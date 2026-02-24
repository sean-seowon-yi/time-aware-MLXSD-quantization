"""
Phase 2: Diagnostic profiling of post-GELU FFN activations for SD3 / MMDiT.

This script consumes the Phase 1 calibration dataset (DiT_cali_data.npz),
replays the MMDiT backbone at those (x, t, conditioning) points, and
records per-layer, per-timestep statistics of the post-GELU FFN outputs.

The goal is to answer the questions from TaQ-DiT for this specific
architecture and sampler:

- Do post-GELU activations exhibit strong asymmetry / heavy mass near 0?
- How do their means / variances / ranges shift across Euler timesteps?

Strategy for correctness:
- Group calibration points by prompt so we can batch all timesteps for a
  single prompt into one cache_modulation_params call (avoiding the
  adaLN weight offload problem that occurs with single-timestep calls).
- Double x_t to batch=2 to match the cfg_batch=2 conditioning from
  Phase 1, replicating CFGDenoiser's batching.

CLI (typical dry run, using a subset of calibration points):

  cd /Users/seanyi/Documents/time-aware-MLXSD-quantization
  python -m src.activation_diagnostics.profile_postgelu \\
      --calibration-file DiT_cali_data.npz \\
      --num-samples 512 \\
      --output activation_stats_postgelu.npz
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mlx.core as mx

from .activation_tracer import (
    ActivationTracer,
    HISTOGRAM_NUM_BINS,
    HISTOGRAM_RANGE,
    install_tracing,
    remove_tracing,
)


_ROOT = Path(__file__).resolve().parents[2]


def _ensure_diffusionkit_on_path() -> None:
    """Make sure DiffusionKit is importable, mirroring Phase 1."""
    try:
        import diffusionkit.mlx  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass

    dk_src = _ROOT / "DiffusionKit" / "python" / "src"
    if dk_src.is_dir() and str(dk_src) not in sys.path:
        sys.path.insert(0, str(dk_src))
    try:
        import diffusionkit.mlx  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "diffusionkit.mlx could not be imported. Make sure DiffusionKit is "
            "available or run from the repo root with "
            "PYTHONPATH=DiffusionKit/python/src"
        ) from exc


def _load_pipeline(model_version: str, low_memory_mode: bool, local_ckpt: str | None):
    """Instantiate the SD3 Medium DiffusionPipeline similarly to Phase 1."""
    _ensure_diffusionkit_on_path()
    from diffusionkit.mlx import DiffusionPipeline  # type: ignore

    pipeline = DiffusionPipeline(
        w16=True,
        shift=3.0,
        use_t5=True,
        model_version=model_version,
        low_memory_mode=low_memory_mode,
        a16=True,
        local_ckpt=local_ckpt,
    )
    return pipeline


def _load_calibration_npz(path: str):
    """Load Phase 1 calibration data."""
    data = np.load(path, allow_pickle=True)
    required_keys = [
        "xs", "ts", "prompt_indices", "cs", "cs_pooled", "prompts", "cfg_scale",
    ]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Calibration file missing key '{k}'")
    return data


def _select_indices(n_cal: int, num_samples: int, seed: int) -> np.ndarray:
    """Choose which calibration points to profile."""
    num_samples = min(num_samples, n_cal)
    rng = np.random.default_rng(seed)
    return rng.choice(n_cal, size=num_samples, replace=False)


def _group_by_prompt(
    sel_idx: np.ndarray,
    prompt_indices: np.ndarray,
    ts: np.ndarray,
) -> dict[int, list[int]]:
    """
    Group selected calibration indices by prompt_index.

    Returns {prompt_idx: [cal_idx, ...]} sorted by prompt_idx.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in sel_idx:
        p = int(prompt_indices[idx])
        groups[p].append(int(idx))
    return dict(sorted(groups.items()))


def _run_single_forward(
    pipeline,
    x_np: np.ndarray,
    t_value: float,
    cond_mx: mx.array,
) -> None:
    """
    Run a single MMDiT forward pass at (x_t, t, conditioning).

    x_np has shape (H, W, C). We double x to batch=2 to match the
    cfg_batch=2 conditioning (positive + negative), replicating
    CFGDenoiser's internal batching.
    """
    x_single = mx.array(x_np[None, ...]).astype(pipeline.activation_dtype)
    x_doubled = mx.concatenate([x_single] * 2, axis=0)

    t_mx = mx.array([t_value], dtype=pipeline.activation_dtype)
    t_broadcast = mx.broadcast_to(t_mx, [x_doubled.shape[0]])

    mmdit_input = {
        "latent_image_embeddings": x_doubled,
        "token_level_text_embeddings": mx.expand_dims(cond_mx, 2),
        "timestep": t_broadcast,
    }

    out = pipeline.mmdit(**mmdit_input)
    mx.eval(out)


def run_diagnostics(
    calibration_file: str,
    model_version: str,
    num_samples: int,
    seed: int,
    low_memory_mode: bool,
    local_ckpt: str | None,
) -> Tuple[ActivationTracer, List[float]]:
    """
    Core Phase 2 diagnostic routine.

    To avoid the adaLN weight offload bug, we group calibration points by
    prompt. For each prompt group we:
    1. Collect all unique timesteps for that group.
    2. Call cache_modulation_params once with all those timesteps.
    3. Run forward passes for every point in the group.
    4. Reload the adaLN weights (via pipeline.load_mmdit) before moving
       to the next prompt group, since cache_modulation_params zeroes them.
    """
    data = _load_calibration_npz(calibration_file)
    xs = data["xs"]
    ts = data["ts"]
    prompt_indices = data["prompt_indices"]
    cs = data["cs"]
    cs_pooled = data["cs_pooled"]

    n_cal = xs.shape[0]
    sel_idx = _select_indices(n_cal, num_samples, seed)

    print(
        f"Loaded calibration set from {calibration_file}: "
        f"{n_cal} points, profiling {len(sel_idx)} of them."
    )

    pipeline = _load_pipeline(
        model_version=model_version,
        low_memory_mode=low_memory_mode,
        local_ckpt=local_ckpt,
    )

    groups = _group_by_prompt(sel_idx, prompt_indices, ts)
    print(f"  Grouped into {len(groups)} prompt group(s)")

    tracer = install_tracing(pipeline.mmdit)
    all_unique_ts: set[float] = set()
    processed = 0

    try:
        for group_num, (prompt_idx, cal_indices) in enumerate(groups.items()):
            # Collect unique timestep values for this prompt group
            group_ts_values = sorted(set(float(ts[i]) for i in cal_indices))
            all_unique_ts.update(group_ts_values)

            # Build conditioning tensors (already on the right dtype from Phase 1)
            pooled_mx = mx.array(cs_pooled[prompt_idx]).astype(
                pipeline.activation_dtype
            )
            cond_mx = mx.array(cs[prompt_idx]).astype(pipeline.activation_dtype)

            # Cache modulation params for ALL timesteps in this group at once.
            # This is the same pattern Phase 1 uses.
            ts_mx = mx.array(group_ts_values, dtype=pipeline.activation_dtype)
            pipeline.mmdit.cache_modulation_params(
                pooled_text_embeddings=pooled_mx,
                timesteps=ts_mx,
            )

            for cal_idx in cal_indices:
                x_np = xs[cal_idx]
                t_value = float(ts[cal_idx])

                _run_single_forward(
                    pipeline=pipeline,
                    x_np=x_np,
                    t_value=t_value,
                    cond_mx=cond_mx,
                )
                processed += 1

                if processed % 50 == 0 or processed == len(sel_idx):
                    print(
                        f"  Processed {processed}/{len(sel_idx)} points "
                        f"(prompt group {group_num+1}/{len(groups)}, "
                        f"unique timesteps: {len(all_unique_ts)})"
                    )

            # Restore adaLN weights that cache_modulation_params offloaded,
            # so the next prompt group starts fresh.
            pipeline.mmdit.load_weights(
                pipeline.load_mmdit(only_modulation_dict=True), strict=False
            )
            gc.collect()

    finally:
        remove_tracing()

    return tracer, sorted(all_unique_ts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: Profile post-GELU FFN activations for SD3 / MMDiT "
            "using the Phase 1 calibration dataset."
        )
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default="DiT_cali_data.npz",
        help="Path to Phase 1 calibration .npz",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="argmaxinc/mlx-stable-diffusion-3-medium",
        help="DiffusionKit model key",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help=(
            "Number of calibration points to profile. "
            "They are sampled uniformly at random without replacement."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for selecting calibration points",
    )
    parser.add_argument(
        "--low-memory-mode",
        action="store_true",
        default=True,
        help="Enable DiffusionKit low_memory_mode (recommended on laptops)",
    )
    parser.add_argument(
        "--no-low-memory-mode",
        action="store_false",
        dest="low_memory_mode",
    )
    parser.add_argument(
        "--local-ckpt",
        type=str,
        default=None,
        help="Optional path to a local MMDiT checkpoint",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="activation_stats_postgelu.npz",
        help="Output path for activation statistics (.npz)",
    )

    args = parser.parse_args()

    tracer, unique_ts = run_diagnostics(
        calibration_file=args.calibration_file,
        model_version=args.model_version,
        num_samples=args.num_samples,
        seed=args.seed,
        low_memory_mode=args.low_memory_mode,
        local_ckpt=args.local_ckpt,
    )

    summary = tracer.summarize()

    flat: dict[str, np.ndarray] = {}
    for layer_id, tdict in summary.items():
        for t_key, stats in tdict.items():
            prefix = f"{layer_id}::t={t_key}"
            for stat_name, arr in stats.items():
                flat[f"{prefix}::{stat_name}"] = arr

    flat["timesteps_unique"] = np.array(unique_ts, dtype=np.float32)
    flat["histogram_bin_edges"] = np.linspace(
        HISTOGRAM_RANGE[0], HISTOGRAM_RANGE[1], HISTOGRAM_NUM_BINS + 1,
        dtype=np.float32,
    )

    np.savez_compressed(args.output, **flat)
    print(f"\nSaved activation statistics to {args.output}")
    print(f"  Layers traced: {len(summary)}")
    print(f"  Unique timesteps: {len(unique_ts)}")
    print(f"  Histogram bins: {HISTOGRAM_NUM_BINS} in [{HISTOGRAM_RANGE[0]}, {HISTOGRAM_RANGE[1]}]")
    print("You can now inspect per-layer, per-timestep stats/histograms in a notebook.")


if __name__ == "__main__":
    main()
