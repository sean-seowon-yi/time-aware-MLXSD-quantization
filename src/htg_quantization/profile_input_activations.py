"""
Phase 3a: Profile input activations for HTG quantization.

Consumes the Phase 1 calibration dataset (DiT_cali_data.npz) and records
per-channel (min, max) statistics for the three HTG target layers in each
MMDiT TransformerBlock:

    fc1     → input to mlp.fc1 (affine_transform output before FFN)
    qkv     → input to attn.q/k/v_proj (affine_transform output before attention)
    oproj   → input to attn.o_proj (SDPA output)

These statistics are used by compute_htg_params.py to derive the HTG shift
vectors (z_t) and scaling vector (s) from the paper.

Design mirrors profile_postgelu.py (Phase 2):
    - Group calibration points by prompt to batch cache_modulation_params calls.
    - Double x_t to batch=2 to match CFG conditioning.
    - Reload adaLN weights between prompt groups.

CLI:
    python -m src.htg_quantization.profile_input_activations \\
        --calibration-file DiT_cali_data.npz \\
        --output htg_input_activation_stats.npz \\
        [--num-samples 512]
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import mlx.core as mx

from .input_activation_tracer import (
    InputActivationTracer,
    install_input_tracing,
    remove_input_tracing,
)
from .htg_config import MODEL_VERSION, DEFAULT_CALIBRATION_FILE, DEFAULT_INPUT_STATS_FILE


_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Pipeline helpers (mirrors profile_postgelu.py)
# ---------------------------------------------------------------------------

def _ensure_diffusionkit_on_path() -> None:
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
    except ImportError as exc:
        raise ImportError(
            "diffusionkit.mlx could not be imported. Make sure DiffusionKit is "
            "available or run from the repo root."
        ) from exc


def _load_pipeline(model_version: str, low_memory_mode: bool, local_ckpt: str | None):
    _ensure_diffusionkit_on_path()
    from diffusionkit.mlx import DiffusionPipeline  # type: ignore

    return DiffusionPipeline(
        w16=True,
        shift=3.0,
        use_t5=True,
        model_version=model_version,
        low_memory_mode=low_memory_mode,
        a16=True,
        local_ckpt=local_ckpt,
    )


def _load_calibration_npz(path: str):
    data = np.load(path, allow_pickle=True)
    for k in ("xs", "ts", "prompt_indices", "cs", "cs_pooled"):
        if k not in data:
            raise KeyError(f"Calibration file missing key '{k}'")
    return data


def _select_indices(n_cal: int, num_samples: int, seed: int) -> np.ndarray:
    num_samples = min(num_samples, n_cal)
    return np.random.default_rng(seed).choice(n_cal, size=num_samples, replace=False)


def _group_by_prompt(
    sel_idx: np.ndarray, prompt_indices: np.ndarray
) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in sel_idx:
        groups[int(prompt_indices[idx])].append(int(idx))
    return dict(sorted(groups.items()))


def _run_single_forward(pipeline, x_np: np.ndarray, t_value: float, cond_mx: mx.array) -> None:
    """
    Single MMDiT forward pass. Doubles x to batch=2 to match cfg_batch=2
    conditioning (positive + negative) — identical to Phase 2 strategy.
    """
    x_single = mx.array(x_np[None, ...]).astype(pipeline.activation_dtype)
    x_doubled = mx.concatenate([x_single] * 2, axis=0)

    t_mx = mx.array([t_value], dtype=pipeline.activation_dtype)
    t_broadcast = mx.broadcast_to(t_mx, [x_doubled.shape[0]])

    out = pipeline.mmdit(
        latent_image_embeddings=x_doubled,
        token_level_text_embeddings=mx.expand_dims(cond_mx, 2),
        timestep=t_broadcast,
    )
    mx.eval(out)


# ---------------------------------------------------------------------------
# Core profiling routine
# ---------------------------------------------------------------------------

def run_input_profiling(
    calibration_file: str,
    model_version: str,
    num_samples: int,
    seed: int,
    low_memory_mode: bool,
    local_ckpt: str | None,
) -> Tuple[InputActivationTracer, List[float]]:
    """
    Run forward passes on calibration data with input activation tracing.

    Returns (tracer, sorted_unique_timesteps).
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
        f"Loaded calibration set: {n_cal} points total, "
        f"profiling {len(sel_idx)} points."
    )

    pipeline = _load_pipeline(
        model_version=model_version,
        low_memory_mode=low_memory_mode,
        local_ckpt=local_ckpt,
    )

    groups = _group_by_prompt(sel_idx, prompt_indices)
    print(f"  Grouped into {len(groups)} prompt group(s)")

    tracer = install_input_tracing(pipeline.mmdit)
    all_unique_ts: Set[float] = set()
    processed = 0

    try:
        for group_num, (prompt_idx, cal_indices) in enumerate(groups.items()):
            group_ts_values = sorted({float(ts[i]) for i in cal_indices})
            all_unique_ts.update(group_ts_values)

            pooled_mx = mx.array(cs_pooled[prompt_idx]).astype(pipeline.activation_dtype)
            cond_mx = mx.array(cs[prompt_idx]).astype(pipeline.activation_dtype)

            # Cache modulation params for all timesteps in this prompt group at once.
            # This is the key workaround for the adaLN weight-offload issue.
            ts_mx = mx.array(group_ts_values, dtype=pipeline.activation_dtype)
            pipeline.mmdit.cache_modulation_params(
                pooled_text_embeddings=pooled_mx,
                timesteps=ts_mx,
            )

            for cal_idx in cal_indices:
                _run_single_forward(
                    pipeline=pipeline,
                    x_np=xs[cal_idx],
                    t_value=float(ts[cal_idx]),
                    cond_mx=cond_mx,
                )
                processed += 1

                if processed % 50 == 0 or processed == len(sel_idx):
                    print(
                        f"  Processed {processed}/{len(sel_idx)} points "
                        f"(group {group_num + 1}/{len(groups)}, "
                        f"unique timesteps seen: {len(all_unique_ts)})"
                    )

            # Restore adaLN weights offloaded by cache_modulation_params
            pipeline.mmdit.load_weights(
                pipeline.load_mmdit(only_modulation_dict=True), strict=False
            )
            gc.collect()

    finally:
        remove_input_tracing()

    return tracer, sorted(all_unique_ts)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_input_stats(
    tracer: InputActivationTracer,
    unique_ts: List[float],
    output_path: str,
) -> None:
    """
    Flatten tracer summary into a single .npz file.

    Key format: "{layer_id}::t={timestep_key}::{stat_name}"
    where stat_name ∈ {min, max, count}.

    Also stores:
        timesteps_unique    float32 (N_ts,)
        layer_ids           object  (N_layers,)  — list of unique full layer ids
    """
    summary = tracer.summarize()

    flat: Dict[str, np.ndarray] = {}
    all_layer_ids: List[str] = []

    for full_layer_id, tdict in summary.items():
        all_layer_ids.append(full_layer_id)
        for t_key, stats in tdict.items():
            prefix = f"{full_layer_id}::t={t_key}"
            for stat_name, arr in stats.items():
                flat[f"{prefix}::{stat_name}"] = arr

    flat["timesteps_unique"] = np.array(unique_ts, dtype=np.float32)
    flat["layer_ids"] = np.array(all_layer_ids, dtype=object)

    np.savez_compressed(output_path, **flat)
    print(f"\nSaved input activation stats to {output_path}")
    print(f"  Full layer IDs traced: {len(all_layer_ids)}")
    print(f"  Unique timesteps: {len(unique_ts)}")

    # Quick sanity summary
    layer_types = {"fc1": 0, "qkv": 0, "oproj": 0}
    for lid in all_layer_ids:
        for t in layer_types:
            if lid.endswith(f"_{t}"):
                layer_types[t] += 1
    for t, n in layer_types.items():
        print(f"  {t} layers: {n}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3a: Profile input activations for HTG quantization of SD3 / MMDiT. "
            "Records per-channel (min, max) for fc1, qkv, and oproj inputs."
        )
    )
    parser.add_argument(
        "--calibration-file", type=str, default=DEFAULT_CALIBRATION_FILE,
        help="Path to Phase 1 calibration .npz (default: %(default)s)",
    )
    parser.add_argument(
        "--model-version", type=str, default=MODEL_VERSION,
        help="DiffusionKit model key (default: %(default)s)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=512,
        help="Number of calibration points to profile (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for point selection (default: %(default)s)",
    )
    parser.add_argument(
        "--low-memory-mode", action="store_true", default=True,
        help="Enable DiffusionKit low_memory_mode (default: on)",
    )
    parser.add_argument(
        "--no-low-memory-mode", action="store_false", dest="low_memory_mode",
    )
    parser.add_argument(
        "--local-ckpt", type=str, default=None,
        help="Optional path to a local MMDiT checkpoint",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=DEFAULT_INPUT_STATS_FILE,
        help="Output path for input activation statistics .npz (default: %(default)s)",
    )

    args = parser.parse_args()

    tracer, unique_ts = run_input_profiling(
        calibration_file=args.calibration_file,
        model_version=args.model_version,
        num_samples=args.num_samples,
        seed=args.seed,
        low_memory_mode=args.low_memory_mode,
        local_ckpt=args.local_ckpt,
    )

    save_input_stats(tracer, unique_ts, args.output)


if __name__ == "__main__":
    main()
