"""
Phase 1 EDA: Forward-pass profiling to collect all 6 activation families.

Consumes eda_output/coco_cali_data.npz (3,000 points: 100 trajectories × 30 steps),
replays the MMDiT backbone at those (x_t, t, conditioning) points, and records
per-(family, layer, timestep) statistics via EDATracer.

Strategy (same as profile_postgelu.py):
  - Group calibration points by prompt to avoid the adaLN offload bug.
  - Per group: call cache_modulation_params once, run all forward passes, then
    reload adaLN weights before moving to the next group.

Subsampling (to reduce runtime):
  - Layers: every 3rd block from 24 (blocks 0,3,6,9,12,15,18,21,23) × img+txt = 18 layers
  - Timesteps: every 2nd from 30 sorted unique timesteps → 15 timesteps

CLI:
  python -m src.eda.profile_activations \\
      --calibration-file eda_output/coco_cali_data.npz \\
      --weight-output   eda_output/weight_stats.npz \\
      --output          eda_output/activation_stats_full.npz
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import mlx.core as mx

from .eda_tracer import (
    EDATracer,
    install_eda_tracing,
    remove_eda_tracing,
    save_tracer_stats,
)
from .weight_profiler import collect_weight_stats


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CALI = str(_ROOT / "eda_output" / "coco_cali_data.npz")
_DEFAULT_WEIGHT_OUT = str(_ROOT / "eda_output" / "weight_stats.npz")
_DEFAULT_ACT_OUT = str(_ROOT / "eda_output" / "activation_stats_full.npz")

# Subsampling parameters
BLOCK_SAMPLE_STEP = 3    # every 3rd block from 24 → [0, 3, 6, 9, 12, 15, 18, 21] + last (23)
TIMESTEP_SAMPLE_STEP = 2  # every 2nd from 25 sorted timesteps → 13 timesteps


def _build_profiled_layer_ids(n_blocks: int = 24) -> Set[str]:
    """Build the set of layer IDs to profile (every 3rd block + last, both streams)."""
    sampled = list(range(0, n_blocks, BLOCK_SAMPLE_STEP))
    if (n_blocks - 1) not in sampled:
        sampled.append(n_blocks - 1)
    ids: Set[str] = set()
    for blk in sampled:
        ids.add(f"mm_{blk:02d}_img")
        ids.add(f"mm_{blk:02d}_txt")
    return ids


def _ensure_diffusionkit() -> None:
    try:
        import diffusionkit.mlx  # noqa: F401
        return
    except ImportError:
        pass
    dk_src = _ROOT / "DiffusionKit" / "python" / "src"
    if dk_src.is_dir() and str(dk_src) not in sys.path:
        sys.path.insert(0, str(dk_src))
    import diffusionkit.mlx  # noqa: F401


def _load_pipeline(model_version: str, low_memory_mode: bool, local_ckpt: str | None):
    _ensure_diffusionkit()
    from diffusionkit.mlx import DiffusionPipeline  # type: ignore
    return DiffusionPipeline(
        w16=True, shift=3.0, use_t5=True,
        model_version=model_version,
        low_memory_mode=low_memory_mode,
        a16=True, local_ckpt=local_ckpt,
    )


def _load_calibration(path: str):
    data = np.load(path, allow_pickle=True)
    required = ["xs", "ts", "prompt_indices", "cs", "cs_pooled", "prompts", "cfg_scale"]
    for k in required:
        if k not in data:
            raise KeyError(f"Calibration file missing key '{k}'")
    return data


def _group_by_prompt(
    prompt_indices: np.ndarray,
    ts: np.ndarray,
) -> dict[int, List[int]]:
    groups: dict[int, List[int]] = defaultdict(list)
    for idx in range(len(prompt_indices)):
        groups[int(prompt_indices[idx])].append(idx)
    return dict(sorted(groups.items()))


def _run_single_forward(pipeline, x_np: np.ndarray, t_value: float, cond_mx: mx.array) -> None:
    """Run a single MMDiT forward pass at (x_t, t, conditioning)."""
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


def run_profiling(
    calibration_file: str,
    model_version: str,
    low_memory_mode: bool,
    local_ckpt: str | None,
    weight_output: str,
    act_output: str,
) -> None:
    """
    Full EDA profiling pipeline:
      1. Load pipeline + calibration data.
      2. Collect weight statistics (one-time).
      3. Install EDA tracing.
      4. Run all forward passes grouped by prompt.
      5. Save activation stats.
    """
    data = _load_calibration(calibration_file)
    xs = data["xs"]
    ts = data["ts"]
    prompt_indices = data["prompt_indices"]
    cs = data["cs"]
    cs_pooled = data["cs_pooled"]
    n_cal = xs.shape[0]

    print(f"Loaded calibration data: {n_cal} points from {calibration_file}")

    # Compute subsampled timesteps (every 2nd from sorted unique set)
    all_ts_sorted = sorted(set(float(ts[i]) for i in range(len(ts))))
    profiled_ts = set(all_ts_sorted[i] for i in range(0, len(all_ts_sorted), TIMESTEP_SAMPLE_STEP))

    # Compute subsampled layer IDs
    profiled_layer_ids = _build_profiled_layer_ids()

    print(f"Subsampling: {len(profiled_layer_ids)} layers, {len(profiled_ts)}/{len(all_ts_sorted)} timesteps")

    pipeline = _load_pipeline(model_version, low_memory_mode, local_ckpt)

    # --- Weight stats (before any cache_modulation_params call) ---
    Path(weight_output).parent.mkdir(parents=True, exist_ok=True)
    print("\nCollecting weight statistics...")
    weight_stats = collect_weight_stats(pipeline.mmdit)
    np.savez_compressed(weight_output, **weight_stats)
    print(f"Saved weight stats to {weight_output}  ({len(weight_stats)} entries)")

    # --- Activation profiling ---
    groups = _group_by_prompt(prompt_indices, ts)
    print(f"\nProfiling activations: {n_cal} total points, {len(groups)} prompt groups")
    print(f"  Profiling {len(profiled_layer_ids)} layers × {len(profiled_ts)} timesteps")

    tracer = install_eda_tracing(pipeline.mmdit, profiled_layer_ids=profiled_layer_ids)
    all_unique_ts: set[float] = set()
    processed = 0
    skipped = 0

    try:
        for group_num, (prompt_idx, cal_indices) in enumerate(groups.items()):
            group_ts = sorted(set(float(ts[i]) for i in cal_indices))
            all_unique_ts.update(t for t in group_ts if t in profiled_ts)

            pooled_mx = mx.array(cs_pooled[prompt_idx]).astype(pipeline.activation_dtype)
            cond_mx = mx.array(cs[prompt_idx]).astype(pipeline.activation_dtype)

            ts_mx = mx.array(group_ts, dtype=pipeline.activation_dtype)
            pipeline.mmdit.cache_modulation_params(
                pooled_text_embeddings=pooled_mx,
                timesteps=ts_mx,
            )

            for cal_idx in cal_indices:
                t_val = float(ts[cal_idx])
                if t_val not in profiled_ts:
                    skipped += 1
                    continue
                _run_single_forward(
                    pipeline=pipeline,
                    x_np=xs[cal_idx],
                    t_value=t_val,
                    cond_mx=cond_mx,
                )
                processed += 1
                if processed % 50 == 0:
                    print(
                        f"  [{processed} profiled / {skipped} skipped] "
                        f"group {group_num+1}/{len(groups)}  "
                        f"unique_ts={len(all_unique_ts)}"
                    )

            # Reload adaLN weights offloaded by cache_modulation_params
            pipeline.mmdit.load_weights(
                pipeline.load_mmdit(only_modulation_dict=True), strict=False
            )
            gc.collect()

    finally:
        remove_eda_tracing()

    # Explicitly release MLX/Metal resources while Python runtime is still alive.
    # Without this, Metal buffers are freed during interpreter shutdown and segfault.
    del pipeline
    gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception:
        pass

    Path(act_output).parent.mkdir(parents=True, exist_ok=True)
    save_tracer_stats(tracer, act_output, sorted(all_unique_ts))
    summary = tracer.summarize()
    total_entries = sum(
        len(t_map)
        for fam_map in summary.values()
        for t_map in fam_map.values()
    )
    print(f"\nSaved activation stats to {act_output}")
    print(f"  Families:       {list(summary.keys())}")
    print(f"  Layers profiled:  {len(profiled_layer_ids)}")
    print(f"  Unique timesteps: {len(all_unique_ts)}")
    print(f"  Total (fam, layer, t) entries: {total_entries}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 EDA: Profile all activation families in SD3 MMDiT"
    )
    parser.add_argument("--calibration-file", type=str, default=_DEFAULT_CALI)
    parser.add_argument(
        "--model-version", type=str,
        default="argmaxinc/mlx-stable-diffusion-3-medium",
    )
    parser.add_argument("--low-memory-mode", action="store_true", default=True)
    parser.add_argument("--no-low-memory-mode", action="store_false", dest="low_memory_mode")
    parser.add_argument("--local-ckpt", type=str, default=None)
    parser.add_argument("--weight-output", type=str, default=_DEFAULT_WEIGHT_OUT)
    parser.add_argument("--output", "-o", type=str, default=_DEFAULT_ACT_OUT)
    args = parser.parse_args()

    run_profiling(
        calibration_file=args.calibration_file,
        model_version=args.model_version,
        low_memory_mode=args.low_memory_mode,
        local_ckpt=args.local_ckpt,
        weight_output=args.weight_output,
        act_output=args.output,
    )


if __name__ == "__main__":
    main()
