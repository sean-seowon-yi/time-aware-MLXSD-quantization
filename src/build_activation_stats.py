"""
Build activation statistics from AdaRound cache for polynomial schedule generation.

Replays each block's cached I/O tensors from calibration_data_512/adaround_cache/
through the live model with activation tracing installed, collecting per-layer
per-timestep histograms + min/max.  Writes the same activations/ directory format
that generate_poly_schedule.py and explore_curve_fits.py expect.

Usage:
    conda run --no-capture-output -n diffusionkit python -m src.build_activation_stats \\
        --adaround-cache calibration_data_512/adaround_cache \\
        --output calibration_data_512/activations \\
        --resume
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import mlx.core as mx

from src.activation_diagnostics.activation_tracer import (
    ActivationTracer,
    HISTOGRAM_NUM_BINS,
    HISTOGRAM_RANGE,
    install_tracing,
    remove_tracing,
)


# ---------------------------------------------------------------------------
# Layer name mapping helpers
# ---------------------------------------------------------------------------

def tracer_key_to_output_layers(tracer_layer_id: str, tensor_type: str) -> List[str]:
    """
    Convert tracer (layer_id, tensor_type) to one or more dot-separated output layer names.

    Tracer layer_ids look like "mm_00_img" or "mm_00_txt".
    tensor_type is one of "qkv_in", "o_in", "fc1_in", "fc2_in".

    Returns a list because qkv_in expands to three separate proj entries.
    """
    # "mm_00_img" -> block_num=0, stream="img"
    parts = tracer_layer_id.split("_")  # ["mm", "00", "img"]
    block_num = int(parts[1])
    stream = parts[2]
    prefix = f"mm{block_num}.{stream}"

    if tensor_type == "qkv_in":
        return [
            f"{prefix}.attn.q_proj",
            f"{prefix}.attn.k_proj",
            f"{prefix}.attn.v_proj",
        ]
    if tensor_type == "o_in":
        return [f"{prefix}.attn.o_proj"]
    if tensor_type == "fc1_in":
        return [f"{prefix}.mlp.fc1"]
    if tensor_type == "fc2_in":
        return [f"{prefix}.mlp.fc2"]
    raise ValueError(f"Unknown tensor_type: {tensor_type!r}")


def parse_tracer_full_id(full_id: str) -> Optional[Tuple[str, str]]:
    """
    Parse "mm_00_img:qkv_in" into ("mm_00_img", "qkv_in").
    Returns None for unrecognized formats.
    """
    if ":" not in full_id:
        return None
    layer_id, tensor_type = full_id.split(":", 1)
    if not layer_id.startswith("mm_"):
        return None  # Skip uni blocks (not in adaround_cache mm-only blocks)
    return layer_id, tensor_type


# ---------------------------------------------------------------------------
# Histogram percentile computation
# ---------------------------------------------------------------------------

def compute_percentiles(histogram: np.ndarray) -> Dict[str, float]:
    """
    Compute four-corner percentiles from a fixed-range histogram.

    histogram: shape [HISTOGRAM_NUM_BINS] with bin counts.
    Returns dict with hist_p999, hist_p99, hist_p01, hist_p001.
    """
    bin_edges = np.linspace(HISTOGRAM_RANGE[0], HISTOGRAM_RANGE[1], HISTOGRAM_NUM_BINS + 1)
    cumsum = np.cumsum(histogram.astype(np.float64))
    total = float(cumsum[-1])

    if total == 0:
        return {
            "hist_p999": 0.0, "hist_p99": 0.0,
            "hist_p01": 0.0, "hist_p001": 0.0,
        }

    def pct(frac: float) -> float:
        idx = int(np.searchsorted(cumsum, frac * total))
        idx = min(idx, HISTOGRAM_NUM_BINS)
        return float(bin_edges[idx])

    return {
        "hist_p999": pct(0.999),
        "hist_p99":  pct(0.99),
        "hist_p01":  pct(0.01),
        "hist_p001": pct(0.001),
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_output(
    tracer: ActivationTracer,
    output_dir: Path,
    sigma_map: Dict[int, float],
    key_timesteps: List[int],
    t_to_step: Dict[float, int],
    n_images: int,
) -> None:
    """Convert tracer stats to activations/ format and write all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = output_dir / "timestep_stats"
    ts_dir.mkdir(exist_ok=True)

    # Aggregate: step_idx → layer_dot → index entry / channel arrays
    step_index: Dict[int, Dict[str, dict]] = defaultdict(dict)
    step_channels: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)

    for full_id, t_dict in tracer.stats.items():
        parsed = parse_tracer_full_id(full_id)
        if parsed is None:
            continue
        layer_id, tensor_type = parsed

        try:
            output_layers = tracer_key_to_output_layers(layer_id, tensor_type)
        except ValueError:
            continue

        for t_key_str, per_t_stats in t_dict.items():
            t_val = float(t_key_str)
            step_idx = t_to_step.get(t_val)
            if step_idx is None:
                # Try nearest within 1.0 t-units
                if t_to_step:
                    closest = min(t_to_step.keys(), key=lambda x: abs(x - t_val))
                    if abs(closest - t_val) < 1.0:
                        step_idx = t_to_step[closest]
                    else:
                        print(f"  WARNING: no step for t={t_val:.3f}, skipping")
                        continue
                else:
                    continue

            finalized = per_t_stats.finalize()
            pcts = compute_percentiles(finalized["histogram"])

            global_min = float(finalized["min"].min())
            global_max = float(finalized["max"].max())
            abs_max = float(max(abs(global_min), abs(global_max)))

            index_entry = {
                "n_batches": n_images,
                "tensor_absmax": abs_max,
                "tensor_min": global_min,
                "tensor_max": global_max,
                "has_shift": False,
                "has_hist": True,
                **pcts,
            }

            for layer_dot in output_layers:
                step_index[step_idx][layer_dot] = index_entry
                layer_raw = layer_dot.replace(".", "_")
                # avg_max / avg_min are per-channel running max/min across all samples
                step_channels[step_idx][f"{layer_raw}__avg_max"] = finalized["max"]
                step_channels[step_idx][f"{layer_raw}__avg_min"] = finalized["min"]

    # Write per-step files
    step_keys_written: List[str] = []
    for step_idx in sorted(step_index.keys()):
        idx_path = ts_dir / f"step_{step_idx}_index.json"
        with open(idx_path, "w") as f:
            json.dump(step_index[step_idx], f)

        npz_path = ts_dir / f"step_{step_idx}.npz"
        np.savez_compressed(npz_path, **step_channels[step_idx])

        step_keys_written.append(str(step_idx))
        print(f"  step {step_idx}: {len(step_index[step_idx])} layers")

    # Write layer_statistics.json
    sigma_map_str = {str(k): float(v) for k, v in sigma_map.items() if k in key_timesteps}
    meta = {
        "format": "activation_stats_v1",
        "sigma_map": sigma_map_str,
        "step_keys": step_keys_written,
        "metadata": {
            "num_images": n_images,
            "key_timesteps": key_timesteps,
            "source": "adaround_cache",
        },
    }
    stats_path = output_dir / "layer_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {stats_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adaround-cache", type=Path, required=True,
        help="Path to calibration_data_512/adaround_cache",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for activations/ format",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip if output/layer_statistics.json already exists",
    )
    args = parser.parse_args()

    stats_path = args.output / "layer_statistics.json"
    if args.resume and stats_path.exists():
        print(f"Output already exists at {args.output} (--resume), skipping.")
        return

    cache_dir = args.adaround_cache
    metadata = json.loads((cache_dir / "metadata.json").read_text())

    key_timesteps: List[int] = metadata["key_timesteps"]
    block_names: List[str] = metadata["block_names"]
    selected_image_ids: List[int] = metadata["selected_image_ids"]
    samples_list: List[dict] = metadata["samples"]

    # Group samples by img_idx
    img_to_samples: Dict[int, List[dict]] = defaultdict(list)
    for s in samples_list:
        img_to_samples[s["img_idx"]].append(s)

    # Build sigma_map (step_idx → sigma) and t_to_step (t_value → step_idx)
    # by reading one sample file per step_idx.
    print("Building sigma/timestep map from sample files...")
    samples_dir = cache_dir / "samples"
    sigma_map: Dict[int, float] = {}
    t_to_step: Dict[float, int] = {}

    for step_idx in key_timesteps:
        for s in samples_list:
            if s["step_idx"] == step_idx:
                d = np.load(samples_dir / s["file"], allow_pickle=True)
                sigma_map[step_idx] = float(d["__sigma__"][0])
                t_to_step[float(d["mm0__arg2"][0])] = step_idx
                break

    print(f"  {len(key_timesteps)} timesteps: {key_timesteps}")
    print(f"  sigma range: {min(sigma_map.values()):.3f} – {max(sigma_map.values()):.3f}")

    # Load pipeline
    print("\nLoading pipeline...")
    from diffusionkit.mlx import DiffusionPipeline  # type: ignore

    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    mmdit = pipeline.mmdit
    activation_dtype = pipeline.activation_dtype
    print("✓ Pipeline loaded")

    # Install activation tracing
    print("Installing activation tracing...")
    tracer = install_tracing(mmdit)
    print("✓ Tracing installed")

    pooled_dir = cache_dir / "pooled"
    n_images = len(selected_image_ids)

    print(f"\nProcessing {n_images} images × {len(key_timesteps)} steps "
          f"× {len(block_names)} blocks...\n")

    for img_progress, img_idx in enumerate(selected_image_ids):
        img_samples = sorted(img_to_samples[img_idx], key=lambda s: s["step_idx"])
        if not img_samples:
            print(f"  [{img_progress+1}/{n_images}] img {img_idx}: no samples, skipping")
            continue

        print(f"  [{img_progress+1}/{n_images}] img {img_idx} "
              f"({len(img_samples)} steps)", flush=True)

        # --- Reload adaLN weights zeroed by previous image's cache_modulation_params ---
        # (On the first image the weights are already loaded; this is a no-op then.)
        try:
            mmdit.clear_modulation_params_cache()
        except Exception:
            pass
        try:
            mmdit.load_weights(
                pipeline.load_mmdit(only_modulation_dict=True), strict=False
            )
        except Exception as e:
            print(f"    WARNING: adaLN reload failed: {e}")

        # --- Cache modulation params for all this image's timesteps ---
        pooled_np = np.load(pooled_dir / f"{img_idx:04d}.npz")["pooled"]
        pooled_mx = mx.array(pooled_np).astype(activation_dtype)

        # Collect the t-values (arg2[0]) in step order
        t_values = []
        for s in img_samples:
            d = np.load(samples_dir / s["file"], allow_pickle=True)
            t_values.append(float(d["mm0__arg2"][0]))

        timesteps_mx = mx.array(t_values).astype(activation_dtype)
        mmdit.cache_modulation_params(pooled_mx, timesteps_mx)
        mx.eval()

        # --- Run each (step, block) forward pass; tracer records internally ---
        for s in img_samples:
            d = np.load(samples_dir / s["file"], allow_pickle=True)

            for block_name in block_names:
                block_idx = int(block_name[2:])  # "mm3" → 3
                block = mmdit.multimodal_transformer_blocks[block_idx]

                arg0 = mx.array(d[f"{block_name}__arg0"]).astype(activation_dtype)
                arg1 = mx.array(d[f"{block_name}__arg1"]).astype(activation_dtype)
                arg2 = mx.array(d[f"{block_name}__arg2"]).astype(activation_dtype)

                block(arg0, arg1, arg2)

            mx.eval()  # Flush after each step to avoid unbounded graph growth

    remove_tracing()
    print("\n✓ Tracing complete")

    # Write output
    print(f"\nWriting output to {args.output} ...")
    write_output(
        tracer=tracer,
        output_dir=args.output,
        sigma_map=sigma_map,
        key_timesteps=key_timesteps,
        t_to_step=t_to_step,
        n_images=n_images,
    )
    print("Done!")


if __name__ == "__main__":
    main()
