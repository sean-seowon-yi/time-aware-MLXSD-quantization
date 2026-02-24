"""
Analyze collected activation statistics and generate a W4A8 quantization config.

Faithful TaQ-DiT baseline: all layers use fixed 8-bit activations with per-tensor
absmax scale. Post-GELU layers additionally carry per-channel shift vectors (momentum
0.95, collected by collect_layer_activations.py) for centering before quantization.

Reads layer_statistics.json produced by collect_layer_activations.py and outputs:
  1. Per-timestep per-layer scale (from tensor_absmax or hist_p999)
  2. Per-channel shift vectors for post-GELU layers (passed through unchanged)
  3. sigma_map for inference-time timestep lookup

For experimental multi-tier (A4/A6/A8) dynamic switching, see:
  src/analyze_activations_multitier.py

Usage:
    conda run -n diffusionkit python -m src.analyze_activations \\
        --stats /path/to/calibration_data/activations/layer_statistics.json \\
        --output /path/to/calibration_data/activations/quant_config.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_stats_v2(stats_path: Path):
    """
    Load per_timestep_npz_v2 format from collect_layer_activations.py.
    Returns: (timesteps_dict, per_step_full, layer_names, metadata, sigma_map)
    where:
      - timesteps_dict[step_key][layer_name] = {tensor_absmax, hist_p999, ...}  (index only)
      - per_step_full[step_key][layer_name] = {avg_min, avg_max, shift, hist_counts, hist_edges, ...}
      - sigma_map[step_key] = sigma value
    """
    with open(stats_path) as f:
        manifest = json.load(f)

    if manifest.get("format") != "per_timestep_npz_v2":
        raise ValueError(f"Expected per_timestep_npz_v2, got {manifest.get('format')}")

    ts_dir = Path(manifest["timestep_dir"])
    step_keys = manifest["step_keys"]
    metadata = manifest.get("metadata", {})
    sigma_map = {int(k): float(v) for k, v in manifest.get("sigma_map", {}).items()}

    timesteps = {}
    per_step_full = {}
    layer_names = set()

    for step_key in step_keys:
        npz_path = ts_dir / f"step_{step_key}.npz"
        index_path = ts_dir / f"step_{step_key}_index.json"

        with open(index_path) as f:
            index = json.load(f)

        npz = np.load(npz_path)

        timesteps[step_key] = index
        per_step_full[step_key] = {}
        layer_names.update(index.keys())

        for layer_name in index.keys():
            safe = layer_name.replace(".", "_")
            per_step_full[step_key][layer_name] = {
                "avg_min": npz[f"{safe}__avg_min"].copy() if f"{safe}__avg_min" in npz else None,
                "avg_max": npz[f"{safe}__avg_max"].copy() if f"{safe}__avg_max" in npz else None,
                "shift": npz[f"{safe}__shift"].copy() if f"{safe}__shift" in npz else None,
                **index[layer_name],  # tensor_absmax, hist_p999, etc.
            }

    return timesteps, per_step_full, sorted(layer_names), metadata, sigma_map


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def identify_outlier_channels(
    avg_min: np.ndarray,
    avg_max: np.ndarray,
    threshold_multiplier: float = 2.5,
    bits: int = 8,
) -> Dict:
    """
    Identify channels with extreme per-channel ranges and compute a per-channel
    multiplier vector for two-scale activation quantization (TaQ-DiT §3.3).

    Channels where range_c > threshold_multiplier × median(range) are outliers.
    Returns a dict with keys: outlier_indices, multiplier_vector, scale_normal,
    scale_outlier.  Returns {} if no outliers are found or data is degenerate.
    """
    per_ch_absmax = np.maximum(np.abs(avg_min), np.abs(avg_max))
    per_ch_range = avg_max - avg_min
    median_range = float(np.median(per_ch_range))
    if median_range < 1e-6:
        return {}

    outlier_mask = per_ch_range > threshold_multiplier * median_range
    if not outlier_mask.any():
        return {}

    qmax = 2 ** (bits - 1) - 1  # 127 for int8

    normal_absmax = per_ch_absmax[~outlier_mask]
    outlier_absmax = per_ch_absmax[outlier_mask]

    if normal_absmax.size == 0 or normal_absmax.max() < 1e-8:
        return {}

    scale_normal = float(normal_absmax.max()) / qmax
    scale_outlier = float(outlier_absmax.max()) / qmax

    if scale_normal < 1e-8:
        return {}

    # Per-channel multiplier: 1.0 for normal, round(scale_outlier/scale_normal) for outliers
    multiplier_vector = np.ones(len(avg_min), dtype=np.float32)
    raw_multiplier = scale_outlier / scale_normal
    multiplier_vector[outlier_mask] = max(1.0, round(raw_multiplier))

    return {
        "outlier_indices": np.where(outlier_mask)[0].tolist(),
        "multiplier_vector": multiplier_vector.tolist(),
        "scale_normal": scale_normal,
        "scale_outlier": scale_outlier,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate W4A8 quantization config (faithful TaQ-DiT baseline)"
    )
    parser.add_argument("--stats", type=Path, required=True,
                        help="layer_statistics.json from collect_layer_activations.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output quant_config.json path")
    parser.add_argument("--use-hist-p999", action="store_true", default=False,
                        help="Use hist_p999 (percentile clipping) instead of tensor_absmax for scale")
    args = parser.parse_args()

    output_path = args.output or (args.stats.parent / "quant_config.json")

    print(f"Loading {args.stats}")
    timesteps, per_step_full, layer_names, metadata, sigma_map = load_stats_v2(args.stats)

    print(f"Calibration: {metadata.get('num_images', 0)} images  "
          f"× {metadata.get('num_timesteps', 0)} timesteps  "
          f"= {metadata.get('total_processed', 0)} forward passes")
    print(f"Timesteps collected: {len(timesteps)}  Layers: {len(layer_names)}")
    print(f"Using {'hist_p999 (percentile)' if args.use_hist_p999 else 'tensor_absmax'} for scale")
    print(f"Activation bits: fixed A8 (faithful TaQ-DiT baseline)")

    step_keys_sorted = sorted(timesteps.keys(), key=int)

    # Build per-timestep config: fixed A8, scale from absmax, shift passthrough
    per_timestep_config = {}
    layer_temporal_stats: Dict[str, List[float]] = {}

    for step_key in step_keys_sorted:
        per_timestep_config[step_key] = {}
        for layer_name in layer_names:
            if layer_name not in per_step_full[step_key]:
                continue
            data = per_step_full[step_key][layer_name]

            scale_val = (
                data.get("hist_p999") or data.get("tensor_absmax", 0.0)
                if args.use_hist_p999
                else data.get("tensor_absmax", 0.0)
            )

            entry: Dict = {
                "bits": 8,
                "scale": float(scale_val),
            }

            # Post-GELU layers: include per-channel shift vector
            shift = data.get("shift")
            if shift is not None:
                entry["shift"] = shift.tolist()

            per_timestep_config[step_key][layer_name] = entry

            # Accumulate for temporal summary
            if layer_name not in layer_temporal_stats:
                layer_temporal_stats[layer_name] = []
            layer_temporal_stats[layer_name].append(float(scale_val))

    # Temporal scale summary per layer
    layer_scale_summary = {}
    for layer_name, vals in layer_temporal_stats.items():
        arr = np.array(vals)
        layer_scale_summary[layer_name] = {
            "mean_scale": float(np.mean(arr)),
            "min_scale": float(np.min(arr)),
            "max_scale": float(np.max(arr)),
        }

    # Shift effectiveness: max shift magnitude per post-GELU layer (informational only)
    shift_summary = {}
    for layer_name in layer_names:
        if not layer_name.endswith(".mlp.fc2"):
            continue
        max_shift_mags = []
        for step_key in step_keys_sorted:
            data = per_step_full.get(step_key, {}).get(layer_name, {})
            shift = data.get("shift")
            if shift is not None:
                max_shift_mags.append(float(np.abs(shift).max()))
        if max_shift_mags:
            shift_summary[layer_name] = {
                "max_shift_magnitude": float(np.max(max_shift_mags)),
                "mean_shift_magnitude": float(np.mean(max_shift_mags)),
            }

    n_post_gelu = sum(1 for n in layer_names if n.endswith(".mlp.fc2"))
    print(f"\nPost-GELU layers with shift: {n_post_gelu}")
    if shift_summary:
        top = sorted(shift_summary.items(), key=lambda x: -x[1]["max_shift_magnitude"])[:5]
        for name, s in top:
            print(f"  {name}: max_shift={s['max_shift_magnitude']:.3f}")

    # ------------------------------------------------------------------
    # Outlier channel detection (per-layer, not per-timestep)
    # Use the median collected timestep as the reference calibration pass.
    # ------------------------------------------------------------------
    ref_step = step_keys_sorted[len(step_keys_sorted) // 2]
    outlier_config: Dict = {}
    for layer_name in layer_names:
        data = per_step_full.get(ref_step, {}).get(layer_name, {})
        avg_min = data.get("avg_min")
        avg_max = data.get("avg_max")
        if avg_min is not None and avg_max is not None:
            result = identify_outlier_channels(avg_min, avg_max, bits=8)
            if result:
                outlier_config[layer_name] = result

    n_outlier_layers = len(outlier_config)
    if n_outlier_layers:
        total_ch_flagged = sum(len(v["outlier_indices"]) for v in outlier_config.values())
        total_ch = sum(
            len(v["multiplier_vector"]) for v in outlier_config.values()
        )
        mean_pct = 100.0 * total_ch_flagged / max(1, total_ch)
        print(f"\n{n_outlier_layers} layers have outlier channels "
              f"(mean {mean_pct:.1f}% channels flagged)")
        top_oc = sorted(outlier_config.items(),
                        key=lambda x: len(x[1]["outlier_indices"]), reverse=True)[:5]
        for name, oc in top_oc:
            n_out = len(oc["outlier_indices"])
            n_total = len(oc["multiplier_vector"])
            print(f"  {name}: {n_out}/{n_total} outlier channels  "
                  f"scale_outlier/scale_normal={oc['scale_outlier']:.4f}/{oc['scale_normal']:.4f}")
    else:
        print("\nNo outlier channels detected (threshold_multiplier=2.5)")

    print(f"\n=== Generating Quantization Config ===")
    total_decisions = sum(
        len(per_timestep_config[sk]) for sk in step_keys_sorted
    )
    print(f"Per-timestep decisions: {total_decisions} total  (all A8)")

    quant_config = {
        "format": "per_timestep_quant_config_v4",
        "per_timestep": per_timestep_config,
        "sigma_map": {str(k): v for k, v in sigma_map.items()},
        "outlier_config": outlier_config,
        "summary": {
            "total_timesteps": len(step_keys_sorted),
            "total_layers": len(layer_names),
            "total_decisions": total_decisions,
            "activation_bits": 8,
            "post_gelu_layers_with_shift": n_post_gelu,
            "layers_with_outlier_channels": n_outlier_layers,
            "scale_source": "hist_p999" if args.use_hist_p999 else "tensor_absmax",
        },
        "metadata": metadata,
    }

    with open(output_path, "w") as f:
        json.dump(quant_config, f, indent=2)
    print(f"\n✓ Quant config -> {output_path}")

    # Save temporal analysis
    analysis_path = args.stats.parent / "layer_temporal_analysis.json"
    temporal_analysis = {
        "format": "layer_temporal_analysis_v4",
        "layer_scale_summary": layer_scale_summary,
        "shift_summary": shift_summary,
        "sigma_map": {str(k): v for k, v in sigma_map.items()},
    }
    with open(analysis_path, "w") as f:
        json.dump(temporal_analysis, f, indent=2)
    print(f"✓ Temporal analysis -> {analysis_path}")


if __name__ == "__main__":
    main()
