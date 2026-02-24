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
from typing import Dict, List

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
                "shift": npz[f"{safe}__shift"].copy() if f"{safe}__shift" in npz else None,
                **index[layer_name],  # tensor_absmax, hist_p999, etc.
            }

    return timesteps, per_step_full, sorted(layer_names), metadata, sigma_map


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

    print(f"\n=== Generating Quantization Config ===")
    total_decisions = sum(
        len(per_timestep_config[sk]) for sk in step_keys_sorted
    )
    print(f"Per-timestep decisions: {total_decisions} total  (all A8)")

    quant_config = {
        "format": "per_timestep_quant_config_v4",
        "per_timestep": per_timestep_config,
        "sigma_map": {str(k): v for k, v in sigma_map.items()},
        "summary": {
            "total_timesteps": len(step_keys_sorted),
            "total_layers": len(layer_names),
            "total_decisions": total_decisions,
            "activation_bits": 8,
            "post_gelu_layers_with_shift": n_post_gelu,
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
