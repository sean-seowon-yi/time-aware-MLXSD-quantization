"""
Generate a per-timestep lookup table (LUT) schedule from calibration stats.

Reads timestep_stats/step_*_index.json and produces a JSON mapping each layer
to an array of tensor_absmax values (one per calibrated timestep).  At inference,
the nearest sigma is found and its alpha used directly — no polynomial fitting.

Usage:
    conda run -n diffusionkit python -m src.generate_lut_schedule \
        --activations-dir calibration_data_512/activations \
        --output lut_schedule_512.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-dir", required=True,
                        help="Path to activations dir with layer_statistics.json and timestep_stats/")
    parser.add_argument("--output", required=True, help="Output LUT JSON path")
    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    meta_path = activations_dir / "layer_statistics.json"
    ts_dir = activations_dir / "timestep_stats"

    with open(meta_path) as f:
        meta = json.load(f)

    step_keys = sorted(meta["step_keys"], key=int)
    sigma_map = {k: float(v) for k, v in meta["sigma_map"].items()}
    sigmas = [sigma_map[s] for s in step_keys]

    # Load all step index JSONs
    step_indices = {}
    for s in step_keys:
        idx_path = ts_dir / f"step_{s}_index.json"
        with open(idx_path) as f:
            step_indices[s] = json.load(f)

    # Discover layers from first step
    all_layers_dot = sorted(step_indices[step_keys[0]].keys())

    # Build per-layer alpha arrays from tensor_absmax
    layers = {}
    for layer_dot in all_layers_dot:
        alphas = []
        for s in step_keys:
            info = step_indices[s].get(layer_dot, {})
            v = info.get("tensor_absmax")
            if v is not None:
                alphas.append(abs(float(v)))
            else:
                # Fallback: use previous value or 0
                alphas.append(alphas[-1] if alphas else 0.0)
        # Convert dot name to underscore key (mm0.img.attn.q_proj -> mm0_img_attn_q_proj)
        layer_key = layer_dot.replace(".", "_")
        layers[layer_key] = {"alphas": alphas}

    sigma_range = [min(sigmas), max(sigmas)]

    schedule = {
        "version": "lut_v1",
        "percentile": "p100_absmax",
        "sigma_range": sigma_range,
        "sigmas": sigmas,
        "layers": layers,
    }

    with open(args.output, "w") as f:
        json.dump(schedule, f, indent=2)

    print(f"LUT schedule: {len(layers)} layers, {len(sigmas)} timesteps")
    print(f"  Sigma range: [{sigma_range[0]:.4f}, {sigma_range[1]:.4f}]")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
