"""
Absorb SmoothQuant per-channel scales into stored AdaRound weights.

Transform: W' = W * diag(s)   (multiply each input channel column by s[c])

Procedure per layer:
  1. Dequantize:  W_fp = weight_int.astype(float32) * scale   (per-output-channel scale)
  2. Apply SQ:    W_smooth = W_fp * s[None, :]                (broadcast over output rows)
  3. Requantize:  W_smooth_int, new_scale = quantize_per_output_channel(W_smooth, bits=4)
  4. Save new NPZ with updated weight_int, scale, and a_scale (preserved from source)

The activation-side inverse scale (1/s[c]) is stored in the scales JSON and
applied at inference time in load_adaround_model.py.

Note: AdaRound alpha (soft-quantization parameter) is **not** preserved —
after absorbing SQ scales the weight distribution shifts, so alpha becomes
invalid. A fresh AdaRound run should be performed on the smoothed weights.

Usage:
    conda run -n diffusionkit python -m src.apply_smoothquant \\
        --weights-dir quantized_weights_w4a8_adaround_poly_p100 \\
        --scales smoothquant_scales.json \\
        --output-dir quantized_weights_w4a8_smoothquant_absorbed
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Re-use naming utilities from compute_smoothquant_scales
# ---------------------------------------------------------------------------

from src.compute_smoothquant_scales import (
    weight_path_to_calib_name,
    weight_path_to_npz_key,
)


# ---------------------------------------------------------------------------
# Per-output-channel quantization (W4)
# ---------------------------------------------------------------------------

def quantize_per_output_channel(w_fp: np.ndarray, bits: int = 4) -> tuple:
    """
    Symmetric per-output-channel quantization.

    w_fp shape: (out_channels, in_channels)

    Returns:
        w_int : int8 ndarray, shape (out_channels, in_channels)
        scale : float16 ndarray, shape (out_channels, 1)
    """
    qmax = 2 ** (bits - 1) - 1
    per_row_max = np.max(np.abs(w_fp), axis=1, keepdims=True)  # (out, 1)
    scale = (per_row_max / qmax).astype(np.float16)
    scale_safe = np.where(scale == 0, np.float16(1.0), scale)
    w_int = np.round(w_fp / scale_safe.astype(np.float32))
    w_int = np.clip(w_int, -qmax, qmax).astype(np.int8)
    return w_int, scale


# ---------------------------------------------------------------------------
# Apply SQ scales to a weights directory
# ---------------------------------------------------------------------------

def apply_smoothquant(
    weights_dir: Path,
    scales: dict,
    output_dir: Path,
    bits: int = 4,
):
    """
    Create a new weights directory with SQ scales absorbed into weights.

    Parameters
    ----------
    weights_dir : Path
        Source AdaRound output dir (contains config.json + weights/).
    scales : dict
        Loaded from smoothquant_scales.json: {"layers": {calib_name: [s0, s1, ...]}}
    output_dir : Path
        Destination directory. Created if absent.
    bits : int
        Weight quantization bits (default 4).
    """
    weights_dir = Path(weights_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config.json with annotation
    config_path = weights_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["smoothquant"] = True
    config["smoothquant_alpha"] = scales.get("alpha", "unknown")
    config["smoothquant_source"] = str(weights_dir)
    config["adaround_alpha_preserved"] = False  # SQ shifts distribution; alpha invalid

    out_config_path = output_dir / "config.json"
    with open(out_config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Build block → quant_paths from config
    path_lookup = {}
    for bm in config.get("block_metrics", []):
        bname = bm.get("block_name")
        paths = bm.get("quant_paths", [])
        if bname and paths:
            path_lookup[bname] = paths

    layer_scales = scales.get("layers", {})

    (output_dir / "weights").mkdir(exist_ok=True)

    n_absorbed = 0
    n_skipped = 0

    npz_dir = weights_dir / "weights"
    for npz_path in sorted(npz_dir.glob("*.npz")):
        block_name = npz_path.stem
        data = dict(np.load(npz_path))
        new_data = dict(data)  # start with a copy

        known_paths = path_lookup.get(block_name, [])
        for weight_path in known_paths:
            wi_key = weight_path_to_npz_key(weight_path, "weight_int")
            sc_key = weight_path_to_npz_key(weight_path, "scale")

            if wi_key not in data or sc_key not in data:
                continue

            calib_name = weight_path_to_calib_name(block_name, weight_path)
            if calib_name not in layer_scales:
                n_skipped += 1
                continue

            s = np.array(layer_scales[calib_name], dtype=np.float32)  # (in_channels,)

            w_int = data[wi_key].astype(np.float32)   # (out, in)
            w_scale = data[sc_key].astype(np.float32)  # (out, 1)
            w_fp = w_int * w_scale                     # dequantized

            # Absorb SQ scale: multiply each input channel column by s[c]
            w_smooth = w_fp * s[None, :]               # (out, in)

            # Re-quantize
            w_int_new, scale_new = quantize_per_output_channel(w_smooth, bits=bits)

            new_data[wi_key] = w_int_new
            new_data[sc_key] = scale_new

            # Drop alpha — no longer valid after weight distribution shift
            alpha_key = weight_path_to_npz_key(weight_path, "alpha")
            if alpha_key in new_data:
                del new_data[alpha_key]

            n_absorbed += 1

        out_npz_path = output_dir / "weights" / npz_path.name
        np.savez(out_npz_path, **new_data)

    print(f"  Absorbed SQ scales into {n_absorbed} layers "
          f"({n_skipped} layers missing from scales → kept original)")
    print(f"  Output written to {output_dir}")
    return n_absorbed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path,
                        default=Path("quantized_weights_w4a8_adaround_poly_p100"),
                        help="Source AdaRound output directory")
    parser.add_argument("--scales", type=Path,
                        default=Path("smoothquant_scales.json"),
                        help="SmoothQuant scales JSON (from compute_smoothquant_scales.py)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("quantized_weights_w4a8_smoothquant_absorbed"),
                        help="Output directory for SQ-absorbed weights")
    parser.add_argument("--bits", type=int, default=4,
                        help="Weight quantization bits (default 4)")
    args = parser.parse_args()

    print(f"Loading SQ scales from {args.scales}...")
    with open(args.scales) as f:
        scales = json.load(f)
    print(f"  alpha={scales.get('alpha')}  layers={scales.get('n_layers')}")

    print(f"Absorbing SQ scales into weights from {args.weights_dir}...")
    apply_smoothquant(args.weights_dir, scales, args.output_dir, bits=args.bits)


if __name__ == "__main__":
    main()
