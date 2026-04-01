"""
Compute SmoothQuant per-channel migration scales from calibration data.

SmoothQuant transform: y = W * x = (W * diag(s)) * (diag(1/s) * x)

Per-channel scale s[c] is chosen to balance quantization difficulty between
activations and weights:

    s[c] = s_act[c]^alpha / s_w[c]^(1-alpha)

where:
    s_act[c] = max over timesteps of avg_max[c, t]   (per-channel activation range)
    s_w[c]   = max over output rows of |W[:, c]|     (per-column weight range)
    alpha    ∈ [0.5, 1.0]  (0.5 = balanced; 1.0 = push all to weight side)

After applying s:
    Activations become x' = x / s[c]  →  smaller range, easier to quantize
    Weights become W' = W * s[c]       →  absorbed offline before AdaRound

Usage:
    conda run -n diffusionkit python -m src.compute_smoothquant_scales \\
        --activations-dir calibration_data_512/activations \\
        --weights-dir quantized_weights_w4a8_adaround_poly_p100 \\
        --output smoothquant_scales.json \\
        --alpha 0.5

Output JSON format:
    {
        "alpha": 0.5,
        "layers": {
            "mm0_img_attn_q_proj": [s_0, s_1, ..., s_{C-1}],
            ...
        }
    }
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Layer name conversions
# ---------------------------------------------------------------------------

def calib_name_to_weight_key(calib_name: str):
    """
    Convert calibration layer name to (block_name, weight_path) tuple.

    Calibration naming:   mm{N}_{stream}_{sublayer}_{layer}
      stream  ∈ {img, txt}
      sublayer ∈ {attn, mlp}
      layer   ∈ {q_proj, k_proj, v_proj, o_proj, fc1, fc2}

    Weight path naming:
      img → image_transformer_block
      txt → text_transformer_block

    Examples:
      mm0_img_attn_q_proj → ("mm0", "image_transformer_block.attn.q_proj")
      mm3_txt_mlp_fc1     → ("mm3", "text_transformer_block.mlp.fc1")
    """
    _STREAM_MAP = {
        "img": "image_transformer_block",
        "txt": "text_transformer_block",
    }
    # calib_name: mm{N}_rest
    underscore_idx = calib_name.index("_")
    block = calib_name[:underscore_idx]           # e.g. "mm0"
    rest = calib_name[underscore_idx + 1:]         # e.g. "img_attn_q_proj"

    parts = rest.split("_", 2)  # ["img", "attn", "q_proj"]
    if len(parts) != 3:
        return None, None
    stream, sublayer, layer = parts
    block_type = _STREAM_MAP.get(stream)
    if block_type is None:
        return None, None
    weight_path = f"{block_type}.{sublayer}.{layer}"
    return block, weight_path


def weight_path_to_npz_key(weight_path: str, field: str) -> str:
    """
    Convert weight path to NPZ key suffix.

    weight_path: "image_transformer_block.attn.q_proj"
    field: "weight_int"
    → "image_transformer_block_attn_q_proj__weight_int"
    """
    safe = weight_path.replace(".", "_")
    return f"{safe}__{field}"


# ---------------------------------------------------------------------------
# Load calibration per-channel activation stats
# ---------------------------------------------------------------------------

def load_per_channel_act_stats(activations_dir: Path):
    """
    Load per-channel avg_max across all timestep NPZ files.

    Returns:
        dict: calib_layer_name → np.ndarray of shape (num_channels,)
              Each value is max_t(avg_max[c, t]) — worst-case per-channel range.
    """
    ts_dir = activations_dir / "timestep_stats"
    npz_files = sorted(ts_dir.glob("step_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No step_*.npz files found in {ts_dir}")

    # Accumulate per-channel absmax across all timesteps.
    # Use max(|avg_max|, |avg_min|) to handle channels that are predominantly
    # negative (where avg_max is negative but avg_min captures the true range).
    per_channel_max = {}   # layer_name → running per-channel absmax

    for npz_path in npz_files:
        data = np.load(npz_path)
        for key in data.files:
            if not key.endswith("__avg_max"):
                continue
            layer_name = key[:-9]  # strip "__avg_max"
            arr_max = np.abs(data[key].astype(np.float32))

            # Include avg_min contribution if available
            min_key = key[:-9] + "__avg_min"
            if min_key in data.files:
                arr_min = np.abs(data[min_key].astype(np.float32))
                arr_absmax = np.maximum(arr_max, arr_min)
            else:
                arr_absmax = arr_max

            if layer_name not in per_channel_max:
                per_channel_max[layer_name] = arr_absmax
            else:
                per_channel_max[layer_name] = np.maximum(per_channel_max[layer_name], arr_absmax)

    return per_channel_max


# ---------------------------------------------------------------------------
# Load per-column weight range from quantized weights
# ---------------------------------------------------------------------------

def load_per_column_weight_range(weights_dir: Path):
    """
    Load per-column (input-channel) weight range from quantized weight NPZs.

    For each layer, computes s_w[c] = max over output rows of |W_fp[:, c]|
    where W_fp = weight_int * scale (dequantized).

    Returns:
        dict: calib_layer_name → np.ndarray of shape (in_channels,)
    """
    weights_dir = Path(weights_dir)
    config_path = weights_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Build block → quant_paths mapping
    path_lookup = {}
    for bm in config.get("block_metrics", []):
        bname = bm.get("block_name")
        paths = bm.get("quant_paths", [])
        if bname and paths:
            path_lookup[bname] = paths

    # Reverse mapping: (block, weight_path) → calib_name
    # Build it from path_lookup
    # We also need to build calib_name → (block, weight_path) from all calib layers
    # But we don't have calib layer names here — we'll build it from the weight side

    per_col_range = {}  # calib_layer_name → per-column max abs weight

    npz_dir = weights_dir / "weights"
    for npz_path in sorted(npz_dir.glob("*.npz")):
        block_name = npz_path.stem
        if block_name not in path_lookup:
            continue
        data = np.load(npz_path)
        for weight_path in path_lookup[block_name]:
            wi_key = weight_path_to_npz_key(weight_path, "weight_int")
            sc_key = weight_path_to_npz_key(weight_path, "scale")
            if wi_key not in data.files or sc_key not in data.files:
                continue
            w_int = data[wi_key].astype(np.float32)   # (out, in)
            w_scale = data[sc_key].astype(np.float32)  # (out, 1)
            w_fp = w_int * w_scale                     # dequantized (out, in)
            per_col = np.max(np.abs(w_fp), axis=0)     # (in,)

            # Convert weight_path to calib_name
            calib_name = weight_path_to_calib_name(block_name, weight_path)
            if calib_name is not None:
                per_col_range[calib_name] = per_col

    return per_col_range


def weight_path_to_calib_name(block_name: str, weight_path: str) -> str:
    """
    Convert (block_name, weight_path) → calibration layer name.

    block_name:  "mm0"
    weight_path: "image_transformer_block.attn.q_proj"
    → "mm0_img_attn_q_proj"
    """
    _STREAM_RMAP = {
        "image_transformer_block": "img",
        "text_transformer_block": "txt",
        "transformer_block": "uni",  # unified blocks (if present)
    }
    parts = weight_path.split(".")
    # parts: ["image_transformer_block", "attn", "q_proj"]
    if len(parts) < 3:
        return None
    block_type = parts[0]
    stream = _STREAM_RMAP.get(block_type)
    if stream is None:
        return None
    sublayer = parts[1]
    layer = ".".join(parts[2:]).replace(".", "_")
    return f"{block_name}_{stream}_{sublayer}_{layer}"


# ---------------------------------------------------------------------------
# Compute SmoothQuant scales
# ---------------------------------------------------------------------------

def compute_smoothquant_scales(
    activations_dir: Path,
    weights_dir: Path,
    alpha: float = 0.5,
    scale_clip: float = 0.0,
) -> dict:
    """
    Compute per-channel SmoothQuant migration scales.

    s[c] = s_act[c]^alpha / s_w[c]^(1-alpha)

    Both s_act and s_w are clamped to a minimum of 1e-5 to avoid
    division-by-zero and degenerate scales.

    Parameters
    ----------
    scale_clip : float
        If > 0, clamp each per-channel scale to [1/scale_clip, scale_clip].
        Useful for W4 where extreme scales collapse weight precision.
        Typical value: 32 or 64.  Default 0 = no clamping.

    Returns dict of {layer_name: scales_array}.
    """
    print(f"Loading per-channel activation stats from {activations_dir}...")
    s_act_map = load_per_channel_act_stats(activations_dir)
    print(f"  Found {len(s_act_map)} layers with per-channel activation stats")

    print(f"Loading per-column weight ranges from {weights_dir}...")
    s_w_map = load_per_column_weight_range(weights_dir)
    print(f"  Found {len(s_w_map)} layers with per-column weight ranges")

    scales = {}
    n_shape_mismatch = 0

    for layer_name in sorted(s_act_map):
        s_act = s_act_map[layer_name]
        s_w = s_w_map.get(layer_name)

        if s_w is None:
            # Layer not in quantized weights (e.g. not quantized) — skip
            continue

        if s_act.shape != s_w.shape:
            print(f"  WARNING: shape mismatch for {layer_name}: "
                  f"s_act={s_act.shape} s_w={s_w.shape} — skipping")
            n_shape_mismatch += 1
            continue

        # Clamp to avoid degenerate scales
        s_act = np.maximum(s_act, 1e-5)
        s_w = np.maximum(s_w, 1e-5)

        s = (s_act ** alpha) / (s_w ** (1.0 - alpha))

        # Optional scale clipping (important for W4 where extreme scales
        # collapse weight precision into a few output rows)
        if scale_clip > 0:
            s = np.clip(s, 1.0 / scale_clip, scale_clip)

        scales[layer_name] = s

    print(f"  Computed scales for {len(scales)} layers "
          f"({n_shape_mismatch} shape mismatches skipped)")

    # Summary statistics
    all_scales = np.concatenate([v for v in scales.values()])
    print(f"\n  Scale statistics (all channels):")
    print(f"    min={all_scales.min():.4f}  max={all_scales.max():.4f}  "
          f"mean={all_scales.mean():.4f}  median={np.median(all_scales):.4f}")
    print(f"    >10x outliers: {(all_scales > 10).sum()}")
    print(f"    <0.1x outliers: {(all_scales < 0.1).sum()}")

    return scales


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-dir", type=Path,
                        default=Path("calibration_data_512/activations"),
                        help="Directory containing timestep_stats/ subfolder")
    parser.add_argument("--weights-dir", type=Path,
                        default=Path("quantized_weights_w4a8_adaround_poly_p100"),
                        help="AdaRound output dir (contains config.json + weights/)")
    parser.add_argument("--output", type=Path,
                        default=Path("smoothquant_scales.json"),
                        help="Output JSON file for per-channel scales")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Migration strength: 0.5 = balanced, 1.0 = all to weights")
    parser.add_argument("--scale-clip", type=float, default=0.0,
                        help="Clamp per-channel scales to [1/clip, clip]. "
                             "Recommended for W4 (e.g. --scale-clip 32) to prevent "
                             "extreme outlier scales from collapsing weight precision. "
                             "Default 0 = no clamping.")
    args = parser.parse_args()

    scales = compute_smoothquant_scales(args.activations_dir, args.weights_dir, args.alpha,
                                        scale_clip=args.scale_clip)

    out = {
        "alpha": args.alpha,
        "scale_clip": args.scale_clip,
        "activations_dir": str(args.activations_dir),
        "weights_dir": str(args.weights_dir),
        "n_layers": len(scales),
        "layers": {k: v.tolist() for k, v in scales.items()},
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved scales to {args.output}")


if __name__ == "__main__":
    main()
