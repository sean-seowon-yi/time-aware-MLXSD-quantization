"""
Phase 3c: Apply HTG re-parameterization and quantize the SD3 MMDiT model.

Implements steps 5-9 of Algorithm 2 from arXiv:2503.06930:

    5.  Rescale weight matrices: Ŵ = diag(s) W
    6.  Absorb z_g and s into AdaLN parameters
    7.  Quantize Ŵ and the activation quantizer (HTG layers)
    8.  Quantize remaining layers with standard uniform quantizer

Re-parameterization for SD3 MMDiT
----------------------------------
In vanilla DiT (paper), γ and β are fixed learned parameters, so z_g can be
absorbed directly into the AdaLN bias.

In SD3 MMDiT, γ and β are computed dynamically by adaLN_modulation:
    [β₁, γ₁, α₁, β₂, γ₂, α₂] = Linear(SiLU(c))

Approach: store per-group HTG correction arrays alongside the model.
At inference, a patched cache_modulation_params applies:
    β̂₁_g = (β₁ - z_g_qkv) / s_qkv     (for QKV layers)
    γ̂₁   = (1 + γ₁) / s_qkv - 1       (DiffusionKit stores raw γ; affine_transform uses 1+γ)
    β̂₂_g = (β₂ - z_g_fc1) / s_fc1     (for fc1 layers)
    γ̂₂   = (1 + γ₂) / s_fc1 - 1
    β̂_oproj = (β_oproj - z_g_oproj) / s_oproj  (stored separately)

The weight rescaling Ŵ = W * s[None, :] (column-wise) is applied once here.

Weight Quantization (MLX)
--------------------------
Uses mlx.nn.QuantizedLinear.from_linear(layer, bits=WEIGHT_BITS, group_size=64)
which applies block-wise integer quantization.  The rescaled weight is set on
the layer before quantizing.

Activation Quantizer Ranges
----------------------------
Records per-tensor [min, max] ranges for each HTG target layer's smoothed
activations X̂ = (X - z_g) / s.  These are derived analytically from the
stored stats (no forward passes needed).

Output
------
Saves to output_dir/:
    htg_mmdit_weights.npz       modified (rescaled + quantized) weight arrays
    htg_corrections.npz         z_g, s, group_assignments, timesteps_sorted
    htg_activation_ranges.npz   per-layer per-group [min, max] for static quantizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .htg_config import (
    MODEL_VERSION,
    WEIGHT_BITS,
    QUANTIZATION_GROUP_SIZE,
    DEFAULT_HTG_PARAMS_FILE,
    DEFAULT_OUTPUT_DIR,
)

_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# DiffusionKit helpers
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


def _load_pipeline(model_version: str, local_ckpt: Optional[str]):
    _ensure_diffusionkit_on_path()
    from diffusionkit.mlx import DiffusionPipeline  # type: ignore

    return DiffusionPipeline(
        w16=True, shift=3.0, use_t5=True,
        model_version=model_version,
        low_memory_mode=True, a16=True,
        local_ckpt=local_ckpt,
    )


# ---------------------------------------------------------------------------
# Load HTG params
# ---------------------------------------------------------------------------

def load_htg_params(path: str) -> Dict[str, np.ndarray]:
    """Load the .npz produced by compute_htg_params.py."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def get_layer_ids_from_params(htg_params: Dict[str, np.ndarray]) -> List[str]:
    """Extract unique full layer IDs from HTG params keys."""
    ids = set()
    for key in htg_params:
        if "::" in key:
            ids.add(key.split("::")[0])
    return sorted(ids)


# ---------------------------------------------------------------------------
# Resolve MMDiT layers
# ---------------------------------------------------------------------------

def _resolve_target_layer(mmdit, full_layer_id: str):
    """
    Return (block, layer_type, linear_layer) for a full_layer_id.

    layer_type ∈ {"fc1", "qkv", "oproj"}
    For "qkv", we return a list [q_proj, k_proj, v_proj] since all three
    share the same input activation and scaling vector.
    """
    parts = full_layer_id.split("_")

    if full_layer_id.startswith("mm_"):
        block_idx = int(parts[1])
        stream = parts[2]
        layer_type = parts[3]
        blocks = getattr(mmdit, "multimodal_transformer_blocks", [])
        if block_idx >= len(blocks):
            return None, None, None
        block = blocks[block_idx]
        tb = block.image_transformer_block if stream == "img" else block.text_transformer_block

    elif full_layer_id.startswith("uni_"):
        block_idx = int(parts[1])
        layer_type = parts[2]
        blocks = getattr(mmdit, "unified_transformer_blocks", [])
        if block_idx >= len(blocks):
            return None, None, None
        tb = blocks[block_idx].transformer_block
    else:
        return None, None, None

    if layer_type == "fc1":
        return tb, "fc1", [tb.mlp.fc1]
    elif layer_type == "qkv":
        return tb, "qkv", [tb.attn.q_proj, tb.attn.k_proj, tb.attn.v_proj]
    elif layer_type == "oproj":
        return tb, "oproj", [tb.attn.o_proj]

    return None, None, None


def _resolve_adaln_linear(tb) -> Optional[object]:
    """
    Return the Linear layer inside adaLN_modulation (layers[1]).
    Returns None if not found.
    """
    mod = getattr(tb, "adaLN_modulation", None)
    if mod is None:
        return None
    layers = getattr(mod, "layers", None)
    if layers is None or len(layers) < 2:
        return None
    return layers[1]


# ---------------------------------------------------------------------------
# Core weight rescaling
# ---------------------------------------------------------------------------

def rescale_weight(weight_np: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Rescale target linear layer weights.

    Implements Ŵ = diag(s) W (paper Eq. 6), which in MLX convention
    (weight shape = (Cout, Cin)) translates to multiplying each column
    (input channel) by s[i]:

        weight_hat[j, i] = weight[j, i] * s[i]   for all j (output channels)

    So: weight_hat = weight * s[None, :]

    This is the row-scaling of W_paper = weight.T, which is
    diag(s) @ W_paper = (W_paper.T * s[:, None]).T = weight * s[None, :].
    """
    assert weight_np.ndim == 2, f"Expected 2D weight, got shape {weight_np.shape}"
    assert s.shape[0] == weight_np.shape[1], (
        f"s dim {s.shape[0]} != weight Cin {weight_np.shape[1]}"
    )
    return (weight_np * s[None, :]).astype(weight_np.dtype)


# ---------------------------------------------------------------------------
# Activation quantizer ranges
# ---------------------------------------------------------------------------

def compute_activation_ranges(
    input_stats: Dict[str, Dict[str, np.ndarray]],
    z_g: np.ndarray,
    s: np.ndarray,
    group_assignments: np.ndarray,
    timesteps_sorted: np.ndarray,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute per-group activation quantizer [min, max] ranges for
    the smoothed activation X̂ = (X - z_g) / s.

    For each group g, the range is derived from the constituent timesteps.

    Returns {group_idx: (q_min, q_max)} for static per-tensor quantizer.
    """
    def _t_key(val: float) -> str:
        return f"{val:.6f}"

    G = z_g.shape[0]
    group_ranges: Dict[int, Tuple[float, float]] = {}

    for g in range(G):
        ts_in_group = [
            timesteps_sorted[t]
            for t in range(len(group_assignments))
            if group_assignments[t] == g
        ]
        if not ts_in_group:
            group_ranges[g] = (-1.0, 1.0)
            continue

        q_min = np.inf
        q_max = -np.inf
        for t_val in ts_in_group:
            key = _t_key(float(t_val))
            entry = input_stats.get(key)
            if entry is None:
                continue
            # Smoothed: (X - z_g) / s  →  smoothed_min = (min_raw - z_g) / s
            smoothed_min = (entry["min"].astype(np.float64) - z_g[g]) / np.maximum(s, 1e-8)
            smoothed_max = (entry["max"].astype(np.float64) - z_g[g]) / np.maximum(s, 1e-8)
            q_min = min(q_min, float(smoothed_min.min()))
            q_max = max(q_max, float(smoothed_max.max()))

        group_ranges[g] = (
            q_min if np.isfinite(q_min) else -1.0,
            q_max if np.isfinite(q_max) else 1.0,
        )

    return group_ranges


# ---------------------------------------------------------------------------
# Main reparameterization pipeline
# ---------------------------------------------------------------------------

def reparameterize_and_quantize(
    htg_params_path: str,
    input_stats_path: Optional[str],
    model_version: str,
    weight_bits: int,
    group_size: int,
    output_dir: str,
    local_ckpt: Optional[str],
    quantize_weights: bool,
) -> None:
    """
    Main entry point: load model + HTG params, rescale weights, quantize,
    and save outputs to output_dir.
    """
    import mlx.core as mx
    import mlx.nn as nn

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading HTG parameters from {htg_params_path} ...")
    htg_params = load_htg_params(htg_params_path)
    layer_ids = get_layer_ids_from_params(htg_params)
    timesteps_sorted = htg_params["timesteps_sorted"].astype(np.float64)
    print(f"  Layers with HTG params: {len(layer_ids)}")

    # Optionally load input stats for activation range computation
    input_stats_all: Optional[Dict] = None
    if input_stats_path is not None and Path(input_stats_path).exists():
        from .profile_input_activations import _load_calibration_npz  # reuse loader shape
        raw = np.load(input_stats_path, allow_pickle=True)
        # Parse into {full_layer_id: {t_key: {min, max}}}
        input_stats_all = {}
        for key in raw.files:
            if "::" not in key or key in ("timesteps_unique", "layer_ids"):
                continue
            parts = key.split("::")
            if len(parts) != 3:
                continue
            full_lid, t_part, stat = parts
            t_key = t_part[2:]
            input_stats_all.setdefault(full_lid, {}).setdefault(t_key, {})[stat] = raw[key]

    print(f"\nLoading model ({model_version}) ...")
    pipeline = _load_pipeline(model_version, local_ckpt)
    mmdit = pipeline.mmdit

    # Collect weight modifications and HTG corrections for saving
    weight_updates: Dict[str, np.ndarray] = {}          # path → rescaled weight (numpy)
    corrections: Dict[str, np.ndarray] = {}             # corrections npz payload
    activation_ranges: Dict[str, np.ndarray] = {}       # activation ranges payload

    print(f"\nApplying HTG re-parameterization ({len(layer_ids)} layers) ...")

    for full_layer_id in layer_ids:
        s = htg_params[f"{full_layer_id}::s"].astype(np.float64)
        z_g = htg_params[f"{full_layer_id}::z_g"].astype(np.float64)
        z_t = htg_params[f"{full_layer_id}::z_t"].astype(np.float64)
        group_assignments = htg_params[f"{full_layer_id}::group_assignments"]

        tb, layer_type, linear_layers = _resolve_target_layer(mmdit, full_layer_id)
        if linear_layers is None:
            print(f"  [SKIP] {full_layer_id}: layer not found in model")
            continue

        # --- Weight rescaling ---
        for linear in linear_layers:
            w_np = np.array(linear.weight, dtype=np.float32)
            w_hat = rescale_weight(w_np, s.astype(np.float32))

            # Apply to layer weight in-place (MLX allows direct assignment)
            linear.weight = mx.array(w_hat)
            mx.eval(linear.weight)

            if quantize_weights:
                # Convert to QuantizedLinear (block-wise integer quantization)
                qlinear = nn.QuantizedLinear.from_linear(
                    linear, group_size=group_size, bits=weight_bits
                )
                # Record quantized params for saving
                layer_path = _layer_to_path(full_layer_id, linear)
                weight_updates[f"{layer_path}::weight_scale"] = np.array(qlinear.scales)
                weight_updates[f"{layer_path}::weight_bias"] = np.array(qlinear.biases)
                weight_updates[f"{layer_path}::weight_int"] = np.array(qlinear.weight)
                weight_updates[f"{layer_path}::group_size"] = np.array(group_size)
                weight_updates[f"{layer_path}::bits"] = np.array(weight_bits)
            else:
                layer_path = _layer_to_path(full_layer_id, linear)
                weight_updates[f"{layer_path}::weight"] = w_hat

        # --- Store HTG corrections for AdaLN absorption at inference ---
        corrections[f"{full_layer_id}::z_g"] = z_g.astype(np.float32)
        corrections[f"{full_layer_id}::s"] = s.astype(np.float32)
        corrections[f"{full_layer_id}::group_assignments"] = group_assignments

        # --- Activation quantizer ranges ---
        if input_stats_all is not None and full_layer_id in input_stats_all:
            layer_input_stats = input_stats_all[full_layer_id]
            ranges = compute_activation_ranges(
                layer_input_stats, z_g, s, group_assignments, timesteps_sorted
            )
            for g_idx, (q_min, q_max) in ranges.items():
                activation_ranges[f"{full_layer_id}::g{g_idx}::act_min"] = np.array(q_min, dtype=np.float32)
                activation_ranges[f"{full_layer_id}::g{g_idx}::act_max"] = np.array(q_max, dtype=np.float32)

        print(f"  {full_layer_id}: s_mean={s.mean():.4f}, z_g_max={np.abs(z_g).max():.4f}")

    # --- Save corrections (z_g, s, group_assignments, timesteps) ---
    corrections["timesteps_sorted"] = timesteps_sorted.astype(np.float32)
    corrections["num_groups"] = htg_params["num_groups"]
    corrections["ema_alpha"] = htg_params["ema_alpha"]
    corrections_path = out_path / "htg_corrections.npz"
    np.savez_compressed(str(corrections_path), **corrections)
    print(f"\nSaved HTG corrections to {corrections_path}")

    # --- Save weight updates ---
    if weight_updates:
        weights_path = out_path / "htg_mmdit_weights.npz"
        np.savez_compressed(str(weights_path), **weight_updates)
        print(f"Saved modified weights to {weights_path}")

    # --- Save activation ranges ---
    if activation_ranges:
        ranges_path = out_path / "htg_activation_ranges.npz"
        np.savez_compressed(str(ranges_path), **activation_ranges)
        print(f"Saved activation quantizer ranges to {ranges_path}")

    print("\nRe-parameterization complete.")
    _print_summary(htg_params, layer_ids)


def _layer_to_path(full_layer_id: str, linear) -> str:
    """Construct a string path key for a linear layer."""
    # Use full_layer_id + class name as a unique key
    layer_name = type(linear).__name__
    return f"{full_layer_id}::{id(linear)}"


def _print_summary(htg_params: Dict[str, np.ndarray], layer_ids: List[str]) -> None:
    """Print a concise summary of the HTG parameters."""
    s_vals = []
    z_maxes = []
    for lid in layer_ids:
        s_key = f"{lid}::s"
        z_key = f"{lid}::z_g"
        if s_key in htg_params:
            s_vals.extend(htg_params[s_key].tolist())
        if z_key in htg_params:
            z_maxes.append(float(np.abs(htg_params[z_key]).max()))

    if s_vals:
        s_arr = np.array(s_vals)
        print(f"  s statistics  — mean: {s_arr.mean():.4f}, max: {s_arr.max():.4f}, min: {s_arr.min():.4f}")
    if z_maxes:
        print(f"  |z_g| max per layer — mean: {np.mean(z_maxes):.4f}, max: {np.max(z_maxes):.4f}")

    layer_types = {"fc1": 0, "qkv": 0, "oproj": 0}
    for lid in layer_ids:
        for t in layer_types:
            if lid.endswith(f"_{t}"):
                layer_types[t] += 1
    print("  Layer type counts:", ", ".join(f"{t}={n}" for t, n in layer_types.items()))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3c: Apply HTG re-parameterization and quantize SD3 / MMDiT. "
            "Rescales target linear weights by s, stores per-group z_g corrections, "
            "and optionally applies MLX block-wise weight quantization."
        )
    )
    parser.add_argument(
        "--htg-params", type=str, default=DEFAULT_HTG_PARAMS_FILE,
        help="Path to HTG parameters .npz from compute_htg_params.py",
    )
    parser.add_argument(
        "--input-stats", type=str, default=None,
        help="Optional path to input activation stats for activation range computation",
    )
    parser.add_argument(
        "--model-version", type=str, default=MODEL_VERSION,
        help="DiffusionKit model key (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-bits", type=int, default=WEIGHT_BITS,
        choices=[4, 8],
        help="Weight quantization bit-width (default: %(default)s)",
    )
    parser.add_argument(
        "--group-size", type=int, default=QUANTIZATION_GROUP_SIZE,
        help="MLX block-wise quantization group size (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--local-ckpt", type=str, default=None,
        help="Optional local checkpoint path",
    )
    parser.add_argument(
        "--no-quantize", action="store_true", default=False,
        help="Skip weight quantization (just rescale and save corrections)",
    )

    args = parser.parse_args()

    reparameterize_and_quantize(
        htg_params_path=args.htg_params,
        input_stats_path=args.input_stats,
        model_version=args.model_version,
        weight_bits=args.weight_bits,
        group_size=args.group_size,
        output_dir=args.output_dir,
        local_ckpt=args.local_ckpt,
        quantize_weights=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
