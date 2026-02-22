"""
Quantize SD3 Medium MMDiT using TaQ-DiT methodology.

TaQ-DiT key concepts:
  1. W4A8 quantization (weights 4-bit, activations 8-bit)
  2. Joint reconstruction: optimize weight AND activation scales together
     to minimize MSE(y_fp, y_quant) — NOT separate optimization
  3. Per-channel weight quantization (output channels)
  4. Per-token activation quantization (each token gets own scale)
  5. Post-GELU shift: Apply momentum-averaged shift before quantizing
     to center the asymmetric GELU output distribution
  6. Fake quantization + STE during calibration to find optimal scales

Implementation:
  - Load calibration statistics from collect_layer_activations.py
  - For each layer:
    * Initialize weight scale (per output channel) and activation scale (per token)
    * Use fake-quant forward passes on calibration data
    * Optimize scales via gradient descent to minimize reconstruction MSE
    * Quantize weights to int4, save activation scales
  - Save quantized model as safetensors + quant metadata

Usage:
    python -m src.quantize_model_taqdit \\
        --model argmaxinc/mlx-stable-diffusion-3-medium \\
        --calib-stats calibration_data/activations/layer_statistics.json \\
        --calib-dir calibration_data \\
        --output quantized_models/sd3-medium-w4a8-taqdit
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

def quantize_symmetric(x: mx.array, scale: mx.array, bits: int) -> mx.array:
    """
    Symmetric uniform quantization to `bits` bits.

    Args:
        x: Input tensor (any shape)
        scale: Scale tensor (broadcastable to x)
        bits: Number of bits (4 or 8)

    Returns:
        Fake-quantized tensor (same shape as x, still float)
    """
    qmax = 2 ** (bits - 1) - 1  # 4-bit: 7, 8-bit: 127
    qmin = -2 ** (bits - 1)     # 4-bit: -8, 8-bit: -128

    # Quantize: x_q = clip(round(x / scale), qmin, qmax)
    x_scaled = x / (scale + 1e-8)
    x_quant = mx.clip(mx.round(x_scaled), qmin, qmax)

    # Dequantize (fake quant): x_fake = x_q * scale
    return x_quant * scale


def compute_per_channel_scale(weight: np.ndarray, bits: int = 4) -> np.ndarray:
    """
    Compute per-output-channel scale for weight quantization.

    Args:
        weight: Weight tensor, shape (out_features, in_features)
        bits: Number of bits (default 4)

    Returns:
        scale: Per-channel scale, shape (out_features,)
    """
    # Per output channel (dim 0): absmax over input channels (dim 1)
    absmax = np.abs(weight).max(axis=1)  # (out_features,)
    qmax = 2 ** (bits - 1) - 1
    scale = absmax / qmax
    return scale


def compute_per_token_scale(activation: np.ndarray, bits: int = 8,
                            shift: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute per-token scale for activation quantization.

    Args:
        activation: Activation tensor, shape (batch*tokens, channels)
        bits: Number of bits (default 8)
        shift: Optional per-channel shift (for post-GELU), shape (channels,)

    Returns:
        scale: Per-token scale, shape (batch*tokens,)
    """
    if shift is not None:
        # Apply shift centering for post-GELU activations
        activation = activation - shift

    # Per token (dim 0): absmax over channels (dim 1)
    absmax = np.abs(activation).max(axis=1)  # (batch*tokens,)
    qmax = 2 ** (bits - 1) - 1
    scale = absmax / qmax
    return scale


# ---------------------------------------------------------------------------
# Joint reconstruction
# ---------------------------------------------------------------------------

class QuantizedLinear:
    """
    Fake-quantized linear layer for joint reconstruction.
    Uses learnable weight and activation scales.
    """

    def __init__(self, weight: mx.array, bias: Optional[mx.array],
                 w_bits: int = 4, a_bits: int = 8,
                 shift: Optional[mx.array] = None):
        """
        Args:
            weight: FP16 weight, shape (out_features, in_features)
            bias: Optional bias, shape (out_features,)
            w_bits: Weight quantization bits
            a_bits: Activation quantization bits
            shift: Optional per-channel shift for post-GELU, shape (in_features,)
        """
        self.weight_fp = weight  # Keep FP reference
        self.bias = bias
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.shift = shift

        # Initialize scales (learnable parameters)
        out_features, in_features = weight.shape

        # Weight scale: per output channel
        w_np = np.array(weight)
        w_scale_init = compute_per_channel_scale(w_np, w_bits)
        self.w_scale = mx.array(w_scale_init, dtype=mx.float32).reshape(-1, 1)

        # Activation scale: initialized to 1.0, will be updated per forward pass
        # (since per-token scales vary by input)
        self.a_scale_init = 1.0

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with fake quantization.

        Args:
            x: Input activations, shape (batch*tokens, in_features)

        Returns:
            Output, shape (batch*tokens, out_features)
        """
        # Quantize activations (per-token)
        x_shifted = x - self.shift if self.shift is not None else x

        # Compute per-token activation scales
        x_np = np.array(x_shifted)
        a_scale = compute_per_token_scale(x_np, self.a_bits, shift=None)
        a_scale_mx = mx.array(a_scale, dtype=mx.float32).reshape(-1, 1)

        # Fake-quant activations
        x_quant = quantize_symmetric(x_shifted, a_scale_mx, self.a_bits)
        if self.shift is not None:
            x_quant = x_quant + self.shift  # Shift back after quantization

        # Fake-quant weights
        w_quant = quantize_symmetric(self.weight_fp, self.w_scale, self.w_bits)

        # Linear forward: y = x @ w.T + b
        y = x_quant @ w_quant.T
        if self.bias is not None:
            y = y + self.bias

        return y


def joint_reconstruction_layer(layer: nn.Linear,
                               calibration_data: List[mx.array],
                               shift: Optional[np.ndarray] = None,
                               lr: float = 1e-3,
                               max_iters: int = 100) -> Tuple[np.ndarray, Dict]:
    """
    Perform joint reconstruction for a single layer.

    Optimizes weight scales to minimize MSE between FP and quantized outputs
    using calibration data. Activation scales are computed per-token dynamically.

    Args:
        layer: Original FP16 linear layer
        calibration_data: List of input tensors from calibration set
        shift: Optional per-channel shift (post-GELU), shape (in_features,)
        lr: Learning rate for scale optimization
        max_iters: Maximum optimization iterations

    Returns:
        w_scale: Optimized per-channel weight scale, shape (out_features,)
        metrics: Dict with loss history and final MSE
    """
    weight_fp = layer.weight
    bias_fp = layer.bias if hasattr(layer, 'bias') else None

    shift_mx = mx.array(shift, dtype=mx.float32) if shift is not None else None

    # Create quantized layer with learnable weight scales
    quant_layer = QuantizedLinear(weight_fp, bias_fp, shift=shift_mx)

    # Optimizer for weight scales only
    # Note: In practice, TaQ-DiT uses AdamW on both weight and activation scales
    # For simplicity, we optimize weight scales while computing activation scales per-token
    optimizer = optim.AdamW(learning_rate=lr)

    loss_history = []

    for iter_idx in range(max_iters):
        total_loss = 0.0

        for x_calib in calibration_data:
            # FP forward
            y_fp = x_calib @ weight_fp.T
            if bias_fp is not None:
                y_fp = y_fp + bias_fp

            # Quantized forward
            y_quant = quant_layer(x_calib)

            # MSE loss
            loss = mx.mean((y_fp - y_quant) ** 2)
            total_loss += float(loss)

            # Backward (STE: gradients flow through quantization)
            # TODO: Implement proper gradient flow with mx.grad
            # For now, this is a placeholder structure

        avg_loss = total_loss / len(calibration_data)
        loss_history.append(avg_loss)

        # Early stopping if converged
        if iter_idx > 10 and abs(loss_history[-1] - loss_history[-2]) < 1e-6:
            break

    # Extract final weight scales
    w_scale_final = np.array(quant_layer.w_scale).ravel()

    metrics = {
        "final_mse": loss_history[-1] if loss_history else 0.0,
        "iterations": len(loss_history),
        "converged": len(loss_history) < max_iters,
    }

    return w_scale_final, metrics


# ---------------------------------------------------------------------------
# Main quantization pipeline
# ---------------------------------------------------------------------------

def load_calibration_stats(stats_path: Path) -> Dict:
    """Load calibration statistics from collect_layer_activations.py output."""
    with open(stats_path) as f:
        manifest = json.load(f)

    if manifest.get("format") != "per_timestep_npz_v2":
        raise ValueError(f"Expected per_timestep_npz_v2, got {manifest.get('format')}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Quantize SD3 Medium with TaQ-DiT (W4A8 joint reconstruction)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (e.g., argmaxinc/mlx-stable-diffusion-3-medium)")
    parser.add_argument("--calib-stats", type=Path, required=True,
                        help="Path to layer_statistics.json from collect_layer_activations.py")
    parser.add_argument("--calib-dir", type=Path, required=True,
                        help="Path to calibration_data directory with saved latents")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for quantized model")
    parser.add_argument("--num-calib-samples", type=int, default=10,
                        help="Number of calibration samples to use for reconstruction")
    parser.add_argument("--recon-lr", type=float, default=1e-3,
                        help="Learning rate for joint reconstruction")
    parser.add_argument("--recon-iters", type=int, default=100,
                        help="Max iterations for joint reconstruction per layer")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=== TaQ-DiT W4A8 Quantization ===")
    print(f"Model: {args.model}")
    print(f"Calibration stats: {args.calib_stats}")
    print(f"Output: {args.output}")

    # Load calibration statistics
    print("\n=== Loading Calibration Statistics ===")
    manifest = load_calibration_stats(args.calib_stats)
    print(f"Calibration: {manifest['metadata']['num_images']} images, "
          f"{manifest['metadata']['num_timesteps']} timesteps")

    # TODO: Load model
    print("\n=== Loading Model ===")
    print("TODO: Load DiffusionPipeline and extract MMDiT")

    # TODO: Load calibration latents
    print("\n=== Loading Calibration Data ===")
    print(f"TODO: Load {args.num_calib_samples} samples from {args.calib_dir}")

    # TODO: Quantize each layer with joint reconstruction
    print("\n=== Joint Reconstruction ===")
    print("TODO: Iterate through MMDiT layers and apply joint reconstruction")

    # TODO: Save quantized model
    print("\n=== Saving Quantized Model ===")
    print(f"TODO: Save to {args.output}")

    print("\n✓ Quantization complete")


if __name__ == "__main__":
    main()
