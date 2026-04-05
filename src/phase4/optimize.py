"""Phase 4 AdaRound optimisation: layer-wise weight rounding.

For each target linear layer:
  1. Compute symmetric per-group scale from the FP16 balanced weights.
  2. Initialise learnable rounding offsets alpha = 0  (sigmoid(0) = 0.5).
  3. Minimise reconstruction MSE:
         L = mean( (X_eff @ W_fp.T  -  X_eff @ W_soft.T)^2 )
     where  W_soft = (floor(W/s) + sigmoid(alpha)) * s
     and    X_eff  = X * b_inv   for o_proj / fc2   (b_inv from phase 2)
                   = X           for all other layers
  4. Hard-binarise: round_delta = (alpha > 0).astype(int8)
  5. Build nn.QuantizedLinear via repack.build_quantized_linear.
  6. Wrap in phase2 W4A8Linear (preserving b_inv and per_token flag).

No block-level backprop: gradients flow only through the linear op, so there
is no risk of NaN from attention softmax backward on extreme activations.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..phase2.quantize import W4A8Linear
from .repack import build_quantized_linear

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scale / floor helpers
# ---------------------------------------------------------------------------

def _compute_quant_params(
    W: np.ndarray,
    bits: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-group symmetric scale, floor(W/s), and scale expanded to W shape.

    Returns
    -------
    scales      : (d_out, n_groups)  float32
    floor_w     : (d_out, d_in)      float32  — integer part of W/s
    scales_exp  : (d_out, d_in)      float32  — scale broadcast to weight shape
    """
    d_out, d_in = W.shape
    n_groups = d_in // group_size
    qmax = 2 ** (bits - 1) - 1

    W_g = W.reshape(d_out, n_groups, group_size)                # (d_out, n_grp, gs)
    scales = np.max(np.abs(W_g), axis=-1) / qmax                # (d_out, n_grp)
    scales = np.maximum(scales, 1e-8).astype(np.float32)

    scales_exp = np.repeat(scales, group_size, axis=1)          # (d_out, d_in)
    floor_w = np.floor(W / scales_exp).astype(np.float32)       # (d_out, d_in)
    return scales, floor_w, scales_exp


# ---------------------------------------------------------------------------
# AdaRound optimisation (single layer)
# ---------------------------------------------------------------------------

def _adaround_layer(
    W_fp: np.ndarray,
    scales_exp: np.ndarray,
    floor_w: np.ndarray,
    inputs: np.ndarray,
    b_inv: np.ndarray | None,
    bits: int,
    config: dict,
) -> np.ndarray:
    """Run AdaRound for one linear layer.

    Args:
        W_fp      : (d_out, d_in)        float32 — FP16 balanced weights
        scales_exp: (d_out, d_in)        float32 — per-group scale, expanded
        floor_w   : (d_out, d_in)        float32 — floor(W/s)
        inputs    : (n_calls, T, d_in)   float16 — collected layer inputs
        b_inv     : (d_in,)              float32 or None
        bits      : weight bit width
        config    : PHASE4_CONFIG

    Returns
    -------
    w_int : (d_out, d_in) int8 — optimal rounded integer weights in [-qmax, qmax]
    """
    n_iters = config["n_iters"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    qmax = 2 ** (bits - 1) - 1

    W_fp_mx = mx.array(W_fp, dtype=mx.float32)
    scales_mx = mx.array(scales_exp, dtype=mx.float32)
    floor_mx = mx.array(floor_w, dtype=mx.float32)
    b_inv_mx = mx.array(b_inv, dtype=mx.float32) if b_inv is not None else None

    # Flatten inputs to 2D: (n_tokens_total, d_in)
    n, t, d = inputs.shape
    X_all = inputs.reshape(n * t, d).astype(np.float32)
    n_total = X_all.shape[0]

    alpha = mx.zeros_like(W_fp_mx)  # (d_out, d_in)

    # Manual Adam state
    m = mx.zeros_like(alpha)
    v = mx.zeros_like(alpha)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    # Store X in a closure variable (avoids tracing it as a gradient argument)
    X_batch_holder: list[mx.array] = [mx.zeros((batch_size, d), dtype=mx.float32)]

    def loss_fn(alpha: mx.array) -> mx.array:
        w_soft = (floor_mx + mx.sigmoid(alpha)) * scales_mx   # (d_out, d_in)
        X = X_batch_holder[0]                                   # (B, d_in)
        if b_inv_mx is not None:
            X = X * b_inv_mx
        diff = X @ (W_fp_mx - w_soft).T                        # (B, d_out)
        return mx.mean(diff ** 2)

    loss_and_grad = mx.value_and_grad(loss_fn)

    for step in range(1, n_iters + 1):
        idx = np.random.randint(0, n_total, size=batch_size)
        X_batch_holder[0] = mx.array(X_all[idx], dtype=mx.float32)

        loss, grad = loss_and_grad(alpha)

        # Adam update
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        alpha = alpha - lr * m_hat / (mx.sqrt(v_hat) + eps_adam)
        mx.eval(alpha, m, v)

        if step % 200 == 0 or step == 1:
            logger.debug("  step %4d  loss=%.6f", step, float(loss.item()))

    # Hard binarise
    alpha_np = np.array(alpha)
    round_delta = (alpha_np > 0).astype(np.int8)
    w_int = np.array(floor_w, dtype=np.int8) + round_delta
    w_int = np.clip(w_int, -qmax, qmax).astype(np.int8)
    return w_int


# ---------------------------------------------------------------------------
# Model-wide optimisation
# ---------------------------------------------------------------------------

def optimize_all_layers(
    pipeline,
    registry: list[dict],
    calibration: dict,
    data_dir: Path,
    phase2_meta: dict,
    config: dict,
    output_dir: Path,
) -> None:
    """Run AdaRound for all target layers and save a phase2-compatible checkpoint.

    Layers for which no calibration NPZ is found are left at their current
    (CSB-balanced FP16) weights and quantised with RTN as a fallback.

    Args:
        pipeline   : Loaded pipeline with CSB-balanced FP16 model.
        registry   : Full layer registry (with ``module`` refs).
        calibration: Phase 2 calibration dict (balancing_vectors, b_inv_layers…).
        data_dir   : Directory containing ``<layer_name>.npz`` files from collect.
        phase2_meta: Loaded ``quantize_config.json`` from phase 2.
        config     : PHASE4_CONFIG.
        output_dir : Where to write the new checkpoint.
    """
    from mlx.utils import tree_flatten
    from ..phase2.quantize import (
        _navigate_to_parent,
        patch_pipeline_for_quantized_inference,
        save_quantized_model,
    )

    bits = config["bits"]
    group_size = config["group_size"]
    b_inv_set = set(calibration["b_inv_layers"])
    b_inv_store = {
        name: (1.0 / calibration["balancing_vectors"][name]).astype(np.float32)
        for name in b_inv_set
    }
    mean_rhos = calibration.get("mean_rhos", {})
    rho_threshold = phase2_meta.get("per_token_rho_threshold", 0.5)
    exclude = set(phase2_meta.get("exclude_layers", ["context_embedder"]))

    layer_meta: dict[str, dict] = {}
    n_adaround = 0
    n_rtn = 0

    for entry in registry:
        name = entry["name"]
        if name in exclude:
            continue

        linear = entry.get("module")
        if linear is None:
            continue

        family = entry.get("family", "")
        layer_bits = (
            phase2_meta.get("final_layer_bits", bits)
            if family == "final_linear" else bits
        )
        if layer_bits >= 16:
            continue

        W_fp = np.array(linear.weight, dtype=np.float32)   # (d_out, d_in)
        d_out, d_in = W_fp.shape
        bias = np.array(linear.bias) if getattr(linear, "bias", None) is not None else None
        b_inv = b_inv_store.get(name)
        per_token = mean_rhos.get(name, 0.0) > rho_threshold

        scales, floor_w, scales_exp = _compute_quant_params(W_fp, layer_bits, group_size)

        # Try to load calibration data
        npz_path = data_dir / f"{name}.npz"
        if npz_path.exists():
            t0 = time.time()
            inputs = np.load(str(npz_path))["inputs"]    # (n_calls, T, d_in)
            w_int = _adaround_layer(
                W_fp, scales_exp, floor_w, inputs, b_inv, layer_bits, config,
            )
            elapsed = time.time() - t0
            logger.info(
                "AdaRound %-45s  %d samples  %.1f s",
                name, inputs.shape[0], elapsed,
            )
            n_adaround += 1
        else:
            # RTN fallback
            logger.warning("No calibration data for %s — using RTN", name)
            qmax = 2 ** (layer_bits - 1) - 1
            w_int = np.clip(np.round(W_fp / scales_exp), -qmax, qmax).astype(np.int8)
            n_rtn += 1

        b_inv_arr = b_inv if b_inv is not None else None
        qlinear = build_quantized_linear(w_int, scales, bias, layer_bits, group_size)
        b_inv_mx = mx.array(b_inv_arr, dtype=mx.float32) if b_inv_arr is not None else None
        w4a8 = W4A8Linear(qlinear, b_inv=b_inv_mx, per_token=per_token)

        parent, attr = _navigate_to_parent(pipeline.mmdit, name)
        setattr(parent, attr, w4a8)
        mx.eval(pipeline.mmdit.parameters())

        layer_meta[name] = {
            "d_in": d_in,
            "d_out": d_out,
            "has_bias": bias is not None,
            "bits": layer_bits,
            "has_b_inv": b_inv is not None,
            "per_token": per_token,
        }

    logger.info(
        "Optimisation complete: %d AdaRound, %d RTN fallback", n_adaround, n_rtn,
    )

    patch_pipeline_for_quantized_inference(pipeline)

    p2_cfg = {
        "model_version": phase2_meta["model_version"],
        "group_size": group_size,
        "bits": bits,
        "a_bits": phase2_meta.get("a_bits", 8),
        "final_layer_bits": phase2_meta.get("final_layer_bits", bits),
        "alpha": phase2_meta["alpha"],
        "qkv_method": phase2_meta["qkv_method"],
        "ssc_tau": phase2_meta.get("ssc_tau", 1.0),
        "per_token_rho_threshold": rho_threshold,
        "exclude_layers": list(exclude),
        "weight_quant": "adaround",
    }
    save_quantized_model(
        pipeline.mmdit,
        output_dir,
        p2_cfg,
        layer_meta,
        list(b_inv_set),
    )
    logger.info("Saved AdaRound checkpoint to %s", output_dir)
