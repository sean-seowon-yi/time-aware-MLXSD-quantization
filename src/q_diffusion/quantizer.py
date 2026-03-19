"""Quantization primitives: scale computation, fake quantization, AdaRound math."""

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Scale computation
# ---------------------------------------------------------------------------

def compute_weight_scale(
    weight: mx.array,
    bits: int,
    symmetric: bool = True,
    per_channel: bool = True,
) -> mx.array:
    """Compute per-channel (or per-tensor) scale for weight quantization.

    Args:
        weight: shape (out_features, in_features)
        bits: quantization bit width
        symmetric: if True, use symmetric range [-2^(b-1), 2^(b-1)-1]
        per_channel: if True, one scale per output channel

    Returns:
        scale: shape (out_features, 1) if per_channel else scalar
    """
    if per_channel:
        alpha = mx.max(mx.abs(weight), axis=1, keepdims=True)
    else:
        alpha = mx.max(mx.abs(weight))

    if symmetric:
        qmax = (1 << (bits - 1)) - 1  # 2^(b-1) - 1
        scale = alpha / qmax
    else:
        qmax = (1 << bits) - 1
        scale = alpha / qmax

    # Prevent division by zero
    scale = mx.maximum(scale, mx.array(1e-8))
    return scale


def compute_activation_scale(
    alpha: mx.array,
    bits: int,
    symmetric: bool = True,
) -> mx.array:
    """Derive quantization scale from clipping range alpha.

    Args:
        alpha: clipping range (scalar or per-tensor)
        bits: quantization bit width
        symmetric: if True, range is [-alpha, +alpha]

    Returns:
        scale: quantization step size
    """
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
    else:
        qmax = (1 << bits) - 1
    scale = alpha / qmax
    return mx.maximum(scale, mx.array(1e-8))


# ---------------------------------------------------------------------------
# Fake quantization (STE)
# ---------------------------------------------------------------------------

def fake_quantize_symmetric(x: mx.array, scale: mx.array, bits: int) -> mx.array:
    """Symmetric fake quantization with STE (straight-through estimator).

    Quantize: x_hat = scale * clamp(round(x / scale), -qmax, qmax)
    Gradient passes through round and clamp via STE.
    """
    qmax = (1 << (bits - 1)) - 1
    x_scaled = x / scale
    # STE: use stop_gradient on the rounding residual
    x_rounded = x_scaled + mx.stop_gradient(mx.round(x_scaled) - x_scaled)
    x_clamped = mx.clip(x_rounded, -qmax, qmax)
    return x_clamped * scale


def fake_quantize_asymmetric(
    x: mx.array,
    scale: mx.array,
    zero_point: mx.array,
    bits: int,
) -> mx.array:
    """Asymmetric fake quantization with STE.

    Used for fc2 inputs (post-GELU, non-negative).
    """
    qmax = (1 << bits) - 1
    x_scaled = x / scale + zero_point
    x_rounded = x_scaled + mx.stop_gradient(mx.round(x_scaled) - x_scaled)
    x_clamped = mx.clip(x_rounded, 0, qmax)
    return (x_clamped - zero_point) * scale


# ---------------------------------------------------------------------------
# AdaRound
# ---------------------------------------------------------------------------

def init_v_from_weights(weight: mx.array, scale: mx.array) -> mx.array:
    """Initialize AdaRound V parameters from the fractional part of w/s.

    V is initialized so that sigmoid(V) ≈ fractional part,
    giving round-to-nearest as the starting point.

    Returns:
        v: same shape as weight, initialized for rectified sigmoid
    """
    w_scaled = weight / scale
    frac = w_scaled - mx.floor(w_scaled)  # fractional part in [0, 1)

    # Invert the rectified sigmoid: h(V) = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    # where zeta = 1.1, gamma = -0.1
    # To get h(V) ≈ frac, we need sigmoid(V) ≈ (frac - gamma) / (zeta - gamma)
    zeta = 1.1
    gamma = -0.1
    # Clamp to avoid log(0) or log(inf)
    target = (frac - gamma) / (zeta - gamma)
    target = mx.clip(target, 1e-6, 1.0 - 1e-6)
    # Inverse sigmoid: V = log(target / (1 - target))
    v = mx.log(target / (1.0 - target))
    return v


def adaround_soft_round(v: mx.array) -> mx.array:
    """Compute soft rounding offset h(V) using rectified sigmoid.

    h(V) = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    where zeta = 1.1, gamma = -0.1
    """
    zeta = 1.1
    gamma = -0.1
    h = mx.clip(mx.sigmoid(v) * (zeta - gamma) + gamma, 0.0, 1.0)
    return h


def adaround_quantize(
    weight: mx.array,
    v: mx.array,
    scale: mx.array,
    bits: int,
) -> mx.array:
    """Fake-quantize weights using AdaRound soft rounding.

    w_hat = scale * clamp(floor(w/s) + h(V), -qmax, qmax)
    """
    qmax = (1 << (bits - 1)) - 1
    w_scaled = weight / scale
    w_floor = mx.floor(w_scaled)
    h = adaround_soft_round(v)
    w_rounded = w_floor + h
    w_clamped = mx.clip(w_rounded, -qmax, qmax)
    return w_clamped * scale


def adaround_hard_quantize(
    weight: mx.array,
    v: mx.array,
    scale: mx.array,
    bits: int,
) -> mx.array:
    """Hard-quantize weights using frozen V (threshold at 0.5)."""
    qmax = (1 << (bits - 1)) - 1
    w_scaled = weight / scale
    w_floor = mx.floor(w_scaled)
    h = (adaround_soft_round(v) >= 0.5).astype(weight.dtype)
    w_rounded = w_floor + h
    w_clamped = mx.clip(w_rounded, -qmax, qmax)
    return w_clamped * scale


def adaround_reg(v: mx.array, beta: float) -> mx.array:
    """AdaRound regularization loss pushing V toward 0 or 1.

    reg = mean(1 - |2 * h(V) - 1|^beta)

    Small beta = gentle push (allows exploration).
    Large beta = strong push toward hard rounding.
    """
    h = adaround_soft_round(v)
    reg = 1.0 - mx.power(mx.abs(2.0 * h - 1.0), beta)
    return mx.mean(reg)


def compute_beta(
    iteration: int,
    total_iters: int,
    warmup_fraction: float,
    beta_start: float,
    beta_end: float,
) -> float:
    """Compute current beta for AdaRound regularization annealing.

    Beta anneals from beta_start -> beta_end over the warmup period.
    After warmup, beta stays at beta_end.
    """
    warmup_iters = int(total_iters * warmup_fraction)
    if iteration >= warmup_iters:
        return beta_end
    t = iteration / max(warmup_iters, 1)
    return beta_start + (beta_end - beta_start) * t


# ---------------------------------------------------------------------------
# Activation Quantizer
# ---------------------------------------------------------------------------

class ActivationQuantizer:
    """Statistics-based activation quantizer with enable/disable support.

    Clipping range alpha is set from calibration data (percentile or MSE search),
    not learned via gradient descent.
    """

    def __init__(self, bits: int = 8, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.alpha = None       # Clipping range (set during calibration)
        self.scale = None       # Derived from alpha
        self.zero_point = None  # Only for asymmetric
        self.enabled = False    # Disabled until calibrated

    def set_alpha(self, alpha: mx.array, alpha_min: mx.array = None):
        """Set clipping range and derive scale.

        For symmetric: alpha defines [-alpha, +alpha]
        For asymmetric: alpha_min, alpha define [alpha_min, alpha]
        """
        self.alpha = alpha
        if self.symmetric:
            self.scale = compute_activation_scale(alpha, self.bits, symmetric=True)
            self.zero_point = None
        else:
            alpha_range = alpha - alpha_min
            self.scale = alpha_range / ((1 << self.bits) - 1)
            self.scale = mx.maximum(self.scale, mx.array(1e-8))
            self.zero_point = mx.round(-alpha_min / self.scale)
        self.enabled = True

    def __call__(self, x: mx.array) -> mx.array:
        """Fake-quantize activation tensor. Pass through if disabled."""
        if not self.enabled:
            return x
        if self.symmetric:
            return fake_quantize_symmetric(x, self.scale, self.bits)
        else:
            return fake_quantize_asymmetric(x, self.scale, self.zero_point, self.bits)
