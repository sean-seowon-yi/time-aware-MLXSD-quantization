"""Polynomial evaluation primitive for MLX arrays.

Evaluates ``p(σ) = c₀ + c₁σ + c₂σ² + …`` via Horner's method.
Used at inference time inside :class:`W4A8PolyLinear` to compute the
per-layer clipping bound from the polynomial schedule.

Supports two coefficient shapes:

- **1-D** ``[d+1]`` — per-tensor: returns a scalar α(σ).
- **2-D** ``[d_in, d+1]`` — per-channel: returns a vector ``[d_in]``
  with one clipping bound per input channel.
"""

from __future__ import annotations

import mlx.core as mx


def poly_eval(coeffs: mx.array, sigma: mx.array) -> mx.array:
    """Evaluate a polynomial at *sigma* using Horner's method.

    Parameters
    ----------
    coeffs : mx.array
        Coefficients in **ascending-power** order.

        - Shape ``[d+1]``: per-tensor — ``[c₀, c₁, …, c_d]``.
        - Shape ``[d_in, d+1]``: per-channel — one row per channel.

    sigma : mx.array, scalar
        The point at which to evaluate (current noise level).

    Returns
    -------
    mx.array
        Scalar when *coeffs* is 1-D, or shape ``[d_in]`` when 2-D.
    """
    n = coeffs.shape[-1]
    if n == 0:
        return mx.array(0.0)
    if n == 1:
        return coeffs[..., 0]
    # Horner: ((c_d · σ + c_{d-1}) · σ + … ) · σ + c_0
    result = coeffs[..., n - 1]
    for k in range(n - 2, -1, -1):
        result = result * sigma + coeffs[..., k]
    return result
