"""Repack GPTQ symmetric-quantized weights into MLX QuantizedLinear format.

GPTQ produces per-(row, group) symmetric int weights:
    dequant = q_int * gptq_scale,  q_int ∈ [-qmax, qmax]

MLX QuantizedLinear expects per-group affine unsigned:
    dequant = q_uint * mlx_scale + mlx_bias,  q_uint ∈ [0, 2^bits - 1]

Mapping:
    q_uint    = q_int + qmax           (shift to unsigned)
    mlx_scale = gptq_scale
    mlx_bias  = -qmax * gptq_scale
"""

import logging

import numpy as np
import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def repack_to_mlx(
    W_q_int: np.ndarray,
    gptq_scales: np.ndarray,
    bits: int,
    group_size: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Convert GPTQ int output to MLX packed format.

    Args:
        W_q_int: (d_out, d_in) int8 — GPTQ quantized weights in [-qmax, qmax].
        gptq_scales: (d_out, n_groups) or (d_out,) float32 — GPTQ symmetric scales.
        bits: quantization bit width.
        group_size: GPTQ group size.

    Returns:
        (packed_weight, mlx_scales, mlx_biases) ready for QuantizedLinear.
    """
    d_out, d_in = W_q_int.shape
    qmax = 2 ** (bits - 1) - 1
    n_per_u32 = 32 // bits

    if gptq_scales.ndim == 1:
        gptq_scales = gptq_scales[:, None]

    q_uint = (W_q_int.astype(np.int16) + qmax).astype(np.uint8)

    packed = np.zeros((d_out, d_in // n_per_u32), dtype=np.uint32)
    for k in range(n_per_u32):
        packed |= q_uint[:, k::n_per_u32].astype(np.uint32) << (bits * k)

    mlx_scales = mx.array(gptq_scales, dtype=mx.float32)
    mlx_biases = mx.array(-qmax * gptq_scales, dtype=mx.float32)
    packed_mx = mx.array(packed)

    return packed_mx, mlx_scales, mlx_biases


def build_quantized_linear(
    W_q_int: np.ndarray,
    gptq_scales: np.ndarray,
    bias: np.ndarray | None,
    bits: int,
    group_size: int,
) -> nn.QuantizedLinear:
    """Create an nn.QuantizedLinear from GPTQ output.

    Args:
        W_q_int: (d_out, d_in) int8.
        gptq_scales: (d_out, n_groups) or (d_out,) float32.
        bias: (d_out,) float32 or None — original layer bias.
        bits: bit width.
        group_size: group size.

    Returns:
        Configured nn.QuantizedLinear ready for inference.
    """
    d_out, d_in = W_q_int.shape
    packed, mlx_scales, mlx_biases = repack_to_mlx(
        W_q_int, gptq_scales, bits, group_size,
    )

    ql = nn.QuantizedLinear(
        input_dims=d_in,
        output_dims=d_out,
        bias=bias is not None,
        group_size=group_size,
        bits=bits,
    )
    ql.weight = packed
    ql.scales = mlx_scales
    ql.biases = mlx_biases
    if bias is not None:
        ql.bias = mx.array(bias, dtype=mx.float32)

    return ql
