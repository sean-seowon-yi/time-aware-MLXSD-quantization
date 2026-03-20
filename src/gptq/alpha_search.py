"""Bidirectional alpha_scale search for activation clipping.

Searches for the optimal scalar multiplier on the polynomial clipping
schedule that minimizes layer-wise MSE between quantized and FP16 outputs.
"""

from typing import List, Optional, Tuple

import numpy as np

from .utils import get_poly_alpha, fake_quant_symmetric


def _evaluate_mse(
    W_q_dequant: np.ndarray,
    bias: Optional[np.ndarray],
    cached_inputs: List[Tuple[np.ndarray, float]],
    cached_outputs: List[np.ndarray],
    poly_entry: dict,
    alpha_scale: float,
) -> float:
    """Compute MSE of quantized-weight + fake-quantized-activation output vs FP16 reference."""
    total_se = 0.0
    total_elements = 0

    for (x_np, sigma), y_ref in zip(cached_inputs, cached_outputs):
        if sigma is None or poly_entry is None:
            x_fq = x_np
        else:
            alpha = alpha_scale * get_poly_alpha(poly_entry, sigma)
            scale = alpha / 127.0
            x_fq = fake_quant_symmetric(x_np, scale, 8)

        # y_q = x_fq @ W^T + bias
        # x_fq: (rows, d_in), W_q_dequant: (d_out, d_in)
        y_q = x_fq @ W_q_dequant.T
        if bias is not None:
            y_q = y_q + bias

        total_se += float(np.sum((y_q - y_ref) ** 2))
        total_elements += y_ref.size

    if total_elements == 0:
        return float("inf")
    return total_se / total_elements


def search_alpha_scale(
    W_q_dequant: np.ndarray,
    bias: Optional[np.ndarray],
    cached_inputs: List[Tuple[np.ndarray, float]],
    cached_outputs: List[np.ndarray],
    poly_entry: Optional[dict],
) -> Tuple[float, float]:
    """Bidirectional search from alpha_scale=1.0.

    Range: [0.2, 0.3, ..., 5.0], step 0.1.
    Early stop: 2 consecutive non-improvements per direction.

    Returns (best_alpha_scale, best_activation_mse).
    """
    if poly_entry is None or len(cached_inputs) == 0:
        return 1.0, float("inf")

    baseline_mse = _evaluate_mse(
        W_q_dequant, bias, cached_inputs, cached_outputs, poly_entry, 1.0
    )

    best_alpha = 1.0
    best_mse = baseline_mse

    # Search downward: 0.9, 0.8, ..., 0.2
    down_best_mse = baseline_mse
    no_improve_count = 0
    for step in range(9, 1, -1):  # 0.9 .. 0.2
        candidate = step * 0.1
        mse = _evaluate_mse(
            W_q_dequant, bias, cached_inputs, cached_outputs, poly_entry, candidate
        )
        if mse < down_best_mse:
            down_best_mse = mse
            if mse < best_mse:
                best_mse = mse
                best_alpha = candidate
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 2:
                break

    # Search upward: 1.1, 1.2, ..., 5.0
    # Compare against baseline (not downward best)
    up_best_mse = baseline_mse
    no_improve_count = 0
    for step in range(11, 51):  # 1.1 .. 5.0
        candidate = step * 0.1
        mse = _evaluate_mse(
            W_q_dequant, bias, cached_inputs, cached_outputs, poly_entry, candidate
        )
        if mse < up_best_mse:
            up_best_mse = mse
            if mse < best_mse:
                best_mse = mse
                best_alpha = candidate
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 2:
                break

    return best_alpha, best_mse


# ---------------------------------------------------------------------------
# Static (timestep-agnostic) alpha search
# ---------------------------------------------------------------------------

def _evaluate_static_mse(
    W_q_dequant: np.ndarray,
    bias: Optional[np.ndarray],
    cached_inputs: List[Tuple[np.ndarray, float]],
    cached_outputs: List[np.ndarray],
    static_alpha: float,
    alpha_scale: float,
) -> float:
    """MSE with a single fixed clipping alpha (no sigma dependence)."""
    total_se = 0.0
    total_elements = 0

    alpha = alpha_scale * static_alpha
    scale = alpha / 127.0

    for (x_np, _sigma), y_ref in zip(cached_inputs, cached_outputs):
        x_fq = fake_quant_symmetric(x_np, scale, 8)
        y_q = x_fq @ W_q_dequant.T
        if bias is not None:
            y_q = y_q + bias
        total_se += float(np.sum((y_q - y_ref) ** 2))
        total_elements += y_ref.size

    if total_elements == 0:
        return float("inf")
    return total_se / total_elements


def search_alpha_scale_static(
    W_q_dequant: np.ndarray,
    bias: Optional[np.ndarray],
    cached_inputs: List[Tuple[np.ndarray, float]],
    cached_outputs: List[np.ndarray],
    static_alpha: float,
) -> Tuple[float, float]:
    """Bidirectional alpha_scale search for static (non-poly) activation clipping.

    Same algorithm as search_alpha_scale but uses a fixed alpha per layer
    instead of sigma-dependent polynomial evaluation.

    Returns (best_alpha_scale, best_activation_mse).
    """
    if len(cached_inputs) == 0:
        return 1.0, float("inf")

    baseline_mse = _evaluate_static_mse(
        W_q_dequant, bias, cached_inputs, cached_outputs, static_alpha, 1.0
    )

    best_alpha = 1.0
    best_mse = baseline_mse

    # Search downward
    down_best_mse = baseline_mse
    no_improve_count = 0
    for step in range(9, 1, -1):
        candidate = step * 0.1
        mse = _evaluate_static_mse(
            W_q_dequant, bias, cached_inputs, cached_outputs, static_alpha, candidate
        )
        if mse < down_best_mse:
            down_best_mse = mse
            if mse < best_mse:
                best_mse = mse
                best_alpha = candidate
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 2:
                break

    # Search upward
    up_best_mse = baseline_mse
    no_improve_count = 0
    for step in range(11, 51):
        candidate = step * 0.1
        mse = _evaluate_static_mse(
            W_q_dequant, bias, cached_inputs, cached_outputs, static_alpha, candidate
        )
        if mse < up_best_mse:
            up_best_mse = mse
            if mse < best_mse:
                best_mse = mse
                best_alpha = candidate
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 2:
                break

    return best_alpha, best_mse
