"""Core GPTQ algorithm — pure NumPy, no MLX dependency.

Implements column-wise quantization with Hessian-weighted error compensation.
"""

import warnings

import numpy as np

from .utils import compute_per_channel_scale


def gptq_quantize(
    W: np.ndarray,
    H: np.ndarray,
    bits: int,
    damp_percent: float = 0.01,
    block_size: int = 128,
):
    """GPTQ quantization of a single weight matrix.

    Args:
        W: (d_out, d_in) float32 — original weight matrix.
        H: (d_in, d_in) float32 — Hessian (2 * X^T X).
        bits: quantization bit-width (e.g. 4 or 8).
        damp_percent: diagonal damping factor.
        block_size: column block size for error compensation.

    Returns:
        W_q_int: (d_out, d_in) int8 — quantized integer weights.
        scales: (d_out,) float32 — per-channel scales.
        weight_mse: float — total squared error ||W - dequant(W_q)||^2.
    """
    d_out, d_in = W.shape
    qmax = 2 ** (bits - 1) - 1
    W_orig = W.copy()
    W = W.copy()

    # Per-channel scales
    scales = compute_per_channel_scale(W_orig, bits)

    # Sanitize: replace NaN/Inf with 0 (can occur from float16 overflow
    # in upstream accumulation, though this should be fixed now).
    nan_mask = ~np.isfinite(H)
    if nan_mask.any():
        warnings.warn(
            f"Hessian has {nan_mask.sum()} non-finite values — zeroing them"
        )
        H = np.where(nan_mask, 0.0, H)

    # Damping
    diag_max = float(np.diag(H).max())
    if diag_max <= 0:
        diag_max = 1.0  # degenerate Hessian, use unit damping
    damp = damp_percent * diag_max
    H = H + damp * np.eye(d_in, dtype=H.dtype)

    # Inverse + Cholesky of inverse (upper triangular)
    try:
        H_inv = np.linalg.inv(H)
        H_inv_chol = np.linalg.cholesky(H_inv).T
    except np.linalg.LinAlgError:
        warnings.warn("Cholesky failed, increasing damping 10x")
        H = H + 9 * damp * np.eye(d_in, dtype=H.dtype)
        try:
            H_inv = np.linalg.inv(H)
            H_inv_chol = np.linalg.cholesky(H_inv).T
        except np.linalg.LinAlgError:
            warnings.warn("Cholesky still failed, using pseudoinverse + diagonal")
            try:
                H_inv = np.linalg.pinv(H)
            except np.linalg.LinAlgError:
                warnings.warn("pinv also failed, falling back to diagonal")
                H_inv = np.diag(1.0 / np.diag(H))
            # Force positive definite for Cholesky
            eigvals = np.linalg.eigvalsh(H_inv)
            min_eig = float(eigvals.min())
            if min_eig <= 0:
                H_inv = H_inv + (abs(min_eig) + 1e-6) * np.eye(d_in, dtype=H_inv.dtype)
            H_inv_chol = np.linalg.cholesky(H_inv).T

    W_q_int = np.zeros_like(W, dtype=np.int8)

    # Column-wise quantization with block error compensation
    for i in range(0, d_in, block_size):
        j_end = min(i + block_size, d_in)
        bsz = j_end - i

        E = np.zeros((d_out, bsz), dtype=np.float32)

        for j in range(i, j_end):
            w_col = W[:, j]
            # Per-channel quantize
            w_q_col = np.clip(np.round(w_col / scales), -qmax, qmax).astype(np.int8)
            W_q_int[:, j] = w_q_col
            w_dequant = w_q_col.astype(np.float32) * scales
            err = (w_col - w_dequant) / H_inv_chol[j, j]
            E[:, j - i] = err
            # Intra-block compensation
            if j + 1 < j_end:
                W[:, j + 1:j_end] -= np.outer(err, H_inv_chol[j, j + 1:j_end])

        # Inter-block compensation
        if j_end < d_in:
            W[:, j_end:] -= E @ H_inv_chol[i:j_end, j_end:]

    # Compute total MSE
    W_dequant = W_q_int.astype(np.float32) * scales[:, None]
    weight_mse = float(np.sum((W_orig - W_dequant) ** 2))

    return W_q_int, scales, weight_mse
