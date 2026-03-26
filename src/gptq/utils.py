"""Shared utilities for the GPTQ pipeline."""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np


def load_prompt_file(path: Path) -> List[Tuple[int, str]]:
    """Load tab-separated (seed, prompt) pairs from a prompt file.

    Each line is: <seed>\\t<prompt text>
    Returns list of (seed, prompt) tuples.
    """
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        seed_str, prompt = line.split("\t", 1)
        entries.append((int(seed_str), prompt))
    return entries

# ---------------------------------------------------------------------------
# Linear layer enumeration (copied from collect_activation_stats.py)
# ---------------------------------------------------------------------------

_LINEAR_PATHS = [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "mlp.fc1",
    "mlp.fc2",
]


def _get_nested(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if "[" in part:
            attr, idx_s = part.split("[", 1)
            obj = getattr(obj, attr)[int(idx_s.rstrip("]"))]
        else:
            obj = getattr(obj, part)
    return obj


def _set_nested(obj: Any, path: str, val: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        if "[" in part:
            attr, idx_s = part.split("[", 1)
            obj = getattr(obj, attr)[int(idx_s.rstrip("]"))]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if "[" in last:
        attr, idx_s = last.split("[", 1)
        getattr(obj, attr)[int(idx_s.rstrip("]"))] = val
    else:
        setattr(obj, last, val)


def _get_block_linears(block, is_mm: bool = True) -> List[Tuple[str, Any]]:
    """Return (dotted_path, layer) for all quantisable linears in a block."""
    prefixes = (
        ["image_transformer_block", "text_transformer_block"]
        if is_mm else ["transformer_block"]
    )
    results = []
    for prefix in prefixes:
        for local in _LINEAR_PATHS:
            full = f"{prefix}.{local}"
            try:
                layer = _get_nested(block, full)
            except AttributeError:
                continue
            if not hasattr(layer, "weight"):
                continue
            results.append((full, layer))
    return results


# ---------------------------------------------------------------------------
# Poly schedule key mapping
# ---------------------------------------------------------------------------

def full_path_to_poly_key(block_idx: int, full_path: str) -> str:
    """Convert block index + full_path to poly schedule key.

    Example: block_idx=0, full_path="image_transformer_block.attn.q_proj"
             -> "mm0_img_attn_q_proj"
    """
    short = (
        full_path
        .replace("image_transformer_block", "img")
        .replace("text_transformer_block", "txt")
    )
    key = f"mm{block_idx}.{short}"
    return key.replace(".", "_")


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def compute_per_channel_scale(W: np.ndarray, bits: int) -> np.ndarray:
    """Per-channel symmetric quantization scales.

    W: (d_out, d_in). Returns (d_out,) scales.
    """
    qmax = 2 ** (bits - 1) - 1
    row_absmax = np.abs(W).max(axis=1)
    scales = row_absmax / qmax
    scales = np.maximum(scales, 1e-10)
    return scales


def compute_group_scales(W: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    """Per-group symmetric quantization scales.

    W: (d_out, d_in). Returns (d_out, n_groups) scales.
    """
    d_out, d_in = W.shape
    qmax = 2 ** (bits - 1) - 1
    n_groups = (d_in + group_size - 1) // group_size

    if d_in % group_size != 0:
        pad_width = group_size * n_groups - d_in
        W_padded = np.pad(W, ((0, 0), (0, pad_width)), constant_values=0)
    else:
        W_padded = W

    W_grouped = W_padded.reshape(d_out, n_groups, group_size)
    group_absmax = np.abs(W_grouped).max(axis=2)  # (d_out, n_groups)
    scales = group_absmax / qmax
    scales = np.maximum(scales, 1e-10)
    return scales


def compute_scales(W: np.ndarray, bits: int, group_size: int = 0) -> np.ndarray:
    """Compute quantization scales — per-channel or per-group.

    group_size <= 0 or >= d_in: per-channel, returns (d_out,).
    Otherwise: per-group, returns (d_out, n_groups).
    """
    if group_size <= 0 or group_size >= W.shape[1]:
        return compute_per_channel_scale(W, bits)
    return compute_group_scales(W, bits, group_size)


def dequantize(W_q_int: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize W_q_int using scales (per-channel or per-group).

    scales shape (d_out,): per-channel broadcast.
    scales shape (d_out, n_groups): per-group, expanded to match d_in.
    """
    if scales.ndim == 1:
        return W_q_int.astype(np.float32) * scales[:, None]
    # Per-group: expand (d_out, n_groups) -> (d_out, d_in)
    d_in = W_q_int.shape[1]
    n_groups = scales.shape[1]
    group_size = (d_in + n_groups - 1) // n_groups
    scales_expanded = np.repeat(scales, group_size, axis=1)[:, :d_in]
    return W_q_int.astype(np.float32) * scales_expanded


def fake_quant_symmetric(x: np.ndarray, scale, bits: int) -> np.ndarray:
    """Fake-quantize x with symmetric per-channel scales."""
    qmax = 2 ** (bits - 1) - 1
    return np.clip(np.round(x / scale), -qmax, qmax) * scale


def get_poly_alpha(poly_entry: dict, sigma: float) -> float:
    """Evaluate polynomial clipping alpha, clamped to >= 0.01."""
    alpha = float(np.polyval(poly_entry["coeffs"], sigma))
    return max(alpha, 0.01)


# ---------------------------------------------------------------------------
# Modulation cache reset
# ---------------------------------------------------------------------------

def _reset_modulation_cache(pipeline):
    """Reload adaLN weights that were offloaded by cache_modulation_params.

    Raises on failure — silent failure would corrupt subsequent prompts
    by reusing stale modulation parameters.
    """
    pipeline.mmdit.load_weights(
        pipeline.load_mmdit(only_modulation_dict=True), strict=False
    )
