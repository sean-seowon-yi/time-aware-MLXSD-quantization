"""Shared utilities for alpha search and poly-schedule workflows.

Migrated from ``src.phase4.utils`` after removing the GPTQ pipeline.
"""

from __future__ import annotations

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
# Linear layer enumeration
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


def _is_linear_like(layer) -> bool:
    """True if layer is nn.Linear, nn.QuantizedLinear, or W4A8StaticLinear."""
    if hasattr(layer, "weight"):
        return True
    if hasattr(layer, "qlinear") and hasattr(layer.qlinear, "weight"):
        return True
    return False


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
            if not _is_linear_like(layer):
                continue
            results.append((full, layer))
    return results


# ---------------------------------------------------------------------------
# Poly schedule key mapping
# ---------------------------------------------------------------------------

def full_path_to_poly_key(block_idx: int, full_path: str) -> str:
    """Convert block index + full_path to poly schedule key.

    Must produce the same key format as Phase 1/2/3:
        blocks.{block_idx}.{side}.{rest}

    Example: block_idx=0, full_path="image_transformer_block.attn.q_proj"
             -> "blocks.0.image.attn.q_proj"
    """
    side = (
        "image" if full_path.startswith("image_transformer_block") else "text"
    )
    prefix = (
        "image_transformer_block."
        if side == "image"
        else "text_transformer_block."
    )
    rest = full_path[len(prefix):]
    return f"blocks.{block_idx}.{side}.{rest}"


# ---------------------------------------------------------------------------
# Poly alpha evaluation (NumPy)
# ---------------------------------------------------------------------------

def get_poly_alpha_raw(poly_entry: dict, sigma: float) -> float:
    """Evaluate poly α(σ) with **no** floor.

    Use when multiplying by a per-layer ``alpha_multiplier`` *m* so that
    ``max(raw * m, 0.01)`` matches :class:`~src.phase3.quantize_poly.W4A8PolyLinear`
    (which applies the floor **after** ``poly(σ) * m``).
    """
    coeffs = poly_entry["coeffs"]
    if isinstance(coeffs[0], (list, np.ndarray)):
        arr = np.asarray(coeffs, dtype=np.float64)
        alphas = np.polynomial.polynomial.polyval(sigma, arr.T)
        return float(np.max(alphas))
    return float(np.polynomial.polynomial.polyval(sigma, coeffs))


def get_poly_alpha(poly_entry: dict, sigma: float) -> float:
    """Evaluate polynomial clipping alpha from a poly_schedule entry.

    Result is clamped to >= 0.01 — must stay in sync with
    ``W4A8PolyLinear`` in ``phase3.quantize_poly``.

    For objectives that scale α by a multiplier *m*, prefer
    ``max(get_poly_alpha_raw(...) * m, 0.01)`` rather than
    ``m * get_poly_alpha(...)``.
    """
    return max(get_poly_alpha_raw(poly_entry, sigma), 0.01)


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
