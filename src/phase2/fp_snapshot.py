"""Float weight snapshot (post-CSB, pre-RTN) for alpha-search FP reference.

Saved next to the Phase 2 checkpoint so :mod:`src.phase4_1.alpha_search` can
compare poly int8 proxies to ``x @ W_fp.T + b`` using the same ``nn.Linear``
weights that were about to be RTN-quantized (not dequantized W4).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    FP_PRE_RTN_MANIFEST_FILENAME,
    FP_PRE_RTN_WEIGHTS_FILENAME,
    PHASE2_CONFIG,
    QUANTIZE_CONFIG_FILENAME,
)

logger = logging.getLogger(__name__)

Manifest = Dict[str, Any]


def save_fp_pre_rtn_snapshot(
    registry: List[dict],
    config: dict | None,
    output_dir: Path,
) -> bool:
    """Save float ``Linear`` weights immediately before RTN (after CSB).

    Returns True if a non-empty snapshot was written.
    """
    cfg = {**PHASE2_CONFIG, **(config or {})}
    exclude = set(cfg["exclude_layers"])
    bits = cfg["bits"]
    final_bits = cfg["final_layer_bits"]

    layer_names: List[str] = []
    arrays: Dict[str, np.ndarray] = {}

    for entry in registry:
        name = entry["name"]
        if name in exclude:
            continue
        linear = entry["module"]
        layer_bits = final_bits if entry["family"] == "final_linear" else bits
        if layer_bits >= 16:
            continue

        W = np.asarray(np.array(linear.weight), dtype=np.float32)
        idx = len(layer_names)
        layer_names.append(name)
        arrays[f"{idx}_W"] = W
        if getattr(linear, "bias", None) is not None:
            arrays[f"{idx}_b"] = np.asarray(np.array(linear.bias), dtype=np.float32)
        else:
            arrays[f"{idx}_b"] = np.zeros(W.shape[0], dtype=np.float32)

    if not layer_names:
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Manifest = {
        "version": 1,
        "layer_names": layer_names,
        "weights_file": FP_PRE_RTN_WEIGHTS_FILENAME,
    }
    man_path = output_dir / FP_PRE_RTN_MANIFEST_FILENAME
    man_path.write_text(json.dumps(manifest, indent=2))

    npz_path = output_dir / FP_PRE_RTN_WEIGHTS_FILENAME
    np.savez_compressed(npz_path, **arrays)

    logger.info(
        "Saved FP pre-RTN snapshot for %d layers (%s, %s)",
        len(layer_names),
        man_path.name,
        npz_path.name,
    )
    return True


def load_fp_pre_rtn_weights(quantized_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load ``{registry_name -> (W_fp, bias_fp)}``. W shape (d_out, d_in).

    Returns empty dict if manifest or npz is missing.
    """
    qdir = Path(quantized_dir)
    man_path = qdir / FP_PRE_RTN_MANIFEST_FILENAME
    meta_path = qdir / QUANTIZE_CONFIG_FILENAME
    if not man_path.is_file():
        return {}

    try:
        manifest: Manifest = json.loads(man_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    layer_names = manifest.get("layer_names")
    if not isinstance(layer_names, list) or not layer_names:
        return {}

    wfile = manifest.get("weights_file", FP_PRE_RTN_WEIGHTS_FILENAME)
    npz_path = qdir / str(wfile)
    if not npz_path.is_file():
        logger.warning("FP snapshot manifest without %s — ignoring snapshot", npz_path.name)
        return {}

    try:
        z = np.load(npz_path)
    except OSError:
        return {}

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(layer_names):
        kw = f"{i}_W"
        kb = f"{i}_b"
        if kw not in z.files or kb not in z.files:
            logger.warning("FP snapshot missing arrays for index %d (%s)", i, name)
            continue
        W = np.asarray(z[kw], dtype=np.float32)
        b = np.asarray(z[kb], dtype=np.float32)
        out[str(name)] = (W, b)

    z.close()

    if out and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            ql = meta.get("quantized_layers") or {}
            missing = [n for n in ql if n not in out]
            if missing:
                logger.warning(
                    "FP snapshot missing %d / %d quantized layer names (e.g. %r)",
                    len(missing),
                    len(ql),
                    missing[:3],
                )
        except (json.JSONDecodeError, OSError):
            pass

    return out


def attach_fp_pre_rtn_meta(meta: dict, output_dir: Path) -> None:
    """If snapshot files exist under *output_dir*, record paths in *meta*."""
    d = Path(output_dir)
    if (d / FP_PRE_RTN_MANIFEST_FILENAME).is_file() and (
        d / FP_PRE_RTN_WEIGHTS_FILENAME
    ).is_file():
        meta["fp_pre_rtn"] = {
            "manifest": FP_PRE_RTN_MANIFEST_FILENAME,
            "weights": FP_PRE_RTN_WEIGHTS_FILENAME,
            "note": "Post-CSB float Linear weights captured immediately before RTN W4",
        }
