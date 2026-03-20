"""Load Phase 1 diagnostic data and compute SSC-weighted balancing vectors.

This module performs the calibration step of Phase 2: for each target layer,
it loads the activation trajectory and weight salience from Phase 1 diagnostics,
computes SSC weights (time-aware calibration), and produces the per-channel
balancing vector used by CSB.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..phase1.analyze import compute_spearman_trajectory, compute_ssc_weights
from .config import PHASE2_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 data loading
# ---------------------------------------------------------------------------

def load_phase1_data(
    layer_name: str,
    diagnostics_dir: Path,
) -> dict:
    """Load activation trajectory and weight salience for one layer.

    Returns
    -------
    dict with:
        "act_trajectory" : ndarray [T, d_in]
        "wt_salience"    : ndarray [d_in]
    """
    act_path = diagnostics_dir / "activation_stats" / f"{layer_name}.npz"
    act_data = np.load(act_path)
    act_trajectory = act_data["act_channel_max"]

    wt_path = diagnostics_dir / "weight_stats.npz"
    wt_data = np.load(wt_path)
    wt_salience = wt_data[f"{layer_name}/w_channel_max"]

    return {"act_trajectory": act_trajectory, "wt_salience": wt_salience}


# ---------------------------------------------------------------------------
# Balancing vector computation
# ---------------------------------------------------------------------------

def compute_balancing_vector(
    act_trajectory: np.ndarray,
    wt_salience: np.ndarray,
    alpha: float = 0.5,
    b_min: float = 1e-5,
    b_max: float = 1e5,
    w_eps: float = 1e-12,
) -> np.ndarray:
    """Compute the SSC-weighted balancing vector for a single layer.

    Steps:
      1. Spearman rho trajectory → SSC weights eta_t.
      2. Weighted activation salience s_bar(X_j) = sum_t eta_t * s(X_j^t).
      3. b_j = (s_bar(X_j) / s(W_j))^alpha, clamped.

    Returns: b_vector of shape [d_in].
    """
    rho_traj = compute_spearman_trajectory(act_trajectory, wt_salience)
    ssc_weights = compute_ssc_weights(rho_traj)
    weighted_act = ssc_weights @ act_trajectory  # [d_in]

    b = (weighted_act / (wt_salience + w_eps)) ** alpha
    b = np.clip(b, b_min, b_max)

    dead_mask = wt_salience < w_eps
    b[dead_mask] = 1.0

    return b


def compute_qkv_balancing(
    block_idx: int,
    side: str,
    diagnostics_dir: Path,
    method: str = "max",
    alpha: float = 0.5,
    b_min: float = 1e-5,
    b_max: float = 1e5,
    w_eps: float = 1e-12,
) -> np.ndarray:
    """Compute the shared balancing vector for Q/K/V projections.

    Q, K, V share the same input (modulated pre-attention), so a single
    balancing vector is computed.  Weight salience is merged across projections
    using the specified *method* ("max" or "geomean").

    Returns: b_qkv of shape [hidden_size].
    """
    q_name = f"blocks.{block_idx}.{side}.attn.q_proj"
    q_data = load_phase1_data(q_name, diagnostics_dir)
    act_trajectory = q_data["act_trajectory"]

    wt = {}
    for proj in ("q_proj", "k_proj", "v_proj"):
        name = f"blocks.{block_idx}.{side}.attn.{proj}"
        wt[proj] = load_phase1_data(name, diagnostics_dir)["wt_salience"]

    # SSC weights from q_proj's rho trajectory (shared input → representative)
    rho_traj = compute_spearman_trajectory(act_trajectory, wt["q_proj"])
    ssc_weights = compute_ssc_weights(rho_traj)
    weighted_act = ssc_weights @ act_trajectory

    if method == "max":
        merged_wt = np.maximum(np.maximum(wt["q_proj"], wt["k_proj"]), wt["v_proj"])
    elif method == "geomean":
        merged_wt = (wt["q_proj"] * wt["k_proj"] * wt["v_proj"]) ** (1.0 / 3.0)
    else:
        raise ValueError(f"Unknown QKV merge method: {method!r}")

    b = (weighted_act / (merged_wt + w_eps)) ** alpha
    b = np.clip(b, b_min, b_max)

    for proj in ("q_proj", "k_proj", "v_proj"):
        b[wt[proj] < w_eps] = 1.0

    return b


# ---------------------------------------------------------------------------
# Calibrate all target layers
# ---------------------------------------------------------------------------

def calibrate_all_layers(
    registry: list[dict],
    diagnostics_dir: Path,
    config: dict | None = None,
) -> dict:
    """Compute balancing vectors for every target layer.

    Parameters
    ----------
    registry : list[dict]
        Layer registry (from phase1.registry or build_lightweight_registry).
    diagnostics_dir : Path
        Directory containing Phase 1 diagnostic outputs.
    config : dict, optional
        Override PHASE2_CONFIG entries.

    Returns
    -------
    dict with:
        "balancing_vectors" : dict[layer_name → ndarray]
        "b_inv_layers"      : list[str]  (layers requiring online b_inv)
    """
    cfg = {**PHASE2_CONFIG, **(config or {})}
    alpha = cfg["alpha"]
    b_min, b_max = cfg["b_min"], cfg["b_max"]
    w_eps = cfg["w_eps"]
    qkv_method = cfg["qkv_method"]
    exclude = set(cfg["exclude_layers"])

    balancing_vectors: dict[str, np.ndarray] = {}
    b_inv_layers: list[str] = []
    processed_qkv: set[tuple[int, str]] = set()

    for entry in registry:
        name = entry["name"]
        if name in exclude:
            continue

        family = entry["family"]
        block = entry["block"]
        side = entry["side"]

        if family in ("q_proj", "k_proj", "v_proj"):
            qkv_key = (block, side)
            if qkv_key not in processed_qkv:
                processed_qkv.add(qkv_key)
                b_qkv = compute_qkv_balancing(
                    block, side, diagnostics_dir,
                    method=qkv_method, alpha=alpha,
                    b_min=b_min, b_max=b_max, w_eps=w_eps,
                )
                for proj in ("q_proj", "k_proj", "v_proj"):
                    proj_name = f"blocks.{block}.{side}.attn.{proj}"
                    balancing_vectors[proj_name] = b_qkv
                logger.info(
                    "Block %d %s QKV (%s): b in [%.4f, %.4f]",
                    block, side, qkv_method, b_qkv.min(), b_qkv.max(),
                )

        elif family in ("o_proj", "fc1", "fc2", "final_linear"):
            data = load_phase1_data(name, diagnostics_dir)
            b = compute_balancing_vector(
                data["act_trajectory"], data["wt_salience"],
                alpha=alpha, b_min=b_min, b_max=b_max, w_eps=w_eps,
            )
            balancing_vectors[name] = b
            if family in ("o_proj", "fc2"):
                b_inv_layers.append(name)
            logger.info(
                "Layer %-45s  b in [%.4f, %.4f]", name, b.min(), b.max(),
            )

    logger.info(
        "Calibrated %d layers (%d need online b_inv)",
        len(balancing_vectors), len(b_inv_layers),
    )
    return {"balancing_vectors": balancing_vectors, "b_inv_layers": b_inv_layers}


# ---------------------------------------------------------------------------
# Lightweight registry (no model load required)
# ---------------------------------------------------------------------------

def build_lightweight_registry(diagnostics_dir: Path) -> list[dict]:
    """Build a minimal registry from the Phase 1 config.json.

    This avoids loading the full model when only calibration is needed.
    """
    cfg_path = diagnostics_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())

    registry: list[dict] = []
    for name in cfg["layer_names"]:
        parts = name.split(".")
        if name == "context_embedder":
            family, side, block = "context_embedder", "shared", -1
        elif name == "final_layer.linear":
            family, side, block = "final_linear", "image", -1
        else:
            block = int(parts[1])
            side = parts[2]
            family = parts[4]
        registry.append({
            "name": name,
            "block": block,
            "family": family,
            "side": side,
        })
    return registry


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_calibration(calibration: dict, output_dir: Path) -> None:
    """Save calibration data (balancing vectors + metadata)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    bv = calibration["balancing_vectors"]
    np.savez_compressed(output_dir / "calibration.npz", **bv)

    meta = {
        "b_inv_layers": calibration["b_inv_layers"],
        "layer_names": list(bv.keys()),
    }
    (output_dir / "calibration_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved calibration to %s (%d layers)", output_dir, len(bv))


def load_calibration(output_dir: Path) -> dict:
    """Load previously saved calibration data."""
    meta = json.loads((output_dir / "calibration_meta.json").read_text())
    data = np.load(output_dir / "calibration.npz")

    balancing_vectors = {name: data[name] for name in meta["layer_names"]}
    return {
        "balancing_vectors": balancing_vectors,
        "b_inv_layers": meta["b_inv_layers"],
    }
