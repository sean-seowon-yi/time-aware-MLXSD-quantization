"""Analysis utilities: aggregation, derived metrics, and summary-table builder.

All functions operate on numpy arrays (post-collection). No MLX dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

from .config import ACTIVATION_STATS_DIR, DIAG_CONFIG, OUTPUT_DIR

logger = logging.getLogger(__name__)

K = DIAG_CONFIG["top_k"]


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1-D array of non-negative values."""
    v = np.sort(values.ravel())
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * v) / (n * np.sum(v))) - (n + 1) / n)


def salience_concentration(s: np.ndarray, k: int = K) -> dict:
    """Compute concentration metrics for a 1-D salience vector."""
    median_val = np.median(s)
    top_k_vals = np.sort(s)[-k:]
    return {
        "top1_over_median": float(s.max() / (median_val + 1e-12)),
        "top1_over_topk_mean": float(s.max() / (top_k_vals.mean() + 1e-12)),
        "gini": gini_coefficient(s),
        "topk_mass_fraction": float(top_k_vals.sum() / (s.sum() + 1e-12)),
    }


def quantize_uniform_8bit(tensor: np.ndarray) -> np.ndarray:
    """Simulate 8-bit uniform min-max quantization (round-to-nearest)."""
    t_min, t_max = tensor.min(), tensor.max()
    if t_max - t_min < 1e-12:
        return tensor.copy()
    scale = (t_max - t_min) / 255.0
    zero_point = np.round(-t_min / scale).clip(0, 255)
    q = np.clip(np.round(tensor / scale + zero_point), 0, 255)
    return (q - zero_point) * scale


def per_channel_quant_mse_activation(X_flat: np.ndarray) -> np.ndarray:
    """Per-channel MSE from tensor-wise 8-bit quantization of activations.

    X_flat: [n_tokens, d_in]. Returns [d_in].
    """
    X_q = quantize_uniform_8bit(X_flat)
    return np.mean((X_flat - X_q) ** 2, axis=0)


def per_channel_quant_mse_weight(W: np.ndarray) -> np.ndarray:
    """Per-channel MSE from channel-wise 8-bit quantization of weights.

    W: [d_out, d_in]. Quantize each column independently. Returns [d_in].
    """
    d_in = W.shape[1]
    mse = np.zeros(d_in)
    for j in range(d_in):
        col = W[:, j]
        col_q = quantize_uniform_8bit(col)
        mse[j] = np.mean((col - col_q) ** 2)
    return mse


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_trajectory(layer_name: str,
                    output_dir: Optional[Path] = None) -> dict:
    """Load saved activation trajectory for one layer."""
    act_dir = output_dir or ACTIVATION_STATS_DIR
    path = act_dir / f"{layer_name}.npz"
    data = np.load(path)
    return {
        "sigma_values": data["sigma_values"],
        "act_channel_max": data["act_channel_max"],
        "act_channel_mean": data["act_channel_mean"],
    }


def load_weight_stats(output_dir: Optional[Path] = None) -> dict:
    path = (output_dir or OUTPUT_DIR) / "weight_stats.npz"
    data = np.load(path)
    weight_stats: dict = {}
    for key in data.files:
        layer_name, stat_key = key.rsplit("/", 1)
        if layer_name not in weight_stats:
            weight_stats[layer_name] = {}
        weight_stats[layer_name][stat_key] = data[key]
    return weight_stats


# ---------------------------------------------------------------------------
# Correlation & complementarity
# ---------------------------------------------------------------------------

def compute_spearman_trajectory(
    act_trajectory: np.ndarray,
    wt_salience: np.ndarray,
) -> np.ndarray:
    """Compute Spearman ρ for each sigma step.

    act_trajectory: [num_steps, d_in]
    wt_salience:    [d_in]
    Returns:        [num_steps]
    """
    num_steps = act_trajectory.shape[0]
    rhos = np.zeros(num_steps)
    for t in range(num_steps):
        rho, _ = spearmanr(act_trajectory[t], wt_salience)
        rhos[t] = rho if not np.isnan(rho) else 0.0
    return rhos


def compute_jaccard_topk(a: np.ndarray, b: np.ndarray, k: int = K) -> float:
    """Jaccard overlap between top-k indices of two vectors."""
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    union = top_a | top_b
    if not union:
        return 0.0
    return len(top_a & top_b) / len(union)


def compute_ssc_weights(rho_trajectory: np.ndarray) -> np.ndarray:
    """SSC weights η_t = exp(-ρ_t) / Σ exp(-ρ_τ), PTQ4DiT Eq. 11."""
    neg_rho = -rho_trajectory
    neg_rho -= neg_rho.max()  # numerical stability
    exp_vals = np.exp(neg_rho)
    return exp_vals / (exp_vals.sum() + 1e-12)


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------

def temporal_cov_per_channel(trajectory: np.ndarray) -> np.ndarray:
    """Coefficient of variation per channel across sigma steps.

    trajectory: [num_steps, d_in]. Returns [d_in].
    """
    mean = trajectory.mean(axis=0)
    std = trajectory.std(axis=0)
    return std / (mean + 1e-10)


def topk_stability(trajectory: np.ndarray, k: int = K) -> dict:
    """Top-k identity stability across sigma steps."""
    num_steps = trajectory.shape[0]
    top_k_sets = [set(np.argsort(trajectory[t])[-k:]) for t in range(num_steps)]

    early_late_jaccard = (
        len(top_k_sets[0] & top_k_sets[-1])
        / max(len(top_k_sets[0] | top_k_sets[-1]), 1)
    )

    consecutive = []
    for t in range(num_steps - 1):
        union = top_k_sets[t] | top_k_sets[t + 1]
        consecutive.append(
            len(top_k_sets[t] & top_k_sets[t + 1]) / max(len(union), 1)
        )

    return {
        "early_late_jaccard": early_late_jaccard,
        "consecutive_jaccards": np.array(consecutive),
        "mean_consecutive_jaccard": float(np.mean(consecutive)) if consecutive else 0.0,
    }


def classify_temporal_behavior(
    trajectory: np.ndarray, k: int = K,
) -> str:
    """Classify temporal behavior: stable / monotonic / regime_shift / oscillatory."""
    cov = temporal_cov_per_channel(trajectory)
    stability = topk_stability(trajectory, k)

    mean_cov = float(cov.mean())
    el_jaccard = stability["early_late_jaccard"]

    if mean_cov < 0.1 and el_jaccard > 0.8:
        return "stable"

    medians = np.median(trajectory, axis=1)
    diffs = np.diff(medians)
    if np.all(diffs > 0) or np.all(diffs < 0):
        return "monotonic"

    consec = stability["consecutive_jaccards"]
    if len(consec) > 0 and consec.min() < 0.3:
        return "regime_shift"

    return "oscillatory"


# ---------------------------------------------------------------------------
# Pairwise Jaccard heatmap data
# ---------------------------------------------------------------------------

def pairwise_topk_jaccard(trajectory: np.ndarray, k: int = K) -> np.ndarray:
    """Symmetric [num_steps, num_steps] Jaccard matrix of top-k channel sets."""
    num_steps = trajectory.shape[0]
    top_k_sets = [set(np.argsort(trajectory[t])[-k:]) for t in range(num_steps)]
    mat = np.zeros((num_steps, num_steps))
    for i in range(num_steps):
        for j in range(i, num_steps):
            union = top_k_sets[i] | top_k_sets[j]
            val = len(top_k_sets[i] & top_k_sets[j]) / max(len(union), 1)
            mat[i, j] = val
            mat[j, i] = val
    return mat


# ---------------------------------------------------------------------------
# Summary table builder
# ---------------------------------------------------------------------------

def build_summary_table(
    registry: list[dict],
    weight_stats: dict,
    output_dir: Optional[Path] = None,
    k: int = K,
) -> List[dict]:
    """Build the diagnostic summary table (Section 10.18).

    Returns a list of dicts, one per layer, with all diagnostic columns.
    """
    rows = []
    act_dir = output_dir or ACTIVATION_STATS_DIR

    for entry in registry:
        name = entry["name"]
        layer_path = act_dir / f"{name}.npz"
        if not layer_path.exists():
            logger.warning("No activation data for %s — skipping", name)
            continue

        traj_data = load_trajectory(name, output_dir)
        trajectory = traj_data["act_channel_max"]  # [num_steps, d_in]
        sigma_values = traj_data["sigma_values"]

        wt = weight_stats.get(name)
        if wt is None:
            logger.warning("No weight data for %s — skipping", name)
            continue
        wt_salience = wt["w_channel_max"]

        rho_traj = compute_spearman_trajectory(trajectory, wt_salience)
        cov = temporal_cov_per_channel(trajectory)
        stability = topk_stability(trajectory, k)

        act_conc_vals = []
        jaccard_vals = []
        for t in range(trajectory.shape[0]):
            act_conc_vals.append(salience_concentration(trajectory[t], k))
            jaccard_vals.append(compute_jaccard_topk(trajectory[t], wt_salience, k))

        wt_conc = salience_concentration(wt_salience, k)

        rows.append({
            "layer_name": name,
            "family": entry["family"],
            "side": entry["side"],
            "block": entry["block"],
            "d_in": entry["d_in"],
            "mean_act_salience": float(trajectory.mean()),
            "max_act_salience": float(trajectory.max()),
            "mean_wt_salience": float(wt_salience.mean()),
            "max_wt_salience": float(wt_salience.max()),
            "top1_median_ratio_act": float(
                np.mean([v["top1_over_median"] for v in act_conc_vals])
            ),
            "top1_median_ratio_wt": wt_conc["top1_over_median"],
            "gini_act": float(np.mean([v["gini"] for v in act_conc_vals])),
            "gini_wt": wt_conc["gini"],
            "mean_spearman_rho": float(rho_traj.mean()),
            "std_spearman_rho": float(rho_traj.std()),
            "min_spearman_rho": float(rho_traj.min()),
            "max_spearman_rho": float(rho_traj.max()),
            "mean_jaccard_topk": float(np.mean(jaccard_vals)),
            "cov_temporal": float(cov.mean()),
            "early_late_topk_jaccard": stability["early_late_jaccard"],
        })

    _add_risk_scores(rows)

    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    logger.info("Built summary table with %d rows", len(rows))
    return rows


def _add_risk_scores(rows: List[dict]):
    """Add normalized composite risk_score to each row in-place."""
    if not rows:
        return

    def _normalize(vals):
        arr = np.array(vals, dtype=np.float64)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    max_act = _normalize([r["max_act_salience"] for r in rows])
    max_wt = _normalize([r["max_wt_salience"] for r in rows])
    mean_rho = _normalize([r["mean_spearman_rho"] for r in rows])
    cov_t = _normalize([r["cov_temporal"] for r in rows])

    for i, row in enumerate(rows):
        row["risk_score"] = float(
            0.4 * max_act[i]
            + 0.2 * max_wt[i]
            + 0.2 * mean_rho[i]
            + 0.2 * cov_t[i]
        )


def save_summary_table(rows: List[dict], output_dir: Optional[Path] = None):
    """Save summary table as CSV."""
    import csv

    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary_table.csv"

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved summary table to %s", path)
