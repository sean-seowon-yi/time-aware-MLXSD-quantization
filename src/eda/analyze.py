"""
Phase 1 EDA: Statistical analysis producing CSV tables.

Input:  eda_output/activation_stats_full.npz  (from profile_activations.py)
        eda_output/weight_stats.npz            (from profile_activations.py)

Outputs (all in eda_output/tables/):
  tvc_ranking.csv          — TVC per (family, layer), sorted desc, with SSB flag
  outlier_channel_count.csv — per-(family, layer, timestep) outlier channel fraction
  ssb_group_assignment.csv — K=4 k-means group boundaries for high-TVC layers
  cross_stream_stats.csv   — Wilcoxon test: txt-stream outlier fraction vs. baseline
  qkv_outlier_source.csv   — per-block fraction of joint-sequence outliers from img/txt

CLI:
  python -m src.eda.analyze
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ACT = str(_ROOT / "eda_output" / "activation_stats_full.npz")
_DEFAULT_WGT = str(_ROOT / "eda_output" / "weight_stats.npz")
_DEFAULT_TABLES = str(_ROOT / "eda_output" / "tables")

TVC_THRESHOLD = 0.2         # SSB candidacy
OUTLIER_Z_THRESHOLD = 5.0   # robust z-score
SSB_K = 4                   # number of k-means clusters
TXT_SEQ_LEN = 77
IMG_SEQ_LEN = 1024
TXT_BASELINE = TXT_SEQ_LEN / (TXT_SEQ_LEN + IMG_SEQ_LEN)  # ≈ 0.0700
WILCOXON_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_activation_stats(
    path: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Load activation stats → {family: {layer_id: {t_key: {stat_name: array}}}}"""
    from .eda_tracer import load_tracer_stats
    return load_tracer_stats(path)


def _channel_range(layer_t_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-channel range at one (layer, timestep)."""
    return layer_t_stats["max"] - layer_t_stats["min"]


def _sorted_timesteps(t_map: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
    return sorted(t_map.keys(), key=lambda k: float(k))


def _robust_outlier_mask(values: np.ndarray, threshold: float = OUTLIER_Z_THRESHOLD) -> np.ndarray:
    """Return boolean mask of outlier channels via MAD-based z-score."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-10:
        return np.zeros(len(values), dtype=bool)
    z = np.abs(values - median) / (1.4826 * mad)
    return z > threshold


# ---------------------------------------------------------------------------
# TVC (temporal variation coefficient)
# ---------------------------------------------------------------------------

def compute_tvc(
    act_stats: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
) -> List[Dict]:
    """
    Compute TVC = std_t(median_channel_range) / mean_t(median_channel_range)
    for each (family, layer_id).

    Returns list of dicts with keys: family, layer_id, tvc, ssb_candidate,
    block_idx, stream, median_range_mean, median_range_std.
    """
    rows = []
    for family, layer_map in act_stats.items():
        for layer_id, t_map in layer_map.items():
            t_keys = _sorted_timesteps(t_map)
            if len(t_keys) < 2:
                continue

            median_ranges = []
            for t_key in t_keys:
                ch_range = _channel_range(t_map[t_key])
                median_ranges.append(float(np.median(ch_range)))

            arr = np.array(median_ranges)
            mean_r = float(arr.mean())
            std_r = float(arr.std())
            tvc = std_r / mean_r if mean_r > 1e-8 else 0.0

            # Parse block_idx and stream from layer_id (e.g. "mm_03_img")
            parts = layer_id.split("_")
            block_idx = int(parts[1]) if len(parts) >= 2 else -1
            stream = parts[2] if len(parts) >= 3 else "unk"

            rows.append({
                "family": family,
                "layer_id": layer_id,
                "block_idx": block_idx,
                "stream": stream,
                "tvc": round(tvc, 6),
                "ssb_candidate": tvc > TVC_THRESHOLD,
                "median_range_mean": round(mean_r, 6),
                "median_range_std": round(std_r, 6),
                "n_timesteps": len(t_keys),
            })

    rows.sort(key=lambda r: r["tvc"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Outlier channel counts
# ---------------------------------------------------------------------------

def compute_outlier_counts(
    act_stats: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
) -> List[Dict]:
    """
    For each (family, layer_id, timestep), count outlier channels and fraction.
    Outlier: robust z-score of per-channel range > OUTLIER_Z_THRESHOLD.
    """
    rows = []
    for family, layer_map in act_stats.items():
        for layer_id, t_map in layer_map.items():
            parts = layer_id.split("_")
            block_idx = int(parts[1]) if len(parts) >= 2 else -1
            stream = parts[2] if len(parts) >= 3 else "unk"

            for t_key in _sorted_timesteps(t_map):
                ch_range = _channel_range(t_map[t_key])
                n_channels = len(ch_range)
                outlier_mask = _robust_outlier_mask(ch_range)
                n_outliers = int(outlier_mask.sum())
                rows.append({
                    "family": family,
                    "layer_id": layer_id,
                    "block_idx": block_idx,
                    "stream": stream,
                    "timestep": float(t_key),
                    "n_channels": n_channels,
                    "n_outliers": n_outliers,
                    "outlier_fraction": round(n_outliers / max(n_channels, 1), 6),
                })
    return rows


# ---------------------------------------------------------------------------
# SSB group assignment (K-means on timestep activation range vectors)
# ---------------------------------------------------------------------------

def compute_ssb_groups(
    act_stats: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
    tvc_rows: List[Dict],
    k: int = SSB_K,
) -> List[Dict]:
    """
    For layers with TVC > TVC_THRESHOLD, cluster the 25 timesteps into K groups
    using K-means on per-channel activation range vectors.

    Feature vector for timestep t: per-channel range (D,).
    K-means on the (n_timesteps, D) matrix → group assignments.

    Returns list of dicts with group assignments and within-group variance fraction.
    """
    from sklearn.cluster import KMeans  # type: ignore

    ssb_layer_ids = {
        (r["family"], r["layer_id"])
        for r in tvc_rows
        if r["ssb_candidate"]
    }

    rows = []
    for family, layer_map in act_stats.items():
        for layer_id, t_map in layer_map.items():
            if (family, layer_id) not in ssb_layer_ids:
                continue

            t_keys = _sorted_timesteps(t_map)
            if len(t_keys) < k:
                continue

            # Feature matrix: (n_timesteps, D)
            X = np.stack([_channel_range(t_map[tk]) for tk in t_keys], axis=0)

            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)

            # Within-group variance fraction
            total_var = float(np.var(X, axis=0).mean())
            within_var = 0.0
            for g in range(k):
                mask = labels == g
                if mask.sum() > 1:
                    within_var += float(np.var(X[mask], axis=0).mean()) * mask.sum()
            within_var /= len(t_keys)
            wgvf = within_var / total_var if total_var > 1e-10 else 0.0

            for i, (tk, label) in enumerate(zip(t_keys, labels)):
                rows.append({
                    "family": family,
                    "layer_id": layer_id,
                    "timestep_idx": i,
                    "timestep": float(tk),
                    "group": int(label),
                    "within_group_var_fraction": round(wgvf, 6),
                })
    return rows


# ---------------------------------------------------------------------------
# Cross-stream significance (Wilcoxon test for A12)
# ---------------------------------------------------------------------------

def compute_cross_stream_stats(
    outlier_rows: List[Dict],
    alpha: float = WILCOXON_ALPHA,
) -> List[Dict]:
    """
    For each block (0-23), test whether the txt-stream outlier fraction
    for q_proj/k_proj/v_proj deviates significantly from the baseline
    expected txt fraction (TXT_BASELINE ≈ 0.070).

    We compare img-stream and txt-stream outlier fractions per (block, timestep)
    using the Wilcoxon signed-rank test (txt_frac vs img_frac, testing symmetry
    around 0 of the difference). Also report the txt-vs-baseline test.

    Bonferroni correction: 24 blocks → threshold α/24 ≈ 0.0021.
    """
    from scipy.stats import wilcoxon  # type: ignore

    n_blocks = 24
    bonferroni_alpha = alpha / n_blocks
    rows = []

    for proj in ("q_proj", "k_proj", "v_proj"):
        for block_idx in range(n_blocks):
            img_id = f"mm_{block_idx:02d}_img"
            txt_id = f"mm_{block_idx:02d}_txt"

            img_fracs = [
                r["outlier_fraction"]
                for r in outlier_rows
                if r["family"] == proj and r["layer_id"] == img_id
            ]
            txt_fracs = [
                r["outlier_fraction"]
                for r in outlier_rows
                if r["family"] == proj and r["layer_id"] == txt_id
            ]

            if not img_fracs or not txt_fracs:
                continue

            img_arr = np.array(img_fracs)
            txt_arr = np.array(txt_fracs[:len(img_arr)])  # align length if needed

            # Median outlier fractions
            img_median = float(np.median(img_arr))
            txt_median = float(np.median(txt_arr))

            # Relative txt contribution: txt_median / (img_median + txt_median)
            total = img_median + txt_median
            txt_relative = txt_median / total if total > 1e-10 else float("nan")

            # Wilcoxon test: txt_frac - img_frac != 0
            diffs = txt_arr - img_arr
            if len(diffs) > 0 and not np.allclose(diffs, 0):
                try:
                    stat, p_val = wilcoxon(diffs)
                except Exception:
                    stat, p_val = float("nan"), float("nan")
            else:
                stat, p_val = float("nan"), 1.0

            significant = (
                (not np.isnan(p_val)) and
                (p_val < bonferroni_alpha) and
                (txt_median > 2 * TXT_BASELINE)
            )

            rows.append({
                "projection": proj,
                "block_idx": block_idx,
                "img_median_outlier_frac": round(img_median, 6),
                "txt_median_outlier_frac": round(txt_median, 6),
                "txt_relative_fraction": round(txt_relative, 6),
                "txt_baseline_fraction": round(TXT_BASELINE, 6),
                "txt_exceeds_2x_baseline": txt_median > 2 * TXT_BASELINE,
                "wilcoxon_stat": round(float(stat), 4) if not np.isnan(stat) else float("nan"),
                "p_value": round(float(p_val), 6) if not np.isnan(p_val) else float("nan"),
                "bonferroni_alpha": round(bonferroni_alpha, 6),
                "significant_after_bonferroni": significant,
            })

    return rows


# ---------------------------------------------------------------------------
# QKV outlier source (per-block img vs. txt contribution)
# ---------------------------------------------------------------------------

def compute_qkv_outlier_source(outlier_rows: List[Dict]) -> List[Dict]:
    """
    For each (block, projection, timestep), compute what fraction of total
    outlier channels come from the img vs. txt stream.

    Since img and txt have the same channel dimensionality D (1536), we compare
    per-stream outlier COUNTS. The "joint" outlier count is the union:
      joint_outliers = img_outliers + txt_outliers - both_outliers
    We report img_fraction = img_outliers / joint_count and txt_fraction similarly.

    Note: for this analysis we treat channels independently across streams
    (they share the same D-dimensional hidden space).
    """
    from collections import defaultdict

    # Build lookup: (family, layer_id, timestep) → outlier count
    lookup: dict = defaultdict(float)
    for r in outlier_rows:
        lookup[(r["family"], r["layer_id"], r["timestep"])] = r["n_outliers"]

    rows = []
    for proj in ("q_proj", "k_proj", "v_proj"):
        for block_idx in range(24):
            img_id = f"mm_{block_idx:02d}_img"
            txt_id = f"mm_{block_idx:02d}_txt"

            # Collect matching timesteps
            img_timesteps = {
                r["timestep"]: r["n_outliers"]
                for r in outlier_rows
                if r["family"] == proj and r["layer_id"] == img_id
            }
            txt_timesteps = {
                r["timestep"]: r["n_outliers"]
                for r in outlier_rows
                if r["family"] == proj and r["layer_id"] == txt_id
            }

            shared_ts = sorted(set(img_timesteps) & set(txt_timesteps))
            if not shared_ts:
                continue

            img_total = sum(img_timesteps[t] for t in shared_ts)
            txt_total = sum(txt_timesteps[t] for t in shared_ts)
            joint_total = img_total + txt_total  # union (channels are independent per stream)

            img_frac = img_total / joint_total if joint_total > 0 else float("nan")
            txt_frac = txt_total / joint_total if joint_total > 0 else float("nan")

            rows.append({
                "projection": proj,
                "block_idx": block_idx,
                "n_timesteps": len(shared_ts),
                "img_total_outliers": img_total,
                "txt_total_outliers": txt_total,
                "joint_total_outliers": joint_total,
                "img_fraction": round(img_frac, 6),
                "txt_fraction": round(txt_frac, 6),
                "txt_baseline_fraction": round(TXT_BASELINE, 6),
                "txt_exceeds_2x_baseline": (txt_frac > 2 * TXT_BASELINE) if not np.isnan(txt_frac) else False,
            })

    return rows


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_csv(rows: List[Dict], path: str) -> None:
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    act_stats_path: str,
    weight_stats_path: str,
    tables_dir: str,
) -> None:
    """Run all analyses and write CSV tables."""
    print(f"Loading activation stats from {act_stats_path}...")
    act_stats = _parse_activation_stats(act_stats_path)

    families = list(act_stats.keys())
    n_layers = sum(len(lm) for lm in act_stats.values())
    print(f"  Families: {families}")
    print(f"  Total (family, layer) pairs: {n_layers}")

    tables = Path(tables_dir)

    # TVC ranking
    print("Computing TVC...")
    tvc_rows = compute_tvc(act_stats)
    _save_csv(tvc_rows, str(tables / "tvc_ranking.csv"))
    ssb_count = sum(1 for r in tvc_rows if r["ssb_candidate"])
    print(f"  {len(tvc_rows)} (family, layer) pairs; {ssb_count} SSB candidates (TVC > {TVC_THRESHOLD})")

    # Outlier counts
    print("Computing outlier channel counts...")
    outlier_rows = compute_outlier_counts(act_stats)
    _save_csv(outlier_rows, str(tables / "outlier_channel_count.csv"))
    print(f"  {len(outlier_rows)} (family, layer, timestep) entries")

    # SSB group assignment
    print(f"Computing SSB K={SSB_K} groups for high-TVC layers...")
    try:
        ssb_rows = compute_ssb_groups(act_stats, tvc_rows)
        _save_csv(ssb_rows, str(tables / "ssb_group_assignment.csv"))
        print(f"  {len(ssb_rows)} (layer, timestep) group assignments")
    except ImportError:
        print("  scikit-learn not available; skipping SSB group assignment")
        ssb_rows = []

    # Cross-stream significance
    print("Computing cross-stream Wilcoxon tests...")
    try:
        cross_rows = compute_cross_stream_stats(outlier_rows)
        _save_csv(cross_rows, str(tables / "cross_stream_stats.csv"))
        sig_count = sum(1 for r in cross_rows if r["significant_after_bonferroni"])
        print(f"  {len(cross_rows)} (proj, block) tests; {sig_count} significant after Bonferroni")
    except ImportError:
        print("  scipy not available; skipping Wilcoxon tests")
        cross_rows = []

    # QKV outlier source
    print("Computing QKV outlier source fractions...")
    qkv_rows = compute_qkv_outlier_source(outlier_rows)
    _save_csv(qkv_rows, str(tables / "qkv_outlier_source.csv"))
    txt_excess = sum(1 for r in qkv_rows if r["txt_exceeds_2x_baseline"])
    print(f"  {len(qkv_rows)} (proj, block) entries; {txt_excess} with txt > 2x baseline")

    print(f"\nAll tables written to {tables_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 EDA: Statistical analysis")
    parser.add_argument("--act-stats", type=str, default=_DEFAULT_ACT)
    parser.add_argument("--weight-stats", type=str, default=_DEFAULT_WGT)
    parser.add_argument("--tables-dir", type=str, default=_DEFAULT_TABLES)
    args = parser.parse_args()

    run_analysis(args.act_stats, args.weight_stats, args.tables_dir)


if __name__ == "__main__":
    main()
