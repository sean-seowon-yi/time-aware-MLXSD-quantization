"""Post-quantization diagnostic visualizations.

Comparison plots between FP16 baseline (Phase 1) and W4A8 quantized model:
  - Activation trajectory overlays
  - Weight error bar charts and heatmaps
  - Per-layer SNR ranking
  - Summary dashboard

Color conventions (extending Phase 1):
  FP16 baseline → solid blue   (#3498db)
  W4A8 quantized → dashed red  (#e74c3c)
  error/diff    → orange       (#e67e22)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PLOTS_DIR = Path("post_quant_diagnostics/plots")

COLOR_FP16 = "#3498db"
COLOR_W4A8 = "#e74c3c"
COLOR_ERROR = "#e67e22"
COLOR_IMAGE = "#3498db"
COLOR_TEXT = "#e67e22"


def _save(fig, name: str, output_dir: Optional[Path] = None):
    d = output_dir or DEFAULT_PLOTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", name)


# ===================================================================
# 1. Activation trajectory overlay (FP16 vs W4A8)
# ===================================================================

def plot_trajectory_overlay(
    name: str,
    fp16_traj: np.ndarray,
    w4a8_traj: np.ndarray,
    sigma_values: np.ndarray | None = None,
    output_dir: Optional[Path] = None,
):
    """Overlay FP16 and W4A8 per-channel-max activation trajectories.

    Shows mean across channels with shaded min/max envelopes.
    """
    n_steps = min(fp16_traj.shape[0], w4a8_traj.shape[0])
    x = sigma_values[:n_steps] if sigma_values is not None else np.arange(n_steps)
    xlabel = "σ (noise level)" if sigma_values is not None else "Step index"

    fp16_mean = fp16_traj[:n_steps].mean(axis=1)
    fp16_max = fp16_traj[:n_steps].max(axis=1)
    w4a8_mean = w4a8_traj[:n_steps].mean(axis=1)
    w4a8_max = w4a8_traj[:n_steps].max(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(f"Activation Trajectory: {name}", fontsize=12)

    ax1.plot(x, fp16_mean, "-", color=COLOR_FP16, lw=1.5, label="FP16 mean")
    ax1.plot(x, w4a8_mean, "--", color=COLOR_W4A8, lw=1.5, label="W4A8 mean")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Mean channel-max activation")
    ax1.legend()
    ax1.set_title("Mean across channels")

    ax2.plot(x, fp16_max, "-", color=COLOR_FP16, lw=1.5, label="FP16 max")
    ax2.plot(x, w4a8_max, "--", color=COLOR_W4A8, lw=1.5, label="W4A8 max")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Max channel-max activation")
    ax2.legend()
    ax2.set_title("Worst-case channel")

    _save(fig, f"traj_overlay_{name}", output_dir)


# ===================================================================
# 2. Activation error per timestep
# ===================================================================

def plot_activation_error_over_time(
    name: str,
    per_step_mse: np.ndarray,
    per_step_snr: np.ndarray,
    sigma_values: np.ndarray | None = None,
    output_dir: Optional[Path] = None,
):
    """MSE and SNR between FP16 and W4A8 activations across timesteps."""
    n_steps = len(per_step_mse)
    x = sigma_values[:n_steps] if sigma_values is not None else np.arange(n_steps)
    xlabel = "σ (noise level)" if sigma_values is not None else "Step index"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    fig.suptitle(f"Activation Error Over Time: {name}", fontsize=12)

    ax1.plot(x, per_step_mse, "o-", color=COLOR_ERROR, markersize=3)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("MSE (FP16 vs W4A8)")
    ax1.set_title("Per-step MSE")

    ax2.plot(x, per_step_snr, "o-", color=COLOR_FP16, markersize=3)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("SNR (dB)")
    ax2.set_title("Per-step SNR")
    ax2.axhline(y=20, color="gray", ls="--", alpha=0.5, label="20 dB")
    ax2.legend()

    _save(fig, f"act_error_time_{name}", output_dir)


# ===================================================================
# 3. Channel-wise activation error heatmap
# ===================================================================

def plot_channel_error_heatmap(
    name: str,
    fp16_traj: np.ndarray,
    w4a8_traj: np.ndarray,
    output_dir: Optional[Path] = None,
):
    """Heatmap of |FP16 - W4A8| per (timestep, channel)."""
    n_steps = min(fp16_traj.shape[0], w4a8_traj.shape[0])
    diff = np.abs(w4a8_traj[:n_steps] - fp16_traj[:n_steps])

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    im = ax.imshow(diff, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Timestep")
    ax.set_title(f"Activation Error |W4A8 − FP16|: {name}")
    fig.colorbar(im, ax=ax, label="Absolute error")
    _save(fig, f"channel_error_heatmap_{name}", output_dir)


# ===================================================================
# 4. Weight error bar chart (per-layer MSE ranking)
# ===================================================================

def plot_weight_error_ranking(
    weight_errors: list[dict],
    top_n: int = 40,
    output_dir: Optional[Path] = None,
):
    """Horizontal bar chart of per-layer weight quantization error (top N)."""
    quantized = [e for e in weight_errors if e["quantized"] and e["mse"] > 0]
    quantized.sort(key=lambda e: e["mse"], reverse=True)
    top = quantized[:top_n]

    if not top:
        logger.warning("No quantized layers with error > 0")
        return

    names = [e["name"] for e in top]
    mses = [e["mse"] for e in top]
    colors = [COLOR_IMAGE if e["side"] == "image" else COLOR_TEXT for e in top]

    fig, ax = plt.subplots(
        figsize=(10, max(6, top_n * 0.3)), constrained_layout=True,
    )
    y = np.arange(len(names))
    ax.barh(y, mses, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Weight MSE (channel-max: FP16 vs dequantized W4)")
    ax.set_title(f"Top-{top_n} Layers by Weight Quantization Error")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=COLOR_IMAGE, label="image"),
                 Patch(color=COLOR_TEXT, label="text")],
        loc="lower right",
    )
    _save(fig, "weight_error_ranking", output_dir)


# ===================================================================
# 5. Weight SNR distribution
# ===================================================================

def plot_weight_snr_distribution(
    weight_errors: list[dict],
    output_dir: Optional[Path] = None,
):
    """Histogram + CDF of per-layer weight SNR (dB)."""
    snrs = [e["snr_db"] for e in weight_errors
            if e["quantized"] and e["snr_db"] is not None and np.isfinite(e["snr_db"])]

    if not snrs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax1.hist(snrs, bins=30, color=COLOR_FP16, alpha=0.7, edgecolor="white")
    ax1.axvline(np.median(snrs), color=COLOR_W4A8, ls="--",
                label=f"median={np.median(snrs):.1f} dB")
    ax1.set_xlabel("Weight SNR (dB)")
    ax1.set_ylabel("Layer count")
    ax1.set_title("Weight SNR Distribution")
    ax1.legend()

    sorted_snrs = np.sort(snrs)
    cdf = np.arange(1, len(sorted_snrs) + 1) / len(sorted_snrs)
    ax2.plot(sorted_snrs, cdf, "-", color=COLOR_FP16)
    ax2.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax2.set_xlabel("Weight SNR (dB)")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_title("Weight SNR CDF")

    _save(fig, "weight_snr_distribution", output_dir)


# ===================================================================
# 6. Activation SNR ranking (per-layer)
# ===================================================================

def plot_activation_snr_ranking(
    comparisons: list[dict],
    top_n: int = 40,
    output_dir: Optional[Path] = None,
):
    """Bar chart of per-layer activation SNR (lowest = most degraded)."""
    sorted_comp = sorted(comparisons, key=lambda c: c["overall_snr"])
    top = sorted_comp[:top_n]

    names = [c["name"] for c in top]
    snrs = [c["overall_snr"] for c in top]

    fig, ax = plt.subplots(
        figsize=(10, max(6, top_n * 0.3)), constrained_layout=True,
    )
    y = np.arange(len(names))
    colors = [COLOR_W4A8 if s < 20 else COLOR_FP16 for s in snrs]
    ax.barh(y, snrs, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(20, color="gray", ls="--", alpha=0.5, label="20 dB threshold")
    ax.set_xlabel("Activation SNR (dB)")
    ax.set_title(f"Bottom-{top_n} Layers by Activation SNR")
    ax.legend()
    _save(fig, "activation_snr_ranking", output_dir)


# ===================================================================
# 7. b_inv distribution analysis
# ===================================================================

def plot_b_inv_distributions(
    calibration_path: Path,
    b_inv_layers: list[str],
    output_dir: Optional[Path] = None,
):
    """Histogram and statistics of b_inv vectors for online-balanced layers."""
    cal = np.load(calibration_path)

    all_binv = []
    layer_stats = []

    for name in b_inv_layers:
        if name not in cal:
            continue
        b = cal[name]
        binv = 1.0 / np.clip(b, 1e-8, None)
        all_binv.append(binv)
        layer_stats.append({
            "name": name,
            "binv_mean": float(binv.mean()),
            "binv_max": float(binv.max()),
            "binv_std": float(binv.std()),
        })

    if not all_binv:
        return

    all_flat = np.concatenate(all_binv)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    axes[0].hist(all_flat, bins=100, color=COLOR_ERROR, alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("b_inv value")
    axes[0].set_ylabel("Channel count")
    axes[0].set_title("b_inv Distribution (all online layers)")
    axes[0].axvline(1.0, color="gray", ls="--", alpha=0.5, label="b_inv=1 (no scaling)")
    axes[0].legend()

    log_binv = np.log10(all_flat + 1e-12)
    axes[1].hist(log_binv, bins=100, color=COLOR_FP16, alpha=0.7, edgecolor="white")
    axes[1].set_xlabel("log₁₀(b_inv)")
    axes[1].set_ylabel("Channel count")
    axes[1].set_title("b_inv Distribution (log scale)")

    layer_stats.sort(key=lambda s: s["binv_max"], reverse=True)
    top = layer_stats[:20]
    names = [s["name"] for s in top]
    maxes = [s["binv_max"] for s in top]
    y = np.arange(len(names))
    axes[2].barh(y, maxes, color=COLOR_W4A8, alpha=0.8)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names, fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Max b_inv")
    axes[2].set_title("Top-20 Layers by Max b_inv")

    _save(fig, "b_inv_distributions", output_dir)


# ===================================================================
# 8. Summary dashboard
# ===================================================================

def plot_summary_dashboard(
    weight_errors: list[dict],
    act_comparisons: list[dict],
    output_dir: Optional[Path] = None,
):
    """2×2 summary dashboard with key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("Post-Quantization Diagnostic Summary (W4A8 vs FP16)", fontsize=14)

    # (0,0) Weight MSE distribution by family
    ax = axes[0, 0]
    families = {}
    for e in weight_errors:
        if not e["quantized"]:
            continue
        f = e["family"]
        families.setdefault(f, []).append(e["mse"])
    if families:
        labels = sorted(families.keys())
        data = [families[f] for f in labels]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR_FP16)
            patch.set_alpha(0.6)
    ax.set_ylabel("Weight MSE")
    ax.set_title("Weight Error by Layer Family")
    ax.tick_params(axis="x", rotation=30)

    # (0,1) Activation SNR distribution by family
    ax = axes[0, 1]
    families = {}
    for c in act_comparisons:
        parts = c["name"].split(".")
        if len(parts) >= 5:
            f = parts[4]
        elif c["name"] == "final_layer.linear":
            f = "final_linear"
        elif c["name"] == "context_embedder":
            f = "context_embedder"
        else:
            f = "other"
        families.setdefault(f, []).append(c["overall_snr"])
    if families:
        labels = sorted(families.keys())
        data = [families[f] for f in labels]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR_W4A8)
            patch.set_alpha(0.6)
    ax.set_ylabel("Activation SNR (dB)")
    ax.set_title("Activation SNR by Layer Family")
    ax.axhline(20, color="gray", ls="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=30)

    # (1,0) Weight SNR vs block depth
    ax = axes[1, 0]
    for side, color in [("image", COLOR_IMAGE), ("text", COLOR_TEXT)]:
        filtered = [e for e in weight_errors
                    if e["quantized"] and e["side"] == side and e["block"] >= 0
                    and e["snr_db"] is not None and np.isfinite(e["snr_db"])]
        blocks = [e["block"] for e in filtered]
        snrs = [e["snr_db"] for e in filtered]
        if blocks:
            ax.scatter(blocks, snrs, c=color, alpha=0.4, s=15, label=side)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Weight SNR (dB)")
    ax.set_title("Weight SNR vs Block Depth")
    ax.legend()

    # (1,1) Activation MSE vs block depth
    ax = axes[1, 1]
    for side, color in [("image", COLOR_IMAGE), ("text", COLOR_TEXT)]:
        items = [(c["name"], c["overall_mse"]) for c in act_comparisons]
        blocks_vals = []
        for name, mse in items:
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "blocks":
                bidx = int(parts[1])
                s = parts[2] if len(parts) > 2 else ""
                if s == side:
                    blocks_vals.append((bidx, mse))
        if blocks_vals:
            bs, ms = zip(*blocks_vals)
            ax.scatter(bs, ms, c=color, alpha=0.4, s=15, label=side)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Activation MSE")
    ax.set_title("Activation Error vs Block Depth")
    ax.legend()

    _save(fig, "post_quant_summary_dashboard", output_dir)


# ===================================================================
# 9. Activation trajectory grid (selected layers)
# ===================================================================

def plot_trajectory_grid(
    comparisons: list[dict],
    selected_layers: list[str] | None = None,
    sigma_values: np.ndarray | None = None,
    output_dir: Optional[Path] = None,
):
    """Grid of trajectory overlays for selected representative layers."""
    comp_map = {c["name"]: c for c in comparisons}

    if selected_layers is None:
        sorted_comp = sorted(comparisons, key=lambda c: c["overall_mse"], reverse=True)
        selected_layers = [c["name"] for c in sorted_comp[:12]]

    n = len(selected_layers)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, name in enumerate(selected_layers):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        comp = comp_map.get(name)
        if comp is None:
            ax.set_visible(False)
            continue

        fp16_mean = comp["fp16_traj"].mean(axis=1)
        w4a8_mean = comp["w4a8_traj"].mean(axis=1)
        n_steps = len(fp16_mean)
        x = sigma_values[:n_steps] if sigma_values is not None else np.arange(n_steps)

        ax.plot(x, fp16_mean, "-", color=COLOR_FP16, lw=1.2, label="FP16")
        ax.plot(x, w4a8_mean, "--", color=COLOR_W4A8, lw=1.2, label="W4A8")
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("FP16 vs W4A8 Activation Trajectories (mean channel-max)", fontsize=12)
    _save(fig, "trajectory_comparison_grid", output_dir)
