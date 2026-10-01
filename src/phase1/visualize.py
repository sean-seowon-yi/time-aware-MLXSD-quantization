"""Phase 1 diagnostic visualizations (Sections 10.1–10.18).

Plots 10.1–10.3: faithful SD3 adaptations of PTQ4DiT Figures 3, 4, 1-Left.
Plots 10.4–10.17: SD3-specific extensions.
Plot 10.18: summary diagnostic table (handled in analyze.py as CSV).

Consistent color conventions:
  image-side  → blue   (#3498db)
  text-side   → orange (#e67e22)
  shared      → green  (#27ae60)
  salient k   → red    (#e74c3c)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from .analyze import (
    compute_jaccard_topk,
    compute_spearman_trajectory,
    compute_ssc_weights,
    gini_coefficient,
    pairwise_topk_jaccard,
    per_channel_quant_mse_activation,
    per_channel_quant_mse_weight,
    temporal_cov_per_channel,
    topk_stability,
)
from .config import DIAG_CONFIG, FAMILY_COLORS, PLOTS_DIR, SIDE_COLORS

logger = logging.getLogger(__name__)

K = DIAG_CONFIG["top_k"]


def _save(fig, name: str, output_dir: Optional[Path] = None):
    d = output_dir or PLOTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", name)


# ===================================================================
# 10.1  Per-channel magnitude + quantization error  (PTQ4DiT Fig. 3)
# ===================================================================

def plot_fig3_reproduction(
    layer_name: str,
    act_channel_max: np.ndarray,
    wt_channel_max: np.ndarray,
    act_channel_mse: np.ndarray,
    wt_channel_mse: np.ndarray,
    k: int = K,
    output_dir: Optional[Path] = None,
):
    fig, (ax_act, ax_wt) = plt.subplots(
        1, 2, figsize=(16, 5), constrained_layout=True,
    )
    d_in = len(act_channel_max)
    ch = np.arange(d_in)

    top_k_act = set(np.argsort(act_channel_max)[-k:])
    c_act = ["#e74c3c" if j in top_k_act else "#3498db" for j in ch]
    ax_act.bar(ch, act_channel_max, color=c_act, width=1.0, edgecolor="none")
    ax_act.set_ylabel("max|activation|", color="#3498db")
    ax_act.set_xlabel("Channel index")
    ax2 = ax_act.twinx()
    ax2.plot(ch, act_channel_mse, color="#2ecc71", lw=0.8, alpha=0.7)
    ax2.set_ylabel("Quant. MSE", color="#2ecc71")
    ax_act.set_title(f"{layer_name} — Activation")

    top_k_wt = set(np.argsort(wt_channel_max)[-k:])
    c_wt = ["#e74c3c" if j in top_k_wt else "#e67e22" for j in ch]
    ax_wt.bar(ch, wt_channel_max, color=c_wt, width=1.0, edgecolor="none")
    ax_wt.set_ylabel("max|weight|", color="#e67e22")
    ax_wt.set_xlabel("Channel index")
    ax3 = ax_wt.twinx()
    ax3.plot(ch, wt_channel_mse, color="#2ecc71", lw=0.8, alpha=0.7)
    ax3.set_ylabel("Quant. MSE", color="#2ecc71")
    ax_wt.set_title(f"{layer_name} — Weight")

    fig.suptitle("Per-channel salience & quantization error (PTQ4DiT Fig.3 analog)")
    _save(fig, f"fig3_{layer_name}", output_dir)
    return fig


# ===================================================================
# 10.2  Temporal boxplot  (PTQ4DiT Fig. 4)
# ===================================================================

def plot_fig4_reproduction(
    layer_name: str,
    trajectory: np.ndarray,
    sigma_values: np.ndarray,
    k: int = K,
    output_dir: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    num_steps = trajectory.shape[0]

    ax.boxplot(
        [trajectory[t] for t in range(num_steps)],
        positions=range(num_steps),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#3498db", alpha=0.4),
        medianprops=dict(color="#e74c3c", linewidth=1.5),
    )

    for t in range(num_steps):
        idx = np.argsort(trajectory[t])[-k:]
        ax.scatter(
            [t] * k, trajectory[t, idx],
            color="#e74c3c", alpha=0.3, s=8, zorder=3,
        )

    ax.set_xticks(range(num_steps))
    ax.set_xticklabels([f"{s:.2f}" for s in sigma_values], rotation=45, fontsize=7)
    ax.set_xlabel("σ value")
    ax.set_ylabel("max|activation| per channel")
    ax.set_title(
        f"{layer_name} — Temporal activation salience (PTQ4DiT Fig.4 analog)"
    )
    _save(fig, f"fig4_{layer_name}", output_dir)
    return fig


# ===================================================================
# 10.3  Complementarity across timesteps  (PTQ4DiT Fig. 1 Left)
# ===================================================================

def plot_fig1_left_reproduction(
    layer_name: str,
    trajectory: np.ndarray,
    wt_salience: np.ndarray,
    sigma_values: np.ndarray,
    step_indices: Sequence[int],
    k: int = K,
    output_dir: Optional[Path] = None,
):
    fig, axes = plt.subplots(
        3, 2, figsize=(16, 12), sharex=True, constrained_layout=True,
    )
    d_in = len(wt_salience)
    ch = np.arange(d_in)
    top_k_wt = set(np.argsort(wt_salience)[-k:])

    for row, si in enumerate(step_indices):
        sigma = sigma_values[si]
        act_s = trajectory[si]
        top_k_act = set(np.argsort(act_s)[-k:])

        rho_val, _ = spearmanr(act_s, wt_salience)
        jac = compute_jaccard_topk(act_s, wt_salience, k)

        c_a = ["#e74c3c" if j in top_k_act else "#3498db" for j in ch]
        axes[row, 0].bar(ch, act_s, color=c_a, width=1.0, edgecolor="none")
        axes[row, 0].set_ylabel(f"σ={sigma:.2f}\nmax|act|")
        axes[row, 0].annotate(
            f"ρ={rho_val:.3f}  J={jac:.3f}",
            xy=(0.98, 0.92), xycoords="axes fraction",
            ha="right", fontsize=9, bbox=dict(boxstyle="round", fc="wheat"),
        )

        c_w = ["#e74c3c" if j in top_k_wt else "#e67e22" for j in ch]
        axes[row, 1].bar(ch, wt_salience, color=c_w, width=1.0, edgecolor="none")
        axes[row, 1].set_ylabel("max|weight|")

    axes[0, 0].set_title("Activation (vary with σ)")
    axes[0, 1].set_title("Weight (fixed)")
    axes[-1, 0].set_xlabel("Channel index")
    axes[-1, 1].set_xlabel("Channel index")
    fig.suptitle(
        f"{layer_name} — Complementarity (PTQ4DiT Fig.1-Left analog)"
    )
    _save(fig, f"fig1left_{layer_name}", output_dir)
    return fig


# ===================================================================
# 10.4  Per-layer salience histogram / KDE
# ===================================================================

def plot_salience_histogram(
    layer_name: str,
    act_salience: np.ndarray,
    wt_salience: np.ndarray,
    k: int = K,
    output_dir: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bins = np.linspace(0, max(act_salience.max(), wt_salience.max()) * 1.05, 80)

    ax.hist(act_salience, bins=bins, alpha=0.5, color="#3498db", label="Activation")
    ax.hist(wt_salience, bins=bins, alpha=0.5, color="#e67e22", label="Weight")

    for v in np.sort(act_salience)[-k:][:3]:
        ax.axvline(v, color="#e74c3c", ls="--", lw=0.7, alpha=0.6)
    for v in np.sort(wt_salience)[-k:][:3]:
        ax.axvline(v, color="#c0392b", ls=":", lw=0.7, alpha=0.6)

    ax.set_xlabel("Channel salience (max|·|)")
    ax.set_ylabel("Count")
    ax.set_title(f"{layer_name} — Salience distribution")
    ax.legend()
    _save(fig, f"hist_{layer_name}", output_dir)
    return fig


# ===================================================================
# 10.5  Heatmap: channel salience × sigma step
# ===================================================================

def plot_salience_heatmap(
    layer_name: str,
    trajectory: np.ndarray,
    sigma_values: np.ndarray,
    output_dir: Optional[Path] = None,
):
    mean_salience = trajectory.mean(axis=0)
    sort_idx = np.argsort(mean_salience)
    sorted_traj = trajectory[:, sort_idx]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    im = ax.imshow(
        np.log10(sorted_traj + 1e-12),
        aspect="auto",
        cmap="inferno",
        origin="upper",
    )
    ax.set_xlabel("Channel (sorted by mean salience →)")
    ax.set_ylabel("σ step (top=early)")
    yticks = np.linspace(0, len(sigma_values) - 1, min(10, len(sigma_values)), dtype=int)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{sigma_values[i]:.2f}" for i in yticks])
    fig.colorbar(im, ax=ax, label="log₁₀(salience)")
    ax.set_title(f"{layer_name} — Channel salience × σ step")
    _save(fig, f"heatmap_{layer_name}", output_dir)
    return fig


def plot_salience_heatmap_grid(
    all_trajectories: dict,
    sigma_values: np.ndarray,
    family: str = "q_proj",
    side: str = "image",
    output_dir: Optional[Path] = None,
):
    """4×6 small-multiples heatmap grid for all 24 blocks of one family."""
    fig, axes = plt.subplots(4, 6, figsize=(24, 14), constrained_layout=True)

    for bidx in range(24):
        r, c = divmod(bidx, 6)
        ax = axes[r, c]
        name = f"blocks.{bidx}.{side}.attn.{family}"
        if name in all_trajectories:
            traj = all_trajectories[name]
            mean_s = traj.mean(axis=0)
            sort_idx = np.argsort(mean_s)
            ax.imshow(
                np.log10(traj[:, sort_idx] + 1e-12),
                aspect="auto", cmap="inferno", origin="upper",
            )
        ax.set_title(f"Block {bidx}", fontsize=8)
        ax.set_xticks([])
        if c > 0:
            ax.set_yticks([])

    fig.suptitle(f"{side} {family} — Salience heatmaps across blocks")
    _save(fig, f"heatmap_grid_{side}_{family}", output_dir)
    return fig


# ===================================================================
# 10.6  Layerwise Spearman ρ bar plot
# ===================================================================

def plot_layerwise_rho(
    summary_rows: List[dict],
    output_dir: Optional[Path] = None,
):
    image_rows = [r for r in summary_rows if r["side"] == "image"]
    text_rows = [r for r in summary_rows if r["side"] == "text"]
    shared_rows = [r for r in summary_rows if r["side"] == "shared"]

    fig, (ax_img, ax_txt) = plt.subplots(
        2, 1, figsize=(20, 8), sharex=False, constrained_layout=True,
    )

    for ax, rows, label in [
        (ax_img, image_rows + shared_rows, "Image / Shared"),
        (ax_txt, text_rows, "Text"),
    ]:
        names = [r["layer_name"] for r in rows]
        rhos = [r["mean_spearman_rho"] for r in rows]
        colors = [FAMILY_COLORS.get(r["family"], "#95a5a6") for r in rows]

        ax.bar(range(len(names)), rhos, color=colors, width=0.8, edgecolor="none")
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(0.5, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("Mean Spearman ρ")
        ax.set_title(f"{label} layers")
        ax.set_xlim(-0.5, len(names) - 0.5)

    handles = [Patch(color=v, label=k) for k, v in FAMILY_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(FAMILY_COLORS), fontsize=8)
    fig.suptitle("Layerwise Spearman ρ(act salience, wt salience)")
    _save(fig, "layerwise_rho", output_dir)
    return fig


# ===================================================================
# 10.7  Temporal ρ trajectory + SSC weight overlay
# ===================================================================

def plot_rho_trajectory(
    layer_name: str,
    rho_traj: np.ndarray,
    sigma_values: np.ndarray,
    output_dir: Optional[Path] = None,
):
    eta = compute_ssc_weights(rho_traj)

    fig, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(rho_traj))

    ax1.plot(x, rho_traj, "o-", color="#2980b9", markersize=3, label="ρ")
    ax1.set_ylabel("Spearman ρ", color="#2980b9")
    ax1.set_xlabel("σ step")
    ax1.axhline(0, color="grey", lw=0.5, ls="--")

    ax2 = ax1.twinx()
    ax2.fill_between(x, eta, alpha=0.3, color="#e67e22", label="η (SSC weight)")
    ax2.set_ylabel("η", color="#e67e22")

    ax1.set_xticks(x[::max(1, len(x) // 10)])
    ax1.set_xticklabels(
        [f"{sigma_values[i]:.2f}" for i in x[::max(1, len(x) // 10)]],
        fontsize=7, rotation=45,
    )
    ax1.set_title(f"{layer_name} — ρ trajectory + SSC weights")
    _save(fig, f"rho_traj_{layer_name}", output_dir)
    return fig


def plot_rho_trajectory_grid(
    all_rho_trajs: dict,
    sigma_values: np.ndarray,
    family: str = "q_proj",
    side: str = "image",
    output_dir: Optional[Path] = None,
):
    """6×4 small-multiples grid of ρ trajectories for all 24 blocks."""
    fig, axes = plt.subplots(6, 4, figsize=(16, 18), constrained_layout=True)
    x = np.arange(len(sigma_values))

    for bidx in range(24):
        r, c = divmod(bidx, 4)
        ax = axes[r, c]
        name = f"blocks.{bidx}.{side}.attn.{family}"
        if name in all_rho_trajs:
            ax.plot(x, all_rho_trajs[name], color="#2980b9", lw=1)
            ax.axhline(0, color="grey", lw=0.3, ls="--")
        ax.set_title(f"Block {bidx}", fontsize=8)
        ax.set_ylim(-1, 1)
        if r < 5:
            ax.set_xticks([])

    fig.suptitle(f"{side} {family} — ρ trajectories across blocks")
    _save(fig, f"rho_grid_{side}_{family}", output_dir)
    return fig


# ===================================================================
# 10.8  Image vs text paired scatter
# ===================================================================

def plot_modality_scatter(
    summary_rows: List[dict],
    metric: str,
    metric_label: str,
    output_dir: Optional[Path] = None,
):
    img_data: dict = {}
    txt_data: dict = {}
    for r in summary_rows:
        key = (r["block"], r["family"])
        if r["side"] == "image":
            img_data[key] = r[metric]
        elif r["side"] == "text":
            txt_data[key] = r[metric]

    common = sorted(set(img_data) & set(txt_data))
    if not common:
        return None

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    blocks = np.array([k[0] for k in common])
    xs = np.array([img_data[k] for k in common])
    ys = np.array([txt_data[k] for k in common])

    sc = ax.scatter(xs, ys, c=blocks, cmap="coolwarm", s=30, edgecolors="k", lw=0.3)
    lims = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel(f"Image {metric_label}")
    ax.set_ylabel(f"Text {metric_label}")
    fig.colorbar(sc, ax=ax, label="Block index")
    ax.set_title(f"Image vs Text — {metric_label}")
    _save(fig, f"modality_scatter_{metric}", output_dir)
    return fig


# ===================================================================
# 10.9  Submodule family violin plot
# ===================================================================

def plot_family_violins(
    summary_rows: List[dict],
    output_dir: Optional[Path] = None,
):
    metrics = [
        ("max_act_salience", "Max activation salience"),
        ("mean_spearman_rho", "Spearman ρ"),
        ("cov_temporal", "Temporal CoV"),
    ]
    families = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for ax, (met, label) in zip(axes, metrics):
        data_img, data_txt, positions, tick_labels = [], [], [], []
        pos = 0
        for fam in families:
            vals_img = [r[met] for r in summary_rows if r["family"] == fam and r["side"] == "image"]
            vals_txt = [r[met] for r in summary_rows if r["family"] == fam and r["side"] == "text"]
            if vals_img:
                data_img.append(vals_img)
                positions.append(pos)
            if vals_txt:
                data_txt.append(vals_txt)
                positions.append(pos + 0.35) if vals_txt else None
            tick_labels.append(fam)
            pos += 1

        img_pos = [i for i in range(len(families)) if any(
            r["family"] == families[i] and r["side"] == "image" for r in summary_rows
        )]
        txt_pos = [i + 0.35 for i in range(len(families)) if any(
            r["family"] == families[i] and r["side"] == "text" for r in summary_rows
        )]

        if data_img:
            vp1 = ax.violinplot(data_img, positions=img_pos, widths=0.3, showmedians=True)
            for body in vp1["bodies"]:
                body.set_facecolor("#3498db")
                body.set_alpha(0.5)
        if data_txt:
            vp2 = ax.violinplot(data_txt, positions=txt_pos, widths=0.3, showmedians=True)
            for body in vp2["bodies"]:
                body.set_facecolor("#e67e22")
                body.set_alpha(0.5)

        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, fontsize=8)
        ax.set_title(label)

    handles = [Patch(color="#3498db", alpha=0.5, label="Image"),
               Patch(color="#e67e22", alpha=0.5, label="Text")]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.suptitle("Submodule family diagnostics")
    _save(fig, "family_violins", output_dir)
    return fig


# ===================================================================
# 10.10  Block depth vs salience profile
# ===================================================================

def plot_block_depth_profile(
    summary_rows: List[dict],
    family: str = "q_proj",
    output_dir: Optional[Path] = None,
):
    block_range = range(24)
    fig, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)

    for side, color, ls_mean, ls_max in [
        ("image", "#3498db", "-", "--"),
        ("text", "#e67e22", "-", "--"),
    ]:
        rows = {r["block"]: r for r in summary_rows
                if r["family"] == family and r["side"] == side}
        blocks = sorted(rows.keys())
        if not blocks:
            continue
        mean_s = [rows[b]["mean_act_salience"] for b in blocks]
        max_s = [rows[b]["max_act_salience"] for b in blocks]
        ax1.plot(blocks, mean_s, ls_mean, color=color, label=f"{side} mean")
        ax1.plot(blocks, max_s, ls_max, color=color, label=f"{side} max", alpha=0.6)

    ax1.set_xlabel("Block index")
    ax1.set_ylabel("Activation salience")
    ax1.legend(fontsize=8)

    ax2 = ax1.twinx()
    for side, color in [("image", "#3498db"), ("text", "#e67e22")]:
        rows = {r["block"]: r for r in summary_rows
                if r["family"] == family and r["side"] == side}
        blocks = sorted(rows.keys())
        if not blocks:
            continue
        rhos = [rows[b]["mean_spearman_rho"] for b in blocks]
        ax2.plot(blocks, rhos, ":", color=color, alpha=0.5, label=f"{side} ρ")

    ax2.set_ylabel("Spearman ρ", alpha=0.5)
    ax1.set_title(f"Block depth profile — {family}")
    _save(fig, f"depth_profile_{family}", output_dir)
    return fig


# ===================================================================
# 10.11  Top-k overlap heatmap across sigma steps
# ===================================================================

def plot_topk_overlap_heatmap(
    layer_name: str,
    trajectory: np.ndarray,
    sigma_values: np.ndarray,
    k: int = K,
    output_dir: Optional[Path] = None,
):
    mat = pairwise_topk_jaccard(trajectory, k)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
    n = len(sigma_values)
    ticks = np.linspace(0, n - 1, min(8, n), dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{sigma_values[i]:.2f}" for i in ticks], fontsize=7, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{sigma_values[i]:.2f}" for i in ticks], fontsize=7)
    fig.colorbar(im, ax=ax, label="Jaccard")
    ax.set_title(f"{layer_name} — Top-{k} overlap across σ steps")
    _save(fig, f"topk_overlap_{layer_name}", output_dir)
    return fig


# ===================================================================
# 10.12  Activation vs weight salience scatter
# ===================================================================

def plot_act_wt_scatter(
    layer_name: str,
    act_salience: np.ndarray,
    wt_salience: np.ndarray,
    k: int = K,
    output_dir: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    d_in = len(act_salience)

    sc = ax.scatter(
        act_salience, wt_salience,
        c=np.arange(d_in), cmap="viridis", s=10, alpha=0.6,
    )
    top_k_act = np.argsort(act_salience)[-k:]
    top_k_wt = np.argsort(wt_salience)[-k:]
    ax.scatter(act_salience[top_k_act], wt_salience[top_k_act],
               facecolors="none", edgecolors="#e74c3c", s=40, lw=1.2, label="Top-k act")
    ax.scatter(act_salience[top_k_wt], wt_salience[top_k_wt],
               facecolors="none", edgecolors="#e67e22", s=40, lw=1.2, label="Top-k wt")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Activation salience s(X_j)")
    ax.set_ylabel("Weight salience s(W_j)")

    rho, _ = spearmanr(act_salience, wt_salience)
    ax.annotate(
        f"ρ = {rho:.3f}",
        xy=(0.05, 0.95), xycoords="axes fraction",
        fontsize=10, bbox=dict(boxstyle="round", fc="wheat"),
    )
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="Channel index")
    ax.set_title(f"{layer_name} — Act vs Wt salience")
    _save(fig, f"scatter_{layer_name}", output_dir)
    return fig


# ===================================================================
# 10.13  Salience rank stability ribbon
# ===================================================================

def plot_rank_stability_ribbon(
    layer_name: str,
    trajectory: np.ndarray,
    sigma_values: np.ndarray,
    n_track: int = 16,
    output_dir: Optional[Path] = None,
):
    num_steps, d_in = trajectory.shape
    tracked = np.argsort(trajectory[0])[-n_track:]

    ranks = np.zeros((num_steps, n_track))
    for t in range(num_steps):
        order = np.argsort(-trajectory[t])
        rank_map = np.empty(d_in, dtype=int)
        rank_map[order] = np.arange(d_in)
        ranks[t] = rank_map[tracked]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    cmap = plt.cm.get_cmap("tab20", n_track)
    for i in range(n_track):
        ax.plot(range(num_steps), ranks[:, i] + 1, color=cmap(i), lw=1.2, alpha=0.8)

    ax.invert_yaxis()
    ax.set_ylabel("Rank (1 = most salient)")
    ax.set_xlabel("σ step")
    xt = np.linspace(0, num_steps - 1, min(10, num_steps), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{sigma_values[i]:.2f}" for i in xt], fontsize=7, rotation=45)
    ax.set_title(f"{layer_name} — Top-{n_track} rank stability")
    _save(fig, f"rank_ribbon_{layer_name}", output_dir)
    return fig


# ===================================================================
# 10.14  Quantization sensitivity ranking
# ===================================================================

def plot_risk_ranking(
    summary_rows: List[dict],
    top_n: int = 40,
    output_dir: Optional[Path] = None,
):
    sorted_rows = sorted(summary_rows, key=lambda r: r["risk_score"], reverse=True)
    top = sorted_rows[:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)), constrained_layout=True)
    names = [r["layer_name"] for r in top][::-1]
    scores = [r["risk_score"] for r in top][::-1]
    colors = [FAMILY_COLORS.get(r["family"], "#95a5a6") for r in top][::-1]

    ax.barh(range(len(names)), scores, color=colors, height=0.7, edgecolor="none")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Risk score")
    ax.set_title(f"Top-{top_n} quantization-sensitive layers")

    handles = [Patch(color=v, label=k) for k, v in FAMILY_COLORS.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    _save(fig, "risk_ranking", output_dir)
    return fig


# ===================================================================
# 10.15  CFG conditioning regime comparison (placeholder)
# ===================================================================

def plot_cfg_comparison(
    layer_name: str,
    cond_salience: np.ndarray,
    uncond_salience: np.ndarray,
    sigma_values: np.ndarray,
    step_indices: Sequence[int],
    output_dir: Optional[Path] = None,
):
    """Only produced if CFG data is collected (cfg_weight > 1)."""
    fig, (ax_box, ax_sc) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True,
    )

    positions = np.arange(len(step_indices)) * 3
    bp1 = ax_box.boxplot(
        [cond_salience[si] for si in step_indices],
        positions=positions, widths=0.8, patch_artist=True, showfliers=False,
    )
    bp2 = ax_box.boxplot(
        [uncond_salience[si] for si in step_indices],
        positions=positions + 1, widths=0.8, patch_artist=True, showfliers=False,
    )
    for patch in bp1["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.5)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#e67e22")
        patch.set_alpha(0.5)
    ax_box.set_xticks(positions + 0.5)
    ax_box.set_xticklabels([f"σ={sigma_values[si]:.2f}" for si in step_indices])
    ax_box.set_title("Conditioned vs Unconditioned salience")
    ax_box.legend(
        [Patch(fc="#3498db", alpha=0.5), Patch(fc="#e67e22", alpha=0.5)],
        ["Conditioned", "Unconditioned"],
    )

    mid = len(step_indices) // 2
    si = step_indices[mid]
    ax_sc.scatter(cond_salience[si], uncond_salience[si], s=8, alpha=0.5)
    lims = [0, max(cond_salience[si].max(), uncond_salience[si].max()) * 1.05]
    ax_sc.plot(lims, lims, "k--", lw=0.5)
    ax_sc.set_xlabel("Conditioned s(X_j)")
    ax_sc.set_ylabel("Unconditioned s(X_j)")
    ax_sc.set_title(f"σ={sigma_values[si]:.2f} ratio scatter")

    fig.suptitle(f"{layer_name} — CFG comparison")
    _save(fig, f"cfg_{layer_name}", output_dir)
    return fig


# ===================================================================
# 10.16  Final layer analysis
# ===================================================================

def plot_final_layer_analysis(
    trajectory: np.ndarray,
    sigma_values: np.ndarray,
    mean_trajectory: np.ndarray,
    wt_salience: np.ndarray,
    midblock_trajectory: Optional[np.ndarray] = None,
    midblock_name: str = "",
    output_dir: Optional[Path] = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # 1. Output channel magnitude bar
    ax = axes[0]
    ch = np.arange(len(wt_salience))
    ax.bar(ch, wt_salience, color="#2c3e50", width=1.0, edgecolor="none")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("max|weight|")
    ax.set_title("final_layer.linear — Weight channel profile")

    # 2. Temporal boxplot (reuse fig4 logic)
    ax = axes[1]
    num_steps = trajectory.shape[0]
    ax.boxplot(
        [trajectory[t] for t in range(num_steps)],
        positions=range(num_steps),
        widths=0.6, patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="#2c3e50", alpha=0.4),
        medianprops=dict(color="#e74c3c", lw=1.5),
    )
    xt = np.linspace(0, num_steps - 1, min(8, num_steps), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{sigma_values[i]:.2f}" for i in xt], fontsize=7, rotation=45)
    ax.set_xlabel("σ")
    ax.set_ylabel("max|activation| per channel")
    ax.set_title("final_layer.linear — Temporal variation")

    # 3. Activation distribution at 3 σ
    ax = axes[2]
    steps = [0, num_steps // 2, num_steps - 1]
    for si in steps:
        vals = mean_trajectory[si]
        ax.hist(vals, bins=50, alpha=0.4, label=f"σ={sigma_values[si]:.2f}")
    ax.set_xlabel("Mean |activation| per channel")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.set_title("final_layer.linear — Distribution shift")

    fig.suptitle("Final layer analysis (velocity prediction)")
    _save(fig, "final_layer_analysis", output_dir)
    return fig


# ===================================================================
# 10.17  Global summary dashboard
# ===================================================================

def plot_summary_dashboard(
    summary_rows: List[dict],
    output_dir: Optional[Path] = None,
):
    families = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # Row 0: block-depth profiles (mean act salience, mean ρ, mean CoV)
    depth_metrics = [
        ("mean_act_salience", "Mean act. salience"),
        ("mean_spearman_rho", "Mean ρ"),
        ("cov_temporal", "Mean CoV"),
    ]
    for col, (met, label) in enumerate(depth_metrics):
        ax = axes[0, col]
        for side, color in [("image", "#3498db"), ("text", "#e67e22")]:
            rows = [r for r in summary_rows
                    if r["side"] == side and r["family"] in families]
            if not rows:
                continue
            by_block: dict = {}
            for r in rows:
                by_block.setdefault(r["block"], []).append(r[met])
            blocks = sorted(by_block)
            means = [np.mean(by_block[b]) for b in blocks]
            ax.plot(blocks, means, "o-", color=color, markersize=3, label=side)
        ax.set_xlabel("Block")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)

    # Row 1: family violins (salience, ρ) + risk histogram
    for col, (met, label) in enumerate([
        ("max_act_salience", "Max act. salience"),
        ("mean_spearman_rho", "Spearman ρ"),
    ]):
        ax = axes[1, col]
        data, pos = [], []
        for i, fam in enumerate(families):
            vals = [r[met] for r in summary_rows if r["family"] == fam]
            if vals:
                data.append(vals)
                pos.append(i)
        if data:
            vp = ax.violinplot(data, positions=pos, widths=0.6, showmedians=True)
            for body in vp["bodies"]:
                body.set_alpha(0.5)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, fontsize=8)
        ax.set_title(label)

    ax = axes[1, 2]
    scores = [r["risk_score"] for r in summary_rows]
    ax.hist(scores, bins=30, color="#2c3e50", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Count")
    ax.set_title("Risk score distribution")

    fig.suptitle("Phase 1 Diagnostic Summary Dashboard", fontsize=14, fontweight="bold")
    _save(fig, "summary_dashboard", output_dir)
    return fig
