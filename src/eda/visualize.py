"""
Phase 1 EDA: Generate C1–C6 plots for Q-Diffusion analysis.

Each plot function reads from pre-loaded stats dicts and writes a PNG/GIF to
eda_output/plots/. All functions are callable independently.

Plot suite:
  C1 — Animated histogram (18 bins, noise→denoised) with per-channel markers
  C2 — Per-channel range boxplot at a single (layer, timestep)
  C3 — Channel mean activation distribution across timesteps (noise→denoised)
  C4 — Q/K/V channel activation: img vs txt across timesteps (3-row × 1-col)
  C5 — Q/K/V channel activation: img vs txt across layers (3-row × 1-col)
  C6 — Txt/img ratio heatmap (max and abs_min) across layers × timesteps

CLI:
  python -m src.eda.visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ACT = str(_ROOT / "eda_output" / "activation_stats_full.npz")
_DEFAULT_WGT = str(_ROOT / "eda_output" / "weight_stats.npz")
_DEFAULT_TABLES = str(_ROOT / "eda_output" / "tables")
_DEFAULT_PLOTS = str(_ROOT / "eda_output" / "plots")

N_BLOCKS = 24
STREAMS = ("img", "txt")
QKV_PROJS = ("q_proj", "k_proj", "v_proj")
TVC_THRESHOLD = 0.2
TXT_BASELINE = 77 / (77 + 1024)

# Consistent channel subset for all C-series per-channel plots
CHANNEL_SUBSET = list(range(768, 818))  # channels 768–817 (middle 50 of 1536)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _sorted_layer_ids(layer_map: Dict) -> List[str]:
    def key(lid):
        parts = lid.split("_")
        try:
            blk = int(parts[1])
        except (IndexError, ValueError):
            blk = 99
        stream_order = {"img": 0, "txt": 1, "joint": 2}
        st = stream_order.get(parts[2] if len(parts) > 2 else "", 3)
        return (blk, st)
    return sorted(layer_map.keys(), key=key)


def _sorted_timesteps(t_map: Dict) -> List[str]:
    return sorted(t_map.keys(), key=lambda k: float(k))


def _channel_range(stats: Dict) -> np.ndarray:
    """Per-channel activation range: max - min across all (B, S) positions."""
    return stats["max"] - stats["min"]


def _channel_range_pcts(stats: Dict, pcts=(0, 10, 25, 50, 75, 90, 100)) -> np.ndarray:
    """Percentiles of the per-channel range distribution."""
    return np.percentile(_channel_range(stats), pcts)


def _iqr(arr: np.ndarray) -> float:
    """Interquartile range of an array."""
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _load_csv(path: str) -> List[Dict]:
    import csv
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows


def _ensure_plots_dir(plots_dir: str) -> Path:
    p = Path(plots_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _make_boxplot_stats(arr: np.ndarray) -> dict:
    """Pre-compute boxplot statistics from a 1-D array (for manual box rendering)."""
    q0, q25, q50, q75, q100 = np.percentile(arr, [0, 25, 50, 75, 100])
    iqr = q75 - q25
    lo_whisk = max(q0, q25 - 1.5 * iqr)
    hi_whisk = min(q100, q75 + 1.5 * iqr)
    return dict(med=q50, q1=q25, q3=q75, whislo=lo_whisk, whishi=hi_whisk, fliers=[])


def _channel_iqr_prompts(stats: Dict) -> np.ndarray:
    """Per-channel IQR across calibration prompts: p75[c] - p25[c]."""
    return stats["p75"] - stats["p25"]


def _channel_metric(stats: Dict, metric: str) -> np.ndarray:
    """Extract per-channel metric array from a stats dict."""
    if metric == "range":
        return stats["max"] - stats["min"]
    elif metric == "abs_max":
        return np.maximum(np.abs(stats["min"]), np.abs(stats["max"]))
    elif metric == "max":
        return stats["max"].copy()
    elif metric == "min":
        return stats["min"].copy()
    raise ValueError(f"Unknown metric: {metric}")


def _bxp_stats_from_channel(stats: Dict, c: int) -> dict:
    """Pre-compute bxp() stats dict for a single channel using stored percentiles."""
    return {
        "med": float(stats["mean"][c]),
        "q1": float(stats["p25"][c]),
        "q3": float(stats["p75"][c]),
        "whislo": float(stats["min"][c]),
        "whishi": float(stats["max"][c]),
        "fliers": [],
    }


# ---------------------------------------------------------------------------
# C1 — Animated histogram with per-channel markers across timesteps
# ---------------------------------------------------------------------------

def plot_C1_channel_histogram_animated(
    act_stats: Dict,
    layer_id: str,
    family: str,
    channel_idx: int,
    plots_dir: str,
    fps: int = 2,
) -> str:
    """
    Animated GIF: true per-channel activation histogram (18 display bins) for a
    single channel across timesteps, ordered noise → denoised (high t first).

    Bars show the actual activation value distribution for channel_idx only
    (not the global all-channel histogram). The stored 512-bin per-channel
    histogram is rebinned to 18 bins for display.

    channel_idx MUST be one of CHANNEL_HISTOGRAM_IDS (20 channels recorded during
    profiling). The available IDs are printed on error and listed in eda_tracer.py
    as CHANNEL_HISTOGRAM_IDS.

    Output: C1_channel_hist_animated_{layer_id}_{family}_ch{channel_idx}.gif
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from .eda_tracer import HISTOGRAM_NUM_BINS, HISTOGRAM_RANGE, CHANNEL_HISTOGRAM_IDS

    DISPLAY_BINS = 18

    out = _ensure_plots_dir(plots_dir)
    t_map = act_stats.get(family, {}).get(layer_id, {})

    if not t_map:
        print(f"  No data for {layer_id}/{family}")
        return ""

    # Validate channel_idx
    ch_key = f"ch_hist_{channel_idx}"
    first_stats = next(iter(t_map.values()))
    if ch_key not in first_stats:
        print(f"  channel_idx={channel_idx} has no stored histogram.")
        print(f"  Available CHANNEL_HISTOGRAM_IDS: {list(CHANNEL_HISTOGRAM_IDS)}")
        return ""

    # Noise → denoised: high t first
    t_keys = list(reversed(_sorted_timesteps(t_map)))
    t_vals = [float(k) for k in t_keys]
    n_ts = len(t_keys)

    # Original 512-bin edges/centers (for rebinning)
    orig_edges = np.linspace(HISTOGRAM_RANGE[0], HISTOGRAM_RANGE[1], HISTOGRAM_NUM_BINS + 1)
    orig_centers = 0.5 * (orig_edges[:-1] + orig_edges[1:])

    # Display 18-bin edges
    disp_edges = np.linspace(HISTOGRAM_RANGE[0], HISTOGRAM_RANGE[1], DISPLAY_BINS + 1)
    disp_centers = 0.5 * (disp_edges[:-1] + disp_edges[1:])
    disp_width = disp_edges[1] - disp_edges[0]

    def _rebin(hist):
        rebinned, _ = np.histogram(orig_centers, bins=disp_edges, weights=hist)
        return rebinned

    all_rebinned = [_rebin(t_map[tk][ch_key]) for tk in t_keys
                    if ch_key in t_map[tk]]
    hist_ymax = max(float(r.max()) for r in all_rebinned) * 2.0 if all_rebinned else 1.0

    cmap_ts = plt.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.82)

    def update(frame_idx: int):
        ax.cla()
        tk = t_keys[frame_idx]
        t_val = t_vals[frame_idx]
        stats = t_map[tk]
        color = cmap_ts(frame_idx / max(n_ts - 1, 1))

        ch_hist = stats.get(ch_key)
        if ch_hist is not None:
            rebinned = _rebin(ch_hist)
            ax.bar(disp_centers, rebinned, width=disp_width, color=color, alpha=0.85,
                   linewidth=0.5, edgecolor="white")
            ax.set_yscale("log")
            ax.set_ylim(1, hist_ymax)
            ax.set_xlim(HISTOGRAM_RANGE)

        ax.set_xlabel("Activation value")
        ax.set_ylabel("Count (log)")
        fig.suptitle(
            f"Layer {layer_id} · {family} · channel {channel_idx} · t={t_val:.0f}",
            fontsize=12,
        )

    anim = FuncAnimation(fig, update, frames=n_ts, interval=int(1000 / fps))
    fname = f"C1_channel_hist_animated_{layer_id}_{family}_ch{channel_idx}.gif"
    gif_path = str(out / fname)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Saved {fname}  ({n_ts} frames @ {fps} fps)")
    return gif_path


# ---------------------------------------------------------------------------
# C2 — Per-channel range boxplot at a single (layer, timestep)
# ---------------------------------------------------------------------------

def plot_C2_channel_range_boxplot(
    act_stats: Dict,
    layer_id: str,
    family: str,
    t_key: str,
    plots_dir: str,
    channel_subset: Optional[List[int]] = None,
) -> None:
    """
    For a single (layer, family, timestep), show per-channel activation range as a
    custom boxplot using stored min/p25/mean/p75/max per channel.

    Each box whiskers = [min[c], max[c]], box = [p25[c], p75[c]], median = mean[c].

    Output: C2_channel_range_boxplot_{layer_id}_{family}_t{t:.0f}.png
    """
    import matplotlib.pyplot as plt

    if channel_subset is None:
        channel_subset = CHANNEL_SUBSET

    out = _ensure_plots_dir(plots_dir)
    t_map = act_stats.get(family, {}).get(layer_id, {})

    if t_key not in t_map:
        available = list(t_map.keys())[:5]
        print(f"  t_key '{t_key}' not found in {layer_id}/{family}. Available: {available}...")
        return

    stats = t_map[t_key]
    t_val = float(t_key)

    bxp_stats = [_bxp_stats_from_channel(stats, c) for c in channel_subset]

    fig, ax = plt.subplots(figsize=(max(12, len(channel_subset) * 0.35), 5))
    positions = list(range(len(channel_subset)))
    ax.bxp(bxp_stats, positions=positions, widths=0.6, patch_artist=True,
           flierprops=dict(marker=".", markersize=2, alpha=0.3),
           medianprops=dict(color="black", linewidth=1.5),
           boxprops=dict(facecolor="steelblue", alpha=0.7))

    step = max(1, len(channel_subset) // 10)
    tick_idx = list(range(0, len(channel_subset), step))
    ax.set_xticks([positions[i] for i in tick_idx])
    ax.set_xticklabels([str(channel_subset[i]) for i in tick_idx],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Activation value")
    ax.set_title(f"{layer_id} · {family} · t={t_val:.0f}")

    fname = f"C2_channel_range_boxplot_{layer_id}_{family}_t{t_val:.0f}.png"
    plt.tight_layout()
    plt.savefig(str(out / fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# C3 — Channel activation distribution across timesteps (noise → denoised)
# ---------------------------------------------------------------------------

def plot_C3_layer_range_across_time(
    act_stats: Dict,
    layer_id: str,
    family: str,
    plots_dir: str,
) -> None:
    """
    Distribution of per-channel mean activations across all profiled timesteps,
    ordered noise → denoised (high t first on x-axis).

    Each box = distribution of abs_max[c] = max(|min[c]|, max[c]) across all
    D=1536 channels at one timestep. This is the per-channel dynamic range that
    a quantizer's clipping scale must cover.
    Plasma colormap marks temporal position.

    Output: C3_layer_range_across_time_{layer_id}_{family}.png
    """
    import matplotlib.pyplot as plt

    out = _ensure_plots_dir(plots_dir)
    t_map = act_stats.get(family, {}).get(layer_id, {})

    if not t_map:
        print(f"  No data for {layer_id}/{family}")
        return

    # Noise → denoised: high t first
    t_keys = list(reversed(_sorted_timesteps(t_map)))
    t_vals = [float(k) for k in t_keys]
    box_data = [
        np.maximum(np.abs(t_map[tk]["min"]), t_map[tk]["max"])
        for tk in t_keys
    ]

    fig, ax = plt.subplots(figsize=(max(12, len(t_keys) * 0.8), 5))
    cmap = plt.get_cmap("plasma")

    bp = ax.boxplot(box_data, positions=range(len(t_keys)), patch_artist=True, widths=0.5,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    medianprops=dict(color="black", linewidth=1.5))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / max(len(t_keys) - 1, 1)))
        patch.set_alpha(0.6)

    ax.set_xticks(range(len(t_keys)))
    ax.set_xticklabels([f"{v:.0f}" for v in t_vals], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Timestep (noise → denoised)")
    ax.set_ylabel("Channel abs_max  (max(|min|, max))")
    ax.set_title(f"{layer_id} · {family} — per-channel dynamic range across timesteps")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(len(t_keys) - 1, 1)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Frame index (0 = most noisy)")

    fname = f"C3_layer_range_across_time_{layer_id}_{family}.png"
    plt.tight_layout()
    plt.savefig(str(out / fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# C4 — Q/K/V channel activation by stream across timesteps (for one block)
# ---------------------------------------------------------------------------

def plot_C4_qkv_range_by_stream_across_time(
    act_stats: Dict,
    layer_id_base: str,
    plots_dir: str,
    projections: Optional[List[str]] = None,
) -> None:
    """
    For a specific block, show the channel mean activation distribution for img
    and txt streams across timesteps, ordered noise → denoised.

    Layout: N_proj rows × 1 col (one row per projection).
    X-axis: timestep (noise → denoised, high t first).
    Y-axis: channel mean activation (distribution over D=1536 channels).
    Grouping: img (steelblue) vs txt (tomato) as paired boxes at each timestep.

    Args:
        layer_id_base: e.g. "mm_04" — appends "_img" / "_txt"
        projections:   subset of QKV_PROJS to show, e.g. ["q_proj"]. None = all 3.

    Output: C4_channel_activation_stream_across_time_{layer_id_base}.png
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if projections is None:
        projections = list(QKV_PROJS)

    out = _ensure_plots_dir(plots_dir)

    # Collect all available timestep keys (noise → denoised: high t first)
    all_t_keys: set = set()
    for proj in projections:
        for stream in STREAMS:
            lid = f"{layer_id_base}_{stream}"
            all_t_keys.update(act_stats.get(proj, {}).get(lid, {}).keys())

    if not all_t_keys:
        print(f"  No data for {layer_id_base}")
        return

    t_keys = list(reversed(sorted(all_t_keys, key=lambda k: float(k))))
    t_vals = [float(k) for k in t_keys]
    n_ts = len(t_keys)
    n_rows = len(projections)

    STREAM_COLORS = {"img": "steelblue", "txt": "tomato"}
    WIDTH = 0.35

    fig, axes = plt.subplots(n_rows, 1, figsize=(max(n_ts * 1.2, 14), n_rows * 4),
                              squeeze=False)

    for row_idx, proj in enumerate(projections):
        ax = axes[row_idx, 0]

        for s_idx, stream in enumerate(["img", "txt"]):
            lid = f"{layer_id_base}_{stream}"
            t_map = act_stats.get(proj, {}).get(lid, {})
            data_list = []
            pos_list = []
            for t_idx, tk in enumerate(t_keys):
                if tk in t_map:
                    data_list.append(t_map[tk]["mean"])
                    pos_list.append(t_idx + (s_idx - 0.5) * WIDTH)

            if data_list:
                bp = ax.boxplot(data_list, positions=pos_list, widths=WIDTH * 0.8,
                                patch_artist=True,
                                flierprops=dict(marker=".", markersize=1, alpha=0.2),
                                medianprops=dict(color="black", linewidth=1.2))
                for patch in bp["boxes"]:
                    patch.set_facecolor(STREAM_COLORS[stream])
                    patch.set_alpha(0.7)

        ax.set_xticks(range(n_ts))
        ax.set_xticklabels([f"{v:.0f}" for v in t_vals], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Channel mean activation")
        ax.set_title(f"{proj}")

        legend_handles = [Patch(facecolor=STREAM_COLORS[s], alpha=0.7, label=s)
                          for s in ["img", "txt"]]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Timestep (noise → denoised)")

    blk_str = layer_id_base.split("_")[1] if "_" in layer_id_base else layer_id_base
    fig.suptitle(f"mm_{blk_str} — Channel activation: img vs txt across timesteps", fontsize=13)
    plt.tight_layout()

    fname = f"C4_channel_activation_stream_across_time_{layer_id_base}.png"
    plt.savefig(str(out / fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# C5 — Q/K/V channel activation by stream across layers (at one timestep)
# ---------------------------------------------------------------------------

def plot_C5_qkv_range_by_stream_across_layers(
    act_stats: Dict,
    t_key: str,
    plots_dir: str,
    projections: Optional[List[str]] = None,
) -> None:
    """
    At a single timestep, show the channel mean activation distribution for img
    and txt streams across all profiled blocks.

    Layout: N_proj rows × 1 col (one row per projection).
    X-axis: block depth (shallow → deep).
    Y-axis: channel mean activation (distribution over D=1536 channels).
    Grouping: img (steelblue) vs txt (tomato) as paired boxes at each block.

    Args:
        t_key:       Timestep key string (e.g. "999.000000").
        projections: subset of QKV_PROJS to show. None = all 3.

    Output: C5_channel_activation_stream_across_layers_t{t:.0f}.png
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if projections is None:
        projections = list(QKV_PROJS)

    out = _ensure_plots_dir(plots_dir)

    # Derive profiled blocks from data
    available_blks = sorted(set(
        int(lid.split("_")[1])
        for proj in projections
        for lid in act_stats.get(proj, {})
        if lid.startswith("mm_")
    ))

    if not available_blks:
        print("  No profiled blocks found in act_stats")
        return

    t_val = float(t_key)
    n_blks = len(available_blks)
    n_rows = len(projections)

    STREAM_COLORS = {"img": "steelblue", "txt": "tomato"}
    WIDTH = 0.35

    fig, axes = plt.subplots(n_rows, 1, figsize=(max(n_blks * 1.2, 14), n_rows * 4),
                              squeeze=False)

    for row_idx, proj in enumerate(projections):
        ax = axes[row_idx, 0]

        for s_idx, stream in enumerate(["img", "txt"]):
            data_list = []
            pos_list = []
            for b_idx, blk in enumerate(available_blks):
                lid = f"mm_{blk:02d}_{stream}"
                t_map = act_stats.get(proj, {}).get(lid, {})
                if t_key in t_map:
                    data_list.append(t_map[t_key]["mean"])
                    pos_list.append(b_idx + (s_idx - 0.5) * WIDTH)

            if data_list:
                bp = ax.boxplot(data_list, positions=pos_list, widths=WIDTH * 0.8,
                                patch_artist=True,
                                flierprops=dict(marker=".", markersize=1, alpha=0.2),
                                medianprops=dict(color="black", linewidth=1.2))
                for patch in bp["boxes"]:
                    patch.set_facecolor(STREAM_COLORS[stream])
                    patch.set_alpha(0.7)

        ax.set_xticks(range(n_blks))
        ax.set_xticklabels([f"mm_{b:02d}" for b in available_blks],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Channel mean activation")
        ax.set_title(f"{proj}")

        legend_handles = [Patch(facecolor=STREAM_COLORS[s], alpha=0.7, label=s)
                          for s in ["img", "txt"]]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Block (shallow → deep)")

    fig.suptitle(f"Channel activation: img vs txt across layers · t={t_val:.0f}", fontsize=13)
    plt.tight_layout()

    fname = f"C5_channel_activation_stream_across_layers_t{t_val:.0f}.png"
    plt.savefig(str(out / fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# C6 — Txt/img activation ratio heatmap
# ---------------------------------------------------------------------------

def plot_C6_txt_img_ratio(
    act_stats: Dict,
    family: str,
    plots_dir: str,
) -> None:
    """
    Per-(layer, timestep) heatmap of txt-to-img stream ratio for average max
    and average abs_min values. Shows systematic stream imbalance → informs
    whether per-stream quantization ranges are needed for joint SDPA.

    Layout: 2 panels side-by-side
      Panel 0: mean(max[c]) for txt / mean(max[c]) for img
      Panel 1: mean(|min[c]|) for txt / mean(|min[c]|) for img
    Colormap: RdBu_r, centered at 1.0.

    Output: C6_txt_img_ratio_{family}.png
    """
    import matplotlib.pyplot as plt

    out = _ensure_plots_dir(plots_dir)
    layer_map = act_stats.get(family, {})

    # Derive profiled blocks and timesteps from data
    available_blks = sorted(set(
        int(lid.split("_")[1])
        for lid in layer_map
        if lid.startswith("mm_")
    ))
    all_t_keys: set = set()
    for t_map in layer_map.values():
        all_t_keys.update(t_map.keys())
    # Noise → denoised: high t first
    t_keys = list(reversed(sorted(all_t_keys, key=lambda k: float(k))))
    t_vals = [float(k) for k in t_keys]

    if not available_blks or not t_keys:
        print(f"  No data for family={family}")
        return

    n_blks = len(available_blks)
    n_ts = len(t_keys)

    max_ratio = np.full((n_blks, n_ts), np.nan)
    min_ratio = np.full((n_blks, n_ts), np.nan)

    for i, blk in enumerate(available_blks):
        img_id = f"mm_{blk:02d}_img"
        txt_id = f"mm_{blk:02d}_txt"
        img_map = layer_map.get(img_id, {})
        txt_map = layer_map.get(txt_id, {})

        for j, tk in enumerate(t_keys):
            img_s = img_map.get(tk)
            txt_s = txt_map.get(tk)
            if img_s is None or txt_s is None:
                continue

            img_max_mean = float(np.mean(img_s["max"]))
            txt_max_mean = float(np.mean(txt_s["max"]))
            if img_max_mean > 1e-8:
                max_ratio[i, j] = txt_max_mean / img_max_mean

            img_absmin_mean = float(np.mean(np.abs(img_s["min"])))
            txt_absmin_mean = float(np.mean(np.abs(txt_s["min"])))
            if img_absmin_mean > 1e-8:
                min_ratio[i, j] = txt_absmin_mean / img_absmin_mean

    # Symmetric colorscale centered at 1.0
    all_vals = np.concatenate([
        max_ratio[~np.isnan(max_ratio)],
        min_ratio[~np.isnan(min_ratio)],
    ])
    if len(all_vals) == 0:
        print(f"  No valid ratio data for family={family}")
        return
    spread = max(abs(float(all_vals.max()) - 1.0), abs(float(all_vals.min()) - 1.0))
    spread = max(spread, 0.1)  # minimum spread for colorscale
    vmin, vmax = 1.0 - spread, 1.0 + spread

    fig, axes = plt.subplots(1, 2, figsize=(max(n_ts * 0.8, 10), max(n_blks * 0.7, 5)))

    for ax_idx, (data, title) in enumerate([
        (max_ratio, "mean(max[c]):  txt / img"),
        (min_ratio, "mean(|min[c]|):  txt / img"),
    ]):
        ax = axes[ax_idx]
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       interpolation="nearest")

        for i in range(n_blks):
            for j in range(n_ts):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black")

        ax.set_xticks(range(n_ts))
        ax.set_xticklabels([f"{v:.0f}" for v in t_vals], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_blks))
        ax.set_yticklabels([f"mm_{b:02d}" for b in available_blks], fontsize=7)
        ax.set_xlabel("Timestep (noise → denoised)")
        if ax_idx == 0:
            ax.set_ylabel("Block")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="txt / img ratio")

    fig.suptitle(f"C6: Txt/Img Stream Ratio — {family}", fontsize=13)
    plt.tight_layout()

    fname = f"C6_txt_img_ratio_{family}.png"
    plt.savefig(str(out / fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Run all plots
# ---------------------------------------------------------------------------

def run_all_plots(
    act_stats_path: str,
    weight_stats_path: str,
    tables_dir: str,
    plots_dir: str,
    cali_path: Optional[str] = None,
) -> None:
    from .eda_tracer import load_tracer_stats
    from .weight_profiler import load_weight_stats

    print(f"Loading activation stats from {act_stats_path}...")
    act_stats = load_tracer_stats(act_stats_path)

    print(f"Loading weight stats from {weight_stats_path}...")
    load_weight_stats(weight_stats_path)  # loaded for potential future use

    print("\nGenerating C-series plots...")

    # Determine demo layer, timestep, and block base from data
    sample_family = "q_proj"
    layer_map = act_stats.get(sample_family, {})
    available_blks: List[int] = []
    demo_layer = "mm_00_img"
    demo_t_key: Optional[str] = None
    demo_blk_base = "mm_03"

    if layer_map:
        all_layer_ids = _sorted_layer_ids(layer_map)
        img_layers = [lid for lid in all_layer_ids if lid.endswith("_img")]
        demo_layer = img_layers[len(img_layers) // 2] if img_layers else all_layer_ids[0]
        demo_t_map = layer_map.get(demo_layer, {})
        if demo_t_map:
            all_t_keys = _sorted_timesteps(demo_t_map)
            demo_t_key = all_t_keys[len(all_t_keys) // 2]
        available_blks = sorted(set(
            int(lid.split("_")[1])
            for lid in layer_map
            if lid.startswith("mm_")
        ))
        if available_blks:
            mid_blk = available_blks[len(available_blks) // 2]
            demo_blk_base = f"mm_{mid_blk:02d}"

    # C3 — channel ranges across time for every profiled img layer (q_proj + post_gelu)
    for family in ("q_proj", "post_gelu"):
        for lid in _sorted_layer_ids(act_stats.get(family, {})):
            if lid.endswith("_img"):
                plot_C3_layer_range_across_time(act_stats, lid, family, plots_dir)

    # C4 — channel activation by stream across timesteps for every profiled block
    for blk in available_blks:
        plot_C4_qkv_range_by_stream_across_time(act_stats, f"mm_{blk:02d}", plots_dir)

    # C5 — channel activation by stream across layers at the middle timestep
    if demo_t_key is not None:
        plot_C5_qkv_range_by_stream_across_layers(act_stats, demo_t_key, plots_dir)

    # C2 — per-channel range boxplot at demo layer + demo timestep
    if demo_t_key is not None:
        for family in ("q_proj", "sdpa_out"):
            plot_C2_channel_range_boxplot(act_stats, demo_layer, family, demo_t_key, plots_dir)

    # C1 — animated per-channel histogram for each tracked channel (CHANNEL_HISTOGRAM_IDS)
    from .eda_tracer import CHANNEL_HISTOGRAM_IDS
    for ch in CHANNEL_HISTOGRAM_IDS:
        plot_C1_channel_histogram_animated(act_stats, demo_layer, "q_proj", ch, plots_dir)

    # C6 — txt/img ratio heatmaps for each target family
    for family in ("q_proj", "sdpa_out", "post_gelu"):
        plot_C6_txt_img_ratio(act_stats, family, plots_dir)

    print(f"\nAll plots written to {plots_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 EDA: Generate C1–C6 plots")
    parser.add_argument("--act-stats", type=str, default=_DEFAULT_ACT)
    parser.add_argument("--weight-stats", type=str, default=_DEFAULT_WGT)
    parser.add_argument("--tables-dir", type=str, default=_DEFAULT_TABLES)
    parser.add_argument("--plots-dir", type=str, default=_DEFAULT_PLOTS)
    parser.add_argument("--calibration-file", type=str, default=None)
    args = parser.parse_args()

    run_all_plots(
        act_stats_path=args.act_stats,
        weight_stats_path=args.weight_stats,
        tables_dir=args.tables_dir,
        plots_dir=args.plots_dir,
        cali_path=args.calibration_file,
    )


if __name__ == "__main__":
    main()
