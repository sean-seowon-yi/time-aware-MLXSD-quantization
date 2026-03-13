"""
Phase 2 visualization: diagnostic plots for post-GELU FFN activations.

Loads the .npz output from profile_postgelu.py and generates the TaQ-DiT
paper-style diagnostic figures:

  1. Activation range vs. timestep (per layer)           — paper Fig 2(d)
  2. Post-GELU histogram for a chosen (layer, timestep)  — paper Fig 3(a)
  3. Channel-wise activation ranges                      — paper Fig 3(d)
  4. Temporal drift of mean/std (heatmaps across layers)
  5. Console summary table of all layers

Usage:

  python -m src.activation_diagnostics.visualize_postgelu \
      --stats-file dry_run_activation_stats_postgelu.npz \
      --output-dir activation_plots/
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError:
    raise ImportError("matplotlib is required for visualization: pip install matplotlib")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def load_stats(path: str, kind: str = "post") -> Tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    np.ndarray,
    np.ndarray,
]:
    """
    Load the flat .npz and reconstruct the nested structure.

    Parameters
    ----------
    path : str
        Path to the .npz file from profile_postgelu.py.
    kind : ``"post"`` | ``"pre"``
        ``"post"`` (default) — post-GELU activations (D = 4 × hidden_size).
        ``"pre"``  — pre-fc1 input activations (D = hidden_size).
        Use ``kind="pre"`` with a ``corrections_path`` for shifted/scaled plots,
        since z_g and s are computed for the fc1 input space.

    Returns
    -------
    stats : {layer_id: {t_key: {stat_name: array}}}
    timesteps_unique : sorted float array
    histogram_bin_edges : array of bin edges
    """
    if kind not in ("post", "pre"):
        raise ValueError(f"kind must be 'post' or 'pre', got '{kind}'")

    data = np.load(path, allow_pickle=True)
    timesteps_unique = data["timesteps_unique"]
    histogram_bin_edges = data["histogram_bin_edges"]

    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    if kind == "post":
        # Keys: {layer_id}::t={t_key}::{stat_name}  (no "pre::" prefix)
        for key in data.files:
            if "::t=" not in key or key.startswith("pre::"):
                continue
            layer_id, rest = key.split("::t=", 1)
            t_key, stat_name = rest.split("::", 1)
            stats[layer_id][t_key][stat_name] = data[key]
    else:
        # Keys: pre::{layer_id}::t={t_key}::{stat_name}
        for key in data.files:
            if not key.startswith("pre::") or "::t=" not in key:
                continue
            stripped = key[len("pre::"):]
            layer_id, rest = stripped.split("::t=", 1)
            t_key, stat_name = rest.split("::", 1)
            stats[layer_id][t_key][stat_name] = data[key]

        if not stats:
            raise KeyError(
                f"No pre-fc1 activation stats found in '{path}'. "
                "Re-run profile_postgelu.py to regenerate with pre-fc1 recording."
            )

    return dict(stats), timesteps_unique, histogram_bin_edges


def _sorted_timestep_keys(t_dict: Dict[str, Dict]) -> List[str]:
    """Sort timestep key strings by their numeric value."""
    return sorted(t_dict.keys(), key=lambda k: float(k))


def _t_key_to_float(t_key: str) -> float:
    return float(t_key)


def _layer_sort_key(layer_id: str) -> Tuple[int, str, int]:
    """Parse layer_id like 'mm_05_img' into (0, 'img', 5) for sorting."""
    parts = layer_id.split("_")
    if parts[0] == "mm":
        return (0, parts[2], int(parts[1]))
    elif parts[0] == "uni":
        return (1, "", int(parts[1]))
    return (2, layer_id, 0)


# ---------------------------------------------------------------------------
# Plot 1: Activation range vs. timestep (all layers, paper Fig 2d)
# ---------------------------------------------------------------------------

def plot_activation_range_vs_timestep(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    """
    For each layer, compute the median channel range at each timestep and
    plot all layers on one figure. Also produces per-modality subplots.
    """
    sorted_layers = sorted(stats.keys(), key=_layer_sort_key)
    img_layers = [l for l in sorted_layers if l.endswith("_img")]
    txt_layers = [l for l in sorted_layers if l.endswith("_txt")]

    for subset_name, subset_layers in [
        ("all", sorted_layers),
        ("image", img_layers),
        ("text", txt_layers),
    ]:
        if not subset_layers:
            continue

        fig, ax = plt.subplots(figsize=(14, 5))

        for layer_id in subset_layers:
            t_keys = _sorted_timestep_keys(stats[layer_id])
            t_vals = [_t_key_to_float(k) for k in t_keys]
            ranges = []
            for tk in t_keys:
                s = stats[layer_id][tk]
                ch_range = s["max"] - s["min"]
                ranges.append(np.median(ch_range))

            depth = _layer_sort_key(layer_id)[2]
            alpha = 0.3 + 0.7 * (depth / max(len(subset_layers), 1))
            ax.plot(t_vals, ranges, marker=".", markersize=3, alpha=alpha,
                    label=layer_id if len(subset_layers) <= 12 else None)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Median channel activation range")
        ax.set_title(f"Post-GELU activation range vs. timestep ({subset_name} layers)")
        ax.invert_xaxis()
        if len(subset_layers) <= 12:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"range_vs_timestep_{subset_name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved range_vs_timestep_{subset_name}.png")


# ---------------------------------------------------------------------------
# Plot 2: Histogram of post-GELU activations (paper Fig 3a)
# ---------------------------------------------------------------------------

def plot_histograms(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    bin_edges: np.ndarray,
    output_dir: Path,
    layers_to_plot: List[str] | None = None,
    max_layers: int = 6,
) -> None:
    """
    For selected layers, plot the activation histogram at every timestep.
    """
    sorted_layers = sorted(stats.keys(), key=_layer_sort_key)
    if layers_to_plot is None:
        img_layers = [l for l in sorted_layers if l.endswith("_img")]
        step = max(1, len(img_layers) // max_layers)
        layers_to_plot = img_layers[::step][:max_layers]

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for layer_id in layers_to_plot:
        t_keys = _sorted_timestep_keys(stats[layer_id])
        n_ts = len(t_keys)
        fig, axes = plt.subplots(1, n_ts, figsize=(4 * n_ts, 3.5), squeeze=False)

        for col, tk in enumerate(t_keys):
            ax = axes[0, col]
            hist = stats[layer_id][tk]["histogram"].astype(np.float64)
            ax.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]),
                   color="steelblue", edgecolor="none", alpha=0.8)
            ax.set_title(f"t={_t_key_to_float(tk):.0f}", fontsize=9)
            ax.set_xlabel("Activation value", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_yscale("log")

        fig.suptitle(f"Post-GELU histogram: {layer_id}", fontsize=11)
        fig.tight_layout()
        safe_name = layer_id.replace("/", "_")
        fig.savefig(output_dir / f"histogram_{safe_name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved histogram_{safe_name}.png")


# ---------------------------------------------------------------------------
# Plot 3: Channel-wise ranges (paper Fig 3d)
# ---------------------------------------------------------------------------

def plot_channel_ranges(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path,
    layers_to_plot: List[str] | None = None,
    max_layers: int = 4,
) -> None:
    """
    For selected layers, show per-channel (max-min) at each timestep as
    a stacked bar-like plot, highlighting outlier channels.
    """
    sorted_layers = sorted(stats.keys(), key=_layer_sort_key)
    if layers_to_plot is None:
        img_layers = [l for l in sorted_layers if l.endswith("_img")]
        step = max(1, len(img_layers) // max_layers)
        layers_to_plot = img_layers[::step][:max_layers]

    for layer_id in layers_to_plot:
        t_keys = _sorted_timestep_keys(stats[layer_id])
        n_ts = len(t_keys)
        fig, axes = plt.subplots(1, n_ts, figsize=(5 * n_ts, 3), squeeze=False)

        for col, tk in enumerate(t_keys):
            ax = axes[0, col]
            s = stats[layer_id][tk]
            ch_range = s["max"] - s["min"]
            n_ch = len(ch_range)
            ax.bar(range(n_ch), ch_range, width=1.0,
                   color="steelblue", edgecolor="none", alpha=0.7)

            # Highlight top 2% outlier channels (paper's threshold)
            top_k = max(1, int(n_ch * 0.02))
            outlier_idx = np.argsort(ch_range)[-top_k:]
            ax.bar(outlier_idx, ch_range[outlier_idx], width=1.0,
                   color="crimson", edgecolor="none", alpha=0.9)

            ax.set_title(f"t={_t_key_to_float(tk):.0f}", fontsize=9)
            ax.set_xlabel("Channel index", fontsize=8)
            ax.set_ylabel("Range (max-min)", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"Channel-wise ranges: {layer_id}  (red = top 2% outliers)",
                     fontsize=10)
        fig.tight_layout()
        safe_name = layer_id.replace("/", "_")
        fig.savefig(output_dir / f"channel_ranges_{safe_name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved channel_ranges_{safe_name}.png")


# ---------------------------------------------------------------------------
# Plot 4: Temporal drift heatmaps (mean and std across layers x timesteps)
# ---------------------------------------------------------------------------

def plot_temporal_heatmaps(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    """
    Two heatmaps: median-of-channel-means and median-of-channel-stds,
    rows = layers (sorted), columns = timesteps (sorted descending = denoising order).
    """
    sorted_layers = sorted(stats.keys(), key=_layer_sort_key)
    sample_layer = sorted_layers[0]
    t_keys = _sorted_timestep_keys(stats[sample_layer])
    t_vals = [_t_key_to_float(k) for k in t_keys]

    for stat_name, cmap, title in [
        ("mean", "RdBu_r", "Median channel mean"),
        ("std", "viridis", "Median channel std"),
    ]:
        matrix = np.zeros((len(sorted_layers), len(t_keys)))
        for row, layer_id in enumerate(sorted_layers):
            layer_t_keys = _sorted_timestep_keys(stats[layer_id])
            for col, tk in enumerate(layer_t_keys):
                arr = stats[layer_id][tk][stat_name]
                matrix[row, col] = np.median(arr)

        fig, ax = plt.subplots(figsize=(max(6, len(t_keys) * 1.2), max(8, len(sorted_layers) * 0.22)))
        if stat_name == "mean":
            vmax = max(abs(matrix.min()), abs(matrix.max()))
            norm = Normalize(vmin=-vmax, vmax=vmax)
        else:
            norm = None

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(range(len(t_vals)))
        ax.set_xticklabels([f"{v:.0f}" for v in t_vals], fontsize=7, rotation=45)
        ax.set_yticks(range(len(sorted_layers)))
        ax.set_yticklabels(sorted_layers, fontsize=6)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Layer")
        ax.set_title(f"Post-GELU {title} (per layer × timestep)")
        fig.colorbar(im, ax=ax, shrink=0.6)
        fig.tight_layout()
        fig.savefig(output_dir / f"heatmap_{stat_name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved heatmap_{stat_name}.png")


# ---------------------------------------------------------------------------
# Plot 5: Asymmetry analysis — fraction of negative activations
# ---------------------------------------------------------------------------

def plot_negative_fraction(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    bin_edges: np.ndarray,
    output_dir: Path,
) -> None:
    """
    For each (layer, timestep), compute what fraction of histogram mass
    falls in the negative range. The paper notes ~79.7% negative for
    Post-GELU in DiT.
    """
    sorted_layers = sorted(stats.keys(), key=_layer_sort_key)
    sample_layer = sorted_layers[0]
    t_keys = _sorted_timestep_keys(stats[sample_layer])
    t_vals = [_t_key_to_float(k) for k in t_keys]

    zero_bin = np.searchsorted(bin_edges, 0.0) - 1

    matrix = np.zeros((len(sorted_layers), len(t_keys)))
    for row, layer_id in enumerate(sorted_layers):
        layer_t_keys = _sorted_timestep_keys(stats[layer_id])
        for col, tk in enumerate(layer_t_keys):
            hist = stats[layer_id][tk]["histogram"].astype(np.float64)
            total = hist.sum()
            neg_mass = hist[:zero_bin].sum() if total > 0 else 0
            matrix[row, col] = neg_mass / max(total, 1)

    fig, ax = plt.subplots(figsize=(max(6, len(t_keys) * 1.2), max(8, len(sorted_layers) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(t_vals)))
    ax.set_xticklabels([f"{v:.0f}" for v in t_vals], fontsize=7, rotation=45)
    ax.set_yticks(range(len(sorted_layers)))
    ax.set_yticklabels(sorted_layers, fontsize=6)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Layer")
    ax.set_title("Fraction of Post-GELU activations < 0 (paper reports ~80%)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Fraction negative")
    fig.tight_layout()
    fig.savefig(output_dir / "negative_fraction_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved negative_fraction_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 6: Boxplot of a single channel's activations across timesteps
# ---------------------------------------------------------------------------

def plot_channel_boxplot_across_timesteps(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    layer_id: str,
    channel_idx: int,
    t_range: Tuple[float, float] | None = None,
    output_dir: Path | None = None,
    ax=None,
):
    """
    Draw a boxplot showing the activation distribution of one channel
    across profiled timesteps (x-axis ordered high→low, denoising order).

    Since only summary statistics are stored (not raw samples), the box
    is constructed from Gaussian approximations:
      - whisker low  = min[channel]
      - Q1           ≈ mean[channel] - 0.674 * std[channel]
      - median       ≈ mean[channel]
      - Q3           ≈ mean[channel] + 0.674 * std[channel]
      - whisker high = max[channel]

    Parameters
    ----------
    stats : nested dict from load_stats()
    layer_id : str
        Layer to visualise, e.g. ``"mm_05_img"``.
    channel_idx : int
        Zero-based channel (hidden-unit) index.
    t_range : (t_min, t_max) or None
        Inclusive timestep range to display, e.g. ``(200, 800)`` shows only
        timesteps between 200 and 800.  ``None`` shows all profiled timesteps.
    output_dir : Path or None
        If given, saves the figure as
        ``{output_dir}/channel_boxplot_{layer_id}_ch{channel_idx}.png``.
    ax : matplotlib Axes or None
        If given, draws into this axes object (useful in notebooks).
        If None, a new figure is created.

    Returns
    -------
    fig : matplotlib Figure
    """
    if layer_id not in stats:
        raise KeyError(
            f"Layer '{layer_id}' not found. Available layers: {sorted(stats.keys())}"
        )

    all_t_keys = _sorted_timestep_keys(stats[layer_id])

    # Apply timestep range filter
    if t_range is not None:
        t_min, t_max = float(t_range[0]), float(t_range[1])
        t_keys = [k for k in all_t_keys if t_min <= _t_key_to_float(k) <= t_max]
        if not t_keys:
            raise ValueError(
                f"No timesteps in range [{t_min}, {t_max}]. "
                f"Available: {[_t_key_to_float(k) for k in all_t_keys]}"
            )
    else:
        t_keys = all_t_keys

    n_channels = stats[layer_id][t_keys[0]]["mean"].shape[0]
    if channel_idx < 0 or channel_idx >= n_channels:
        raise ValueError(
            f"channel_idx={channel_idx} out of range for layer '{layer_id}' "
            f"which has {n_channels} channels."
        )

    # Build bxp-compatible dicts, one per timestep (denoising order: high→low)
    box_stats = []
    for tk in reversed(t_keys):
        s = stats[layer_id][tk]
        mu = float(s["mean"][channel_idx])
        sigma = float(s["std"][channel_idx])
        lo = float(s["min"][channel_idx])
        hi = float(s["max"][channel_idx])
        box_stats.append({
            "med":    mu,
            "q1":     mu - 0.674 * sigma,
            "q3":     mu + 0.674 * sigma,
            "whislo": lo,
            "whishi": hi,
            "fliers": [],
        })

    t_labels = [f"{_t_key_to_float(k):.0f}" for k in reversed(t_keys)]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, len(t_keys) * 0.2), 4))
    else:
        fig = ax.get_figure()

    _spacing = 0.3  # center-to-center distance; reduce to compress gaps
    _positions = [i * _spacing for i in range(len(box_stats))]
    ax.bxp(
        box_stats,
        positions=_positions,
        widths=0.1,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color="steelblue"),
        capprops=dict(color="steelblue"),
    )

    ax.set_xticks(_positions)
    ax.set_xticklabels(t_labels, rotation=90, ha="center", fontsize=8)
    ax.set_xlabel("Timestep (denoising order →)", fontsize=9)
    ax.set_ylabel("Activation value", fontsize=9)

    range_str = f"  t ∈ [{t_range[0]:.0f}, {t_range[1]:.0f}]" if t_range else ""
    ax.set_title(
        f"Channel {channel_idx} activation distribution across timesteps\n"
        f"Layer: {layer_id}{range_str}  "
        f"(box = mean ± 0.674σ, whiskers = [min, max])",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)

    if own_fig:
        fig.tight_layout()

    if output_dir is not None:
        safe_layer = layer_id.replace("/", "_")
        fname = output_dir / f"channel_boxplot_{safe_layer}_ch{channel_idx:04d}.png"
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname.name}")

    return fig


# ---------------------------------------------------------------------------
# Plot 7 (helper): Group boundary computation for HTG analysis
# ---------------------------------------------------------------------------

def _group_boundaries(t_vals_asc: np.ndarray, divisor: int) -> np.ndarray:
    """
    Return timestep values at which HTG group transitions occur for a given divisor.

    With T timesteps and divisor d, G = T // d equal-count groups are formed.
    The boundary between group i and i+1 is the midpoint between the last
    timestep of group i and the first timestep of group i+1.

    Parameters
    ----------
    t_vals_asc : (T,) array of timestep values sorted ascending
    divisor : int  — e.g. 2 → T/2 groups, 5 → T/5 groups, 10 → T/10 groups
    """
    T = len(t_vals_asc)
    G = max(1, T // divisor)
    step = T // G
    boundaries = []
    for i in range(1, G):
        idx = i * step
        if idx < T:
            mid = (float(t_vals_asc[idx - 1]) + float(t_vals_asc[idx])) / 2.0
            boundaries.append(mid)
    return np.array(boundaries)


# ---------------------------------------------------------------------------
# Plot 8: Channel activation range vs timestep (continuous band + group lines)
# ---------------------------------------------------------------------------

def plot_channel_range_vs_timestep(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    layer_id: str,
    channel_idx: int,
    group_divisors: List[int] | None = None,
    output_dir: Path | None = None,
    ax=None,
) -> plt.Figure:
    """
    Plot a single channel's activation statistics as a continuous band over
    all profiled timesteps, with optional HTG group-boundary overlays.

    Useful for investigating whether T/2, T/5, or T/10 grouping aligns with
    where activation statistics actually change across the diffusion trajectory.

    Visual elements
    ---------------
    - Faint fill  (alpha=0.15): [min, max] envelope
    - Medium fill (alpha=0.35): mean ± std band
    - Solid line:               mean
    - Dashed black line:        y = 0
    - Vertical dashed lines:    group boundaries for each entry in
                                ``group_divisors`` (one colour per divisor)

    Parameters
    ----------
    stats : dict from load_stats()
        Works with both ``kind="post"`` (D=6144) and ``kind="pre"`` (D=1536).
    layer_id : str
        e.g. ``"mm_05_img"``.
    channel_idx : int
        Zero-based channel index.
    group_divisors : list[int] or None
        Divisors for overlaying group boundaries.  e.g. ``[2, 5, 10]`` draws
        boundaries for T/2, T/5, and T/10 groups.  ``None`` = no overlays.
    output_dir : Path or None
        If given, saves as ``channel_range_{layer_id}_ch{channel_idx:04d}.png``.
    ax : matplotlib Axes or None
        Draw into an existing axes (notebook); ``None`` creates a new figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    if layer_id not in stats:
        raise KeyError(
            f"Layer '{layer_id}' not found. Available: {sorted(stats.keys())}"
        )

    t_keys = _sorted_timestep_keys(stats[layer_id])
    n_ch = stats[layer_id][t_keys[0]]["mean"].shape[0]
    if not (0 <= channel_idx < n_ch):
        raise ValueError(
            f"channel_idx={channel_idx} out of range; layer '{layer_id}' has {n_ch} channels."
        )

    # Collect per-timestep stats for this channel (denoising order: high → low)
    t_vals = np.array([_t_key_to_float(k) for k in reversed(t_keys)])
    means  = np.array([stats[layer_id][k]["mean"][channel_idx] for k in reversed(t_keys)])
    stds   = np.array([stats[layer_id][k]["std"][channel_idx]  for k in reversed(t_keys)])
    lows   = np.array([stats[layer_id][k]["min"][channel_idx]  for k in reversed(t_keys)])
    highs  = np.array([stats[layer_id][k]["max"][channel_idx]  for k in reversed(t_keys)])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, len(t_vals) * 0.18), 4))
    else:
        fig = ax.get_figure()

    color = "steelblue"
    ax.fill_between(t_vals, lows,  highs,            color=color, alpha=0.15, label="[min, max]")
    ax.fill_between(t_vals, means - stds, means + stds, color=color, alpha=0.35, label="mean ± std")
    ax.plot(t_vals, means, color=color, linewidth=1.2, label="mean")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    # Group boundary overlays
    if group_divisors:
        t_asc = np.sort(t_vals)   # ascending for boundary computation
        _COLORS = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
        for i, d in enumerate(sorted(group_divisors)):
            boundaries = _group_boundaries(t_asc, d)
            G = max(1, len(t_vals) // d)
            c = _COLORS[i % len(_COLORS)]
            for j, b in enumerate(boundaries):
                ax.axvline(
                    b, color=c, linewidth=0.9, linestyle=":",
                    label=f"T/{d} ({G} groups)" if j == 0 else "_nolegend_",
                )

    # X-axis: denoising direction label (high → low already)
    ax.invert_xaxis()
    ax.set_xlabel("Timestep (denoising direction →)", fontsize=9)
    ax.set_ylabel("Activation value", fontsize=9)
    ax.set_title(
        f"Ch {channel_idx} activation range vs timestep  |  layer: {layer_id}",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="upper right", ncol=max(1, 1 + len(group_divisors or [])))
    ax.grid(True, axis="y", alpha=0.25)

    if own_fig:
        fig.tight_layout()

    if output_dir is not None:
        safe_layer = layer_id.replace("/", "_")
        fname = output_dir / f"channel_range_{safe_layer}_ch{channel_idx:04d}.png"
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname.name}")

    return fig


# ---------------------------------------------------------------------------
# Plot 9: Unified channel-wise distribution with HTG transform modes
# ---------------------------------------------------------------------------

def _load_corrections(corrections_path: str) -> Dict[str, np.ndarray]:
    """Load htg_corrections.npz or htg_params.npz into a flat dict."""
    data = np.load(corrections_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _find_htg_layer(corrections: Dict[str, np.ndarray], layer_id: str) -> str | None:
    """
    Match a stats layer_id (e.g. ``"mm_05_img"``) to the corresponding HTG
    layer key in the corrections file.

    Post-GELU stats correspond to the output of fc1+GELU, so we look for
    ``{layer_id}_fc1`` first, then any key that begins with ``{layer_id}_``.
    """
    preferred = f"{layer_id}_fc1"
    htg_layer_ids = {k.split("::")[0] for k in corrections if "::" in k}
    if preferred in htg_layer_ids:
        return preferred
    for lid in sorted(htg_layer_ids):
        if lid.startswith(layer_id + "_"):
            return lid
    return None


def _snap_timestep(stats_layer: Dict[str, Dict[str, np.ndarray]], timestep: float) -> str:
    """Return the t_key whose float value is closest to ``timestep``."""
    t_keys = list(stats_layer.keys())
    return min(t_keys, key=lambda k: abs(_t_key_to_float(k) - timestep))


def _apply_transform(
    mode: str,
    mean: np.ndarray,
    std: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    z_g: np.ndarray,
    s: np.ndarray,
) -> tuple:
    """
    Apply the requested HTG transform to per-channel summary statistics.

    Parameters
    ----------
    mode : ``"original"`` | ``"shifted"`` | ``"scaled"``
    z_g  : (D,) per-channel group shift
    s    : (D,) per-channel scale (safe, ≥ 1e-8)

    Returns
    -------
    (mean, std, lo, hi) after transformation
    """
    if mode == "original":
        return mean, std, lo, hi
    if mode == "shifted":
        return mean - z_g, std, lo - z_g, hi - z_g
    if mode == "scaled":
        return (mean - z_g) / s, std / s, (lo - z_g) / s, (hi - z_g) / s
    raise ValueError(f"Unknown mode '{mode}'. Choose from: 'original', 'shifted', 'scaled'.")


_MODE_COLOR = {
    "original": "steelblue",
    "shifted":  "darkorange",
    "scaled":   "seagreen",
}

_MODE_LABEL = {
    "original": "X  (raw)",
    "shifted":  "X − z_g",
    "scaled":   "(X − z_g) / s",
}


def plot_channel_distribution(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    layer_id: str,
    timestep: float,
    corrections_path: str | None = None,
    show: List[str] = ("original",),
    channel_idx: "int | List[int] | None" = None,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Plot the channel-wise activation distribution for a given layer and
    timestep, with optional HTG transforms applied.

    One panel is drawn per entry in ``show``, stacked vertically and sharing
    the x-axis (channel index). Each panel displays:
      - Faint envelope:  [min, max] across calibration samples
      - Shaded band:     mean ± std
      - Line:            mean
      - Dashed line:     y = 0

    Supported transform modes
    -------------------------
    ``"original"``
        Raw activations X.  Works with both ``kind="post"`` and ``kind="pre"``.
    ``"shifted"``
        X − z_g[group]  — subtracts the per-group HTG shifting vector.
        Requires ``stats`` loaded with ``load_stats(path, kind="pre")`` so that
        D matches z_g (both = hidden_size).
    ``"scaled"``
        (X − z_g[group]) / s  — full HTG normalization (shift + scale).
        Also requires ``kind="pre"`` stats.

    Parameters
    ----------
    stats : dict from load_stats()
        For ``"shifted"`` / ``"scaled"`` modes use ``load_stats(path, kind="pre")``
        to obtain fc1 input activation stats (D = hidden_size) that align with z_g.
    layer_id : str
        Stats layer ID, e.g. ``"mm_05_img"``. For ``"shifted"`` and
        ``"scaled"`` modes the matching HTG layer (``"mm_05_img_fc1"``)
        is resolved automatically from the corrections file.
    timestep : float
        Target denoising timestep (snapped to nearest recorded value).
    corrections_path : str or None
        Path to ``htg_corrections.npz`` or ``htg_params.npz``. Required
        when ``show`` includes ``"shifted"`` or ``"scaled"``.
    show : sequence of str
        Ordered list of modes to display. Default: ``("original",)``.
        Each entry produces one panel.
    channel_idx : int, list of int, or None
        Channels to display. A single int selects one channel; a list selects
        multiple (plotted at their true channel-index positions on the x-axis);
        ``None`` (default) displays all channels.
    output_dir : Path or None
        If given, saves the figure as
        ``channel_dist_{layer_id}_t{t:.0f}_{modes}.png``.

    Returns
    -------
    fig : matplotlib Figure
    """
    _valid_modes = {"original", "shifted", "scaled"}
    show = list(show)
    for m in show:
        if m not in _valid_modes:
            raise ValueError(f"Unknown mode '{m}'. Choose from: {sorted(_valid_modes)}.")

    needs_corrections = any(m in ("shifted", "scaled") for m in show)
    if needs_corrections and corrections_path is None:
        raise ValueError(
            "corrections_path is required for 'shifted' and 'scaled' modes."
        )

    if layer_id not in stats:
        raise KeyError(
            f"Layer '{layer_id}' not found. Available: {sorted(stats.keys())}"
        )

    # --- Snap to nearest recorded timestep ---
    t_key = _snap_timestep(stats[layer_id], timestep)
    t_actual = _t_key_to_float(t_key)
    if abs(t_actual - timestep) > 1.0:
        print(f"  Note: requested t={timestep:.1f}, snapped to t={t_actual:.1f}")

    s_dict = stats[layer_id][t_key]
    mean = s_dict["mean"].astype(np.float64)   # (D,)
    std  = s_dict["std"].astype(np.float64)    # (D,)
    lo   = s_dict["min"].astype(np.float64)    # (D,)
    hi   = s_dict["max"].astype(np.float64)    # (D,)
    D = mean.shape[0]
    ch = np.arange(D)

    # --- Load HTG corrections if needed ---
    z_g = np.zeros(D, dtype=np.float64)
    s_scale = np.ones(D, dtype=np.float64)
    group_label = ""
    htg_lid = None

    if needs_corrections:
        corrections = _load_corrections(corrections_path)
        htg_lid = _find_htg_layer(corrections, layer_id)
        if htg_lid is None:
            print(
                f"  Warning: no HTG corrections found for '{layer_id}'. "
                f"z_g=0, s=1 will be used."
            )
        else:
            z_g_all   = corrections[f"{htg_lid}::z_g"].astype(np.float64)   # (G, D)
            s_raw     = corrections[f"{htg_lid}::s"].astype(np.float64)      # (D,)
            group_asn = corrections[f"{htg_lid}::group_assignments"]          # (T,)
            ts_sorted = corrections["timesteps_sorted"].astype(np.float64)   # (T,)

            t_idx = int(np.argmin(np.abs(ts_sorted - t_actual)))
            g_idx = int(group_asn[t_idx])
            z_g     = z_g_all[g_idx]
            s_scale = np.maximum(s_raw, 1e-8)
            group_label = f"group {g_idx}  ({htg_lid})"

            # Dimension sanity check: z_g is for fc1 *input* (D=hidden),
            # but post-GELU stats are for fc1 *output* (D=4×hidden).
            if z_g.shape[0] != D:
                raise ValueError(
                    f"Dimension mismatch for layer '{layer_id}': "
                    f"activation stats have D={D} (post-GELU, fc1 output = 4 × hidden_size) "
                    f"but z_g has D={z_g.shape[0]} (fc1 input = hidden_size). "
                    f"Load fc1 input stats with load_stats(path, kind='pre') — these have "
                    f"D={z_g.shape[0]} and align with z_g / s from the corrections file."
                )

    # --- Channel selection ---
    if channel_idx is None:
        sel = np.arange(D)
    elif isinstance(channel_idx, int):
        sel = np.array([channel_idx])
    else:
        sel = np.array(sorted(channel_idx))

    ch      = sel
    mean    = mean[sel]
    std     = std[sel]
    lo      = lo[sel]
    hi      = hi[sel]
    z_g     = z_g[sel]
    s_scale = s_scale[sel]

    D_plot = len(sel)
    ch_label = (
        "Channel index"
        if channel_idx is None
        else f"Channel index  (showing {D_plot} of {D})"
    )

    # --- Build figure ---
    _spacing = 0.3
    _positions = [i * _spacing for i in range(D_plot)]

    n_panels = len(show)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(min(16, max(10, D_plot // 100)), 3.5 * n_panels),
        sharex=True,
        squeeze=False,
    )

    def _draw_panel(ax, mu, sg, vlo, vhi, color, title):
        box_stats = [
            {
                "med":    float(mu[i]),
                "q1":     float(mu[i] - 0.674 * sg[i]),
                "q3":     float(mu[i] + 0.674 * sg[i]),
                "whislo": float(vlo[i]),
                "whishi": float(vhi[i]),
                "fliers": [],
            }
            for i in range(D_plot)
        ]
        ax.bxp(
            box_stats,
            positions=_positions,
            widths=0.1,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="white", linewidth=1.5),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_ylabel("Activation value", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

    # Pre-compute all transforms to find shared y range
    transformed = [
        _apply_transform(mode, mean, std, lo, hi, z_g, s_scale) for mode in show
    ]
    y_min = min(float(vlo.min()) for _, _, vlo, _ in transformed)
    y_max = max(float(vhi.max()) for _, _, _, vhi in transformed)
    y_pad = (y_max - y_min) * 0.05
    y_lim = (y_min - y_pad, y_max + y_pad)

    for row, (mode, (mu, sg, vlo, vhi)) in enumerate(zip(show, transformed)):
        ax = axes[row, 0]

        subtitle = f"{_MODE_LABEL[mode]}"
        if mode != "original" and group_label:
            subtitle += f"  |  {group_label}"

        _draw_panel(ax, mu, sg, vlo, vhi, color=_MODE_COLOR[mode], title=subtitle)
        ax.set_ylim(y_lim)

    axes[-1, 0].set_xticks(_positions)
    axes[-1, 0].set_xticklabels([str(c) for c in sel], rotation=90, ha="center", fontsize=8)
    axes[-1, 0].set_xlabel(ch_label, fontsize=9)
    fig.suptitle(
        f"Channel-wise activation distribution  |  layer: {layer_id}  |  t ≈ {t_actual:.0f}",
        fontsize=11,
    )
    fig.tight_layout()

    if output_dir is not None:
        safe_layer = layer_id.replace("/", "_")
        modes_tag = "_".join(show)
        fname = output_dir / f"channel_dist_{safe_layer}_t{t_actual:.0f}_{modes_tag}.png"
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname.name}")

    return fig


# ---------------------------------------------------------------------------
# Table: Console summary
# ---------------------------------------------------------------------------

def print_summary_table(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    bin_edges: np.ndarray,
) -> None:
    """Print a concise text table summarizing each layer."""
    sorted_layers = sorted(stats.keys(), key=_layer_sort_key)
    sample_layer = sorted_layers[0]
    t_keys = _sorted_timestep_keys(stats[sample_layer])

    zero_bin = np.searchsorted(bin_edges, 0.0) - 1

    header = (
        f"{'Layer':<16} {'Med.Mean':>9} {'Med.Std':>9} "
        f"{'Med.Range':>10} {'Max.Range':>10} {'%Neg':>6} {'Outlier Ch':>10}"
    )
    print("\n" + "=" * len(header))
    print("Post-GELU Activation Summary (aggregated across timesteps)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for layer_id in sorted_layers:
        all_means = []
        all_stds = []
        all_ranges = []
        all_neg_frac = []

        for tk in _sorted_timestep_keys(stats[layer_id]):
            s = stats[layer_id][tk]
            all_means.append(np.median(s["mean"]))
            all_stds.append(np.median(s["std"]))
            ch_range = s["max"] - s["min"]
            all_ranges.append(ch_range)

            hist = s["histogram"].astype(np.float64)
            total = hist.sum()
            neg = hist[:zero_bin].sum() if total > 0 else 0
            all_neg_frac.append(neg / max(total, 1))

        agg_mean = np.mean(all_means)
        agg_std = np.mean(all_stds)
        all_ch_ranges = np.stack(all_ranges, axis=0)
        med_range = np.median(all_ch_ranges)
        max_range = np.max(all_ch_ranges)
        neg_frac = np.mean(all_neg_frac)

        # Outlier channels: those in the top 2% of range at any timestep
        n_ch = all_ch_ranges.shape[1]
        top_k = max(1, int(n_ch * 0.02))
        outlier_set = set()
        for row in all_ch_ranges:
            outlier_set.update(np.argsort(row)[-top_k:].tolist())

        print(
            f"{layer_id:<16} {agg_mean:>9.4f} {agg_std:>9.4f} "
            f"{med_range:>10.4f} {max_range:>10.4f} {neg_frac:>5.1%} "
            f"{len(outlier_set):>10d}"
        )

    print("-" * len(header))
    print(f"Total layers: {len(sorted_layers)}, Timesteps: {len(t_keys)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Phase 2 post-GELU activation diagnostics."
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default="activation_stats_postgelu.npz",
        help="Path to the .npz from profile_postgelu.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="activation_plots",
        help="Directory to save plot PNGs",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="*",
        default=None,
        help="Specific layer IDs for per-layer plots (default: auto-select)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading stats from {args.stats_file} ...")
    stats, timesteps_unique, bin_edges = load_stats(args.stats_file)
    print(f"  {len(stats)} layers, {len(timesteps_unique)} timesteps")
    print(f"  Saving plots to {output_dir}/\n")

    print("[1/5] Activation range vs. timestep ...")
    plot_activation_range_vs_timestep(stats, output_dir)

    print("[2/5] Post-GELU histograms ...")
    plot_histograms(stats, bin_edges, output_dir, layers_to_plot=args.layers)

    print("[3/5] Channel-wise ranges ...")
    plot_channel_ranges(stats, output_dir, layers_to_plot=args.layers)

    print("[4/5] Temporal drift heatmaps ...")
    plot_temporal_heatmaps(stats, output_dir)

    print("[5/5] Asymmetry (negative fraction) heatmap ...")
    plot_negative_fraction(stats, bin_edges, output_dir)

    print_summary_table(stats, bin_edges)
    print(f"All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
