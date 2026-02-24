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

def load_stats(path: str) -> Tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    np.ndarray,
    np.ndarray,
]:
    """
    Load the flat .npz and reconstruct the nested structure.

    Returns:
        stats: {layer_id: {t_key: {stat_name: array}}}
        timesteps_unique: sorted float array
        histogram_bin_edges: array of bin edges
    """
    data = np.load(path, allow_pickle=True)
    timesteps_unique = data["timesteps_unique"]
    histogram_bin_edges = data["histogram_bin_edges"]

    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for key in data.files:
        if "::t=" not in key:
            continue
        layer_id, rest = key.split("::t=", 1)
        t_key, stat_name = rest.split("::", 1)
        stats[layer_id][t_key][stat_name] = data[key]

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
