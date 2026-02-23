"""
Visualize per-layer activation statistics from layer_statistics.json.

Two main views:
  1. Snapshot view  — all layers at a single timestep (absmax bar chart, colored by type)
  2. Temporal view  — selected layers across all timesteps (absmax vs sigma line plot)

Also produces:
  3. Heatmap        — all layers × all timesteps (absmax), sorted by variability
  4. Shift view     — post-GELU shift_absmax vs sigma for all fc2 layers

Usage:
    python -m src.visualize_activations \\
        --stats /path/to/calibration_data/activations/layer_statistics.json \\
        --output-dir /path/to/calibration_data/activations/plots
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LAYER_COLORS = {
    "img.attn.q_proj": "#4e79a7",
    "img.attn.k_proj": "#4e79a7",
    "img.attn.v_proj": "#59a14f",
    "img.attn.o_proj": "#f28e2b",
    "img.mlp.fc1":     "#e15759",
    "img.mlp.fc2":     "#b07aa1",  # post-GELU
    "txt.attn.q_proj": "#76b7b2",
    "txt.attn.k_proj": "#76b7b2",
    "txt.attn.v_proj": "#9c755f",
    "txt.attn.o_proj": "#bab0ac",
    "txt.mlp.fc1":     "#edc948",
    "txt.mlp.fc2":     "#ff9da7",  # post-GELU
}

def layer_color(name: str) -> str:
    for suffix, color in LAYER_COLORS.items():
        if name.endswith(suffix):
            return color
    return "#aaaaaa"

def layer_type(name: str) -> str:
    """Return short type label for legend."""
    for suffix in LAYER_COLORS:
        if name.endswith(suffix):
            return suffix
    return "other"

def block_index(name: str) -> int:
    """Extract block number from layer name like mm12.img.attn.q_proj."""
    try:
        return int(name.split(".")[0].replace("mm", "").replace("uni", ""))
    except Exception:
        return -1


def load_data(stats_path: Path):
    with open(stats_path) as f:
        data = json.load(f)
    sigma_map = {int(k): float(v) for k, v in data["sigma_map"].items()}

    # Support both old JSON format and new per-timestep npz format
    if "timesteps" in data:
        timesteps = data["timesteps"]
    else:
        # New format: load per-timestep npz files
        ts_dir = Path(data["timestep_dir"])
        timesteps = {}
        for step_key in data["step_keys"]:
            npz = np.load(ts_dir / f"step_{step_key}.npz")
            with open(ts_dir / f"step_{step_key}_index.json") as f:
                index = json.load(f)
            layers = {}
            for layer_name, meta in index.items():
                safe = layer_name.replace(".", "_")
                entry = dict(meta)
                entry["avg_min"] = npz[f"{safe}__avg_min"].tolist()
                entry["avg_max"] = npz[f"{safe}__avg_max"].tolist()
                if meta.get("has_shift"):
                    entry["shift"] = npz[f"{safe}__shift"].tolist()
                layers[layer_name] = entry
            timesteps[step_key] = layers

    step_keys = sorted(timesteps.keys(), key=int)
    layer_names = sorted(timesteps[step_keys[0]].keys())
    return timesteps, sigma_map, step_keys, layer_names


# ---------------------------------------------------------------------------
# Plot 1: Snapshot — all layers at one timestep
# ---------------------------------------------------------------------------

def plot_snapshot(timesteps, sigma_map, step_keys, layer_names, step_idx: int, out_dir: Path):
    key = str(step_idx)
    if key not in timesteps:
        # Find nearest
        key = min(step_keys, key=lambda k: abs(int(k) - step_idx))
    sigma = sigma_map[int(key)]
    layers = timesteps[key]

    names = [n for n in layer_names if n in layers]
    absmax = [layers[n]["tensor_absmax"] for n in names]
    colors = [layer_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(22, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, absmax, color=colors, width=0.8, edgecolor="none")

    # Mark post-GELU with a dot on top
    for i, n in enumerate(names):
        if n.endswith(".mlp.fc2"):
            ax.plot(i, absmax[i] + max(absmax) * 0.01, "v", color="white",
                    markersize=4, markeredgecolor="#333", markeredgewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=5.5)
    ax.set_ylabel("tensor absmax (avg over calibration images)")
    ax.set_title(f"All Layers — Step {key}  (σ={sigma:.3f})")
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    legend_items = [
        Line2D([0], [0], color=c, lw=6, label=s)
        for s, c in LAYER_COLORS.items()
    ]
    legend_items.append(Line2D([0], [0], marker="v", color="white",
                               markeredgecolor="#333", markersize=6,
                               label="post-GELU (fc2)", lw=0))
    ax.legend(handles=legend_items, loc="upper right", fontsize=7,
              ncol=2, framealpha=0.8)

    fig.tight_layout()
    path = out_dir / f"snapshot_step{key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


# ---------------------------------------------------------------------------
# Plot 2: Temporal — selected layers over all timesteps
# ---------------------------------------------------------------------------

def plot_temporal(timesteps, sigma_map, step_keys, layer_names, layers_to_plot, out_dir: Path,
                  use_shift: bool = False, title_suffix: str = ""):
    sigmas = [sigma_map[int(k)] for k in step_keys]

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = cm.get_cmap("tab20", len(layers_to_plot))

    for i, layer in enumerate(layers_to_plot):
        vals = []
        for k in step_keys:
            s = timesteps[k].get(layer, {})
            if use_shift and "shift_absmax" in s:
                vals.append(s["shift_absmax"])
            else:
                vals.append(s.get("tensor_absmax", np.nan))
        label = layer.replace("mm", "mm").replace(".img.", " img.").replace(".txt.", " txt.")
        ax.plot(sigmas, vals, "o-", color=cmap(i), label=label,
                linewidth=1.8, markersize=4)

    ax.invert_xaxis()  # sigma goes 1.0 → 0.0 (high noise → clean)
    ax.set_xlabel("σ (noise level)  →  denoising direction →")
    ylabel = "shift_absmax" if use_shift else "tensor absmax"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Layer Activations vs Timestep{title_suffix}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.8)

    fig.tight_layout()
    suffix = "_shift" if use_shift else ""
    fname = f"temporal{'_' + title_suffix.strip() if title_suffix else ''}{suffix}.png"
    fname = fname.replace(" ", "_").replace("(", "").replace(")", "")
    path = out_dir / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


# ---------------------------------------------------------------------------
# Plot 3: Heatmap — all layers × all timesteps
# ---------------------------------------------------------------------------

def plot_heatmap(timesteps, sigma_map, step_keys, layer_names, out_dir: Path,
                 sort_by: str = "variability", max_layers: int = 120):
    sigmas = [sigma_map[int(k)] for k in step_keys]

    # Build matrix
    matrix = np.full((len(layer_names), len(step_keys)), np.nan)
    for j, k in enumerate(step_keys):
        for i, name in enumerate(layer_names):
            s = timesteps[k].get(name, {})
            if s:
                matrix[i, j] = s["tensor_absmax"]

    # Sort rows
    if sort_by == "variability":
        valid = ~np.isnan(matrix).all(axis=1)
        row_range = np.nanmax(matrix, axis=1) / (np.nanmin(matrix, axis=1) + 1e-6)
        row_range[~valid] = 0
        order = np.argsort(-row_range)
    elif sort_by == "absmax":
        order = np.argsort(-np.nanmax(matrix, axis=1))
    else:
        order = np.arange(len(layer_names))

    # Limit to top N layers
    order = order[:max_layers]
    matrix = matrix[order]
    names_sorted = [layer_names[i] for i in order]

    # Log scale for better visibility
    log_matrix = np.log1p(matrix)

    fig, ax = plt.subplots(figsize=(18, max(8, len(names_sorted) * 0.18)))
    im = ax.imshow(log_matrix, aspect="auto", cmap="YlOrRd",
                   extent=[sigmas[0], sigmas[-1], len(names_sorted) - 0.5, -0.5])

    ax.set_yticks(np.arange(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=6)
    ax.set_xlabel("σ (noise level)")
    ax.set_title(f"Activation Heatmap — log(1+absmax)  [sorted by {sort_by}]")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("log(1 + absmax)")

    # Highlight post-GELU rows
    for i, name in enumerate(names_sorted):
        if name.endswith(".mlp.fc2"):
            ax.axhline(i, color="cyan", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    path = out_dir / f"heatmap_{sort_by}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


# ---------------------------------------------------------------------------
# Plot 4: Per-channel distribution for a single layer at a single step
# ---------------------------------------------------------------------------

def plot_channel_dist(timesteps, sigma_map, step_keys, layer_name: str,
                      step_idx: int, out_dir: Path):
    key = str(step_idx)
    if key not in timesteps:
        key = min(step_keys, key=lambda k: abs(int(k) - step_idx))

    s = timesteps[key].get(layer_name)
    if not s:
        print(f"  Layer {layer_name} not found at step {key}")
        return

    sigma = sigma_map[int(key)]
    avg_min = np.array(s["avg_min"])
    avg_max = np.array(s["avg_max"])
    absmax = np.maximum(np.abs(avg_min), np.abs(avg_max))
    C = len(absmax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: sorted per-channel absmax
    ax = axes[0]
    sorted_abs = np.sort(absmax)[::-1]
    ax.bar(np.arange(C), sorted_abs, width=1.0, color=layer_color(layer_name), edgecolor="none")
    p50 = np.percentile(absmax, 50)
    p99 = np.percentile(absmax, 99)
    ax.axhline(p50, color="blue", linewidth=1.2, linestyle="--", label=f"p50={p50:.2f}")
    ax.axhline(p99, color="red", linewidth=1.2, linestyle="--", label=f"p99={p99:.2f}")
    ax.set_xlabel("channel (sorted by absmax)")
    ax.set_ylabel("absmax")
    ax.set_title(f"Per-channel absmax — {layer_name}\nStep {key}  σ={sigma:.3f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: histogram of absmax values
    ax = axes[1]
    ax.hist(absmax, bins=60, color=layer_color(layer_name), edgecolor="none", alpha=0.8)
    ax.axvline(p50, color="blue", linewidth=1.5, linestyle="--", label=f"p50={p50:.2f}")
    ax.axvline(p99, color="red", linewidth=1.5, linestyle="--", label=f"p99={p99:.2f}")
    if "shift" in s:
        shift = np.array(s["shift"])
        shift_abs = np.abs(shift)
        ax.axvline(np.median(shift_abs), color="green", linewidth=1.5,
                   linestyle=":", label=f"shift median={np.median(shift_abs):.2f}")
    ax.set_xlabel("absmax value")
    ax.set_ylabel("# channels")
    ax.set_title("Distribution of per-channel absmax")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    safe_name = layer_name.replace(".", "_")
    path = out_dir / f"channel_dist_{safe_name}_step{key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


# ---------------------------------------------------------------------------
# Plot 5: Temporal variability summary — scatter
# ---------------------------------------------------------------------------

def plot_variability_scatter(timesteps, sigma_map, step_keys, layer_names, out_dir: Path):
    """Scatter: x=mean absmax, y=max/min ratio (variability), colored by layer type."""
    means, ratios, colors, names = [], [], [], []

    for name in layer_names:
        vals = [timesteps[k][name]["tensor_absmax"]
                for k in step_keys if name in timesteps[k]]
        if not vals:
            continue
        mn, mx = min(vals), max(vals)
        means.append(np.mean(vals))
        ratios.append(mx / (mn + 1e-6))
        colors.append(layer_color(name))
        names.append(name)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(means, ratios, c=colors, s=30, alpha=0.7, edgecolors="none")

    # Label the most extreme points
    ratios_arr = np.array(ratios)
    means_arr = np.array(means)
    top_var = np.argsort(-ratios_arr)[:10]
    top_abs = np.argsort(-means_arr)[:5]
    to_label = set(top_var.tolist()) | set(top_abs.tolist())
    for i in to_label:
        ax.annotate(names[i], (means[i], ratios[i]), fontsize=6.5,
                    xytext=(4, 2), textcoords="offset points")

    ax.axhline(1.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5,
               label="1.5x variability")
    ax.axhline(2.0, color="orange", linewidth=0.8, linestyle="--", alpha=0.7,
               label="2.0x variability")
    ax.set_xlabel("Mean absmax across timesteps")
    ax.set_ylabel("Temporal variability (max/min absmax)")
    ax.set_title("Layer Temporal Variability vs Magnitude")
    ax.grid(alpha=0.3)

    legend_items = [Line2D([0], [0], color=c, lw=5, label=s)
                    for s, c in LAYER_COLORS.items()]
    legend_items += [
        Line2D([0], [0], color="gray", lw=1, linestyle="--", label="1.5x"),
        Line2D([0], [0], color="orange", lw=1, linestyle="--", label="2.0x"),
    ]
    ax.legend(handles=legend_items, fontsize=7, ncol=2, loc="upper right")

    fig.tight_layout()
    path = out_dir / "variability_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


# ---------------------------------------------------------------------------
# Plot 6: Raw distribution histograms with shift and quantization levels
# ---------------------------------------------------------------------------

def plot_distribution_raw(timesteps, sigma_map, step_keys, layer_name: str,
                         step_idx: int, out_dir: Path, quant_config: Dict = None):
    """
    Plot raw activation distribution from histogram with:
    - Original distribution
    - Shift line (if post-GELU)
    - Quantization level overlays (A4/A6/A8 boundaries)
    """
    key = str(step_idx)
    if key not in timesteps:
        key = min(step_keys, key=lambda k: abs(int(k) - step_idx))

    # Load histogram data from npz
    with open(timesteps[key]) as f:  # This is wrong, need to fix
        pass
    # Actually need to load from the npz format
    # Let me check the load_data function to see how it's done

    sigma = sigma_map[int(key)]
    # TODO: Complete this after checking load_data pattern


def plot_distribution_with_shift(ts_dir: Path, step_key: str, layer_name: str,
                                 sigma: float, out_dir: Path,
                                 quant_config: Dict = None):
    """
    Plot activation distribution with shift and quantization overlays.

    Args:
        ts_dir: Directory with npz files
        step_key: Timestep key (e.g., "0", "24")
        layer_name: Layer to plot
        sigma: Noise level at this timestep
        out_dir: Output directory
        quant_config: Optional quantization config for level overlays
    """
    npz_path = ts_dir / f"step_{step_key}.npz"
    index_path = ts_dir / f"step_{step_key}_index.json"

    if not npz_path.exists():
        return None

    npz = np.load(npz_path)
    with open(index_path) as f:
        index = json.load(f)

    if layer_name not in index:
        return None

    meta = index[layer_name]
    safe = layer_name.replace(".", "_")

    # Load histogram
    hist_counts = npz[f"{safe}__hist_counts"]
    hist_edges = npz[f"{safe}__hist_edges"]
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

    # Normalize to probability
    hist_prob = hist_counts / hist_counts.sum()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot distribution
    ax.bar(bin_centers, hist_prob, width=np.diff(hist_edges),
           color=layer_color(layer_name), alpha=0.7, edgecolor='none',
           label='Activation distribution')

    # Add shift line if available
    if f"{safe}__shift" in npz:
        shift = npz[f"{safe}__shift"]
        shift_mean = float(shift.mean())
        ax.axvline(shift_mean, color='red', linewidth=2, linestyle='--',
                  label=f'Shift (mean={shift_mean:.2f})')
        # Show shifted distribution
        shifted_centers = bin_centers - shift_mean
        ax.bar(shifted_centers, hist_prob, width=np.diff(hist_edges),
               color='green', alpha=0.3, edgecolor='none',
               label='After shift')

    # Add zero line
    ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)

    # Add quantization levels if config provided
    if quant_config and step_key in quant_config.get("per_timestep", {}):
        layer_cfg = quant_config["per_timestep"][step_key].get(layer_name)
        if layer_cfg:
            bits = layer_cfg["bits"]
            scale = layer_cfg["scale"]
            qmax = 2 ** (bits - 1) - 1

            # Show quantization boundaries
            for i in range(-qmax-1, qmax+2):
                q_val = i * scale
                if hist_edges[0] <= q_val <= hist_edges[-1]:
                    ax.axvline(q_val, color='orange', linewidth=0.5,
                              alpha=0.3, linestyle=':')

            # Label quantization info
            ax.text(0.98, 0.95, f'{bits}-bit quantization\n{2**bits} levels\nscale={scale:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Activation value')
    ax.set_ylabel('Probability density')
    ax.set_title(f'{layer_name} — Step {step_key} (σ={sigma:.3f})')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    safe_name = layer_name.replace(".", "_")
    path = out_dir / f"dist_{safe_name}_step{step_key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


def plot_distribution_temporal_evolution(ts_dir: Path, layer_name: str,
                                        step_keys: list, sigma_map: dict,
                                        out_dir: Path):
    """
    Plot how distribution evolves across timesteps for a single layer.
    Shows multiple histograms overlaid or in subplots.
    """
    # Select representative timesteps (beginning, middle, end)
    selected_steps = [step_keys[0], step_keys[len(step_keys)//2], step_keys[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, step_key in enumerate(selected_steps):
        npz_path = ts_dir / f"step_{step_key}.npz"
        index_path = ts_dir / f"step_{step_key}_index.json"

        if not npz_path.exists():
            continue

        npz = np.load(npz_path)
        with open(index_path) as f:
            index = json.load(f)

        if layer_name not in index:
            continue

        safe = layer_name.replace(".", "_")
        hist_counts = npz[f"{safe}__hist_counts"]
        hist_edges = npz[f"{safe}__hist_edges"]
        bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        hist_prob = hist_counts / hist_counts.sum()

        ax = axes[idx]
        sigma = sigma_map[int(step_key)]

        ax.bar(bin_centers, hist_prob, width=np.diff(hist_edges),
              color=layer_color(layer_name), alpha=0.7, edgecolor='none')

        # Add shift if available
        if f"{safe}__shift" in npz:
            shift_mean = float(npz[f"{safe}__shift"].mean())
            ax.axvline(shift_mean, color='red', linewidth=2, linestyle='--',
                      label=f'Shift={shift_mean:.2f}')

        ax.axvline(0, color='black', linewidth=1, alpha=0.5)
        ax.set_xlabel('Activation value')
        ax.set_title(f'Step {step_key} (σ={sigma:.3f})')
        ax.grid(alpha=0.3, axis='y')
        if idx == 0:
            ax.set_ylabel('Probability density')
        if f"{safe}__shift" in npz:
            ax.legend(fontsize=8)

    fig.suptitle(f'{layer_name} — Distribution Evolution Across Timesteps', fontsize=14)
    fig.tight_layout()

    safe_name = layer_name.replace(".", "_")
    path = out_dir / f"dist_temporal_{safe_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--snapshot-steps", type=int, nargs="+", default=[0, 12, 24, 40, 48],
                        help="Which step indices to snapshot")
    parser.add_argument("--quant-config", type=Path, default=None,
                        help="Optional quant_config.json for overlaying quantization levels")
    parser.add_argument("--plot-distributions", action="store_true",
                        help="Generate raw distribution plots with shift/quant overlays")
    args = parser.parse_args()

    out_dir = args.output_dir or (args.stats.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.stats}")
    with open(args.stats) as f:
        manifest = json.load(f)

    # Get ts_dir for loading npz files
    ts_dir = Path(manifest["timestep_dir"])
    sigma_map = {int(k): float(v) for k, v in manifest["sigma_map"].items()}
    step_keys = manifest["step_keys"]

    # Load old format for existing plots
    timesteps, _, step_keys, layer_names = load_data(args.stats)
    print(f"  {len(step_keys)} timesteps × {len(layer_names)} layers\n")

    # Load quant config if provided
    quant_config = None
    if args.quant_config and args.quant_config.exists():
        with open(args.quant_config) as f:
            quant_config = json.load(f)
        print(f"Loaded quant config: {quant_config.get('format')}\n")

    # --- Snapshots at key steps ---
    print("=== Snapshot plots ===")
    for step in args.snapshot_steps:
        plot_snapshot(timesteps, sigma_map, step_keys, layer_names, step, out_dir)

    # --- Temporal: highest-variability layers ---
    print("\n=== Temporal plots ===")
    # Compute variability
    variability = []
    for name in layer_names:
        vals = [timesteps[k][name]["tensor_absmax"]
                for k in step_keys if name in timesteps[k]]
        if vals:
            variability.append((name, max(vals) / (min(vals) + 1e-6)))
    variability.sort(key=lambda x: -x[1])

    top_variable = [n for n, _ in variability[:16]]
    plot_temporal(timesteps, sigma_map, step_keys, layer_names,
                  top_variable, out_dir, title_suffix="(top 16 variable)")

    # Temporal: most problematic (highest absmax)
    by_absmax = sorted(layer_names,
                       key=lambda n: max(timesteps[k][n]["tensor_absmax"]
                                        for k in step_keys if n in timesteps[k]),
                       reverse=True)
    plot_temporal(timesteps, sigma_map, step_keys, layer_names,
                  by_absmax[:12], out_dir, title_suffix="(top 12 absmax)")

    # Temporal: img MLP fc1 layers (most variable class)
    img_fc1 = [n for n in layer_names if n.endswith("img.mlp.fc1")]
    if img_fc1:
        plot_temporal(timesteps, sigma_map, step_keys, layer_names,
                      img_fc1, out_dir, title_suffix="(img mlp fc1 all blocks)")

    # Temporal: txt MLP layers
    txt_mlp = [n for n in layer_names if "txt.mlp" in n]
    if txt_mlp:
        plot_temporal(timesteps, sigma_map, step_keys, layer_names,
                      txt_mlp, out_dir, title_suffix="(txt mlp all blocks)")

    # Temporal: post-GELU shift values
    pg_layers = [n for n in layer_names if n.endswith(".mlp.fc2")]
    if pg_layers:
        # Split into img and txt for readability
        img_pg = [n for n in pg_layers if ".img." in n]
        txt_pg = [n for n in pg_layers if ".txt." in n]
        if img_pg:
            plot_temporal(timesteps, sigma_map, step_keys, layer_names,
                          img_pg, out_dir, use_shift=True,
                          title_suffix="(img fc2 shift)")
        if txt_pg:
            plot_temporal(timesteps, sigma_map, step_keys, layer_names,
                          txt_pg, out_dir, use_shift=True,
                          title_suffix="(txt fc2 shift)")

    # --- Heatmaps ---
    print("\n=== Heatmaps ===")
    plot_heatmap(timesteps, sigma_map, step_keys, layer_names, out_dir,
                 sort_by="variability", max_layers=120)
    plot_heatmap(timesteps, sigma_map, step_keys, layer_names, out_dir,
                 sort_by="absmax", max_layers=60)

    # --- Per-channel distributions for interesting layers ---
    print("\n=== Per-channel distributions ===")
    interesting = [
        ("mm22.txt.mlp.fc1", 24),   # extreme spike case
        ("mm1.txt.mlp.fc2",  24),   # bimodal post-GELU
        ("mm9.img.mlp.fc1",  16),   # most temporally variable
        ("mm0.img.attn.q_proj", 0), # stable reference
        ("mm12.img.attn.q_proj", 24), # increasing late-step
    ]
    for layer_name, step in interesting:
        if layer_name in layer_names:
            plot_channel_dist(timesteps, sigma_map, step_keys, layer_name, step, out_dir)

    # --- Variability scatter ---
    print("\n=== Variability scatter ===")
    plot_variability_scatter(timesteps, sigma_map, step_keys, layer_names, out_dir)

    # --- Raw distribution plots with shift and quantization overlays ---
    if args.plot_distributions:
        print("\n=== Raw distribution plots ===")

        # Select interesting layers for distribution visualization
        dist_layers = [
            "mm1.txt.mlp.fc2",   # Post-GELU with shift, early block
            "mm15.txt.mlp.fc2",  # Post-GELU mid block
            "mm22.txt.mlp.fc1",  # Extreme outlier case
            "mm7.img.mlp.fc1",   # High variability
            "mm12.img.attn.q_proj",  # Stable attention layer
        ]

        # Plot single-timestep distributions at key points
        for layer_name in dist_layers:
            if layer_name not in layer_names:
                continue
            for step_key in ["0", "12", "24"]:
                if step_key in step_keys:
                    plot_distribution_with_shift(
                        ts_dir, step_key, layer_name,
                        sigma_map[int(step_key)], out_dir, quant_config
                    )

        # Plot temporal evolution for selected layers
        for layer_name in dist_layers:
            if layer_name in layer_names:
                plot_distribution_temporal_evolution(
                    ts_dir, layer_name, step_keys, sigma_map, out_dir
                )

    print(f"\n✓ All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
