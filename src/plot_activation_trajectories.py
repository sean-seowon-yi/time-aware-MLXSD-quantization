"""
Plot activation percentile trajectories across the denoising schedule.

Generates per-block figures showing how p99, p999, and absmax (p100) change
across noise levels (σ), plus the absmax/p999 ratio and per-channel spread.

Layout: layers as columns, 3 panel rows (percentiles, tail ratio, channel spread).
Img and txt streams get separate figures for readability.

Usage:
    python -m src.plot_activation_trajectories \
        --stats calibration_data_512/activations \
        --output calibration_data_512/activation_trajectory_plots
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_sigma_map(stats_dir: Path) -> Dict[int, float]:
    with open(stats_dir / "layer_statistics.json") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["sigma_map"].items()}


def load_step_data(stats_dir: Path, step: int) -> Tuple[Dict, Dict]:
    timestep_dir = stats_dir / "timestep_stats"
    with open(timestep_dir / f"step_{step}_index.json") as f:
        index_data = json.load(f)
    npz_data = np.load(timestep_dir / f"step_{step}.npz")
    return index_data, npz_data


def get_block_from_layer(layer_key: str) -> str:
    return layer_key.split('.')[0]


def get_layer_short_name(layer_key: str) -> str:
    parts = layer_key.split('.')
    return '.'.join(parts[1:])


def get_stream(layer_key: str) -> str:
    parts = layer_key.split('.')
    return parts[1] if len(parts) > 1 else 'unknown'


def compute_channel_stats(npz_data, layer_key: str):
    layer_raw = layer_key.replace('.', '_')
    avg_max_key = f"{layer_raw}__avg_max"
    avg_min_key = f"{layer_raw}__avg_min"
    if avg_max_key not in npz_data or avg_min_key not in npz_data:
        return None, None
    avg_max = npz_data[avg_max_key]
    avg_min = npz_data[avg_min_key]
    per_channel_absmax = np.maximum(np.abs(avg_max), np.abs(avg_min))
    return float(np.mean(per_channel_absmax)), float(np.std(per_channel_absmax))


def group_layers_by_block(layers: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for layer in sorted(layers):
        groups[get_block_from_layer(layer)].append(layer)
    return groups


def _natural_block_sort(block_name: str) -> int:
    """Extract numeric index for natural sort: mm0 -> 0, mm23 -> 23."""
    return int(block_name.replace('mm', ''))


def plot_stream_figure(
    block_name: str,
    stream: str,
    layers: List[str],
    trajectories: Dict[str, Dict],
    sigmas: List[float],
    output_path: Path,
):
    n_layers = len(layers)
    if n_layers == 0:
        return

    # Layout: 3 rows (percentiles, ratio, channel spread) × N columns (layers)
    col_width = 4.0
    row_height = 3.5
    fig, axes = plt.subplots(
        3, n_layers,
        figsize=(col_width * n_layers, row_height * 3),
        squeeze=False,
    )

    # Sigma in denoising order: high σ (noisy) on left, low σ (clean) on right
    sigmas_arr = np.array(sigmas)

    for col, layer_key in enumerate(layers):
        short_name = get_layer_short_name(layer_key)
        # Strip stream prefix for column title (e.g. "img.attn.q_proj" -> "attn.q_proj")
        col_label = '.'.join(short_name.split('.')[1:])
        traj = trajectories[layer_key]

        p99 = np.array(traj['p99'])
        p999 = np.array(traj['p999'])
        absmax = np.array(traj['absmax'])
        ch_mean = np.array(traj['channel_mean']) if traj['channel_mean'] else None
        ch_std = np.array(traj['channel_std']) if traj['channel_std'] else None

        # --- Row 0: Absolute percentiles ---
        ax = axes[0, col]
        ax.plot(sigmas_arr, p99, 'b-o', label='p99', linewidth=1.8, markersize=3)
        ax.plot(sigmas_arr, p999, '-o', color='orange', label='p999', linewidth=1.8, markersize=3)
        ax.plot(sigmas_arr, absmax, 'r--s', label='absmax', linewidth=1.8, markersize=3)
        ax.set_title(col_label, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        if col == 0:
            ax.set_ylabel('Activation Range', fontsize=10)

        # --- Row 1: Tail ratio ---
        ax = axes[1, col]
        ratio = absmax / np.maximum(p999, 1e-6)
        ax.plot(sigmas_arr, ratio, 'g-o', linewidth=1.8, markersize=3, label='absmax/p999')
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(1.5, color='red', linestyle='--', alpha=0.7, linewidth=1, label='1.5×')
        y_top = max(np.max(ratio) * 1.15, 2.0)
        ax.fill_between(sigmas_arr, 1.5, y_top, color='red', alpha=0.08)
        ax.set_ylim(bottom=0.8, top=y_top)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        if col == 0:
            ax.set_ylabel('Tail Ratio', fontsize=10)

        # --- Row 2: Per-channel spread ---
        ax = axes[2, col]
        if ch_mean is not None:
            ax.plot(sigmas_arr, ch_mean, '-o', color='purple', linewidth=1.8, markersize=3, label='mean')
            ax.fill_between(
                sigmas_arr,
                np.maximum(0, ch_mean - ch_std),
                ch_mean + ch_std,
                color='purple', alpha=0.2, label='± 1σ',
            )
            # Also overlay p999 and absmax as thin lines for reference
            ax.plot(sigmas_arr, p999, '-', color='orange', linewidth=0.8, alpha=0.5, label='p999')
            ax.plot(sigmas_arr, absmax, '--', color='red', linewidth=0.8, alpha=0.5, label='absmax')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.set_xlabel('σ', fontsize=10)
        if col == 0:
            ax.set_ylabel('Channel Spread', fontsize=10)

    stream_label = 'Image Stream' if stream == 'img' else 'Text Stream'
    fig.suptitle(
        f'{block_name} — {stream_label}',
        fontsize=14, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats', type=Path, required=True,
                        help='Path to activations directory')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output directory for PNG figures')
    args = parser.parse_args()

    stats_dir = Path(args.stats)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sigma map...")
    sigma_map = load_sigma_map(stats_dir)
    steps_sorted = sorted(sigma_map.keys())
    sigmas = [sigma_map[s] for s in steps_sorted]
    print(f"Found {len(steps_sorted)} timesteps: σ = {sigmas[0]:.3f} → {sigmas[-1]:.3f}")

    print("Loading activation data across all timesteps...")
    all_trajectories: Dict[str, Dict] = defaultdict(lambda: {
        'p99': [], 'p999': [], 'absmax': [],
        'channel_mean': [], 'channel_std': [],
    })
    all_layers = None

    for step in steps_sorted:
        index_data, npz_data = load_step_data(stats_dir, step)
        if all_layers is None:
            all_layers = list(index_data.keys())
            print(f"  {len(all_layers)} layers")

        for layer_key in all_layers:
            if layer_key not in index_data:
                continue
            s = index_data[layer_key]
            all_trajectories[layer_key]['p99'].append(s.get('hist_p99', 0.0))
            all_trajectories[layer_key]['p999'].append(s.get('hist_p999', 0.0))
            all_trajectories[layer_key]['absmax'].append(s.get('tensor_absmax', 0.0))

            mean_val, std_val = compute_channel_stats(npz_data, layer_key)
            if mean_val is not None:
                all_trajectories[layer_key]['channel_mean'].append(mean_val)
                all_trajectories[layer_key]['channel_std'].append(std_val)

    # Group by block and stream, then plot
    print("\nGenerating figures...")
    blocks = group_layers_by_block(all_layers)

    for block_name in sorted(blocks.keys(), key=_natural_block_sort):
        block_layers = blocks[block_name]
        img_layers = [l for l in block_layers if get_stream(l) == 'img']
        txt_layers = [l for l in block_layers if get_stream(l) == 'txt']

        if img_layers:
            plot_stream_figure(
                block_name, 'img', img_layers, all_trajectories, sigmas,
                output_dir / f"{block_name}_img_trajectories.png",
            )
        if txt_layers:
            plot_stream_figure(
                block_name, 'txt', txt_layers, all_trajectories, sigmas,
                output_dir / f"{block_name}_txt_trajectories.png",
            )

    n_figs = sum(1 for _ in output_dir.glob("*.png"))
    print(f"\nDone! {n_figs} figures in {output_dir}")


if __name__ == '__main__':
    main()
