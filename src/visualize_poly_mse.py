"""
Visualize polynomial schedule fit quality (MSE) per block for selected layer types.

Plots a line chart where each line is a layer type (e.g. img_attn_q_proj, txt_mlp_fc2),
x-axis is block index, y-axis is MSE between the polynomial prediction and actual
calibration data.

Usage:
    python -m src.visualize_poly_mse \
        --schedule polynomial_clipping_schedule.json \
        --activations-dir calibration_data_100/activations \
        --layers img_attn_q_proj txt_mlp_fc1 txt_mlp_fc2 \
        --output poly_mse.png

    # View all layer types:
    python -m src.visualize_poly_mse \
        --schedule polynomial_clipping_schedule.json \
        --activations-dir calibration_data_100/activations \
        --output poly_mse.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALL_LAYER_TYPES = [
    "img_attn_q_proj", "img_attn_k_proj", "img_attn_v_proj", "img_attn_o_proj",
    "img_mlp_fc1", "img_mlp_fc2",
    "txt_attn_q_proj", "txt_attn_k_proj", "txt_attn_v_proj", "txt_attn_o_proj",
    "txt_mlp_fc1", "txt_mlp_fc2",
]


def load_actual_trajectories(
    activations_dir: Path,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load per-layer absmax trajectories from collected activation stats.

    Returns (trajectories, sigmas) where trajectories maps
    layer_key (underscore format) -> array of absmax values per step.
    """
    meta_path = activations_dir / "layer_statistics.json"
    with open(meta_path) as f:
        meta = json.load(f)

    ts_dir = activations_dir / "timestep_stats"
    step_keys = sorted(meta["step_keys"], key=int)
    sigma_map = {k: float(v) for k, v in meta["sigma_map"].items()}
    sigmas = np.array([sigma_map[s] for s in step_keys])

    step_indices = {}
    for s in step_keys:
        with open(ts_dir / f"step_{s}_index.json") as f:
            step_indices[s] = json.load(f)

    all_layers_dot = sorted(step_indices[step_keys[0]].keys())

    trajectories = {}
    for layer_dot in all_layers_dot:
        vals = []
        for s in step_keys:
            info = step_indices[s].get(layer_dot, {})
            v = info.get("tensor_absmax")
            vals.append(float(v) if v is not None else np.nan)
        layer_raw = layer_dot.replace(".", "_")
        trajectories[layer_raw] = np.array(vals)

    return trajectories, sigmas


def compute_mse_per_block(
    schedule: dict,
    trajectories: Dict[str, np.ndarray],
    sigmas: np.ndarray,
    layer_types: List[str],
) -> Dict[str, Dict[int, float]]:
    """Compute MSE between poly prediction and actual data for each (layer_type, block).

    Returns dict: layer_type -> {block_idx: mse}.
    """
    results = {}
    for lt in layer_types:
        block_mse = {}
        for block_idx in range(24):
            key = f"mm{block_idx}_{lt}"
            if key not in schedule["layers"]:
                continue
            if key not in trajectories:
                continue

            coeffs = schedule["layers"][key]["coeffs"]
            actual = trajectories[key]
            predicted = np.polyval(coeffs, sigmas)

            mask = ~np.isnan(actual)
            if mask.sum() == 0:
                continue
            mse = float(np.mean((actual[mask] - predicted[mask]) ** 2))
            block_mse[block_idx] = mse

        if block_mse:
            results[lt] = block_mse

    return results


def plot_mse(
    mse_data: Dict[str, Dict[int, float]],
    layer_types: List[str],
    output_path: Path,
    title: str = "Polynomial Fit MSE by Block",
):
    """Plot MSE line chart: one line per layer type, x=block, y=MSE."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(layer_types)))
    markers = ["o", "s", "^", "v", "D", "P", "X", "p", "h", "*", "+", "x"]

    for i, lt in enumerate(layer_types):
        if lt not in mse_data:
            continue
        block_mse = mse_data[lt]
        blocks = sorted(block_mse.keys())
        mses = [block_mse[b] for b in blocks]

        # Shorten label for readability
        label = lt.replace("attn_", "").replace("mlp_", "")
        ax.plot(
            blocks, mses,
            color=colors[i],
            marker=markers[i % len(markers)],
            markersize=4,
            linewidth=1.5,
            label=label,
        )

    ax.set_xlabel("Block Index", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(24))
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_layer_fit(
    block_idx: int,
    layer_type: str,
    schedule_path: str = "polynomial_clipping_schedule.json",
    activations_dir: str = "calibration_data_100/activations",
    output_path: Optional[str] = None,
):
    """Plot actual data points vs polynomial fit for a single (block, layer_type).

    Shows the raw calibration data as scatter points and the polynomial curve,
    with per-point residuals, fit metadata (degree, R², MSE), and residual subplot.

    Args:
        block_idx: Block index (0-23)
        layer_type: e.g. "img_attn_q_proj", "txt_mlp_fc2"
        schedule_path: Path to polynomial_clipping_schedule.json
        activations_dir: Path to activations directory
        output_path: Output PNG path. If None, defaults to
                     "poly_fit_mm{block_idx}_{layer_type}.png"
    """
    with open(schedule_path) as f:
        schedule = json.load(f)

    trajectories, sigmas = load_actual_trajectories(Path(activations_dir))

    key = f"mm{block_idx}_{layer_type}"
    if key not in schedule["layers"]:
        print(f"Layer '{key}' not found in schedule. Available layers with block {block_idx}:")
        for k in sorted(schedule["layers"].keys()):
            if k.startswith(f"mm{block_idx}_"):
                print(f"  {k}")
        return
    if key not in trajectories:
        print(f"Layer '{key}' not found in activation data.")
        return

    info = schedule["layers"][key]
    coeffs = info["coeffs"]
    degree = info["degree"]
    r2 = info["r2"]

    actual = trajectories[key]
    mask = ~np.isnan(actual)
    sigmas_valid = sigmas[mask]
    actual_valid = actual[mask]

    # Dense curve for the polynomial
    sigma_fine = np.linspace(float(sigmas.min()), float(sigmas.max()), 200)
    predicted_fine = np.polyval(coeffs, sigma_fine)
    predicted_at_data = np.polyval(coeffs, sigmas_valid)

    residuals = actual_valid - predicted_at_data
    mse = float(np.mean(residuals ** 2))

    # Plot
    if output_path is None:
        output_path = f"poly_fit_mm{block_idx}_{layer_type}.png"

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(10, 7), height_ratios=[3, 1], sharex=True,
    )

    # Main plot: data points + polynomial curve
    ax_main.scatter(
        sigmas_valid, actual_valid,
        color="steelblue", s=40, zorder=3, label="Calibration data", edgecolors="white",
        linewidths=0.5,
    )
    ax_main.plot(
        sigma_fine, predicted_fine,
        color="crimson", linewidth=2, label=f"Poly fit (degree {degree})",
    )

    # Annotate
    coeffs_str = ", ".join(f"{c:.4f}" for c in coeffs)
    ax_main.set_title(
        f"{key}\n"
        f"degree={degree}  R²={r2:.4f}  MSE={mse:.6f}  coeffs=[{coeffs_str}]",
        fontsize=11,
    )
    ax_main.set_ylabel("Activation absmax", fontsize=11)
    ax_main.legend(fontsize=9)
    ax_main.grid(True, linestyle="--", alpha=0.4)

    # Residual subplot
    ax_res.stem(
        sigmas_valid, residuals,
        linefmt="gray", markerfmt="o", basefmt="k-",
    )
    ax_res.axhline(0, color="black", linewidth=0.8)
    ax_res.set_xlabel("σ (noise level)", fontsize=11)
    ax_res.set_ylabel("Residual", fontsize=10)
    ax_res.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_layer_fits(
    layers: List[Tuple[int, str]],
    schedule_path: str = "polynomial_clipping_schedule.json",
    activations_dir: str = "calibration_data_100/activations",
    output_path: str = "poly_fits_diagnostic.png",
):
    """Plot actual vs polynomial fit for multiple (block_idx, layer_type) pairs in a grid.

    Args:
        layers: List of (block_idx, layer_type) tuples,
                e.g. [(5, "txt_mlp_fc2"), (12, "img_attn_o_proj")]
        schedule_path: Path to polynomial_clipping_schedule.json
        activations_dir: Path to activations directory
        output_path: Output PNG path
    """
    with open(schedule_path) as f:
        schedule = json.load(f)

    trajectories, sigmas = load_actual_trajectories(Path(activations_dir))

    sigma_fine = np.linspace(float(sigmas.min()), float(sigmas.max()), 200)

    n = len(layers)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).ravel()

    for i, (block_idx, layer_type) in enumerate(layers):
        ax = axes_flat[i]
        key = f"mm{block_idx}_{layer_type}"

        if key not in schedule["layers"] or key not in trajectories:
            ax.set_title(f"{key}\nNOT FOUND", fontsize=9, color="red")
            ax.set_visible(True)
            continue

        info = schedule["layers"][key]
        coeffs = info["coeffs"]
        degree = info["degree"]
        r2 = info["r2"]

        actual = trajectories[key]
        mask = ~np.isnan(actual)
        sigmas_valid = sigmas[mask]
        actual_valid = actual[mask]

        predicted_fine = np.polyval(coeffs, sigma_fine)
        predicted_at_data = np.polyval(coeffs, sigmas_valid)
        mse = float(np.mean((actual_valid - predicted_at_data) ** 2))

        ax.scatter(sigmas_valid, actual_valid, color="steelblue", s=30, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.plot(sigma_fine, predicted_fine, color="crimson", linewidth=1.5)

        ax.set_title(f"{key}\ndeg={degree}  R²={r2:.3f}  MSE={mse:.4f}", fontsize=9)
        ax.set_xlabel("σ", fontsize=9)
        ax.set_ylabel("absmax", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Polynomial Fit Diagnostics", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_poly_mse(
    schedule_path: str = "polynomial_clipping_schedule.json",
    activations_dir: str = "calibration_data_100/activations",
    layer_types: Optional[List[str]] = None,
    output_path: str = "poly_mse.png",
    title: str = "Polynomial Fit MSE by Block",
):
    """Main function: load data, compute MSE, plot.

    Args:
        schedule_path: Path to polynomial_clipping_schedule.json
        activations_dir: Path to activations directory
        layer_types: List of layer types to plot, e.g. ["img_attn_q_proj", "txt_mlp_fc2"].
                     If None, plots all 12 types.
        output_path: Output PNG path
        title: Plot title
    """
    with open(schedule_path) as f:
        schedule = json.load(f)

    trajectories, sigmas = load_actual_trajectories(Path(activations_dir))

    if layer_types is None:
        layer_types = ALL_LAYER_TYPES

    mse_data = compute_mse_per_block(schedule, trajectories, sigmas, layer_types)

    if not mse_data:
        print("No matching layers found. Available types:", sorted(
            {k.split("_", 2)[2] for k in schedule["layers"].keys()}
        ))
        return

    plot_mse(mse_data, layer_types, Path(output_path), title=title)

    # Print summary table
    print(f"\n{'Layer Type':<25} {'Mean MSE':>10} {'Max MSE':>10} {'Max Block':>10}")
    print("-" * 57)
    for lt in layer_types:
        if lt not in mse_data:
            continue
        mses = list(mse_data[lt].values())
        blocks = list(mse_data[lt].keys())
        max_idx = int(np.argmax(mses))
        label = lt.replace("attn_", "").replace("mlp_", "")
        print(f"{label:<25} {np.mean(mses):>10.4f} {np.max(mses):>10.4f} {blocks[max_idx]:>10}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schedule", type=str,
                        default="polynomial_clipping_schedule.json")
    parser.add_argument("--activations-dir", type=str,
                        default="calibration_data_100/activations")
    parser.add_argument("--layers", nargs="+", default=None,
                        help=f"Layer types to plot. Options: {ALL_LAYER_TYPES}")
    parser.add_argument("--output", type=str, default="poly_mse.png")
    parser.add_argument("--title", type=str,
                        default="Polynomial Fit MSE by Block")
    args = parser.parse_args()

    plot_poly_mse(
        schedule_path=args.schedule,
        activations_dir=args.activations_dir,
        layer_types=args.layers,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
