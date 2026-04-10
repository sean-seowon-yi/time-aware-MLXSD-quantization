"""Visualize polynomial clipping fits against activation trajectories.

Generates per-layer plots showing:
- Pre-CSB per-tensor absmax trajectory (raw Phase 1 activations)
- Post-CSB per-tensor absmax trajectory (what the quantized weight sees)
- Fitted polynomial α(σ) clipping bound
- Reference horizontal line: Phase 2 static α = 127·scale from ``static_scales.npz``
  when present, else max post-CSB absmax over timesteps

Also generates summary grid and comparison plots.

Usage
-----
# Plot representative layers
python -m src.phase3.visualize_poly \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \
    --output-dir plots/poly_clipping

# Plot specific layers
python -m src.phase3.visualize_poly \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \
    --layers "blocks.12.image.mlp.fc2" "blocks.14.image.attn.q_proj" \
    --output-dir plots/poly_clipping

# Plot all layers with degree > 0
python -m src.phase3.visualize_poly \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \
    --non-static-only \
    --output-dir plots/poly_clipping
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _eval_poly(coeffs: list[float], sigma: np.ndarray) -> np.ndarray:
    """Evaluate polynomial with ascending-power coefficients."""
    return np.polynomial.polynomial.polyval(sigma, coeffs)


def _load_phase2_static_alphas(calibration_dir: Path) -> dict[str, float]:
    """Map layer name → static A8 clipping bound α = 127·scale from checkpoint.

    Reads ``static_scales.npz`` only (no MLX import — matches
    ``phase2.quantize_static.STATIC_SCALES_FILENAME``).
    """
    path = calibration_dir / "static_scales.npz"
    if not path.exists():
        return {}
    raw = np.load(path)
    out: dict[str, float] = {}
    for k in raw.files:
        arr = np.asarray(raw[k], dtype=np.float64).ravel()
        out[k] = float(np.max(arr) * 127.0)
    return out


def plot_layer_poly_fit(
    name: str,
    sigmas: np.ndarray,
    pre_csb_absmax: np.ndarray,
    post_csb_absmax: np.ndarray,
    poly_coeffs: list[float],
    degree: int,
    r2: float,
    reference_alpha: float | None = None,
    reference_label: str | None = None,
    output_path: Path | None = None,
    post_csb_channels: np.ndarray | None = None,
) -> None:
    """Plot a single layer's activation trajectory with poly fit overlay.

    Parameters
    ----------
    name : str
        Layer name (e.g. ``blocks.12.image.mlp.fc2``).
    sigmas : ndarray [T]
        Sigma values (noise levels) for each timestep.
    pre_csb_absmax : ndarray [T]
        Per-tensor max|x| before CSB balancing.
    post_csb_absmax : ndarray [T]
        Per-tensor max|x / b| after CSB balancing.
    poly_coeffs : list[float]
        Polynomial coefficients in ascending-power order.
    degree : int
        Polynomial degree.
    r2 : float
        R² of the fit.
    reference_alpha : float or None
        Constant clipping bound α (same units as poly α(σ)).
    reference_label : str or None
        Legend text for the horizontal reference line.
    output_path : Path or None
        Save path. If None, plt.show().
    post_csb_channels : ndarray [T, d_in] or None
        If provided, draws faint per-channel trajectories.
    """
    if not HAS_MPL:
        logger.warning("matplotlib not available — skipping plot for %s", name)
        return

    sigma_fine = np.linspace(float(sigmas.min()), float(sigmas.max()), 200)
    poly_vals = np.maximum(_eval_poly(poly_coeffs, sigma_fine), 0.01)
    poly_at_data = np.maximum(_eval_poly(poly_coeffs, sigmas), 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False)

    # --- Left panel: Pre-CSB vs Post-CSB ---
    ax = axes[0]
    ax.plot(sigmas, pre_csb_absmax, "o-", color="#2196F3", ms=4, lw=1.5,
            label="Pre-CSB  max|x|", alpha=0.9)
    ax.plot(sigmas, post_csb_absmax, "s-", color="#E91E63", ms=4, lw=1.5,
            label="Post-CSB  max|x/b|", alpha=0.9)

    ax.set_xlabel("σ (noise level)", fontsize=11)
    ax.set_ylabel("Activation magnitude", fontsize=11)
    ax.set_title("CSB Effect on Activations", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ratio = pre_csb_absmax / np.maximum(post_csb_absmax, 1e-12)
    ax2 = ax.twinx()
    ax2.fill_between(sigmas, ratio, alpha=0.08, color="gray")
    ax2.plot(sigmas, ratio, "--", color="gray", lw=1, alpha=0.5, label="Ratio pre/post")
    ax2.set_ylabel("Pre/Post ratio", fontsize=9, color="gray")
    ax2.tick_params(axis="y", colors="gray")

    # --- Right panel: Post-CSB + Poly Fit ---
    ax = axes[1]

    if post_csb_channels is not None:
        n_ch = post_csb_channels.shape[1]
        step = max(1, n_ch // 50)
        for c in range(0, n_ch, step):
            ax.plot(sigmas, post_csb_channels[:, c], color="#E0E0E0", lw=0.3,
                    alpha=0.4, zorder=1)

    ax.plot(sigmas, post_csb_absmax, "s", color="#E91E63", ms=5, zorder=3,
            label="Post-CSB  max|x/b|")

    if reference_alpha is not None:
        lbl = reference_label or f"α ref = {reference_alpha:.2f}"
        ax.axhline(reference_alpha, color="#9E9E9E", ls="--", lw=2, alpha=0.7,
                   label=lbl, zorder=2)

    poly_label = f"Poly deg {degree}  (R²={r2:.3f})"
    ax.plot(sigma_fine, poly_vals, "-", color="#4CAF50", lw=2.5,
            label=poly_label, zorder=4)

    residual = post_csb_absmax - poly_at_data
    under = residual > 0
    if np.any(under):
        ax.scatter(sigmas[under], post_csb_absmax[under], marker="^",
                   color="red", s=40, zorder=5, label="Under-clipped points")

    ax.fill_between(sigma_fine, poly_vals, alpha=0.08, color="#4CAF50")

    ax.set_xlabel("σ (noise level)", fontsize=11)
    ax.set_ylabel("Activation magnitude", fontsize=11)
    ax.set_title("Polynomial Clipping Fit", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    short_name = name.replace("blocks.", "B").replace(".image.", ".I.").replace(".text.", ".T.")
    fig.suptitle(short_name, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", output_path)
    else:
        plt.show()


def plot_summary_grid(
    layer_data: list[dict],
    output_path: Path | None = None,
    max_cols: int = 4,
) -> None:
    """Plot a grid of poly fits for multiple layers.

    Parameters
    ----------
    layer_data : list of dict
        Each dict has keys: name, sigmas, post_csb_absmax, poly_coeffs,
        degree, r2, reference_alpha (optional).
    """
    if not HAS_MPL:
        return

    n = len(layer_data)
    if n == 0:
        return

    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for i, entry in enumerate(layer_data):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sigmas = entry["sigmas"]
        post = entry["post_csb_absmax"]
        coeffs = entry["poly_coeffs"]
        sigma_fine = np.linspace(float(sigmas.min()), float(sigmas.max()), 200)
        poly_vals = np.maximum(_eval_poly(coeffs, sigma_fine), 0.01)

        ax.plot(sigmas, post, "s", color="#E91E63", ms=3)
        ax.plot(sigma_fine, poly_vals, "-", color="#4CAF50", lw=2)

        if entry.get("reference_alpha") is not None:
            ax.axhline(entry["reference_alpha"], color="#9E9E9E", ls="--",
                       lw=1.5, alpha=0.6)

        short = entry["name"].replace("blocks.", "B").replace(
            ".image.", ".I.").replace(".text.", ".T.")
        ax.set_title(f"{short}\ndeg={entry['degree']}  R²={entry['r2']:.3f}",
                     fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Polynomial Clipping Fits — Summary Grid", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved grid to %s", output_path)
    else:
        plt.show()


def plot_wasted_range_comparison(
    layer_data: list[dict],
    output_path: Path | None = None,
) -> None:
    """Bar chart showing int8 utilization: static vs polynomial.

    For each layer, computes the mean fraction of the int8 range actually
    used across timesteps under each scheme.
    """
    if not HAS_MPL:
        return
    if not layer_data:
        return

    names = []
    static_util = []
    poly_util = []

    for entry in layer_data:
        sigmas = entry["sigmas"]
        post = entry["post_csb_absmax"]
        coeffs = entry["poly_coeffs"]
        ref_a = entry.get("reference_alpha")
        if ref_a is None or ref_a < 1e-8:
            continue

        poly_at_data = _eval_poly(coeffs, sigmas)
        # Match inference floor on α (see quantize_poly / phase4_1 get_poly_alpha).
        poly_at_data = np.maximum(poly_at_data, 0.01)

        util_static = post / ref_a
        util_poly = post / poly_at_data

        names.append(
            entry["name"].replace("blocks.", "B")
            .replace(".image.", ".I.")
            .replace(".text.", ".T.")
        )
        static_util.append(float(np.mean(util_static)))
        poly_util.append(float(np.mean(np.minimum(util_poly, 1.0))))

    if not names:
        return

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 5))
    ax.bar(x - w / 2, static_util, w, color="#9E9E9E", alpha=0.8,
           label="Static (fixed bound)")
    ax.bar(x + w / 2, poly_util, w, color="#4CAF50", alpha=0.8,
           label="Poly (σ-adaptive)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Mean int8 utilization (actual / bound)", fontsize=10)
    ax.set_title("Int8 Range Utilization: Static vs Polynomial Clipping",
                 fontsize=12, fontweight="bold")
    ax.axhline(1.0, color="black", ls=":", lw=0.8)
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved utilization comparison to %s", output_path)
    else:
        plt.show()


def generate_all_plots(
    diagnostics_dir: Path,
    calibration_dir: Path,
    output_dir: Path,
    layer_names: list[str] | None = None,
    non_static_only: bool = False,
    max_layers: int = 50,
) -> None:
    """Load data and generate all poly-clipping visualization plots.

    Parameters
    ----------
    diagnostics_dir : Path
        Phase 1 output (``diagnostics/``).
    calibration_dir : Path
        Phase 2 quantized output containing ``calibration.npz``,
        ``calibration_meta.json``, and ``poly_schedule.json``.
    output_dir : Path
        Directory for output PNGs.
    layer_names : list[str] or None
        Specific layers to plot. None = auto-select.
    non_static_only : bool
        If True, only plot layers with degree > 0.
    max_layers : int
        Maximum number of individual layer plots.
    """
    from .poly_clipping import (
        POLY_SCHEDULE_FILENAME,
        _load_calibration_data,
        _load_sigma_values,
    )

    sigmas = _load_sigma_values(diagnostics_dir)
    b_vectors, _ = _load_calibration_data(calibration_dir)
    act_dir = diagnostics_dir / "activation_stats"

    schedule_path = calibration_dir / POLY_SCHEDULE_FILENAME
    if not schedule_path.exists():
        raise FileNotFoundError(
            f"Missing {schedule_path} — run generate_schedule first"
        )
    with open(schedule_path) as f:
        schedule = json.load(f)
    layers_schedule = schedule["layers"]
    static_alpha_by_layer = _load_phase2_static_alphas(calibration_dir)

    if layer_names is not None:
        target_names = layer_names
    elif non_static_only:
        target_names = [
            k for k, v in layers_schedule.items() if v["degree"] > 0
        ]
    else:
        non_static = [
            k for k, v in layers_schedule.items() if v["degree"] > 0
        ]
        static_sample = [
            k for k, v in layers_schedule.items() if v["degree"] == 0
        ][:5]
        target_names = non_static + static_sample

    if len(target_names) > max_layers:
        logger.info(
            "Limiting to %d layers (from %d)", max_layers, len(target_names)
        )
        target_names = target_names[:max_layers]

    logger.info("Generating plots for %d layers", len(target_names))

    grid_data = []

    for name in target_names:
        if name not in layers_schedule:
            logger.warning("%s not in poly schedule — skipping", name)
            continue
        if name not in b_vectors:
            logger.warning("%s not in calibration — skipping", name)
            continue

        npz_path = act_dir / f"{name}.npz"
        if not npz_path.exists():
            logger.warning("No activation stats for %s — skipping", name)
            continue

        data = np.load(npz_path)
        act_traj = data["act_channel_max"]  # [T, d_in]
        b = b_vectors[name]

        pre_csb_absmax = act_traj.max(axis=1)  # [T]

        b_safe = np.maximum(b, 1e-12)
        post_csb = act_traj / b_safe[np.newaxis, :]  # [T, d_in]
        post_csb_absmax = post_csb.max(axis=1)  # [T]

        entry = layers_schedule[name]
        coeffs = entry["coeffs"]
        degree = entry["degree"]
        r2 = entry.get("r2", 0.0)

        if entry.get("granularity") == "per_channel":
            logger.info(
                "Skipping per-channel layer %s (needs multi-curve plot — not yet)",
                name,
            )
            continue
        plot_coeffs = coeffs

        worst_post = float(np.max(post_csb_absmax))
        if name in static_alpha_by_layer:
            ref_a = static_alpha_by_layer[name]
            ref_lbl = f"Phase 2 static α = {ref_a:.2f}"
        else:
            ref_a = worst_post
            ref_lbl = f"maxₜ post-CSB absmax = {ref_a:.2f}"

        safe_name = name.replace(".", "_")
        plot_layer_poly_fit(
            name=name,
            sigmas=sigmas,
            pre_csb_absmax=pre_csb_absmax,
            post_csb_absmax=post_csb_absmax,
            poly_coeffs=plot_coeffs,
            degree=degree,
            r2=r2,
            reference_alpha=ref_a,
            reference_label=ref_lbl,
            output_path=output_dir / f"{safe_name}.png",
            post_csb_channels=post_csb,
        )

        grid_data.append({
            "name": name,
            "sigmas": sigmas,
            "post_csb_absmax": post_csb_absmax,
            "poly_coeffs": plot_coeffs,
            "degree": degree,
            "r2": r2,
            "reference_alpha": ref_a,
        })

    if grid_data:
        plot_summary_grid(grid_data, output_dir / "summary_grid.png")
        plot_wasted_range_comparison(
            grid_data, output_dir / "utilization_comparison.png"
        )

    logger.info(
        "Done — %d individual plots + grid + utilization chart in %s",
        len(grid_data), output_dir,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Visualize polynomial clipping fits vs activation trajectories",
    )
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=Path("diagnostics"),
        help="Phase 1 diagnostics directory",
    )
    parser.add_argument(
        "--calibration-dir", type=Path, required=True,
        help="Phase 2 quantized output with calibration + poly_schedule.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots/poly_clipping"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--layers", type=str, nargs="*", default=None,
        help="Specific layer names to plot (default: auto-select)",
    )
    parser.add_argument(
        "--non-static-only", action="store_true",
        help="Only plot layers with polynomial degree > 0",
    )
    parser.add_argument(
        "--max-layers", type=int, default=50,
        help="Max number of individual layer plots (default: 50)",
    )
    args = parser.parse_args()

    if not HAS_MPL:
        logger.error("matplotlib is required — pip install matplotlib")
        return

    generate_all_plots(
        diagnostics_dir=args.diagnostics_dir,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir,
        layer_names=args.layers,
        non_static_only=args.non_static_only,
        max_layers=args.max_layers,
    )


if __name__ == "__main__":
    main()
