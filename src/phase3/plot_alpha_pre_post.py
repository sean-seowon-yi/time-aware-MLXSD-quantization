"""Plot polynomial α(σ) before and after alpha search.

For each layer, produces PNGs under ``--output-dir``:

- ``pre_alpha/{layer}.png`` — post-CSB absmax vs σ + original poly α(σ)
  (from the pre-alpha-search schedule).

- ``pre_alpha/{layer}_weight_ref.png`` — same, **plus** horizontal ``max|W·b|``
  when ``weight_stats.npz`` has the layer.

- ``post_alpha/{layer}.png`` — post-CSB absmax vs σ + scaled poly
  ``α(σ) × alpha_multiplier`` (from the post-alpha-search schedule).

- ``post_alpha/{layer}_weight_ref.png`` — same, **plus** horizontal ``max|W·b|``.

- ``pre_post_alpha/{layer}.png`` — **both** α(σ) curves (pre- and
  post-alpha-search) overlaid on the same figure together with the
  post-CSB absmax trajectory.

Layer auto-selection, sorting, and per-channel skipping follow the same
conventions as ``plot_poly_pre_post``.

Usage
-----
python -m src.phase3.plot_alpha_pre_post \
    --diagnostics-dir diagnostics \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static_alpha_search \
    --output-dir plots/alpha_pre_post

# Specific layers only
python -m src.phase3.plot_alpha_pre_post \
    --diagnostics-dir diagnostics \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static_alpha_search \
    --layers "blocks.12.image.mlp.fc2" \
    --output-dir plots/alpha_pre_post
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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


def _short_name(name: str) -> str:
    return (
        name.replace("blocks.", "B")
        .replace(".image.", ".I.")
        .replace(".text.", ".T.")
    )


_FAM_ORDER = {
    "q_proj": 0, "k_proj": 1, "v_proj": 2,
    "o_proj": 3, "fc1": 4, "fc2": 5,
}

_SUBLAYER_LABEL = {
    "attn.q_proj": "Q_Proj",
    "attn.k_proj": "K_Proj",
    "attn.v_proj": "V_Proj",
    "attn.o_proj": "O_Proj",
    "mlp.fc1": "FC1",
    "mlp.fc2": "FC2",
}


def _friendly_title(name: str, prefix: str = "Polynomial curve fit") -> str:
    """Format a layer name like ``blocks.12.image.mlp.fc2`` into
    ``"Polynomial curve fit for block 12, Image FC2"``.
    """
    parts = name.split(".")
    if len(parts) >= 5 and parts[0] == "blocks" and parts[2] in ("image", "text"):
        try:
            bidx = int(parts[1])
        except ValueError:
            return f"{prefix} for {_short_name(name)}"
        side = "Image" if parts[2] == "image" else "Text"
        sublayer = ".".join(parts[3:])
        nice = _SUBLAYER_LABEL.get(sublayer, sublayer)
        return f"{prefix} for block {bidx}, {side} {nice}"
    return f"{prefix} for {_short_name(name)}"


# Canonical colours used across per-layer plots.
_COL_ACT = "#9E9E9E"        # grey — Abs Max Activation
_COL_POLY = "#1976D2"       # blue — Poly curve (baseline, m = 1.0)
_COL_POLY_ALPHA = "#C62828"  # red  — Poly curve shifted by α
_COL_WEIGHT = "#6A1B9A"     # purple — max|W·b| reference
_LBL_ACT = "Abs Max Activation"
_LBL_POLY = "Poly curve"
_LBL_POLY_ALPHA = "Poly curve shifted by α"


def _layer_sort_key(name: str) -> tuple[int, int, int, str]:
    """Sort key: block index, image before text, attention/MLP family."""
    parts = name.split(".")
    if name == "context_embedder":
        return (-1, 0, 0, name)
    if name == "final_layer.linear":
        return (1 << 20, 0, 0, name)
    if len(parts) >= 5 and parts[0] == "blocks" and parts[2] in ("image", "text"):
        bidx = int(parts[1])
        side = 0 if parts[2] == "image" else 1
        fam = parts[4]
        fo = _FAM_ORDER.get(fam, 99)
        return (bidx, side, fo, name)
    return (1 << 19, 0, 0, name)


def _plot_single_alpha(
    name: str,
    sigmas: np.ndarray,
    absmax: np.ndarray,
    poly_coeffs: list[float],
    degree: int,
    r2: float,
    alpha_multiplier: float,
    *,
    mode: str,
    output_path: Path,
    weight_scalar: float | None = None,
) -> None:
    """Render one PNG: trajectory + polynomial (optionally scaled by alpha_multiplier)."""
    if not HAS_MPL:
        return

    sigma_fine = np.linspace(float(sigmas.min()), float(sigmas.max()), 300)
    raw_poly = _eval_poly(poly_coeffs, sigma_fine)
    poly_vals = np.maximum(raw_poly * alpha_multiplier, 0.01)

    fig, ax = plt.subplots(figsize=(7, 4))

    if mode == "pre_alpha":
        poly_colour = _COL_POLY
        poly_label = _LBL_POLY
    else:
        poly_colour = _COL_POLY_ALPHA
        poly_label = _LBL_POLY_ALPHA

    ax.scatter(
        sigmas, absmax, c=_COL_ACT, s=28, marker="s",
        edgecolors="none", alpha=0.9, label=_LBL_ACT, zorder=2,
    )
    ax.plot(
        sigma_fine, poly_vals, "-", color=poly_colour, lw=2,
        label=poly_label, zorder=3,
    )

    if weight_scalar is not None:
        ax.axhline(
            weight_scalar, color=_COL_WEIGHT, ls="--", lw=1.5, alpha=0.8,
            label=f"max|W·b| = {weight_scalar:.1f}", zorder=1,
        )

    if mode != "pre_alpha":
        ax.text(
            0.02, 0.97, f"α = {alpha_multiplier:.2f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

    ax.invert_xaxis()
    ax.set_xlabel("σ  (noisy → clean)")
    ax.set_ylabel("Absmax")
    ax.set_ylim(bottom=0)
    ax.set_title(_friendly_title(name), fontsize=11)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def _plot_alpha_overlay(
    name: str,
    sigmas: np.ndarray,
    absmax: np.ndarray,
    poly_coeffs: list[float],
    degree: int,
    alpha_multiplier: float,
    output_path: Path,
    *,
    weight_scalar: float | None = None,
) -> None:
    """Pre- and post-alpha-search α(σ) curves overlaid with the trajectory."""
    if not HAS_MPL:
        return

    sigma_fine = np.linspace(float(sigmas.min()), float(sigmas.max()), 300)
    raw_poly = _eval_poly(poly_coeffs, sigma_fine)
    pre_vals = np.maximum(raw_poly, 0.01)
    post_vals = np.maximum(raw_poly * alpha_multiplier, 0.01)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(
        sigmas, absmax, c=_COL_ACT, s=24, marker="s",
        edgecolors="none", alpha=0.85, label=_LBL_ACT, zorder=2,
    )
    ax.plot(
        sigma_fine, pre_vals, "-", color=_COL_POLY, lw=2,
        label=_LBL_POLY, zorder=3,
    )
    ax.plot(
        sigma_fine, post_vals, "-", color=_COL_POLY_ALPHA, lw=2,
        label=_LBL_POLY_ALPHA, zorder=4,
    )

    if weight_scalar is not None:
        ax.axhline(
            weight_scalar, color=_COL_WEIGHT, ls="--", lw=1.5, alpha=0.8,
            label=f"max|W·b| = {weight_scalar:.1f}", zorder=1,
        )

    ax.text(
        0.02, 0.97, f"α = {alpha_multiplier:.2f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )

    ax.invert_xaxis()
    ax.set_xlabel("σ  (noisy → clean)")
    ax.set_ylabel("Absmax")
    ax.set_ylim(bottom=0)
    ax.set_title(_friendly_title(name), fontsize=11)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def generate_plots(
    diagnostics_dir: Path,
    quantized_dir: Path,
    output_dir: Path,
    layer_names: list[str] | None = None,
    non_static_only: bool = False,
    max_layers: int = 0,
) -> None:
    from .poly_clipping import (
        POLY_SCHEDULE_FILENAME,
        _load_calibration_data,
        _load_sigma_values,
    )

    post_schedule_path = quantized_dir / POLY_SCHEDULE_FILENAME
    pre_schedule_path = quantized_dir / (POLY_SCHEDULE_FILENAME + ".pre_alpha_search.bak")

    if not post_schedule_path.exists():
        raise FileNotFoundError(
            f"Missing {post_schedule_path} — run alpha search first"
        )
    if not pre_schedule_path.exists():
        raise FileNotFoundError(
            f"Missing {pre_schedule_path} — the pre-alpha-search backup "
            f"poly_schedule.json.pre_alpha_search.bak is required"
        )

    with open(post_schedule_path) as f:
        post_schedule = json.load(f)
    with open(pre_schedule_path) as f:
        pre_schedule = json.load(f)

    post_layers = post_schedule["layers"]
    pre_layers = pre_schedule["layers"]

    sigmas = _load_sigma_values(diagnostics_dir)
    b_vectors, _ = _load_calibration_data(quantized_dir)
    act_dir = diagnostics_dir / "activation_stats"

    if layer_names is not None:
        target_names = layer_names
    elif non_static_only:
        target_names = [
            k for k, v in post_layers.items() if v["degree"] > 0
        ]
    else:
        target_names = list(post_layers.keys())

    if layer_names is None:
        target_names = sorted(target_names, key=_layer_sort_key)

    if max_layers > 0 and len(target_names) > max_layers:
        logger.info("Limiting to %d layers (from %d)", max_layers, len(target_names))
        target_names = target_names[:max_layers]

    weight_stats_path = diagnostics_dir / "weight_stats.npz"
    weight_pre_all: dict[str, np.ndarray] = {}
    if weight_stats_path.exists():
        from ..phase2.plot_weight_profile import load_weight_stats
        weight_pre_all = load_weight_stats(diagnostics_dir)
    else:
        logger.warning(
            "Missing %s — no *_weight_ref.png plots",
            weight_stats_path,
        )

    n_with_alpha = sum(
        1 for n in target_names
        if n in post_layers and "alpha_multiplier" in post_layers[n]
    )
    logger.info(
        "Generating alpha pre/post plots for %d layers (%d with alpha_multiplier)",
        len(target_names), n_with_alpha,
    )

    count = 0
    for name in target_names:
        if name not in post_layers:
            logger.warning("%s not in post-alpha schedule — skipping", name)
            continue
        if name not in b_vectors:
            logger.warning("%s not in calibration — skipping", name)
            continue

        npz_path = act_dir / f"{name}.npz"
        if not npz_path.exists():
            logger.warning("No activation stats for %s — skipping", name)
            continue

        post_entry = post_layers[name]
        if post_entry.get("granularity") == "per_channel":
            logger.info("Skipping per-channel layer %s", name)
            continue

        pre_entry = pre_layers.get(name, post_entry)
        alpha_multiplier = float(post_entry.get("alpha_multiplier", 1.0))

        coeffs = pre_entry["coeffs"]
        degree = pre_entry["degree"]
        r2 = float(pre_entry.get("r2", 0.0))

        data = np.load(npz_path)
        act_traj = data["act_channel_max"]  # [T, d_in]
        b = b_vectors[name]
        b_safe = np.maximum(b, 1e-12)
        post_csb = act_traj / b_safe[np.newaxis, :]
        post_csb_absmax = post_csb.max(axis=1)

        safe_name = name.replace(".", "_")

        w_arr = weight_pre_all.get(name)
        post_w_scalar: float | None = None
        if w_arr is not None:
            w_pre_vec = np.asarray(w_arr, dtype=np.float64).ravel()
            b_vec = np.asarray(b, dtype=np.float64).ravel()
            if w_pre_vec.shape == b_vec.shape:
                post_w_scalar = float(np.max(w_pre_vec * b_vec))

        # --- pre-alpha: poly(σ) with multiplier = 1.0 ---
        _plot_single_alpha(
            name, sigmas, post_csb_absmax,
            poly_coeffs=coeffs, degree=degree, r2=r2,
            alpha_multiplier=1.0,
            mode="pre_alpha",
            output_path=output_dir / "pre_alpha" / f"{safe_name}.png",
        )
        if post_w_scalar is not None:
            _plot_single_alpha(
                name, sigmas, post_csb_absmax,
                poly_coeffs=coeffs, degree=degree, r2=r2,
                alpha_multiplier=1.0,
                mode="pre_alpha",
                output_path=output_dir / "pre_alpha" / f"{safe_name}_weight_ref.png",
                weight_scalar=post_w_scalar,
            )

        # --- post-alpha: poly(σ) × alpha_multiplier ---
        _plot_single_alpha(
            name, sigmas, post_csb_absmax,
            poly_coeffs=coeffs, degree=degree, r2=r2,
            alpha_multiplier=alpha_multiplier,
            mode="post_alpha",
            output_path=output_dir / "post_alpha" / f"{safe_name}.png",
        )
        if post_w_scalar is not None:
            _plot_single_alpha(
                name, sigmas, post_csb_absmax,
                poly_coeffs=coeffs, degree=degree, r2=r2,
                alpha_multiplier=alpha_multiplier,
                mode="post_alpha",
                output_path=output_dir / "post_alpha" / f"{safe_name}_weight_ref.png",
                weight_scalar=post_w_scalar,
            )

        # --- overlay: both curves on one figure ---
        _plot_alpha_overlay(
            name, sigmas, post_csb_absmax,
            poly_coeffs=coeffs, degree=degree,
            alpha_multiplier=alpha_multiplier,
            output_path=output_dir / "pre_post_alpha" / f"{safe_name}.png",
        )
        if post_w_scalar is not None:
            _plot_alpha_overlay(
                name, sigmas, post_csb_absmax,
                poly_coeffs=coeffs, degree=degree,
                alpha_multiplier=alpha_multiplier,
                output_path=output_dir / "pre_post_alpha" / f"{safe_name}_weight_ref.png",
                weight_scalar=post_w_scalar,
            )

        count += 1

    logger.info(
        "Done — %d layers: pre_alpha/, post_alpha/ (plain + optional "
        "*_weight_ref.png), pre_post_alpha/ in %s",
        count, output_dir,
    )


# ---------------------------------------------------------------------------
# Alpha-by-block line plot  (analogous to plot_alpha_by_block from GPTQ)
# ---------------------------------------------------------------------------

_LAYER_KEY_RE = re.compile(
    r"^blocks\.(\d+)\.(image|text)\.(.+)$"
)


def plot_alpha_by_block(
    schedule_path: Path,
    output_path: Path,
    layer_types: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Plot per-block ``alpha_multiplier`` as a line plot, one line per layer type.

    Layer types are ``{side}.{sublayer}`` strings, e.g.
    ``image.mlp.fc1``, ``text.mlp.fc2``.  Pass *layer_types* to filter
    (default: all types present in the schedule).
    """
    if not HAS_MPL:
        logger.error("matplotlib is required")
        return

    with open(schedule_path) as f:
        schedule = json.load(f)

    layers = schedule["layers"]

    by_type: dict[str, dict[int, float]] = {}
    for key, entry in layers.items():
        m = _LAYER_KEY_RE.match(key)
        if not m:
            continue
        block_idx = int(m.group(1))
        side = m.group(2)
        sublayer = m.group(3)
        layer_type = f"{side}.{sublayer}"
        if layer_types is not None and layer_type not in layer_types:
            continue
        alpha = float(entry.get("alpha_multiplier", 1.0))
        by_type.setdefault(layer_type, {})[block_idx] = alpha

    if not by_type:
        logger.warning("No matching layers found for layer_types=%s", layer_types)
        return

    all_blocks = sorted({b for d in by_type.values() for b in d})
    n_blocks = len(all_blocks)

    fig, ax = plt.subplots(figsize=(14, 6))

    markers = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "d", "*"]
    sorted_types = sorted(by_type.keys())

    def _draw(ax, *, style: str) -> None:
        for i, layer_type in enumerate(sorted_types):
            block_alphas = by_type[layer_type]
            blocks = sorted(block_alphas.keys())
            values = [block_alphas[b] for b in blocks]
            if style == "scatter":
                ax.scatter(
                    blocks, values,
                    marker=markers[i % len(markers)], s=40,
                    label=layer_type, alpha=0.85, edgecolors="none", zorder=2,
                )
            else:
                ax.plot(
                    blocks, values,
                    marker="o", markersize=3, linewidth=1.5,
                    label=layer_type, zorder=2,
                )
        ax.axhline(
            y=1.0, color="gray", linestyle="--", linewidth=1,
            alpha=0.5, label="baseline (1.0)",
        )
        ax.set_xlabel("Block index")
        ax.set_ylabel("Alpha multiplier")
        ax.set_xticks(range(n_blocks))
        suffix = " (scatter)" if style == "scatter" else " (line)"
        ax.set_title((title or "alpha_multiplier by block and layer type") + suffix)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    ext = output_path.suffix or ".png"

    for style in ("scatter", "line"):
        fig, ax = plt.subplots(figsize=(14, 6))
        _draw(ax, style=style)
        fig.tight_layout()
        p = output_path.with_name(f"{stem}_{style}{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", p)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Plot α(σ) before and after alpha search — comparing "
                    "original poly schedule vs scaled α(σ)×multiplier.",
    )
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=Path("diagnostics"),
        help="Phase 1 diagnostics directory",
    )
    parser.add_argument(
        "--quantized-dir", type=Path, required=True,
        help="Quantized directory with poly_schedule.json (post-alpha) "
             "and poly_schedule.json.pre_alpha_search.bak (pre-alpha)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots/alpha_pre_post"),
        help="Output directory for plots (default: plots/alpha_pre_post)",
    )
    parser.add_argument(
        "--layers", type=str, nargs="*", default=None,
        help="Specific layer names to plot (default: auto-select all)",
    )
    parser.add_argument(
        "--non-static-only", action="store_true",
        help="Only plot layers with polynomial degree > 0",
    )
    parser.add_argument(
        "--max-layers", type=int, default=0,
        help="Max layers to plot; 0 = no limit (default). "
             "Layers are taken in block order.",
    )
    parser.add_argument(
        "--alpha-by-block", action="store_true",
        help="Also generate an alpha-multiplier-by-block line plot.",
    )
    parser.add_argument(
        "--alpha-by-block-only", action="store_true",
        help="Only generate the alpha-by-block line plot (skip per-layer PNGs).",
    )
    parser.add_argument(
        "--alpha-layer-types", type=str, nargs="*", default=None,
        help="Layer types for the alpha-by-block plot, e.g. image.mlp.fc1 "
             "text.mlp.fc2.  Default: all types.",
    )
    args = parser.parse_args()

    if not HAS_MPL:
        logger.error("matplotlib is required — pip install matplotlib")
        return

    from .poly_clipping import POLY_SCHEDULE_FILENAME

    if not args.alpha_by_block_only:
        generate_plots(
            diagnostics_dir=args.diagnostics_dir,
            quantized_dir=args.quantized_dir,
            output_dir=args.output_dir,
            layer_names=args.layers,
            non_static_only=args.non_static_only,
            max_layers=args.max_layers,
        )

    if args.alpha_by_block or args.alpha_by_block_only:
        schedule_path = args.quantized_dir / POLY_SCHEDULE_FILENAME
        plot_alpha_by_block(
            schedule_path=schedule_path,
            output_path=args.output_dir / "alpha_by_block.png",
            layer_types=args.alpha_layer_types,
        )


if __name__ == "__main__":
    main()
