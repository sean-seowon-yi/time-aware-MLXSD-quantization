"""Plot polynomial clipping vs pre-CSB and post-CSB activations (and weight profiles).

For each layer, produces PNGs under ``--output-dir``:

- ``pre_csb/{layer}.png`` — pre-CSB absmax vs σ + poly fit **without** weight line.

- ``pre_csb/{layer}_weight_ref.png`` — same, **plus** horizontal ``max|W|`` when
  ``weight_stats.npz`` has the layer.

- ``post_csb/{layer}.png`` — post-CSB + schedule **without** weight line.

- ``post_csb/{layer}_weight_ref.png`` — same, **plus** horizontal ``max|W·b|`` when
  weights and ``b`` shapes match.

- ``pre_post/{layer}.png`` — **pre-CSB and post-CSB** absmax vs σ on one
  figure (no polynomial).

- ``weight_pre_post/{layer}.png`` — per-channel weight profile: **pre-CSB/SSC**
  ``max_i |W[i,j]|`` vs **post-CSB/SSC** ``max_i |W[i,j]|·b[j]`` (same
  convention as ``src.phase2.plot_weight_profile``), when
  ``diagnostics/weight_stats.npz`` contains that layer.

With ``--block-act-weight``, also writes **one PNG per layer** (see
``src.phase3.plot_block_act_weight_overlay``):

- ``pre_act_weight/{layer}.png`` — per-channel **max over σ** pre activation vs ``|W|``.
- ``post_act_weight/{layer}.png`` — same for post (``|x/b|`` vs ``|W|·b``).

Auto-selected layers are sorted by **block index → image/text → q,k,v,o,fc1,fc2**.
By default **every layer** listed in ``poly_schedule.json`` is a plot target
(``--max-layers 0`` = no cap). Use ``--non-static-only`` if you only want
``degree > 0``. Layers with ``granularity: per_channel`` are still skipped until
a multi-curve plot exists.

Usage
-----
python -m src.phase3.plot_poly_pre_post \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \
    --output-dir plots/poly_pre_post

# Quick subset: first 40 layers in block order (same dirs as above)
#   --max-layers 40

# Specific layers only
python -m src.phase3.plot_poly_pre_post \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \
    --layers "blocks.12.image.mlp.fc2" \
    --output-dir plots/poly_pre_post
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


def _short_name(name: str) -> str:
    return (
        name.replace("blocks.", "B")
        .replace(".image.", ".I.")
        .replace(".text.", ".T.")
    )


# Same family ordering as ``src.phase2.plot_post_csb`` (natural layer order).
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
    """Format ``blocks.12.image.mlp.fc2`` into
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


_COL_ACT = "#9E9E9E"
_COL_POLY = "#1976D2"
_COL_POLY_ALPHA = "#C62828"
_COL_WEIGHT = "#6A1B9A"
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


def _fit_plot_poly(
    sigmas: np.ndarray,
    vals: np.ndarray,
    max_degree: int,
) -> tuple[int, list[float], float]:
    """Tiered poly fit (same rules as schedule) capped at ``max_degree``."""
    from .poly_clipping import poly_r2, select_degree

    if max_degree == 0:
        return 0, [float(np.max(vals))], 1.0

    degree, coeffs, r2, _ = select_degree(sigmas, vals)
    if degree > max_degree:
        r2_cap, coeffs_cap = poly_r2(sigmas, vals, max_degree)
        return max_degree, [float(c) for c in coeffs_cap], float(r2_cap)
    return degree, coeffs, float(r2)


def _plot_single(
    name: str,
    sigmas: np.ndarray,
    absmax: np.ndarray,
    poly_coeffs: list[float],
    degree: int,
    r2: float,
    *,
    mode: str,
    output_path: Path,
    weight_scalar: float | None = None,
) -> None:
    """Render one PNG: trajectory + polynomial + optional weight reference line."""
    if not HAS_MPL:
        return

    sigma_fine = np.linspace(float(sigmas.min()), float(sigmas.max()), 300)
    poly_vals = np.maximum(_eval_poly(poly_coeffs, sigma_fine), 0.01)

    fig, ax = plt.subplots(figsize=(7, 4))

    marker = "o" if mode == "pre_csb" else "s"
    w_label = "max|W|" if mode == "pre_csb" else "max|W·b|"

    ax.scatter(
        sigmas, absmax, c=_COL_ACT, s=28, marker=marker,
        edgecolors="none", alpha=0.9, label=_LBL_ACT, zorder=2,
    )
    ax.plot(
        sigma_fine, poly_vals, "-", color=_COL_POLY, lw=2,
        label=_LBL_POLY, zorder=3,
    )

    if weight_scalar is not None:
        ax.axhline(
            weight_scalar, color=_COL_WEIGHT, ls="--", lw=1.5, alpha=0.8,
            label=f"{w_label} = {weight_scalar:.1f}", zorder=1,
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


def _plot_pre_post_overlay(
    name: str,
    sigmas: np.ndarray,
    pre_csb_absmax: np.ndarray,
    post_csb_absmax: np.ndarray,
    output_path: Path,
) -> None:
    """Pre- and post-CSB absmax vs σ only (no polynomial)."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        sigmas, pre_csb_absmax, c=_COL_POLY, s=28, marker="o",
        edgecolors="none", alpha=0.9, label="Abs Max Activation (pre-CSB)", zorder=2,
    )
    ax.scatter(
        sigmas, post_csb_absmax, c=_COL_POLY_ALPHA, s=28, marker="s",
        edgecolors="none", alpha=0.9, label="Abs Max Activation (post-CSB)", zorder=3,
    )
    ax.invert_xaxis()
    ax.set_xlabel("σ  (noisy → clean)")
    ax.set_ylabel("Absmax")
    ax.set_ylim(bottom=0)
    ax.set_title(_friendly_title(name, prefix="Pre/Post-CSB trajectory"), fontsize=11)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def _plot_weight_pre_post(
    name: str,
    w_pre: np.ndarray,
    w_post: np.ndarray,
    output_path: Path,
) -> None:
    """Per-channel weight absmax: pre vs post CSB/SSC (no polynomial)."""
    if not HAS_MPL:
        return

    w_pre = np.asarray(w_pre, dtype=np.float64).ravel()
    w_post = np.asarray(w_post, dtype=np.float64).ravel()
    ch = np.arange(w_pre.size, dtype=np.float64)

    n = w_pre.size
    pt = max(3, min(24, int(4000 // max(n, 1))))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        ch, w_pre, c=_COL_POLY, s=pt, marker="o",
        edgecolors="none", alpha=0.75, label="max_i |W[i,j]| (pre-CSB/SSC)",
        zorder=2,
    )
    ax.scatter(
        ch, w_post, c=_COL_POLY_ALPHA, s=pt, marker="s",
        edgecolors="none", alpha=0.75, label="max_i |W[i,j]|·b[j] (post-CSB/SSC)",
        zorder=3,
    )

    ax.set_xlabel("Input channel j")
    ax.set_ylabel("Weight magnitude")
    ax.set_ylim(bottom=0)
    ax.set_title(_friendly_title(name, prefix="Weight profile"), fontsize=11)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def generate_plots(
    diagnostics_dir: Path,
    calibration_dir: Path,
    output_dir: Path,
    layer_names: list[str] | None = None,
    non_static_only: bool = False,
    max_layers: int = 0,
    block_act_weight: bool = False,
) -> None:
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
    sched_max_deg = int(schedule.get("max_degree", 4))

    if layer_names is not None:
        target_names = layer_names
    elif non_static_only:
        target_names = [
            k for k, v in layers_schedule.items() if v["degree"] > 0
        ]
    else:
        # Plot every scheduled layer (static and non-static).  The old
        # ``non_static + 5 static`` sample omitted most degree-0 blocks
        # (e.g. entire block 15 when all layers there are constant α).
        target_names = list(layers_schedule.keys())

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
            "Missing %s — no weight_pre_post/ plots (Phase 1 weight stats)",
            weight_stats_path,
        )

    logger.info("Generating pre/post-CSB poly plots for %d layers", len(target_names))
    count = 0
    weight_plots = 0

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

        entry = layers_schedule[name]
        if entry.get("granularity") == "per_channel":
            logger.info("Skipping per-channel layer %s", name)
            continue

        data = np.load(npz_path)
        act_traj = data["act_channel_max"]  # [T, d_in]
        b = b_vectors[name]
        b_safe = np.maximum(b, 1e-12)

        pre_csb_absmax = act_traj.max(axis=1)
        post_csb = act_traj / b_safe[np.newaxis, :]
        post_csb_absmax = post_csb.max(axis=1)

        post_coeffs = entry["coeffs"]
        post_degree = entry["degree"]
        post_r2 = float(entry.get("r2", 0.0))

        pre_deg, pre_coeffs, pre_r2 = _fit_plot_poly(
            sigmas, pre_csb_absmax, sched_max_deg,
        )

        safe_name = name.replace(".", "_")

        w_arr = weight_pre_all.get(name)
        if w_arr is not None:
            w_pre_vec = np.asarray(w_arr, dtype=np.float64).ravel()
            b_vec = np.asarray(b, dtype=np.float64).ravel()
            pre_w_scalar = float(np.max(w_pre_vec))
            if w_pre_vec.shape == b_vec.shape:
                post_w_scalar = float(np.max(w_pre_vec * b_vec))
            else:
                post_w_scalar = None
        else:
            w_pre_vec = None
            pre_w_scalar = None
            post_w_scalar = None

        _plot_single(
            name, sigmas, pre_csb_absmax,
            poly_coeffs=pre_coeffs, degree=pre_deg, r2=pre_r2,
            mode="pre_csb",
            output_path=output_dir / "pre_csb" / f"{safe_name}.png",
            weight_scalar=None,
        )
        if pre_w_scalar is not None:
            _plot_single(
                name, sigmas, pre_csb_absmax,
                poly_coeffs=pre_coeffs, degree=pre_deg, r2=pre_r2,
                mode="pre_csb",
                output_path=output_dir / "pre_csb" / f"{safe_name}_weight_ref.png",
                weight_scalar=pre_w_scalar,
            )

        _plot_single(
            name, sigmas, post_csb_absmax,
            poly_coeffs=post_coeffs, degree=post_degree, r2=post_r2,
            mode="post_csb",
            output_path=output_dir / "post_csb" / f"{safe_name}.png",
            weight_scalar=None,
        )
        if post_w_scalar is not None:
            _plot_single(
                name, sigmas, post_csb_absmax,
                poly_coeffs=post_coeffs, degree=post_degree, r2=post_r2,
                mode="post_csb",
                output_path=output_dir / "post_csb" / f"{safe_name}_weight_ref.png",
                weight_scalar=post_w_scalar,
            )

        _plot_pre_post_overlay(
            name, sigmas, pre_csb_absmax, post_csb_absmax,
            output_path=output_dir / "pre_post" / f"{safe_name}.png",
        )

        if w_pre_vec is not None:
            b_vec = np.asarray(b, dtype=np.float64).ravel()
            if w_pre_vec.shape == b_vec.shape:
                w_post_vec = w_pre_vec * b_vec
                _plot_weight_pre_post(
                    name, w_pre_vec, w_post_vec,
                    output_path=output_dir / "weight_pre_post" / f"{safe_name}.png",
                )
                weight_plots += 1
            else:
                logger.warning(
                    "%s: weight channels %d vs b length %d — skip weight plot",
                    name, w_pre_vec.size, b_vec.size,
                )

        count += 1

    logger.info(
        "Done — %d layers: pre_csb/post_csb (plain + optional *_weight_ref.png), "
        "pre_post/; weight_pre_post/ %d/%d in %s",
        count, weight_plots, count, output_dir,
    )

    if block_act_weight:
        from .plot_block_act_weight_overlay import generate_block_act_weight_plots

        generate_block_act_weight_plots(
            diagnostics_dir, calibration_dir, output_dir,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Plot pre/post-CSB trajectories (with optional poly "
                    "overlays) as separate PNGs.",
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
        "--output-dir", type=Path, default=Path("plots/poly_pre_post"),
        help="Output directory for plots (default: plots/poly_pre_post)",
    )
    parser.add_argument(
        "--layers", type=str, nargs="*", default=None,
        help="Specific layer names to plot (default: auto-select)",
    )
    parser.add_argument(
        "--non-static-only", action="store_true",
        help="Only plot layers with polynomial degree > 0 (default is all layers)",
    )
    parser.add_argument(
        "--max-layers", type=int, default=0,
        help="Max layers to plot; 0 = no limit (default). "
             "When limited, layers are taken in block order (0.image.q, …).",
    )
    parser.add_argument(
        "--block-act-weight", action="store_true",
        help="Also write pre_act_weight/ and post_act_weight/ one PNG per layer "
             "(activation vs weight twin-y on channel index).",
    )
    args = parser.parse_args()

    if not HAS_MPL:
        logger.error("matplotlib is required — pip install matplotlib")
        return

    generate_plots(
        diagnostics_dir=args.diagnostics_dir,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir,
        layer_names=args.layers,
        non_static_only=args.non_static_only,
        max_layers=args.max_layers,
        block_act_weight=args.block_act_weight,
    )


if __name__ == "__main__":
    main()
