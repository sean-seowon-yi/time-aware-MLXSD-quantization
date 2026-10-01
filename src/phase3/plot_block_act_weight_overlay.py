"""Standalone figures: activation vs weight on channel axis (twin y), one PNG per layer.

For each quantized linear layer (same scope as calibration), writes two files:

- ``pre_act_weight/{layer}.png`` — **Pre-CSB/SSC**: per channel *j*,
  ``max_t act_channel_max[t,j]`` (blue, left y) vs ``max_i |W[i,j]|`` (red, right y).

- ``post_act_weight/{layer}.png`` — **Post-CSB/SSC**:
  ``max_t |x_{t,j}/b_j|`` vs ``max_i |W[i,j]|·b_j``.

Layer names use underscores like ``blocks_12_image_mlp_fc2.png``.

Usage
-----
python -m src.phase3.plot_block_act_weight_overlay \\
    --diagnostics-dir diagnostics \\
    --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \\
    --output-dir plots/block_act_weight
"""

from __future__ import annotations

import argparse
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

from ..phase2.plot_post_csb import load_calibration, parse_layer_name
from ..phase2.plot_weight_profile import load_weight_stats

_FAM_ORDER = {
    "q_proj": 0, "k_proj": 1, "v_proj": 2,
    "o_proj": 3, "fc1": 4, "fc2": 5,
}


def _layer_sort_key(name: str) -> tuple[int, int, int, str]:
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


def _short_title(name: str) -> str:
    return (
        name.replace("blocks.", "B")
        .replace(".image.", ".I.")
        .replace(".text.", ".T.")
    )


def _draw_act_weight_on_ax(
    ax,
    name: str,
    act_dir: Path,
    b_vectors: dict[str, np.ndarray],
    weight_pre: dict[str, np.ndarray],
    *,
    mode: str,
) -> bool:
    """Twin-y scatter on *ax*. Returns True if data was plotted."""
    npz_path = act_dir / f"{name}.npz"
    if not npz_path.exists():
        ax.text(0.5, 0.5, "no activations", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return False

    b = b_vectors.get(name)
    if b is None:
        ax.text(0.5, 0.5, "no b", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return False

    w_arr = weight_pre.get(name)
    if w_arr is None:
        ax.text(0.5, 0.5, "no weight stats", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return False

    act = np.asarray(np.load(npz_path)["act_channel_max"], dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    w = np.asarray(w_arr, dtype=np.float64).ravel()

    if act.ndim != 2 or act.shape[1] != b.size or w.size != b.size:
        ax.text(0.5, 0.5, "shape mismatch", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return False

    d_in = act.shape[1]
    ch = np.arange(d_in, dtype=np.float64)
    b_safe = np.maximum(b, 1e-12)
    pt = max(2, min(16, int(6000 // max(d_in, 1))))

    if mode == "pre":
        y_act = np.max(act, axis=0)
        y_w = w
        ylab_act = r"$\max_t |x|$"
        ylab_w = r"$\max_i|W|$"
        mode_title = "pre-CSB/SSC"
    else:
        y_act = np.max(act / b_safe[np.newaxis, :], axis=0)
        y_w = w * b
        ylab_act = r"$\max_t|x/b|$"
        ylab_w = r"$\max_i|W|\!\cdot\!b$"
        mode_title = "post-CSB/SSC"

    ax.scatter(
        ch, y_act, s=pt, c="#1976D2", edgecolors="none", alpha=0.75,
        label=ylab_act, zorder=2,
    )
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylab_act, color="#1976D2", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#1976D2", labelsize=8)
    ax.set_xlabel("channel j", fontsize=9)
    ax.grid(True, alpha=0.2)

    ax2 = ax.twinx()
    ax2.scatter(
        ch, y_w, s=pt, c="#C62828", marker="s", edgecolors="none", alpha=0.65,
        label=ylab_w, zorder=3,
    )
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel(ylab_w, color="#C62828", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#C62828", labelsize=8)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", framealpha=0.9)

    ax.set_title(f"{_short_title(name)} — {mode_title}", fontsize=10)
    return True


def _save_standalone_layer(
    name: str,
    act_dir: Path,
    b_vectors: dict[str, np.ndarray],
    weight_pre: dict[str, np.ndarray],
    out_dir: Path,
    *,
    mode: str,
) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_act_weight_on_ax(ax, name, act_dir, b_vectors, weight_pre, mode=mode)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = name.replace(".", "_")
    p = out_dir / f"{safe}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)


def generate_block_act_weight_plots(
    diagnostics_dir: Path,
    calibration_dir: Path,
    output_dir: Path,
) -> None:
    """Write one PNG per layer under ``pre_act_weight/`` and ``post_act_weight/``."""
    if not HAS_MPL:
        logger.error("matplotlib required")
        return

    act_dir = diagnostics_dir / "activation_stats"
    w_path = diagnostics_dir / "weight_stats.npz"
    if not w_path.exists():
        logger.warning("Missing %s — cannot build act+weight overlays", w_path)
        return

    b_vectors = load_calibration(calibration_dir)
    weight_pre = load_weight_stats(diagnostics_dir)

    layer_names = sorted(
        (n for n in b_vectors if parse_layer_name(n) is not None),
        key=_layer_sort_key,
    )

    out_pre = output_dir / "pre_act_weight"
    out_post = output_dir / "post_act_weight"

    n_ok = 0
    for name in layer_names:
        _save_standalone_layer(
            name, act_dir, b_vectors, weight_pre, out_pre, mode="pre",
        )
        _save_standalone_layer(
            name, act_dir, b_vectors, weight_pre, out_post, mode="post",
        )
        n_ok += 1

    logger.info(
        "Act+weight overlays: %d layers × 2 PNGs → %s/ and %s/",
        n_ok, out_pre, out_post,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Per-layer pre/post CSB activation + weight twin-y overlays.",
    )
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=Path("diagnostics"),
    )
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots/block_act_weight"),
    )
    args = parser.parse_args()

    generate_block_act_weight_plots(
        diagnostics_dir=args.diagnostics_dir,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
