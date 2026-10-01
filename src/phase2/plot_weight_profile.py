#!/usr/bin/env python3
"""Plot per-channel weight absmax before/after CSB for every quantized layer.

Standalone pipeline that reads Phase 1 weight statistics and calibration
balancing vectors to show how CSB/SSC reshapes the weight magnitude profile.

For ALL quantized layers (balance.py applies ``balance_weight`` universally):
    pre_csb[j]  = max_i |W[i, j]|
    post_csb[j] = max_i |W[i, j]| * b[j]      (W → W * diag(b))

Online layers (o_proj, fc2) additionally store ``b_inv = 1/b`` for runtime
input scaling, but their weight is ALSO multiplied by ``b``.

Output
------
One PNG + PDF per MMDiT block, plus one for special layers.
Each figure arranges subplots as [image, text] × [q, k, v, o, fc1, fc2].

Usage
-----
    python -m src.phase2.plot_weight_profile \\
        --diagnostics-dir diagnostics \\
        --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \\
        --output-dir plots/weight_profile
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = {
    "q_proj": 0, "k_proj": 1, "v_proj": 2,
    "o_proj": 3, "fc1": 4, "fc2": 5,
}
SIDES = ["image", "text"]


def load_weight_stats(diag_dir: Path) -> dict[str, np.ndarray]:
    path = diag_dir / "weight_stats.npz"
    raw = np.load(path)
    stats: dict[str, np.ndarray] = {}
    for key in raw.files:
        if key.endswith("/w_channel_max"):
            layer_name = key.rsplit("/", 1)[0]
            stats[layer_name] = raw[key]
    return stats


def load_calibration(cal_dir: Path) -> dict[str, np.ndarray]:
    """Return b_vectors dict."""
    meta = json.loads((cal_dir / "calibration_meta.json").read_text())
    data = np.load(cal_dir / "calibration.npz")
    return {name: data[name] for name in meta["layer_names"]}



def parse_layer_name(name: str) -> tuple[int, str, str] | None:
    if name in ("context_embedder", "final_layer.linear"):
        return None
    parts = name.split(".")
    return int(parts[1]), parts[2], parts[4]


def compute_post_csb_weights(
    weight_data: dict[str, np.ndarray],
    b_vectors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Post-CSB weight absmax: ALL layers get w*b (balance.py line 184)."""
    post: dict[str, np.ndarray] = {}
    for name, w in weight_data.items():
        b = b_vectors.get(name)
        if b is None:
            post[name] = w.copy()
        else:
            post[name] = w * b
    return post


def _draw(ax, channels, pre, post, title, is_bottom, is_left):
    ax.scatter(channels, pre, s=3, color="silver",
               edgecolors="none", alpha=0.6, label="Pre-CSB", zorder=1)
    ax.scatter(channels, post, s=3, color="#1f77b4",
               edgecolors="none", alpha=0.7, label="Post-CSB", zorder=2)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    if is_bottom:
        ax.set_xlabel("Channel")
    if is_left:
        ax.set_ylabel("|W| max")
    ax.legend(fontsize=6, loc="upper right")


def plot_block(
    block_idx: int,
    block_layers: dict[tuple[int, str], list[tuple[str, str]]],
    weight_pre: dict[str, np.ndarray],
    weight_post: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    sides_present = [s for s in SIDES if (block_idx, s) in block_layers]
    nrows = len(sides_present)
    if nrows == 0:
        return

    max_cols = max(len(block_layers[(block_idx, s)]) for s in sides_present)
    fig, axes = plt.subplots(
        nrows, max_cols,
        figsize=(4 * max_cols, 3.5 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"Block {block_idx} — Per-Channel Weight Absmax (before / after CSB)",
        fontsize=13, fontweight="bold",
    )

    for row, side in enumerate(sides_present):
        layers = block_layers[(block_idx, side)]
        for col, (name, family) in enumerate(layers):
            ax = axes[row, col]
            pre = weight_pre.get(name)
            post = weight_post.get(name)
            if pre is None:
                ax.set_visible(False)
                continue

            _draw(ax, np.arange(len(pre)), pre, post,
                  f"{side}.{family}", row == nrows - 1, col == 0)

        for col in range(len(layers), max_cols):
            axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    stem = f"block_{block_idx:02d}"
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_single_layer(
    name: str,
    title: str,
    weight_pre: dict[str, np.ndarray],
    weight_post: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    pre = weight_pre.get(name)
    if pre is None:
        return
    post = weight_post.get(name)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _draw(ax, np.arange(len(pre)), pre, post,
          title, True, True)
    plt.tight_layout()
    stem = name.replace(".", "_")
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-channel weight absmax before/after CSB.",
    )
    parser.add_argument(
        "--diagnostics-dir", type=str, default="diagnostics",
        help="Phase 1 diagnostics directory (default: diagnostics/).",
    )
    parser.add_argument(
        "--calibration-dir", type=str, required=True,
        help="Directory with calibration.npz + calibration_meta.json.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots/weight_profile",
        help="Where to write plots (default: plots/weight_profile/).",
    )
    args = parser.parse_args()

    diag_dir = Path(args.diagnostics_dir)
    cal_dir  = Path(args.calibration_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Diagnostics : {diag_dir}")
    print(f"Calibration : {cal_dir}")
    print(f"Output      : {out_dir}")

    weight_pre = load_weight_stats(diag_dir)
    b_vectors = load_calibration(cal_dir)
    weight_post = compute_post_csb_weights(weight_pre, b_vectors)

    print(f"Loaded {len(weight_pre)} layers, {len(b_vectors)} b-vectors")

    block_layers: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    special: list[tuple[str, str]] = []

    for name in weight_pre:
        parsed = parse_layer_name(name)
        if parsed is None:
            if name == "final_layer.linear":
                special.append((name, "Final Layer Linear"))
            elif name == "context_embedder":
                special.append((name, "Context Embedder"))
            continue
        block_idx, side, family = parsed
        block_layers[(block_idx, side)].append((name, family))

    for key in block_layers:
        block_layers[key].sort(key=lambda x: FAMILY_ORDER.get(x[1], 99))

    blocks = sorted({b for b, _ in block_layers})
    for block_idx in blocks:
        plot_block(block_idx, block_layers, weight_pre, weight_post, out_dir)
        print(f"  Block {block_idx:2d} ✓")

    for name, title in special:
        plot_single_layer(name, title, weight_pre, weight_post, out_dir)
        print(f"  {title} ✓")

    total = len(blocks) + len(special)
    print(f"\nDone — {total} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
