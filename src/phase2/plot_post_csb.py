#!/usr/bin/env python3
"""Plot post-CSB/SSC activation absmax vs sigma for every quantized layer.

Standalone pipeline that reads Phase 1 diagnostic data and calibration
balancing vectors to show what the quantizer sees at each denoising step.

For each layer the per-tensor absmax at step *t* is:

    pre_csb[t]  = max_j  act_channel_max[t, j]
    post_csb[t] = max_j( act_channel_max[t, j] / b[j] )

where *b* is the SSC-weighted balancing vector from calibration.

If a static-quantised model directory is given, a horizontal clip line at
``static_scale * 127`` is overlaid so you can see where clipping occurs.

Output
------
One PNG + PDF per MMDiT block, plus one for ``final_layer.linear``.
Each figure arranges subplots as [image, text] × [q, k, v, o, fc1, fc2].

Usage
-----
    python -m src.phase2.plot_post_csb \\
        --diagnostics-dir diagnostics \\
        --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \\
        --output-dir plots/post_csb
"""

from __future__ import annotations

import argparse
import json
import sys
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


def load_calibration(cal_dir: Path) -> dict[str, np.ndarray]:
    meta = json.loads((cal_dir / "calibration_meta.json").read_text())
    data = np.load(cal_dir / "calibration.npz")
    return {name: data[name] for name in meta["layer_names"]}


def load_static_scales(cal_dir: Path) -> dict[str, float] | None:
    path = cal_dir / "static_scales.npz"
    if not path.exists():
        return None
    raw = np.load(path)
    scales: dict[str, float] = {}
    for name in raw.files:
        s = raw[name]
        scales[name] = float(s.item() if s.ndim == 0 else s[0])
    return scales


def parse_layer_name(name: str) -> tuple[int, str, str] | None:
    """Return (block_idx, side, family) or None for special layers."""
    if name in ("context_embedder", "final_layer.linear"):
        return None
    parts = name.split(".")
    return int(parts[1]), parts[2], parts[4]


def compute_absmax(
    diag_dir: Path,
    b_vectors: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Return per-layer dict with sigma, pre_csb, post_csb arrays."""
    results: dict[str, dict] = {}
    for name, b in b_vectors.items():
        act_path = diag_dir / "activation_stats" / f"{name}.npz"
        if not act_path.exists():
            continue
        npz = np.load(act_path)
        raw = npz["act_channel_max"]            # [T, d_in]
        sigmas = npz["sigma_values"]             # [T]
        pre  = raw.max(axis=1)                   # [T]
        post = (raw / b[np.newaxis, :]).max(axis=1)
        results[name] = {"sigma": sigmas, "pre_csb": pre, "post_csb": post}
    return results


def plot_block(
    block_idx: int,
    block_layers: dict[str, list[tuple[str, str]]],
    layer_data: dict[str, dict],
    static_scales: dict[str, float] | None,
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
        squeeze=False, sharex=True,
    )
    fig.suptitle(
        f"Block {block_idx} — Post-CSB Activation Absmax vs σ",
        fontsize=13, fontweight="bold",
    )

    for row, side in enumerate(sides_present):
        layers = block_layers[(block_idx, side)]
        for col, (name, family) in enumerate(layers):
            ax = axes[row, col]
            d = layer_data.get(name)
            if d is None:
                ax.set_visible(False)
                continue

            sigma = d["sigma"]
            ax.scatter(sigma, d["pre_csb"], color="silver", s=18,
                       edgecolors="none", label="Pre-CSB", zorder=1)
            ax.scatter(sigma, d["post_csb"], color="#1f77b4", s=18,
                       edgecolors="none", label="Post-CSB", zorder=2)

            if static_scales and name in static_scales:
                clip = static_scales[name] * 127.0
                ax.axhline(clip, color="#d62728", ls="--", lw=1.2,
                           label=f"Static clip", zorder=3)

            ax.set_title(f"{side}.{family}", fontsize=10)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.25)
            if row == nrows - 1:
                ax.set_xlabel("σ")
            if col == 0:
                ax.set_ylabel("Absmax")
            ax.legend(fontsize=7, loc="upper right")

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
    layer_data: dict[str, dict],
    static_scales: dict[str, float] | None,
    out_dir: Path,
) -> None:
    d = layer_data.get(name)
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sigma = d["sigma"]
    ax.scatter(sigma, d["pre_csb"], color="silver", s=24,
               edgecolors="none", label="Pre-CSB")
    ax.scatter(sigma, d["post_csb"], color="#1f77b4", s=24,
               edgecolors="none", label="Post-CSB")
    if static_scales and name in static_scales:
        clip = static_scales[name] * 127.0
        ax.axhline(clip, color="#d62728", ls="--", lw=1.2, label="Static clip")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("σ")
    ax.set_ylabel("Absmax")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    stem = name.replace(".", "_")
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot post-CSB/SSC activation absmax vs σ for every quantized layer.",
    )
    parser.add_argument(
        "--diagnostics-dir", type=str, default="diagnostics",
        help="Phase 1 diagnostics directory (default: diagnostics/)",
    )
    parser.add_argument(
        "--calibration-dir", type=str, required=True,
        help="Directory containing calibration.npz + calibration_meta.json "
             "(e.g. quantized/w4a8_l2_a0.50_gs32_static).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots/post_csb",
        help="Where to write plots (default: plots/post_csb/).",
    )
    args = parser.parse_args()

    diag_dir = Path(args.diagnostics_dir)
    cal_dir  = Path(args.calibration_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Diagnostics : {diag_dir}")
    print(f"Calibration : {cal_dir}")
    print(f"Output      : {out_dir}")

    b_vectors = load_calibration(cal_dir)
    static_scales = load_static_scales(cal_dir)
    print(f"Loaded {len(b_vectors)} balancing vectors"
          + (f", {len(static_scales)} static scales" if static_scales else ""))

    layer_data = compute_absmax(diag_dir, b_vectors)
    print(f"Computed absmax trajectories for {len(layer_data)} layers")

    # Organize by block
    block_layers: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    special: list[tuple[str, str]] = []

    for name in b_vectors:
        parsed = parse_layer_name(name)
        if parsed is None:
            if name == "final_layer.linear":
                special.append((name, "Final Layer Linear"))
            continue
        block_idx, side, family = parsed
        block_layers[(block_idx, side)].append((name, family))

    for key in block_layers:
        block_layers[key].sort(key=lambda x: FAMILY_ORDER.get(x[1], 99))

    blocks = sorted({b for b, _ in block_layers})
    for block_idx in blocks:
        plot_block(block_idx, block_layers, layer_data, static_scales, out_dir)
        print(f"  Block {block_idx:2d} ✓")

    for name, title in special:
        plot_single_layer(name, title, layer_data, static_scales, out_dir)
        print(f"  {title} ✓")

    total = len(blocks) + len(special)
    print(f"\nDone — {total} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
