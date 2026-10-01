#!/usr/bin/env python3
"""Plot per-group quantized vs original weight absmax for every quantized layer.

Standalone pipeline that loads the quantized safetensors, dequantizes each
layer, and compares per-group weight absmax against the post-CSB original.

For each group g (of ``group_size`` consecutive input channels):

    original_group_max[g] = max_{i, j in group_g} |W_csb[i, j]|
    quantized_group_max[g] = max_{i, j in group_g} |dequantize(W_q)[i, j]|

Output
------
One PNG + PDF per MMDiT block, plus one for special layers.
Each figure arranges subplots as [image, text] × [q, k, v, o, fc1, fc2].

Usage
-----
    conda run -n diffusionkit python -m src.phase2.plot_quantized_weight \\
        --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \\
        --output-dir plots/quantized_weight
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np


FAMILY_ORDER = {
    "q_proj": 0, "k_proj": 1, "v_proj": 2,
    "o_proj": 3, "fc1": 4, "fc2": 5,
}
SIDES = ["image", "text"]

SAFETENSOR_PREFIX_MAP = {
    "q_proj": "attn.q_proj",
    "k_proj": "attn.k_proj",
    "v_proj": "attn.v_proj",
    "o_proj": "attn.o_proj",
    "fc1": "mlp.fc1",
    "fc2": "mlp.fc2",
}


def layer_name_to_safetensor_prefix(name: str) -> str:
    """Map e.g. 'blocks.5.image.attn.q_proj' → safetensors key prefix."""
    if name == "final_layer.linear":
        return "final_layer.linear.qlinear"
    if name == "context_embedder":
        return "context_embedder.qlinear"
    parts = name.split(".")
    block = parts[1]
    side = parts[2]
    family = parts[4]
    side_key = "image_transformer_block" if side == "image" else "text_transformer_block"
    sub = SAFETENSOR_PREFIX_MAP[family]
    return f"multimodal_transformer_blocks.{block}.{side_key}.{sub}.qlinear"


def load_weight_stats(diag_dir: Path) -> dict[str, np.ndarray]:
    path = diag_dir / "weight_stats.npz"
    raw = np.load(path)
    stats: dict[str, np.ndarray] = {}
    for key in raw.files:
        if key.endswith("/w_channel_max"):
            layer_name = key.rsplit("/", 1)[0]
            stats[layer_name] = raw[key]
    return stats


def load_calibration_b(cal_dir: Path) -> dict[str, np.ndarray]:
    meta = json.loads((cal_dir / "calibration_meta.json").read_text())
    data = np.load(cal_dir / "calibration.npz")
    return {name: data[name] for name in meta["layer_names"]}


def parse_layer_name(name: str) -> tuple[int, str, str] | None:
    if name in ("context_embedder", "final_layer.linear"):
        return None
    parts = name.split(".")
    return int(parts[1]), parts[2], parts[4]


def compute_per_group(arr: np.ndarray, group_size: int) -> np.ndarray:
    """Per-group max from a per-channel array [d_in] → [n_groups]."""
    n = len(arr)
    n_groups = n // group_size
    return arr[: n_groups * group_size].reshape(n_groups, group_size).max(axis=1)


def _draw(ax, groups, orig, quant, title, is_bottom, is_left):
    ax.scatter(groups, orig, s=12, color="silver",
               edgecolors="none", alpha=0.6, label="Post-CSB (FP)", zorder=1)
    ax.scatter(groups, quant, s=12, color="#1f77b4",
               edgecolors="none", alpha=0.7, label="Dequantized (W4)", zorder=2)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    if is_bottom:
        ax.set_xlabel("Group index")
    if is_left:
        ax.set_ylabel("|W| group max")
    ax.legend(fontsize=6, loc="upper right")


def plot_block(
    block_idx: int,
    block_layers: dict[tuple[int, str], list[tuple[str, str]]],
    orig_groups: dict[str, np.ndarray],
    quant_groups: dict[str, np.ndarray],
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
        f"Block {block_idx} — Per-Group Weight Absmax: Post-CSB vs W4 Dequantized",
        fontsize=13, fontweight="bold",
    )
    for row, side in enumerate(sides_present):
        layers = block_layers[(block_idx, side)]
        for col, (name, family) in enumerate(layers):
            ax = axes[row, col]
            o = orig_groups.get(name)
            q = quant_groups.get(name)
            if o is None or q is None:
                ax.set_visible(False)
                continue
            _draw(ax, np.arange(len(o)), o, q,
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
    orig_groups: dict[str, np.ndarray],
    quant_groups: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    o = orig_groups.get(name)
    q = quant_groups.get(name)
    if o is None or q is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _draw(ax, np.arange(len(o)), o, q, title, True, True)
    plt.tight_layout()
    stem = name.replace(".", "_")
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-group quantized vs original weight absmax.",
    )
    parser.add_argument(
        "--diagnostics-dir", type=str, default="diagnostics",
    )
    parser.add_argument(
        "--quantized-dir", type=str, required=True,
        help="Directory with mmdit_quantized.safetensors + calibration.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots/quantized_weight",
    )
    args = parser.parse_args()

    diag_dir = Path(args.diagnostics_dir)
    q_dir    = Path(args.quantized_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((q_dir / "quantize_config.json").read_text())
    group_size = cfg["group_size"]
    bits = cfg["bits"]
    print(f"W{bits} group_size={group_size}")

    weight_pre = load_weight_stats(diag_dir)
    b_vectors = load_calibration_b(q_dir)
    safetensor_data = mx.load(str(q_dir / "mmdit_quantized.safetensors"))
    st_keys = set(safetensor_data.keys())

    calibrated_layers = set(b_vectors.keys()) & set(weight_pre.keys())
    print(f"Layers to plot: {len(calibrated_layers)}")

    orig_groups: dict[str, np.ndarray] = {}
    quant_groups: dict[str, np.ndarray] = {}

    for name in sorted(calibrated_layers):
        w_pre = weight_pre[name]
        b = b_vectors[name]
        w_post_csb = w_pre * b

        prefix = layer_name_to_safetensor_prefix(name)
        wk = f"{prefix}.weight"
        sk = f"{prefix}.scales"
        bk = f"{prefix}.biases"
        if wk not in st_keys:
            print(f"  SKIP {name} (key {wk} not found)")
            continue

        deq = mx.dequantize(safetensor_data[wk], safetensor_data[sk],
                            safetensor_data[bk], group_size, bits)
        mx.eval(deq)
        deq_np = np.array(deq).astype(np.float32)

        deq_ch_max = np.max(np.abs(deq_np), axis=0)

        orig_groups[name] = compute_per_group(w_post_csb, group_size)
        quant_groups[name] = compute_per_group(deq_ch_max, group_size)

    print(f"Dequantized {len(quant_groups)} layers")

    block_layers: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    special: list[tuple[str, str]] = []

    for name in quant_groups:
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
        plot_block(block_idx, block_layers, orig_groups, quant_groups, out_dir)
        print(f"  Block {block_idx:2d} ✓")

    for name, title in special:
        plot_single_layer(name, title, orig_groups, quant_groups, out_dir)
        print(f"  {title} ✓")

    total = len(blocks) + len(special)
    print(f"\nDone — {total} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
