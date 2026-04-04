#!/usr/bin/env python3
"""Plot quantization MSE vs block depth for weights (W4) and activations (A8).

Standalone pipeline producing two figures:

1. **Weight MSE** — Analytical W4 quantization noise per layer:
       MSE_w = mean(scales²) / 12
   where ``scales`` are the per-group quantization scales from
   ``nn.QuantizedLinear``.

2. **Activation MSE** — Expected dynamic A8 quantization noise per layer:
       MSE_a = mean_t( (max_j(act[t,j]/b[j]) / 127)² ) / 12
   averaged across all denoising timesteps.

Each figure has two subplots (image / text) with one line per layer family
(q_proj, k_proj, v_proj, o_proj, fc1, fc2).

Usage
-----
    conda run -n diffusionkit python -m src.phase2.plot_mse_vs_block \\
        --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \\
        --output-dir plots/mse_vs_block
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

FAMILY_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
FAMILY_COLORS = {
    "q_proj": "#1f77b4", "k_proj": "#ff7f0e", "v_proj": "#2ca02c",
    "o_proj": "#d62728", "fc1": "#9467bd", "fc2": "#8c564b",
}
SIDES = ["image", "text"]

SAFETENSOR_SUB = {
    "q_proj": "attn.q_proj", "k_proj": "attn.k_proj", "v_proj": "attn.v_proj",
    "o_proj": "attn.o_proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2",
}


def layer_to_st_prefix(name: str) -> str:
    if name == "final_layer.linear":
        return "final_layer.linear.qlinear"
    parts = name.split(".")
    blk, side, fam = parts[1], parts[2], parts[4]
    side_key = "image_transformer_block" if side == "image" else "text_transformer_block"
    return f"multimodal_transformer_blocks.{blk}.{side_key}.{SAFETENSOR_SUB[fam]}.qlinear"


def parse_layer(name: str) -> tuple[int, str, str] | None:
    if name in ("context_embedder", "final_layer.linear"):
        return None
    parts = name.split(".")
    return int(parts[1]), parts[2], parts[4]


def compute_weight_mse(
    st_data: dict,
    layer_names: list[str],
    group_size: int,
    bits: int,
) -> dict[str, float]:
    """Analytical W4 MSE = mean(scales²) / 12 per layer."""
    mse: dict[str, float] = {}
    for name in layer_names:
        prefix = layer_to_st_prefix(name)
        sk = f"{prefix}.scales"
        if sk not in st_data:
            continue
        s = np.array(st_data[sk]).astype(np.float32)
        mse[name] = float(np.mean(s ** 2) / 12.0)
    return mse


def compute_activation_mse(
    diag_dir: Path,
    b_vectors: dict[str, np.ndarray],
) -> dict[str, float]:
    """Expected dynamic A8 MSE = mean_t( scale_t² ) / 12 per layer."""
    mse: dict[str, float] = {}
    for name, b in b_vectors.items():
        act_path = diag_dir / "activation_stats" / f"{name}.npz"
        if not act_path.exists():
            continue
        raw = np.load(act_path)["act_channel_max"]        # [T, d_in]
        post_csb_absmax = (raw / b[None, :]).max(axis=1)  # [T]
        scale_t = post_csb_absmax / 127.0
        mse[name] = float(np.mean(scale_t ** 2) / 12.0)
    return mse


def _plot_mse(
    mse: dict[str, float],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    # Organize by (block, side, family)
    data: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    special: dict[str, float] = {}

    for name, val in mse.items():
        parsed = parse_layer(name)
        if parsed is None:
            special[name] = val
            continue
        block_idx, side, family = parsed
        data[(side, family)].append((block_idx, val))

    for key in data:
        data[key].sort()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for col, side in enumerate(SIDES):
        ax = axes[col]
        for family in FAMILY_ORDER:
            key = (side, family)
            if key not in data:
                continue
            blocks, vals = zip(*data[key])
            ax.plot(blocks, vals, marker="o", markersize=4, lw=1.5,
                    color=FAMILY_COLORS[family], label=family)

        if "final_layer.linear" in special and col == 0:
            ax.axhline(special["final_layer.linear"], color="gray",
                       ls="--", lw=1, label="final_layer")

        ax.set_title(f"{side} side", fontsize=12)
        ax.set_xlabel("Block index")
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(range(0, 24, 2))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot quantization MSE vs block depth.",
    )
    parser.add_argument("--diagnostics-dir", type=str, default="diagnostics")
    parser.add_argument("--quantized-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="plots/mse_vs_block")
    args = parser.parse_args()

    diag_dir = Path(args.diagnostics_dir)
    q_dir    = Path(args.quantized_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((q_dir / "quantize_config.json").read_text())
    group_size = cfg["group_size"]
    bits = cfg["bits"]

    # Calibration b-vectors
    meta = json.loads((q_dir / "calibration_meta.json").read_text())
    cal_data = np.load(q_dir / "calibration.npz")
    b_vectors = {n: cal_data[n] for n in meta["layer_names"]}

    # Quantized model
    st_data = mx.load(str(q_dir / "mmdit_quantized.safetensors"))

    # Weight MSE
    print("Computing weight MSE (W4) ...")
    w_mse = compute_weight_mse(st_data, list(b_vectors.keys()), group_size, bits)
    print(f"  {len(w_mse)} layers")
    _plot_mse(
        w_mse,
        title=f"W{bits} Weight Quantization MSE vs Block (gs={group_size})",
        ylabel="MSE  [ mean(scale²)/12 ]",
        out_path=out_dir / "weight_mse",
    )

    # Activation MSE
    print("Computing activation MSE (A8 dynamic) ...")
    a_mse = compute_activation_mse(diag_dir, b_vectors)
    print(f"  {len(a_mse)} layers")
    _plot_mse(
        a_mse,
        title="A8 Dynamic Activation Quantization MSE vs Block",
        ylabel="MSE  [ mean_t(scale_t²)/12 ]",
        out_path=out_dir / "activation_mse",
    )

    print(f"\nDone — 2 figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
