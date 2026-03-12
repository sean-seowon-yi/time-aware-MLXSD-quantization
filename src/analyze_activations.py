"""
Analyze SD3 MM-DiT calibration data to reveal quantization challenges:
  1. Dual-stream activation scale heatmaps (img vs txt stream)
  2. Cross-stream scale divergence (joint attention quantization problem)
  3. Temporal activation drift under rectified flow
  4. adaLN shift magnitude contribution
  5. Per-channel outlier analysis (img vs txt)
  6. Block depth vs temporal variability
  7. Scale trajectory linearity analysis

Reads from calibration_data_100/activations/timestep_stats/*.npz

Usage:
    conda run -n diffusionkit python -m src.analyze_activations \
        --activations-dir calibration_data_100/activations \
        --output-dir analysis_results
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

warnings.filterwarnings("ignore")

# ─── helpers ──────────────────────────────────────────────────────────────────

def block_num(layer_raw: str) -> int:
    """Extract block number from layer name like mm0_img_attn_q_proj."""
    return int(layer_raw.split("_")[0][2:])


def stream(layer_raw: str) -> str:
    """Return 'img' or 'txt'."""
    parts = layer_raw.split("_")
    return parts[1] if len(parts) > 1 else "?"


def absmax_from_stats(data, layer_raw: str) -> float:
    """Compute per-layer mean absmax: mean over channels of max(|avg_max|, |avg_min|)."""
    mx = data.get(f"{layer_raw}__avg_max")
    mn = data.get(f"{layer_raw}__avg_min")
    if mx is None or mn is None:
        return np.nan
    return float(np.maximum(np.abs(mx), np.abs(mn)).mean())


def channel_absmax_vec(data, layer_raw: str):
    """Per-channel absmax vector."""
    mx = data.get(f"{layer_raw}__avg_max")
    mn = data.get(f"{layer_raw}__avg_min")
    if mx is None or mn is None:
        return None
    return np.maximum(np.abs(mx), np.abs(mn))


def load_step(ts_dir: Path, step: str):
    p = ts_dir / f"step_{step}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


# ─── main analysis ────────────────────────────────────────────────────────────

def run(activations_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = activations_dir / "layer_statistics.json"
    with open(meta_path) as f:
        meta = json.load(f)

    ts_dir = activations_dir / "timestep_stats"
    step_keys = sorted(meta["step_keys"], key=int)
    sigma_map = {k: float(v) for k, v in meta["sigma_map"].items()}
    sigmas = np.array([sigma_map[s] for s in step_keys])

    temporal_path = activations_dir / "layer_temporal_analysis.json"
    with open(temporal_path) as f:
        temporal = json.load(f)

    # Discover all layer names from first step
    data0 = load_step(ts_dir, step_keys[0])
    def _layer_sort_key(name: str):
        """Sort by block number (numeric), then stream, then sublayer name."""
        parts = name.split("_", 2)  # ['mmN', 'img'/'txt', 'rest']
        block = int(parts[0][2:]) if parts[0].startswith("mm") else 0
        stream_order = 0 if (len(parts) > 1 and parts[1] == "img") else 1
        rest = parts[2] if len(parts) > 2 else ""
        return (block, stream_order, rest)

    all_layers_raw = sorted({k.rsplit("__", 1)[0] for k in data0.keys()}, key=_layer_sort_key)
    img_layers = [l for l in all_layers_raw if "_img_" in l]
    txt_layers = [l for l in all_layers_raw if "_txt_" in l]

    dual_blocks = sorted({block_num(l) for l in img_layers})
    print(f"Dual-stream blocks: {len(dual_blocks)}  (0..{dual_blocks[-1]})")
    print(f"img layers: {len(img_layers)}, txt layers: {len(txt_layers)}")
    print(f"Timesteps: {len(step_keys)} steps  σ {sigmas[0]:.3f}→{sigmas[-1]:.3f}")

    # ── Build [layers × timesteps] absmax matrices ────────────────────────────
    print("\nLoading activation stats across all timesteps...")
    absmax_img = np.full((len(img_layers), len(step_keys)), np.nan)
    absmax_txt = np.full((len(txt_layers), len(step_keys)), np.nan)

    for si, step in enumerate(step_keys):
        d = load_step(ts_dir, step)
        if d is None:
            continue
        for li, lr in enumerate(img_layers):
            absmax_img[li, si] = absmax_from_stats(d, lr)
        for li, lr in enumerate(txt_layers):
            absmax_txt[li, si] = absmax_from_stats(d, lr)
        if si % 5 == 0:
            print(f"  step {step} (σ={sigmas[si]:.3f}) loaded")

    # ── Fig 1: Dual-stream scale heatmap ──────────────────────────────────────
    print("\n[Fig 1] Dual-stream scale heatmap")
    vmax_shared = np.nanpercentile(np.concatenate([absmax_img, absmax_txt]), 97)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    for ax, matrix, title, layers in [
        (axes[0], absmax_img, "Image stream (img tokens)", img_layers),
        (axes[1], absmax_txt, "Text stream (txt tokens)", txt_layers),
    ]:
        im = ax.imshow(
            matrix, aspect="auto", interpolation="nearest",
            cmap="inferno", vmin=0, vmax=vmax_shared,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Noise level σ (decreasing = denoising)", fontsize=10)
        ax.set_ylabel("Layer index", fontsize=10)
        xticks = list(range(0, len(step_keys), 4))
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{sigmas[i]:.2f}" for i in xticks], rotation=45, ha="right")
        yticks = list(range(0, len(layers), 6))
        ax.set_yticks(yticks)
        ax.set_yticklabels([layers[i] for i in yticks], fontsize=6)
        plt.colorbar(im, ax=ax, label="Mean absmax activation")

    fig.suptitle(
        "SD3 MM-DiT Dual-Stream Activation Scales Across Timesteps\n"
        "img and txt streams show qualitatively different scale profiles;\n"
        "per-tensor quantization assigns one clipping range to activations from both streams",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_dual_stream_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_dual_stream_heatmap.png")

    # ── Fig 2: Cross-stream scale divergence (attn q/k/v) ────────────────────
    print("\n[Fig 2] Cross-stream scale divergence (joint attention projections)")
    attn_projs = ["attn_q_proj", "attn_k_proj", "attn_v_proj"]
    q_ratios, k_ratios, v_ratios, blk_labels = [], [], [], []

    for b in dual_blocks:
        row = {}
        for proj in attn_projs:
            img_raw = f"mm{b}_img_{proj}"
            txt_raw = f"mm{b}_txt_{proj}"
            if img_raw in img_layers and txt_raw in txt_layers:
                iv = absmax_img[img_layers.index(img_raw)]
                tv = absmax_txt[txt_layers.index(txt_raw)]
                with np.errstate(divide="ignore", invalid="ignore"):
                    row[proj] = np.where(tv > 0, iv / tv, np.nan)
        if row:
            q_ratios.append(row.get("attn_q_proj", np.full(len(step_keys), np.nan)))
            k_ratios.append(row.get("attn_k_proj", np.full(len(step_keys), np.nan)))
            v_ratios.append(row.get("attn_v_proj", np.full(len(step_keys), np.nan)))
            blk_labels.append(f"mm{b}")

    if q_ratios:
        fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
        norm = mcolors.LogNorm(vmin=0.2, vmax=5.0)

        for ax, arr, proj_name in [
            (axes[0], np.array(q_ratios), "Q projection"),
            (axes[1], np.array(k_ratios), "K projection"),
            (axes[2], np.array(v_ratios), "V projection"),
        ]:
            im = ax.imshow(arr, aspect="auto", interpolation="nearest",
                           cmap="RdBu_r", norm=norm)
            ax.set_title(f"img/txt scale ratio — {proj_name}", fontsize=12)
            ax.set_xlabel("Noise level σ", fontsize=10)
            xticks = list(range(0, len(step_keys), 4))
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{sigmas[i]:.2f}" for i in xticks],
                               rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(blk_labels)))
            ax.set_yticklabels(blk_labels, fontsize=7)
            plt.colorbar(im, ax=ax, label="img/txt scale ratio (log scale)")

        # Add reference line at ratio=1 annotation
        for ax in axes:
            ax.text(0.02, 0.98, "white = ratio 1.0\n(perfect match)",
                    transform=ax.transAxes, fontsize=7, va="top",
                    color="gray", style="italic")

        fig.suptitle(
            "Cross-Stream Scale Divergence in MM-DiT Joint Attention\n"
            "Ratio ≠ 1 means img and txt activations have different dynamic ranges;\n"
            "a single per-tensor quantization range clips one stream or wastes bits on the other",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_dir / "fig2_cross_stream_divergence.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig2_cross_stream_divergence.png")

    # ── Fig 3: Temporal drift under rectified flow ────────────────────────────
    print("\n[Fig 3] Temporal activation drift (rectified flow)")
    # Select layers with the most interesting / diverse trajectory shapes.
    # Normalize each curve to [0,1] so shape differences are visible regardless of
    # absolute scale — we care about the trajectory, not the magnitude.
    candidates = [
        ("mm0_img_mlp_fc1",     "mm0 img mlp_fc1   (U-shape, trough σ≈0.48, 16% raw var)",   img_layers),
        ("mm18_txt_mlp_fc1",    "mm18 txt mlp_fc1  (hill: peaks σ≈0.82, 22% raw var)",        txt_layers),
        ("mm18_img_attn_q_proj","mm18 img q_proj   (monotone rise, 39% raw var)",              img_layers),
        ("mm0_img_attn_q_proj", "mm0 img q_proj    (peaks σ≈0.97, declines, 23% raw var)",    img_layers),
        ("mm9_img_mlp_fc1",     "mm9 img mlp_fc1   (U-shape, only 3.9% raw var)",             img_layers),
        ("mm18_img_mlp_fc1",    "mm18 img mlp_fc1  (rise then plateau, 16% raw var)",         img_layers),
    ]
    candidates = [(r, l, ll) for r, l, ll in candidates if r in ll]

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(18, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(candidates)))

    for (raw, label, ll), color in zip(candidates, colors):
        idx = ll.index(raw)
        matrix = absmax_img if ll is img_layers else absmax_txt
        vals = matrix[idx]
        ls = "--" if "txt" in raw else "-"
        # Raw values
        ax_raw.plot(sigmas, vals, ls + "o", markersize=3,
                    label=label, color=color, linewidth=1.5)
        # Normalised to [0,1] to expose trajectory shape
        vmin, vmax_v = np.nanmin(vals), np.nanmax(vals)
        norm_vals = (vals - vmin) / (vmax_v - vmin + 1e-10)
        ax_norm.plot(sigmas, norm_vals, ls + "o", markersize=3,
                     label=label, color=color, linewidth=1.5)

    for ax in (ax_raw, ax_norm):
        ax.axvline(0.7, color="gray", linestyle="--", alpha=0.6, linewidth=1.2)
        ax.axvline(0.3, color="gray", linestyle=":",  alpha=0.6, linewidth=1.2)
        ax.text(0.85, 0.02, "early\nbucket", transform=ax.transAxes,
                fontsize=7, color="gray", ha="center")
        ax.text(0.5, 0.02, "mid bucket", transform=ax.transAxes,
                fontsize=7, color="gray", ha="center")
        ax.text(0.15, 0.02, "late bucket", transform=ax.transAxes,
                fontsize=7, color="gray", ha="center")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Noise level σ  (right→left = denoising direction)", fontsize=10)

    ax_raw.set_ylabel("Mean absmax activation (raw)", fontsize=10)
    ax_raw.set_title("Raw activation scale trajectories", fontsize=11)
    ax_raw.legend(fontsize=7, loc="upper right")

    ax_norm.set_ylabel("Normalised scale (0=min, 1=max per layer)", fontsize=10)
    ax_norm.set_title("Normalised trajectories — shape comparison", fontsize=11)
    ax_norm.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Activation Scale Drift Across Denoising Steps — SD3 Rectified Flow Schedule\n"
        "Layers show opposite trends and non-monotone shapes; "
        "one calibration range per HTG bucket cannot fit all trajectories",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_temporal_drift_rectified_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_temporal_drift_rectified_flow.png")

    # ── Fig 3 sub-figures: one per sublayer type, all blocks as lines ─────────
    print("\n[Fig 3 sub-figures] Per-sublayer-type block trajectories")

    # k/q/v are always identical within a block — show q as the representative
    sublayer_specs = [
        ("attn_q_proj", "img", "a", "img  attn_q_proj  (= k_proj = v_proj)"),
        ("attn_o_proj", "img", "b", "img  attn_o_proj"),
        ("mlp_fc1",     "img", "c", "img  mlp_fc1"),
        ("mlp_fc2",     "img", "d", "img  mlp_fc2"),
        ("attn_q_proj", "txt", "e", "txt  attn_q_proj  (= k_proj = v_proj)"),
        ("attn_o_proj", "txt", "f", "txt  attn_o_proj"),
        ("mlp_fc1",     "txt", "g", "txt  mlp_fc1"),
        ("mlp_fc2",     "txt", "h", "txt  mlp_fc2"),
    ]

    cmap = plt.cm.plasma
    block_colors = {b: cmap(i / max(len(dual_blocks) - 1, 1))
                    for i, b in enumerate(dual_blocks)}

    for sublayer, stream_name, letter, title_suffix in sublayer_specs:
        layer_list = img_layers if stream_name == "img" else txt_layers
        matrix    = absmax_img  if stream_name == "img" else absmax_txt

        fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(16, 5))

        for b in dual_blocks:
            raw = f"mm{b}_{stream_name}_{sublayer}"
            if raw not in layer_list:
                continue
            idx  = layer_list.index(raw)
            vals = matrix[idx]
            color = block_colors[b]

            ax_raw.plot(sigmas, vals, "-", color=color, linewidth=1.2, alpha=0.85)
            ax_raw.text(sigmas[-1] - 0.01, vals[-1], f"mm{b}",
                        fontsize=5, color=color, va="center", ha="right")

            vmin_l, vmax_l = np.nanmin(vals), np.nanmax(vals)
            norm_vals = (vals - vmin_l) / (vmax_l - vmin_l + 1e-10)
            ax_norm.plot(sigmas, norm_vals, "-", color=color, linewidth=1.2, alpha=0.85)

        for ax in (ax_raw, ax_norm):
            ax.axvline(0.7, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.axvline(0.3, color="gray", linestyle=":",  alpha=0.5, linewidth=1)
            ax.invert_xaxis()
            ax.set_xlabel("Noise level σ  (right→left = denoising)", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.text(0.84, 0.97, "early", transform=ax.transAxes,
                    fontsize=7, color="gray", va="top", ha="center")
            ax.text(0.50, 0.97, "mid",   transform=ax.transAxes,
                    fontsize=7, color="gray", va="top", ha="center")
            ax.text(0.14, 0.97, "late",  transform=ax.transAxes,
                    fontsize=7, color="gray", va="top", ha="center")

        ax_raw.set_ylabel("Mean absmax activation", fontsize=9)
        ax_raw.set_title("Raw scale (colour = block depth: purple→yellow = mm0→mm23)", fontsize=9)

        ax_norm.set_ylabel("Normalised [0, 1] per block", fontsize=9)
        ax_norm.set_title("Normalised trajectories (shape only — ignores absolute magnitude)", fontsize=9)

        # Colourbar legend
        sm = plt.cm.ScalarMappable(cmap=cmap,
             norm=plt.Normalize(vmin=dual_blocks[0], vmax=dual_blocks[-1]))
        sm.set_array([])
        plt.colorbar(sm, ax=ax_norm, label="Block index (mm0 → mm23)", shrink=0.8)

        fig.suptitle(
            f"Fig 3{letter} — {title_suffix}\n"
            "All 24 MM-DiT blocks across σ  |  gray dashed = HTG bucket boundaries",
            fontsize=11,
        )
        fig.tight_layout()
        fname = f"fig3{letter}_{stream_name}_{sublayer}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── Fig 4: adaLN shift magnitude ──────────────────────────────────────────
    print("\n[Fig 4] adaLN shift magnitude")
    shift_summary = temporal.get("shift_summary", {})
    if shift_summary:
        shift_layers = sorted(shift_summary.keys(),
                              key=lambda l: -shift_summary[l]["max_shift_magnitude"])
        max_shifts = [shift_summary[l]["max_shift_magnitude"] for l in shift_layers]
        mean_shifts = [shift_summary[l]["mean_shift_magnitude"] for l in shift_layers]

        fig, ax = plt.subplots(figsize=(15, 6))
        x = np.arange(len(shift_layers))
        ax.bar(x, max_shifts, alpha=0.8, label="Max shift magnitude", color="crimson", width=0.8)
        ax.bar(x, mean_shifts, alpha=0.8, label="Mean shift magnitude", color="steelblue", width=0.8)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5,
                   label="Shift = 1.0 (equal to typical activation scale)")
        ax.set_xticks(x)
        ax.set_xticklabels(shift_layers, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("Activation shift magnitude (adaLN-modulated layers)", fontsize=11)
        ax.set_title(
            "adaLN Timestep-Conditioning Shift Across Denoising Steps\n"
            "High shift = the activation distribution translates significantly between timesteps;\n"
            "standard AdaRound minimises rounding error at one fixed distribution — shift invalidates this",
            fontsize=10,
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "fig4_adaln_shift_magnitude.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig4_adaln_shift_magnitude.png")
    else:
        print("  No shift_summary data — skipping.")

    # ── Fig 5: Per-channel outlier profiles ───────────────────────────────────
    print("\n[Fig 5] Per-channel outlier profiles (img vs txt)")
    mid_step = step_keys[len(step_keys) // 2]
    d_mid = load_step(ts_dir, mid_step)
    mid_sigma = sigma_map[mid_step]

    sample_blocks = [0, 9, 18, 27]
    sample_blocks = [b for b in sample_blocks if b in dual_blocks]

    fig, axes = plt.subplots(2, len(sample_blocks), figsize=(16, 8))
    if len(sample_blocks) == 1:
        axes = axes.reshape(2, 1)

    for col, b in enumerate(sample_blocks):
        for row, (stream_name, layer_list) in enumerate([
            ("img", img_layers), ("txt", txt_layers)
        ]):
            ax = axes[row, col]
            raw = f"mm{b}_{stream_name}_attn_q_proj"
            if raw not in layer_list:
                ax.set_visible(False)
                continue
            ch_vec = channel_absmax_vec(d_mid, raw)
            if ch_vec is None:
                ax.set_visible(False)
                continue
            sorted_vals = np.sort(ch_vec)[::-1]
            color = "steelblue" if stream_name == "img" else "darkorange"
            ax.semilogy(sorted_vals, marker=".", markersize=2, color=color, linewidth=0.8)
            ax.axhline(sorted_vals.mean(), color="red", linestyle="--",
                       linewidth=1.2, alpha=0.8, label=f"mean={sorted_vals.mean():.2f}")
            # Mark top-1% outlier boundary
            p99 = np.percentile(sorted_vals, 99)
            ax.axhline(p99, color="purple", linestyle=":", linewidth=1,
                       alpha=0.7, label=f"p99={p99:.2f}")
            ax.set_title(f"mm{b} {stream_name} q_proj\nσ={mid_sigma:.2f}", fontsize=9)
            ax.set_xlabel("Channel rank (sorted)", fontsize=8)
            ax.set_ylabel("Absmax", fontsize=8)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)

    axes[0, 0].set_ylabel("Image stream\nChannel absmax (log scale)", fontsize=9)
    axes[1, 0].set_ylabel("Text stream\nChannel absmax (log scale)", fontsize=9)
    fig.suptitle(
        "Per-Channel Activation Outliers: img vs txt Q-Projection  (σ≈{:.2f})\n"
        "Heavy-tailed channels force per-tensor quantizers to use a large range,\n"
        "wasting bits on typical channels; img and txt streams have different outlier severity".format(mid_sigma),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_channel_outliers_img_vs_txt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5_channel_outliers_img_vs_txt.png")

    # ── Fig 6: Block depth vs temporal scale variability ─────────────────────
    print("\n[Fig 6] Block depth vs temporal variability (CV)")
    sublayers_to_check = ["mlp_fc2", "attn_q_proj", "attn_v_proj", "mlp_fc1"]

    fig, axes = plt.subplots(len(sublayers_to_check), 1, figsize=(12, 10), sharex=True)

    for ax, sl in zip(axes, sublayers_to_check):
        img_cv, txt_cv, blocks_with_data = [], [], []
        for b in dual_blocks:
            img_raw = f"mm{b}_img_{sl}"
            txt_raw = f"mm{b}_txt_{sl}"
            if img_raw not in img_layers or txt_raw not in txt_layers:
                continue
            iv = absmax_img[img_layers.index(img_raw)]
            tv = absmax_txt[txt_layers.index(txt_raw)]
            if np.nanmean(iv) > 0 and np.nanmean(tv) > 0:
                img_cv.append(np.nanstd(iv) / np.nanmean(iv))
                txt_cv.append(np.nanstd(tv) / np.nanmean(tv))
                blocks_with_data.append(b)

        if blocks_with_data:
            x = np.array(blocks_with_data)
            ax.plot(x, img_cv, "b-o", markersize=4, label="img stream", linewidth=1.5)
            ax.plot(x, txt_cv, "r-s", markersize=4, label="txt stream", linewidth=1.5)
            ax.axhline(0.2, color="gray", linestyle="--", linewidth=1, alpha=0.5,
                       label="CV=0.2 (moderate drift)")
            ax.set_ylabel("CV (σ/μ)", fontsize=9)
            ax.set_title(f"{sl}", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("MM-DiT block index (deeper = later in network)", fontsize=11)
    fig.suptitle(
        "Temporal Variability of Activation Scales by Block Depth\n"
        "CV = std/mean of absmax across all σ steps;\n"
        "High CV means the layer's scale shifts substantially — calibration at one timestep is insufficient",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_depth_vs_temporal_variability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig6_depth_vs_temporal_variability.png")

    # ── Fig 7: Cross-stream ratio summary heatmap + distribution ─────────────
    print("\n[Fig 7] Summary: per-block img/txt scale ratio")
    img_mean_per_block_step = np.full((len(dual_blocks), len(step_keys)), np.nan)
    txt_mean_per_block_step = np.full((len(dual_blocks), len(step_keys)), np.nan)

    for bi, b in enumerate(dual_blocks):
        b_img = [l for l in img_layers if l.startswith(f"mm{b}_")]
        b_txt = [l for l in txt_layers if l.startswith(f"mm{b}_")]
        for si in range(len(step_keys)):
            img_vals = [absmax_img[img_layers.index(l), si] for l in b_img]
            txt_vals = [absmax_txt[txt_layers.index(l), si] for l in b_txt]
            img_mean_per_block_step[bi, si] = np.nanmean(img_vals) if img_vals else np.nan
            txt_mean_per_block_step[bi, si] = np.nanmean(txt_vals) if txt_vals else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_arr = np.where(
            txt_mean_per_block_step > 0,
            img_mean_per_block_step / txt_mean_per_block_step,
            np.nan,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im = ax1.imshow(ratio_arr, aspect="auto", interpolation="nearest",
                    cmap="RdBu_r", norm=mcolors.LogNorm(vmin=0.25, vmax=4.0))
    ax1.set_xlabel("Noise level σ", fontsize=10)
    ax1.set_ylabel("MM-DiT block index", fontsize=10)
    xticks = list(range(0, len(step_keys), 4))
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{sigmas[i]:.2f}" for i in xticks], rotation=45, ha="right")
    ax1.set_yticks(range(0, len(dual_blocks), 4))
    ax1.set_yticklabels([str(dual_blocks[i]) for i in range(0, len(dual_blocks), 4)])
    plt.colorbar(im, ax=ax1, label="img/txt mean absmax ratio (log)")
    ax1.set_title("Mean activation scale ratio: img / txt  per block × σ\n"
                  "(white/gray = balanced; red/blue = mismatched)", fontsize=10)

    valid = ratio_arr[~np.isnan(ratio_arr)].flatten()
    ax2.hist(np.log2(valid), bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Perfect match (ratio=1)")
    ax2.axvline(np.log2(1.5), color="orange", linestyle=":", linewidth=1.5,
                label=f"1.5× mismatch ({np.mean(valid > 1.5)*100:.1f}% of pairs)")
    ax2.axvline(-np.log2(1.5), color="orange", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("log₂(img/txt scale ratio)", fontsize=10)
    ax2.set_ylabel("Count (block × timestep pairs)", fontsize=10)
    ax2.set_title(
        f"Distribution of cross-stream scale ratios\n"
        f"Median={np.median(valid):.2f}, "
        f">1.5×: {np.mean(valid > 1.5)*100:.1f}%,  "
        f"<0.67×: {np.mean(valid < 0.67)*100:.1f}%",
        fontsize=10,
    )
    ax2.legend()

    fig.suptitle(
        "SD3 MM-DiT Cross-Stream Activation Scale Mismatch\n"
        "Core evidence that per-tensor quantization of joint attention is suboptimal",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig7_cross_stream_ratio_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig7_cross_stream_ratio_summary.png")

    # ── Fig 8: Scale trajectory linearity (rectified flow) ───────────────────
    print("\n[Fig 8] Activation scale trajectory: linear vs quadratic fit in σ")
    candidates_for_fit = [
        ("mm0_img_mlp_fc2",  "block 0 img mlp_fc2",  img_layers),
        ("mm9_img_mlp_fc2",  "block 9 img mlp_fc2",  img_layers),
        ("mm18_img_mlp_fc2", "block 18 img mlp_fc2", img_layers),
        ("mm27_img_mlp_fc2", "block 27 img mlp_fc2", img_layers),
        ("mm0_txt_mlp_fc2",  "block 0 txt mlp_fc2",  txt_layers),
        ("mm18_txt_mlp_fc2", "block 18 txt mlp_fc2", txt_layers),
    ]
    candidates_for_fit = [(r, l, ll) for r, l, ll in candidates_for_fit if r in ll]

    fig, axes = plt.subplots(1, len(candidates_for_fit),
                             figsize=(4 * len(candidates_for_fit), 5))
    if len(candidates_for_fit) == 1:
        axes = [axes]

    for ax, (raw, label, ll) in zip(axes, candidates_for_fit):
        idx = ll.index(raw)
        matrix = absmax_img if ll is img_layers else absmax_txt
        vals = matrix[idx]
        mask = ~np.isnan(vals)
        x = sigmas[mask]
        y = vals[mask]
        ax.scatter(x, y, s=20, zorder=5, color="black", label="data", alpha=0.8)
        xf = np.linspace(x.min(), x.max(), 100)

        p1 = np.polyfit(x, y, 1)
        y_lin = np.polyval(p1, xf)
        ss_res = np.sum((y - np.polyval(p1, x)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.plot(xf, y_lin, "b--", linewidth=1.5, label=f"Linear R²={r2_lin:.3f}")

        p2 = np.polyfit(x, y, 2)
        y_quad = np.polyval(p2, xf)
        ss_res2 = np.sum((y - np.polyval(p2, x)) ** 2)
        r2_quad = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0
        ax.plot(xf, y_quad, "r-", linewidth=1.5, label=f"Quadratic R²={r2_quad:.3f}")

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("σ (denoising →)", fontsize=8)
        ax.set_ylabel("Mean absmax", fontsize=8)
        ax.legend(fontsize=7)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Activation Scale vs σ: Is the Drift Linear?\n"
        "Low linear R² or higher quadratic R² indicates non-linear drift;\n"
        "HTG timestep groupings based on σ spacing will be misaligned",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig8_scale_linearity_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig8_scale_linearity_analysis.png")

    # ── Summary statistics ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY STATISTICS")
    print("=" * 65)

    valid = ratio_arr[~np.isnan(ratio_arr)].flatten()
    print(f"\nCross-stream (img/txt) activation scale ratio:")
    print(f"  Median:  {np.median(valid):.3f}  (1.0 = perfectly matched)")
    print(f"  Mean:    {np.mean(valid):.3f}")
    print(f"  Std:     {np.std(valid):.3f}")
    print(f"  > 1.5×:  {np.mean(valid > 1.5)*100:.1f}% of (block, σ) pairs")
    print(f"  < 0.67×: {np.mean(valid < 0.67)*100:.1f}% of (block, σ) pairs")
    print(f"  Worst block: "
          f"mm{dual_blocks[np.argmax(np.nanmean(np.abs(np.log(np.clip(ratio_arr, 0.01, 100))), axis=1))]} "
          f"(highest log-ratio magnitude)")

    if shift_summary:
        high_shift = {k: v for k, v in shift_summary.items()
                      if v["max_shift_magnitude"] > 2.0}
        print(f"\nadaLN layers with shift > 2.0 activation units: {len(high_shift)}")
        for k in sorted(high_shift, key=lambda l: -high_shift[l]["max_shift_magnitude"])[:8]:
            print(f"  {k}: max={high_shift[k]['max_shift_magnitude']:.2f}  "
                  f"mean={high_shift[k]['mean_shift_magnitude']:.2f}")

    print(f"\nTop 10 layers by temporal variability (CV = std/mean across σ steps):")
    cvs = []
    for li, lr in enumerate(img_layers):
        v = absmax_img[li]
        if np.nanmean(v) > 0:
            cvs.append((np.nanstd(v) / np.nanmean(v), lr, "img"))
    for li, lr in enumerate(txt_layers):
        v = absmax_txt[li]
        if np.nanmean(v) > 0:
            cvs.append((np.nanstd(v) / np.nanmean(v), lr, "txt"))
    cvs.sort(reverse=True)
    for cv, lr, st in cvs[:10]:
        print(f"  {lr} ({st}): CV={cv:.3f}")

    print(f"\nAll figures saved to: {output_dir}/")
    print("  fig1_dual_stream_heatmap.png")
    print("  fig2_cross_stream_divergence.png")
    print("  fig3_temporal_drift_rectified_flow.png")
    print("  fig4_adaln_shift_magnitude.png")
    print("  fig5_channel_outliers_img_vs_txt.png")
    print("  fig6_depth_vs_temporal_variability.png")
    print("  fig7_cross_stream_ratio_summary.png")
    print("  fig8_scale_linearity_analysis.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SD3 MM-DiT calibration activations for quantization challenges"
    )
    parser.add_argument(
        "--activations-dir", default="calibration_data_100/activations",
        help="Path to activations directory (default: calibration_data_100/activations)",
    )
    parser.add_argument(
        "--output-dir", default="analysis_results",
        help="Directory to save figures (default: analysis_results)",
    )
    args = parser.parse_args()

    base = Path(".")
    activations_dir = base / args.activations_dir
    output_dir = base / args.output_dir

    if not activations_dir.exists():
        print(f"ERROR: {activations_dir} does not exist")
        return

    run(activations_dir, output_dir)


if __name__ == "__main__":
    main()
