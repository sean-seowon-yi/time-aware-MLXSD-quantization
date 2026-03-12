"""
Curve fitting exploration for SD3 MM-DiT activation scale trajectories.

For each layer, fits linear / quadratic / cubic models to the absmax vs σ
trajectory and generates diagnostic plots:

  fig_cf1  — Per-sublayer-type grid: all blocks overlaid, img (blue) vs txt
              (red/orange), data points + quadratic fit lines.
  fig_cf2  — R² improvement: linear vs quadratic scatter, coloured by
              sublayer type.  Highlights how much the quadratic gains.
  fig_cf3  — "Opposite directions": img vs txt attn_q_proj trajectories with
              quadratic fits, showing streams diverge in opposite directions.
  fig_cf4  — Fit-quality heatmap: quadratic R² for every (block, sublayer)
              combination, split img / txt.

Usage:
    conda run -n diffusionkit python -m src.explore_curve_fits \
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

# ── helpers ───────────────────────────────────────────────────────────────────

SUBLAYER_ORDER = [
    "attn_q_proj", "attn_k_proj", "attn_v_proj",
    "attn_o_proj", "mlp_fc1", "mlp_fc2",
]

IMG_CMAP = plt.cm.Blues
TXT_CMAP = plt.cm.Reds


def block_num(layer_raw: str) -> int:
    return int(layer_raw.split("_")[0][2:])


def sublayer_type(layer_raw: str) -> str:
    """e.g. mm3_img_attn_q_proj -> attn_q_proj"""
    parts = layer_raw.split("_", 2)
    return parts[2] if len(parts) > 2 else layer_raw


def stream(layer_raw: str) -> str:
    parts = layer_raw.split("_")
    return parts[1] if len(parts) > 1 else "?"


def absmax_from_step(data, layer_raw: str) -> float:
    mx = data.get(f"{layer_raw}__avg_max")
    mn = data.get(f"{layer_raw}__avg_min")
    if mx is None or mn is None:
        return np.nan
    return float(np.maximum(np.abs(mx), np.abs(mn)).mean())


def load_step(ts_dir: Path, step: str):
    p = ts_dir / f"step_{step}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def poly_r2(x, y, deg):
    coeffs = np.polyfit(x, y, deg)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return r2, coeffs


# ── data loading ──────────────────────────────────────────────────────────────

def load_trajectories(activations_dir: Path):
    """Return dict: layer_raw -> (sigmas_array, absmax_array)."""
    meta_path = activations_dir / "layer_statistics.json"
    with open(meta_path) as f:
        meta = json.load(f)

    ts_dir = activations_dir / "timestep_stats"
    step_keys = sorted(meta["step_keys"], key=int)
    sigma_map = {k: float(v) for k, v in meta["sigma_map"].items()}
    sigmas = np.array([sigma_map[s] for s in step_keys])

    # Discover layer names from first step
    data0 = load_step(ts_dir, step_keys[0])
    all_layers = sorted(
        {k.rsplit("__", 1)[0] for k in data0.keys()},
        key=lambda n: (block_num(n), 0 if "_img_" in n else 1, sublayer_type(n)),
    )

    # Build trajectories
    trajectories = {}
    step_data = {s: load_step(ts_dir, s) for s in step_keys}

    for layer in all_layers:
        vals = np.array([
            absmax_from_step(step_data[s], layer)
            for s in step_keys
        ])
        mask = ~np.isnan(vals)
        if mask.sum() > 3:
            trajectories[layer] = (sigmas[mask], vals[mask])

    return trajectories, sigmas


def load_percentile_trajectories(activations_dir: Path):
    """Return dict of percentile -> {layer_raw -> (sigmas, values)}.

    Loads p999 (99.9th), p99 (99th), and tensor_absmax (100th) from each step's
    index JSON, building per-layer trajectories at each percentile level.
    Also returns a 'mean_absmax' trajectory (from avg_min/avg_max arrays).
    """
    meta_path = activations_dir / "layer_statistics.json"
    with open(meta_path) as f:
        meta = json.load(f)

    ts_dir = activations_dir / "timestep_stats"
    step_keys = sorted(meta["step_keys"], key=int)
    sigma_map = {k: float(v) for k, v in meta["sigma_map"].items()}
    sigmas = np.array([sigma_map[s] for s in step_keys])

    # Load all step indices
    step_indices = {}
    step_npz = {}
    for s in step_keys:
        idx_path = ts_dir / f"step_{s}_index.json"
        with open(idx_path) as f:
            step_indices[s] = json.load(f)
        step_npz[s] = load_step(ts_dir, s)

    # Discover layers
    all_layers_dot = sorted(step_indices[step_keys[0]].keys())

    percentile_trajs = {}  # percentile_name -> {layer_raw -> (sigmas, values)}

    for pname, field in [("p100_absmax", "tensor_absmax"),
                          ("p999", "hist_p999"),
                          ("p99", "hist_p99")]:
        trajs = {}
        for layer_dot in all_layers_dot:
            vals = []
            for s in step_keys:
                info = step_indices[s].get(layer_dot, {})
                v = info.get(field)
                if v is not None:
                    # For percentiles, take abs value (they can be negative for min-side)
                    vals.append(abs(float(v)))
                else:
                    vals.append(np.nan)
            vals = np.array(vals)
            mask = ~np.isnan(vals)
            if mask.sum() > 3:
                layer_raw = layer_dot.replace(".", "_")
                trajs[layer_raw] = (sigmas[mask], vals[mask])
        percentile_trajs[pname] = trajs

    # Also add mean_absmax (from npz arrays)
    trajs = {}
    for layer_dot in all_layers_dot:
        layer_raw = layer_dot.replace(".", "_")
        vals = np.array([
            absmax_from_step(step_npz[s], layer_raw)
            for s in step_keys
        ])
        mask = ~np.isnan(vals)
        if mask.sum() > 3:
            trajs[layer_raw] = (sigmas[mask], vals[mask])
    percentile_trajs["mean_absmax"] = trajs

    return percentile_trajs, sigmas


# ── figures ───────────────────────────────────────────────────────────────────

def fig_cf1_per_sublayer_fits(trajectories, output_dir: Path):
    """Grid of sublayer types; all blocks overlaid, img blue / txt red, with quad fits."""
    print("[fig_cf1] Per-sublayer-type curve fits (img vs txt, all blocks)")

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
    fig.suptitle(
        "Quadratic fits to activation absmax vs σ  |  blue = img stream, red = txt stream\n"
        "dots = data, solid line = quadratic fit",
        fontsize=12,
    )

    for ax_idx, stype in enumerate(SUBLAYER_ORDER):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        ax.set_title(stype, fontsize=10)
        ax.set_xlabel("σ (noise level)")
        ax.set_ylabel("mean absmax")
        ax.invert_xaxis()

        img_layers = sorted(
            [l for l in trajectories if sublayer_type(l) == stype and "_img_" in l],
            key=block_num,
        )
        txt_layers = sorted(
            [l for l in trajectories if sublayer_type(l) == stype and "_txt_" in l],
            key=block_num,
        )
        n_img = max(len(img_layers), 1)
        n_txt = max(len(txt_layers), 1)

        for i, layer in enumerate(img_layers):
            sigmas, vals = trajectories[layer]
            alpha = 0.5 + 0.5 * i / n_img
            col = IMG_CMAP(0.4 + 0.5 * i / n_img)
            ax.scatter(sigmas, vals, s=8, color=col, alpha=0.6, zorder=2)
            _, coeffs = poly_r2(sigmas, vals, 2)
            xf = np.linspace(sigmas.min(), sigmas.max(), 100)
            ax.plot(xf, np.polyval(coeffs, xf), color=col, linewidth=1.2, alpha=alpha)

        for i, layer in enumerate(txt_layers):
            sigmas, vals = trajectories[layer]
            alpha = 0.5 + 0.5 * i / n_txt
            col = TXT_CMAP(0.4 + 0.5 * i / n_txt)
            ax.scatter(sigmas, vals, s=8, color=col, alpha=0.6, zorder=2)
            _, coeffs = poly_r2(sigmas, vals, 2)
            xf = np.linspace(sigmas.min(), sigmas.max(), 100)
            ax.plot(xf, np.polyval(coeffs, xf), color=col, linewidth=1.2,
                    linestyle="--", alpha=alpha)

        # legend proxy
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=IMG_CMAP(0.65), lw=2, label="img"),
            Line2D([0], [0], color=TXT_CMAP(0.65), lw=2, ls="--", label="txt"),
        ]
        ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    out = output_dir / "fig_cf1_per_sublayer_fits.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def fig_cf2_r2_improvement(trajectories, output_dir: Path):
    """Scatter: linear R² vs quadratic R² for every layer, coloured by sublayer type."""
    print("[fig_cf2] R² improvement: linear vs quadratic")

    cmap = plt.cm.tab10
    stype_colors = {s: cmap(i / len(SUBLAYER_ORDER)) for i, s in enumerate(SUBLAYER_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Linear vs Quadratic fit quality (R²) across all layers", fontsize=12)

    ax_scatter = axes[0]
    ax_hist = axes[1]

    r2_lin_all, r2_quad_all, colors_all, labels_all = [], [], [], []

    for layer, (sigmas, vals) in trajectories.items():
        stype = sublayer_type(layer)
        r2_lin, _ = poly_r2(sigmas, vals, 1)
        r2_quad, _ = poly_r2(sigmas, vals, 2)
        r2_lin_all.append(r2_lin)
        r2_quad_all.append(r2_quad)
        colors_all.append(stype_colors.get(stype, "gray"))
        labels_all.append(stype)

    r2_lin_all = np.array(r2_lin_all)
    r2_quad_all = np.array(r2_quad_all)

    # Scatter
    ax_scatter.scatter(r2_lin_all, r2_quad_all, c=colors_all, s=20, alpha=0.7, zorder=3)
    lim = [min(r2_lin_all.min(), r2_quad_all.min()) - 0.05, 1.02]
    ax_scatter.plot(lim, lim, "k--", linewidth=1, label="linear = quadratic", zorder=2)
    ax_scatter.set_xlim(lim)
    ax_scatter.set_ylim(lim)
    ax_scatter.set_xlabel("Linear R²")
    ax_scatter.set_ylabel("Quadratic R²")
    ax_scatter.set_title("Points above diagonal: quadratic gains over linear")

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=stype_colors[s],
                      markersize=8, label=s) for s in SUBLAYER_ORDER]
    ax_scatter.legend(handles=handles, fontsize=8)

    # Histogram of improvement
    improvement = r2_quad_all - r2_lin_all
    ax_hist.hist(improvement, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    ax_hist.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_hist.axvline(np.median(improvement), color="red", linestyle="-", linewidth=1.5,
                    label=f"median Δ = {np.median(improvement):.3f}")
    ax_hist.set_xlabel("Quadratic R² − Linear R²  (improvement)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of R² improvement from adding quadratic term")
    ax_hist.legend(fontsize=9)

    # Print summary stats
    print(f"  Layers with quad R² > 0.9:  {(r2_quad_all > 0.9).sum()} / {len(r2_quad_all)}")
    print(f"  Layers with quad > lin + 0.1: {(improvement > 0.1).sum()} / {len(r2_quad_all)}")
    print(f"  Median improvement: {np.median(improvement):.3f}")

    plt.tight_layout()
    out = output_dir / "fig_cf2_r2_improvement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def fig_cf3_opposite_directions(trajectories, output_dir: Path):
    """Show img vs txt attn_q_proj trajectories + fits to confirm opposite drift directions."""
    print("[fig_cf3] Opposite drift directions: img vs txt attn_q_proj")

    img_layers = sorted(
        [l for l in trajectories if sublayer_type(l) == "attn_q_proj" and "_img_" in l],
        key=block_num,
    )
    txt_layers = sorted(
        [l for l in trajectories if sublayer_type(l) == "attn_q_proj" and "_txt_" in l],
        key=block_num,
    )

    plasma = plt.cm.plasma
    n_blocks = 24

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "img vs txt attn_q_proj: opposite temporal drift directions\n"
        "Left = raw absmax, Right = normalised [0,1] per block",
        fontsize=12,
    )

    for row, (layers, label, cmap) in enumerate([
        (img_layers, "img", IMG_CMAP),
        (txt_layers, "txt", TXT_CMAP),
    ]):
        ax_raw = axes[row][0]
        ax_norm = axes[row][1]
        ax_raw.set_title(f"{label} stream — raw absmax", fontsize=10)
        ax_norm.set_title(f"{label} stream — normalised [0,1]", fontsize=10)

        for ax in (ax_raw, ax_norm):
            ax.set_xlabel("σ")
            ax.set_ylabel("absmax")
            ax.invert_xaxis()

        for i, layer in enumerate(layers):
            sigmas, vals = trajectories[layer]
            blk = block_num(layer)
            col = plasma(blk / n_blocks)
            r2_q, coeffs = poly_r2(sigmas, vals, 2)
            xf = np.linspace(sigmas.min(), sigmas.max(), 200)
            yf = np.polyval(coeffs, xf)

            ax_raw.scatter(sigmas, vals, s=6, color=col, alpha=0.5, zorder=3)
            ax_raw.plot(xf, yf, color=col, linewidth=1.0, alpha=0.8)

            vmin, vmax = vals.min(), vals.max()
            vals_n = (vals - vmin) / (vmax - vmin) if vmax > vmin else vals * 0
            yf_n = (yf - vmin) / (vmax - vmin) if vmax > vmin else yf * 0
            ax_norm.scatter(sigmas, vals_n, s=6, color=col, alpha=0.5, zorder=3)
            ax_norm.plot(xf, yf_n, color=col, linewidth=1.0, alpha=0.8)

        # Colourbar for block depth
        sm = plt.cm.ScalarMappable(cmap=plasma, norm=mcolors.Normalize(0, n_blocks - 1))
        sm.set_array([])
        for ax in (ax_raw, ax_norm):
            cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("block index", fontsize=8)

    plt.tight_layout()
    out = output_dir / "fig_cf3_opposite_directions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def fig_cf4_r2_heatmap(trajectories, output_dir: Path):
    """Heatmap of quadratic R² per (block, sublayer_type) for img and txt."""
    print("[fig_cf4] Quadratic R² heatmap: block × sublayer type")

    n_blocks = 24
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Quadratic R² per block × sublayer type  (higher = tighter fit)", fontsize=12)

    for ax, stream_name in zip(axes, ["img", "txt"]):
        matrix = np.full((len(SUBLAYER_ORDER), n_blocks), np.nan)
        for layer, (sigmas, vals) in trajectories.items():
            if stream(layer) != stream_name:
                continue
            stype = sublayer_type(layer)
            if stype not in SUBLAYER_ORDER:
                continue
            blk = block_num(layer)
            r2_q, _ = poly_r2(sigmas, vals, 2)
            row = SUBLAYER_ORDER.index(stype)
            if blk < n_blocks:
                matrix[row, blk] = r2_q

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n_blocks))
        ax.set_xticklabels([str(i) for i in range(n_blocks)], fontsize=7)
        ax.set_yticks(range(len(SUBLAYER_ORDER)))
        ax.set_yticklabels(SUBLAYER_ORDER, fontsize=9)
        ax.set_xlabel("Block index")
        ax.set_title(f"{stream_name} stream")

        # Annotate cells with R² value
        for r in range(len(SUBLAYER_ORDER)):
            for c in range(n_blocks):
                v = matrix[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5, color="black" if v > 0.3 else "white")

        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Quadratic R²")

    plt.tight_layout()
    out = output_dir / "fig_cf4_r2_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def fig_cf5_cubic_vs_quadratic(trajectories, output_dir: Path):
    """Identify layers where cubic meaningfully improves over quadratic, and show their fits."""
    print("[fig_cf5] Cubic vs quadratic: where does the extra parameter help?")

    # Compute all R² values
    records = []
    for layer, (sigmas, vals) in trajectories.items():
        r2_q, coeff_q = poly_r2(sigmas, vals, 2)
        r2_c, coeff_c = poly_r2(sigmas, vals, 3)
        records.append({
            "layer": layer, "stream": stream(layer), "stype": sublayer_type(layer),
            "block": block_num(layer), "r2_q": r2_q, "r2_c": r2_c,
            "delta": r2_c - r2_q, "coeff_q": coeff_q, "coeff_c": coeff_c,
            "sigmas": sigmas, "vals": vals,
        })

    # ── Panel layout: 3 rows ──
    # Row 1: scatter of quad R² vs cubic R², + histogram of improvement
    # Row 2: the top-N layers where cubic gains most, with both fits overlaid
    # Row 3: heatmap of cubic-minus-quadratic R² gain by block × sublayer

    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1], hspace=0.35, wspace=0.3)

    # ── Row 1 left: scatter ──
    ax_scat = fig.add_subplot(gs[0, 0])
    r2_q_arr = np.array([r["r2_q"] for r in records])
    r2_c_arr = np.array([r["r2_c"] for r in records])
    delta_arr = r2_c_arr - r2_q_arr

    cmap_scat = plt.cm.tab10
    stype_colors = {s: cmap_scat(i / len(SUBLAYER_ORDER)) for i, s in enumerate(SUBLAYER_ORDER)}
    colors = [stype_colors.get(r["stype"], "gray") for r in records]

    ax_scat.scatter(r2_q_arr, r2_c_arr, c=colors, s=18, alpha=0.7, zorder=3)
    lim = [min(r2_q_arr.min(), r2_c_arr.min()) - 0.05, 1.02]
    ax_scat.plot(lim, lim, "k--", linewidth=1, zorder=2)
    ax_scat.set_xlim(lim); ax_scat.set_ylim(lim)
    ax_scat.set_xlabel("Quadratic R²"); ax_scat.set_ylabel("Cubic R²")
    ax_scat.set_title("Cubic vs Quadratic R²\n(above diagonal = cubic helps)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=stype_colors[s],
                      markersize=7, label=s) for s in SUBLAYER_ORDER]
    ax_scat.legend(handles=handles, fontsize=7, loc="lower right")

    # ── Row 1 right: histogram of improvement ──
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_hist.hist(delta_arr, bins=40, color="mediumpurple", edgecolor="white", linewidth=0.5)
    ax_hist.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_hist.axvline(np.median(delta_arr), color="red", linewidth=1.5,
                    label=f"median Δ = {np.median(delta_arr):.3f}")
    # Mark the threshold we'll use for "meaningful"
    threshold = 0.15
    ax_hist.axvline(threshold, color="orange", linestyle=":", linewidth=1.5,
                    label=f"threshold Δ = {threshold}")
    n_above = (delta_arr > threshold).sum()
    ax_hist.set_xlabel("Cubic R² − Quadratic R²")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Cubic improvement distribution\n{n_above} / {len(delta_arr)} layers gain >{threshold}")
    ax_hist.legend(fontsize=9)

    # ── Row 2: top-12 layers where cubic gains most ──
    sorted_recs = sorted(records, key=lambda r: r["delta"], reverse=True)
    top_n = 12
    n_cols = 4
    n_rows_detail = 3
    for idx in range(top_n):
        rec = sorted_recs[idx]
        ax = fig.add_subplot(gs[1, :], label=f"detail_{idx}")
        # Actually need a sub-gridspec for the detail panels
        break  # We'll use a different approach

    # Clear the gs[1,:] and use a nested gridspec
    gs_detail = gs[1, :].subgridspec(n_rows_detail, n_cols, hspace=0.4, wspace=0.3)
    for idx in range(top_n):
        rec = sorted_recs[idx]
        ax = fig.add_subplot(gs_detail[idx // n_cols, idx % n_cols])

        sigmas, vals = rec["sigmas"], rec["vals"]
        xf = np.linspace(sigmas.min(), sigmas.max(), 200)

        ax.scatter(sigmas, vals, s=10, color="black", alpha=0.6, zorder=4)
        y_q = np.polyval(rec["coeff_q"], xf)
        y_c = np.polyval(rec["coeff_c"], xf)
        ax.plot(xf, y_q, "b--", linewidth=1.5, label=f"quad R²={rec['r2_q']:.2f}")
        ax.plot(xf, y_c, "r-", linewidth=1.5, label=f"cubic R²={rec['r2_c']:.2f}")
        ax.invert_xaxis()
        ax.set_title(f"{rec['layer']}\nΔ = +{rec['delta']:.3f}", fontsize=7.5)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    # ── Row 3: heatmap of cubic−quadratic gain ──
    n_blocks = 24
    gs_heat = gs[2, :].subgridspec(1, 2, wspace=0.3)
    for panel_idx, stream_name in enumerate(["img", "txt"]):
        ax = fig.add_subplot(gs_heat[0, panel_idx])
        matrix = np.full((len(SUBLAYER_ORDER), n_blocks), np.nan)
        for rec in records:
            if rec["stream"] != stream_name:
                continue
            stype = rec["stype"]
            if stype not in SUBLAYER_ORDER:
                continue
            blk = rec["block"]
            if blk < n_blocks:
                matrix[SUBLAYER_ORDER.index(stype), blk] = rec["delta"]

        im = ax.imshow(matrix, vmin=0, vmax=0.5, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n_blocks))
        ax.set_xticklabels([str(i) for i in range(n_blocks)], fontsize=6)
        ax.set_yticks(range(len(SUBLAYER_ORDER)))
        ax.set_yticklabels(SUBLAYER_ORDER, fontsize=8)
        ax.set_xlabel("Block index")
        ax.set_title(f"{stream_name} stream: Cubic R² gain over Quadratic")

        for r in range(len(SUBLAYER_ORDER)):
            for c in range(n_blocks):
                v = matrix[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=4.5, color="black" if v < 0.35 else "white")

        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Cubic − Quadratic R²")

    fig.suptitle("fig_cf5: Where does cubic (4 params) meaningfully beat quadratic (3 params)?",
                 fontsize=13, y=0.99)
    out = output_dir / "fig_cf5_cubic_vs_quadratic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print the layers that need cubic
    print(f"  Layers where cubic gains > {threshold} R² over quadratic:")
    for rec in sorted_recs:
        if rec["delta"] < threshold:
            break
        print(f"    {rec['layer']:<35} quad={rec['r2_q']:.3f}  cubic={rec['r2_c']:.3f}  Δ=+{rec['delta']:.3f}")
    print(f"  Saved {out.name}")


def fig_cf6_precision_loss(trajectories, output_dir: Path):
    """Quantify effective bit-precision lost when using fitted curves vs true per-σ absmax.

    If the true clipping range at timestep i is α_true[i] and the fitted range is α_fit[i]:
      - If α_fit > α_true: the quantisation grid is wider than needed → wasted precision.
        Effective bits lost ≈ log₂(α_fit / α_true).
      - If α_fit < α_true: real values get clipped → clipping error.
        We measure the overshoot ratio α_true / α_fit.

    We compute these for three strategies: static (single global α), quadratic fit, and
    the best-of quad/cubic (adaptive). Then compare.
    """
    print("[fig_cf6] Precision loss analysis: fitted curves vs true per-σ absmax")

    from matplotlib.lines import Line2D

    # ── Compute per-layer, per-timestep metrics ──
    all_records = []
    for layer, (sigmas, vals) in trajectories.items():
        # True absmax at each σ
        alpha_true = vals

        # Strategy 1: static (worst-case)
        alpha_static = np.full_like(vals, vals.max())

        # Strategy 2: quadratic fit
        _, coeff_q = poly_r2(sigmas, vals, 2)
        alpha_quad = np.polyval(coeff_q, sigmas)

        # Strategy 3: best-of quad/cubic (use cubic if gain > 0.15)
        r2_q, _ = poly_r2(sigmas, vals, 2)
        r2_c, coeff_c = poly_r2(sigmas, vals, 3)
        if (r2_c - r2_q) > 0.15:
            alpha_adaptive = np.polyval(coeff_c, sigmas)
            fit_label = "cubic"
        else:
            alpha_adaptive = alpha_quad.copy()
            fit_label = "quad"

        # Bits wasted = log2(α_fit / α_true) when α_fit > α_true
        # Clipping overshoot = α_true / α_fit when α_fit < α_true
        for name, alpha_fit in [("static", alpha_static), ("quadratic", alpha_quad),
                                 ("adaptive", alpha_adaptive)]:
            ratio = alpha_fit / alpha_true  # >1 means wider than needed, <1 means clipping
            bits_wasted = np.where(ratio > 1, np.log2(ratio), 0)
            clip_ratio = np.where(ratio < 1, alpha_true / alpha_fit, 1.0)

            all_records.append({
                "layer": layer, "stream": stream(layer), "stype": sublayer_type(layer),
                "block": block_num(layer), "strategy": name,
                "mean_bits_wasted": float(bits_wasted.mean()),
                "max_bits_wasted": float(bits_wasted.max()),
                "mean_clip_ratio": float(clip_ratio.mean()),
                "max_clip_ratio": float(clip_ratio.max()),
                "pct_clipping": float((ratio < 0.99).mean() * 100),
            })

    # ── Fig layout: 2×2 ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "fig_cf6: Effective precision lost — static vs quadratic-fit vs adaptive (quad/cubic)\n"
        "For W4 (4-bit), 0.5 bits wasted = losing 12.5% of your quantisation levels",
        fontsize=12,
    )

    strategies = ["static", "quadratic", "adaptive"]
    strat_colors = {"static": "firebrick", "quadratic": "steelblue", "adaptive": "seagreen"}

    # ── Panel A: histogram of mean bits wasted ──
    ax = axes[0][0]
    for strat in strategies:
        vals = [r["mean_bits_wasted"] for r in all_records if r["strategy"] == strat]
        ax.hist(vals, bins=40, alpha=0.6, color=strat_colors[strat], label=strat,
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean bits wasted per layer (across all σ steps)")
    ax.set_ylabel("Count (layers)")
    ax.set_title("Distribution of wasted precision")
    ax.legend(fontsize=9)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="0.5-bit threshold")

    # ── Panel B: max bits wasted per sublayer type ──
    ax = axes[0][1]
    x_pos = np.arange(len(SUBLAYER_ORDER))
    width = 0.25
    for i, strat in enumerate(strategies):
        means = []
        for stype in SUBLAYER_ORDER:
            stype_vals = [r["max_bits_wasted"] for r in all_records
                          if r["strategy"] == strat and r["stype"] == stype]
            means.append(np.mean(stype_vals) if stype_vals else 0)
        ax.bar(x_pos + i * width, means, width, color=strat_colors[strat],
               label=strat, alpha=0.8)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(SUBLAYER_ORDER, fontsize=8, rotation=20)
    ax.set_ylabel("Avg of max bits wasted per layer")
    ax.set_title("Worst-case wasted precision by sublayer type")
    ax.legend(fontsize=8)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

    # ── Panel C: clipping percentage (how often does the fit under-estimate?) ──
    ax = axes[1][0]
    for strat in ["quadratic", "adaptive"]:
        vals = [r["pct_clipping"] for r in all_records if r["strategy"] == strat]
        ax.hist(vals, bins=30, alpha=0.6, color=strat_colors[strat], label=strat,
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("% of σ steps where fit clips (α_fit < α_true)")
    ax.set_ylabel("Count (layers)")
    ax.set_title("Clipping frequency — how often does the fit underestimate?")
    ax.legend(fontsize=9)

    # ── Panel D: per-block mean bits wasted — static vs adaptive ──
    ax = axes[1][1]
    n_blocks = 24
    for strat in strategies:
        block_means = []
        for b in range(n_blocks):
            bvals = [r["mean_bits_wasted"] for r in all_records
                     if r["strategy"] == strat and r["block"] == b]
            block_means.append(np.mean(bvals) if bvals else 0)
        ax.plot(range(n_blocks), block_means, "o-", color=strat_colors[strat],
                markersize=4, label=strat, alpha=0.8)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Mean bits wasted (averaged over sublayers)")
    ax.set_title("Wasted precision by block depth")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(range(n_blocks))
    ax.set_xticklabels([str(i) for i in range(n_blocks)], fontsize=7)

    plt.tight_layout()
    out = output_dir / "fig_cf6_precision_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Summary stats ──
    for strat in strategies:
        recs = [r for r in all_records if r["strategy"] == strat]
        mean_bw = np.mean([r["mean_bits_wasted"] for r in recs])
        max_bw = np.max([r["max_bits_wasted"] for r in recs])
        mean_clip = np.mean([r["pct_clipping"] for r in recs])
        print(f"  {strat:12s}:  mean bits wasted = {mean_bw:.3f},  "
              f"worst-case = {max_bw:.3f},  mean clipping% = {mean_clip:.1f}%")

    print(f"  Saved {out.name}")


def fig_cf7_percentile_clipping(pct_trajectories, output_dir: Path):
    """Compare curve fitting at different percentile levels.

    The key question: if you fit to the 99.9th percentile trajectory instead of
    the mean absmax, do you get a tighter clipping range with controlled clipping
    rate?  And does the curve still fit well at higher percentiles?

    Plots:
      Row 1: R² distribution at each percentile level (quad fit quality stays good?)
      Row 2: Example layers showing trajectories at all 4 percentile levels + fits
      Row 3: Precision analysis — bits wasted + clipping rate at each percentile
    """
    print("[fig_cf7] Percentile-based clipping: which percentile to fit?")

    pct_names = ["p99", "p999", "mean_absmax", "p100_absmax"]
    pct_labels = ["99th %ile", "99.9th %ile", "Mean ch. absmax", "Tensor absmax (100%)"]
    pct_colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.3, 1], hspace=0.35)

    # ── Row 1: R² distributions at each percentile ──
    gs_r1 = gs[0].subgridspec(1, 4, wspace=0.3)
    for i, (pname, plabel, pcol) in enumerate(zip(pct_names, pct_labels, pct_colors)):
        ax = fig.add_subplot(gs_r1[0, i])
        trajs = pct_trajectories.get(pname, {})
        r2_vals = []
        for layer, (sigmas, vals) in trajs.items():
            r2, _ = poly_r2(sigmas, vals, 2)
            r2_vals.append(r2)
        r2_vals = np.array(r2_vals)
        ax.hist(r2_vals, bins=30, color=pcol, edgecolor="white", linewidth=0.5, alpha=0.8)
        ax.axvline(np.median(r2_vals), color="black", linewidth=1.5,
                   label=f"median = {np.median(r2_vals):.3f}")
        ax.set_title(f"{plabel}\n(n={len(r2_vals)})", fontsize=9)
        ax.set_xlabel("Quadratic R²")
        ax.set_ylabel("Count")
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=7)

    # ── Row 2: Example layers — 8 selected, showing all percentile trajectories ──
    example_layers = [
        "mm0_img_mlp_fc2", "mm9_img_mlp_fc2", "mm18_img_mlp_fc2",
        "mm22_txt_mlp_fc2", "mm14_txt_mlp_fc2", "mm18_img_attn_q_proj",
        "mm0_txt_attn_q_proj", "mm20_img_attn_o_proj",
    ]
    gs_r2 = gs[1].subgridspec(2, 4, hspace=0.4, wspace=0.3)
    for idx, layer in enumerate(example_layers):
        ax = fig.add_subplot(gs_r2[idx // 4, idx % 4])
        ax.set_title(layer.replace("_", " "), fontsize=7.5)
        ax.invert_xaxis()
        ax.set_xlabel("σ", fontsize=7)
        ax.set_ylabel("activation scale", fontsize=7)
        ax.tick_params(labelsize=6)

        for pname, plabel, pcol in zip(pct_names, pct_labels, pct_colors):
            trajs = pct_trajectories.get(pname, {})
            if layer not in trajs:
                continue
            sigmas, vals = trajs[layer]
            r2, coeffs = poly_r2(sigmas, vals, 2)
            xf = np.linspace(sigmas.min(), sigmas.max(), 200)
            yf = np.polyval(coeffs, xf)
            ax.scatter(sigmas, vals, s=6, color=pcol, alpha=0.4, zorder=3)
            ax.plot(xf, yf, color=pcol, linewidth=1.2, alpha=0.8,
                    label=f"{plabel} R²={r2:.2f}")

        ax.legend(fontsize=5, loc="best")

    # ── Row 3: Precision analysis at each percentile ──
    # For each percentile level, fit a quadratic and compute:
    #   - bits wasted (using p100 as "true" ceiling)
    #   - clipping rate (% of σ steps where fit < p100)
    gs_r3 = gs[2].subgridspec(1, 3, wspace=0.3)

    # Panel A: mean bits wasted vs percentile
    ax = fig.add_subplot(gs_r3[0, 0])
    for pname, plabel, pcol in zip(pct_names, pct_labels, pct_colors):
        trajs = pct_trajectories.get(pname, {})
        true_trajs = pct_trajectories.get("p100_absmax", {})
        bits_wasted_list = []
        for layer in trajs:
            if layer not in true_trajs:
                continue
            sigmas, vals = trajs[layer]
            true_sigmas, true_vals = true_trajs[layer]
            # Align (should be same sigmas)
            if len(vals) != len(true_vals):
                continue
            _, coeffs = poly_r2(sigmas, vals, 2)
            alpha_fit = np.polyval(coeffs, sigmas)
            # Compare fitted percentile curve to true absmax
            ratio = alpha_fit / true_vals
            bw = np.where(ratio > 1, np.log2(ratio), 0)
            bits_wasted_list.append(float(bw.mean()))
        ax.hist(bits_wasted_list, bins=30, alpha=0.5, color=pcol, label=plabel,
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean bits wasted vs tensor absmax")
    ax.set_ylabel("Count")
    ax.set_title("Wasted precision by percentile level")
    ax.legend(fontsize=7)

    # Panel B: clipping rate vs percentile
    ax = fig.add_subplot(gs_r3[0, 1])
    for pname, plabel, pcol in zip(pct_names, pct_labels, pct_colors):
        trajs = pct_trajectories.get(pname, {})
        true_trajs = pct_trajectories.get("p100_absmax", {})
        clip_rates = []
        for layer in trajs:
            if layer not in true_trajs:
                continue
            sigmas, vals = trajs[layer]
            true_sigmas, true_vals = true_trajs[layer]
            if len(vals) != len(true_vals):
                continue
            _, coeffs = poly_r2(sigmas, vals, 2)
            alpha_fit = np.polyval(coeffs, sigmas)
            clip_rate = float((alpha_fit < true_vals * 0.99).mean() * 100)
            clip_rates.append(clip_rate)
        ax.hist(clip_rates, bins=30, alpha=0.5, color=pcol, label=plabel,
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("% of σ steps where fit < tensor absmax")
    ax.set_ylabel("Count")
    ax.set_title("Clipping frequency by percentile level")
    ax.legend(fontsize=7)

    # Panel C: the trade-off scatter — mean bits wasted vs mean clip rate per layer
    ax = fig.add_subplot(gs_r3[0, 2])
    for pname, plabel, pcol in zip(pct_names, pct_labels, pct_colors):
        trajs = pct_trajectories.get(pname, {})
        true_trajs = pct_trajectories.get("p100_absmax", {})
        bw_list, cr_list = [], []
        for layer in trajs:
            if layer not in true_trajs:
                continue
            sigmas, vals = trajs[layer]
            true_sigmas, true_vals = true_trajs[layer]
            if len(vals) != len(true_vals):
                continue
            _, coeffs = poly_r2(sigmas, vals, 2)
            alpha_fit = np.polyval(coeffs, sigmas)
            ratio = alpha_fit / true_vals
            bw = np.where(ratio > 1, np.log2(ratio), 0).mean()
            cr = float((alpha_fit < true_vals * 0.99).mean() * 100)
            bw_list.append(bw)
            cr_list.append(cr)
        ax.scatter(cr_list, bw_list, s=8, alpha=0.4, color=pcol, label=plabel)
    ax.set_xlabel("Clipping % (fit < absmax)")
    ax.set_ylabel("Bits wasted (fit > absmax)")
    ax.set_title("Trade-off: precision waste vs clipping risk")
    ax.legend(fontsize=7)

    fig.suptitle(
        "fig_cf7: Which percentile should we fit the curve to?\n"
        "Lower percentile = tighter clip (more precision) but more clipping risk",
        fontsize=12, y=0.99,
    )
    out = output_dir / "fig_cf7_percentile_clipping.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    true_trajs = pct_trajectories.get("p100_absmax", {})
    print(f"  {'Percentile':<20} {'Med R²':>8} {'Mean BW':>10} {'Mean Clip%':>12}")
    print(f"  {'-'*52}")
    for pname, plabel in zip(pct_names, pct_labels):
        trajs = pct_trajectories.get(pname, {})
        r2s, bws, clips = [], [], []
        for layer in trajs:
            sigmas, vals = trajs[layer]
            r2, coeffs = poly_r2(sigmas, vals, 2)
            r2s.append(r2)
            if layer in true_trajs:
                ts, tv = true_trajs[layer]
                if len(vals) == len(tv):
                    af = np.polyval(coeffs, sigmas)
                    ratio = af / tv
                    bws.append(np.where(ratio > 1, np.log2(ratio), 0).mean())
                    clips.append(float((af < tv * 0.99).mean() * 100))
        print(f"  {plabel:<20} {np.median(r2s):>8.3f} {np.mean(bws):>10.4f} {np.mean(clips):>11.1f}%")

    print(f"  Saved {out.name}")


def print_fit_summary(trajectories):
    """Print a table of fit quality for every layer, sorted by quadratic R²."""
    print("\n── Fit quality summary (sorted by quadratic R², ascending) ──")
    print(f"{'Layer':<35} {'Lin R²':>8} {'Quad R²':>9} {'Cub R²':>8} {'Δ(Q-L)':>8}")
    print("-" * 72)

    rows = []
    for layer, (sigmas, vals) in trajectories.items():
        r2_l, _ = poly_r2(sigmas, vals, 1)
        r2_q, _ = poly_r2(sigmas, vals, 2)
        r2_c, _ = poly_r2(sigmas, vals, 3)
        rows.append((layer, r2_l, r2_q, r2_c, r2_q - r2_l))

    for layer, r2_l, r2_q, r2_c, delta in sorted(rows, key=lambda x: x[2]):
        print(f"{layer:<35} {r2_l:>8.3f} {r2_q:>9.3f} {r2_c:>8.3f} {delta:>+8.3f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-dir", default="calibration_data_100/activations")
    parser.add_argument("--output-dir", default="analysis_results")
    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trajectories...")
    trajectories, sigmas = load_trajectories(activations_dir)
    print(f"  Loaded {len(trajectories)} layer trajectories, {len(sigmas)} σ steps")

    fig_cf1_per_sublayer_fits(trajectories, output_dir)
    fig_cf2_r2_improvement(trajectories, output_dir)
    fig_cf3_opposite_directions(trajectories, output_dir)
    fig_cf4_r2_heatmap(trajectories, output_dir)
    fig_cf5_cubic_vs_quadratic(trajectories, output_dir)
    fig_cf6_precision_loss(trajectories, output_dir)

    print("\nLoading percentile trajectories...")
    pct_trajectories, _ = load_percentile_trajectories(activations_dir)
    for pname, trajs in pct_trajectories.items():
        print(f"  {pname}: {len(trajs)} layers")
    fig_cf7_percentile_clipping(pct_trajectories, output_dir)

    print_fit_summary(trajectories)

    print("\nDone. Outputs in", output_dir)


if __name__ == "__main__":
    main()
