"""
Validate polynomial clipping schedule generalizability across independent COCO image groups.

Runs full SD3 denoising on 2 independent groups of COCO prompts, builds polynomial
schedules for each group in-memory, and compares them against each other and against
the reference schedule fit on calibration data.

Generalisation claim: curves are a property of SD3 rectified-flow physics, not the
specific calibration images. Success criterion: median nRMSE < 5% for Group A vs B;
< 8% for either group vs reference.

Usage:
    conda run -n diffusionkit python -m src.validate_poly_generalization \\
        --coco-prompts coco_prompts.csv \\
        --reference-schedule polynomial_clipping_schedule.json \\
        --group-size 30 \\
        --num-groups 2 \\
        --output-dir generalization_results \\
        --num-steps 25
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import mlx.core as mx

from src.adaround_optimize import _get_nested, _set_nested, get_block_linears
from src.generate_poly_schedule import select_degree


# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

class LinearHook:
    """Wraps a linear layer to capture its last output for stat collection."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._last_output = None

    def __call__(self, *args, **kwargs):
        result = self._wrapped(*args, **kwargs)
        self._last_output = result
        return result

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def clear(self):
        self._last_output = None


def install_linear_hooks(pipeline):
    """Install LinearHooks on all MM transformer block linears.

    Returns dict: layer_key -> (LinearHook, block_idx, full_path).
    Layer keys use underscores, e.g. 'mm0_img_attn_q_proj'.
    """
    hooks = {}
    for i, block in enumerate(pipeline.mmdit.multimodal_transformer_blocks):
        for full_path, layer, _ in get_block_linears(block, is_mm=True):
            short = full_path.replace("image_transformer_block", "img") \
                             .replace("text_transformer_block", "txt")
            key = f"mm{i}_{short.replace('.', '_')}"
            hook = LinearHook(layer)
            _set_nested(block, full_path, hook)
            hooks[key] = (hook, i, full_path)
    return hooks


def remove_linear_hooks(pipeline, hooks):
    """Restore original layers by removing hooks."""
    for key, (hook, block_idx, full_path) in hooks.items():
        block = pipeline.mmdit.multimodal_transformer_blocks[block_idx]
        _set_nested(block, full_path, hook._wrapped)


# ---------------------------------------------------------------------------
# Per-step stat collection
# ---------------------------------------------------------------------------

def collect_step_stats(hooks):
    """Batch-eval pending arrays, compute p99.9 per layer, clear hooks.

    Returns dict: layer_key -> float p999 value.
    """
    pending = [h._last_output for h, _, _ in hooks.values()
               if h._last_output is not None and isinstance(h._last_output, mx.array)]
    if pending:
        mx.eval(*pending)

    stats = {}
    for key, (hook, _, _) in hooks.items():
        if hook._last_output is not None:
            arr = np.abs(np.array(hook._last_output))
            stats[key] = float(np.percentile(arr.ravel(), 99.9))
        hook.clear()
    return stats


# ---------------------------------------------------------------------------
# Full denoising for a single prompt
# ---------------------------------------------------------------------------

def run_prompt(pipeline, denoiser, prompt, hooks, num_steps, cfg_weight=7.5):
    """Run full SD3 denoising, collect per-step activation stats.

    Returns (step_stats list, sigmas np.ndarray).
    """
    conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
    mx.eval(conditioning, pooled)

    sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
    timesteps = pipeline.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
    denoiser.cache_modulation_params(pooled, timesteps)

    latent_shape = (1, 64, 64, 16)
    noise = mx.random.normal(latent_shape).astype(pipeline.activation_dtype)
    x = pipeline.sampler.noise_scaling(
        sigmas[0], noise, mx.zeros(latent_shape), pipeline.max_denoise(sigmas)
    )
    mx.eval(x)

    step_stats, recorded_sigmas = [], []
    for i in range(len(sigmas) - 1):
        denoised = denoiser(
            x, timesteps[i], sigmas[i],
            conditioning=conditioning, cfg_weight=cfg_weight,
        )
        step_stats.append(collect_step_stats(hooks))
        recorded_sigmas.append(float(sigmas[i].item()))
        d = (x - denoised) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
        mx.eval(x)

    return step_stats, np.array(recorded_sigmas)


# ---------------------------------------------------------------------------
# Group trajectory aggregation
# ---------------------------------------------------------------------------

def _reset_modulation_cache(pipeline):
    """Reset per-prompt adaLN modulation cache (mirrors cache_adaround_data.py)."""
    try:
        pipeline.mmdit.clear_modulation_params_cache()
    except Exception:
        pass
    try:
        pipeline.mmdit.load_weights(
            pipeline.load_mmdit(only_modulation_dict=True), strict=False
        )
    except Exception:
        pass


def collect_group(pipeline, prompts, num_steps):
    """Run all prompts in the group; aggregate per-step p999 stats.

    Returns (trajs dict: layer_key -> (sigmas_array, p999_means_array), all_sigmas).
    """
    from diffusionkit.mlx import CFGDenoiser

    hooks = install_linear_hooks(pipeline)
    denoiser = CFGDenoiser(pipeline)

    # acc[step_idx][layer_key] = list of per-prompt p999 values
    acc = defaultdict(lambda: defaultdict(list))
    all_sigmas = None

    for prompt in tqdm(prompts, desc="Prompts"):
        try:
            step_stats, sigmas = run_prompt(pipeline, denoiser, prompt, hooks, num_steps)
            if all_sigmas is None:
                all_sigmas = sigmas
            for i, stats in enumerate(step_stats):
                for key, val in stats.items():
                    acc[i][key].append(val)
        except Exception as e:
            print(f"WARNING: skipping prompt: {e}")
        _reset_modulation_cache(pipeline)

    remove_linear_hooks(pipeline, hooks)

    if all_sigmas is None:
        raise RuntimeError("All prompts failed — no data collected.")

    # Build per-layer trajectories
    trajs = {}
    all_layers = set(k for d in acc.values() for k in d.keys())
    for layer_key in all_layers:
        vals = np.array([
            np.mean(acc[i][layer_key])
            for i in range(num_steps)
            if layer_key in acc[i]
        ])
        sigs = np.array([
            all_sigmas[i]
            for i in range(num_steps)
            if layer_key in acc[i]
        ])
        if len(vals) >= 3:
            trajs[layer_key] = (sigs, vals)

    return trajs, all_sigmas


# ---------------------------------------------------------------------------
# In-memory schedule building
# ---------------------------------------------------------------------------

def build_schedule(trajs, sigmas):
    """Fit polynomial to each layer trajectory. Returns schedule dict."""
    layers = {}
    for layer_raw, (layer_sigmas, vals) in sorted(trajs.items()):
        degree, coeffs, r2, cv = select_degree(layer_sigmas, vals)
        layers[layer_raw] = {
            "degree": degree,
            "coeffs": coeffs,
            "r2": round(r2, 4),
            "cv": round(cv, 4),
        }
    return {
        "version": "poly_v1",
        "percentile": "p999",
        "sigma_range": [float(sigmas.min()), float(sigmas.max())],
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# Schedule comparison
# ---------------------------------------------------------------------------

def compare_schedules(sched_a, sched_b, sigma_grid):
    """Return per-layer {rmse, nrmse} using sched_a as reference for normalisation."""
    results = {}
    for layer in sched_a["layers"]:
        if layer not in sched_b["layers"]:
            continue
        ya = np.polyval(sched_a["layers"][layer]["coeffs"], sigma_grid)
        yb = np.polyval(sched_b["layers"][layer]["coeffs"], sigma_grid)
        rmse = float(np.sqrt(np.mean((ya - yb) ** 2)))
        mean_ref = float(np.mean(np.abs(ya)))
        results[layer] = {
            "rmse": rmse,
            "nrmse": rmse / mean_ref if mean_ref > 1e-8 else 0.0,
        }
    return results


def comparison_summary(per_layer):
    """Aggregate per-layer comparison results."""
    nrmses = [v["nrmse"] for v in per_layer.values()]
    return {
        "median_nrmse": float(np.median(nrmses)),
        "p95_nrmse": float(np.percentile(nrmses, 95)),
        "n_layers": len(nrmses),
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_SUBLAYER_TYPES = ["attn_q_proj", "attn_k_proj", "attn_v_proj", "attn_out_proj",
                   "mlp_fc1", "mlp_fc2"]


def _sublayer_type(key):
    for t in _SUBLAYER_TYPES:
        if t in key:
            return t
    return "other"


def plot_comparison_bar(summary, output_path):
    """Grouped bar chart: median and p95 nRMSE per comparison pair, by sublayer type."""
    pairs = list(summary.keys())

    # Collect per-sublayer nRMSEs for each pair
    sublayer_data = {}  # pair -> sublayer_type -> [nrmse, ...]
    for pair, pair_data in summary.items():
        by_sub = defaultdict(list)
        for layer, v in pair_data["per_layer"].items():
            by_sub[_sublayer_type(layer)].append(v["nrmse"])
        sublayer_data[pair] = by_sub

    subtypes = sorted(set(st for sd in sublayer_data.values() for st in sd.keys()))
    n_sub = len(subtypes)
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "darkorange", "forestgreen"]

    for ax_idx, metric in enumerate(["median", "p95"]):
        ax = axes[ax_idx]
        x = np.arange(n_sub)
        width = 0.8 / n_pairs

        for pi, pair in enumerate(pairs):
            vals = []
            for st in subtypes:
                arr = sublayer_data[pair].get(st, [0.0])
                if metric == "median":
                    vals.append(float(np.median(arr)) * 100)
                else:
                    vals.append(float(np.percentile(arr, 95)) * 100)
            offset = (pi - n_pairs / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, label=pair, color=colors[pi % len(colors)])

        ax.set_xticks(x)
        ax.set_xticklabels(subtypes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("nRMSE (%)")
        ax.set_title(f"{metric.capitalize()} nRMSE by sublayer type")
        ax.legend(fontsize=7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved {output_path}")


def plot_curve_overlays(sched_a, sched_b, sched_ref, sigmas, output_path):
    """6-panel plot: one representative layer per sublayer type, 3 curves each."""
    sigma_grid = np.linspace(float(sigmas.min()), float(sigmas.max()), 200)

    # Pick one representative layer per sublayer type (prefer middle MM block)
    rep_layers = {}
    all_layers = set(sched_a["layers"]) & set(sched_b["layers"])
    if sched_ref:
        all_layers &= set(sched_ref["layers"])

    for st in _SUBLAYER_TYPES:
        candidates = sorted(l for l in all_layers if st in l)
        if candidates:
            # Prefer mm12 or mm13 (middle of 24 blocks), fallback to any
            mid = [c for c in candidates if "mm12_" in c or "mm13_" in c]
            rep_layers[st] = (mid or candidates)[0]

    n = len(rep_layers)
    if n == 0:
        print("WARNING: no common layers found for curve overlays — skipping plot.")
        return

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes_flat = np.array(axes).ravel()

    for ax_idx, (st, layer) in enumerate(sorted(rep_layers.items())):
        ax = axes_flat[ax_idx]
        ya = np.polyval(sched_a["layers"][layer]["coeffs"], sigma_grid)
        yb = np.polyval(sched_b["layers"][layer]["coeffs"], sigma_grid)
        ax.plot(sigma_grid, ya, color="steelblue", label="Group A", linewidth=1.5)
        ax.plot(sigma_grid, yb, color="darkorange", label="Group B",
                linewidth=1.5, linestyle="--")
        if sched_ref and layer in sched_ref["layers"]:
            yr = np.polyval(sched_ref["layers"][layer]["coeffs"], sigma_grid)
            ax.plot(sigma_grid, yr, color="crimson", label="Reference",
                    linewidth=1.5, linestyle=":")
        ax.set_title(f"{st}\n{layer}", fontsize=7)
        ax.set_xlabel("σ", fontsize=8)
        ax.set_ylabel("p99.9 activation", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide unused axes
    for ax_idx in range(n, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    plt.suptitle("Polynomial curve overlays — Group A / B / Reference", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-prompts", type=Path, default=Path("coco_prompts.csv"))
    parser.add_argument("--reference-schedule", type=Path,
                        default=Path("polynomial_clipping_schedule.json"))
    parser.add_argument("--group-size", type=int, default=30)
    parser.add_argument("--num-groups", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("generalization_results"))
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--cfg-weight", type=float, default=7.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Prompt selection ----
    df = pd.read_csv(args.coco_prompts)
    total = len(df)
    n_total = args.num_groups * args.group_size
    stride = total // n_total
    selected = [df.iloc[i * stride]["prompt"] for i in range(n_total)]
    groups = [
        selected[g * args.group_size:(g + 1) * args.group_size]
        for g in range(args.num_groups)
    ]
    print(f"Selected {n_total} prompts (stride={stride}) from {total} total.")
    for g_idx, grp in enumerate(groups):
        print(f"  Group {g_idx}: {len(grp)} prompts, "
              f"first='{grp[0][:60]}...', last='{grp[-1][:60]}...'")

    # ---- Load pipeline ----
    from diffusionkit.mlx import DiffusionPipeline
    print("\nLoading SD3 pipeline...")
    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    print("Pipeline loaded.")

    # ---- Load reference schedule ----
    ref_schedule = None
    if args.reference_schedule.exists():
        with open(args.reference_schedule) as f:
            ref_schedule = json.load(f)
        print(f"Loaded reference schedule: {len(ref_schedule['layers'])} layers.")
    else:
        print(f"WARNING: reference schedule not found at {args.reference_schedule}")

    # ---- Collect per-group trajectories ----
    group_schedules = []
    group_sigmas_list = []

    for g_idx, prompts in enumerate(groups):
        print(f"\n{'='*60}")
        print(f"Collecting Group {g_idx} ({len(prompts)} prompts, {args.num_steps} steps)...")
        trajs, all_sigmas = collect_group(pipeline, prompts, args.num_steps)
        schedule = build_schedule(trajs, all_sigmas)
        group_schedules.append(schedule)
        group_sigmas_list.append(all_sigmas)

        out_path = args.output_dir / f"group_{g_idx}_schedule.json"
        with open(out_path, "w") as f:
            json.dump(schedule, f, indent=2)
        print(f"Saved {out_path} ({len(schedule['layers'])} layers).")

    sched_a, sched_b = group_schedules[0], group_schedules[1]
    all_sigmas = group_sigmas_list[0]

    # ---- Comparisons ----
    sigma_grid = np.linspace(float(all_sigmas.min()), float(all_sigmas.max()), 200)

    results_ab = compare_schedules(sched_a, sched_b, sigma_grid)
    summary_ab = comparison_summary(results_ab)

    summary = {"groupA_vs_groupB": summary_ab}

    if ref_schedule is not None:
        results_ar = compare_schedules(sched_a, ref_schedule, sigma_grid)
        results_br = compare_schedules(sched_b, ref_schedule, sigma_grid)
        summary["groupA_vs_reference"] = comparison_summary(results_ar)
        summary["groupB_vs_reference"] = comparison_summary(results_br)

    # Strip per_layer from top-level summary output (keep in file only)
    summary_out = {}
    for pair, data in summary.items():
        summary_out[pair] = {
            k: v for k, v in data.items() if k != "per_layer"
        }
        summary_out[pair]["per_layer"] = data["per_layer"]

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved {summary_path}")

    # ---- Print results ----
    print("\n=== Generalisation Validation Results ===")
    for pair, data in summary_out.items():
        print(f"  {pair}:")
        print(f"    median nRMSE = {data['median_nrmse']*100:.2f}%")
        print(f"    p95    nRMSE = {data['p95_nrmse']*100:.2f}%")
        print(f"    n_layers     = {data['n_layers']}")

    print("\nSuccess thresholds:")
    ab_ok = summary_out["groupA_vs_groupB"]["median_nrmse"] < 0.05
    print(f"  Group A vs B  median nRMSE < 5%: {'PASS' if ab_ok else 'FAIL'}")
    if "groupA_vs_reference" in summary_out:
        ar_ok = summary_out["groupA_vs_reference"]["median_nrmse"] < 0.08
        br_ok = summary_out["groupB_vs_reference"]["median_nrmse"] < 0.08
        print(f"  Group A vs Ref median nRMSE < 8%: {'PASS' if ar_ok else 'FAIL'}")
        print(f"  Group B vs Ref median nRMSE < 8%: {'PASS' if br_ok else 'FAIL'}")

    # ---- Plots ----
    bar_path = args.output_dir / "comparison_bar.png"
    plot_comparison_bar(summary_out, bar_path)

    overlay_path = args.output_dir / "curve_overlays.png"
    plot_curve_overlays(sched_a, sched_b, ref_schedule, all_sigmas, overlay_path)


if __name__ == "__main__":
    main()
