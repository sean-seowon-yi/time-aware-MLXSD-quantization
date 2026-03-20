"""
Validate polynomial clipping schedule generalizability across independent COCO image groups.

Runs full SD3 denoising on 3 independent groups of COCO prompts, builds polynomial
schedules for each group in-memory, and compares them against each other.

Generalisation claim: curves are a property of SD3 rectified-flow physics, not the
specific calibration images. Success criterion: median nRMSE < 5% for Group A vs B.

Usage:
    conda run -n diffusionkit python -m src.validate_poly_generalization \\
        --coco-prompts coco_prompts.csv \\
        --group-size 30 \\
        --num-groups 3 \\
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
    """Batch-eval pending arrays, compute p100 absmax per layer, clear hooks.

    Returns dict: layer_key -> float absmax value.
    """
    pending = [h._last_output for h, _, _ in hooks.values()
               if h._last_output is not None and isinstance(h._last_output, mx.array)]
    if pending:
        mx.eval(*pending)

    stats = {}
    for key, (hook, _, _) in hooks.items():
        if hook._last_output is not None:
            arr = np.abs(np.array(hook._last_output))
            stats[key] = float(arr.max())
        hook.clear()
    return stats


# ---------------------------------------------------------------------------
# Full denoising for a single prompt
# ---------------------------------------------------------------------------

def run_prompt(pipeline, denoiser, prompt, hooks, num_steps, cfg_weight=7.5, latent_size=64):
    """Run full SD3 denoising, collect per-step activation stats.

    Returns (step_stats list, sigmas np.ndarray).
    """
    conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
    mx.eval(conditioning, pooled)

    sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
    timesteps = pipeline.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
    denoiser.cache_modulation_params(pooled, timesteps)

    latent_shape = (1, latent_size, latent_size, 16)
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


def collect_group(pipeline, prompts, num_steps, latent_size=64):
    """Run all prompts in the group; aggregate per-step absmax stats.

    Returns (trajs dict: layer_key -> (sigmas_array, absmax_means_array), all_sigmas).
    """
    from diffusionkit.mlx import CFGDenoiser

    hooks = install_linear_hooks(pipeline)
    denoiser = CFGDenoiser(pipeline)

    # acc[step_idx][layer_key] = list of per-prompt absmax values
    acc = defaultdict(lambda: defaultdict(list))
    all_sigmas = None

    for prompt in tqdm(prompts, desc="Prompts"):
        try:
            step_stats, sigmas = run_prompt(pipeline, denoiser, prompt, hooks, num_steps, latent_size=latent_size)
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
        "percentile": "p100_absmax",
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


def plot_curve_overlays(named_schedules, sigmas, output_path):
    """6-panel plot: one representative layer per sublayer type, one curve per schedule.

    named_schedules: list of (label, schedule_dict) in display order.
    """
    sigma_grid = np.linspace(float(sigmas.min()), float(sigmas.max()), 200)
    palette = ["steelblue", "darkorange", "forestgreen", "crimson", "purple", "saddlebrown"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    # Intersect layers across all schedules
    all_layers = set(named_schedules[0][1]["layers"])
    for _, sched in named_schedules[1:]:
        all_layers &= set(sched["layers"])

    rep_layers = {}
    for st in _SUBLAYER_TYPES:
        candidates = sorted(l for l in all_layers if st in l)
        if candidates:
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
        for i, (label, sched) in enumerate(named_schedules):
            if layer not in sched["layers"]:
                continue
            y = np.polyval(sched["layers"][layer]["coeffs"], sigma_grid)
            ax.plot(sigma_grid, y, color=palette[i % len(palette)],
                    linestyle=linestyles[i % len(linestyles)],
                    label=label, linewidth=1.5)
        ax.set_title(f"{st}\n{layer}", fontsize=7)
        ax.set_xlabel("σ", fontsize=8)
        ax.set_ylabel("absmax activation", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax_idx in range(n, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    labels_str = " / ".join(label for label, _ in named_schedules)
    plt.suptitle(f"Polynomial curve overlays — {labels_str}", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# High-nRMSE layer visualization
# ---------------------------------------------------------------------------

def plot_high_nrmse_layers(summary, named_schedules, sigmas, output_path, top_n=20):
    """Plot polynomial curves for the top-N layers by worst-case nRMSE across all pairs.

    For each layer, all group curves are overlaid so divergence is immediately visible.
    """
    sigma_grid = np.linspace(float(sigmas.min()), float(sigmas.max()), 200)

    # Find worst-case nRMSE per layer across all pairs
    worst = {}
    for pair, data in summary.items():
        for layer, v in data["per_layer"].items():
            if layer not in worst or v["nrmse"] > worst[layer]:
                worst[layer] = v["nrmse"]

    top_layers = sorted(worst, key=lambda l: worst[l], reverse=True)[:top_n]

    ncols = 4
    nrows = (len(top_layers) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes_flat = np.array(axes).ravel()

    palette = ["steelblue", "darkorange", "forestgreen", "crimson", "purple"]
    linestyles = ["-", "--", "-."]

    for ax_idx, layer in enumerate(top_layers):
        ax = axes_flat[ax_idx]
        for i, (label, sched) in enumerate(named_schedules):
            if layer not in sched["layers"]:
                continue
            y = np.polyval(sched["layers"][layer]["coeffs"], sigma_grid)
            ax.plot(sigma_grid, y,
                    color=palette[i % len(palette)],
                    linestyle=linestyles[i % len(linestyles)],
                    label=label, linewidth=1.5)
        ax.set_title(f"{layer}\nmax nRMSE={worst[layer]*100:.1f}%", fontsize=7)
        ax.set_xlabel("σ", fontsize=8)
        ax.set_ylabel("absmax", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax_idx in range(len(top_layers), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    plt.suptitle(f"Top-{top_n} layers by worst-case nRMSE across all pairs", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Ranked MSE output
# ---------------------------------------------------------------------------

def print_ranked_mse(summary, top_n=None):
    """Print per-layer RMSE ranked highest to lowest for each pair."""
    for pair, data in summary.items():
        rows = sorted(data["per_layer"].items(),
                      key=lambda x: x[1]["rmse"], reverse=True)
        if top_n:
            rows = rows[:top_n]
        print(f"\n--- {pair} — layers ranked by MSE (high → low) ---")
        print(f"  {'Layer':<45} {'RMSE':>10}  {'nRMSE':>8}")
        print(f"  {'-'*45} {'-'*10}  {'-'*8}")
        for layer, v in rows:
            print(f"  {layer:<45} {v['rmse']:>10.4f}  {v['nrmse']*100:>7.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-prompts", type=Path, default=Path("coco_prompts.csv"))
    parser.add_argument("--group-size", type=int, default=30)
    parser.add_argument("--num-groups", type=int, default=2,
                        help="Number of NEW groups to run (ignored groups use --load-group-schedules)")
    parser.add_argument("--prompt-start-group", type=int, default=0,
                        help="Offset into the CSV: new prompts start at this group index "
                             "(set to number of already-run groups to avoid reusing prompts)")
    parser.add_argument("--load-group-schedules", type=Path, nargs="*", default=[],
                        metavar="JSON",
                        help="Paths to previously computed group schedule JSONs to compare "
                             "against new groups without re-running them")
    parser.add_argument("--output-dir", type=Path, default=Path("generalization_results"))
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--cfg-weight", type=float, default=7.5)
    parser.add_argument("--latent-size", type=int, default=64,
                        help="Spatial size of latent (e.g. 32 → 256px, 64 → 512px)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load any pre-existing group schedules ----
    # named_schedules: list of (label, schedule_dict) in order
    named_schedules = []
    for i, path in enumerate(args.load_group_schedules):
        with open(path) as f:
            sched = json.load(f)
        label = f"Group {chr(65 + i)}"  # A, B, C, ...
        named_schedules.append((label, sched))
        print(f"Loaded {label} schedule from {path} ({len(sched['layers'])} layers).")

    n_loaded = len(named_schedules)

    # ---- Prompt selection for new groups ----
    # New groups start at prompt-start-group (default = n_loaded to avoid overlap)
    effective_start = args.prompt_start_group if args.prompt_start_group else n_loaded

    df = pd.read_csv(args.coco_prompts)
    total = len(df)
    # Use a fixed stride based on total / group_size so the grid is stable
    stride = total // args.group_size
    new_groups = []
    for g in range(args.num_groups):
        group_idx = effective_start + g
        prompts = [
            df.iloc[(group_idx * args.group_size + j) % total]["prompt"]
            for j in range(args.group_size)
        ]
        new_groups.append(prompts)

    print(f"\nNew groups: {args.num_groups} × {args.group_size} prompts "
          f"(start_group={effective_start}, stride={stride}, total={total})")
    for g_idx, grp in enumerate(new_groups):
        label = chr(65 + n_loaded + g_idx)
        print(f"  Group {label}: first='{grp[0][:60]}...', last='{grp[-1][:60]}...'")

    # ---- Load pipeline (only if new groups need running) ----
    if new_groups:
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

    # ---- Collect new group trajectories ----
    all_sigmas = None
    for g_idx, prompts in enumerate(new_groups):
        label = chr(65 + n_loaded + g_idx)
        file_idx = n_loaded + g_idx
        print(f"\n{'='*60}")
        print(f"Collecting Group {label} ({len(prompts)} prompts, {args.num_steps} steps)...")
        trajs, group_sigmas = collect_group(pipeline, prompts, args.num_steps,
                                             latent_size=args.latent_size)
        schedule = build_schedule(trajs, group_sigmas)
        named_schedules.append((f"Group {label}", schedule))
        if all_sigmas is None:
            all_sigmas = group_sigmas

        out_path = args.output_dir / f"group_{file_idx}_schedule.json"
        with open(out_path, "w") as f:
            json.dump(schedule, f, indent=2)
        print(f"Saved {out_path} ({len(schedule['layers'])} layers).")

        npz_path = args.output_dir / f"group_{file_idx}_trajectories.npz"
        np.savez(npz_path, sigmas=group_sigmas,
                 **{k: v[1] for k, v in trajs.items()})
        print(f"Saved {npz_path} ({len(trajs)} layer trajectories).")

    # Fall back to sigma range from first loaded schedule if no new groups were run
    if all_sigmas is None:
        sigma_range = named_schedules[0][1]["sigma_range"]
        all_sigmas = np.linspace(sigma_range[0], sigma_range[1], args.num_steps)

    # ---- All pairwise comparisons ----
    sigma_grid = np.linspace(float(all_sigmas.min()), float(all_sigmas.max()), 200)
    summary = {}
    for i in range(len(named_schedules)):
        for j in range(i + 1, len(named_schedules)):
            label_i, sched_i = named_schedules[i]
            label_j, sched_j = named_schedules[j]
            key = f"{label_i.replace(' ', '')}_vs_{label_j.replace(' ', '')}"
            results = compare_schedules(sched_i, sched_j, sigma_grid)
            summary[key] = comparison_summary(results)

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}")

    ranked = {}
    for pair, data in summary.items():
        ranked[pair] = sorted(
            [{"layer": k, **v} for k, v in data["per_layer"].items()],
            key=lambda x: x["rmse"], reverse=True
        )
    with open(args.output_dir / "ranked_mse.json", "w") as f:
        json.dump(ranked, f, indent=2)
    print(f"Saved {args.output_dir / 'ranked_mse.json'}")

    print_ranked_mse(summary)

    # ---- Print results ----
    print("\n=== Generalisation Validation Results ===")
    for pair, data in summary.items():
        print(f"  {pair}:")
        print(f"    median nRMSE = {data['median_nrmse']*100:.2f}%")
        print(f"    p95    nRMSE = {data['p95_nrmse']*100:.2f}%")
        print(f"    n_layers     = {data['n_layers']}")

    # ---- Plots ----
    bar_path = args.output_dir / "comparison_bar.png"
    plot_comparison_bar(summary, bar_path)

    overlay_path = args.output_dir / "curve_overlays.png"
    plot_curve_overlays(named_schedules, all_sigmas, overlay_path)

    high_nrmse_path = args.output_dir / "high_nrmse_layers.png"
    plot_high_nrmse_layers(summary, named_schedules, all_sigmas, high_nrmse_path)


if __name__ == "__main__":
    main()
