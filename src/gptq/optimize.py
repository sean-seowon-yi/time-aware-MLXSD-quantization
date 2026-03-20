"""GPTQ pipeline orchestrator for SD3 MMDiT.

Two-phase design for ~5× speedup over naive per-block collection:

  Phase A — Global Hessian pass: install collectors on ALL 285 linears
            across all 24 blocks. Run all prompts once (900 fwd passes).
            No I/O caching — only Hessian accumulation.

  Phase B — Per-block alpha search: for each block, install cache-only
            collectors, run a small subset of prompts (e.g. 5), cache I/O,
            GPTQ-quantize using Phase A Hessians, then alpha search.

Usage:
    python -m src.gptq.optimize \
        --prompts src/calibration_sample_generation/sample_prompts.txt \
        --poly-schedule polynomial_clipping_schedule.json \
        --bits-w 4 \
        --output-dir gptq_output \
        --num-steps 30 --cfg-weight 4.0 --seed 42 \
        --damp-percent 0.01 --block-size 128 \
        --max-prompts 30 --alpha-prompts 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .utils import _get_block_linears, full_path_to_poly_key
from .gptq_quantize import gptq_quantize
from .hessian_collector import (
    collect_hessians_global,
    collect_io_for_block,
)
from .alpha_search import search_alpha_scale, search_alpha_scale_static
from .inference import compute_static_alphas


def save_block(output_dir: Path, block_idx: int, block_weight_results: dict):
    """Save quantized weights for one block to an .npz file."""
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for poly_key, (W_q_int, scales, _mse) in block_weight_results.items():
        arrays[f"{poly_key}__weight_int"] = W_q_int
        arrays[f"{poly_key}__scale"] = scales

    np.savez(weights_dir / f"mm{block_idx}.npz", **arrays)


def save_config(output_dir: Path, args, all_metrics: dict):
    """Save config.json with all hyperparameters and per-layer metrics."""
    alpha_scales = {}
    weight_mses = {}
    activation_mses = {}

    for poly_key, metrics in all_metrics.items():
        alpha_scales[poly_key] = metrics["alpha_scale"]
        weight_mses[poly_key] = metrics["weight_mse"]
        activation_mses[poly_key] = metrics["activation_mse"]

    config = {
        "method": "gptq",
        "bits_w": args.bits_w,
        "act_quant_mode": "static" if args.static_act_quant else "poly",
        "damp_percent": args.damp_percent,
        "block_size": args.block_size,
        "poly_schedule": str(args.poly_schedule),
        "num_prompts": args.max_prompts or "all",
        "alpha_prompts": args.alpha_prompts,
        "num_steps": args.num_steps,
        "cfg_weight": args.cfg_weight,
        "seed": args.seed,
        "alpha_scales": alpha_scales,
        "weight_mse": weight_mses,
        "activation_mse": activation_mses,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ quantization for SD3 MMDiT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompts", type=Path, required=True,
                        help="Text file with one prompt per line")
    parser.add_argument("--poly-schedule", type=Path, required=True,
                        help="Path to polynomial_clipping_schedule.json")
    parser.add_argument("--bits-w", type=int, default=4, choices=[4, 8],
                        help="Weight quantization bits (default: 4)")
    parser.add_argument("--output-dir", type=Path, default=Path("gptq_output"))
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts for Hessian collection")
    parser.add_argument("--alpha-prompts", type=int, default=5,
                        help="Number of prompts for Phase B alpha search I/O cache")
    parser.add_argument("--alpha-max-cache", type=int, default=50,
                        help="Max cached I/O samples per layer for alpha search")
    parser.add_argument("--hessian-cache", type=Path, default=None,
                        help="Path to save/load Hessian .npz checkpoint. "
                        "If the file exists, Phase A is skipped and Hessians "
                        "are loaded from it. Otherwise Phase A runs and saves here.")
    parser.add_argument("--static-act-quant", action="store_true",
                        help="Use static (timestep-agnostic) activation clipping "
                        "instead of polynomial clipping for alpha search")
    args = parser.parse_args()

    # Load prompts
    all_prompts = [
        line.strip() for line in args.prompts.read_text().splitlines()
        if line.strip()
    ]
    if args.max_prompts is not None:
        all_prompts = all_prompts[:args.max_prompts]
    print(f"Loaded {len(all_prompts)} prompts from {args.prompts}")

    # Split prompts: all for Phase A, first alpha_prompts for Phase B
    alpha_prompts = all_prompts[:args.alpha_prompts]
    print(f"Phase A: {len(all_prompts)} prompts for Hessian collection")
    print(f"Phase B: {len(alpha_prompts)} prompts for alpha search I/O cache")

    # Load poly schedule
    with open(args.poly_schedule) as f:
        poly_schedule = json.load(f)
    print(f"Loaded poly schedule: {len(poly_schedule.get('layers', {}))} layers")

    # Compute static alphas if needed
    static_alphas = None
    if args.static_act_quant:
        static_alphas = compute_static_alphas(poly_schedule)
        print(f"Static mode: computed {len(static_alphas)} fixed alphas from poly schedule")

    # Load pipeline
    print("Loading pipeline...")
    from diffusionkit.mlx import DiffusionPipeline, CFGDenoiser

    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    denoiser = CFGDenoiser(pipeline)
    print("Pipeline loaded")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Phase A: Global Hessian collection (all blocks, one pass)
    # -----------------------------------------------------------------------
    hessian_cache_path = args.hessian_cache
    if hessian_cache_path is None:
        hessian_cache_path = args.output_dir / "hessians.npz"

    if hessian_cache_path.exists():
        print(f"Loading cached Hessians from {hessian_cache_path}")
        phase_a_t0 = time.time()
        loaded = np.load(hessian_cache_path)
        hessians = {key: loaded[key] for key in loaded.files}
        phase_a_elapsed = time.time() - phase_a_t0
        print(f"Loaded {len(hessians)} Hessians in {phase_a_elapsed:.1f}s")
    else:
        phase_a_t0 = time.time()
        all_collectors = collect_hessians_global(
            pipeline, denoiser, all_prompts, poly_schedule,
            num_steps=args.num_steps,
            cfg_weight=args.cfg_weight,
            seed=args.seed,
            latent_size=args.latent_size,
            static_alphas=static_alphas,
        )
        phase_a_elapsed = time.time() - phase_a_t0
        print(f"Phase A done: {phase_a_elapsed:.0f}s")

        # Extract Hessians into NumPy (free MLX memory)
        hessians = {}  # poly_key -> np.ndarray
        for block_idx, collectors in all_collectors.items():
            for poly_key, collector in collectors.items():
                hessians[poly_key] = collector.get_hessian()
        del all_collectors

        # Save checkpoint
        hessian_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(hessian_cache_path, **hessians)
        print(f"Saved Hessian checkpoint to {hessian_cache_path}")

    # -----------------------------------------------------------------------
    # Phase B: Per-block GPTQ + alpha search
    # -----------------------------------------------------------------------
    all_metrics = {}
    n_blocks = len(pipeline.mmdit.multimodal_transformer_blocks)
    phase_b_t0 = time.time()

    for block_idx in tqdm(range(n_blocks), desc="Blocks (GPTQ + alpha)"):
        block_t0 = time.time()
        block = pipeline.mmdit.multimodal_transformer_blocks[block_idx]

        # 1. GPTQ quantize using Phase A Hessians
        block_weight_results = {}
        for full_path, layer in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            W = np.array(layer.weight, dtype=np.float32)
            H = hessians[poly_key]
            W_q_int, scales, weight_mse = gptq_quantize(
                W, H, args.bits_w, args.damp_percent, args.block_size
            )
            block_weight_results[poly_key] = (W_q_int, scales, weight_mse)

        # 2. Collect I/O cache for alpha search (short pass)
        io_collectors = collect_io_for_block(
            pipeline, denoiser, block_idx, alpha_prompts, poly_schedule,
            num_steps=args.num_steps,
            max_cache=args.alpha_max_cache,
            cfg_weight=args.cfg_weight,
            seed=args.seed,
            latent_size=args.latent_size,
            static_alphas=static_alphas,
        )

        # 3. Alpha search for each linear
        block_alpha_results = {}
        layers_dict = poly_schedule.get("layers", {})
        static_alphas = compute_static_alphas(poly_schedule) if args.static_act_quant else {}
        for poly_key, (W_q_int, scales, weight_mse) in block_weight_results.items():
            io_collector = io_collectors[poly_key]
            W_q_dequant = W_q_int.astype(np.float32) * scales[:, None]

            wrapped = io_collector._wrapped
            bias = None
            if hasattr(wrapped, "bias") and wrapped.bias is not None:
                bias = np.array(wrapped.bias, dtype=np.float32)

            cached_inputs, cached_outputs = io_collector.get_cached_io()

            if args.static_act_quant:
                static_alpha = static_alphas.get(poly_key, 1.0)
                best_alpha, best_act_mse = search_alpha_scale_static(
                    W_q_dequant, bias, cached_inputs, cached_outputs, static_alpha
                )
            else:
                poly_entry = layers_dict.get(poly_key)
                best_alpha, best_act_mse = search_alpha_scale(
                    W_q_dequant, bias, cached_inputs, cached_outputs, poly_entry
                )
            block_alpha_results[poly_key] = (best_alpha, best_act_mse)

        # 4. Save block weights
        save_block(args.output_dir, block_idx, block_weight_results)

        # 5. Accumulate metrics
        for poly_key in block_weight_results:
            _, _, weight_mse = block_weight_results[poly_key]
            alpha_scale, act_mse = block_alpha_results[poly_key]
            all_metrics[poly_key] = {
                "weight_mse": float(weight_mse),
                "activation_mse": float(act_mse),
                "alpha_scale": float(alpha_scale),
            }

        # Free per-block I/O cache
        del io_collectors

        block_elapsed = time.time() - block_t0
        n_layers = len(block_weight_results)
        print(f"  Block {block_idx}: {n_layers} layers, {block_elapsed:.1f}s")

    phase_b_elapsed = time.time() - phase_b_t0

    # Save config
    save_config(args.output_dir, args, all_metrics)

    elapsed = time.time() - t0
    print(f"\nPhase A (Hessian): {phase_a_elapsed:.0f}s")
    print(f"Phase B (GPTQ + alpha): {phase_b_elapsed:.0f}s")
    print(f"Total: {len(all_metrics)} layers quantized in {elapsed:.0f}s")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
