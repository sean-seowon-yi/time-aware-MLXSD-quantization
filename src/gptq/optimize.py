"""GPTQ pipeline orchestrator for SD3 MMDiT.

Two-phase design:

  Phase A — Global Hessian pass: install collectors on ALL 285 linears
            across all 24 blocks. Run all prompts once (900 fwd passes).
            Only Hessian accumulation, no I/O caching.

  Phase B — GPTQ quantize all layers, then global alpha search: install
            alpha accumulators on ALL linears, run a small subset of
            prompts. Each accumulator evaluates 22 alpha_scale candidates
            (0.01–100.0) via vectorized matmuls and accumulates MSE.

Weight modes (--weight-mode):
  gptq   — Full GPTQ pipeline (Phase A + B). Default.
  rtn    — Round-to-nearest per-channel quantization (skip Phase A).
  fp16   — No weight quantization; alpha search only (activation-only W16A8).

Prompt file format: tab-separated ``seed<TAB>prompt`` per line.

Usage:
    python -m src.gptq.optimize \\
        --prompts src/calibration_sample_generation/sample_prompts.txt \\
        --poly-schedule polynomial_clipping_schedule.json \\
        --bits-w 4 \\
        --output-dir gptq_output \\
        --num-steps 30 --cfg-weight 4.0 \\
        --damp-percent 0.01 --block-size 128 \\
        --max-prompts 30 --alpha-prompts 5

    # RTN W4A8:
    python -m src.gptq.optimize \\
        --prompts sample_prompts.txt \\
        --poly-schedule polynomial_clipping_schedule.json \\
        --weight-mode rtn --bits-w 4 \\
        --output-dir rtn_w4a8_output --alpha-prompts 5

    # Activation-only W16A8:
    python -m src.gptq.optimize \\
        --prompts sample_prompts.txt \\
        --poly-schedule polynomial_clipping_schedule.json \\
        --weight-mode fp16 \\
        --output-dir w16a8_output --alpha-prompts 5
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .utils import (
    _get_block_linears, full_path_to_poly_key, load_prompt_file,
    compute_scales, dequantize,
)
from .gptq_quantize import gptq_quantize
from .hessian_collector import (
    collect_hessians_global,
    collect_alpha_mse_global,
)
from .inference import compute_static_alphas, load_gptq_weights


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

    weight_mode = getattr(args, "weight_mode", "gptq")
    config = {
        "method": weight_mode,
        "bits_w": args.bits_w if weight_mode != "fp16" else 16,
        "group_size": args.group_size,
        "act_quant_mode": "static" if args.static_act_quant else "poly",
        "damp_percent": args.damp_percent,
        "block_size": args.block_size,
        "poly_schedule": str(args.poly_schedule),
        "num_prompts": args.max_prompts or "all",
        "alpha_prompts": args.alpha_prompts,
        "num_steps": args.num_steps,
        "cfg_weight": args.cfg_weight,
        "seeds": "per-prompt",
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
                        help="Tab-separated file: seed<TAB>prompt per line")
    parser.add_argument("--poly-schedule", type=Path, required=True,
                        help="Path to polynomial_clipping_schedule.json")
    parser.add_argument("--bits-w", type=int, default=4, choices=[4, 8],
                        help="Weight quantization bits (default: 4)")
    parser.add_argument("--output-dir", type=Path, default=Path("gptq_output"))
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-weight", type=float, default=4.0)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for per-group weight quantization (default: 128). "
                        "Use 0 for per-channel (no grouping).")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts for Hessian collection")
    parser.add_argument("--alpha-prompts", type=int, default=5,
                        help="Number of prompts for Phase B alpha search")
    parser.add_argument("--subsample-rows", type=int, default=128,
                        help="Max activation rows sampled per forward call in Phase B "
                        "alpha search (default: 128). Lower = faster but noisier MSE.")
    parser.add_argument("--hessian-cache", type=Path, default=None,
                        help="Path to save/load Hessian .npz checkpoint. "
                        "If the file exists, Phase A is skipped and Hessians "
                        "are loaded from it. Otherwise Phase A runs and saves here.")
    parser.add_argument("--static-act-quant", action="store_true",
                        help="Use static (timestep-agnostic) activation clipping "
                        "instead of polynomial clipping for alpha search")
    parser.add_argument("--skip-gptq", action="store_true",
                        help="Skip Phase A and B.1 (Hessian collection and GPTQ "
                        "quantization). Loads existing weights from output-dir/weights/ "
                        "and reruns alpha search only.")
    parser.add_argument("--weight-mode", type=str, default="gptq",
                        choices=["gptq", "rtn", "fp16"],
                        help="Weight quantization method: "
                        "'gptq' = full GPTQ (default), "
                        "'rtn' = round-to-nearest per-channel, "
                        "'fp16' = no weight quant (activation-only alpha search)")
    parser.add_argument("--raw-hessian", action="store_true",
                        help="Use full-precision activations (not fake-quantized) "
                        "when accumulating Hessians in Phase A. Avoids conditioning "
                        "weight quantization on activation clipping parameters that "
                        "are not finalized until Phase B alpha search.")
    args = parser.parse_args()

    # Load prompts (seed, prompt) pairs
    all_prompt_entries = load_prompt_file(args.prompts)
    if args.max_prompts is not None:
        all_prompt_entries = all_prompt_entries[:args.max_prompts]
    print(f"Loaded {len(all_prompt_entries)} prompts from {args.prompts}")

    # Split prompts: all for Phase A, first alpha_prompts for Phase B
    alpha_prompt_entries = all_prompt_entries[:args.alpha_prompts]
    print(f"Phase A: {len(all_prompt_entries)} prompts for Hessian collection")
    print(f"Phase B: {len(alpha_prompt_entries)} prompts for alpha search I/O cache")

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
    phase_a_elapsed = 0.0

    if args.weight_mode == "fp16":
        # -----------------------------------------------------------------
        # FP16 mode: no weight quantization, use original weights as-is
        # -----------------------------------------------------------------
        print("Weight mode: fp16 (no weight quantization)")
        n_blocks = len(pipeline.mmdit.multimodal_transformer_blocks)
        all_weight_results = {}
        for block_idx in range(n_blocks):
            block = pipeline.mmdit.multimodal_transformer_blocks[block_idx]
            for full_path, layer in _get_block_linears(block, is_mm=True):
                poly_key = full_path_to_poly_key(block_idx, full_path)
                W = np.array(layer.weight, dtype=np.float32)
                # 8-bit identity quantization to fit the (W_q_int, scales) format
                scales = compute_scales(W, 8, args.group_size)
                if scales.ndim == 1:
                    s_expand = scales[:, None]
                else:
                    n_groups = scales.shape[1]
                    gs = (W.shape[1] + n_groups - 1) // n_groups
                    s_expand = np.repeat(scales, gs, axis=1)[:, :W.shape[1]]
                W_q_int = np.clip(np.round(W / s_expand), -127, 127).astype(np.int8)
                W_dequant = dequantize(W_q_int, scales)
                weight_mse = float(np.sum((W - W_dequant) ** 2))
                all_weight_results[poly_key] = (W_q_int, scales, weight_mse)

        weight_mses = {
            pk: float(wmse)
            for pk, (_, _, wmse) in all_weight_results.items()
        }
        print(f"FP16 mode: {len(all_weight_results)} layers (no weight quant)")

    elif args.weight_mode == "rtn":
        # -----------------------------------------------------------------
        # RTN mode: round-to-nearest per-channel quantization
        # -----------------------------------------------------------------
        print(f"Weight mode: rtn (round-to-nearest, {args.bits_w}-bit, group_size={args.group_size})")
        n_blocks = len(pipeline.mmdit.multimodal_transformer_blocks)
        all_weight_results = {}
        qmax = 2 ** (args.bits_w - 1) - 1

        for block_idx in tqdm(range(n_blocks), desc="RTN quantize"):
            block = pipeline.mmdit.multimodal_transformer_blocks[block_idx]
            for full_path, layer in _get_block_linears(block, is_mm=True):
                poly_key = full_path_to_poly_key(block_idx, full_path)
                W = np.array(layer.weight, dtype=np.float32)
                scales = compute_scales(W, args.bits_w, args.group_size)
                if scales.ndim == 1:
                    s_expand = scales[:, None]
                else:
                    n_groups = scales.shape[1]
                    gs = (W.shape[1] + n_groups - 1) // n_groups
                    s_expand = np.repeat(scales, gs, axis=1)[:, :W.shape[1]]
                W_q_int = np.clip(
                    np.round(W / s_expand), -qmax, qmax
                ).astype(np.int8)
                W_dequant = dequantize(W_q_int, scales)
                weight_mse = float(np.sum((W - W_dequant) ** 2))
                all_weight_results[poly_key] = (W_q_int, scales, weight_mse)

            # Save block weights
            prefix = f"mm{block_idx}_"
            block_results = {
                k: v for k, v in all_weight_results.items()
                if k.startswith(prefix) and k[len(prefix):].split("_")[0] in ("img", "txt")
            }
            save_block(args.output_dir, block_idx, block_results)

        weight_mses = {
            pk: float(wmse)
            for pk, (_, _, wmse) in all_weight_results.items()
        }
        print(f"RTN done: {len(all_weight_results)} layers quantized")

    elif args.skip_gptq:
        # -----------------------------------------------------------------
        # Skip Phase A and B.1 — load existing weights from disk
        # -----------------------------------------------------------------
        print("Skipping Phase A and B.1 (--skip-gptq)")
        print(f"Loading existing GPTQ weights from {args.output_dir}/weights/")
        existing_config, gptq_weights = load_gptq_weights(args.output_dir)

        # Reconstruct all_weight_results with weight_mse from existing config
        existing_wmses = existing_config.get("weight_mse", {})
        all_weight_results = {}
        for poly_key, (W_q_int, scales) in gptq_weights.items():
            wmse = existing_wmses.get(poly_key, 0.0)
            all_weight_results[poly_key] = (W_q_int, scales, wmse)

        weight_mses = {
            poly_key: float(wmse)
            for poly_key, wmse in existing_wmses.items()
        }
        print(f"Loaded {len(all_weight_results)} layers from disk")

    else:
        # -----------------------------------------------------------------
        # GPTQ mode: Phase A (Hessian) + B.1 (GPTQ quantize)
        # -----------------------------------------------------------------
        hessian_cache_path = args.hessian_cache
        if hessian_cache_path is None:
            hessian_cache_path = args.output_dir / "hessians.npz"

        if hessian_cache_path.exists():
            print(f"Loading cached Hessians from {hessian_cache_path}")
            phase_a_t0 = time.time()
            with np.load(hessian_cache_path) as loaded:
                hessians = {key: loaded[key] for key in loaded.files}
            phase_a_elapsed = time.time() - phase_a_t0
            print(f"Loaded {len(hessians)} Hessians in {phase_a_elapsed:.1f}s")
        else:
            phase_a_t0 = time.time()
            all_collectors = collect_hessians_global(
                pipeline, denoiser, all_prompt_entries, poly_schedule,
                num_steps=args.num_steps,
                cfg_weight=args.cfg_weight,
                latent_size=args.latent_size,
                static_alphas=static_alphas,
                raw_hessian=args.raw_hessian,
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

        # -------------------------------------------------------------------
        # B.1: GPTQ quantize all blocks (pure NumPy, fast)
        # -------------------------------------------------------------------
        n_blocks = len(pipeline.mmdit.multimodal_transformer_blocks)
        all_weight_results = {}  # poly_key -> (W_q_int, scales, weight_mse)
        for block_idx in tqdm(range(n_blocks), desc="GPTQ quantize"):
            block = pipeline.mmdit.multimodal_transformer_blocks[block_idx]
            for full_path, layer in _get_block_linears(block, is_mm=True):
                poly_key = full_path_to_poly_key(block_idx, full_path)
                W = np.array(layer.weight, dtype=np.float32)
                H = hessians[poly_key]
                W_q_int, scales, weight_mse = gptq_quantize(
                    W, H, args.bits_w, args.damp_percent, args.block_size,
                    group_size=args.group_size,
                )
                all_weight_results[poly_key] = (W_q_int, scales, weight_mse)

            # Save block weights immediately
            prefix = f"mm{block_idx}_"
            block_results = {
                k: v for k, v in all_weight_results.items()
                if k.startswith(prefix) and k[len(prefix):].split("_")[0] in ("img", "txt")
            }
            save_block(args.output_dir, block_idx, block_results)

        print(f"GPTQ done: {len(all_weight_results)} layers quantized")

        # Free Hessians — no longer needed, already saved to disk
        del hessians

        # Extract weight MSEs before alpha search (so we can free weight arrays)
        weight_mses = {
            poly_key: float(weight_mse)
            for poly_key, (_W_q_int, _scales, weight_mse) in all_weight_results.items()
        }

    # -----------------------------------------------------------------------
    # B.2: Global alpha search (all blocks, one pass)
    # -----------------------------------------------------------------------
    phase_b_t0 = time.time()
    alpha_results = collect_alpha_mse_global(
        pipeline, denoiser, alpha_prompt_entries, poly_schedule,
        all_weight_results,
        num_steps=args.num_steps,
        cfg_weight=args.cfg_weight,
        latent_size=args.latent_size,
        static_alphas=static_alphas,
        subsample_rows=args.subsample_rows,
    )

    # Free weight arrays — no longer needed (already saved to disk per-block)
    del all_weight_results

    # B.3: Assemble metrics
    all_metrics = {}
    for poly_key, wmse in weight_mses.items():
        alpha_scale, act_mse = alpha_results.get(poly_key, (1.0, float("inf")))
        all_metrics[poly_key] = {
            "weight_mse": wmse,
            "activation_mse": float(act_mse),
            "alpha_scale": float(alpha_scale),
        }

    phase_b_elapsed = time.time() - phase_b_t0

    # Save config
    save_config(args.output_dir, args, all_metrics)

    elapsed = time.time() - t0
    print(f"\nWeight mode: {args.weight_mode}")
    if args.weight_mode == "gptq":
        print(f"Phase A (Hessian): {phase_a_elapsed:.0f}s")
    print(f"Phase B (alpha search): {phase_b_elapsed:.0f}s")
    print(f"Total: {len(all_metrics)} layers in {elapsed:.0f}s")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
