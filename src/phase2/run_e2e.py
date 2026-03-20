#!/usr/bin/env python3
"""End-to-end quantization pipeline: data collection → calibration → CSB → quantize → save.

Loads the model once, runs Phase 1 diagnostic collection (prompts through the
denoiser with hooks), then immediately performs Phase 2 (SSC calibration,
CSB absorption, W4A8 quantization) and saves the quantized model.

Usage
-----
# Full run with defaults (100 prompts, seed 42, 30 steps, CFG 4.0)
python -m src.phase2.run_e2e --output-dir quantized/

# Quick test with 2 prompts
python -m src.phase2.run_e2e --output-dir quantized/ --num-prompts 2

# Override Phase 2 hyperparameters
python -m src.phase2.run_e2e --output-dir quantized/ --alpha 0.3 --qkv-method geomean

# Override generation settings
python -m src.phase2.run_e2e --output-dir quantized/ --num-steps 20 --cfg-weight 5.0

# Skip Phase 1 collection (use existing diagnostics)
python -m src.phase2.run_e2e --output-dir quantized/ --skip-collection

Steps
-----
1. Load SD3 Medium pipeline (once)
2. Phase 1: run prompts through denoiser with hooks → activation trajectories + weight salience
3. Phase 2a: SSC-weighted calibration → balancing vectors
4. Phase 2b: CSB absorption into adaLN + weight balancing
5. Phase 2c: W4A8 quantization (4-bit weights, 8-bit activations)
6. Save quantized model + metadata
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end: data collection → calibration → CSB → W4A8 quantization",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir", type=str, default="quantized",
        help="Directory for quantized model output (default: quantized/)",
    )
    parser.add_argument(
        "--diagnostics-dir", type=str, default="diagnostics",
        help="Directory for Phase 1 diagnostic data (default: diagnostics/)",
    )

    # --- Phase 1: collection settings ---
    p1 = parser.add_argument_group("Phase 1 — data collection")
    p1.add_argument(
        "--skip-collection", action="store_true",
        help="Skip Phase 1 collection; use existing data in --diagnostics-dir",
    )
    p1.add_argument(
        "--num-prompts", type=int, default=None,
        help="Number of calibration prompts (default: all 100)",
    )
    p1.add_argument(
        "--num-seeds", type=int, default=None,
        help="Number of seeds (default: 1, i.e. seed 42)",
    )
    p1.add_argument("--num-steps", type=int, default=None, help="Denoising steps (default: 30)")
    p1.add_argument("--cfg-weight", type=float, default=None, help="CFG scale (default: 4.0)")
    p1.add_argument("--seed", type=int, default=None, help="Override seed (default: 42)")

    # --- Phase 2: quantization settings ---
    p2 = parser.add_argument_group("Phase 2 — quantization")
    p2.add_argument("--alpha", type=float, default=None, help="CSB exponent (default: 0.5)")
    p2.add_argument("--qkv-method", type=str, default=None, choices=["max", "geomean"])
    p2.add_argument("--group-size", type=int, default=None, help="W4 group size (default: 64)")
    p2.add_argument("--bits", type=int, default=None, help="Weight bits (default: 4)")
    p2.add_argument("--final-layer-bits", type=int, default=None, help="Final layer bits (default: 4)")

    args = parser.parse_args()

    from pathlib import Path
    import mlx.core as mx

    from .config import (
        DIFFUSIONKIT_SRC,
        MODEL_VERSION,
        PHASE2_CONFIG,
        PIPELINE_KWARGS,
    )

    output_dir = Path(args.output_dir)
    diagnostics_dir = Path(args.diagnostics_dir)

    # --- Build Phase 2 config ---
    p2_cfg = {**PHASE2_CONFIG}
    if args.alpha is not None:
        p2_cfg["alpha"] = args.alpha
    if args.qkv_method is not None:
        p2_cfg["qkv_method"] = args.qkv_method
    if args.group_size is not None:
        p2_cfg["group_size"] = args.group_size
    if args.bits is not None:
        p2_cfg["bits"] = args.bits
    if args.final_layer_bits is not None:
        p2_cfg["final_layer_bits"] = args.final_layer_bits
    p2_cfg["model_version"] = MODEL_VERSION

    t_total = time.time()

    # ==================================================================
    # Step 1: Load model
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 1/6 — Loading SD3 Medium pipeline")
    logger.info("=" * 60)

    sys.path.insert(0, DIFFUSIONKIT_SRC)
    from diffusionkit.mlx import DiffusionPipeline

    pipeline = DiffusionPipeline(
        **PIPELINE_KWARGS,
        model_version=MODEL_VERSION,
    )
    logger.info("Pipeline loaded. dtype=%s", pipeline.dtype)

    # Build registry (with module references — needed for both phases)
    from ..phase1.registry import build_layer_registry
    registry = build_layer_registry(pipeline.mmdit)
    logger.info("Registry: %d linear layers", len(registry))

    # ==================================================================
    # Step 2: Phase 1 — Diagnostic data collection
    # ==================================================================
    if args.skip_collection:
        logger.info("=" * 60)
        logger.info("STEP 2/6 — SKIPPED (using existing data in %s)", diagnostics_dir)
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("STEP 2/6 — Phase 1: running prompts through denoiser with hooks")
        logger.info("=" * 60)

        from ..phase1.config import DIAG_CONFIG, DIAGNOSTIC_PROMPTS
        from ..phase1.collect import (
            collect_adaln_stats,
            compute_weight_salience,
            run_diagnostic_collection,
            save_activation_stats,
            save_adaln_stats,
            save_config,
            save_weight_stats,
        )
        from ..phase1.hooks import ChannelStatsCollector, install_hooks, remove_hooks

        num_steps = args.num_steps or DIAG_CONFIG["num_steps"]
        cfg_weight = args.cfg_weight if args.cfg_weight is not None else DIAG_CONFIG["cfg_weight"]
        latent_size = tuple(DIAG_CONFIG["latent_size"])

        n_prompts = args.num_prompts or len(DIAGNOSTIC_PROMPTS)
        prompts = DIAGNOSTIC_PROMPTS[:n_prompts]

        if args.seed is not None:
            seeds = [args.seed]
        elif args.num_seeds is not None:
            seeds = DIAG_CONFIG["seed_range"][:args.num_seeds]
        else:
            seeds = DIAG_CONFIG["seed_range"]

        logger.info(
            "Collection settings: %d prompts, %d seeds, %d steps, CFG %.1f",
            len(prompts), len(seeds), num_steps, cfg_weight,
        )

        # Weight salience (time-independent, computed once)
        weight_stats = compute_weight_salience(registry)
        save_weight_stats(weight_stats, diagnostics_dir)

        # Install hooks and run collection
        collector = ChannelStatsCollector()
        hooks = install_hooks(registry, collector)

        t_collect = time.time()
        run_diagnostic_collection(
            pipeline, prompts, seeds, collector,
            num_steps=num_steps,
            latent_size=latent_size,
            cfg_weight=cfg_weight,
        )
        logger.info("Collection finished in %.1f s", time.time() - t_collect)

        # Collect adaLN stats
        # adaLN weights are already restored after run_diagnostic_collection;
        # capture from memory instead of reloading full checkpoint from disk.
        from mlx.utils import tree_flatten
        adaln_cache = [
            (k, v) for k, v in tree_flatten(pipeline.mmdit.parameters())
            if "adaLN" in k
        ]
        pipeline.mmdit.load_weights(adaln_cache, strict=False)
        conditioning, pooled_conditioning = pipeline.encode_text(
            prompts[0], cfg_weight=cfg_weight,
        )
        if cfg_weight <= 0:
            conditioning = conditioning[:1]
            pooled_conditioning = pooled_conditioning[:1]
        conditioning = conditioning.astype(pipeline.activation_dtype)
        pooled_conditioning = pooled_conditioning.astype(pipeline.activation_dtype)
        mx.eval(conditioning, pooled_conditioning)
        sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
        timesteps = pipeline.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
        pipeline.mmdit.cache_modulation_params(pooled_conditioning, timesteps)
        adaln_stats = collect_adaln_stats(pipeline.mmdit)
        pipeline.mmdit.clear_modulation_params_cache()

        # Remove hooks
        remove_hooks(hooks)

        # Restore adaLN weights (cache_modulation_params offloads them
        # to empty arrays; clear_modulation_params_cache does NOT restore)
        pipeline.mmdit.load_weights(adaln_cache, strict=False)

        # cache_modulation_params leaves a `to_offload` list of module refs
        # on the MMDiT; remove it so tree_flatten during save doesn't pick
        # up duplicate keys like `to_offload.0.weight`.
        if hasattr(pipeline.mmdit, "to_offload"):
            delattr(pipeline.mmdit, "to_offload")

        # Save Phase 1 data
        save_activation_stats(collector, registry, diagnostics_dir / "activation_stats")
        save_adaln_stats(adaln_stats, diagnostics_dir)
        save_config(prompts, seeds, registry, diagnostics_dir)

        logger.info("Phase 1 data saved to %s", diagnostics_dir)

    # ==================================================================
    # Step 3: Phase 2a — Calibration (SSC + balancing vectors)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 3/6 — Phase 2a: SSC calibration")
    logger.info("=" * 60)

    from .calibrate import calibrate_all_layers, save_calibration

    t_cal = time.time()
    # Build lightweight registry entries for layers without module refs
    # (calibration only needs name/block/family/side, not module)
    light_registry = []
    for entry in registry:
        light_registry.append({
            "name": entry["name"],
            "block": entry["block"],
            "family": entry["family"],
            "side": entry["side"],
        })

    calibration = calibrate_all_layers(light_registry, diagnostics_dir, p2_cfg)
    save_calibration(calibration, output_dir)
    logger.info("Calibration finished in %.1f s", time.time() - t_cal)
    logger.info(
        "  %d balancing vectors, %d need online b_inv",
        len(calibration["balancing_vectors"]), len(calibration["b_inv_layers"]),
    )

    # ==================================================================
    # Step 4: Phase 2b — CSB absorption + weight balancing
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 4/6 — Phase 2b: CSB re-parameterization")
    logger.info("=" * 60)

    from .balance import apply_csb_to_model
    from .quantize import patch_pipeline_for_quantized_inference

    t_csb = time.time()
    b_inv_map = apply_csb_to_model(pipeline.mmdit, registry, calibration)
    logger.info("CSB applied in %.1f s", time.time() - t_csb)

    patch_pipeline_for_quantized_inference(pipeline)

    # ==================================================================
    # Step 5: Phase 2c — W4A8 quantization
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 5/6 — Phase 2c: W4A8 quantization")
    logger.info("=" * 60)

    from .quantize import quantize_model

    t_quant = time.time()
    layer_meta = quantize_model(pipeline.mmdit, registry, b_inv_map, p2_cfg)
    logger.info("Quantization finished in %.1f s", time.time() - t_quant)

    # ==================================================================
    # Step 6: Save
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 6/6 — Saving quantized model")
    logger.info("=" * 60)

    from .quantize import save_quantized_model

    save_quantized_model(
        pipeline.mmdit, output_dir, p2_cfg, layer_meta, calibration["b_inv_layers"],
    )

    total_elapsed = time.time() - t_total
    n_quantized = len(layer_meta)
    n_online = len(b_inv_map)

    logger.info(
        "\n" + "=" * 60 + "\n"
        "  END-TO-END COMPLETE\n"
        "=" * 60 + "\n"
        "  Total time:        %.1f s (%.1f min)\n"
        "  Quantized layers:  %d\n"
        "  Online b_inv:      %d\n"
        "  Group size:        %d\n"
        "  Weight bits:       %d\n"
        "  Alpha:             %.2f\n"
        "  QKV method:        %s\n"
        "  Diagnostics:       %s\n"
        "  Quantized model:   %s",
        total_elapsed, total_elapsed / 60,
        n_quantized, n_online,
        p2_cfg["group_size"], p2_cfg["bits"],
        p2_cfg["alpha"], p2_cfg["qkv_method"],
        diagnostics_dir, output_dir,
    )


if __name__ == "__main__":
    main()
