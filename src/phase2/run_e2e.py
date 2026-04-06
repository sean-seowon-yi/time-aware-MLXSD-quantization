#!/usr/bin/env python3
"""End-to-end quantization pipeline: data collection → calibration → CSB → quantize → save.

Loads the model once, runs Phase 1 diagnostic collection (prompts through the
denoiser with hooks), then immediately performs Phase 2 (SSC calibration,
CSB absorption, W4A8 quantization) and saves the quantized model.

Usage
-----
# Full run with defaults (dynamic A8) → quantized/w4a8_max_a0.50_gs64/
python -m src.phase2.run_e2e --output-dir quantized/

# Static A8 per-tensor → quantized/w4a8_max_a0.50_gs64_static/
python -m src.phase2.run_e2e --output-dir quantized/ --act-quant static --skip-collection

# Static A8 per-channel → quantized/w4a8_max_a0.50_gs64_staticpc/
python -m src.phase2.run_e2e --output-dir quantized/ --act-quant static \
    --static-granularity per_channel --skip-collection

# Static A8 with global-max scale mode
python -m src.phase2.run_e2e --output-dir quantized/ --act-quant static \
    --static-mode global_max --skip-collection

# Override alpha → quantized/w4a8_max_a0.30_gs64/
python -m src.phase2.run_e2e --output-dir quantized/ --alpha 0.3

# Skip Phase 1 collection (use existing diagnostics)
python -m src.phase2.run_e2e --output-dir quantized/ --skip-collection

Steps
-----
1. Load SD3 Medium pipeline (once)
2. Phase 1: run prompts through denoiser with hooks → activation trajectories + weight salience
3. Phase 2a: SSC-weighted calibration → balancing vectors
4. Phase 2b: CSB absorption into adaLN + weight balancing
5. Phase 2c: W4A8 quantization (4-bit weights, dynamic or static 8-bit activations)
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
        help="Number of calibration prompt-seed pairs (default: all 100)",
    )
    p1.add_argument("--num-steps", type=int, default=None, help="Denoising steps (default: 30)")
    p1.add_argument("--cfg-weight", type=float, default=None, help="CFG scale (default: 4.0)")

    # --- Phase 2: quantization settings ---
    p2 = parser.add_argument_group("Phase 2 — quantization")
    p2.add_argument("--alpha", type=float, default=None, help="CSB exponent (default: 0.5)")
    p2.add_argument("--qkv-method", type=str, default=None, choices=["max", "geomean", "l2"])
    p2.add_argument("--group-size", type=int, default=None, help="W4 group size (default: 64)")
    p2.add_argument("--bits", type=int, default=None, help="Weight bits (default: 4)")
    p2.add_argument("--final-layer-bits", type=int, default=None, help="Final layer bits (default: 4)")
    p2.add_argument("--ssc-tau", type=float, default=None,
                     help="SSC temperature (default: 1.0). Values < 1 sharpen time-weighting.")
    p2.add_argument("--per-token-rho-threshold", type=float, default=None,
                     help="Layers with mean rho above this use per-token A8 (default: 0.5)")

    # --- Activation quantization mode ---
    p3 = parser.add_argument_group("Activation quantization mode")
    p3.add_argument("--act-quant", type=str, default="dynamic",
                     choices=["dynamic", "static"],
                     help="Activation quantization mode: 'dynamic' (default) computes "
                          "scale per-forward; 'static' uses pre-computed scales from "
                          "Phase 1 calibration.")
    p3.add_argument("--static-mode", type=str, default="ssc_weighted",
                     choices=["ssc_weighted", "global_max"],
                     help="How to aggregate calibration data across timesteps for "
                          "static scales (default: ssc_weighted).")
    p3.add_argument("--static-granularity", type=str, default="per_tensor",
                     choices=["per_tensor", "per_channel"],
                     help="Static scale granularity: one scale per layer (per_tensor) "
                          "or one per input channel (per_channel). Default: per_tensor.")

    args = parser.parse_args()

    from pathlib import Path
    import mlx.core as mx

    from .config import (
        DIFFUSIONKIT_SRC,
        MODEL_VERSION,
        PHASE2_CONFIG,
        PIPELINE_KWARGS,
        config_tag,
    )

    output_root = Path(args.output_dir)
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
    if args.ssc_tau is not None:
        p2_cfg["ssc_tau"] = args.ssc_tau
    if args.per_token_rho_threshold is not None:
        p2_cfg["per_token_rho_threshold"] = args.per_token_rho_threshold
    p2_cfg["model_version"] = MODEL_VERSION
    if args.act_quant == "static":
        p2_cfg["static_granularity"] = args.static_granularity

    tag = config_tag(p2_cfg, act_quant=args.act_quant)
    output_dir = output_root / tag
    logger.info("Config tag: %s  →  output: %s", tag, output_dir)

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

        from ..phase1.config import CALIBRATION_PAIRS, DIAG_CONFIG
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

        n = args.num_prompts or len(CALIBRATION_PAIRS)
        pairs = CALIBRATION_PAIRS[:n]

        logger.info(
            "Collection settings: %d prompt-seed pairs, %d steps, CFG %.1f",
            len(pairs), num_steps, cfg_weight,
        )

        weight_stats = compute_weight_salience(registry)
        save_weight_stats(weight_stats, diagnostics_dir)

        collector = ChannelStatsCollector()
        hooks = install_hooks(registry, collector)

        t_collect = time.time()
        run_diagnostic_collection(
            pipeline, pairs, collector,
            num_steps=num_steps,
            latent_size=latent_size,
            cfg_weight=cfg_weight,
        )
        logger.info("Collection finished in %.1f s", time.time() - t_collect)

        from mlx.utils import tree_flatten
        adaln_cache = [
            (k, v) for k, v in tree_flatten(pipeline.mmdit.parameters())
            if "adaLN" in k
        ]
        pipeline.mmdit.load_weights(adaln_cache, strict=False)
        first_prompt = pairs[0][1]
        conditioning, pooled_conditioning = pipeline.encode_text(
            first_prompt, cfg_weight=cfg_weight,
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
        for _blk in pipeline.mmdit.multimodal_transformer_blocks:
            if hasattr(_blk.image_transformer_block, "_modulation_params"):
                _blk.image_transformer_block._modulation_params = {}
            if hasattr(_blk.text_transformer_block, "_modulation_params"):
                _blk.text_transformer_block._modulation_params = {}
        if hasattr(pipeline.mmdit.final_layer, "_modulation_params"):
            pipeline.mmdit.final_layer._modulation_params = {}

        remove_hooks(hooks)

        pipeline.mmdit.load_weights(adaln_cache, strict=False)
        if hasattr(pipeline.mmdit, "to_offload"):
            pipeline.mmdit.to_offload = []

        save_activation_stats(collector, registry, diagnostics_dir / "activation_stats")
        save_adaln_stats(adaln_stats, diagnostics_dir)
        save_config(pairs, registry, diagnostics_dir)

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
    logger.info("STEP 5/6 — Phase 2c: W4A8 quantization (act_quant=%s)", args.act_quant)
    logger.info("=" * 60)

    t_quant = time.time()

    if args.act_quant == "static":
        from .quantize_static import (
            compute_static_scales,
            quantize_model_static,
            save_quantized_model_static,
        )

        static_scales = compute_static_scales(
            light_registry,
            diagnostics_dir,
            calibration,
            config=p2_cfg,
            mode=args.static_mode,
            granularity=args.static_granularity,
        )
        layer_meta = quantize_model_static(
            pipeline.mmdit, registry, b_inv_map, static_scales, config=p2_cfg,
        )
    else:
        from .quantize import quantize_model

        layer_meta = quantize_model(
            pipeline.mmdit, registry, b_inv_map, p2_cfg,
            mean_rhos=calibration.get("mean_rhos"),
        )

    logger.info("Quantization finished in %.1f s", time.time() - t_quant)

    # ==================================================================
    # Step 6: Save
    # ==================================================================
    logger.info("=" * 60)
    logger.info("STEP 6/6 — Saving quantized model")
    logger.info("=" * 60)

    if args.act_quant == "static":
        save_quantized_model_static(
            pipeline.mmdit, output_dir, p2_cfg, layer_meta,
            calibration["b_inv_layers"], static_scales,
            granularity=args.static_granularity,
            mode=args.static_mode,
        )
    else:
        from .quantize import save_quantized_model

        save_quantized_model(
            pipeline.mmdit, output_dir, p2_cfg, layer_meta,
            calibration["b_inv_layers"],
        )

    total_elapsed = time.time() - t_total
    n_quantized = len(layer_meta)
    n_online = len(b_inv_map)

    if args.act_quant == "static":
        n_per_channel = sum(
            1 for m in layer_meta.values() if m.get("per_channel", False)
        )
        logger.info(
            "\n" + "=" * 60 + "\n"
            "  END-TO-END COMPLETE (static A8)\n"
            "=" * 60 + "\n"
            "  Total time:        %.1f s (%.1f min)\n"
            "  Quantized layers:  %d\n"
            "  Online b_inv:      %d\n"
            "  Static mode:       %s\n"
            "  Granularity:       %s\n"
            "  Per-channel layers:%d\n"
            "  Group size:        %d\n"
            "  Weight bits:       %d\n"
            "  Alpha:             %.2f\n"
            "  QKV method:        %s\n"
            "  SSC tau:           %.2f\n"
            "  Diagnostics:       %s\n"
            "  Quantized model:   %s",
            total_elapsed, total_elapsed / 60,
            n_quantized, n_online,
            args.static_mode, args.static_granularity, n_per_channel,
            p2_cfg["group_size"], p2_cfg["bits"],
            p2_cfg["alpha"], p2_cfg["qkv_method"],
            p2_cfg.get("ssc_tau", 1.0),
            diagnostics_dir, output_dir,
        )
    else:
        n_per_token = sum(
            1 for m in layer_meta.values() if m.get("per_token", False)
        )
        logger.info(
            "\n" + "=" * 60 + "\n"
            "  END-TO-END COMPLETE\n"
            "=" * 60 + "\n"
            "  Total time:        %.1f s (%.1f min)\n"
            "  Quantized layers:  %d\n"
            "  Online b_inv:      %d\n"
            "  Per-token A8:      %d\n"
            "  Group size:        %d\n"
            "  Weight bits:       %d\n"
            "  Alpha:             %.2f\n"
            "  QKV method:        %s\n"
            "  SSC tau:           %.2f\n"
            "  Diagnostics:       %s\n"
            "  Quantized model:   %s",
            total_elapsed, total_elapsed / 60,
            n_quantized, n_online, n_per_token,
            p2_cfg["group_size"], p2_cfg["bits"],
            p2_cfg["alpha"], p2_cfg["qkv_method"],
            p2_cfg.get("ssc_tau", 1.0),
            diagnostics_dir, output_dir,
        )


if __name__ == "__main__":
    main()
