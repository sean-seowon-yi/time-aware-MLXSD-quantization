#!/usr/bin/env python3
"""Entry point: collect Phase 1 diagnostic data.

Usage:
    python -m src.phase1.run_collection [--pilot]

With --pilot, runs 2 prompts × 1 seed for a quick sanity check.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import mlx.core as mx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 diagnostic collection")
    parser.add_argument(
        "--pilot", action="store_true",
        help="Quick pilot run: 2 prompts, 1 seed, verify hooks fire correctly",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Limit number of prompts (default: all 100)",
    )
    args = parser.parse_args()

    from .config import CALIBRATION_PAIRS, DIAG_CONFIG
    from .collect import (
        collect_adaln_stats,
        compute_weight_salience,
        run_diagnostic_collection,
        save_activation_stats,
        save_adaln_stats,
        save_config,
        save_weight_stats,
    )
    from .hooks import ChannelStatsCollector, install_hooks, remove_hooks
    from .registry import build_layer_registry

    if args.pilot:
        pairs = CALIBRATION_PAIRS[:2]
        logger.info("=== PILOT RUN: 2 prompt-seed pairs ===")
    else:
        n = args.num_prompts or len(CALIBRATION_PAIRS)
        pairs = CALIBRATION_PAIRS[:n]

    logger.info("Prompt-seed pairs: %d", len(pairs))

    # --- Load pipeline ---
    logger.info("Loading DiffusionPipeline ...")
    sys.path.insert(0, "DiffusionKit/python/src")
    from diffusionkit.mlx import DiffusionPipeline

    pipeline = DiffusionPipeline(
        w16=DIAG_CONFIG["w16"],
        shift=DIAG_CONFIG["shift"],
        use_t5=DIAG_CONFIG["use_t5"],
        model_version=DIAG_CONFIG["model_version"],
        low_memory_mode=DIAG_CONFIG["low_memory_mode"],
    )
    logger.info("Pipeline loaded. Model dtype: %s", pipeline.dtype)

    # --- Build registry ---
    registry = build_layer_registry(pipeline.mmdit)
    logger.info("Registry: %d linear layers", len(registry))
    assert len(registry) == 287, f"Expected 287, got {len(registry)}"

    # --- Weight salience (once) ---
    weight_stats = compute_weight_salience(registry)
    save_weight_stats(weight_stats)

    # --- Install hooks ---
    collector = ChannelStatsCollector()
    hooks = install_hooks(registry, collector)

    # --- Run collection ---
    cfg_weight = DIAG_CONFIG["cfg_weight"]
    t0 = time.time()
    run_diagnostic_collection(
        pipeline, pairs, collector,
        num_steps=DIAG_CONFIG["num_steps"],
        latent_size=tuple(DIAG_CONFIG["latent_size"]),
        cfg_weight=cfg_weight,
    )
    elapsed = time.time() - t0
    logger.info("Collection finished in %.1f s", elapsed)

    # --- Collect adaLN stats ---
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
    sigmas = pipeline.get_sigmas(pipeline.sampler, DIAG_CONFIG["num_steps"])
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

    # --- Remove hooks ---
    remove_hooks(hooks)

    # --- Save ---
    save_activation_stats(collector, registry)
    save_adaln_stats(adaln_stats)
    save_config(pairs, registry)

    # --- Pilot validation ---
    if args.pilot:
        expected_calls = 287 * DIAG_CONFIG["num_steps"] * len(pairs)
        actual = collector.call_count
        logger.info("Pilot hook calls: expected=%d  actual=%d", expected_calls, actual)
        if actual != expected_calls:
            logger.error("MISMATCH — some layers may not have fired!")
        else:
            logger.info("Pilot PASSED: all layers fired correctly.")

        logger.info("Layers observed: %d", len(collector.layer_names()))
        logger.info("Steps observed: %d", collector.num_steps())

    logger.info("All data saved to diagnostics/")


if __name__ == "__main__":
    main()
