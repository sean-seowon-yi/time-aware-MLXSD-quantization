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
        help="Limit number of prompts (default: all 20)",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=None,
        help="Limit number of seeds (default: 8)",
    )
    args = parser.parse_args()

    from .config import DIAG_CONFIG, DIAGNOSTIC_PROMPTS
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
        prompts = DIAGNOSTIC_PROMPTS[:2]
        seeds = [42]
        logger.info("=== PILOT RUN: 2 prompts, 1 seed ===")
    else:
        n_prompts = args.num_prompts or len(DIAGNOSTIC_PROMPTS)
        n_seeds = args.num_seeds or len(DIAG_CONFIG["seed_range"])
        prompts = DIAGNOSTIC_PROMPTS[:n_prompts]
        seeds = DIAG_CONFIG["seed_range"][:n_seeds]

    logger.info("Prompts: %d,  Seeds: %d", len(prompts), len(seeds))

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
    t0 = time.time()
    run_diagnostic_collection(
        pipeline, prompts, seeds, collector,
        num_steps=DIAG_CONFIG["num_steps"],
        latent_size=tuple(DIAG_CONFIG["latent_size"]),
    )
    elapsed = time.time() - t0
    logger.info("Collection finished in %.1f s", elapsed)

    # --- Collect adaLN stats ---
    # adaLN weights were offloaded during the last denoising run; restore them
    # so we can re-cache modulation params for analysis.
    pipeline.mmdit.load_weights(
        pipeline.load_mmdit(only_modulation_dict=True), strict=False,
    )
    conditioning, pooled_conditioning = pipeline.encode_text(
        DIAGNOSTIC_PROMPTS[0], cfg_weight=0.0,
    )
    conditioning = conditioning[:1]
    pooled_conditioning = pooled_conditioning[:1]
    mx.eval(conditioning, pooled_conditioning)
    sigmas = pipeline.get_sigmas(pipeline.sampler, DIAG_CONFIG["num_steps"])
    timesteps = pipeline.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
    pipeline.mmdit.cache_modulation_params(pooled_conditioning, timesteps)
    adaln_stats = collect_adaln_stats(pipeline.mmdit)
    pipeline.mmdit.clear_modulation_params_cache()

    # --- Remove hooks ---
    remove_hooks(hooks)

    # --- Save ---
    save_activation_stats(collector, registry)
    save_adaln_stats(adaln_stats)
    save_config(prompts, seeds, registry)

    # --- Pilot validation ---
    if args.pilot:
        expected_calls = 287 * DIAG_CONFIG["num_steps"] * len(prompts) * len(seeds)
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
