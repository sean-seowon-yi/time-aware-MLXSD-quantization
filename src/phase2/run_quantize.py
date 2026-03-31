#!/usr/bin/env python3
"""Entry point: run the Phase 2 quantization pipeline.

Usage examples
--------------
# Full pipeline: calibrate → balance → quantize → save
python -m src.phase2.run_quantize --output-dir quantized/

# Calibrate only (save calibration for reuse, no model load)
python -m src.phase2.run_quantize --calibrate-only --output-dir quantized/

# Partial: use saved calibration, skip calibration step
python -m src.phase2.run_quantize --from-calibration quantized/ --output-dir quantized/

# Override defaults
python -m src.phase2.run_quantize --output-dir quantized/ --qkv-method geomean --alpha 0.3
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


def _build_config(args) -> dict:
    """Merge CLI overrides into the default config."""
    from .config import MODEL_VERSION, PHASE2_CONFIG

    cfg = {**PHASE2_CONFIG}
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    if args.qkv_method is not None:
        cfg["qkv_method"] = args.qkv_method
    if args.group_size is not None:
        cfg["group_size"] = args.group_size
    if args.bits is not None:
        cfg["bits"] = args.bits
    if args.final_layer_bits is not None:
        cfg["final_layer_bits"] = args.final_layer_bits
    cfg["model_version"] = MODEL_VERSION
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: W4A8 quantization via CSB + SSC",
    )
    parser.add_argument(
        "--output-dir", type=str, default="quantized",
        help="Directory for quantized model output (default: quantized/)",
    )
    parser.add_argument(
        "--diagnostics-dir", type=str, default="diagnostics",
        help="Phase 1 diagnostics directory (default: diagnostics/)",
    )

    # --- Mode flags ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--calibrate-only", action="store_true",
        help="Only compute and save calibration data (no model load)",
    )
    mode_group.add_argument(
        "--from-calibration", type=str, default=None, metavar="DIR",
        help="Load calibration from DIR instead of recomputing",
    )

    # --- Config overrides ---
    parser.add_argument("--alpha", type=float, default=None, help="CSB exponent (default: 0.5)")
    parser.add_argument("--qkv-method", type=str, default=None, choices=["max", "geomean"])
    parser.add_argument("--group-size", type=int, default=None, help="W4 group size (default: 64)")
    parser.add_argument("--bits", type=int, default=None, help="Weight bits (default: 4)")
    parser.add_argument("--final-layer-bits", type=int, default=None, help="Final layer bits (default: 4)")

    args = parser.parse_args()

    from pathlib import Path
    from .calibrate import (
        build_lightweight_registry,
        calibrate_all_layers,
        load_calibration,
        save_calibration,
    )
    from .balance import apply_csb_to_model
    from .quantize import (
        patch_pipeline_for_quantized_inference,
        quantize_model,
        save_quantized_model,
    )

    output_dir = Path(args.output_dir)
    diagnostics_dir = Path(args.diagnostics_dir)
    cfg = _build_config(args)

    logger.info("Phase 2 config: %s", {k: v for k, v in cfg.items() if k != "exclude_layers"})

    # ===================================================================
    # Mode 1: calibrate-only (no model load)
    # ===================================================================
    if args.calibrate_only:
        logger.info("=== CALIBRATE-ONLY MODE ===")
        registry = build_lightweight_registry(diagnostics_dir)
        logger.info("Lightweight registry: %d layers", len(registry))

        t0 = time.time()
        calibration = calibrate_all_layers(registry, diagnostics_dir, cfg)
        logger.info("Calibration finished in %.1f s", time.time() - t0)

        save_calibration(calibration, output_dir)
        logger.info("Done. Calibration saved to %s", output_dir)
        return

    # ===================================================================
    # Load calibration (either from file or compute fresh)
    # ===================================================================
    if args.from_calibration:
        calib_dir = Path(args.from_calibration)
        logger.info("Loading calibration from %s", calib_dir)
        calibration = load_calibration(calib_dir)
        logger.info(
            "Loaded %d balancing vectors (%d b_inv layers)",
            len(calibration["balancing_vectors"]),
            len(calibration["b_inv_layers"]),
        )
    else:
        logger.info("=== CALIBRATION ===")
        registry_light = build_lightweight_registry(diagnostics_dir)
        t0 = time.time()
        calibration = calibrate_all_layers(registry_light, diagnostics_dir, cfg)
        logger.info("Calibration finished in %.1f s", time.time() - t0)
        save_calibration(calibration, output_dir)

    # ===================================================================
    # Load model
    # ===================================================================
    logger.info("=== LOADING MODEL ===")
    from .config import DIFFUSIONKIT_SRC, PIPELINE_KWARGS
    sys.path.insert(0, DIFFUSIONKIT_SRC)
    from diffusionkit.mlx import DiffusionPipeline

    pipeline = DiffusionPipeline(
        **PIPELINE_KWARGS,
        model_version=cfg["model_version"],
    )
    logger.info("Pipeline loaded. dtype=%s", pipeline.dtype)

    # Build full registry (with module references)
    from ..phase1.registry import build_layer_registry
    registry = build_layer_registry(pipeline.mmdit)
    logger.info("Registry: %d linear layers", len(registry))

    # ===================================================================
    # Apply CSB
    # ===================================================================
    logger.info("=== APPLYING CSB ===")
    t0 = time.time()
    b_inv_map = apply_csb_to_model(pipeline.mmdit, registry, calibration)
    logger.info("CSB applied in %.1f s", time.time() - t0)

    # Patch pipeline immediately after CSB so adaLN weights are preserved
    patch_pipeline_for_quantized_inference(pipeline)

    # ===================================================================
    # Quantize
    # ===================================================================
    logger.info("=== QUANTIZING ===")
    t0 = time.time()
    layer_meta = quantize_model(pipeline.mmdit, registry, b_inv_map, cfg)
    logger.info("Quantization finished in %.1f s", time.time() - t0)

    # ===================================================================
    # Save
    # ===================================================================
    logger.info("=== SAVING ===")
    save_quantized_model(
        pipeline.mmdit,
        output_dir,
        cfg,
        layer_meta,
        calibration["b_inv_layers"],
    )
    logger.info("All outputs saved to %s", output_dir)

    # --- Summary ---
    n_quantized = len(layer_meta)
    n_online = len(b_inv_map)
    logger.info(
        "\n=== SUMMARY ===\n"
        "  Quantized layers:  %d\n"
        "  Online b_inv:      %d\n"
        "  Group size:        %d\n"
        "  Weight bits:       %d\n"
        "  QKV method:        %s\n"
        "  Alpha:             %.2f\n"
        "  Output:            %s",
        n_quantized, n_online, cfg["group_size"], cfg["bits"],
        cfg["qkv_method"], cfg["alpha"], output_dir,
    )


if __name__ == "__main__":
    main()
