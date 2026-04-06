"""Phase 4 CLI: collect calibration data → AdaRound weight optimisation → save.

Usage
-----
# Full run (collection + optimisation)
python -m src.phase4.run_phase4 \\
    --phase2-dir quantized/<tag>/ \\
    --prompts-file src/settings/coco_100_calibration_prompts.txt \\
    --output-dir quantized/<tag>_adaround/

# Skip collection (reuse saved calibration data)
python -m src.phase4.run_phase4 \\
    --phase2-dir quantized/<tag>/ \\
    --skip-collection \\
    --calibration-dir phase4_calibration/ \\
    --output-dir quantized/<tag>_adaround/

# Quick test (4 prompts, 200 iters)
python -m src.phase4.run_phase4 \\
    --phase2-dir quantized/<tag>/ \\
    --prompts-file src/settings/coco_100_calibration_prompts.txt \\
    --num-prompts 4 --n-iters 200 \\
    --output-dir quantized/<tag>_adaround_test/

Prerequisites
-------------
  Phase 2 checkpoint (run_quantize.py): quantize_config.json + calibration.npz
  Phase 3 (optional): poly_schedule.json for activation quantisation at inference
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_pairs(prompts_file: Path, n: int | None) -> list[tuple[int, str]]:
    pairs = []
    for line in prompts_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        seed_str, prompt = line.split("\t", 1)
        pairs.append((int(seed_str), prompt.strip()))
    if n is not None:
        pairs = pairs[:n]
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 4: AdaRound weight optimisation",
    )

    # --- I/O ---
    p.add_argument(
        "--phase2-dir", type=Path, required=True,
        help="Phase 2 checkpoint dir (contains quantize_config.json + calibration.npz)",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for the AdaRound checkpoint",
    )
    p.add_argument(
        "--calibration-dir", type=Path, default=None,
        help="Directory for per-layer input NPZs (default: <output-dir>/calibration/)",
    )

    # --- Collection ---
    p.add_argument(
        "--skip-collection", action="store_true",
        help="Skip data collection; reuse NPZs in --calibration-dir",
    )
    p.add_argument(
        "--prompts-file", type=Path, default=None,
        help="Tab-separated seed<TAB>prompt file",
    )
    p.add_argument("--num-prompts", type=int, default=None,
                   help="Limit calibration prompts (default: all)")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Denoising steps per prompt (default: 30)")
    p.add_argument("--cfg-weight", type=float, default=None,
                   help="CFG scale (default: 4.0)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Tokens to capture per forward pass (default: 64)")

    # --- Optimisation ---
    p.add_argument("--n-iters", type=int, default=None,
                   help="AdaRound iterations per layer (default: 1000)")
    p.add_argument("--lr", type=float, default=None,
                   help="Adam learning rate (default: 1e-3)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Calibration samples per iteration (default: 8)")
    p.add_argument("--blocks", type=str, default=None,
                   help="Comma-separated block indices to optimize (default: all)")

    args = p.parse_args()

    if not args.skip_collection and args.prompts_file is None:
        p.error("--prompts-file is required unless --skip-collection is set")

    from .config import PHASE4_CONFIG
    from ..phase2.calibrate import load_calibration
    from ..phase2.config import DIFFUSIONKIT_SRC, PIPELINE_KWARGS, QUANTIZE_CONFIG_FILENAME
    from ..phase1.registry import build_layer_registry

    cfg = {**PHASE4_CONFIG}
    if args.num_steps is not None:
        cfg["n_steps"] = args.num_steps
    if args.cfg_weight is not None:
        cfg["cfg_weight"] = args.cfg_weight
    if args.max_tokens is not None:
        cfg["max_tokens_per_sample"] = args.max_tokens
    if args.n_iters is not None:
        cfg["n_iters"] = args.n_iters
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    calib_dir = args.calibration_dir or (args.output_dir / "calibration")
    phase2_dir = args.phase2_dir

    # Load phase 2 metadata + calibration
    meta_path = phase2_dir / QUANTIZE_CONFIG_FILENAME
    if not meta_path.exists():
        p.error(f"quantize_config.json not found in {phase2_dir}")
    phase2_meta = json.loads(meta_path.read_text())
    logger.info("Phase 2 meta: W%dA%d, group_size=%d, alpha=%.2f",
                phase2_meta["bits"], phase2_meta.get("a_bits", 8),
                phase2_meta["group_size"], phase2_meta["alpha"])

    calibration = load_calibration(phase2_dir)
    logger.info(
        "Calibration loaded: %d layers, %d b_inv",
        len(calibration["balancing_vectors"]),
        len(calibration["b_inv_layers"]),
    )

    # ===================================================================
    # Load pipeline + apply CSB
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Loading SD3 Medium pipeline")
    logger.info("=" * 60)

    sys.path.insert(0, DIFFUSIONKIT_SRC)
    from diffusionkit.mlx import DiffusionPipeline

    pipeline = DiffusionPipeline(
        **PIPELINE_KWARGS,
        model_version=phase2_meta.get("model_version"),
    )
    logger.info("Pipeline loaded. dtype=%s", pipeline.dtype)

    registry = build_layer_registry(pipeline.mmdit)
    logger.info("Registry: %d linear layers", len(registry))

    from ..phase2.balance import apply_csb_to_model
    from ..phase2.quantize import patch_pipeline_for_quantized_inference

    logger.info("Applying CSB...")
    apply_csb_to_model(pipeline.mmdit, registry, calibration)
    patch_pipeline_for_quantized_inference(pipeline)
    logger.info("CSB applied and pipeline patched.")

    # ===================================================================
    # Data collection
    # ===================================================================
    if args.skip_collection:
        logger.info("=" * 60)
        logger.info("SKIPPING collection — using %s", calib_dir)
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("Collecting layer inputs (%d prompts × %d steps)",
                    args.num_prompts or len(_load_pairs(args.prompts_file, None)),
                    cfg["n_steps"])
        logger.info("=" * 60)

        from .collect import collect_block_io

        pairs = _load_pairs(args.prompts_file, args.num_prompts)
        logger.info("Calibration pairs: %d", len(pairs))

        block_subset = None
        if args.blocks is not None:
            block_subset = set(int(b) for b in args.blocks.split(","))

        t0 = time.time()
        collect_block_io(pipeline, pairs, calib_dir, cfg, block_subset=block_subset)
        logger.info("Collection done in %.1f s", time.time() - t0)

    # ===================================================================
    # AdaRound optimisation
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Running block-wise AdaRound (%d iters/block, lr=%.1e, batch=%d)",
                cfg["n_iters"], cfg["lr"], cfg["batch_size"])
    logger.info("=" * 60)

    from .optimize import optimize_all_blocks

    # Reload fresh FP16 + CSB for optimisation (collection pipeline is still clean)
    pipeline2 = DiffusionPipeline(
        **PIPELINE_KWARGS,
        model_version=phase2_meta.get("model_version"),
    )
    registry2 = build_layer_registry(pipeline2.mmdit)
    apply_csb_to_model(pipeline2.mmdit, registry2, calibration)
    patch_pipeline_for_quantized_inference(pipeline2)

    block_subset = None
    if args.blocks is not None:
        block_subset = set(int(b) for b in args.blocks.split(","))
        logger.info("Optimising blocks: %s", sorted(block_subset))

    t0 = time.time()
    optimize_all_blocks(
        pipeline2, registry2, calibration, calib_dir, phase2_meta, cfg, args.output_dir,
        block_subset=block_subset,
    )
    elapsed = time.time() - t0

    logger.info(
        "\n" + "=" * 60 + "\n"
        "  PHASE 4 COMPLETE\n"
        "=" * 60 + "\n"
        "  Optimisation time:  %.1f s (%.1f min)\n"
        "  Iters per layer:    %d\n"
        "  Calibration data:   %s\n"
        "  Output:             %s",
        elapsed, elapsed / 60,
        cfg["n_iters"], calib_dir, args.output_dir,
    )


if __name__ == "__main__":
    main()
