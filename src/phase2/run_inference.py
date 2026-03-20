#!/usr/bin/env python3
"""Generate images with FP16 baseline or W4A8-quantized SD3 Medium.

Supports single-prompt and batch modes.  Batch mode reads prompts from a
file (one per line) and saves numbered images under a mode-specific
subdirectory, making FP16 ↔ W4A8 comparison straightforward.

Usage
-----
# FP16 baseline — single prompt
python -m src.phase2.run_inference --mode fp16 \\
    --prompt "a cat on a red couch" --output-dir results/

# W4A8 quantized — single prompt
python -m src.phase2.run_inference --mode w4a8 --quantized-dir quantized/ \\
    --prompt "a cat on a red couch" --output-dir results/

# FP16 baseline — evaluation batch
python -m src.phase2.run_inference --mode fp16 \\
    --prompts-file src/settings/evaluation_set.txt --output-dir results/

# W4A8 quantized — evaluation batch
python -m src.phase2.run_inference --mode w4a8 --quantized-dir quantized/ \\
    --prompts-file src/settings/evaluation_set.txt --output-dir results/

# Limit number of prompts for quick testing
python -m src.phase2.run_inference --mode fp16 \\
    --prompts-file src/settings/evaluation_set.txt --num-prompts 5 --output-dir results/

Output layout
-------------
results/
├── fp16/
│   ├── 000.png
│   ├── 001.png
│   └── ...
└── w4a8/
    ├── 000.png
    ├── 001.png
    └── ...
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


def _load_prompts_file(path: Path) -> list[str]:
    """Read prompts from a text file (one prompt per line, skip blanks)."""
    lines = path.read_text().strip().splitlines()
    return [l.strip() for l in lines if l.strip()]


def _generate_batch(
    pipeline,
    prompts: list[str],
    output_dir: Path,
    *,
    seed: int,
    num_steps: int,
    cfg_weight: float,
    latent_size: tuple[int, int],
) -> list[float]:
    """Generate images for a list of prompts and save to *output_dir*.

    Returns a list of per-image generation times (seconds).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    for idx, prompt in enumerate(prompts):
        logger.info(
            "[%d/%d] seed=%d  prompt=%r", idx + 1, len(prompts), seed, prompt[:80],
        )
        t0 = time.time()
        image, _log = pipeline.generate_image(
            prompt,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            latent_size=latent_size,
            seed=seed,
        )
        elapsed = time.time() - t0
        timings.append(elapsed)

        out_path = output_dir / f"{idx:03d}.png"
        image.save(str(out_path))
        logger.info("  saved %s  (%.1f s)", out_path.name, elapsed)

    return timings


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with FP16 or W4A8-quantized SD3 Medium",
    )

    # --- Mode ---
    parser.add_argument(
        "--mode", type=str, required=True, choices=["fp16", "w4a8"],
        help="Model mode: fp16 (baseline) or w4a8 (quantized)",
    )
    parser.add_argument(
        "--quantized-dir", type=str, default=None,
        help="Quantized model directory (required when --mode w4a8)",
    )

    # --- Prompts (mutually exclusive: single vs batch) ---
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Single text prompt")
    prompt_group.add_argument(
        "--prompts-file", type=str,
        help="Path to text file with one prompt per line",
    )

    # --- Generation settings ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-weight", type=float, default=4.0)
    parser.add_argument(
        "--latent-size", type=int, nargs=2, default=[64, 64], metavar=("H", "W"),
        help="Latent size (default: 64 64 → 512×512)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Limit number of prompts from --prompts-file",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Root output directory (default: results/)",
    )

    args = parser.parse_args()

    # --- Validate ---
    if args.mode == "w4a8" and args.quantized_dir is None:
        parser.error("--quantized-dir is required when --mode w4a8")

    # --- Build prompt list ---
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = _load_prompts_file(Path(args.prompts_file))
        if args.num_prompts is not None:
            prompts = prompts[: args.num_prompts]

    logger.info("Mode: %s  |  Prompts: %d  |  Seed: %d  |  Steps: %d  |  CFG: %.1f",
                args.mode, len(prompts), args.seed, args.num_steps, args.cfg_weight)

    # --- Load pipeline ---
    logger.info("Loading DiffusionPipeline ...")
    from .config import DIFFUSIONKIT_SRC, MODEL_VERSION, PIPELINE_KWARGS, QUANTIZE_CONFIG_FILENAME
    sys.path.insert(0, DIFFUSIONKIT_SRC)
    from diffusionkit.mlx import DiffusionPipeline

    model_version = MODEL_VERSION

    if args.mode == "w4a8":
        meta = json.loads(
            (Path(args.quantized_dir) / QUANTIZE_CONFIG_FILENAME).read_text()
        )
        model_version = meta.get("model_version", model_version)

    pipeline = DiffusionPipeline(
        **PIPELINE_KWARGS,
        model_version=model_version,
    )
    logger.info("Pipeline loaded. dtype=%s", pipeline.dtype)

    # --- Load quantized weights (W4A8 mode only) ---
    if args.mode == "w4a8":
        from .quantize import load_quantized_model

        logger.info("Loading quantized model from %s ...", args.quantized_dir)
        load_quantized_model(pipeline, Path(args.quantized_dir))
        logger.info("Quantized model ready.")

    # --- Generate ---
    output_root = Path(args.output_dir) / args.mode
    logger.info("=== GENERATING %d IMAGES → %s ===", len(prompts), output_root)

    t_total = time.time()
    timings = _generate_batch(
        pipeline,
        prompts,
        output_root,
        seed=args.seed,
        num_steps=args.num_steps,
        cfg_weight=args.cfg_weight,
        latent_size=tuple(args.latent_size),
    )
    total_elapsed = time.time() - t_total

    # --- Summary ---
    import numpy as np
    timings_arr = np.array(timings)
    logger.info(
        "\n=== DONE ===\n"
        "  Mode:           %s\n"
        "  Images:         %d\n"
        "  Output:         %s\n"
        "  Total time:     %.1f s\n"
        "  Mean per image: %.1f s\n"
        "  Min / Max:      %.1f / %.1f s",
        args.mode, len(prompts), output_root,
        total_elapsed, timings_arr.mean(),
        timings_arr.min(), timings_arr.max(),
    )

    # Save generation metadata for reproducibility
    run_meta = {
        "mode": args.mode,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "cfg_weight": args.cfg_weight,
        "latent_size": list(args.latent_size),
        "num_prompts": len(prompts),
        "prompts": prompts,
        "timings_s": timings,
        "total_time_s": round(total_elapsed, 2),
    }
    meta_path = output_root / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    logger.info("Run metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
