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


def _load_seed_prompt_file(path: Path) -> list[tuple[int, str]]:
    """Load ``(seed, prompt)`` pairs from a tab-separated text file.

    Each non-blank line has the format ``<seed>\\t<prompt>``.
    """
    pairs: list[tuple[int, str]] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        seed_str, prompt = line.split("\t", 1)
        pairs.append((int(seed_str), prompt.strip()))
    return pairs


def _generate_batch(
    pipeline,
    seed_prompt_pairs: list[tuple[int, str]],
    output_dir: Path,
    *,
    num_steps: int,
    cfg_weight: float,
    latent_size: tuple[int, int],
) -> list[float]:
    """Generate images for a list of ``(seed, prompt)`` pairs and save to
    *output_dir*.

    Returns a list of per-image generation times (seconds).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    total = len(seed_prompt_pairs)
    for idx, (seed, prompt) in enumerate(seed_prompt_pairs):
        logger.info(
            "[%d/%d] seed=%d  prompt=%r", idx + 1, total, seed, prompt[:80],
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
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for single --prompt mode (ignored when using --prompts-file)",
    )
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-weight", type=float, default=4.0)
    parser.add_argument(
        "--latent-size", type=int, nargs=2, default=[64, 64], metavar=("H", "W"),
        help="Latent size (default: 64 64 → 512×512)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Limit number of prompt-seed pairs from --prompts-file",
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

    # --- Build prompt-seed pairs ---
    if args.prompt:
        pairs: list[tuple[int, str]] = [(args.seed, args.prompt)]
    else:
        pairs = _load_seed_prompt_file(Path(args.prompts_file))
        if args.num_prompts is not None:
            pairs = pairs[: args.num_prompts]

    logger.info("Mode: %s  |  Pairs: %d  |  Steps: %d  |  CFG: %.1f",
                args.mode, len(pairs), args.num_steps, args.cfg_weight)

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
    logger.info("=== GENERATING %d IMAGES → %s ===", len(pairs), output_root)

    t_total = time.time()
    timings = _generate_batch(
        pipeline,
        pairs,
        output_root,
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
        args.mode, len(pairs), output_root,
        total_elapsed, timings_arr.mean(),
        timings_arr.min(), timings_arr.max(),
    )

    run_meta = {
        "mode": args.mode,
        "num_steps": args.num_steps,
        "cfg_weight": args.cfg_weight,
        "latent_size": list(args.latent_size),
        "num_pairs": len(pairs),
        "seed_prompt_pairs": [[seed, prompt] for seed, prompt in pairs],
        "timings_s": timings,
        "total_time_s": round(total_elapsed, 2),
    }
    meta_path = output_root / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    logger.info("Run metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
