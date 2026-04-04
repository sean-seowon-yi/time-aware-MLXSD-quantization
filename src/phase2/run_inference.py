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

# W4A8 quantized — single prompt (directory with quantize_config.json, e.g. run_e2e output)
python -m src.phase2.run_inference --mode w4a8 --quantized-dir quantized/<tag>/ \\
    --prompt "a cat on a red couch" --output-dir results/

# FP16 baseline — evaluation batch
python -m src.phase2.run_inference --mode fp16 \\
    --prompts-file src/settings/evaluation_set.txt --output-dir results/

# W4A8 quantized — evaluation batch
python -m src.phase2.run_inference --mode w4a8 --quantized-dir quantized/<tag>/ \\
    --prompts-file src/settings/evaluation_set.txt --output-dir results/

# Limit number of prompts for quick testing
python -m src.phase2.run_inference --mode fp16 \\
    --prompts-file src/settings/evaluation_set.txt --num-prompts 5 --output-dir results/

Output layout
-------------
results/
├── fp16/                        ← baseline (always "fp16")
│   ├── 000.png
│   └── ...
├── w4a8_max_a0.50_gs64/         ← auto-named from quantize_config.json
│   ├── 000.png
│   └── ...
└── w4a8_geomean_a0.50_gs64/     ← different config → different folder
    ├── 000.png
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
    eval_indices: set[int] | None = None,
) -> tuple[int, int, list[float]]:
    """Generate images for a list of ``(seed, prompt)`` pairs and save to
    *output_dir*.

    When *eval_indices* is provided, only those prompt indices are generated
    (others are silently skipped).  Images that already exist on disk are also
    skipped, making staged evaluation efficient.

    Returns ``(n_generated, n_skipped, timings)`` where *timings* has one
    entry per generated image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    n_generated = 0
    n_skipped = 0
    total = len(eval_indices) if eval_indices is not None else len(seed_prompt_pairs)
    progress = 0

    for idx, (seed, prompt) in enumerate(seed_prompt_pairs):
        if eval_indices is not None and idx not in eval_indices:
            continue

        out_path = output_dir / f"{idx:03d}.png"
        if out_path.exists():
            n_skipped += 1
            progress += 1
            continue

        progress += 1
        logger.info(
            "[%d/%d] seed=%d  prompt=%r", progress, total, seed, prompt[:80],
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
        n_generated += 1

        image.save(str(out_path))
        logger.info("  saved %s  (%.1f s)", out_path.name, elapsed)

    if n_skipped:
        logger.info("Skipped %d existing images, generated %d new", n_skipped, n_generated)

    return n_generated, n_skipped, timings


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
    parser.add_argument(
        "--eval-indices-file", type=str, default=None,
        help="JSON file listing prompt indices to generate (overrides --num-prompts)",
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
    eval_indices: set[int] | None = None
    if args.prompt:
        pairs: list[tuple[int, str]] = [(args.seed, args.prompt)]
    else:
        pairs = _load_seed_prompt_file(Path(args.prompts_file))
        if args.eval_indices_file is not None:
            eval_indices = set(json.loads(Path(args.eval_indices_file).read_text()))
            logger.info("Using eval-indices file: %d indices", len(eval_indices))
        elif args.num_prompts is not None:
            pairs = pairs[: args.num_prompts]

    n_effective = len(eval_indices) if eval_indices is not None else len(pairs)
    logger.info("Mode: %s  |  Pairs: %d  |  Steps: %d  |  CFG: %.1f",
                args.mode, n_effective, args.num_steps, args.cfg_weight)

    # --- Load pipeline ---
    logger.info("Loading DiffusionPipeline ...")
    from .config import (
        DIFFUSIONKIT_SRC, MODEL_VERSION, PIPELINE_KWARGS,
        QUANTIZE_CONFIG_FILENAME, config_tag_from_meta,
    )
    sys.path.insert(0, DIFFUSIONKIT_SRC)
    from diffusionkit.mlx import DiffusionPipeline

    model_version = MODEL_VERSION
    quant_meta: dict | None = None

    if args.mode == "w4a8":
        meta = json.loads(
            (Path(args.quantized_dir) / QUANTIZE_CONFIG_FILENAME).read_text()
        )
        quant_meta = meta
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
    if args.mode == "w4a8" and quant_meta is not None:
        subdir = config_tag_from_meta(quant_meta)
    else:
        subdir = args.mode
    output_root = Path(args.output_dir) / subdir
    logger.info("=== GENERATING %d IMAGES → %s ===", n_effective, output_root)

    t_total = time.time()
    n_generated, n_skipped, timings = _generate_batch(
        pipeline,
        pairs,
        output_root,
        num_steps=args.num_steps,
        cfg_weight=args.cfg_weight,
        latent_size=tuple(args.latent_size),
        eval_indices=eval_indices,
    )
    total_elapsed = time.time() - t_total

    # --- Summary ---
    import numpy as np

    summary_lines = (
        "\n=== DONE ===\n"
        "  Mode:           %s\n"
        "  Total images:   %d  (generated: %d, reused: %d)\n"
        "  Output:         %s\n"
        "  Total time:     %.1f s"
    )
    summary_args: list = [args.mode, n_effective, n_generated, n_skipped, output_root, total_elapsed]

    if timings:
        timings_arr = np.array(timings)
        summary_lines += (
            "\n  Mean per image: %.1f s\n"
            "  Min / Max:      %.1f / %.1f s"
        )
        summary_args += [timings_arr.mean(), timings_arr.min(), timings_arr.max()]

    logger.info(summary_lines, *summary_args)

    run_meta = {
        "mode": args.mode,
        "config_tag": subdir,
        "num_steps": args.num_steps,
        "cfg_weight": args.cfg_weight,
        "latent_size": list(args.latent_size),
        "num_pairs": n_effective,
        "eval_indices": sorted(eval_indices) if eval_indices is not None else None,
        "timings_s": timings,
        "total_time_s": round(total_elapsed, 2),
    }
    if quant_meta is not None:
        run_meta["quantization"] = {
            "bits": quant_meta.get("bits"),
            "alpha": quant_meta.get("alpha"),
            "qkv_method": quant_meta.get("qkv_method"),
            "group_size": quant_meta.get("group_size"),
            "ssc_tau": quant_meta.get("ssc_tau", 1.0),
            "per_token_rho_threshold": quant_meta.get("per_token_rho_threshold", 0.5),
        }
    meta_path = output_root / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    logger.info("Run metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
