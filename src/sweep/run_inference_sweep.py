#!/usr/bin/env python3
"""Step 2 — Inference sweep: generate W4A8 images for each quantized config.

For every quantized model produced by Step 1 (or a user-specified subset),
runs ``src.phase2.run_inference --mode w4a8`` with the evaluation prompt set.
FP16 reference images are assumed to already exist.

Supports staged evaluation via ``--num-prompts`` (e.g. 32 for a quick scan,
then 256 for the full set on promising configs).

Usage
-----
# Generate all 256 images for every quantized config
python -m src.sweep.run_inference_sweep

# Quick scan: first 32 images only
python -m src.sweep.run_inference_sweep --num-prompts 32

# Only run specific configs
python -m src.sweep.run_inference_sweep --configs 0 3 --num-prompts 64

# Show which configs have quantized models ready
python -m src.sweep.run_inference_sweep --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .sweep_config import (
    PROMPTS_FILE,
    QUANTIZED_ROOT,
    RESULTS_ROOT,
    config_tag,
    load_prompt_pairs,
    resolve_configs,
)


def _count_images(directory: Path) -> int:
    """Count .png files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.png")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: W4A8 inference sweep for each quantized config",
    )
    parser.add_argument(
        "--configs", type=int, nargs="*", default=None,
        help="Indices into SWEEP_MATRIX to run (default: all)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Limit number of prompts (default: all in evaluation set)",
    )
    parser.add_argument(
        "--eval-indices-file", type=str, default=None,
        help="JSON file listing prompt indices to generate (overrides --num-prompts)",
    )
    parser.add_argument(
        "--quantized-root", type=str, default=str(QUANTIZED_ROOT),
        help=f"Root for quantized model directories (default: {QUANTIZED_ROOT})",
    )
    parser.add_argument(
        "--results-root", type=str, default=str(RESULTS_ROOT),
        help=f"Root for result images (default: {RESULTS_ROOT})",
    )
    parser.add_argument(
        "--prompts-file", type=str, default=str(PROMPTS_FILE),
        help=f"Evaluation prompt set (default: {PROMPTS_FILE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show status of each config and exit",
    )
    args = parser.parse_args()

    configs = resolve_configs(args.configs)

    if args.dry_run:
        print(f"\n{'tag':<30}  {'quantized?':<12}  {'images'}")
        print("-" * 60)
        for cfg in configs:
            tag = config_tag(cfg)
            qdir = Path(args.quantized_root) / tag
            rdir = Path(args.results_root) / tag
            has_model = (qdir / "quantize_config.json").exists()
            n_imgs = _count_images(rdir)
            print(f"{tag:<30}  {'YES' if has_model else 'NO':<12}  {n_imgs}")
        return

    print(f"\n{'='*60}")
    print(f"  INFERENCE SWEEP — {len(configs)} configs")
    if args.num_prompts:
        print(f"  Staged: first {args.num_prompts} prompts")
    print(f"{'='*60}\n")

    succeeded, skipped, failed = [], [], []

    eval_indices: list[int] | None = None
    if args.eval_indices_file is not None:
        import json
        eval_indices = json.loads(Path(args.eval_indices_file).read_text())
        target = len(eval_indices)
    elif args.num_prompts is not None:
        target = args.num_prompts
    else:
        target = len(load_prompt_pairs(Path(args.prompts_file)))

    for i, cfg in enumerate(configs):
        tag = config_tag(cfg)
        qdir = Path(args.quantized_root) / tag

        print(f"[{i+1}/{len(configs)}] {tag}")

        if not (qdir / "quantize_config.json").exists():
            print(f"  SKIP — no quantized model at {qdir}")
            skipped.append(tag)
            continue

        rdir = Path(args.results_root) / tag
        if eval_indices is not None:
            all_exist = all((rdir / f"{idx:03d}.png").exists() for idx in eval_indices)
            if all_exist:
                print(f"  SKIP — all {len(eval_indices)} eval-index images exist")
                succeeded.append(tag)
                continue
        else:
            existing = _count_images(rdir)
            if existing >= target:
                print(f"  SKIP — already have {existing} images (>= {target})")
                succeeded.append(tag)
                continue

        cmd = [
            sys.executable, "-m", "src.phase2.run_inference",
            "--mode", "w4a8",
            "--quantized-dir", str(qdir),
            "--prompts-file", args.prompts_file,
            "--output-dir", args.results_root,
        ]
        if args.eval_indices_file is not None:
            cmd += ["--eval-indices-file", args.eval_indices_file]
        elif args.num_prompts is not None:
            cmd += ["--num-prompts", str(args.num_prompts)]

        print(f"  CMD: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode == 0:
            n_out = _count_images(rdir)
            print(f"  OK  ({n_out} images, {elapsed:.0f}s)")
            succeeded.append(tag)
        else:
            print(f"  FAIL (exit {result.returncode}, {elapsed:.0f}s)")
            failed.append(tag)

    print(f"\n{'='*60}")
    print(f"  INFERENCE SWEEP COMPLETE")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print(f"  Failed configs: {', '.join(failed)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
