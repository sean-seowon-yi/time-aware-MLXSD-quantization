#!/usr/bin/env python3
"""Unified CLI for the sweep pipeline.

This script orchestrates the four sweep steps:
1) quantize
2) inference
3) metrics
4) summarize

It reuses existing step scripts so output paths/tags remain consistent.

Usage
-----
# Full pipeline (all prompts/images)
python -m src.sweep.cli

# Quick pass (first 32 prompts/images), then summarize
python -m src.sweep.cli --profile quick

# Run only selected steps
python -m src.sweep.cli --steps quantize inference

# Run subset of configs
python -m src.sweep.cli --configs 0 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sweep_config import (
    DIAGNOSTICS_DIR,
    METRICS_ROOT,
    PROMPTS_FILE,
    QUANTIZED_ROOT,
    RESULTS_ROOT,
    SWEEP_MATRIX,
    SWEEP_SEED,
    config_tag,
    load_prompt_pairs,
)

STEP_ORDER = ("quantize", "inference", "metrics", "summarize")


@dataclass
class RunResult:
    step: str
    returncode: int


def _parse_csv_ints(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_csv_ints_allow_empty(text: str) -> list[int]:
    """Parse comma-separated ints; empty/whitespace means an empty list."""
    if text.strip() == "":
        return []
    return _parse_csv_ints(text)


def _resolve_config_indices(indices: list[int] | None) -> list[int]:
    if indices is None:
        return list(range(len(SWEEP_MATRIX)))
    bad = [i for i in indices if i < 0 or i >= len(SWEEP_MATRIX)]
    if bad:
        raise ValueError(f"invalid config indices: {bad}; valid range is [0, {len(SWEEP_MATRIX)-1}]")
    return indices


def _rank_candidates(metrics_root: Path, config_indices: list[int], expected_num_images: int) -> list[dict]:
    ranked: list[dict] = []
    for idx in config_indices:
        tag = config_tag(SWEEP_MATRIX[idx])
        agg_path = metrics_root / tag / "aggregate.json"
        if not agg_path.exists():
            raise FileNotFoundError(f"missing aggregate metrics for {tag}: {agg_path}")
        agg = json.loads(agg_path.read_text())
        if int(agg["num_images"]) != expected_num_images:
            raise ValueError(
                f"metrics for {tag} have num_images={agg['num_images']} "
                f"but expected {expected_num_images}. "
                "This usually means stage data is incomplete or mismatched."
            )
        ranked.append({
            "index": idx,
            "tag": tag,
            "num_images": agg["num_images"],
            "lpips_mean": float(agg["lpips"]["mean"]),
            "lpips_p90": float(agg["lpips"]["p90"]),
            "clip_w4a8_mean": float(agg["clip_w4a8"]["mean"]),
            "clip_delta_mean": float(agg["clip_delta"]["mean"]),
        })
    # Primary: LPIPS mean (lower is better). Tie-breakers:
    # 1) LPIPS p90 lower; 2) clip_delta closer to zero; 3) CLIPScore higher.
    ranked.sort(
        key=lambda r: (
            r["lpips_mean"],
            r["lpips_p90"],
            abs(r["clip_delta_mean"]),
            -r["clip_w4a8_mean"],
        )
    )
    return ranked


def _save_staged_metadata(metadata: dict, metrics_root: Path) -> Path:
    run_dir = metrics_root / "staged_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = metadata["run_id"]
    path = run_dir / f"{run_id}.json"
    path.write_text(json.dumps(metadata, indent=2))
    latest = run_dir / "latest.json"
    latest.write_text(json.dumps(metadata, indent=2))
    return path


def _run_module(module: str, args: list[str], *, dry_run: bool) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n$ {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def _add_configs(args_list: list[str], configs: list[int] | None) -> list[str]:
    if configs is None:
        return args_list
    return [*args_list, "--configs", *[str(i) for i in configs]]


def _build_quantize_args(args) -> list[str]:
    step_args: list[str] = [
        "--output-root", args.quantized_root,
        "--diagnostics-dir", args.diagnostics_dir,
    ]
    return _add_configs(step_args, args.configs)


def _build_inference_args(args) -> list[str]:
    step_args: list[str] = [
        "--quantized-root", args.quantized_root,
        "--results-root", args.results_root,
        "--prompts-file", args.prompts_file,
    ]
    if args.num_prompts is not None:
        step_args += ["--num-prompts", str(args.num_prompts)]
    return _add_configs(step_args, args.configs)


def _build_metrics_args(args) -> list[str]:
    if args.configs is None:
        step_args: list[str] = ["--all"]
    else:
        step_args = ["--configs", *[str(i) for i in args.configs]]

    step_args += [
        "--results-root", args.results_root,
        "--metrics-root", args.metrics_root,
        "--prompts-file", args.prompts_file,
        "--clip-model", args.clip_model,
        "--lpips-net", args.lpips_net,
    ]
    if args.fp16_dir is not None:
        step_args += ["--fp16-dir", args.fp16_dir]
    if args.num_images is not None:
        step_args += ["--num-images", str(args.num_images)]
    return step_args


def _build_summarize_args(args) -> list[str]:
    step_args: list[str] = [
        "--metrics-root", args.metrics_root,
        "--top-k-worst", str(args.top_k_worst),
    ]
    return _add_configs(step_args, args.configs)


def _generate_shuffled_indices(
    n_total: int,
    stage_sizes: list[int],
    seed: int,
    out_dir: Path,
) -> list[Path]:
    """Generate per-stage eval-index JSON files from a single shuffled permutation.

    Each stage's indices are a superset of the previous stage's, ensuring that
    cached images and metrics carry over.  Returns a list of file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total).tolist()

    paths: list[Path] = []
    for sidx, n in enumerate(stage_sizes):
        indices = sorted(perm[:n])
        path = out_dir / f"stage_{sidx + 1}_indices.json"
        path.write_text(json.dumps(indices))
        paths.append(path)

    perm_path = out_dir / "permutation.json"
    perm_path.write_text(json.dumps(perm))

    return paths


def _run_staged_full_pipeline(args) -> int:
    metrics_root = Path(args.metrics_root)
    selected = _resolve_config_indices(args.configs)

    # Cap stage sizes at evaluation set size and collapse duplicates.
    all_pairs = load_prompt_pairs(Path(args.prompts_file))
    stage_sizes = [min(n, len(all_pairs)) for n in args.stage_prompts]
    stage_topk = list(args.stage_topk)
    while len(stage_sizes) > 1 and stage_sizes[-1] <= stage_sizes[-2]:
        stage_sizes.pop()
        if stage_topk:
            stage_topk.pop()
    if stage_sizes != list(args.stage_prompts) or stage_topk != list(args.stage_topk):
        print(f"\n  NOTE: stages adjusted to evaluation set size ({len(all_pairs)})")
        print(f"  Prompts: {list(args.stage_prompts)} \u2192 {stage_sizes}")
        print(f"  Top-k:   {list(args.stage_topk)} \u2192 {stage_topk}")

    # Generate shuffled per-stage index files (reproducible).
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    indices_dir = metrics_root / "staged_runs" / run_id
    index_files = _generate_shuffled_indices(
        len(all_pairs), stage_sizes, args.sweep_seed, indices_dir,
    )

    print("\n" + "=" * 68)
    print(" STAGED SWEEP PIPELINE (FULL)")
    print("=" * 68)
    print(f" Initial configs:  {selected}")
    print(f" Stage sizes:      {stage_sizes}")
    print(f" Stage top-k:      {stage_topk}")
    print(f" Shuffle seed:     {args.sweep_seed}")
    print(f" Index files:      {indices_dir}")
    print(f" Dry run:          {args.dry_run}")
    print("=" * 68)

    # Step 1: quantize all candidate configs once.
    quantize_args = _add_configs([
        "--output-root", args.quantized_root,
        "--diagnostics-dir", args.diagnostics_dir,
    ], selected)
    print("\n--- [quantize] ---")
    rc = _run_module("src.sweep.run_quantize_sweep", quantize_args, dry_run=args.dry_run)
    if rc != 0:
        return rc

    metadata: dict = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "staged_full",
        "sweep_seed": args.sweep_seed,
        "stage_sizes": stage_sizes,
        "stage_topk": stage_topk,
        "initial_config_indices": selected,
        "initial_tags": [config_tag(SWEEP_MATRIX[i]) for i in selected],
        "stages": [],
    }

    stage_candidates = list(selected)
    for sidx, num in enumerate(stage_sizes):
        idx_file = str(index_files[sidx])
        print(f"\n=== Stage {sidx + 1}/{len(stage_sizes)}  (N={num}, randomized) ===")
        cfg_args = ["--configs", *[str(i) for i in stage_candidates]]

        inf_args = [
            "--quantized-root", args.quantized_root,
            "--results-root", args.results_root,
            "--prompts-file", args.prompts_file,
            "--eval-indices-file", idx_file,
            *cfg_args,
        ]
        print("\n--- [inference] ---")
        rc = _run_module("src.sweep.run_inference_sweep", inf_args, dry_run=args.dry_run)
        if rc != 0:
            return rc

        met_args = [
            "--results-root", args.results_root,
            "--metrics-root", args.metrics_root,
            "--prompts-file", args.prompts_file,
            "--clip-model", args.clip_model,
            "--lpips-net", args.lpips_net,
            "--eval-indices-file", idx_file,
            *cfg_args,
        ]
        if args.fp16_dir is not None:
            met_args += ["--fp16-dir", args.fp16_dir]
        print("\n--- [metrics] ---")
        rc = _run_module("src.sweep.run_metrics", met_args, dry_run=args.dry_run)
        if rc != 0:
            return rc

        if args.dry_run:
            stage_record = {
                "stage_index": sidx + 1,
                "num_images": num,
                "eval_indices_file": idx_file,
                "candidate_indices": list(stage_candidates),
                "candidate_tags": [config_tag(SWEEP_MATRIX[i]) for i in stage_candidates],
                "ranked": [],
                "selected_for_next": list(stage_candidates),
            }
            metadata["stages"].append(stage_record)
            continue

        ranked = _rank_candidates(metrics_root, stage_candidates, expected_num_images=num)
        stage_record = {
            "stage_index": sidx + 1,
            "num_images": num,
            "eval_indices_file": idx_file,
            "candidate_indices": list(stage_candidates),
            "candidate_tags": [config_tag(SWEEP_MATRIX[i]) for i in stage_candidates],
            "ranked": ranked,
        }

        if sidx < len(stage_sizes) - 1:
            keep_k = stage_topk[sidx]
            keep_k = min(keep_k, len(ranked))
            next_indices = [r["index"] for r in ranked[:keep_k]]
            stage_record["selected_for_next"] = next_indices
            stage_candidates = next_indices
            print(f"Promoted to next stage (top-{keep_k}): {[r['tag'] for r in ranked[:keep_k]]}")
        else:
            stage_record["selected_for_next"] = [r["index"] for r in ranked]
            best = ranked[0]
            metadata["final_best"] = best
            print(f"Final best config: {best['tag']} (LPIPS={best['lpips_mean']:.4f})")

        metadata["stages"].append(stage_record)

    # Final summarize on finalists from last stage.
    finalists = stage_candidates
    sum_args = [
        "--metrics-root", args.metrics_root,
        "--top-k-worst", str(args.top_k_worst),
        "--configs", *[str(i) for i in finalists],
    ]
    print("\n--- [summarize] ---")
    rc = _run_module("src.sweep.summarize", sum_args, dry_run=args.dry_run)
    if rc != 0:
        return rc

    if not args.dry_run:
        meta_path = _save_staged_metadata(metadata, metrics_root)
        print(f"\nStaged selection metadata saved to: {meta_path}")
        print(f"Latest metadata alias: {metrics_root / 'staged_runs' / 'latest.json'}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CLI for full sweep pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_ORDER,
        default=list(STEP_ORDER),
        help="Which steps to run (default: all)",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="full",
        help="quick=32 prompt/image staged run, full=all prompts/images",
    )
    parser.add_argument(
        "--configs",
        type=int,
        nargs="*",
        default=None,
        help="Indices into SWEEP_MATRIX; default is all configs",
    )
    parser.add_argument("--quantized-root", type=str, default=str(QUANTIZED_ROOT))
    parser.add_argument("--results-root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--metrics-root", type=str, default=str(METRICS_ROOT))
    parser.add_argument("--diagnostics-dir", type=str, default=str(DIAGNOSTICS_DIR))
    parser.add_argument("--prompts-file", type=str, default=str(PROMPTS_FILE))
    parser.add_argument("--fp16-dir", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--top-k-worst", type=int, default=5)
    parser.add_argument(
        "--stage-prompts",
        type=str,
        default="16,32,64",
        help="Comma-separated prompt counts for staged full run (default: 16,32,64)",
    )
    parser.add_argument(
        "--stage-topk",
        type=str,
        default="4,2",
        help="Comma-separated top-k promotions between stages (default: 4,2)",
    )
    parser.add_argument(
        "--sweep-seed", type=int, default=SWEEP_SEED,
        help=f"Seed for shuffling eval-set indices across stages (default: {SWEEP_SEED})",
    )
    parser.add_argument(
        "--disable-staged-selection",
        action="store_true",
        help="Disable staged full pipeline and run requested steps directly",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.configs == []:
        parser.error("--configs provided without indices. Provide at least one index or omit it.")

    if args.profile == "quick":
        if args.num_prompts is None:
            args.num_prompts = 32
        if args.num_images is None:
            args.num_images = 32

    try:
        args.stage_prompts = _parse_csv_ints(args.stage_prompts)
        args.stage_topk = _parse_csv_ints_allow_empty(args.stage_topk)
    except ValueError as exc:
        parser.error(f"invalid staged arguments: {exc}")
    if len(args.stage_prompts) < 1:
        parser.error("--stage-prompts must include at least one stage")
    if len(args.stage_topk) != max(0, len(args.stage_prompts) - 1):
        parser.error("--stage-topk must have exactly len(stage-prompts)-1 values")
    if any(k < 1 for k in args.stage_topk):
        parser.error("--stage-topk values must be >= 1")
    if args.profile == "full":
        if any(a >= b for a, b in zip(args.stage_prompts, args.stage_prompts[1:])):
            parser.error("for staged full run, --stage-prompts must be strictly increasing")

    step_modules = {
        "quantize": ("src.sweep.run_quantize_sweep", _build_quantize_args),
        "inference": ("src.sweep.run_inference_sweep", _build_inference_args),
        "metrics": ("src.sweep.run_metrics", _build_metrics_args),
        "summarize": ("src.sweep.summarize", _build_summarize_args),
    }

    selected_steps = [s for s in STEP_ORDER if s in set(args.steps)]

    # Default full end-to-end flow now uses staged candidate selection.
    if (
        args.profile == "full"
        and not args.disable_staged_selection
        and selected_steps == list(STEP_ORDER)
    ):
        rc = _run_staged_full_pipeline(args)
        if rc != 0:
            sys.exit(rc)
        return

    print("\n" + "=" * 68)
    print(" SWEEP PIPELINE")
    print("=" * 68)
    print(f" Steps:      {', '.join(selected_steps)}")
    print(f" Profile:    {args.profile}")
    print(f" Configs:    {args.configs if args.configs is not None else 'all'}")
    print(f" Dry run:    {args.dry_run}")
    print("=" * 68)

    results: list[RunResult] = []
    for step in selected_steps:
        module, build_args = step_modules[step]
        print(f"\n--- [{step}] ---")
        rc = _run_module(module, build_args(args), dry_run=args.dry_run)
        results.append(RunResult(step=step, returncode=rc))
        if rc != 0 and not args.continue_on_error:
            break

    failed = [r for r in results if r.returncode != 0]
    print("\n" + "=" * 68)
    print(" PIPELINE RESULT")
    print("=" * 68)
    for r in results:
        status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
        print(f" {r.step:<10} {status}")
    print("=" * 68)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
