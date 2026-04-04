#!/usr/bin/env python3
"""Step 4 — Summarize and rank all sweep configs by their metrics.

Reads per-config metric files produced by Step 3, ranks configurations, and
identifies the worst-performing samples for visual inspection.

Usage
-----
# Summarize all configs
python -m src.sweep.summarize

# Show top-10 worst images per config
python -m src.sweep.summarize --top-k-worst 10

# Only summarize specific configs
python -m src.sweep.summarize --configs 0 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .sweep_config import (
    METRICS_ROOT,
    config_tag,
    resolve_configs,
)


def _load_aggregate(metrics_dir: Path) -> dict | None:
    path = metrics_dir / "aggregate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_per_image(metrics_dir: Path) -> list[dict] | None:
    path = metrics_dir / "per_image.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _rank_table(aggregates: list[dict]) -> None:
    """Print a ranked comparison table sorted by mean LPIPS (ascending)."""
    sorted_agg = sorted(aggregates, key=lambda a: a["lpips"]["mean"])

    header = (
        f"{'rank':>4}  {'config_tag':<30}  {'N':>4}  "
        f"{'LPIPS↓':>8}  {'LPIPS_p90':>9}  "
        f"{'CLIP_w4a8':>9}  {'CLIP_Δ':>8}"
    )
    print(header)
    print("-" * len(header))

    for rank, agg in enumerate(sorted_agg, 1):
        print(
            f"{rank:>4}  {agg['tag']:<30}  {agg['num_images']:>4}  "
            f"{agg['lpips']['mean']:>8.4f}  {agg['lpips']['p90']:>9.4f}  "
            f"{agg['clip_w4a8']['mean']:>9.2f}  "
            f"{agg['clip_delta']['mean']:>+8.2f}"
        )
    print()


def _worst_samples(tag: str, per_image: list[dict], top_k: int) -> None:
    """Print the worst (highest LPIPS) samples for a config."""
    sorted_by_lpips = sorted(per_image, key=lambda r: r["lpips"], reverse=True)

    print(f"  Top-{top_k} worst samples (highest LPIPS):")
    print(f"  {'idx':>4}  {'seed':>6}  {'LPIPS':>7}  {'CLIP_Δ':>7}  prompt")
    print(f"  {'-'*70}")
    for r in sorted_by_lpips[:top_k]:
        prompt_short = r["prompt"][:50] + ("..." if len(r["prompt"]) > 50 else "")
        print(
            f"  {r['index']:>4}  {r['seed']:>6}  "
            f"{r['lpips']:>7.4f}  {r['clip_delta']:>+7.2f}  {prompt_short}"
        )
    print()


def _distribution_stats(per_image: list[dict]) -> None:
    """Print distribution breakdown for LPIPS scores."""
    lpips_vals = np.array([r["lpips"] for r in per_image])
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    print("  LPIPS distribution:")
    for t in thresholds:
        pct = 100.0 * np.mean(lpips_vals <= t)
        print(f"    <= {t:.2f}: {pct:5.1f}%  ({int(np.sum(lpips_vals <= t))} images)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: summarize and rank sweep configs by metrics",
    )
    parser.add_argument(
        "--metrics-root", type=str, default=str(METRICS_ROOT),
        help=f"Root for metric outputs (default: {METRICS_ROOT})",
    )
    parser.add_argument(
        "--configs", type=int, nargs="*", default=None,
        help="Indices into SWEEP_MATRIX (default: all discovered)",
    )
    parser.add_argument(
        "--top-k-worst", type=int, default=5,
        help="Number of worst samples to show per config (default: 5)",
    )
    args = parser.parse_args()

    metrics_root = Path(args.metrics_root)

    # Discover configs
    if args.configs is not None:
        tags = [config_tag(c) for c in resolve_configs(args.configs)]
    else:
        if not metrics_root.exists():
            print(f"No metrics directory found at {metrics_root}")
            return
        tags = sorted([
            d.name for d in metrics_root.iterdir()
            if d.is_dir() and (d / "aggregate.json").exists()
        ])

    if not tags:
        print("No metric results found. Run Step 3 (run_metrics) first.")
        return

    # Load aggregates
    aggregates: list[dict] = []
    per_image_data: dict[str, list[dict]] = {}
    for tag in tags:
        agg = _load_aggregate(metrics_root / tag)
        if agg:
            aggregates.append(agg)
            pi = _load_per_image(metrics_root / tag)
            if pi:
                per_image_data[tag] = pi

    if not aggregates:
        print("No aggregate data found.")
        return

    # Rankings
    print(f"\n{'='*70}")
    print(f"  SWEEP SUMMARY — {len(aggregates)} configs")
    print(f"{'='*70}\n")

    _rank_table(aggregates)

    # Per-config details
    for agg in sorted(aggregates, key=lambda a: a["lpips"]["mean"]):
        tag = agg["tag"]
        print(f"── {tag} ──")
        if tag in per_image_data:
            _distribution_stats(per_image_data[tag])
            _worst_samples(tag, per_image_data[tag], args.top_k_worst)

    # Quick recommendation
    best = min(aggregates, key=lambda a: a["lpips"]["mean"])
    best_clip = max(aggregates, key=lambda a: a["clip_w4a8"]["mean"])

    print(f"{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"  Best LPIPS:     {best['tag']}  "
          f"(mean={best['lpips']['mean']:.4f})")
    print(f"  Best CLIPScore: {best_clip['tag']}  "
          f"(mean={best_clip['clip_w4a8']['mean']:.2f})")
    if best["tag"] == best_clip["tag"]:
        print(f"  → Both metrics agree: {best['tag']}")
    else:
        print(f"  → Metrics disagree — inspect images for the top candidates")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
