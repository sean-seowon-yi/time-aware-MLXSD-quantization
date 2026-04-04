#!/usr/bin/env python3
"""Generate a polynomial clipping schedule from Phase 1/2 artifacts.

Reads per-layer activation trajectories (Phase 1) and CSB balancing
vectors (Phase 2), computes post-CSB absmax trajectories, fits tiered
polynomials, and writes ``poly_schedule.json``.

Usage
-----
# Basic: generate schedule next to the Phase 2 checkpoint
python -m src.phase3.generate_schedule \\
    --diagnostics-dir diagnostics \\
    --calibration-dir quantized/<tag>/ \\
    --output quantized/<tag>/poly_schedule.json

# Hybrid: per-channel poly for layers with ρ > 0.5
python -m src.phase3.generate_schedule \\
    --diagnostics-dir diagnostics \\
    --calibration-dir quantized/<tag>/ \\
    --per-channel-rho-threshold 0.5 \\
    --output quantized/<tag>/poly_schedule.json

# Force all layers to degree 0 (equivalent to static max-absmax)
python -m src.phase3.generate_schedule \\
    --diagnostics-dir diagnostics \\
    --calibration-dir quantized/<tag>/ \\
    --max-degree 0 \\
    --output quantized/<tag>/poly_schedule.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate polynomial clipping schedule from Phase 1/2 data",
    )
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=Path("diagnostics"),
        help="Phase 1 diagnostics directory (default: diagnostics/)",
    )
    parser.add_argument(
        "--calibration-dir", type=Path, required=True,
        help="Phase 2 quantized output with calibration.npz + calibration_meta.json",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for poly_schedule.json (default: <calibration-dir>/poly_schedule.json)",
    )
    parser.add_argument(
        "--max-degree", type=int, default=4,
        help="Maximum polynomial degree (0 = static constant for all layers, default: 4)",
    )
    parser.add_argument(
        "--include-shifts", action="store_true",
        help="(Future) Fit shift polynomials for asymmetric quantisation. "
             "Currently disabled: Phase 1 collects unsigned max|X| only; "
             "signed statistics are needed for meaningful shift polynomials.",
    )
    parser.add_argument(
        "--per-channel-rho-threshold", type=float, default=None,
        help="Enable per-channel poly for layers whose mean Spearman ρ "
             "exceeds this value.  Omit to use per-tensor for all layers.",
    )
    parser.add_argument(
        "--exclude-layers", type=str, nargs="*", default=None,
        help="Layer names to exclude (e.g. context_embedder)",
    )

    args = parser.parse_args()

    output_path = args.output or (args.calibration_dir / "poly_schedule.json")

    from .poly_clipping import generate_schedule_from_diagnostics, print_summary

    logger.info("Generating polynomial schedule ...")
    logger.info("  diagnostics : %s", args.diagnostics_dir)
    logger.info("  calibration : %s", args.calibration_dir)
    logger.info("  max_degree  : %d", args.max_degree)
    logger.info("  shifts      : %s", args.include_shifts)
    if args.per_channel_rho_threshold is not None:
        logger.info("  per-channel ρ threshold : %.3f", args.per_channel_rho_threshold)
    else:
        logger.info("  per-channel : disabled (all per-tensor)")

    schedule = generate_schedule_from_diagnostics(
        diagnostics_dir=args.diagnostics_dir,
        calibration_dir=args.calibration_dir,
        max_degree=args.max_degree,
        include_shifts=args.include_shifts,
        exclude_layers=args.exclude_layers,
        per_channel_rho_threshold=args.per_channel_rho_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(schedule, f, indent=2)
    logger.info("Wrote %s", output_path)

    print_summary(schedule)


if __name__ == "__main__":
    main()
