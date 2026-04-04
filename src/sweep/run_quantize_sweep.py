#!/usr/bin/env python3
"""Step 1 — Quantization sweep: produce a quantized model for each config.

For every entry in the sweep matrix (or a user-specified subset), runs
``src.phase2.run_e2e --skip-collection`` with the appropriate hyperparameters.
Phase 1 diagnostic data is reused across all runs.

Usage
-----
# Quantize all configs in the sweep matrix
python -m src.sweep.run_quantize_sweep

# Quantize only configs 0 and 3
python -m src.sweep.run_quantize_sweep --configs 0 3

# Show the sweep matrix without running
python -m src.sweep.run_quantize_sweep --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .sweep_config import (
    DIAGNOSTICS_DIR,
    QUANTIZED_ROOT,
    config_tag,
    print_matrix,
    resolve_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: quantization sweep over the hyperparameter grid",
    )
    parser.add_argument(
        "--configs", type=int, nargs="*", default=None,
        help="Indices into SWEEP_MATRIX to run (default: all)",
    )
    parser.add_argument(
        "--output-root", type=str, default=str(QUANTIZED_ROOT),
        help=f"Root directory for quantized outputs (default: {QUANTIZED_ROOT})",
    )
    parser.add_argument(
        "--diagnostics-dir", type=str, default=str(DIAGNOSTICS_DIR),
        help=f"Phase 1 diagnostic data directory (default: {DIAGNOSTICS_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the sweep matrix and exit",
    )
    args = parser.parse_args()

    if args.dry_run:
        print_matrix()
        return

    configs = resolve_configs(args.configs)
    print(f"\n{'='*60}")
    print(f"  QUANTIZATION SWEEP — {len(configs)} configs")
    print(f"{'='*60}\n")

    succeeded, failed = [], []

    for i, cfg in enumerate(configs):
        tag = config_tag(cfg)
        out_dir = Path(args.output_root) / tag

        print(f"[{i+1}/{len(configs)}] {tag}")
        if out_dir.exists() and (out_dir / "quantize_config.json").exists():
            print(f"  SKIP — quantized model already exists at {out_dir}")
            succeeded.append(tag)
            continue

        cmd = [
            sys.executable, "-m", "src.phase2.run_e2e",
            "--skip-collection",
            "--output-dir", args.output_root,
            "--diagnostics-dir", args.diagnostics_dir,
            "--qkv-method", cfg["qkv_method"],
            "--alpha", str(cfg["alpha"]),
        ]
        if "group_size" in cfg:
            cmd.extend(["--group-size", str(cfg["group_size"])])
        if "ssc_tau" in cfg:
            cmd.extend(["--ssc-tau", str(cfg["ssc_tau"])])

        print(f"  CMD: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"  OK  ({elapsed:.0f}s)")
            succeeded.append(tag)
        else:
            print(f"  FAIL (exit {result.returncode}, {elapsed:.0f}s)")
            failed.append(tag)

    print(f"\n{'='*60}")
    print(f"  QUANTIZATION SWEEP COMPLETE")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print(f"  Failed configs: {', '.join(failed)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
