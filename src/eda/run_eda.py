"""
Phase 1 EDA: Orchestration CLI.

Runs the full EDA pipeline in three stages:
  1. profile  — forward-pass profiling (weight stats + activation stats)
  2. analyze  — statistical analysis → CSV tables
  3. visualize — C1–C6 PNG/GIF plots (opt-in; use eda_plots.ipynb instead)

Each stage can be skipped or enabled independently.

Usage:
  # Full pipeline: profile + analyze only (plots via notebook)
  python -m src.eda.run_eda

  # Skip profiling (reuse existing stats files)
  python -m src.eda.run_eda --skip-profile

  # Profiling only
  python -m src.eda.run_eda --skip-analyze

  # Enable file-based plot generation (not recommended; use notebook)
  python -m src.eda.run_eda --skip-profile --skip-analyze --visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_EDA_OUT = _ROOT / "eda_output"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 EDA: Full analysis pipeline for SD3 MMDiT"
    )

    # Stage control
    parser.add_argument("--skip-profile", action="store_true",
                        help="Skip forward-pass profiling (reuse existing stats files)")
    parser.add_argument("--skip-analyze", action="store_true",
                        help="Skip statistical analysis (reuse existing CSV tables)")
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Generate C1–C6 PNG/GIF files (off by default; use eda_plots.ipynb)")

    # Profiling options
    parser.add_argument(
        "--calibration-file", type=str,
        default=str(_EDA_OUT / "coco_cali_data.npz"),
        help="Path to COCO calibration .npz (from sample_cali_data.py)",
    )
    parser.add_argument(
        "--model-version", type=str,
        default="argmaxinc/mlx-stable-diffusion-3-medium",
    )
    parser.add_argument("--low-memory-mode", action="store_true", default=True)
    parser.add_argument("--no-low-memory-mode", action="store_false", dest="low_memory_mode")
    parser.add_argument("--local-ckpt", type=str, default=None)

    # Output paths
    parser.add_argument("--weight-output", type=str,
                        default=str(_EDA_OUT / "weight_stats.npz"))
    parser.add_argument("--act-output", type=str,
                        default=str(_EDA_OUT / "activation_stats_full.npz"))
    parser.add_argument("--tables-dir", type=str,
                        default=str(_EDA_OUT / "tables"))
    parser.add_argument("--plots-dir", type=str,
                        default=str(_EDA_OUT / "plots"))

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Stage 1: Profile
    # ------------------------------------------------------------------
    if not args.skip_profile:
        print("=" * 60)
        print("Stage 1: Activation & Weight Profiling")
        print("=" * 60)
        from .profile_activations import run_profiling
        run_profiling(
            calibration_file=args.calibration_file,
            model_version=args.model_version,
            low_memory_mode=args.low_memory_mode,
            local_ckpt=args.local_ckpt,
            weight_output=args.weight_output,
            act_output=args.act_output,
        )
    else:
        print("Stage 1: Skipped (--skip-profile)")
        for path in (args.weight_output, args.act_output):
            if not Path(path).exists():
                print(f"  WARNING: expected file not found: {path}")

    # ------------------------------------------------------------------
    # Stage 2: Analyze
    # ------------------------------------------------------------------
    if not args.skip_analyze:
        print("\n" + "=" * 60)
        print("Stage 2: Statistical Analysis")
        print("=" * 60)
        from .analyze import run_analysis
        run_analysis(
            act_stats_path=args.act_output,
            weight_stats_path=args.weight_output,
            tables_dir=args.tables_dir,
        )
    else:
        print("Stage 2: Skipped (--skip-analyze)")

    # ------------------------------------------------------------------
    # Stage 3: Visualize (opt-in only)
    # ------------------------------------------------------------------
    if args.visualize:
        print("\n" + "=" * 60)
        print("Stage 3: Visualization (C1–C6)")
        print("=" * 60)
        from .visualize import run_all_plots
        run_all_plots(
            act_stats_path=args.act_output,
            weight_stats_path=args.weight_output,
            tables_dir=args.tables_dir,
            plots_dir=args.plots_dir,
            cali_path=args.calibration_file if Path(args.calibration_file).exists() else None,
        )
    else:
        print("Stage 3: Skipped (use eda_plots.ipynb for interactive plots)")

    print("\nEDA pipeline complete.")
    print(f"Outputs in: {_EDA_OUT}/")


if __name__ == "__main__":
    main()
