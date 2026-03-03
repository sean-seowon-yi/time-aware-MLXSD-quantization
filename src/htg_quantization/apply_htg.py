"""
Phase 3 top-level CLI: run the full HTG quantization pipeline for SD3 / MMDiT.

Stages
------
1. Profile input activations   (profile_input_activations.py)
   Runs calibration data through the model and records per-channel (min, max)
   for fc1, qkv, and o_proj inputs.

2. Compute HTG parameters      (compute_htg_params.py)
   Derives z_t, z_g, group assignments, and scaling vector s for every
   target layer from the profiled statistics.

3. Re-parameterize & quantize  (htg_reparameterize.py)
   Rescales linear layer weights by s, stores per-group z_g corrections for
   AdaLN absorption at inference, and optionally applies MLX weight quantization.

Usage
-----
    # Full pipeline from calibration data
    python -m src.htg_quantization.apply_htg \\
        --calibration-file DiT_cali_data.npz \\
        --output-dir htg_output/

    # Skip profiling if htg_input_activation_stats.npz already exists
    python -m src.htg_quantization.apply_htg \\
        --calibration-file DiT_cali_data.npz \\
        --output-dir htg_output/ \\
        --skip-profile

    # Skip profiling AND param computation if htg_params.npz already exists
    python -m src.htg_quantization.apply_htg \\
        --calibration-file DiT_cali_data.npz \\
        --output-dir htg_output/ \\
        --skip-profile --skip-compute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .htg_config import (
    MODEL_VERSION,
    NUM_GROUPS,
    EMA_ALPHA,
    WEIGHT_BITS,
    QUANTIZATION_GROUP_SIZE,
    DEFAULT_CALIBRATION_FILE,
    DEFAULT_INPUT_STATS_FILE,
    DEFAULT_HTG_PARAMS_FILE,
    DEFAULT_OUTPUT_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3: Full HTG quantization pipeline for SD3 / MMDiT "
            "(arXiv:2503.06930). Runs profiling → parameter computation → "
            "re-parameterization in sequence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input / output ---
    parser.add_argument(
        "--calibration-file", type=str, default=DEFAULT_CALIBRATION_FILE,
        help="Path to Phase 1 calibration .npz (DiT_cali_data.npz)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for all output files",
    )

    # --- Model ---
    parser.add_argument(
        "--model-version", type=str, default=MODEL_VERSION,
        help="DiffusionKit model identifier",
    )
    parser.add_argument(
        "--local-ckpt", type=str, default=None,
        help="Optional local checkpoint path",
    )
    parser.add_argument(
        "--low-memory-mode", action="store_true", default=True,
        help="Enable DiffusionKit low_memory_mode (recommended for Apple Silicon)",
    )
    parser.add_argument(
        "--no-low-memory-mode", action="store_false", dest="low_memory_mode",
    )

    # --- Stage control ---
    parser.add_argument(
        "--skip-profile", action="store_true", default=False,
        help=(
            "Skip Stage 1 (input activation profiling). "
            "Requires --input-stats or the default file to already exist."
        ),
    )
    parser.add_argument(
        "--skip-compute", action="store_true", default=False,
        help=(
            "Skip Stage 2 (HTG parameter computation). "
            "Implies --skip-profile. "
            "Requires --htg-params or the default file to already exist."
        ),
    )

    # --- Stage 1 options ---
    parser.add_argument(
        "--num-samples", type=int, default=512,
        help="[Stage 1] Number of calibration points to profile",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="[Stage 1] Random seed for point selection",
    )
    parser.add_argument(
        "--input-stats", type=str, default=None,
        help=(
            "[Stage 1 output / Stage 2 input] Path for input activation stats .npz. "
            f"Default: <output_dir>/{Path(DEFAULT_INPUT_STATS_FILE).name}"
        ),
    )

    # --- Stage 2 options ---
    parser.add_argument(
        "--num-groups", type=int, default=NUM_GROUPS,
        help="[Stage 2] Target number of timestep groups G. None → T // 10",
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=EMA_ALPHA,
        help="[Stage 2] EMA coefficient α for scaling accumulation",
    )
    parser.add_argument(
        "--htg-params", type=str, default=None,
        help=(
            "[Stage 2 output / Stage 3 input] Path for HTG parameters .npz. "
            f"Default: <output_dir>/{Path(DEFAULT_HTG_PARAMS_FILE).name}"
        ),
    )

    # --- Stage 3 options ---
    parser.add_argument(
        "--weight-bits", type=int, default=WEIGHT_BITS, choices=[4, 8],
        help="[Stage 3] Weight quantization bit-width",
    )
    parser.add_argument(
        "--group-size", type=int, default=QUANTIZATION_GROUP_SIZE,
        help="[Stage 3] MLX block-wise quantization group size",
    )
    parser.add_argument(
        "--no-quantize", action="store_true", default=False,
        help="[Stage 3] Skip weight quantization (rescale only, no integer conversion)",
    )

    args = parser.parse_args()

    # Resolve output dir and default file paths
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_stats_path = args.input_stats or str(out_dir / Path(DEFAULT_INPUT_STATS_FILE).name)
    htg_params_path = args.htg_params or str(out_dir / Path(DEFAULT_HTG_PARAMS_FILE).name)

    skip_profile = args.skip_profile or args.skip_compute

    # -----------------------------------------------------------------------
    # Stage 1: Profile input activations
    # -----------------------------------------------------------------------
    if not skip_profile:
        print("=" * 60)
        print("Stage 1: Profiling input activations")
        print("=" * 60)
        from .profile_input_activations import run_input_profiling, save_input_stats

        tracer, unique_ts = run_input_profiling(
            calibration_file=args.calibration_file,
            model_version=args.model_version,
            num_samples=args.num_samples,
            seed=args.seed,
            low_memory_mode=args.low_memory_mode,
            local_ckpt=args.local_ckpt,
        )
        save_input_stats(tracer, unique_ts, input_stats_path)
    else:
        if not Path(input_stats_path).exists():
            print(
                f"ERROR: --skip-profile requested but {input_stats_path} does not exist.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Skipping Stage 1. Using existing stats: {input_stats_path}")

    # -----------------------------------------------------------------------
    # Stage 2: Compute HTG parameters
    # -----------------------------------------------------------------------
    if not args.skip_compute:
        print()
        print("=" * 60)
        print("Stage 2: Computing HTG parameters")
        print("=" * 60)
        from .compute_htg_params import compute_htg_params
        import numpy as np

        flat = compute_htg_params(
            input_stats_path=input_stats_path,
            model_version=args.model_version,
            num_groups=args.num_groups,
            ema_alpha=args.ema_alpha,
            local_ckpt=args.local_ckpt,
        )
        np.savez_compressed(htg_params_path, **flat)
        print(f"Saved HTG parameters to {htg_params_path}")
    else:
        if not Path(htg_params_path).exists():
            print(
                f"ERROR: --skip-compute requested but {htg_params_path} does not exist.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Skipping Stage 2. Using existing params: {htg_params_path}")

    # -----------------------------------------------------------------------
    # Stage 3: Re-parameterize and quantize
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Stage 3: Re-parameterizing and quantizing")
    print("=" * 60)
    from .htg_reparameterize import reparameterize_and_quantize

    reparameterize_and_quantize(
        htg_params_path=htg_params_path,
        input_stats_path=input_stats_path if Path(input_stats_path).exists() else None,
        model_version=args.model_version,
        weight_bits=args.weight_bits,
        group_size=args.group_size,
        output_dir=args.output_dir,
        local_ckpt=args.local_ckpt,
        quantize_weights=not args.no_quantize,
    )

    print()
    print("=" * 60)
    print("HTG quantization pipeline complete.")
    print(f"Outputs written to: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
