"""CLI entry point for Q-Diffusion quantization.

Usage:
    python -m src.q_diffusion.run_quantize --weight-bits 4 --activation-bits 8
    python -m src.q_diffusion.run_quantize --weight-bits 8 --activation-bits 8 --adaround-iters 1000
"""

from __future__ import annotations

import argparse

from .config import QDiffusionConfig
from .pipeline import run_q_diffusion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q-Diffusion: post-training quantization for SD3 MMDiT"
    )

    # Bit widths
    p.add_argument("--weight-bits", type=int, default=4, choices=[4, 8],
                    help="Weight bit width (default: 4)")
    p.add_argument("--activation-bits", type=int, default=8, choices=[4, 8],
                    help="Activation bit width (default: 8)")

    # AdaRound
    p.add_argument("--adaround-iters", type=int, default=3000,
                    help="AdaRound iterations per block (default: 3000)")
    p.add_argument("--adaround-lr", type=float, default=1e-3,
                    help="AdaRound learning rate (default: 1e-3)")
    p.add_argument("--adaround-beta-start", type=float, default=2.0,
                    help="AdaRound initial beta (default: 2.0)")
    p.add_argument("--adaround-beta-end", type=float, default=20.0,
                    help="AdaRound final beta (default: 20.0)")
    p.add_argument("--adaround-warmup", type=float, default=0.2,
                    help="Fraction of iters before beta annealing starts (default: 0.2)")
    p.add_argument("--adaround-reg-weight", type=float, default=0.01,
                    help="AdaRound regularization weight (default: 0.01)")

    # Block reconstruction
    p.add_argument("--batch-size", type=int, default=16,
                    help="Mini-batch size for AdaRound (default: 16)")
    p.add_argument("--adaround-batch-groups", type=int, default=2,
                    help="Timestep groups sampled per AdaRound iteration (default: 2)")
    p.add_argument("--n-samples", type=int, default=256,
                    help="Number of calibration samples per block (default: 256)")

    # Activation calibration
    p.add_argument("--act-calibration-method", type=str, default="mse_search",
                    choices=["mse_search", "percentile"],
                    help="Activation calibration method (default: mse_search)")
    p.add_argument("--act-percentile", type=float, default=99.99,
                    help="Percentile for activation clipping (default: 99.99)")

    # Options
    p.add_argument("--use-fisher", action="store_true",
                    help="Enable Fisher-information weighting")
    p.add_argument("--skip-final-layer", action="store_true",
                    help="Skip quantizing the FinalLayer")
    p.add_argument("--resume", action="store_true",
                    help="Reuse existing .fp_target_cache and .naive_input_cache if valid; "
                         "skips stages 3 and 4.5 (useful after a crash)")

    # Paths
    p.add_argument("--calibration-file", type=str,
                    default="eda_output/coco_cali_data.npz",
                    help="Path to calibration .npz file")
    p.add_argument("--output-dir", type=str, default="q_diffusion_output",
                    help="Output directory for quantized model")

    # Logging
    p.add_argument("--log-every", type=int, default=100,
                    help="Print loss every N iterations (default: 100)")

    return p.parse_args()


def main():
    args = parse_args()

    config = QDiffusionConfig(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        adaround_iters=args.adaround_iters,
        adaround_lr=args.adaround_lr,
        adaround_beta_start=args.adaround_beta_start,
        adaround_beta_end=args.adaround_beta_end,
        adaround_warmup=args.adaround_warmup,
        adaround_reg_weight=args.adaround_reg_weight,
        batch_size=args.batch_size,
        adaround_batch_groups=args.adaround_batch_groups,
        n_samples=args.n_samples,
        act_calibration_method=args.act_calibration_method,
        act_percentile=args.act_percentile,
        use_fisher=args.use_fisher,
        skip_final_layer=args.skip_final_layer,
        calibration_file=args.calibration_file,
        output_dir=args.output_dir,
        log_every=args.log_every,
        resume=args.resume,
    )

    run_q_diffusion(config)


if __name__ == "__main__":
    main()
