"""End-to-end quantization pipeline: Phase 1 -> 2 -> 3 -> 4.

Usage:
    # Full pipeline (all phases)
    python -m src.run_pipeline \\
        --prompts-file src/settings/coco_100_calibration_prompts.txt \\
        --output-dir quantized/

    # Skip Phase 1 collection (reuse existing diagnostics)
    python -m src.run_pipeline --skip-phase1 \\
        --prompts-file src/settings/coco_100_calibration_prompts.txt \\
        --output-dir quantized/

    # Run only up to Phase 3 (no GPTQ)
    python -m src.run_pipeline --stop-after phase3 \\
        --prompts-file src/settings/coco_100_calibration_prompts.txt

    # Skip to Phase 4 only (Phases 1-3 already done)
    python -m src.run_pipeline --start-from phase4 \\
        --quantized-dir quantized/w4a8_l2_a0.50_gs64/ \\
        --prompts-file src/settings/coco_100_calibration_prompts.txt

    # Phases 2-4 only, with Phase 2 hyperparameters (skip Phase 1 collection)
    python -m src.run_pipeline --skip-phase1 \\
        --qkv-method l2 --alpha 0.5 --group-size 32 --act-quant static \\
        --diagnostics-dir diagnostics --output-dir quantized

Phase order:
    Phase 1: Diagnostic collection (activation/weight statistics)
    Phase 2: CSB balancing + RTN W4A8 quantization + SSC weighting
    Phase 3: Polynomial clipping schedule generation
    Phase 4: GPTQ weight quantization with Hessian-weighted error compensation
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHASES = ["phase1", "phase2", "phase3", "phase4"]


def _parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end W4A8 quantization pipeline (Phases 1-4)",
    )

    # --- Global I/O ---
    p.add_argument(
        "--prompts-file", type=str,
        default="src/settings/coco_100_calibration_prompts.txt",
        help="Tab-separated seed<TAB>prompt file",
    )
    p.add_argument(
        "--output-dir", type=str, default="quantized",
        help="Root output directory for Phase 2 checkpoint",
    )
    p.add_argument(
        "--diagnostics-dir", type=str, default="diagnostics",
        help="Directory for Phase 1 diagnostics",
    )
    p.add_argument(
        "--quantized-dir", type=str, default=None,
        help="Explicit Phase 2 checkpoint dir (auto-detected if not set)",
    )

    # --- Phase control ---
    p.add_argument(
        "--start-from", type=str, choices=PHASES, default="phase1",
        help="Start from this phase (skip earlier phases)",
    )
    p.add_argument(
        "--stop-after", type=str, choices=PHASES, default="phase4",
        help="Stop after this phase",
    )
    p.add_argument("--skip-phase1", action="store_true",
                    help="Shorthand for --start-from phase2")

    # --- Phase 1/2 args ---
    p.add_argument("--num-prompts-collection", type=int, default=None,
                    help="Number of prompts for Phase 1 collection")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument(
        "--qkv-method", type=str, default=None,
        choices=["max", "geomean", "l2"],
    )
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--bits", type=int, default=None)
    p.add_argument("--ssc-tau", type=float, default=None)
    p.add_argument("--per-token-rho-threshold", type=float, default=None)
    p.add_argument(
        "--act-quant", type=str, default="dynamic",
        choices=["dynamic", "static"],
        help="A8 mode for Phase 2 (forwarded to run_e2e)",
    )
    p.add_argument(
        "--static-mode", type=str, default="ssc_weighted",
        choices=["ssc_weighted", "global_max"],
        help="Static A8 scale aggregation (when --act-quant static)",
    )
    p.add_argument(
        "--static-granularity", type=str, default="per_tensor",
        choices=["per_tensor", "per_channel"],
        help="Static A8 granularity (when --act-quant static)",
    )

    # --- Phase 3 args ---
    p.add_argument("--max-degree", type=int, default=4)
    p.add_argument("--per-channel-rho-threshold", type=float, default=None)

    # --- Phase 4 args ---
    p.add_argument("--num-prompts-adaround", type=int, default=16,
                    help="Number of prompts for Phase 4 AdaRound calibration")
    p.add_argument("--adaround-iters", type=int, default=None,
                    help="AdaRound iterations per block (default: 1000)")
    p.add_argument("--adaround-lr", type=float, default=None,
                    help="AdaRound Adam learning rate (default: 1e-3)")
    p.add_argument("--adaround-batch-size", type=int, default=None,
                    help="Calibration samples per AdaRound iter (default: 8)")

    # --- Misc ---
    p.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")

    return p.parse_args()


def _run(cmd: list[str], dry_run: bool = False) -> int:
    cmd_str = " \\\n    ".join(cmd)
    logger.info("Running:\n    %s", cmd_str)
    if dry_run:
        return 0
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("Command failed with exit code %d", result.returncode)
    return result.returncode


def _predict_quantized_dir(args) -> str | None:
    """Compute the expected Phase 2 output dir from CLI arguments."""
    try:
        from src.phase2.config import PHASE2_CONFIG, config_tag
    except ImportError:
        return None
    cfg = dict(PHASE2_CONFIG)
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    if args.qkv_method is not None:
        cfg["qkv_method"] = args.qkv_method
    if args.group_size is not None:
        cfg["group_size"] = args.group_size
    if args.bits is not None:
        cfg["bits"] = args.bits
    if args.ssc_tau is not None:
        cfg["ssc_tau"] = args.ssc_tau
    if args.act_quant == "static":
        cfg["static_granularity"] = args.static_granularity
    tag = config_tag(cfg, act_quant=args.act_quant)
    candidate = Path(args.output_dir) / tag
    if (candidate / "quantize_config.json").exists():
        return str(candidate)
    return None


def _find_quantized_dir(output_dir: str) -> str | None:
    """Find the most recently modified Phase 2 checkpoint subdirectory."""
    root = Path(output_dir)
    if (root / "quantize_config.json").exists():
        return str(root)
    candidates = list(root.glob("*/quantize_config.json"))
    if candidates:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(newest.parent)
    return None


def main():
    args = _parse_args()
    py = sys.executable

    if args.skip_phase1:
        args.start_from = "phase2"

    start_idx = PHASES.index(args.start_from)
    stop_idx = PHASES.index(args.stop_after)

    if start_idx > stop_idx:
        raise ValueError(
            f"--start-from {args.start_from} is after --stop-after {args.stop_after}"
        )

    phases_to_run = PHASES[start_idx:stop_idx + 1]
    logger.info("Pipeline phases: %s", " -> ".join(phases_to_run))

    quantized_dir = args.quantized_dir

    # ------------------------------------------------------------------
    # Phase 1 + 2: run_e2e (handles both collection and quantization)
    # ------------------------------------------------------------------
    if "phase1" in phases_to_run or "phase2" in phases_to_run:
        cmd = [
            py, "-m", "src.phase2.run_e2e",
            "--output-dir", args.output_dir,
            "--diagnostics-dir", args.diagnostics_dir,
        ]
        if "phase1" not in phases_to_run:
            cmd.append("--skip-collection")
        if args.num_prompts_collection is not None:
            cmd += ["--num-prompts", str(args.num_prompts_collection)]
        if args.alpha is not None:
            cmd += ["--alpha", str(args.alpha)]
        if args.qkv_method is not None:
            cmd += ["--qkv-method", args.qkv_method]
        if args.group_size is not None:
            cmd += ["--group-size", str(args.group_size)]
        if args.bits is not None:
            cmd += ["--bits", str(args.bits)]
        if args.ssc_tau is not None:
            cmd += ["--ssc-tau", str(args.ssc_tau)]
        if args.per_token_rho_threshold is not None:
            cmd += ["--per-token-rho-threshold", str(args.per_token_rho_threshold)]
        cmd += ["--act-quant", args.act_quant]
        if args.act_quant == "static":
            cmd += [
                "--static-mode", args.static_mode,
                "--static-granularity", args.static_granularity,
            ]

        rc = _run(cmd, args.dry_run)
        if rc != 0:
            return rc

        if quantized_dir is None:
            quantized_dir = _predict_quantized_dir(args)
            if quantized_dir is None:
                quantized_dir = _find_quantized_dir(args.output_dir)
            if quantized_dir is None:
                logger.error(
                    "Could not find quantize_config.json under %s", args.output_dir
                )
                return 1
            logger.info("Auto-detected quantized dir: %s", quantized_dir)

    # ------------------------------------------------------------------
    # Phase 3: generate poly schedule
    # ------------------------------------------------------------------
    if "phase3" in phases_to_run:
        if quantized_dir is None:
            quantized_dir = args.quantized_dir or _find_quantized_dir(args.output_dir)
            if quantized_dir is None:
                logger.error("--quantized-dir required for Phase 3")
                return 1

        cmd = [
            py, "-m", "src.phase3.generate_schedule",
            "--diagnostics-dir", args.diagnostics_dir,
            "--calibration-dir", quantized_dir,
            "--max-degree", str(args.max_degree),
        ]
        if args.per_channel_rho_threshold is not None:
            cmd += ["--per-channel-rho-threshold", str(args.per_channel_rho_threshold)]

        rc = _run(cmd, args.dry_run)
        if rc != 0:
            return rc

    # ------------------------------------------------------------------
    # Phase 4: AdaRound
    # ------------------------------------------------------------------
    if "phase4" in phases_to_run:
        if quantized_dir is None:
            quantized_dir = args.quantized_dir or _find_quantized_dir(args.output_dir)
            if quantized_dir is None:
                logger.error("--quantized-dir required for Phase 4")
                return 1

        adaround_out = str(Path(quantized_dir).parent / (Path(quantized_dir).name + "_adaround"))
        cmd = [
            py, "-m", "src.phase4.run_phase4",
            "--phase2-dir", quantized_dir,
            "--prompts-file", args.prompts_file,
            "--num-prompts", str(args.num_prompts_adaround),
            "--output-dir", adaround_out,
        ]
        if args.adaround_iters is not None:
            cmd += ["--n-iters", str(args.adaround_iters)]
        if args.adaround_lr is not None:
            cmd += ["--lr", str(args.adaround_lr)]
        if args.adaround_batch_size is not None:
            cmd += ["--batch-size", str(args.adaround_batch_size)]

        rc = _run(cmd, args.dry_run)
        if rc != 0:
            return rc
        quantized_dir = adaround_out

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    if quantized_dir:
        logger.info("Output: %s", quantized_dir)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
