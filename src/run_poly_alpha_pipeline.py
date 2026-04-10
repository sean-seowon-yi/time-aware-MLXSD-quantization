"""RTN W4A8 + polynomial clipping + alpha search.

Chains:

1. **Calibration** (Phase 1) — optional via ``--skip-calibration`` (``run_e2e --skip-collection``; reuses ``--diagnostics-dir``). Prompt list is whatever ``run_e2e`` uses; ``--prompts-file`` defaults alpha search only.
2. **CSB / SSC + RTN W4A8** (Phase 2) — ``src.phase2.run_e2e``.
3. **Polynomial clipping schedule** (Phase 3) — ``src.phase3.generate_schedule``.
4. **Alpha search** (Phase 4.1) — ``src.phase4_1.alpha_search`` (merges ``alpha_multiplier`` into ``poly_schedule.json``).
5. **Finalize** — tags ``quantize_config.json`` with ``poly_alpha_w4a8_ready`` for downstream tools.

Inference: ``load_quantized_model_poly(pipeline, Path(quantized_dir))``.

All subprocess steps use the **repository root** as working directory, so relative
paths such as ``quantized/``, ``diagnostics/``, and ``src/settings/…`` resolve
consistently even if you start Python from another directory.

``quantize_config.json`` gains ``poly_alpha_w4a8_ready`` only when this run
requested merging alpha multipliers **and** ``poly_schedule.json`` on disk
contains ``alpha_multiplier`` entries (so the manifest matches deploy).

Usage::

    python -m src.run_poly_alpha_pipeline \\
        --prompts-file src/settings/coco_100_calibration_prompts.txt \\
        --output-dir quantized \\
        --diagnostics-dir diagnostics

    # Reuse diagnostics; only run quantize → poly → alpha → finalize
    python -m src.run_poly_alpha_pipeline --skip-calibration \\
        --output-dir quantized --diagnostics-dir diagnostics

    # Resume from existing checkpoint (poly + alpha + finalize only)
    python -m src.run_poly_alpha_pipeline --start-from poly \\
        --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \\
        --diagnostics-dir diagnostics
"""

from __future__ import annotations

import argparse
import json
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

_START_CHOICES = ("e2e", "poly", "alpha")

# Subprocess cwd so relative paths (diagnostics/, quantized/, src/settings/…) work
# even if the user invoked Python from another directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_under_repo(path_str: str) -> Path:
    """Resolve *path_str* the same way subprocess steps do (cwd = repo root)."""
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _run(cmd: list[str], dry_run: bool = False) -> int:
    cmd_str = " \\\n    ".join(cmd)
    logger.info("Running:\n    %s", cmd_str)
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        logger.error("Command failed with exit code %d", result.returncode)
    return result.returncode


def _phase2_tag_paths(args: argparse.Namespace) -> tuple[dict, str]:
    """Return ``(cfg_dict, tag)`` matching ``run_e2e`` / ``config_tag``."""
    from src.phase2.config import PHASE2_CONFIG, config_tag

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
    cfg["static_granularity"] = args.static_granularity
    tag = config_tag(cfg)
    return cfg, tag


def _expected_quantized_dir(args: argparse.Namespace) -> str:
    """Directory ``run_e2e`` writes to for these CLI args (may not exist yet)."""
    _, tag = _phase2_tag_paths(args)
    return str(_resolve_under_repo(args.output_dir) / tag)


def _predict_quantized_dir(args: argparse.Namespace) -> str | None:
    try:
        candidate = Path(_expected_quantized_dir(args))
    except ImportError:
        return None
    if (candidate / "quantize_config.json").is_file():
        return str(candidate)
    return None


def _find_quantized_dir(output_dir: str) -> str | None:
    root = _resolve_under_repo(output_dir)
    if (root / "quantize_config.json").is_file():
        return str(root)
    candidates = list(root.glob("*/quantize_config.json"))
    if candidates:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(newest.parent.resolve())
    return None


def _detect_quantized_dir_after_e2e(args: argparse.Namespace) -> str | None:
    """Directory written by ``run_e2e`` for the same hyperparameters as *args*."""
    predicted = _predict_quantized_dir(args)
    if predicted is not None:
        return str(Path(predicted).resolve())
    return _find_quantized_dir(args.output_dir)


def _checkpoint_ok_for_poly_schedule(qdir: Path) -> tuple[bool, str]:
    """``generate_schedule`` needs Phase 2 calibration artifacts next to weights."""
    from src.phase2.config import QUANTIZE_CONFIG_FILENAME

    missing = [
        name
        for name, rel in (
            ("quantize_config.json", QUANTIZE_CONFIG_FILENAME),
            ("calibration.npz", "calibration.npz"),
            ("calibration_meta.json", "calibration_meta.json"),
        )
        if not (qdir / rel).is_file()
    ]
    if missing:
        return False, f"checkpoint {qdir} missing: {', '.join(missing)}"
    return True, ""


def _poly_schedule_available_for_alpha(
    quantized_dir: str,
    poly_schedule_arg: str | None,
) -> tuple[bool, str]:
    """``--start-from alpha`` skips ``generate_schedule``; need an existing schedule file."""
    from src.phase3.poly_clipping import POLY_SCHEDULE_FILENAME

    qdir = Path(quantized_dir).resolve()
    if poly_schedule_arg:
        sched_path = _resolve_under_repo(poly_schedule_arg)
        if not sched_path.is_file():
            return False, f"--poly-schedule not found: {sched_path}"
        return True, ""
    default_poly = qdir / POLY_SCHEDULE_FILENAME
    if not default_poly.is_file():
        return False, (
            f"No {POLY_SCHEDULE_FILENAME} under {qdir}. "
            f"Run --start-from poly first, or pass --poly-schedule."
        )
    return True, ""


def _poly_schedule_has_alpha_multipliers(poly_path: Path) -> bool:
    if not poly_path.is_file():
        return False
    try:
        sched = json.loads(poly_path.read_text())
    except json.JSONDecodeError:
        return False
    layers = sched.get("layers") or {}
    return any(
        isinstance(v, dict) and "alpha_multiplier" in v for v in layers.values()
    )


def _finalize_quantize_config(
    quantized_dir: Path,
    *,
    intended_merge: bool,
) -> None:
    """Update ``quantize_config.json`` with poly+alpha pipeline metadata.

    ``poly_alpha_w4a8_ready`` is True only when multipliers were merged into
    ``poly_schedule.json`` on disk (so ``load_quantized_model_poly`` matches search).
    """
    from src.phase2.config import QUANTIZE_CONFIG_FILENAME
    from src.phase3.poly_clipping import POLY_SCHEDULE_FILENAME

    qdir = Path(quantized_dir).resolve()
    path = qdir / QUANTIZE_CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")

    meta = json.loads(path.read_text())
    poly_path = qdir / POLY_SCHEDULE_FILENAME
    has_multipliers = _poly_schedule_has_alpha_multipliers(poly_path)
    ready = intended_merge and has_multipliers
    if intended_merge and not has_multipliers:
        logger.warning(
            "Alpha merge was requested but %s has no per-layer alpha_multiplier entries — "
            "poly_alpha_w4a8_ready=False (check empty poly schedule or merge skip inside alpha_search)",
            poly_path,
        )

    meta["poly_alpha_w4a8_ready"] = ready
    meta["poly_alpha_pipeline"] = {
        "module": "src.run_poly_alpha_pipeline",
        "inference": "src.phase3.quantize_poly.load_quantized_model_poly",
        "alpha_multipliers_merged": intended_merge,
        "deployable_poly_alpha": ready,
    }
    path.write_text(json.dumps(meta, indent=2))
    logger.info(
        "Updated %s (poly_alpha_w4a8_ready=%s, alpha_multipliers_merged=%s)",
        path,
        ready,
        intended_merge,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline: calibration (optional) → RTN W4A8 → poly schedule "
        "→ alpha search → finalize for σ-aware W4A8",
    )

    p.add_argument(
        "--prompts-file",
        type=str,
        default="src/settings/coco_100_calibration_prompts.txt",
        help="Tab-separated seed<TAB>prompt; default for alpha search. "
        "Phase 1 collection uses run_e2e's built-in calibration list unless you "
        "change that script.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="quantized",
        help="Root directory for Phase 2 checkpoint subdirectory",
    )
    p.add_argument(
        "--diagnostics-dir",
        type=str,
        default="diagnostics",
        help="Phase 1 diagnostics directory",
    )
    p.add_argument(
        "--quantized-dir",
        type=str,
        default=None,
        help="Explicit Phase 2 checkpoint (required for --start-from poly or alpha)",
    )

    p.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Pass --skip-collection to run_e2e (reuse diagnostics)",
    )
    p.add_argument(
        "--start-from",
        type=str,
        choices=_START_CHOICES,
        default="e2e",
        help="'e2e': full from run_e2e; 'poly': schedule+alpha+finalize only; "
        "'alpha': alpha+finalize only",
    )

    # run_e2e Phase 1 collection
    p.add_argument(
        "--num-prompts-collection",
        type=int,
        default=None,
        help="Limit calibration prompts for collection (run_e2e --num-prompts)",
    )
    p.add_argument(
        "--collection-num-steps",
        type=int,
        default=None,
        help="Denoising steps during Phase 1 collection (run_e2e --num-steps)",
    )
    p.add_argument(
        "--collection-cfg-weight",
        type=float,
        default=None,
        help="CFG during Phase 1 collection (run_e2e --cfg-weight)",
    )

    # Phase 2 quantization
    p.add_argument("--alpha", type=float, default=None, help="CSB exponent")
    p.add_argument(
        "--qkv-method",
        type=str,
        default=None,
        choices=["max", "geomean", "l2"],
    )
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--bits", type=int, default=None)
    p.add_argument("--final-layer-bits", type=int, default=None)
    p.add_argument("--ssc-tau", type=float, default=None)
    p.add_argument(
        "--static-mode",
        type=str,
        default="ssc_weighted",
        choices=["ssc_weighted", "global_max"],
    )
    p.add_argument(
        "--static-granularity",
        type=str,
        default="per_tensor",
        choices=["per_tensor", "per_channel"],
    )

    # Phase 3
    p.add_argument("--max-degree", type=int, default=4)
    p.add_argument("--per-channel-rho-threshold", type=float, default=None)
    p.add_argument(
        "--exclude-layers",
        type=str,
        nargs="*",
        default=None,
        help="Forwarded to generate_schedule (space-separated layer keys)",
    )
    p.add_argument(
        "--include-shifts",
        action="store_true",
        help="Forwarded to generate_schedule (future / no-op for unsigned Phase 1)",
    )

    # Phase 4.1 alpha search
    p.add_argument(
        "--alpha-prompts-file",
        type=str,
        default=None,
        help="Prompts for alpha search (default: same as --prompts-file)",
    )
    p.add_argument("--alpha-num-prompts", type=int, default=16)
    p.add_argument("--alpha-num-steps", type=int, default=30)
    p.add_argument("--alpha-cfg-weight", type=float, default=4.0)
    p.add_argument("--alpha-latent-size", type=int, default=64)
    p.add_argument("--alpha-subsample-rows", type=int, default=128)
    p.add_argument(
        "--no-merge-poly-schedule",
        action="store_true",
        help="Forward to alpha_search: do not merge multipliers into poly_schedule.json",
    )
    p.add_argument(
        "--no-poly-schedule-backup",
        action="store_true",
        help="Forward to alpha_search: no backup before merge",
    )
    p.add_argument(
        "--poly-schedule",
        type=str,
        default=None,
        help="Forward to alpha_search: override path to poly_schedule.json",
    )
    p.add_argument(
        "--alpha-checkpoint",
        type=str,
        default=None,
        help="Forward to alpha_search: checkpoint JSON path (save after each prompt; resume if file exists)",
    )
    p.add_argument(
        "--alpha-resume",
        action="store_true",
        help=f"Forward to alpha_search --resume (default checkpoint under quantized dir)",
    )
    p.add_argument(
        "--keep-alpha-checkpoint",
        action="store_true",
        help="Forward to alpha_search --keep-checkpoint (do not delete after full run)",
    )
    p.add_argument(
        "--alpha-reference-fp32",
        action="store_true",
        help="Forward to alpha_search --reference-fp32 (FP32 reference matmul instead of FP16)",
    )

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    py = sys.executable

    if args.start_from in ("poly", "alpha") and not args.quantized_dir:
        logger.error("--quantized-dir is required when --start-from is poly or alpha")
        return 1

    # ------------------------------------------------------------------
    # Phase 1+2: run_e2e
    # ------------------------------------------------------------------
    if args.start_from == "e2e":
        cmd = [
            py,
            "-m",
            "src.phase2.run_e2e",
            "--output-dir",
            args.output_dir,
            "--diagnostics-dir",
            args.diagnostics_dir,
        ]
        if args.skip_calibration:
            cmd.append("--skip-collection")
        if args.num_prompts_collection is not None:
            cmd += ["--num-prompts", str(args.num_prompts_collection)]
        if args.collection_num_steps is not None:
            cmd += ["--num-steps", str(args.collection_num_steps)]
        if args.collection_cfg_weight is not None:
            cmd += ["--cfg-weight", str(args.collection_cfg_weight)]
        if args.alpha is not None:
            cmd += ["--alpha", str(args.alpha)]
        if args.qkv_method is not None:
            cmd += ["--qkv-method", args.qkv_method]
        if args.group_size is not None:
            cmd += ["--group-size", str(args.group_size)]
        if args.bits is not None:
            cmd += ["--bits", str(args.bits)]
        if args.final_layer_bits is not None:
            cmd += ["--final-layer-bits", str(args.final_layer_bits)]
        if args.ssc_tau is not None:
            cmd += ["--ssc-tau", str(args.ssc_tau)]
        cmd += [
            "--static-mode",
            args.static_mode,
            "--static-granularity",
            args.static_granularity,
        ]

        rc = _run(cmd, args.dry_run)
        if rc != 0:
            return rc

        if args.dry_run:
            try:
                quantized_dir = _expected_quantized_dir(args)
            except ImportError:
                logger.error("Could not resolve expected Phase 2 output directory")
                return 1
            logger.info("Dry-run: expected quantized dir (no checkpoint written): %s", quantized_dir)
        else:
            detected = _detect_quantized_dir_after_e2e(args)
            if detected is None:
                logger.error(
                    "Could not find quantize_config.json under %s after run_e2e",
                    args.output_dir,
                )
                return 1
            if args.quantized_dir:
                explicit = _resolve_under_repo(args.quantized_dir)
                det_p = Path(detected).resolve()
                if explicit != det_p:
                    logger.warning(
                        "--quantized-dir (%s) does not match this run's Phase 2 output (%s); "
                        "using Phase 2 output for poly + alpha.",
                        explicit,
                        det_p,
                    )
            quantized_dir = detected
            logger.info("Using quantized dir: %s", quantized_dir)

    else:
        quantized_dir = str(_resolve_under_repo(args.quantized_dir))

    # ------------------------------------------------------------------
    # Phase 3: poly schedule
    # ------------------------------------------------------------------
    if args.start_from in ("e2e", "poly"):
        if not args.dry_run:
            ok, err = _checkpoint_ok_for_poly_schedule(Path(quantized_dir))
            if not ok:
                logger.error("%s", err)
                return 1
        cmd = [
            py,
            "-m",
            "src.phase3.generate_schedule",
            "--diagnostics-dir",
            args.diagnostics_dir,
            "--calibration-dir",
            quantized_dir,
            "--max-degree",
            str(args.max_degree),
        ]
        if args.per_channel_rho_threshold is not None:
            cmd += [
                "--per-channel-rho-threshold",
                str(args.per_channel_rho_threshold),
            ]
        if args.exclude_layers:
            cmd.append("--exclude-layers")
            cmd.extend(args.exclude_layers)
        if args.include_shifts:
            cmd.append("--include-shifts")

        rc = _run(cmd, args.dry_run)
        if rc != 0:
            return rc

    # ------------------------------------------------------------------
    # Phase 4.1: alpha search
    # ------------------------------------------------------------------
    if args.start_from in ("e2e", "poly", "alpha"):
        if args.start_from == "alpha" and not args.dry_run:
            ok_poly, poly_err = _poly_schedule_available_for_alpha(
                quantized_dir,
                args.poly_schedule,
            )
            if not ok_poly:
                logger.error("%s", poly_err)
                return 1
        alpha_prompts = args.alpha_prompts_file or args.prompts_file
        prompts_path = _resolve_under_repo(alpha_prompts)
        if not args.dry_run and not prompts_path.is_file():
            logger.error("Prompts file not found: %s", prompts_path)
            return 1
        cmd = [
            py,
            "-m",
            "src.phase4_1.alpha_search",
            "--quantized-dir",
            quantized_dir,
            "--prompts-file",
            alpha_prompts,
            "--num-prompts",
            str(args.alpha_num_prompts),
            "--num-steps",
            str(args.alpha_num_steps),
            "--cfg-weight",
            str(args.alpha_cfg_weight),
            "--latent-size",
            str(args.alpha_latent_size),
            "--subsample-rows",
            str(args.alpha_subsample_rows),
        ]
        if args.no_merge_poly_schedule:
            cmd.append("--no-merge-poly-schedule")
        if args.no_poly_schedule_backup:
            cmd.append("--no-poly-schedule-backup")
        if args.poly_schedule:
            cmd += ["--poly-schedule", args.poly_schedule]
        if args.alpha_checkpoint:
            cmd += ["--checkpoint", args.alpha_checkpoint]
        elif args.alpha_resume:
            cmd.append("--resume")
        if args.keep_alpha_checkpoint:
            cmd.append("--keep-checkpoint")
        if args.alpha_reference_fp32:
            cmd.append("--reference-fp32")

        rc = _run(cmd, args.dry_run)
        if rc != 0:
            return rc

    # ------------------------------------------------------------------
    # Finalize: W4A8 checkpoint metadata
    # ------------------------------------------------------------------
    if not args.dry_run:
        _finalize_quantize_config(
            Path(quantized_dir),
            intended_merge=not args.no_merge_poly_schedule,
        )
    else:
        logger.info("[dry-run] Would finalize quantize_config.json in %s", quantized_dir)

    logger.info("=" * 60)
    logger.info("Poly + alpha pipeline complete.")
    logger.info("Checkpoint: %s", quantized_dir)
    logger.info(
        "Load for inference: load_quantized_model_poly(pipeline, Path(%r))",
        quantized_dir,
    )
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
