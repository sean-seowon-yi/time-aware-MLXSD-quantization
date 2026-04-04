"""Phase 4 CLI: Hessian collection -> GPTQ weight quantization -> save checkpoint.

Usage:
    python -m src.phase4.run_phase4 \\
        --quantized-dir quantized/<tag>/ \\
        --prompts-file src/settings/coco_100_calibration_prompts.txt \\
        --num-prompts 16

    # Raw Hessians (no poly fake-quant):
    python -m src.phase4.run_phase4 \\
        --quantized-dir quantized/<tag>/ \\
        --prompts-file src/settings/coco_100_calibration_prompts.txt \\
        --raw-hessian

    # Skip Hessian collection (reuse saved Hessians):
    python -m src.phase4.run_phase4 \\
        --quantized-dir quantized/<tag>/ \\
        --skip-collection

Prerequisites:
    Phase 2: quantized checkpoint with mmdit_quantized.safetensors
    Phase 3 (optional): poly_schedule.json for sigma-aware Hessians
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HESSIAN_DIR_NAME = "hessians"
GPTQ_META_FILENAME = "gptq_meta.json"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Phase 4: GPTQ weight quantization with Hessian-weighted "
                    "error compensation",
    )

    # --- I/O ---
    p.add_argument(
        "--quantized-dir", type=Path, required=True,
        help="Phase 2 checkpoint directory (must contain quantize_config.json)",
    )
    p.add_argument(
        "--prompts-file", type=Path, default=None,
        help="Tab-separated seed<TAB>prompt file for calibration",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (defaults to --quantized-dir)",
    )

    # --- Collection ---
    p.add_argument(
        "--skip-collection", action="store_true",
        help="Skip Hessian collection; reuse saved Hessians from <output-dir>/hessians/",
    )
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--cfg-weight", type=float, default=4.0)
    p.add_argument(
        "--raw-hessian", action="store_true",
        help="Collect Hessians from full-precision activations (no poly fake-quant)",
    )

    # --- GPTQ ---
    p.add_argument("--bits", type=int, default=4)
    p.add_argument(
        "--group-size", type=int, default=None,
        help="GPTQ group size (default: read from Phase 2 checkpoint)",
    )
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--damp-percent", type=float, default=0.01)

    # --- Skip steps ---
    p.add_argument(
        "--skip-gptq", action="store_true",
        help="Skip GPTQ; only collect + save Hessians",
    )

    return p.parse_args()


def _build_pipeline():
    """Construct DiffusionPipeline using Phase 2 config."""
    from ..phase2.config import DIFFUSIONKIT_SRC, MODEL_VERSION, PIPELINE_KWARGS

    sys.path.insert(0, DIFFUSIONKIT_SRC)
    from diffusionkit.mlx import DiffusionPipeline

    pipeline = DiffusionPipeline(**PIPELINE_KWARGS, model_version=MODEL_VERSION)
    return pipeline


def _build_denoiser(pipeline):
    """Create CFGDenoiser wrapper around the pipeline."""
    from diffusionkit.mlx import CFGDenoiser

    return CFGDenoiser(pipeline)


def _load_poly_schedule(quantized_dir: Path) -> dict:
    """Load poly_schedule.json if it exists, else return empty schedule."""
    poly_path = quantized_dir / "poly_schedule.json"
    if poly_path.exists():
        with open(poly_path) as f:
            schedule = json.load(f)
        logger.info(
            "Loaded poly schedule: %d layers (version: %s)",
            len(schedule.get("layers", {})), schedule.get("version"),
        )
        return schedule
    logger.info("No poly_schedule.json found -- Hessians will use raw activations")
    return {"layers": {}}


def _extract_csb_weights(pipeline, include_final: bool = True) -> dict:
    """Extract CSB-balanced weight matrices from the Phase 2-loaded model.

    Must be called AFTER loading the Phase 2 checkpoint.  Each layer is
    either a ``W4A8StaticLinear`` (static A8) or ``W4A8Linear`` (dynamic A8)
    wrapping an ``nn.QuantizedLinear``.  We dequantize the packed weights
    to recover the CSB-balanced FP32 matrices that GPTQ should optimise.

    For layers that were not quantized (e.g. ``final_layer_bits >= 16``),
    the raw ``nn.Linear`` weight is returned directly.

    Returns {poly_key: (W_fp32, bias_fp32_or_None)}.
    """
    import mlx.core as mx
    import mlx.nn as nn

    from .hessian import FINAL_LAYER_KEY
    from .utils import _get_block_linears, full_path_to_poly_key

    def _extract_weight(layer):
        ql = layer.qlinear if hasattr(layer, "qlinear") else layer
        if isinstance(ql, nn.QuantizedLinear):
            W_deq = mx.dequantize(
                ql.weight, ql.scales, ql.biases,
                bits=ql.bits, group_size=ql.group_size,
            )
            W_np = np.array(W_deq, dtype=np.float32)
        else:
            W_np = np.array(ql.weight, dtype=np.float32)
        bias_np = None
        if "bias" in ql and ql.bias is not None:
            bias_np = np.array(ql.bias, dtype=np.float32)
        return W_np, bias_np

    mmdit = pipeline.mmdit
    weights = {}

    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            weights[poly_key] = _extract_weight(layer)

    if include_final:
        fl = mmdit.final_layer.linear
        weights[FINAL_LAYER_KEY] = _extract_weight(fl)

    logger.info("Extracted %d CSB-balanced weight matrices (dequantized)", len(weights))
    return weights


def _save_hessians(
    all_collectors: dict,
    output_dir: Path,
):
    """Save Hessians to .npz files under output_dir/hessians/."""
    hessian_dir = output_dir / HESSIAN_DIR_NAME
    hessian_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for _block_idx, collectors in all_collectors.items():
        for poly_key, collector in collectors.items():
            H = collector.get_hessian()
            np.savez_compressed(
                hessian_dir / f"{poly_key}.npz",
                H=H, n_samples=collector._n_samples,
            )
            count += 1

    logger.info("Saved %d Hessians to %s", count, hessian_dir)


def _load_hessians(output_dir: Path) -> dict:
    """Load saved Hessians from output_dir/hessians/. Returns {poly_key: H}."""
    hessian_dir = output_dir / HESSIAN_DIR_NAME
    if not hessian_dir.exists():
        raise FileNotFoundError(f"No Hessian directory at {hessian_dir}")

    hessians = {}
    for path in sorted(hessian_dir.glob("*.npz")):
        data = np.load(path)
        poly_key = path.stem
        hessians[poly_key] = data["H"]

    logger.info("Loaded %d Hessians from %s", len(hessians), hessian_dir)
    return hessians


def _run_gptq_all_layers(
    hessians: dict,
    fp_weights: dict,
    bits: int,
    group_size: int,
    block_size: int,
    damp_percent: float,
) -> dict:
    """Run GPTQ on all layers. Returns {poly_key: (W_q_int, scales, mse)}."""
    from tqdm import tqdm

    from .gptq_quantize import gptq_quantize

    results = {}
    total_mse = 0.0

    keys = sorted(set(hessians.keys()) & set(fp_weights.keys()))
    skipped_h = set(fp_weights.keys()) - set(hessians.keys())
    skipped_w = set(hessians.keys()) - set(fp_weights.keys())
    if skipped_h:
        logger.warning(
            "Skipping %d layers (Hessian missing): %s",
            len(skipped_h), sorted(skipped_h)[:5],
        )
    if skipped_w:
        logger.warning(
            "Skipping %d layers (weights missing): %s",
            len(skipped_w), sorted(skipped_w)[:5],
        )
    logger.info(
        "Running GPTQ on %d layers (bits=%d, gs=%d, bs=%d, damp=%.3f)",
        len(keys), bits, group_size, block_size, damp_percent,
    )

    for poly_key in tqdm(keys, desc="GPTQ"):
        W, _ = fp_weights[poly_key]
        H = hessians[poly_key]

        W_q_int, scales, mse = gptq_quantize(
            W, H, bits=bits,
            damp_percent=damp_percent,
            block_size=block_size,
            group_size=group_size,
        )
        results[poly_key] = (W_q_int, scales, mse)
        total_mse += mse

    logger.info(
        "GPTQ complete: %d layers, total weight MSE = %.6f, mean = %.6f",
        len(results), total_mse, total_mse / max(len(results), 1),
    )
    return results


def _repack_and_save(
    pipeline,
    gptq_results: dict,
    fp_weights: dict,
    output_dir: Path,
    quantized_dir: Path,
    bits: int,
    group_size: int,
    args,
):
    """Repack GPTQ weights into model and save checkpoint."""
    import shutil

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from ..phase2.config import QUANTIZE_CONFIG_FILENAME, QUANTIZED_WEIGHTS_FILENAME
    from .repack import build_quantized_linear
    from .utils import _get_block_linears, _set_nested, full_path_to_poly_key

    from .hessian import FINAL_LAYER_KEY

    mmdit = pipeline.mmdit
    layer_mse = {}

    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            if poly_key not in gptq_results:
                continue

            W_q_int, scales, mse = gptq_results[poly_key]
            _, bias_np = fp_weights[poly_key]
            layer_mse[poly_key] = float(mse)

            ql_new = build_quantized_linear(
                W_q_int, scales, bias_np, bits, group_size,
            )

            if hasattr(layer, "qlinear"):
                layer.qlinear = ql_new
            else:
                _set_nested(block, full_path, ql_new)

    if FINAL_LAYER_KEY in gptq_results:
        W_q_int, scales, mse = gptq_results[FINAL_LAYER_KEY]
        _, bias_np = fp_weights[FINAL_LAYER_KEY]
        layer_mse[FINAL_LAYER_KEY] = float(mse)

        ql_new = build_quantized_linear(
            W_q_int, scales, bias_np, bits, group_size,
        )

        fl = mmdit.final_layer.linear
        if hasattr(fl, "qlinear"):
            fl.qlinear = ql_new
        else:
            mmdit.final_layer.linear = ql_new

    logger.info("Repacked %d layers into model", len(layer_mse))

    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = quantized_dir / QUANTIZE_CONFIG_FILENAME
    meta = json.loads(meta_path.read_text())
    meta["weight_quant"] = "gptq"
    meta["gptq_config"] = {
        "bits": bits,
        "group_size": group_size,
        "block_size": args.block_size,
        "damp_percent": args.damp_percent,
        "num_prompts": args.num_prompts,
        "raw_hessian": args.raw_hessian,
    }

    weights = {
        k: v for k, v in tree_flatten(mmdit.parameters())
        if not k.startswith("to_offload.")
    }
    weight_path = output_dir / QUANTIZED_WEIGHTS_FILENAME
    mx.save_safetensors(str(weight_path), weights)
    logger.info("Saved %d tensors to %s", len(weights), weight_path)

    config_path = output_dir / QUANTIZE_CONFIG_FILENAME
    config_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved config to %s", config_path)

    gptq_meta = {
        "layer_mse": layer_mse,
        "total_mse": sum(layer_mse.values()),
        "mean_mse": sum(layer_mse.values()) / max(len(layer_mse), 1),
        "n_layers": len(layer_mse),
        "bits": bits,
        "group_size": group_size,
        "block_size": args.block_size,
        "damp_percent": args.damp_percent,
    }
    gptq_meta_path = output_dir / GPTQ_META_FILENAME
    gptq_meta_path.write_text(json.dumps(gptq_meta, indent=2))
    logger.info("Saved GPTQ meta to %s", gptq_meta_path)

    if output_dir.resolve() != quantized_dir.resolve():
        for name in [
            "poly_schedule.json",
            "calibration.npz",
            "calibration_meta.json",
            "static_scales.npz",
        ]:
            src = quantized_dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)


def main():
    args = _parse_args()
    output_dir = args.output_dir or args.quantized_dir

    # Resolve bits, group_size, and final_layer_bits from Phase 2 meta.
    from ..phase2.config import QUANTIZE_CONFIG_FILENAME
    p2_meta = json.loads((args.quantized_dir / QUANTIZE_CONFIG_FILENAME).read_text())
    p2_bits = p2_meta.get("bits", 4)
    p2_group_size = p2_meta["group_size"]
    p2_final_bits = p2_meta.get("final_layer_bits", p2_bits)
    include_final = (p2_final_bits == args.bits) and (p2_final_bits < 16)

    if args.bits != p2_bits:
        raise ValueError(
            f"--bits {args.bits} differs from Phase 2 checkpoint's "
            f"bits={p2_bits}.  GPTQ bit-width must match Phase 2 to "
            f"preserve tensor shapes in the checkpoint."
        )

    if args.group_size is None:
        args.group_size = p2_group_size
    elif args.group_size != p2_group_size:
        raise ValueError(
            f"--group-size {args.group_size} differs from Phase 2 checkpoint's "
            f"group_size={p2_group_size}.  GPTQ group_size must match Phase 2."
        )

    if not include_final:
        logger.info(
            "final_layer.linear will be SKIPPED by GPTQ "
            "(Phase 2 final_layer_bits=%s, GPTQ bits=%d)",
            p2_final_bits, args.bits,
        )

    logger.info("=" * 60)
    logger.info("Phase 4: GPTQ Weight Quantization")
    logger.info("=" * 60)
    logger.info("Quantized dir : %s", args.quantized_dir)
    logger.info("Output dir    : %s", output_dir)
    logger.info("Skip collection: %s", args.skip_collection)
    logger.info("Skip GPTQ     : %s", args.skip_gptq)
    logger.info("Raw Hessian   : %s", args.raw_hessian)
    logger.info("Include final_layer: %s", include_final)
    logger.info(
        "GPTQ bits=%d  group_size=%d  block_size=%d  damp=%.3f",
        args.bits, args.group_size, args.block_size, args.damp_percent,
    )

    # ------------------------------------------------------------------
    # Step 1: Build pipeline (original FP16 model)
    # ------------------------------------------------------------------
    t0 = time.time()
    pipeline = _build_pipeline()
    logger.info("Built pipeline in %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # Step 2: Load Phase 2 checkpoint (model becomes W4A8)
    # ------------------------------------------------------------------
    t1 = time.time()
    if p2_meta.get("act_quant") == "static":
        from ..phase2.quantize_static import load_quantized_model_static
        load_quantized_model_static(pipeline, args.quantized_dir)
        logger.info(
            "Loaded Phase 2 static checkpoint in %.1fs (mode=%s, granularity=%s)",
            time.time() - t1,
            p2_meta.get("static_mode"), p2_meta.get("static_granularity"),
        )
    else:
        from ..phase2.quantize import load_quantized_model
        load_quantized_model(pipeline, args.quantized_dir)
        logger.info("Loaded Phase 2 checkpoint in %.1fs", time.time() - t1)

    # ------------------------------------------------------------------
    # Step 3: Extract CSB-balanced weights (AFTER Phase 2 load)
    # ------------------------------------------------------------------
    if not args.skip_gptq:
        fp_weights = _extract_csb_weights(pipeline, include_final=include_final)
    else:
        fp_weights = None

    # ------------------------------------------------------------------
    # Step 4: Collect or load Hessians
    # ------------------------------------------------------------------
    if args.skip_collection:
        hessians = _load_hessians(output_dir) if not args.skip_gptq else {}
    else:
        if args.prompts_file is None:
            raise ValueError(
                "--prompts-file is required when not using --skip-collection"
            )

        from .utils import load_prompt_file
        prompt_entries = load_prompt_file(args.prompts_file)[:args.num_prompts]
        logger.info("Using %d prompts for Hessian collection", len(prompt_entries))

        poly_schedule = _load_poly_schedule(args.quantized_dir)
        if args.raw_hessian:
            poly_schedule = {"layers": {}}

        denoiser = _build_denoiser(pipeline)

        from .hessian import collect_hessians_global
        t2 = time.time()
        all_collectors = collect_hessians_global(
            pipeline, denoiser, prompt_entries,
            poly_schedule=poly_schedule,
            num_steps=args.num_steps,
            cfg_weight=args.cfg_weight,
            raw_hessian=args.raw_hessian,
            include_final=include_final,
        )
        logger.info("Hessian collection took %.1fs", time.time() - t2)

        output_dir.mkdir(parents=True, exist_ok=True)
        _save_hessians(all_collectors, output_dir)

        hessians = {}
        for block_collectors in all_collectors.values():
            for poly_key, collector in block_collectors.items():
                hessians[poly_key] = collector.get_hessian()

    if args.skip_gptq:
        logger.info("--skip-gptq set, stopping after Hessian collection")
        return

    # ------------------------------------------------------------------
    # Step 5: Run GPTQ
    # ------------------------------------------------------------------
    t3 = time.time()
    gptq_results = _run_gptq_all_layers(
        hessians, fp_weights,
        bits=args.bits,
        group_size=args.group_size,
        block_size=args.block_size,
        damp_percent=args.damp_percent,
    )
    logger.info("GPTQ took %.1fs", time.time() - t3)

    # ------------------------------------------------------------------
    # Step 6: Repack into MLX format and save
    # ------------------------------------------------------------------
    _repack_and_save(
        pipeline, gptq_results, fp_weights,
        output_dir, args.quantized_dir,
        bits=args.bits, group_size=args.group_size,
        args=args,
    )

    logger.info("=" * 60)
    logger.info("Phase 4 complete. Output: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
