#!/usr/bin/env python3
"""Post-quantization diagnostics: collect W4A8 activation stats, compare
weight/activation distributions against FP16 baselines, and generate plots.

Usage
-----
# ``--quantized-dir`` must be the folder containing ``quantize_config.json``
# (e.g. ``quantized/w4a8_max_a0.50_gs64/`` from ``run_e2e``).

# Full diagnostics (collect + analyze + plot)
python -m src.phase2.run_diagnose --quantized-dir quantized/<tag>/ --output-dir post_quant_diagnostics/

# Quick test with 2 prompts
python -m src.phase2.run_diagnose --quantized-dir quantized/<tag>/ --num-prompts 2 --output-dir post_quant_diagnostics/

# Skip collection (reuse previously collected W4A8 stats)
python -m src.phase2.run_diagnose --quantized-dir quantized/<tag>/ --skip-collection --output-dir post_quant_diagnostics/

# Analysis + plots only (no model loading; requires prior collection)
python -m src.phase2.run_diagnose --analysis-only --output-dir post_quant_diagnostics/

Steps
-----
1. Load pipeline + quantized model
2. Hook all layers and run prompts through quantized denoiser
3. Save W4A8 activation trajectories
4. Compare FP16 vs W4A8 weights (dequantized) and activations
5. Generate diagnostic plots
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Post-quantization diagnostics: W4A8 vs FP16 comparison",
    )

    parser.add_argument(
        "--quantized-dir", type=str, default="quantized",
        help="Quantized model directory (default: quantized/)",
    )
    parser.add_argument(
        "--diagnostics-dir", type=str, default="diagnostics",
        help="Phase 1 FP16 diagnostics directory (default: diagnostics/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="post_quant_diagnostics",
        help="Output directory for W4A8 diagnostics (default: post_quant_diagnostics/)",
    )

    parser.add_argument(
        "--skip-collection", action="store_true",
        help="Skip W4A8 activation collection (use existing data)",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Only run analysis + plots (no model loading)",
    )

    parser.add_argument("--num-prompts", type=int, default=None, help="Number of prompt-seed pairs (default: all)")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-weight", type=float, default=4.0)

    args = parser.parse_args()

    from pathlib import Path
    import json
    import numpy as np

    output_dir = Path(args.output_dir)
    quantized_dir = Path(args.quantized_dir)
    fp16_diag_dir = Path(args.diagnostics_dir)

    w4a8_act_dir = output_dir / "activation_stats"
    plots_dir = output_dir / "plots"

    t_total = time.time()

    # ==================================================================
    # Phase A: Activation collection on quantized model
    # ==================================================================
    if not args.analysis_only and not args.skip_collection:
        logger.info("=" * 60)
        logger.info("PHASE A: Collecting activation stats on W4A8 model")
        logger.info("=" * 60)

        import mlx.core as mx

        from .config import (
            DIFFUSIONKIT_SRC,
            MODEL_VERSION,
            PIPELINE_KWARGS,
            QUANTIZE_CONFIG_FILENAME,
        )

        sys.path.insert(0, DIFFUSIONKIT_SRC)
        from diffusionkit.mlx import DiffusionPipeline

        meta = json.loads(
            (quantized_dir / QUANTIZE_CONFIG_FILENAME).read_text()
        )
        model_version = meta.get("model_version", MODEL_VERSION)

        logger.info("Loading pipeline ...")
        pipeline = DiffusionPipeline(
            **PIPELINE_KWARGS,
            model_version=model_version,
        )

        logger.info("Loading quantized model from %s ...", quantized_dir)
        from .quantize import load_quantized_model
        load_quantized_model(pipeline, quantized_dir)
        logger.info("Quantized model loaded.")

        from ..phase1.config import CALIBRATION_PAIRS
        from ..phase1.hooks import ChannelStatsCollector
        from .diagnose import (
            build_quantized_registry,
            install_quantized_hooks,
            remove_quantized_hooks,
            run_quantized_collection,
            save_quantized_activation_stats,
        )

        registry = build_quantized_registry(pipeline.mmdit)
        logger.info("Quantized registry: %d layers", len(registry))

        n = args.num_prompts or len(CALIBRATION_PAIRS)
        pairs = CALIBRATION_PAIRS[:n]

        logger.info(
            "Collection: %d prompt-seed pairs, %d steps, CFG %.1f",
            len(pairs), args.num_steps, args.cfg_weight,
        )

        collector = ChannelStatsCollector()
        hooks = install_quantized_hooks(registry, collector)

        t_collect = time.time()
        run_quantized_collection(
            pipeline, pairs, collector,
            num_steps=args.num_steps,
            latent_size=(64, 64),
            cfg_weight=args.cfg_weight,
        )
        logger.info("Collection finished in %.1f s", time.time() - t_collect)

        remove_quantized_hooks(hooks)
        save_quantized_activation_stats(collector, registry, w4a8_act_dir)

        # Weight error analysis (needs quantized model in memory)
        logger.info("Computing weight errors ...")
        from ..phase1.collect import load_weight_stats
        from .diagnose import compute_weight_errors, save_weight_errors

        fp16_wt_stats = load_weight_stats(fp16_diag_dir)
        weight_errors = compute_weight_errors(fp16_wt_stats, registry)
        save_weight_errors(weight_errors, output_dir)

        del pipeline
        mx.metal.clear_cache() if hasattr(mx, "metal") else None

    elif not args.analysis_only and args.skip_collection:
        logger.info("=" * 60)
        logger.info("PHASE A: SKIPPED — using existing W4A8 stats in %s", w4a8_act_dir)
        logger.info("=" * 60)

    # ==================================================================
    # Phase B: Analysis + comparison
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE B: Comparing FP16 vs W4A8 distributions")
    logger.info("=" * 60)

    fp16_act_dir = fp16_diag_dir / "activation_stats"

    fp16_cfg_path = fp16_diag_dir / "config.json"
    fp16_cfg = json.loads(fp16_cfg_path.read_text())
    layer_names = fp16_cfg["layer_names"]

    from .diagnose import compare_activation_trajectories, save_activation_comparison

    w4a8_available = [
        n for n in layer_names
        if (w4a8_act_dir / f"{n}.npz").exists()
    ]
    logger.info(
        "Comparing %d layers (%d have W4A8 data)",
        len(layer_names), len(w4a8_available),
    )

    act_comparisons = compare_activation_trajectories(
        fp16_act_dir, w4a8_act_dir, w4a8_available,
    )
    save_activation_comparison(act_comparisons, output_dir / "activation_comparison")

    # Load weight errors if they exist (from collection phase or prior run)
    weight_errors = []
    we_path = output_dir / "weight_error_summary.json"
    if we_path.exists():
        weight_errors = json.loads(we_path.read_text())
        for e in weight_errors:
            e["channel_mse"] = np.zeros(1)
        logger.info("Loaded weight error summary (%d layers)", len(weight_errors))

    # ==================================================================
    # Phase C: Plotting
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE C: Generating diagnostic plots")
    logger.info("=" * 60)

    from .visualize_quant import (
        plot_activation_error_over_time,
        plot_activation_snr_ranking,
        plot_b_inv_distributions,
        plot_channel_error_heatmap,
        plot_summary_dashboard,
        plot_trajectory_grid,
        plot_trajectory_overlay,
        plot_weight_error_ranking,
        plot_weight_snr_distribution,
    )

    sigma_values = None
    if w4a8_available:
        sample = np.load(fp16_act_dir / f"{w4a8_available[0]}.npz")
        sigma_values = sample.get("sigma_values")

    representative = [
        "blocks.0.image.attn.q_proj",
        "blocks.0.text.attn.q_proj",
        "blocks.12.image.attn.o_proj",
        "blocks.12.text.attn.o_proj",
        "blocks.12.image.mlp.fc1",
        "blocks.12.image.mlp.fc2",
        "blocks.0.text.mlp.fc2",
        "final_layer.linear",
    ]

    comp_map = {c["name"]: c for c in act_comparisons}

    for name in representative:
        comp = comp_map.get(name)
        if comp is None:
            continue

        plot_trajectory_overlay(
            name, comp["fp16_traj"], comp["w4a8_traj"],
            sigma_values=sigma_values, output_dir=plots_dir,
        )
        plot_activation_error_over_time(
            name, comp["per_step_mse"], comp["per_step_snr"],
            sigma_values=sigma_values, output_dir=plots_dir,
        )
        plot_channel_error_heatmap(
            name, comp["fp16_traj"], comp["w4a8_traj"],
            output_dir=plots_dir,
        )

    if weight_errors:
        plot_weight_error_ranking(weight_errors, top_n=40, output_dir=plots_dir)
        plot_weight_snr_distribution(weight_errors, output_dir=plots_dir)

    if act_comparisons:
        plot_activation_snr_ranking(act_comparisons, top_n=40, output_dir=plots_dir)
        plot_trajectory_grid(
            act_comparisons, selected_layers=representative,
            sigma_values=sigma_values, output_dir=plots_dir,
        )

    cal_path = quantized_dir / "calibration.npz"
    cal_meta_path = quantized_dir / "calibration_meta.json"
    if cal_path.exists() and cal_meta_path.exists():
        cal_meta = json.loads(cal_meta_path.read_text())
        plot_b_inv_distributions(
            cal_path, cal_meta.get("b_inv_layers", []),
            output_dir=plots_dir,
        )

    if weight_errors and act_comparisons:
        plot_summary_dashboard(weight_errors, act_comparisons, output_dir=plots_dir)

    total_elapsed = time.time() - t_total
    logger.info(
        "\n" + "=" * 60 + "\n"
        "  POST-QUANTIZATION DIAGNOSTICS COMPLETE\n"
        "=" * 60 + "\n"
        "  Total time:          %.1f s\n"
        "  Layers compared:     %d\n"
        "  Plots saved to:      %s\n"
        "  Activation stats:    %s\n"
        "  Weight errors:       %s",
        total_elapsed,
        len(act_comparisons),
        plots_dir,
        w4a8_act_dir,
        output_dir / "weight_error_summary.json",
    )

    if act_comparisons:
        snrs = [c["overall_snr"] for c in act_comparisons]
        mses = [c["overall_mse"] for c in act_comparisons]
        logger.info(
            "\n  --- Activation Summary ---\n"
            "  Mean SNR:   %.1f dB\n"
            "  Median SNR: %.1f dB\n"
            "  Min SNR:    %.1f dB  (%s)\n"
            "  Mean MSE:   %.6f\n"
            "  Max MSE:    %.6f  (%s)",
            np.mean(snrs), np.median(snrs),
            min(snrs), act_comparisons[np.argmin(snrs)]["name"],
            np.mean(mses),
            max(mses), act_comparisons[np.argmax(mses)]["name"],
        )


if __name__ == "__main__":
    main()
