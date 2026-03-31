#!/usr/bin/env python3
"""Entry point: run Phase 1 analysis and generate all diagnostic plots.

Usage:
    python -m src.phase1.run_analysis

Expects data in diagnostics/ from a prior run_collection run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    from .analyze import (
        build_summary_table,
        compute_spearman_trajectory,
        load_trajectory,
        load_weight_stats,
        per_channel_quant_mse_activation,
        per_channel_quant_mse_weight,
        save_summary_table,
    )
    from .config import (
        ACTIVATION_STATS_DIR,
        OUTPUT_DIR,
        PLOTS_DIR,
        REPRESENTATIVE_LAYERS,
    )
    from .visualize import (
        plot_act_wt_scatter,
        plot_block_depth_profile,
        plot_family_violins,
        plot_fig1_left_reproduction,
        plot_fig3_reproduction,
        plot_fig4_reproduction,
        plot_final_layer_analysis,
        plot_layerwise_rho,
        plot_modality_scatter,
        plot_rank_stability_ribbon,
        plot_risk_ranking,
        plot_rho_trajectory,
        plot_rho_trajectory_grid,
        plot_salience_heatmap,
        plot_salience_heatmap_grid,
        plot_salience_histogram,
        plot_summary_dashboard,
        plot_topk_overlap_heatmap,
    )

    config_path = OUTPUT_DIR / "config.json"
    if not config_path.exists():
        logger.error("No config.json found in %s — run collection first.", OUTPUT_DIR)
        return
    config = json.loads(config_path.read_text())

    logger.info("Loading weight stats ...")
    weight_stats = load_weight_stats()

    logger.info("Building registry from config ...")
    registry = []
    for name in config["layer_names"]:
        parts = name.split(".")
        if name == "context_embedder":
            family, side, block = "context_embedder", "shared", -1
        elif name == "final_layer.linear":
            family, side, block = "final_linear", "image", -1
        else:
            block = int(parts[1])
            side = parts[2]
            family = parts[4]

        act_path = ACTIVATION_STATS_DIR / f"{name}.npz"
        d_in = 0
        if act_path.exists():
            d_in = np.load(act_path)["act_channel_max"].shape[1]

        registry.append({
            "name": name,
            "module": None,
            "block": block,
            "family": family,
            "side": side,
            "d_in": d_in,
        })

    logger.info("Registry reconstructed: %d layers", len(registry))

    # -----------------------------------------------------------------
    # Build summary table
    # -----------------------------------------------------------------
    logger.info("Building summary table ...")
    summary_rows = build_summary_table(registry, weight_stats)
    save_summary_table(summary_rows)
    logger.info("Summary table: %d rows, top risk = %.4f",
                len(summary_rows),
                summary_rows[0]["risk_score"] if summary_rows else 0)

    # -----------------------------------------------------------------
    # Load trajectories for representative layers
    # -----------------------------------------------------------------
    trajectories: dict[str, dict] = {}
    for name in REPRESENTATIVE_LAYERS:
        try:
            trajectories[name] = load_trajectory(name)
        except FileNotFoundError:
            logger.warning("No trajectory data for %s", name)
    if not trajectories:
        logger.error("No trajectory data found — aborting plot generation.")
        return

    ref_sigma = next(iter(trajectories.values()))["sigma_values"]
    num_steps = len(ref_sigma)

    # -----------------------------------------------------------------
    # 10.1  Figure 3 reproduction
    # -----------------------------------------------------------------
    logger.info("Generating Fig. 3 reproductions ...")
    for name in REPRESENTATIVE_LAYERS:
        if name not in trajectories:
            continue
        traj = trajectories[name]["act_channel_max"]
        mid_step = num_steps // 2
        act_salience = traj[mid_step]

        wt = weight_stats.get(name)
        if wt is None:
            continue
        wt_salience = wt["w_channel_max"]

        act_flat = np.outer(np.ones(100), act_salience)
        act_mse = per_channel_quant_mse_activation(act_flat)
        wt_full = np.outer(np.ones(64), wt_salience)  # [64, d_in]
        wt_mse = per_channel_quant_mse_weight(wt_full)

        plot_fig3_reproduction(name, act_salience, wt_salience, act_mse, wt_mse)

    # -----------------------------------------------------------------
    # 10.2  Figure 4 reproduction
    # -----------------------------------------------------------------
    logger.info("Generating Fig. 4 reproductions ...")
    for name in REPRESENTATIVE_LAYERS:
        if name not in trajectories:
            continue
        plot_fig4_reproduction(
            name, trajectories[name]["act_channel_max"], ref_sigma,
        )

    # -----------------------------------------------------------------
    # 10.3  Figure 1-Left reproduction
    # -----------------------------------------------------------------
    logger.info("Generating Fig. 1-Left reproductions ...")
    early = 0
    mid = num_steps // 2
    late = num_steps - 1
    step_indices = [early, mid, late]

    for name in REPRESENTATIVE_LAYERS[:3]:
        if name not in trajectories:
            continue
        wt = weight_stats.get(name)
        if wt is None:
            continue
        plot_fig1_left_reproduction(
            name, trajectories[name]["act_channel_max"],
            wt["w_channel_max"], ref_sigma, step_indices,
        )

    # -----------------------------------------------------------------
    # 10.4  Salience histograms
    # -----------------------------------------------------------------
    logger.info("Generating salience histograms ...")
    for name in REPRESENTATIVE_LAYERS:
        if name not in trajectories:
            continue
        wt = weight_stats.get(name)
        if wt is None:
            continue
        act_s = trajectories[name]["act_channel_max"][mid]
        plot_salience_histogram(name, act_s, wt["w_channel_max"])

    # -----------------------------------------------------------------
    # 10.5  Salience heatmaps + grid
    # -----------------------------------------------------------------
    logger.info("Generating salience heatmaps ...")
    for name in REPRESENTATIVE_LAYERS:
        if name not in trajectories:
            continue
        plot_salience_heatmap(name, trajectories[name]["act_channel_max"], ref_sigma)

    all_trajs_for_grid: dict[str, np.ndarray] = {}
    for name in config["layer_names"]:
        if "image.attn.q_proj" in name:
            try:
                data = load_trajectory(name)
                all_trajs_for_grid[name] = data["act_channel_max"]
            except FileNotFoundError:
                pass
    if all_trajs_for_grid:
        plot_salience_heatmap_grid(all_trajs_for_grid, ref_sigma)

    # -----------------------------------------------------------------
    # 10.6  Layerwise Spearman ρ bar plot
    # -----------------------------------------------------------------
    logger.info("Generating layerwise ρ bar plot ...")
    plot_layerwise_rho(summary_rows)

    # -----------------------------------------------------------------
    # 10.7  Temporal ρ trajectory + grid
    # -----------------------------------------------------------------
    logger.info("Generating ρ trajectories ...")
    all_rho_trajs: dict[str, np.ndarray] = {}
    for name in REPRESENTATIVE_LAYERS:
        if name not in trajectories:
            continue
        wt = weight_stats.get(name)
        if wt is None:
            continue
        rho_traj = compute_spearman_trajectory(
            trajectories[name]["act_channel_max"], wt["w_channel_max"],
        )
        all_rho_trajs[name] = rho_traj
        plot_rho_trajectory(name, rho_traj, ref_sigma)

    all_rho_for_grid: dict[str, np.ndarray] = {}
    for name in config["layer_names"]:
        if "image.attn.q_proj" in name:
            wt = weight_stats.get(name)
            if wt is None:
                continue
            try:
                data = load_trajectory(name)
            except FileNotFoundError:
                continue
            all_rho_for_grid[name] = compute_spearman_trajectory(
                data["act_channel_max"], wt["w_channel_max"],
            )
    if all_rho_for_grid:
        plot_rho_trajectory_grid(all_rho_for_grid, ref_sigma)

    # -----------------------------------------------------------------
    # 10.8  Modality scatter
    # -----------------------------------------------------------------
    logger.info("Generating modality scatter plots ...")
    for metric, label in [
        ("mean_spearman_rho", "Mean ρ"),
        ("cov_temporal", "Mean CoV"),
        ("max_act_salience", "Max salience"),
    ]:
        plot_modality_scatter(summary_rows, metric, label)

    # -----------------------------------------------------------------
    # 10.9  Family violins
    # -----------------------------------------------------------------
    logger.info("Generating family violin plots ...")
    plot_family_violins(summary_rows)

    # -----------------------------------------------------------------
    # 10.10  Block depth profiles
    # -----------------------------------------------------------------
    logger.info("Generating block depth profiles ...")
    for fam in ("q_proj", "fc1", "o_proj"):
        plot_block_depth_profile(summary_rows, family=fam)

    # -----------------------------------------------------------------
    # 10.11  Top-k overlap heatmaps
    # -----------------------------------------------------------------
    logger.info("Generating top-k overlap heatmaps ...")
    for name in [
        "blocks.12.image.attn.q_proj",
        "blocks.12.image.mlp.fc1",
        "final_layer.linear",
    ]:
        if name not in trajectories:
            try:
                data = load_trajectory(name)
                trajectories[name] = data
            except FileNotFoundError:
                continue
        plot_topk_overlap_heatmap(
            name, trajectories[name]["act_channel_max"], ref_sigma,
        )

    # -----------------------------------------------------------------
    # 10.12  Act vs Wt scatter
    # -----------------------------------------------------------------
    logger.info("Generating act vs wt scatter plots ...")
    for name in REPRESENTATIVE_LAYERS:
        if name not in trajectories:
            continue
        wt = weight_stats.get(name)
        if wt is None:
            continue
        act_s = trajectories[name]["act_channel_max"][mid]
        plot_act_wt_scatter(name, act_s, wt["w_channel_max"])

    # -----------------------------------------------------------------
    # 10.13  Rank stability ribbon
    # -----------------------------------------------------------------
    logger.info("Generating rank stability ribbons ...")
    for name in REPRESENTATIVE_LAYERS[:3]:
        if name not in trajectories:
            continue
        plot_rank_stability_ribbon(
            name, trajectories[name]["act_channel_max"], ref_sigma,
        )

    # -----------------------------------------------------------------
    # 10.14  Risk ranking
    # -----------------------------------------------------------------
    logger.info("Generating risk ranking ...")
    plot_risk_ranking(summary_rows, top_n=40)

    # -----------------------------------------------------------------
    # 10.16  Final layer analysis
    # -----------------------------------------------------------------
    logger.info("Generating final layer analysis ...")
    fl_name = "final_layer.linear"
    if fl_name in trajectories:
        wt = weight_stats.get(fl_name)
        if wt is not None:
            mean_traj = trajectories[fl_name].get("act_channel_mean")
            if mean_traj is None:
                try:
                    full_data = load_trajectory(fl_name)
                    mean_traj = full_data["act_channel_mean"]
                except (FileNotFoundError, KeyError):
                    mean_traj = trajectories[fl_name]["act_channel_max"]

            plot_final_layer_analysis(
                trajectory=trajectories[fl_name]["act_channel_max"],
                sigma_values=ref_sigma,
                mean_trajectory=mean_traj,
                wt_salience=wt["w_channel_max"],
            )

    # -----------------------------------------------------------------
    # 10.17  Summary dashboard
    # -----------------------------------------------------------------
    logger.info("Generating summary dashboard ...")
    plot_summary_dashboard(summary_rows)

    # -----------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------
    logger.info("All plots saved to %s", PLOTS_DIR)
    logger.info("Summary table saved to %s/summary_table.csv", OUTPUT_DIR)

    if summary_rows:
        logger.info("\n=== Top 10 highest-risk layers ===")
        for i, r in enumerate(summary_rows[:10]):
            logger.info(
                "  %2d. %-45s  risk=%.4f  ρ=%.3f  CoV=%.3f  max_act=%.2f",
                i + 1, r["layer_name"], r["risk_score"],
                r["mean_spearman_rho"], r["cov_temporal"], r["max_act_salience"],
            )


if __name__ == "__main__":
    main()
