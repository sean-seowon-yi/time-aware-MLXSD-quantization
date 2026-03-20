"""Data collection routines: weight salience, adaLN stats, and the main
denoising-loop collection with forward hooks.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List

import mlx.core as mx
import numpy as np

from .config import ACTIVATION_STATS_DIR, DIAG_CONFIG, OUTPUT_DIR
from .hooks import ChannelStatsCollector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight salience (time-independent, computed once)
# ---------------------------------------------------------------------------

def compute_weight_salience(registry: list[dict]) -> dict:
    """Compute per-channel weight statistics for every layer in the registry.

    Returns dict[layer_name] -> {"w_channel_max": ndarray[d_in],
                                  "w_channel_mean": ndarray[d_in]}.
    """
    weight_stats = {}
    for entry in registry:
        W = entry["module"].weight.astype(mx.float32)
        w_abs = mx.abs(W)
        w_max = mx.max(w_abs, axis=0)
        w_mean = mx.mean(w_abs, axis=0)
        mx.eval(w_max, w_mean)
        weight_stats[entry["name"]] = {
            "w_channel_max": np.array(w_max),
            "w_channel_mean": np.array(w_mean),
        }
    logger.info("Computed weight salience for %d layers", len(weight_stats))
    return weight_stats


# ---------------------------------------------------------------------------
# adaLN modulation cache statistics
# ---------------------------------------------------------------------------

def collect_adaln_stats(mmdit) -> dict:
    """Extract per-channel max magnitudes from the pre-cached adaLN modulation
    parameters. Must be called after cache_modulation_params().

    Returns dict[layer_name] -> dict[timestep_key] -> ndarray.
    """
    adaln_records: dict = {}

    for bidx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for side, tb in [
            ("image", block.image_transformer_block),
            ("text", block.text_transformer_block),
        ]:
            name = f"blocks.{bidx}.{side}.adaLN"
            adaln_records[name] = {}
            if not hasattr(tb, "_modulation_params"):
                continue
            for ts_key, params in tb._modulation_params.items():
                params_fp32 = params.astype(mx.float32)
                abs_params = mx.abs(params_fp32)
                channel_max = mx.max(
                    abs_params.reshape(-1, abs_params.shape[-1]), axis=0
                )
                mx.eval(channel_max)
                adaln_records[name][ts_key] = np.array(channel_max)

    if hasattr(mmdit.final_layer, "_modulation_params"):
        name = "final_layer.adaLN"
        adaln_records[name] = {}
        for ts_key, params in mmdit.final_layer._modulation_params.items():
            params_fp32 = params.astype(mx.float32)
            abs_params = mx.abs(params_fp32)
            channel_max = mx.max(
                abs_params.reshape(-1, abs_params.shape[-1]), axis=0
            )
            mx.eval(channel_max)
            adaln_records[name][ts_key] = np.array(channel_max)

    logger.info("Collected adaLN stats for %d layers", len(adaln_records))
    return adaln_records


# ---------------------------------------------------------------------------
# Main collection loop (Euler sampler, optional CFG)
# ---------------------------------------------------------------------------

def run_diagnostic_collection(
    pipeline,
    prompts: List[str],
    seeds: List[int],
    collector: ChannelStatsCollector,
    num_steps: int | None = None,
    latent_size: tuple[int, int] | None = None,
    cfg_weight: float | None = None,
) -> None:
    """Run the Euler denoising loop for every (prompt, seed) pair, firing
    hooks on every denoising step so the collector accumulates statistics.

    When ``cfg_weight > 0``, classifier-free guidance is applied: the MMDiT
    receives batch=2 inputs (conditioned + unconditioned) per step, and the
    outputs are combined via the standard CFG formula before the Euler update.
    """
    num_steps = num_steps or DIAG_CONFIG["num_steps"]
    latent_size = latent_size or tuple(DIAG_CONFIG["latent_size"])
    cfg_weight = cfg_weight if cfg_weight is not None else DIAG_CONFIG["cfg_weight"]
    use_cfg = cfg_weight > 0

    total_runs = len(prompts) * len(seeds)
    run_idx = 0

    # Cache the original adaLN weights from the in-memory model to avoid
    # reloading the full checkpoint from disk after every prompt
    # (DiffusionKit's load_mmdit reads the entire safetensors file each call).
    from mlx.utils import tree_flatten
    _adaln_cache = [
        (k, v) for k, v in tree_flatten(pipeline.mmdit.parameters())
        if "adaLN" in k
    ]

    for prompt_id, prompt in enumerate(prompts):
        conditioning, pooled_conditioning = pipeline.encode_text(
            prompt, cfg_weight=cfg_weight,
        )
        if not use_cfg:
            conditioning = conditioning[:1]
            pooled_conditioning = pooled_conditioning[:1]
        conditioning = conditioning.astype(pipeline.activation_dtype)
        pooled_conditioning = pooled_conditioning.astype(pipeline.activation_dtype)
        mx.eval(conditioning, pooled_conditioning)
        batch_size = conditioning.shape[0]  # 2 with CFG, 1 without

        for seed in seeds:
            run_idx += 1
            t0 = time.time()
            logger.info(
                "Run %d/%d  prompt=%d  seed=%d  cfg=%.1f",
                run_idx, total_runs, prompt_id, seed, cfg_weight,
            )

            mx.random.seed(seed)
            x_T = pipeline.get_empty_latent(*latent_size)
            noise = pipeline.get_noise(seed, x_T)
            sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
            noise_scaled = pipeline.sampler.noise_scaling(
                sigmas[0], noise, x_T, True,
            )

            timesteps = pipeline.sampler.timestep(sigmas).astype(
                pipeline.activation_dtype
            )

            pipeline.mmdit.cache_modulation_params(
                pooled_conditioning, timesteps,
            )

            x = noise_scaled.astype(pipeline.activation_dtype)
            mmdit = pipeline.mmdit

            for i in range(len(sigmas) - 1):
                sigma_val = float(sigmas[i].item())
                collector.set_context(
                    step_idx=i,
                    sigma=sigma_val,
                    prompt_id=str(prompt_id),
                    seed=seed,
                )

                if use_cfg:
                    x_in = mx.concatenate([x] * 2, axis=0)
                else:
                    x_in = x

                token_text = mx.expand_dims(conditioning, 2)
                ts = mx.broadcast_to(timesteps[i], [batch_size])

                mmdit_output = mmdit(
                    latent_image_embeddings=x_in,
                    token_level_text_embeddings=token_text,
                    timestep=ts,
                )

                if use_cfg:
                    eps_pred = pipeline.sampler.calculate_denoised(
                        sigmas[i], mmdit_output, x_in,
                    )
                    eps_cond, eps_uncond = eps_pred.split(2)
                    denoised = eps_uncond + cfg_weight * (eps_cond - eps_uncond)
                else:
                    denoised = pipeline.sampler.calculate_denoised(
                        sigmas[i], mmdit_output, x,
                    )

                d = (x - denoised) / sigmas[i]
                x = x + d * (sigmas[i + 1] - sigmas[i])
                mx.eval(x)

            pipeline.mmdit.load_weights(_adaln_cache, strict=False)
            pipeline.mmdit.clear_modulation_params_cache()

            elapsed = time.time() - t0
            logger.info(
                "  completed in %.1fs  (total hook calls so far: %d)",
                elapsed, collector.call_count,
            )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_weight_stats(weight_stats: dict, output_dir: Path | None = None):
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    flat = {}
    for layer_name, stats in weight_stats.items():
        for stat_key, arr in stats.items():
            flat[f"{layer_name}/{stat_key}"] = arr
    path = out / "weight_stats.npz"
    np.savez_compressed(path, **flat)
    logger.info("Saved weight stats to %s", path)


def load_weight_stats(output_dir: Path | None = None) -> dict:
    path = (output_dir or OUTPUT_DIR) / "weight_stats.npz"
    data = np.load(path)
    weight_stats: dict = {}
    for key in data.files:
        layer_name, stat_key = key.rsplit("/", 1)
        if layer_name not in weight_stats:
            weight_stats[layer_name] = {}
        weight_stats[layer_name][stat_key] = data[key]
    return weight_stats


def save_activation_stats(
    collector: ChannelStatsCollector,
    registry: list[dict],
    output_dir: Path | None = None,
):
    """Save per-layer activation trajectories as individual .npz files."""
    act_dir = output_dir or ACTIVATION_STATS_DIR
    act_dir.mkdir(parents=True, exist_ok=True)

    sigma_values = collector.sigma_values()

    for entry in registry:
        name = entry["name"]
        if name not in collector.layer_names():
            continue
        trajectory = collector.get_trajectory(name)
        mean_trajectory = collector.get_mean_trajectory(name)

        path = act_dir / f"{name}.npz"
        np.savez_compressed(
            path,
            sigma_values=sigma_values,
            act_channel_max=trajectory,
            act_channel_mean=mean_trajectory,
        )
    logger.info("Saved activation stats to %s (%d files)", act_dir, len(registry))


def load_activation_stats(
    layer_name: str, output_dir: Path | None = None,
) -> dict:
    path = (output_dir or ACTIVATION_STATS_DIR) / f"{layer_name}.npz"
    data = np.load(path)
    return {
        "sigma_values": data["sigma_values"],
        "act_channel_max": data["act_channel_max"],
        "act_channel_mean": data["act_channel_mean"],
    }


def save_adaln_stats(adaln_records: dict, output_dir: Path | None = None):
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    flat = {}
    for layer_name, ts_dict in adaln_records.items():
        for ts_key, arr in ts_dict.items():
            flat[f"{layer_name}/{ts_key}"] = arr
    path = out / "adaln_stats.npz"
    np.savez_compressed(path, **flat)
    logger.info("Saved adaLN stats to %s", path)


def save_config(
    prompts: list[str],
    seeds: list[int],
    registry: list[dict],
    output_dir: Path | None = None,
):
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        **{k: v for k, v in DIAG_CONFIG.items() if k != "seed_range"},
        "seeds": seeds,
        "prompts": prompts,
        "num_layers": len(registry),
        "layer_names": [e["name"] for e in registry],
    }
    path = out / "config.json"
    path.write_text(json.dumps(cfg, indent=2))
    logger.info("Saved config to %s", path)
