"""Post-quantization diagnostics: collect W4A8 activation stats and compare
against Phase 1 FP16 baselines.

Reuses Phase 1's ChannelStatsCollector + hook infrastructure (adapted for
W4A8Linear modules) and adds weight-error analysis that compares original
FP16 weights against dequantized 4-bit weights.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..phase1.hooks import ChannelStatsCollector
from .quantize import W4A8Linear

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry builder for quantized model
# ---------------------------------------------------------------------------

def build_quantized_registry(mmdit) -> list[dict]:
    """Walk the quantized MMDiT and build a registry analogous to Phase 1.

    For W4A8Linear modules, d_in comes from qlinear; for unquantized nn.Linear
    modules, it comes from weight.shape[1].
    """
    registry = []

    for bidx, block in enumerate(mmdit.multimodal_transformer_blocks):
        skip_text_post = block.text_transformer_block.skip_post_sdpa

        for side, tb in [
            ("image", block.image_transformer_block),
            ("text", block.text_transformer_block),
        ]:
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                layer = getattr(tb.attn, proj_name)
                d_in = _get_d_in(layer)
                registry.append({
                    "name": f"blocks.{bidx}.{side}.attn.{proj_name}",
                    "module": layer,
                    "block": bidx,
                    "family": proj_name,
                    "side": side,
                    "d_in": d_in,
                    "quantized": isinstance(layer, W4A8Linear),
                })

            if not (side == "text" and skip_text_post):
                o_proj = getattr(tb.attn, "o_proj")
                if not isinstance(o_proj, nn.Identity):
                    registry.append({
                        "name": f"blocks.{bidx}.{side}.attn.o_proj",
                        "module": o_proj,
                        "block": bidx,
                        "family": "o_proj",
                        "side": side,
                        "d_in": _get_d_in(o_proj),
                        "quantized": isinstance(o_proj, W4A8Linear),
                    })

                for ff_name in ("fc1", "fc2"):
                    layer = getattr(tb.mlp, ff_name)
                    registry.append({
                        "name": f"blocks.{bidx}.{side}.mlp.{ff_name}",
                        "module": layer,
                        "block": bidx,
                        "family": ff_name,
                        "side": side,
                        "d_in": _get_d_in(layer),
                        "quantized": isinstance(layer, W4A8Linear),
                    })

    registry.append({
        "name": "context_embedder",
        "module": mmdit.context_embedder,
        "block": -1,
        "family": "context_embedder",
        "side": "shared",
        "d_in": mmdit.context_embedder.weight.shape[1],
        "quantized": False,
    })

    fl = mmdit.final_layer.linear
    registry.append({
        "name": "final_layer.linear",
        "module": fl,
        "block": -1,
        "family": "final_linear",
        "side": "image",
        "d_in": _get_d_in(fl),
        "quantized": isinstance(fl, W4A8Linear),
    })

    return registry


def _get_d_in(layer) -> int:
    if isinstance(layer, W4A8Linear):
        return layer.qlinear.weight.shape[1] * (32 // layer.qlinear.bits)
    return layer.weight.shape[1]


# ---------------------------------------------------------------------------
# Hook adapter for W4A8Linear
# ---------------------------------------------------------------------------

class QuantizedLinearHook:
    """Monkey-patches a W4A8Linear or nn.Linear so that __call__ records the
    input activation to a ChannelStatsCollector.
    """

    def __init__(self, module, name: str, collector: ChannelStatsCollector):
        self.name = name
        self.collector = collector
        self._original_cls = module.__class__

        outer = self
        original_call = module.__class__.__call__
        _dummy_w = mx.zeros((1, 1))

        def hooked_call(self_module, x):
            outer.collector.record(outer.name, x, _dummy_w)
            return original_call(self_module, x)

        module.__class__ = type(
            module.__class__.__name__ + "_Hooked",
            (module.__class__,),
            {"__call__": hooked_call},
        )
        self.module = module

    def remove(self):
        self.module.__class__ = self._original_cls


def install_quantized_hooks(
    registry: list[dict],
    collector: ChannelStatsCollector,
) -> list[QuantizedLinearHook]:
    hooks = []
    for entry in registry:
        hook = QuantizedLinearHook(entry["module"], entry["name"], collector)
        hooks.append(hook)
    logger.info("Installed %d hooks on quantized model", len(hooks))
    return hooks


def remove_quantized_hooks(hooks: list[QuantizedLinearHook]):
    for hook in hooks:
        hook.remove()
    logger.info("Removed %d hooks", len(hooks))


# ---------------------------------------------------------------------------
# Activation collection (reuses Phase 1 denoising loop on quantized model)
# ---------------------------------------------------------------------------

def run_quantized_collection(
    pipeline,
    seed_prompt_pairs: list[tuple[int, str]],
    collector: ChannelStatsCollector,
    num_steps: int = 30,
    latent_size: tuple[int, int] = (64, 64),
    cfg_weight: float = 4.0,
) -> None:
    """Run Euler denoising on the *quantized* pipeline, firing hooks every step.

    *seed_prompt_pairs* is a list of ``(seed, prompt)`` tuples.
    """
    use_cfg = cfg_weight > 0
    total_runs = len(seed_prompt_pairs)

    from mlx.utils import tree_flatten
    _adaln_cache = [
        (k, v) for k, v in tree_flatten(pipeline.mmdit.parameters())
        if "adaLN" in k
    ]

    for run_idx, (seed, prompt) in enumerate(seed_prompt_pairs, 1):
        conditioning, pooled_conditioning = pipeline.encode_text(
            prompt, cfg_weight=cfg_weight,
        )
        if not use_cfg:
            conditioning = conditioning[:1]
            pooled_conditioning = pooled_conditioning[:1]
        conditioning = conditioning.astype(pipeline.activation_dtype)
        pooled_conditioning = pooled_conditioning.astype(pipeline.activation_dtype)
        mx.eval(conditioning, pooled_conditioning)
        batch_size = conditioning.shape[0]

        t0 = time.time()
        logger.info(
            "Run %d/%d  prompt=%d  seed=%d  cfg=%.1f",
            run_idx, total_runs, run_idx - 1, seed, cfg_weight,
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
                prompt_id=str(run_idx - 1),
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
        if hasattr(pipeline.mmdit, "to_offload"):
            delattr(pipeline.mmdit, "to_offload")

        elapsed = time.time() - t0
        logger.info(
            "  completed in %.1fs  (hook calls so far: %d)",
            elapsed, collector.call_count,
        )


# ---------------------------------------------------------------------------
# Weight error analysis (FP16 vs dequantized W4)
# ---------------------------------------------------------------------------

def compute_weight_errors(
    fp16_weight_stats: dict,
    quantized_registry: list[dict],
) -> list[dict]:
    """Compare FP16 weights against dequantized W4 weights.

    Returns per-layer error metrics: MSE, max error, SNR, and channel-wise
    error distribution.
    """
    results = []

    for entry in quantized_registry:
        name = entry["name"]
        module = entry["module"]
        fp16_stats = fp16_weight_stats.get(name)
        if fp16_stats is None:
            continue

        if not entry["quantized"]:
            results.append({
                "name": name,
                "family": entry["family"],
                "side": entry["side"],
                "block": entry["block"],
                "quantized": False,
                "mse": 0.0,
                "max_error": 0.0,
                "snr_db": None,
                "channel_mse": np.zeros(entry["d_in"]),
            })
            continue

        W_deq = np.array(mx.dequantize(
            module.qlinear.weight,
            module.qlinear.scales,
            module.qlinear.biases,
            module.qlinear.group_size,
            module.qlinear.bits,
        ).astype(mx.float32))

        fp16_max = fp16_stats["w_channel_max"]
        deq_abs = np.abs(W_deq)
        deq_channel_max = np.max(deq_abs, axis=0)

        channel_mse = (fp16_max - deq_channel_max) ** 2
        mse = float(channel_mse.mean())
        max_error = float(np.abs(fp16_max - deq_channel_max).max())
        signal_power = float((fp16_max ** 2).mean())
        snr_db = 10.0 * np.log10(signal_power / (mse + 1e-12))

        results.append({
            "name": name,
            "family": entry["family"],
            "side": entry["side"],
            "block": entry["block"],
            "quantized": True,
            "mse": mse,
            "max_error": max_error,
            "snr_db": float(snr_db),
            "channel_mse": channel_mse,
        })

    results.sort(key=lambda r: r["mse"], reverse=True)
    logger.info("Computed weight errors for %d layers", len(results))
    return results


# ---------------------------------------------------------------------------
# Activation comparison (FP16 Phase 1 vs W4A8 trajectories)
# ---------------------------------------------------------------------------

def compare_activation_trajectories(
    fp16_act_dir: Path,
    w4a8_act_dir: Path,
    layer_names: list[str],
) -> list[dict]:
    """Compare FP16 vs W4A8 activation trajectories per layer.

    Returns per-layer comparison metrics.
    """
    results = []

    for name in layer_names:
        fp16_path = fp16_act_dir / f"{name}.npz"
        w4a8_path = w4a8_act_dir / f"{name}.npz"

        if not fp16_path.exists() or not w4a8_path.exists():
            continue

        fp16_data = np.load(fp16_path)
        w4a8_data = np.load(w4a8_path)

        fp16_traj = fp16_data["act_channel_max"]
        w4a8_traj = w4a8_data["act_channel_max"]

        n_steps = min(fp16_traj.shape[0], w4a8_traj.shape[0])
        fp16_traj = fp16_traj[:n_steps]
        w4a8_traj = w4a8_traj[:n_steps]

        diff = w4a8_traj - fp16_traj
        per_step_mse = np.mean(diff ** 2, axis=1)
        per_channel_mse = np.mean(diff ** 2, axis=0)

        fp16_power = np.mean(fp16_traj ** 2, axis=1)
        per_step_snr = 10.0 * np.log10(fp16_power / (per_step_mse + 1e-12))

        fp16_mean_per_step = fp16_traj.mean(axis=1)
        w4a8_mean_per_step = w4a8_traj.mean(axis=1)
        relative_shift = (w4a8_mean_per_step - fp16_mean_per_step) / (fp16_mean_per_step + 1e-12)

        results.append({
            "name": name,
            "fp16_traj": fp16_traj,
            "w4a8_traj": w4a8_traj,
            "per_step_mse": per_step_mse,
            "per_channel_mse": per_channel_mse,
            "per_step_snr": per_step_snr,
            "overall_mse": float(np.mean(diff ** 2)),
            "overall_snr": float(10.0 * np.log10(
                np.mean(fp16_traj ** 2) / (np.mean(diff ** 2) + 1e-12)
            )),
            "mean_relative_shift": float(np.mean(np.abs(relative_shift))),
        })

    results.sort(key=lambda r: r["overall_mse"], reverse=True)
    logger.info("Compared activation trajectories for %d layers", len(results))
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_quantized_activation_stats(
    collector: ChannelStatsCollector,
    registry: list[dict],
    output_dir: Path,
):
    """Save W4A8 activation trajectories (same format as Phase 1)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_values = collector.sigma_values()

    count = 0
    for entry in registry:
        name = entry["name"]
        if name not in collector.layer_names():
            continue
        trajectory = collector.get_trajectory(name)
        mean_trajectory = collector.get_mean_trajectory(name)

        path = output_dir / f"{name}.npz"
        np.savez_compressed(
            path,
            sigma_values=sigma_values,
            act_channel_max=trajectory,
            act_channel_mean=mean_trajectory,
        )
        count += 1
    logger.info("Saved W4A8 activation stats: %d files → %s", count, output_dir)


def save_weight_errors(errors: list[dict], output_dir: Path):
    """Save weight error analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    flat = {}
    summary = []
    for entry in errors:
        flat[f"{entry['name']}/channel_mse"] = entry["channel_mse"]
        summary.append({
            "name": entry["name"],
            "family": entry["family"],
            "side": entry["side"],
            "block": entry["block"],
            "quantized": entry["quantized"],
            "mse": entry["mse"],
            "max_error": entry["max_error"],
            "snr_db": entry["snr_db"],
        })

    np.savez_compressed(output_dir / "weight_errors.npz", **flat)
    (output_dir / "weight_error_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("Saved weight errors to %s", output_dir)


def save_activation_comparison(comparisons: list[dict], output_dir: Path):
    """Save activation comparison results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for entry in comparisons:
        name = entry["name"]
        np.savez_compressed(
            output_dir / f"{name}.npz",
            per_step_mse=entry["per_step_mse"],
            per_channel_mse=entry["per_channel_mse"],
            per_step_snr=entry["per_step_snr"],
        )
        summary.append({
            "name": name,
            "overall_mse": entry["overall_mse"],
            "overall_snr": entry["overall_snr"],
            "mean_relative_shift": entry["mean_relative_shift"],
        })

    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("Saved activation comparisons to %s", output_dir)
