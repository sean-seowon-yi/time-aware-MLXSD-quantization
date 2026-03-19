"""Evaluation utilities for quantized models."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import mlx.core as mx


def compute_per_timestep_mse(
    pipeline_quant,
    pipeline_fp,
    cali_data: dict,
    n_samples: int = 50,
) -> Dict[float, float]:
    """Compute noise prediction MSE per timestep: MSE(eps_FP, eps_quant).

    Args:
        pipeline_quant: Pipeline with quantized MMDiT
        pipeline_fp: Pipeline with FP MMDiT
        cali_data: Calibration data dict
        n_samples: Number of samples to evaluate

    Returns:
        Dict mapping timestep -> MSE value
    """
    xs = cali_data["xs"]
    ts = cali_data["ts"]
    cs = cali_data["cs"]
    cs_pooled = cali_data["cs_pooled"]
    prompt_indices = cali_data["prompt_indices"]

    n = min(n_samples, len(xs))
    mse_by_ts: Dict[float, List[float]] = defaultdict(list)

    for i in range(n):
        t_val = float(ts[i])
        prompt_idx = int(prompt_indices[i])

        x_single = mx.array(xs[i][None, ...]).astype(pipeline_fp.activation_dtype)
        x_doubled = mx.concatenate([x_single] * 2, axis=0)
        t_mx = mx.array([t_val], dtype=pipeline_fp.activation_dtype)
        t_broadcast = mx.broadcast_to(t_mx, [2])

        cond_mx = mx.array(cs[prompt_idx]).astype(pipeline_fp.activation_dtype)
        pooled_mx = mx.array(cs_pooled[prompt_idx]).astype(pipeline_fp.activation_dtype)

        # Cache modulation params for both models
        for p in [pipeline_fp, pipeline_quant]:
            p.mmdit.cache_modulation_params(
                pooled_text_embeddings=pooled_mx,
                timesteps=mx.array([t_val], dtype=p.activation_dtype),
            )

        # FP forward
        out_fp = pipeline_fp.mmdit(
            latent_image_embeddings=x_doubled,
            token_level_text_embeddings=mx.expand_dims(cond_mx, 2),
            timestep=t_broadcast,
        )

        # Quantized forward
        out_quant = pipeline_quant.mmdit(
            latent_image_embeddings=x_doubled,
            token_level_text_embeddings=mx.expand_dims(cond_mx, 2),
            timestep=t_broadcast,
        )

        mse = mx.mean((out_fp - out_quant) ** 2)
        mx.eval(mse)
        mse_by_ts[t_val].append(mse.item())

        # Reload adaLN weights
        for p in [pipeline_fp, pipeline_quant]:
            p.mmdit.load_weights(
                p.load_mmdit(only_modulation_dict=True), strict=False
            )

    # Average per timestep
    result = {t: np.mean(vals) for t, vals in sorted(mse_by_ts.items())}
    return result


def generate_images(
    pipeline,
    prompts: List[str],
    output_dir: str,
    seed: int = 42,
    num_steps: int = 30,
    cfg_weight: float = 4.0,
    height: int = 512,
    width: int = 512,
):
    """Generate images using the (quantized) pipeline.

    Saves PNG images to output_dir.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        image = pipeline.generate_image(
            text=prompt,
            cfg_weight=cfg_weight,
            num_steps=num_steps,
            seed=seed + i,
            height=height,
            width=width,
        )
        image.save(str(out / f"image_{i:03d}.png"))
        print(f"  Generated image {i}: {prompt[:50]}...")

    print(f"Saved {len(prompts)} images to {output_dir}")
