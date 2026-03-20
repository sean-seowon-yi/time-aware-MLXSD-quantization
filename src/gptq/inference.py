"""Load GPTQ-quantized weights and generate images for comparison.

Usage:
    python -m src.gptq.inference \
        --gptq-dir gptq_output \
        --prompt "a cat sitting on a windowsill" \
        --output-dir gptq_comparison \
        --num-steps 30 --cfg-weight 4.0 --seed 42
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

import mlx.core as mx

from .utils import (
    _get_block_linears,
    full_path_to_poly_key,
    get_poly_alpha,
    _set_nested,
)


def load_gptq_weights(gptq_dir: Path):
    """Load all quantized weights and config from a GPTQ output directory.

    Returns (config, weights) where weights is {poly_key: (W_q_int, scales)}.
    """
    with open(gptq_dir / "config.json") as f:
        config = json.load(f)

    weights = {}
    weights_dir = gptq_dir / "weights"
    for npz_path in sorted(weights_dir.glob("mm*.npz")):
        data = np.load(npz_path)
        # Keys are like "mm0_img_attn_q_proj__weight_int", "mm0_img_attn_q_proj__scale"
        layer_keys = set(k.rsplit("__", 1)[0] for k in data.files)
        for poly_key in layer_keys:
            W_q_int = data[f"{poly_key}__weight_int"]
            scales = data[f"{poly_key}__scale"]
            weights[poly_key] = (W_q_int, scales)

    return config, weights


def patch_model_weights(pipeline, gptq_weights: dict):
    """Replace model linear weights with dequantized GPTQ weights.

    Args:
        pipeline: DiffusionPipeline instance.
        gptq_weights: {poly_key: (W_q_int, scales)} from load_gptq_weights.
    """
    mmdit = pipeline.mmdit
    patched = 0
    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            if poly_key not in gptq_weights:
                continue
            W_q_int, scales = gptq_weights[poly_key]
            W_dequant = W_q_int.astype(np.float32) * scales[:, None]
            layer.weight = mx.array(W_dequant, dtype=pipeline.activation_dtype)
            patched += 1

    print(f"Patched {patched} layers with GPTQ weights")


def compute_static_alphas(poly_schedule: dict, n_points: int = 200) -> dict:
    """Compute a single static alpha per layer: max of poly(sigma) across sigma range.

    Returns {poly_key: static_alpha}.
    """
    sigma_range = poly_schedule.get("sigma_range", [0.01, 1.0])
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_points)
    layers_dict = poly_schedule.get("layers", {})
    static_alphas = {}
    for poly_key, entry in layers_dict.items():
        values = [get_poly_alpha(entry, float(s)) for s in sigmas]
        static_alphas[poly_key] = float(max(values))
    return static_alphas


class _ActQuantHook:
    """Transparent proxy that fake-quantizes inputs using poly schedule + alpha_scale.

    Supports two modes:
    - Poly mode (default): alpha varies with sigma via polynomial evaluation
    - Static mode (static_alpha set): alpha is fixed regardless of sigma
    """

    def __init__(self, wrapped, poly_entry: Optional[dict], alpha_scale: float,
                 static_alpha: Optional[float] = None):
        self._wrapped = wrapped
        self._poly_entry = poly_entry
        self._alpha_scale = alpha_scale
        self._static_alpha = static_alpha
        self._sigma: Optional[float] = None

    def __call__(self, x):
        if self._static_alpha is not None:
            # Static mode: fixed alpha regardless of timestep
            alpha = self._alpha_scale * self._static_alpha
            scale = alpha / 127.0
            x = mx.clip(mx.round(x / scale), -127, 127) * scale
        elif self._poly_entry is not None and self._sigma is not None:
            # Poly mode: timestep-varying alpha
            alpha = self._alpha_scale * get_poly_alpha(self._poly_entry, self._sigma)
            scale = alpha / 127.0
            x = mx.clip(mx.round(x / scale), -127, 127) * scale
        return self._wrapped(x)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def set_sigma(self, sigma: float):
        self._sigma = sigma


def install_act_quant_hooks(pipeline, config: dict, poly_schedule: dict):
    """Install activation quantization hooks on all linears.

    Returns list of hooks for sigma updates.
    """
    mmdit = pipeline.mmdit
    alpha_scales = config.get("alpha_scales", {})
    layers_dict = poly_schedule.get("layers", {})
    hooks = []

    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            poly_entry = layers_dict.get(poly_key)
            alpha_scale = alpha_scales.get(poly_key, 1.0)
            hook = _ActQuantHook(layer, poly_entry, alpha_scale)
            _set_nested(block, full_path, hook)
            hooks.append(hook)

    print(f"Installed {len(hooks)} activation quantization hooks")
    return hooks


def install_static_act_quant_hooks(pipeline, config: dict, poly_schedule: dict):
    """Install static (timestep-agnostic) A8 activation quantization hooks.

    Uses max(poly_alpha(sigma)) across the sigma range as the fixed clipping
    range per layer, with alpha_scales from config.

    Returns list of hooks.
    """
    mmdit = pipeline.mmdit
    alpha_scales = config.get("alpha_scales", {})
    static_alphas = compute_static_alphas(poly_schedule)
    hooks = []

    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            static_alpha = static_alphas.get(poly_key, 1.0)
            alpha_scale = alpha_scales.get(poly_key, 1.0)
            hook = _ActQuantHook(layer, None, alpha_scale, static_alpha=static_alpha)
            _set_nested(block, full_path, hook)
            hooks.append(hook)

    print(f"Installed {len(hooks)} static activation quantization hooks")
    return hooks


def remove_act_quant_hooks(pipeline, hooks):
    """Remove activation quantization hooks, restoring original (patched) layers."""
    mmdit = pipeline.mmdit
    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            if isinstance(layer, _ActQuantHook):
                _set_nested(block, full_path, layer._wrapped)


def generate_comparison(
    prompt: str,
    gptq_dir: Path,
    poly_schedule_path: Path,
    output_dir: Path,
    num_steps: int = 30,
    cfg_weight: float = 4.0,
    seed: int = 42,
    latent_size: int = 64,
    quantize_activations: bool = True,
):
    """Generate FP16 baseline and GPTQ-quantized images side by side.

    Saves into output_dir/<prompt_slug>/: fp16.png, gptq.png, comparison.png
    """
    from diffusionkit.mlx import DiffusionPipeline
    from PIL import Image

    # Create prompt-specific subdirectory
    slug = re.sub(r"[^\w\s-]", "", prompt.lower())[:30].strip().replace(" ", "_")
    output_dir = Path(output_dir) / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("Loading pipeline...")
    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()

    # 1. FP16 baseline
    print(f"\nGenerating FP16 baseline (seed={seed})...")
    fp16_img, fp16_log = pipeline.generate_image(
        text=prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        latent_size=(latent_size, latent_size),
        seed=seed,
        verbose=False,
    )
    fp16_img.save(output_dir / "fp16.png")
    print(f"  Saved fp16.png ({fp16_log['total_time']:.1f}s)")

    # 2. Load and apply GPTQ weights
    print("\nApplying GPTQ weights...")
    config, gptq_weights = load_gptq_weights(gptq_dir)
    patch_model_weights(pipeline, gptq_weights)

    # 3. Optionally install activation quantization hooks
    hooks = None
    if quantize_activations:
        with open(poly_schedule_path) as f:
            poly_schedule = json.load(f)
        act_mode = config.get("act_quant_mode", "poly")
        if act_mode == "static":
            hooks = install_static_act_quant_hooks(pipeline, config, poly_schedule)
        else:
            hooks = install_act_quant_hooks(pipeline, config, poly_schedule)

    # 4. Generate GPTQ image
    # If using act quant hooks, we need to manually set sigma each step.
    # generate_image() doesn't expose per-step callbacks, so we use the
    # lower-level denoising loop.
    if hooks:
        print(f"\nGenerating GPTQ W{config['bits_w']}A8 image (seed={seed})...")
        gptq_img = _generate_with_sigma_hooks(
            pipeline, prompt, hooks, num_steps, cfg_weight,
            latent_size, seed,
        )
    else:
        print(f"\nGenerating GPTQ W{config['bits_w']} image (seed={seed})...")
        gptq_img, _ = pipeline.generate_image(
            text=prompt,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            latent_size=(latent_size, latent_size),
            seed=seed,
            verbose=False,
        )

    gptq_img.save(output_dir / "gptq.png")
    print(f"  Saved gptq.png")

    # 5. Side-by-side comparison
    comparison = Image.new("RGB", (fp16_img.width * 2, fp16_img.height))
    comparison.paste(fp16_img, (0, 0))
    comparison.paste(gptq_img, (fp16_img.width, 0))
    comparison.save(output_dir / "comparison.png")
    print(f"  Saved comparison.png (left=FP16, right=GPTQ)")

    # Cleanup
    if hooks:
        remove_act_quant_hooks(pipeline, hooks)

    return fp16_img, gptq_img


def _generate_with_sigma_hooks(
    pipeline, prompt, hooks, num_steps, cfg_weight, latent_size, seed,
):
    """Generate an image while updating sigma on act quant hooks each step."""
    from diffusionkit.mlx import CFGDenoiser
    from PIL import Image

    denoiser = CFGDenoiser(pipeline)

    conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
    mx.eval(conditioning, pooled)

    sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
    timesteps = pipeline.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
    denoiser.cache_modulation_params(pooled, timesteps)

    mx.random.seed(seed)
    latent_shape = (1, latent_size, latent_size, 16)
    noise = mx.random.normal(latent_shape).astype(pipeline.activation_dtype)
    x = pipeline.sampler.noise_scaling(
        sigmas[0], noise, mx.zeros(latent_shape), pipeline.max_denoise(sigmas)
    )
    mx.eval(x)

    for i in range(len(sigmas) - 1):
        sigma_val = float(sigmas[i])
        for h in hooks:
            h.set_sigma(sigma_val)

        denoised = denoiser(
            x, timesteps[i], sigmas[i],
            conditioning=conditioning, cfg_weight=cfg_weight,
        )
        d = (x - denoised) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
        mx.eval(x)

    # Reset modulation cache
    from .utils import _reset_modulation_cache
    _reset_modulation_cache(pipeline)

    # Decode latents to image
    x = pipeline.latent_format.process_out(x)
    decoded = pipeline.decode_latents_to_image(x)
    x = mx.concatenate(decoded, axis=0)
    x = (x * 255).astype(mx.uint8)
    mx.eval(x)

    return Image.fromarray(np.array(x))


def generate_poly_only_comparison(
    prompt: str,
    poly_schedule_path: Path,
    output_dir: Path,
    num_steps: int = 30,
    cfg_weight: float = 4.0,
    seed: int = 42,
    latent_size: int = 64,
):
    """Generate FP16 baseline vs W16A8 (poly clipping only, no weight quantization).

    Saves into output_dir/<prompt_slug>/: fp16.png, w16a8.png, comparison.png
    """
    from diffusionkit.mlx import DiffusionPipeline
    from PIL import Image

    slug = re.sub(r"[^\w\s-]", "", prompt.lower())[:30].strip().replace(" ", "_")
    output_dir = Path(output_dir) / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline...")
    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()

    # 1. FP16 baseline
    print(f"\nGenerating FP16 baseline (seed={seed})...")
    fp16_img, fp16_log = pipeline.generate_image(
        text=prompt,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        latent_size=(latent_size, latent_size),
        seed=seed,
        verbose=False,
    )
    fp16_img.save(output_dir / "fp16.png")
    print(f"  Saved fp16.png ({fp16_log['total_time']:.1f}s)")

    # 2. Install poly activation hooks (alpha_scale=1.0 for all layers, no GPTQ)
    with open(poly_schedule_path) as f:
        poly_schedule = json.load(f)

    # Build a dummy config with all alpha_scales = 1.0
    dummy_config = {"alpha_scales": {}}
    hooks = install_act_quant_hooks(pipeline, dummy_config, poly_schedule)

    # 3. Generate W16A8 image
    print(f"\nGenerating W16A8 (poly clipping only) image (seed={seed})...")
    poly_img = _generate_with_sigma_hooks(
        pipeline, prompt, hooks, num_steps, cfg_weight, latent_size, seed,
    )
    poly_img.save(output_dir / "w16a8.png")
    print(f"  Saved w16a8.png")

    # 4. Side-by-side
    comparison = Image.new("RGB", (fp16_img.width * 2, fp16_img.height))
    comparison.paste(fp16_img, (0, 0))
    comparison.paste(poly_img, (fp16_img.width, 0))
    comparison.save(output_dir / "comparison.png")
    print(f"  Saved comparison.png (left=FP16, right=W16A8)")

    remove_act_quant_hooks(pipeline, hooks)
    return fp16_img, poly_img


def main():
    parser = argparse.ArgumentParser(
        description="Generate FP16 vs quantized comparison images",
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--gptq-dir", type=Path, default=Path("gptq_output"))
    parser.add_argument("--poly-schedule", type=Path,
                        default=Path("polynomial_clipping_schedule.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("gptq_comparison"))
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--no-act-quant", action="store_true",
                        help="Disable activation quantization (weight-only)")
    parser.add_argument("--poly-only", action="store_true",
                        help="W16A8: poly activation clipping only, no weight quantization")
    args = parser.parse_args()

    if args.poly_only:
        generate_poly_only_comparison(
            prompt=args.prompt,
            poly_schedule_path=args.poly_schedule,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            cfg_weight=args.cfg_weight,
            seed=args.seed,
            latent_size=args.latent_size,
        )
    else:
        generate_comparison(
            prompt=args.prompt,
            gptq_dir=args.gptq_dir,
            poly_schedule_path=args.poly_schedule,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            cfg_weight=args.cfg_weight,
            seed=args.seed,
            latent_size=args.latent_size,
            quantize_activations=not args.no_act_quant,
        )


if __name__ == "__main__":
    main()
