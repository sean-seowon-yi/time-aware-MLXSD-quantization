"""
Generate calibration data - FINAL WORKING VERSION.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple
import csv

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
from diffusionkit.mlx import DiffusionPipeline, CFGDenoiser


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def sample_euler_with_calibration(
    model: CFGDenoiser,
    x: mx.array,
    sigmas: mx.array,
    extra_args: dict,
    img_idx: int,
    samples_dir: Path,
) -> Tuple[mx.array, float]:
    """
    Euler sampler exactly matching DiffusionKit's implementation.
    """
    extra_args = {} if extra_args is None else extra_args
    
    # Convert sigmas to timesteps
    timesteps = model.model.sampler.timestep(sigmas).astype(
        model.model.activation_dtype
    )
    
    # Cache modulation once for all timesteps
    model.cache_modulation_params(extra_args.pop("pooled_conditioning"), timesteps)
    
    iter_start = time.time()
    
    for i in range(len(sigmas) - 1):
        # Save calibration sample
        save_data = {
            'x': np.array(x),
            'timestep': np.array(timesteps[i]),
            'sigma': np.array(sigmas[i]),
            'step_index': np.int32(i),
            'image_id': np.int32(img_idx),
            'is_final': np.bool_(False),
        }
        np.savez_compressed(
            samples_dir / f"{img_idx:04d}_{i:03d}.npz",
            **save_data
        )
        
        # Denoise
        denoised = model(x, timesteps[i], sigmas[i], **extra_args)
        
        # Karras ODE derivative
        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]
        
        # Euler step
        x = x + d * dt
        mx.eval(x)
    
    # Save final
    save_data = {
        'x': np.array(x),
        'timestep': np.array(timesteps[-1]),
        'sigma': np.array(sigmas[-1]),
        'step_index': np.int32(len(sigmas) - 1),
        'image_id': np.int32(img_idx),
        'is_final': np.bool_(True),
    }
    np.savez_compressed(
        samples_dir / f"{img_idx:04d}_{len(sigmas)-1:03d}.npz",
        **save_data
    )
    
    # Clear cache
    model.clear_cache()
    
    iter_time = time.time() - iter_start
    return x, iter_time


def load_prompts(csv_path: Path, max_count: int) -> List[str]:
    """Load prompts from CSV."""
    prompts = []
    
    if not csv_path.exists():
        return [
            "a photo of a cat",
            "abstract art with vibrant colors",
            "a landscape with mountains",
        ][:max_count]
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(prompts) >= max_count:
                break
            p = row.get("prompt", "").strip()
            if p:
                prompts.append(p)
    
    return prompts


def initialize_pipeline():
    """Initialize a fresh pipeline."""
    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    return pipeline


def generate_with_calibration(pipeline, prompt: str, seed: int, num_steps: int,
                              cfg_weight: float, latent_size: tuple, 
                              img_idx: int, samples_dir: Path):
    """Generate image with calibration data."""
    mx.random.seed(seed)
    
    # Encode text
    conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
    mx.eval(conditioning)
    mx.eval(pooled)
    
    conditioning = conditioning.astype(pipeline.activation_dtype)
    pooled = pooled.astype(pipeline.activation_dtype)
    
    # Setup
    x_T = pipeline.get_empty_latent(*latent_size)
    noise = pipeline.get_noise(seed, x_T)
    sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
    
    extra_args = {
        "conditioning": conditioning,
        "cfg_weight": cfg_weight,
        "pooled_conditioning": pooled,
    }
    
    noise_scaled = pipeline.sampler.noise_scaling(
        sigmas[0], noise, x_T, pipeline.max_denoise(sigmas)
    )
    
    # Sample
    latent, iter_time = sample_euler_with_calibration(
        CFGDenoiser(pipeline),
        noise_scaled,
        sigmas,
        extra_args,
        img_idx,
        samples_dir
    )
    
    # Process and decode
    latent = pipeline.latent_format.process_out(latent)
    mx.eval(latent)
    latent = latent.astype(pipeline.activation_dtype)
    decoded = pipeline.decode_latents_to_image(latent)
    mx.eval(decoded)
    
    x_img = mx.concatenate([decoded], axis=0)
    x_img = (x_img * 255).astype(mx.uint8)
    pil_image = Image.fromarray(np.array(x_img[0]))
    
    return pil_image, latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-weight", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument("--prompt-csv", type=Path, default=None)
    args = parser.parse_args()
    
    calib_dir = args.calib_dir or (_REPO / "calibration_data")
    prompt_csv = args.prompt_csv or (_REPO / "all_prompts.csv")
    
    calib_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = calib_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    images_dir = calib_dir / "images"
    images_dir.mkdir(exist_ok=True)
    latents_dir = calib_dir / "latents"
    latents_dir.mkdir(exist_ok=True)
    
    prompts = load_prompts(prompt_csv, args.num_images)
    
    print(f"=== Generating {len(prompts)} Images ===\n")
    
    images_metadata = []
    start_time = time.time()
    
    for img_idx, prompt in enumerate(prompts):
        seed = args.seed + img_idx
        
        print(f"[{img_idx + 1}/{len(prompts)}] {prompt[:60]}...")
        
        img_start = time.time()
        
        print(f"  Loading pipeline...", end=" ", flush=True)
        pipeline = initialize_pipeline()
        print("✓")
        
        print(f"  Generating...", end=" ", flush=True)
        image, final_latent = generate_with_calibration(
            pipeline=pipeline,
            prompt=prompt,
            seed=seed,
            num_steps=args.num_steps,
            cfg_weight=args.cfg_weight,
            latent_size=(64, 64),
            img_idx=img_idx,
            samples_dir=samples_dir
        )
        print("✓")
        
        image_path = images_dir / f"{img_idx:04d}.png"
        image.save(image_path)
        np.save(latents_dir / f"{img_idx:04d}.npy", np.array(final_latent))
        
        images_metadata.append({
            'image_id': img_idx,
            'prompt': prompt,
            'seed': seed,
            'cfg_weight': args.cfg_weight,
            'num_steps': args.num_steps,
            'filename': f"{img_idx:04d}.png",
            'latent_filename': f"{img_idx:04d}.npy",
        })
        
        del pipeline
        
        img_time = time.time() - img_start
        print(f"  Total: {img_time:.1f}s\n")
    
    manifest = {
        "n_completed": len(prompts),
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_weight,
        "latent_size": [64, 64],
        "prompt_path": str(prompt_csv),
        "num_images": len(prompts),
        "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
        "use_t5": True,
        "seed_base": args.seed,
        "images_saved": True,
        "latents_saved": True,
        "images": images_metadata,
    }
    
    with open(calib_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"=== Complete ===")
    print(f"✓ {len(prompts)} images")
    print(f"✓ {len(prompts) * args.num_steps} calibration samples")
    print(f"  Time: {total_time / 60:.1f} min")
    print(f"  Images: {images_dir}")


if __name__ == "__main__":
    main()