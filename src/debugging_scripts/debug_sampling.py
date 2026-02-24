"""
Debug the denoising process step by step.
python -m src.debug_sampling
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
from diffusionkit.mlx import DiffusionPipeline, CFGDenoiser

# Initialize
pipeline = DiffusionPipeline(
    shift=3.0,
    use_t5=True,
    model_version="argmaxinc/mlx-stable-diffusion-3-medium",
    low_memory_mode=False,
    a16=True,
    w16=True,
)
pipeline.check_and_load_models()

# Setup
prompt = "a photo of a cat"
seed = 42
cfg_weight = 7.5
num_steps = 50
latent_size = (64, 64)

mx.random.seed(seed)

# Encode
conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")

# Get noise
x_T = pipeline.get_empty_latent(*latent_size)
noise = pipeline.get_noise(seed, x_T)
sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)

# Cache modulation
pipeline.mmdit.cache_modulation_params(pooled, mx.array([float(s) for s in sigmas]))

# Scale noise
noise_scaled = pipeline.sampler.noise_scaling(
    sigmas[0], noise, x_T, pipeline.max_denoise(sigmas)
)

print(f"Initial noise stats:")
print(f"  Shape: {noise_scaled.shape}")
print(f"  Mean: {float(mx.mean(noise_scaled)):.6f}")
print(f"  Std: {float(mx.std(noise_scaled)):.6f}")

# Run denoising manually
denoiser = CFGDenoiser(pipeline)
x = noise_scaled
s_in = mx.ones([x.shape[0]])

print(f"\nDenoising for {len(sigmas)-1} steps...")

# Check a few steps
check_steps = [0, 10, 25, 49]

for i in range(len(sigmas) - 1):
    sigma = sigmas[i]
    
    if i in check_steps:
        print(f"\nStep {i} (sigma={float(sigma):.6f}):")
        print(f"  x mean: {float(mx.mean(x)):.6f}, std: {float(mx.std(x)):.6f}")
    
    # Denoise
    denoised = denoiser(x, sigma * s_in, sigma, 
                       conditioning=conditioning,
                       cfg_weight=cfg_weight,
                       pooled_conditioning=pooled)
    
    if i in check_steps:
        print(f"  denoised mean: {float(mx.mean(denoised)):.6f}, std: {float(mx.std(denoised)):.6f}")
    
    # Euler step
    dt = sigmas[i + 1] - sigma
    x = x + denoised * dt
    mx.eval(x)
    
    if i in check_steps:
        print(f"  x after step mean: {float(mx.mean(x)):.6f}, std: {float(mx.std(x)):.6f}")

# Final state
print(f"\nFinal latent (after {len(sigmas)-1} steps):")
print(f"  Shape: {x.shape}")
print(f"  Mean: {float(mx.mean(x)):.6f}")
print(f"  Std: {float(mx.std(x)):.6f}")
print(f"  Min: {float(mx.min(x)):.6f}")
print(f"  Max: {float(mx.max(x)):.6f}")

# Check if this is still noise-like
# Clean latents should have mean near 0 and std around 0.5-1.5
if abs(float(mx.mean(x))) > 0.5 or float(mx.std(x)) > 3.0:
    print("\n⚠️  WARNING: Final latent looks unusual!")
    print("This might still be noisy or incorrectly denoised")

# Try decoding
print(f"\n=== Decoding ===")
decoded = pipeline.decode_latents_to_image(x)
decoded_np = np.array(decoded)

print(f"Decoded shape: {decoded_np.shape}")
print(f"Decoded range: [{decoded_np.min():.3f}, {decoded_np.max():.3f}]")

# Save
if len(decoded_np.shape) == 4:
    decoded_np = decoded_np[0]
if decoded_np.shape[0] in [3, 4]:
    decoded_np = np.transpose(decoded_np, (1, 2, 0))

decoded_np = (decoded_np * 255).clip(0, 255).astype(np.uint8)
if decoded_np.shape[-1] > 3:
    decoded_np = decoded_np[:, :, :3]

Image.fromarray(decoded_np).save("debug_final.png")
print(f"\nSaved debug_final.png")

# Also try using pipeline's built-in generation
print(f"\n=== Comparing with Pipeline's Built-in Generation ===")
mx.random.seed(seed)

# Use the pipeline's sample method directly
from diffusionkit.mlx.sampler import EulerDiscreteScheduler

scheduler = EulerDiscreteScheduler(shift=3.0)
sigmas_builtin = scheduler.get_sigmas(num_steps)

# Run the built-in sampling
x_builtin = noise_scaled
for i in range(len(sigmas_builtin) - 1):
    sigma = sigmas_builtin[i]
    
    denoised = denoiser(x_builtin, sigma * s_in, sigma,
                       conditioning=conditioning,
                       cfg_weight=cfg_weight,
                       pooled_conditioning=pooled)
    
    # Calculate dt
    dt = sigmas_builtin[i + 1] - sigma
    x_builtin = x_builtin + denoised * dt
    mx.eval(x_builtin)

print(f"\nBuilt-in sampler final latent:")
print(f"  Mean: {float(mx.mean(x_builtin)):.6f}")
print(f"  Std: {float(mx.std(x_builtin)):.6f}")

decoded_builtin = pipeline.decode_latents_to_image(x_builtin)
decoded_builtin_np = np.array(decoded_builtin)

if len(decoded_builtin_np.shape) == 4:
    decoded_builtin_np = decoded_builtin_np[0]
if decoded_builtin_np.shape[0] in [3, 4]:
    decoded_builtin_np = np.transpose(decoded_builtin_np, (1, 2, 0))

decoded_builtin_np = (decoded_builtin_np * 255).clip(0, 255).astype(np.uint8)
if decoded_builtin_np.shape[-1] > 3:
    decoded_builtin_np = decoded_builtin_np[:, :, :3]

Image.fromarray(decoded_builtin_np).save("debug_builtin.png")
print(f"Saved debug_builtin.png")

print(f"\n=== Check both images ===")
print(f"debug_final.png - our sampling")
print(f"debug_builtin.png - built-in sampling")
print(f"\nIf built-in looks good but ours doesn't, there's an issue with our sampler")