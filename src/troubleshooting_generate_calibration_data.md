Troubleshooting Guide: DiffusionKit Calibration Data Generation
This document outlines all the issues encountered and fixes applied while implementing calibration data generation for TaQ-DiT quantization with DiffusionKit on MLX.

Overview
Goal: Generate calibration data for TaQ-DiT quantization, including:

Per-step latent states during denoising
Final decoded images for FID evaluation
Metadata (prompts, seeds, timesteps)

Challenges: Multiple issues with DiffusionKit's API, model state corruption, incorrect sampling implementation, and VAE decoding problems.

Lessons Learned: 
***********************************************

TL;DR
Key phrases to use:

"Read the source first"
"Check how [library] actually implements this"
"Show me the source code for [function]"
"Before suggesting anything, inspect..."
"Let's look at the actual implementation"

And always provide:

File paths where the source lives
Specific functions/classes to check
Permission to use view tool on your files

This shifts me from "guess based on common patterns" to "verify with actual code" mode.



Issue 1: Storage Explosion from Conditioning Data
Problem
Initial implementation saved text conditioning (19 MB) in every calibration sample:
python# Per sample: x (256 KB) + conditioning (19 MB) + pooled (16 KB) = ~19.3 MB
# 10 images × 51 steps × 19.3 MB = ~10 GB
Result: 5 images generated 5.3 GB of data.
Root Cause
Text conditioning is identical across all steps of the same image but was being saved redundantly.
Solution
Store conditioning separately, reference by image_id:
python# In each sample, store only:
{
    'x': latent,           # 256 KB
    'timestep': timestep,  # 4 bytes
    'sigma': sigma,        # 4 bytes
    'image_id': img_idx,   # 4 bytes - reference to conditioning
}
# Total: ~256 KB per sample

# Store conditioning once per image in separate directory
conditioning/{img_id}_conditioning.npz
Storage reduction: 10 GB → 130 MB (98% reduction)
Further Optimization
Realized conditioning can be regenerated from prompts stored in manifest.json:
python# No separate conditioning files needed
# Just store prompts in manifest, regenerate on load
conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
Final storage: ~13 MB per 10 images (99.9% reduction)

Issue 2: MLX Doesn't Support PyTorch-Style Hooks
Problem
Initial attempt to collect layer statistics used PyTorch's register_forward_hook:
pythonhook = module.register_forward_hook(make_stats_hook(name))
```

**Error**: 
```
AttributeError: 'Linear' object has no attribute 'register_forward_hook'
Root Cause
MLX doesn't have PyTorch's hook system for intercepting layer outputs.
Attempted Solutions
1. Monkey-patching layer __call__ methods (Failed)
pythonoriginal_call = module.__call__
module.__call__ = wrapped_call.__get__(module, type(module))
Led to method binding issues and state corruption.
2. Separate statistics collection (Skipped for now)
Decided to collect basic calibration data first, add detailed layer statistics later if needed for advanced quantization beyond TaQ-DiT.
Decision
Focus on per-step latent states first. Layer activation statistics can be added later by running additional forward passes on saved samples.

Issue 3: MLX Doesn't Have no_grad() Context
Problem
Code included PyTorch's no_grad() context manager:
pythonwith mx.no_grad():
    output = model(x)
```

**Error**:
```
AttributeError: module 'mlx.core' has no attribute 'no_grad'
Root Cause
MLX doesn't track gradients by default, so no_grad() isn't needed.
Solution
Simply remove the context manager:
python# MLX doesn't need no_grad - it doesn't track gradients by default
output = model(x)

Issue 4: Model State Corruption Between Images
Problem
After generating the first image successfully, the second image failed:
python[1/10] Image generated successfully ✓
[2/10] ERROR: ValueError in cache_modulation_params
  Last dimension mismatch: (2,1,1,1536) vs (0,1)
Root Cause
DiffusionKit's cache_modulation_params stores cached AdaLN parameters in model state. After the first image, these cached weights became corrupted, likely due to:

Memory not being properly released
State from previous timesteps interfering
Internal caching mechanism not being cleared properly

Solution
Reload the entire pipeline for each image:
pythonfor img_idx in range(num_images):
    # Load fresh pipeline
    pipeline = initialize_pipeline()
    
    # Generate image
    image = generate(pipeline, ...)
    
    # Clean up
    del pipeline
Trade-off: Slower (~10s overhead per image from model loading) but reliable.
Alternative attempted: Tried pipeline.mmdit._modulation_cache = {} to clear cache, but internal state corruption persisted.

Issue 5: VAE Decode Method Discovery
Problem
Initially couldn't find the correct VAE decode method:
pythonpipeline.vae.decode(latent)      # ✗ No 'vae' attribute
pipeline.decode(latent)          # ✗ No 'decode' method
pipeline.vae_decode(latent)      # ✗ No 'vae_decode' method
Discovery Process
Step 1: Inspect available methods
pythonprint([m for m in dir(pipeline) if 'decode' in m.lower()])
# Found: decode_latents_to_image
Step 2: Test the method
pythonimage = pipeline.decode_latents_to_image(latent)  # ✓ Works!
Correct Usage
pythondecoded = pipeline.decode_latents_to_image(latent)
decoded_np = np.array(decoded)

# Handle dimensions
if len(decoded_np.shape) == 4:
    decoded_np = decoded_np[0]  # Remove batch
if decoded_np.shape[0] == 3:
    decoded_np = np.transpose(decoded_np, (1, 2, 0))  # [C,H,W] -> [H,W,C]

# Normalize to [0, 255]
decoded_np = (decoded_np * 255).clip(0, 255).astype(np.uint8)

Issue 6: All Generated Images Were Noise
Problem
Images decoded successfully but looked like pure noise instead of the intended content.
Investigation Process
Step 1: Verify DiffusionKit works
pythonimage, metadata = pipeline.generate_image(
    text="a photo of a cat",
    num_steps=50,
    cfg_weight=7.5,
    seed=42,
)
# Result: Perfect cat image ✓
Conclusion: DiffusionKit's built-in generation works, so the bug is in our custom sampling code.
Step 2: Compare latent statistics
python# Our latent:
Mean: 0.234, Std: 1.456, Range: [-2.1, 1.8]

# Working latent from pipeline.generate_image:
Mean: 0.189, Std: 1.234, Range: [-1.9, 1.6]
Latents looked similar, so the bug was in the sampling loop itself, not the setup.
Step 3: Read DiffusionKit's source code
Found the actual sample_euler implementation and discovered multiple bugs.

Issue 7: Incorrect Euler Sampling Implementation
Bug #1: Wrong Euler Step Formula
Incorrect implementation:
python# Our buggy code
denoised = denoiser(x, sigma * s_in, sigma, **extra_args)
dt = sigmas[i + 1] - sigma
x = x + denoised * dt  # ✗ Wrong!
Correct implementation (from DiffusionKit):
pythondenoised = model(x, timesteps[i], sigmas[i], **extra_args)
d = to_d(x, sigmas[i], denoised)  # Karras ODE derivative
dt = sigmas[i + 1] - sigma
x = x + d * dt  # ✓ Correct
Key difference: Need to compute the derivative d using the to_d function, not use denoised directly.
Bug #2: Missing append_dims for Broadcasting
The to_d function:
pythondef to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)
Without append_dims: Sigma shape mismatch
pythonx.shape = (1, 16, 64, 64)
sigma.shape = ()  # Scalar

# Without broadcasting:
(x - denoised) / sigma  # ✗ Shape error or incorrect broadcasting
With append_dims: Proper broadcasting
pythondef append_dims(x, target_dims):
    """Appends dimensions to match target tensor."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]

# Result: sigma.shape becomes (1, 1, 1, 1) for proper broadcasting
Bug #3: Using Sigmas Instead of Timesteps
Incorrect:
pythondenoised = denoiser(x, sigma * s_in, sigma, **extra_args)
Correct:
python# Convert sigmas to timesteps first
timesteps = model.model.sampler.timestep(sigmas).astype(
    model.model.activation_dtype
)

# Pass timesteps, not sigma * s_in
denoised = model(x, timesteps[i], sigmas[i], **extra_args)
Why: The model expects preprocessed timestep values, not raw sigma values.
Bug #4: Modulation Caching Strategy
Incorrect (our approach):
pythonfor i in range(len(sigmas) - 1):
    # Cache before each step
    pipeline.mmdit.cache_modulation_params(pooled, mx.array([sigma]))
    denoised = denoiser(...)
Correct (DiffusionKit's approach):
python# Cache ONCE for ALL timesteps before the loop
timesteps = model.model.sampler.timestep(sigmas).astype(...)
model.cache_modulation_params(extra_args.pop("pooled_conditioning"), timesteps)

for i in range(len(sigmas) - 1):
    # Just use the cached values
    denoised = model(x, timesteps[i], sigmas[i], **extra_args)
    
# Clear cache after the loop
model.clear_cache()
Why: Pre-caching all timesteps is more efficient and avoids state corruption issues.

Issue 8: VAE Scaling Factor Confusion
Initial Confusion
Tried various VAE scaling factors:

1.5305 (SD3 standard)
0.13025 (inverse)
0.18215 (SD1.5)
1.0 (no scaling)

Result: All produced noise.
Root Cause
The scaling factor wasn't the issue - the latents themselves were wrong due to the Euler sampling bugs above.
Solution
Once Euler sampling was fixed, decoding worked without any special scaling:
pythondecoded = pipeline.decode_latents_to_image(latent)  # No scaling needed!
DiffusionKit's decode_latents_to_image handles scaling internally.

Issue 9: Manifest Not Recording All Images
Problem
Generated 10 images but manifest.json only showed 2.
Root Cause
Manifest was only saved at checkpoints (every 10 images by default). If the script stopped before hitting the checkpoint, progress wasn't saved.
Solution
pythonparser.add_argument("--checkpoint-every", type=int, default=1,
                   help="Save progress every N images")
Changed default from 10 to 1, ensuring manifest updates after each image.
Additional Fix
Added atomic manifest writes to prevent corruption:
pythondef save_manifest(calib_dir: Path, manifest: Dict):
    """Save manifest with atomic write."""
    manifest_path = calib_dir / "manifest.json"
    temp_path = manifest_path.with_suffix('.json.tmp')
    
    with open(temp_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    temp_path.replace(manifest_path)  # Atomic on Unix

Complete Working Solution
Final Implementation
pythondef append_dims(x, target_dims):
    """Appends dimensions for broadcasting."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised):
    """Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

def sample_euler_with_calibration(model, x, sigmas, extra_args, img_idx, samples_dir):
    """Euler sampler matching DiffusionKit exactly."""
    
    # Convert sigmas to timesteps
    timesteps = model.model.sampler.timestep(sigmas).astype(
        model.model.activation_dtype
    )
    
    # Cache modulation ONCE for all timesteps
    model.cache_modulation_params(
        extra_args.pop("pooled_conditioning"), 
        timesteps
    )
    
    for i in range(len(sigmas) - 1):
        # Save calibration sample
        save_calibration(x, timesteps[i], sigmas[i], i, img_idx, samples_dir)
        
        # Denoise
        denoised = model(x, timesteps[i], sigmas[i], **extra_args)
        
        # Karras ODE derivative
        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]
        
        # Euler step
        x = x + d * dt
        mx.eval(x)
    
    # Save final sample
    save_calibration(x, timesteps[-1], sigmas[-1], len(sigmas)-1, img_idx, samples_dir, is_final=True)
    
    model.clear_cache()
    return x
Key Success Factors

Exact DiffusionKit matching: Read source code, implement exactly
Fresh pipeline per image: Avoid state corruption
Proper broadcasting: Use append_dims for sigma
Correct Euler formula: Use to_d function
Timestep conversion: Convert sigmas to timesteps
Single modulation cache: Cache once, not per step
No special VAE scaling: DiffusionKit handles it internally


Performance Characteristics
Final Metrics

Time per image: ~30-40 seconds (M4 Max)

Pipeline loading: ~10s
Text encoding: ~1-2s
Denoising (50 steps): ~20-25s
Image decoding: ~1s


Storage per image: ~1.3 MB

Calibration samples (51 steps): ~1.2 MB
Final latent: ~65 KB
Final image (PNG): ~2 MB


Total for 1000 images:

Time: ~10-12 hours
Storage: ~3.5 GB (samples + latents + images + manifest)




Lessons Learned

Read the source code first: Would have saved hours if we'd started by reading DiffusionKit's actual implementation
Test incrementally: Verify each component (encode → sample → decode) works before combining
MLX ≠ PyTorch: Don't assume PyTorch patterns work (hooks, no_grad, etc.)
Model state is fragile: When in doubt, reload
Storage optimization matters: 10 GB → 13 MB made the difference between feasible and infeasible for 1000 images
Broadcasting is critical: A missing append_dims can silently break everything
Trust the framework: DiffusionKit's built-in methods (like decode) handle complexity correctly


Tools for Debugging
Verify DiffusionKit Works
pythonimage, _ = pipeline.generate_image(
    text="test prompt",
    num_steps=50,
    cfg_weight=7.5,
    seed=42
)
# If this works, bug is in our code, not DiffusionKit
Inspect Latent Statistics
pythonprint(f"Mean: {float(mx.mean(latent)):.6f}")
print(f"Std: {float(mx.std(latent)):.6f}")
print(f"Range: [{float(mx.min(latent)):.6f}, {float(mx.max(latent)):.6f}]")

# Healthy latents: mean ~0, std ~0.5-1.5, range ~[-3, 3]
# Noise latents: uniform distribution, no structure
Compare With Working Implementation
python# Generate with DiffusionKit's built-in
good_image, metadata = pipeline.generate_image(...)

# Generate with our code
our_image = our_generate_function(...)

# Visual comparison immediately shows if our code works

Future Improvements

Resume capability: Already implemented via manifest tracking
Parallel generation: Could parallelize across multiple devices if available
Layer activation statistics: Add back in for advanced quantization methods beyond TaQ-DiT
Memory optimization: Could use low_memory_mode=True at cost of speed
Progress persistence: Currently checkpoint-based, could add SQLite for more granular tracking


References

DiffusionKit source: /DiffusionKit/python/src/diffusionkit/mlx/__init__.py
Karras et al. (2022): "Elucidating the Design Space of Diffusion-Based Generative Models"
TaQ-DiT: "TaQ-DiT: Time-Aware Quantization for Diffusion Transformers"