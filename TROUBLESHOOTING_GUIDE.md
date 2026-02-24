# Troubleshooting Guide

Complete reference for all issues encountered and their solutions.

---

## Issue Index

1. [Storage Explosion from Conditioning Data](#issue-1-storage-explosion)
2. [MLX Missing PyTorch Hooks](#issue-2-mlx-hooks)
3. [MLX Missing no_grad](#issue-3-mlx-no-grad)
4. [Model State Corruption (adaLN Overwrite)](#issue-4-model-corruption)
5. [VAE Decode Method Discovery](#issue-5-vae-decode)
6. [All Images Are Noise](#issue-6-images-noise)
7. [Wrong Euler Formula](#issue-7-euler-formula)
8. [Missing append_dims](#issue-8-append-dims)
9. [Manifest Not Recording All Images](#issue-9-manifest-incomplete)
10. [Out of Memory During Collection](#issue-10-out-of-memory)
11. [Block Hooks Corrupt adaLN Weights on Reload](#issue-11-hook-corruption)

---

<a name="issue-1-storage-explosion"></a>
## Issue 1: Storage Explosion from Conditioning Data

### Symptom
```
5 images generate 5.3 GB of data
10 images would be 10+ GB
```

### Cause
Storing text conditioning (19 MB) in every calibration sample.

### Solution
Store only latents, regenerate conditioning from prompts:

```python
# Don't save conditioning
save_data = {
    'x': np.array(x),
    'timestep': np.array(timestep),
    'image_id': np.int32(img_idx),  # Reference to prompt in manifest
}

# Regenerate when needed
prompt = manifest['images'][img_idx]['prompt']
conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
```

**Result**: 99% storage reduction

---

<a name="issue-2-mlx-hooks"></a>
## Issue 2: MLX Missing PyTorch Hooks

### Symptom
```python
AttributeError: 'Linear' object has no attribute 'register_forward_hook'
```

### Cause
MLX doesn't support PyTorch-style hooks.

### Solution
Use monkey-patching instead:

```python
# Save original
original_call = layer.__call__

# Create wrapper
def wrapper(*args, **kwargs):
    output = original_call(*args, **kwargs)
    activations[layer_name] = np.array(output)
    return output

# Apply patch
layer.__call__ = wrapper
```

---

<a name="issue-3-mlx-no-grad"></a>
## Issue 3: MLX Missing no_grad

### Symptom
```python
AttributeError: module 'mlx.core' has no attribute 'no_grad'
```

### Cause
MLX doesn't track gradients by default.

### Solution
Simply remove `no_grad()` - not needed in MLX.

---

<a name="issue-4-model-corruption"></a>
## Issue 4: Model State Corruption (adaLN Overwrite)

### Symptom
```
[1/10] Image generated successfully ✓
[2/10] ERROR: ValueError in cache_modulation_params
  Last dimension mismatch: (2,1,1,1536) vs (0,1)
```

### Cause
`CFGDenoiser.cache_modulation_params()` overwrites the adaLN weights in the live
model. When a second image starts, those weights are stale/zeroed.

### Solution
After each image's forward passes, reload only the adaLN (modulation) weights —
do **not** reload the entire pipeline:

```python
for img_idx in range(num_images):
    # ... run forward passes for this image ...

    # Reload adaLN weights before next image
    pipeline.mmdit.load_weights(
        pipeline.load_mmdit(only_modulation_dict=True), strict=False
    )
```

This takes milliseconds vs ~10s for a full pipeline reload.

> **Note**: The earlier workaround of reloading the whole pipeline per image also
> works but is unnecessary. The targeted reload above is the correct fix.

---

<a name="issue-5-vae-decode"></a>
## Issue 5: VAE Decode Method Discovery

### Symptom
```python
AttributeError: 'DiffusionPipeline' object has no attribute 'vae'
AttributeError: 'DiffusionPipeline' object has no attribute 'decode'
```

### Investigation
```python
# Check available methods
print([m for m in dir(pipeline) if 'decode' in m.lower()])
# Found: decode_latents_to_image
```

### Solution
```python
# Correct method
image = pipeline.decode_latents_to_image(latent)
```

---

<a name="issue-6-images-noise"></a>
## Issue 6: All Images Are Noise

### Symptom
Decoded images look like pure noise instead of content.

### Investigation
```python
# Verify DiffusionKit works
image, _ = pipeline.generate_image(text="cat", num_steps=50)
# ✓ Produces proper cat image

# Our code produces noise
# → Bug is in our sampling implementation
```

### Cause
Multiple bugs in Euler sampler (see issues #7, #8).

---

<a name="issue-7-euler-formula"></a>
## Issue 7: Wrong Euler Formula

### Symptom
Generated images are noise.

### Wrong Implementation
```python
denoised = denoiser(x, sigma * s_in, sigma, **extra_args)
dt = sigmas[i + 1] - sigma
x = x + denoised * dt  # ✗ Wrong!
```

### Correct Implementation
```python
# Convert sigmas to timesteps
timesteps = model.model.sampler.timestep(sigmas).astype(...)

# Use Karras ODE derivative
denoised = model(x, timesteps[i], sigmas[i], **extra_args)
d = to_d(x, sigmas[i], denoised)
dt = sigmas[i + 1] - sigmas[i]
x = x + d * dt  # ✓ Correct
```

### Solution
Read DiffusionKit's `sample_euler` source and match exactly.

---

<a name="issue-8-append-dims"></a>
## Issue 8: Missing append_dims

### Symptom
Shape mismatch or incorrect results.

### Cause
Sigma needs broadcasting to match tensor dimensions.

### Wrong Implementation
```python
def to_d(x, sigma, denoised):
    return (x - denoised) / sigma  # ✗ Wrong shape!
```

### Correct Implementation
```python
def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)  # ✓ Correct
```

---

<a name="issue-9-manifest-incomplete"></a>
## Issue 9: Manifest Not Recording All Images

### Symptom
Generated 10 images but manifest only shows 2.

### Cause
Manifest saved only at checkpoints (every 10 images).

### Solution
```python
# Set checkpoint frequency to 1
parser.add_argument("--checkpoint-every", type=int, default=1)

# Or rebuild manifest from files
python -m src.rebuild_manifest --calib-dir calibration_data
```

---

<a name="issue-10-out-of-memory"></a>
## Issue 10: Out of Memory During Collection

### Symptom
```
MemoryError during activation collection
Process killed
```

### Solutions

**Reduce batch size**:
```bash
python -m src.collect_layer_activations --num-images 50  # Instead of 100
```

**Clear cache more frequently**:
```python
# In activation collector
if step_idx % 10 == 0:
    mx.metal.clear_cache()
    import gc
    gc.collect()
```

**Use less timesteps**:
Modify `select_key_timesteps()` to return fewer steps.

---

<a name="issue-11-hook-corruption"></a>
## Issue 11: Block Hooks Corrupt adaLN Weights on Reload

### Symptom
Same dimension mismatch as Issue 4, but occurs even after adding the adaLN
reload, or adaLN weights silently reset to zero on the second image.

### Cause
`BlockHook` / `_HookedLayer` proxy objects hold references into the model's
module list. Calling `pipeline.mmdit.load_weights(...)` or `mx.metal.clear_cache()`
while hooks are still installed overwrites the adaLN parameters through those
stale references, corrupting the weights the reload just wrote.

### Solution
Always remove hooks **before** any `load_weights` or `clear_cache` call:

```python
# cache_adaround_data.py pattern
remove_block_hooks(pipeline.mmdit, hooks)   # ← remove first
mx.metal.clear_cache()
pipeline.mmdit.load_weights(
    pipeline.load_mmdit(only_modulation_dict=True), strict=False
)
```

```python
# collect_layer_activations.py pattern
remove_layer_hooks(pipeline.mmdit, hooks)   # ← remove first
pipeline.mmdit.load_weights(
    pipeline.load_mmdit(only_modulation_dict=True), strict=False
)
```

### Rule of thumb
**Install hooks → forward passes → flush → remove hooks → reload weights.**
Never reload weights with hooks still live.

---



### Verification Failed

```bash
# Run verification
python -m src.verify_calibration --calib-dir calibration_data

# If issues found, rebuild
python -m src.rebuild_manifest --calib-dir calibration_data
```

### Activation Collection Failed

```bash
# Force regenerate
python -m src.collect_layer_activations \
    --calib-dir calibration_data \
    --num-images 100 \
    --force
```

### Resume Interrupted Generation

```bash
python -m src.generate_calibration_data \
    --num-images 1000 \
    --num-steps 50 \
    --resume
```

---

## Prevention

### Before Starting

- Check disk space: `df -h`
- Check memory: `free -h` or `htop`
- Test with small dataset first (10 images)
- Use `screen` or `tmux` for long runs

### During Generation

- Monitor progress
- Check intermediate outputs
- Verify first few images look correct

### After Completion

- Run verification script
- Check file counts
- Spot-check random samples

---

## Getting Help

If issues persist:

1. Run verification: `python -m src.verify_calibration`
2. Check error messages carefully
3. Review relevant documentation section
4. Check if it's a known issue in this guide

---

## Issue Template

When reporting new issues, include:

```
Symptom: [What went wrong]
Command: [Exact command run]
Error: [Full error message]
Environment: [M1/M2/M4, RAM, disk space]
Files: [What files exist/don't exist]
```
