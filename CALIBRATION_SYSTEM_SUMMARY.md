# Calibration Data Collection System - Summary

## What We Built

A complete system for generating calibration data and collecting layer activation statistics for quantizing diffusion transformers with timestep-awareness (TaQ-DiT style).

## Key Design Decisions

### 1. **Two-Phase Approach**

**Phase 1: Generate Calibration Data (1000 images)**
- Purpose: Create evaluation dataset + basic calibration samples
- What's saved: Per-step latents, final images, prompts
- What's NOT saved: Conditioning (regenerate from prompts), layer activations
- Storage: ~3.3 GB
- Time: ~11 hours

**Phase 2: Collect Activation Statistics (100 images)**
- Purpose: Get layer-specific statistics for quantization
- Strategy: Subsample 100 images × 17 key timesteps = 1,700 samples
- Why subsample: Statistics stabilize with 50-100 diverse samples
- Storage: ~5 GB
- Time: ~30 minutes

### 2. **Smart Timestep Selection**

Not all 51 timesteps are equally important. Selected ~17 key timesteps:

```
Critical timesteps:
- Boundaries: 0, 50 (start/end)
- Quarters: 12, 25, 37
- Dense in middle: every 3 steps from 16-33 (structure formation)
- Sparse elsewhere: every 5 steps

Why: Captures all distribution phases without redundancy
```

### 3. **Storage Optimization**

**Problem**: Initial naive implementation generated 10 GB for just 10 images

**Solutions Applied**:
1. Don't store conditioning (19 MB per step) → regenerate from prompts
2. Subsample images for activation collection (100 instead of 1000)
3. Subsample timesteps for activation collection (17 instead of 51)

**Result**: 99% storage reduction

### 4. **Handling DiffusionKit Issues**

**Model State Corruption**:
- Problem: Cached AdaLN parameters corrupt between images
- Solution: Reload fresh pipeline for each image
- Trade-off: Slower (~10s overhead) but reliable

**No Hook Support in MLX**:
- Problem: Can't use PyTorch-style hooks for activation collection
- Solution: Monkey-patch layer `__call__` methods
- Trade-off: Hacky but works without modifying DiffusionKit source

**Euler Sampling Bugs**:
- Fixed: Proper `append_dims` for broadcasting
- Fixed: Convert sigmas to timesteps
- Fixed: Cache modulation once, not per step
- Fixed: Use correct `to_d` function

## What You Get

### Files Created

```
src/
├── generate_calibration_data.py        # Main generation script
├── collect_layer_activations.py        # Activation collection
├── analyze_activations.py              # Analysis & recommendations
├── verify_calibration.py               # Data integrity checks
└── rebuild_manifest.py                 # Manifest recovery

calibration_data/
├── samples/                            # 51,000 × .npz files
├── latents/                            # 1,000 × .npy files
├── images/                             # 1,000 × .png files
├── activations/
│   ├── layer_statistics.json          # Per-layer stats
│   └── collection_metadata.json       # Collection info
└── manifest.json                       # All prompts + metadata
```

### What Each File Does

**`generate_calibration_data.py`**:
- Generates 1000 images with per-step latent states
- Saves: latents, images, prompts
- Handles: Model corruption, resume capability
- Time: ~11 hours for 1000 images

**`collect_layer_activations.py`**:
- Collects activation statistics from 100 images
- Uses: Monkey-patching to intercept layer outputs
- Smart sampling: 17 key timesteps per image
- Time: ~30 minutes

**`analyze_activations.py`**:
- Analyzes collected statistics
- Suggests: Bit-widths per layer
- Identifies: Sensitive layers, outliers
- Exports: Quantization config JSON

**`verify_calibration.py`**:
- Checks data integrity
- Reports: Missing files, suspicious values
- Estimates: Storage usage

## How to Use

### Basic Workflow

```bash
# 1. Generate (overnight)
python -m src.generate_calibration_data --num-images 1000 --num-steps 50

# 2. Collect activations (30 min)
python -m src.collect_layer_activations --calib-dir calibration_data --num-images 100

# 3. Analyze (instant)
python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --export-config quantization_config.json
```

### Quick Test

```bash
# Test with 10 images (6 minutes)
python -m src.generate_calibration_data --num-images 10 --num-steps 50
python -m src.collect_layer_activations --calib-dir calibration_data --num-images 10
python -m src.analyze_activations --stats calibration_data/activations/layer_statistics.json
```

## What the Analysis Provides

### Layer Statistics (for each layer)

```json
{
  "min": -2.456,           // For quantization range
  "max": 3.123,
  "mean": 0.234,           // For zero-point calculation
  "std": 0.891,            // For distribution understanding
  "percentiles": {         // For outlier-robust quantization
    "p01": -1.234,
    "p99": 2.456
  },
  "num_samples": 170000    // Statistical confidence
}
```

### Quantization Recommendations

```json
{
  "layer_name": {
    "suggested_weight_bits": 6,
    "suggested_activation_bits": 8,
    "use_percentile_clipping": true,
    "clip_percentile": [1, 99]
  }
}
```

### Analysis Output

```
Layer Statistics Summary:
  Total layers: 28
  Suggested bit-widths:
    8-bit: 12 layers (43%)
    6-bit: 10 layers (36%)
    4-bit:  6 layers (21%)
  
  Sensitive layers (need careful handling):
    - mm_block_0 (large range, needs 8-bit)
    - mm_block_15 (high variation, needs 8-bit)
  
  Layers suitable for aggressive quantization (≤4-bit):
    - mm_block_23 (low variation)
    - unified_block_5 (narrow range)
```

## Performance

### Time (M4 Max)

| Task | Images | Time |
|------|--------|------|
| Generation | 1000 | ~11 hours |
| Generation | 100 | ~1 hour |
| Generation | 10 | ~6 minutes |
| Activation collection | 100 | ~30 minutes |
| Analysis | - | <1 minute |

### Storage

| Dataset | Storage |
|---------|---------|
| 1000 images (no activations) | 3.3 GB |
| 1000 images + activations (100 subset) | 8 GB |
| 10 images (with activations) | 200 MB |

## Advantages of This Approach

1. **Efficient**: Only collects detailed activations from representative subset
2. **Flexible**: Can add more images or timesteps as needed
3. **Resume-able**: Can stop and resume generation
4. **Validated**: Includes verification and integrity checks
5. **Documented**: Extensive documentation and examples
6. **Non-invasive**: Doesn't modify DiffusionKit source code

## Limitations

1. **MLX-specific**: Monkey-patching approach tied to MLX's architecture
2. **Reload overhead**: ~10s per image for fresh pipeline loading
3. **Approximate**: Monkey-patching might miss some internal activations
4. **Memory-intensive**: Collecting activations requires keeping values in RAM

## Next Steps: Using the Data

### 1. Implement TaQ-DiT Quantization

```python
# Load statistics
with open('calibration_data/activations/layer_statistics.json') as f:
    stats = json.load(f)

# Apply per-layer quantization
for layer_name, layer_stats in stats.items():
    # Use p01/p99 for outlier-robust quantization
    qmin = layer_stats['percentiles']['p01']
    qmax = layer_stats['percentiles']['p99']
    
    # Compute scale and zero-point
    scale = (qmax - qmin) / (2**bits - 1)
    zero_point = -qmin / scale
    
    # Quantize
    quantize_layer(layer_name, scale, zero_point, bits)
```

### 2. Implement Timestep-Aware Quantization

```python
# Different strategies for different timestep phases
phase_configs = {
    'early': {'bits': 4, 'steps': range(0, 15)},
    'mid': {'bits': 6, 'steps': range(15, 35)},
    'late': {'bits': 8, 'steps': range(35, 51)},
}

# During inference, switch quantization based on timestep
for step in denoising_steps:
    phase = get_phase(step)
    model = quantize_model(model, phase_configs[phase])
    x = model.forward(x, step)
```

### 3. Evaluate with FID

```python
# Use generated images as reference
from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance()

# Load original images
original_images = load_images('calibration_data/images/')

# Generate with quantized model
quantized_images = generate_with_quantization(prompts, model_quantized)

# Compute FID
fid.update(original_images, real=True)
fid.update(quantized_images, real=False)
score = fid.compute()

print(f"FID: {score}")
```

## Troubleshooting Reference

See `README_CALIBRATION.md` and `QUICKSTART_CALIBRATION.md` for:
- Common issues and solutions
- Verification procedures
- Resume instructions
- Debug commands

## Key Files to Reference

1. **README_CALIBRATION.md**: Comprehensive documentation
2. **QUICKSTART_CALIBRATION.md**: Command reference
3. **layer_statistics.json**: Activation statistics (output)
4. **quantization_config.json**: Ready-to-use config (output)

## Summary

You now have a complete, production-ready system for:
- ✅ Generating large-scale calibration datasets
- ✅ Collecting layer activation statistics efficiently
- ✅ Analyzing and recommending quantization strategies
- ✅ Exporting configs for quantization implementation

**Total development time saved**: Would have taken weeks to debug and optimize without reading DiffusionKit source and understanding the pitfalls.

**Ready to proceed with TaQ-DiT implementation!** 🚀
