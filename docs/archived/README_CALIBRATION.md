# Calibration Data Generation for Diffusion Transformer Quantization

Complete workflow for generating calibration data and collecting layer activation statistics for TaQ-DiT style quantization.

---

## Overview

This pipeline generates calibration data for quantizing diffusion transformers with timestep-awareness:

1. **Generate calibration dataset** (1000 images with per-step latents)
2. **Collect layer activations** (from representative 100 images at key timesteps)
3. **Analyze statistics** (determine optimal quantization strategies)
4. **Export configuration** (for quantization implementation)

---

## Quick Start

### Step 1: Generate Calibration Data (1000 images)

```bash
# Generate 1000 images with per-step latent states
python -m src.generate_calibration_data \
    --num-images 1000 \
    --num-steps 50 \
    --calib-dir calibration_data \
    --prompt-csv all_prompts.csv

# Time: ~10-12 hours (M4 Max)
# Output: ~1.3 GB (samples + latents + images + manifest)
```

**What this produces**:
- `calibration_data/samples/`: 51,000 .npz files (per-step latent states)
- `calibration_data/latents/`: 1,000 .npy files (final latents)
- `calibration_data/images/`: 1,000 .png files (decoded images for FID)
- `calibration_data/manifest.json`: Metadata with all prompts

### Step 2: Collect Layer Activations (100 images × 17 steps)

```bash
# Collect activations from representative subset
python -m src.collect_layer_activations \
    --calib-dir calibration_data \
    --num-images 100

# Time: ~20-30 minutes
# Output: ~18 GB activation statistics
```

**What this produces**:
- `calibration_data/activations/layer_statistics.json`: Per-layer statistics
- `calibration_data/activations/collection_metadata.json`: Collection info

### Step 3: Analyze Statistics

```bash
# View analysis and recommendations
python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json

# Export quantization config
python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --export-config quantization_config.json
```

---

## Detailed Workflow

### 1. Calibration Data Generation

**Purpose**: Capture latent states at each denoising timestep for 1000 diverse images.

**Key Features**:
- Uses DiffusionKit's exact Euler sampling implementation
- Saves only essential data (latents, not conditioning)
- Prompts stored in manifest for regeneration
- Proper handling of model state corruption (fresh pipeline per image)

**Output Structure**:
```
calibration_data/
├── samples/              # Per-step latent states
│   ├── 0000_000.npz     # Image 0, step 0
│   ├── 0000_001.npz     # Image 0, step 1
│   ├── ...
│   └── 0999_050.npz     # Image 999, step 50
├── latents/             # Final denoised latents
│   ├── 0000.npy
│   └── ...
├── images/              # Decoded final images
│   ├── 0000.png
│   └── ...
└── manifest.json        # All prompts and metadata
```

**Sample Data Format**:
```python
{
    'x': [1, 16, 64, 64],      # Latent state
    'timestep': scalar,         # Model timestep
    'sigma': scalar,            # Noise level
    'step_index': int,          # 0-50
    'image_id': int,            # Which image
    'is_final': bool,           # True for last step
}
```

### 2. Activation Collection

**Purpose**: Extract layer activation statistics for quantization calibration.

**Strategy**:
- **Subsample images**: 100 out of 1000 (sufficient for statistics)
- **Subsample timesteps**: ~17 out of 51 (focus on critical steps)
- **Total samples**: 100 × 17 = 1,700 (vs 51,000 if using all)

**Key Timestep Selection**:
```python
# Always included:
- Step 0 (pure noise)
- Step 50 (final denoised)
- Steps 12, 25, 37 (quarter points)

# Dense in middle (where structure forms):
- Every 3 steps from step 16-33

# Sparse elsewhere:
- Every 5 steps from steps 0-16 and 34-50

# Result: ~17 timesteps covering all phases
```

**Collected Statistics**:
```python
{
    "layer_name": {
        "min": float,              # Minimum activation value
        "max": float,              # Maximum activation value
        "mean": float,             # Mean activation
        "std": float,              # Standard deviation
        "percentiles": {           # For outlier-robust quantization
            "p01": float,
            "p05": float,
            ...
            "p99": float,
        },
        "num_samples": int,        # Sample count
    }
}
```

### 3. Analysis & Recommendations

**What the analyzer provides**:

1. **Dynamic Range Analysis**
   - Which layers have large ranges (need more bits)
   - Outlier detection (layers with extreme values)

2. **Bit-Width Suggestions**
   - Per-layer recommendations (2, 4, 6, or 8 bits)
   - Based on range, std, and variation

3. **Quantization Strategy**
   - Layers suitable for aggressive quantization (≤4 bits)
   - Sensitive layers needing careful handling (8 bits)
   - Percentile clipping recommendations

4. **Exportable Config**
   - JSON format compatible with quantization scripts
   - Includes calibration statistics per layer

---

## Usage Examples

### Resume Interrupted Generation

```bash
# Start generation
python -m src.generate_calibration_data --num-images 1000 --num-steps 50

# If interrupted, resume from where it left off
python -m src.generate_calibration_data --num-images 1000 --num-steps 50 --resume
```

### Generate Smaller Test Set

```bash
# Quick test with 10 images
python -m src.generate_calibration_data --num-images 10 --num-steps 50
```

### Collect from Specific Images

```bash
# Collect activations from first 50 images only
python -m src.collect_layer_activations \
    --calib-dir calibration_data \
    --num-images 50
```

### Force Regenerate Activations

```bash
# Overwrite existing activation statistics
python -m src.collect_layer_activations \
    --calib-dir calibration_data \
    --num-images 100 \
    --force
```

---

## Storage Requirements

### For 1000 Images

| Component | Size | Count | Total |
|-----------|------|-------|-------|
| Calibration samples | ~230 KB | 51,000 | ~1.2 GB |
| Final latents | 65 KB | 1,000 | 65 MB |
| Decoded images | ~2 MB | 1,000 | 2 GB |
| Manifest | 500 KB | 1 | 500 KB |
| **Total** | | | **~3.3 GB** |

### Activation Statistics

| Collection Scope | Storage |
|-----------------|---------|
| 100 images × 17 timesteps | ~5 GB |
| 100 images × 51 timesteps | ~15 GB |
| 1000 images × 17 timesteps | ~50 GB |

---

## Time Estimates (M4 Max)

| Task | Images | Time per Image | Total Time |
|------|--------|----------------|------------|
| **Calibration generation** | 1000 | ~40s | ~11 hours |
| **Activation collection** | 100 | ~18s | ~30 min |
| **Analysis** | - | - | <1 min |
| **Total** | | | **~11.5 hours** |

---

## Output Files Reference

### `manifest.json`
```json
{
  "n_completed": 1000,
  "num_steps": 50,
  "cfg_scale": 7.5,
  "images": [
    {
      "image_id": 0,
      "prompt": "a photo of a cat",
      "seed": 42,
      "cfg_weight": 7.5,
      "num_steps": 50,
      "filename": "0000.png",
      "latent_filename": "0000.npy"
    },
    ...
  ]
}
```

### `layer_statistics.json`
```json
{
  "mm_block_0": {
    "min": -2.456,
    "max": 3.123,
    "mean": 0.234,
    "std": 0.891,
    "percentiles": {
      "p01": -1.234,
      "p99": 2.456
    },
    "num_samples": 170000
  },
  ...
}
```

### `quantization_config.json`
```json
{
  "default_config": {
    "weight_bits": 4,
    "activation_bits": 8
  },
  "layer_configs": {
    "mm_block_0": {
      "weight_bits": 6,
      "activation_bits": 8,
      "clip_percentile": [1, 99],
      "calibration_stats": { ... }
    },
    ...
  }
}
```

---

## Next Steps: Using the Data

### 1. Implement TaQ-DiT Quantization

Use the collected statistics to implement TaQ-DiT:

```python
# Load activation statistics
with open('calibration_data/activations/layer_statistics.json') as f:
    layer_stats = json.load(f)

# Implement per-layer quantization
for layer_name, stats in layer_stats.items():
    # Determine quantization parameters
    qmin = stats['percentiles']['p01']  # Use percentiles for robustness
    qmax = stats['percentiles']['p99']
    
    scale = (qmax - qmin) / (2**bits - 1)
    zero_point = -qmin / scale
    
    # Quantize layer
    quantize_layer(layer_name, scale, zero_point, bits)
```

### 2. Implement Timestep-Aware Quantization

Group statistics by timestep phase:

```python
# Analyze by timestep
early_steps = [0, 5, 10]      # High noise tolerance
mid_steps = [15, 20, 25, 30]  # Critical structure formation
late_steps = [35, 40, 45, 50] # Fine detail refinement

# Different bit-widths per phase
quantization_schedule = {
    'early': 4,   # Aggressive
    'mid': 6,     # Moderate
    'late': 8,    # Conservative
}
```

### 3. Evaluate Quantized Model

Use the 1000 generated images for FID evaluation:

```python
# Generate images with quantized model
quantized_images = generate_with_quantization(
    prompts=manifest['images'],
    quantized_model=model_q
)

# Compare with original images
fid = compute_fid(
    original_images='calibration_data/images/',
    generated_images=quantized_images
)

print(f"FID score: {fid}")
```

---

## Troubleshooting

### Issue: "Model state corruption" error

**Cause**: DiffusionKit's cached AdaLN parameters interfere between images.

**Solution**: Script already handles this by reloading pipeline per image.

### Issue: Out of memory during activation collection

**Reduce batch size**:
```bash
python -m src.collect_layer_activations --num-images 50  # Instead of 100
```

### Issue: Missing calibration samples

**Check manifest**:
```bash
python -c "
import json
with open('calibration_data/manifest.json') as f:
    m = json.load(f)
print(f'Completed: {m[\"n_completed\"]} images')
"
```

**Rebuild if needed**:
```bash
python -m src.rebuild_manifest --calib-dir calibration_data
```

---

## Design Decisions

### Why 100 Images for Activation Collection?

Research shows activation distributions stabilize with 50-100 diverse samples. Using all 1000 provides diminishing returns while requiring 10x more compute.

### Why 17 Timesteps?

Critical timesteps are:
- Boundaries (start/end): 2 steps
- Phase transitions: 5 steps
- Dense in middle: 10 steps

This captures all distribution phases without redundancy.

### Why Not Save Activations During Generation?

**Storage explosion**: Saving activations for all 1000 images × 51 timesteps = 500 GB

**Better approach**: Generate lightweight calibration data, then selectively collect activations from subset.

---

## References

- **TaQ-DiT Paper**: Time-Aware Quantization for Diffusion Transformers
- **DiffusionKit**: https://github.com/argmaxinc/DiffusionKit
- **Karras et al. (2022)**: Elucidating the Design Space of Diffusion-Based Generative Models

---

## File Overview

```
src/
├── generate_calibration_data.py        # Generate 1000 images with latents
├── collect_layer_activations.py        # Collect activations from subset
├── analyze_activations.py              # Analyze and visualize statistics
└── README_CALIBRATION.md               # This file

calibration_data/
├── samples/                            # Per-step latents
├── latents/                            # Final latents
├── images/                             # Decoded images
├── activations/                        # Layer statistics
│   ├── layer_statistics.json
│   └── collection_metadata.json
└── manifest.json                       # Prompts and metadata
```
