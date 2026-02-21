# Calibration Data Collection - Quick Reference

## Complete Workflow

```bash
# 1. Generate calibration data (1000 images, ~11 hours)
python -m src.generate_calibration_data \
    --num-images 1000 \
    --num-steps 50 \
    --calib-dir calibration_data \
    --prompt-csv all_prompts.csv

# 2. Verify calibration data
python -m src.verify_calibration --calib-dir calibration_data

# 3. Collect layer activations (100 images, ~30 min)
python -m src.collect_layer_activations \
    --calib-dir calibration_data \
    --num-images 100

# 4. Analyze activation statistics
python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --export-config quantization_config.json

# 5. Verify everything
python -m src.verify_calibration --calib-dir calibration_data
```

## Quick Test (10 images)

```bash
# Generate small test set
python -m src.generate_calibration_data \
    --num-images 10 \
    --num-steps 50 \
    --calib-dir test_calibration

# Collect activations
python -m src.collect_layer_activations \
    --calib-dir test_calibration \
    --num-images 10

# Analyze
python -m src.analyze_activations \
    --stats test_calibration/activations/layer_statistics.json
```

## Resume Interrupted Generation

```bash
# If generation was interrupted, resume:
python -m src.generate_calibration_data \
    --num-images 1000 \
    --num-steps 50 \
    --resume
```

## Regenerate Activations

```bash
# Force overwrite existing activation statistics
python -m src.collect_layer_activations \
    --calib-dir calibration_data \
    --num-images 100 \
    --force
```

## Common Issues

### Issue: Model state corruption
**Symptom**: Second image fails with shape mismatch
**Fix**: Already handled by script (reloads pipeline per image)

### Issue: Out of memory during activation collection
**Fix**: Reduce number of images
```bash
python -m src.collect_layer_activations --num-images 50
```

### Issue: Missing samples
**Check**:
```bash
python -m src.verify_calibration --calib-dir calibration_data
```

## File Locations

```
calibration_data/
├── samples/                 # Per-step latents (1.2 GB)
├── latents/                 # Final latents (65 MB)
├── images/                  # Decoded images (2 GB)
├── activations/             # Layer statistics (varies)
│   ├── layer_statistics.json
│   └── collection_metadata.json
└── manifest.json            # All prompts and metadata
```

## Time Estimates (M4 Max)

| Task | Time |
|------|------|
| Generate 1000 images | ~11 hours |
| Generate 100 images | ~1 hour |
| Generate 10 images | ~6 minutes |
| Collect activations (100 images) | ~30 minutes |
| Analyze statistics | <1 minute |

## Storage Estimates

| Dataset | Storage |
|---------|---------|
| 10 images (with activations) | ~200 MB |
| 100 images (with activations) | ~1.5 GB |
| 1000 images (no activations) | ~3.3 GB |
| 1000 images (with activations) | ~8 GB |

## Next Steps

After collecting calibration data and activation statistics:

1. **Implement TaQ-DiT quantization**
2. **Use `quantization_config.json` for layer-specific settings**
3. **Evaluate quantized model with FID using generated images**
4. **Iterate on quantization strategy based on results**

## Helper Scripts

```bash
# Rebuild manifest if corrupted
python -m src.rebuild_manifest --calib-dir calibration_data

# Decode latents to images (if decode failed during generation)
python -m src.decode_latents --calib-dir calibration_data

# Verify data integrity
python -m src.verify_calibration --calib-dir calibration_data
```
