# Quantization Pipeline — Quick Reference

All commands require the `diffusionkit` conda environment.

## Complete Workflow

```bash
# 1. Generate calibration latents (~6 min for 10 images, ~11h for 1000)
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 1000 --num-steps 50 --calib-dir calibration_data [--resume]

# --- WEIGHT TRACK ---

# 2W. Cache block-level FP16 I/O for AdaRound (~30 min for 5 images)
conda run -n diffusionkit python -m src.cache_adaround_data \
    --calib-dir calibration_data --num-images 5 --stride 5 [--force]

# 3W. Optimize AdaRound W4A8 weights
conda run -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data/adaround_cache \
    --output quantized_weights \
    [--iters 20000] [--batch-size 16] [--bits-w 4] [--bits-a 8]

# --- ACTIVATION TRACK ---

# 2A. Collect per-layer activation statistics (~30 min for 5 images)
conda run -n diffusionkit python -m src.collect_layer_activations \
    --calib-dir calibration_data --num-images 5 --stride 2 [--force]

# 3A. W4A8 baseline config (faithful TaQ-DiT)
conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --output calibration_data/activations/quant_config.json

# 3A-alt. Experimental multi-tier A4/A6/A8 config
conda run -n diffusionkit python -m src.analyze_activations_multitier \
    --stats calibration_data/activations/layer_statistics.json \
    --output calibration_data/activations/quant_config_multitier.json

# --- INFERENCE ---

# 4a. V1: FP16 activations (validate AdaRound weight quality)
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output quantized_weights \
    --prompt "a tabby cat on a table" \
    --output-image quant_test.png [--compare] [--diff-stats]

# 4b. V2: fake-quantized W4A8 activations (requires quant_config.json from Step 3A)
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output quantized_weights \
    --quant-config calibration_data/activations/quant_config.json \
    --prompt "a tabby cat on a table" \
    --output-image quant_w4a8_actquant.png [--compare]
```

## Quick Test (10 images, ~6 min)

```bash
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 10 --num-steps 50 --calib-dir calibration_data

conda run -n diffusionkit python -m src.collect_layer_activations \
    --calib-dir calibration_data --num-images 5 --stride 2

conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json
```

## Resume Interrupted Generation

```bash
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 1000 --num-steps 50 --resume
```

## Force Regenerate Activations

```bash
conda run -n diffusionkit python -m src.collect_layer_activations \
    --calib-dir calibration_data --num-images 5 --force
```

## Optional: Visualize Activation Statistics

```bash
conda run -n diffusionkit python -m src.visualize_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --output-dir calibration_data/activations/plots \
    [--snapshot-steps 0 12 24 40 48] \
    [--quant-config calibration_data/activations/quant_config.json] \
    [--plot-distributions]
```

## Common Issues

**Second image fails with shape mismatch**
Handled automatically — adaLN weights are reloaded between images with a targeted
modulation-only reload. See TROUBLESHOOTING_GUIDE.md Issue 4.

**Out of memory during activation collection**
```bash
conda run -n diffusionkit python -m src.collect_layer_activations --num-images 5
```

**Missing samples / interrupted generation**
```bash
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 1000 --resume
```

## File Locations

```
calibration_data/
├── manifest.json
├── samples/                         # per-step latents
├── adaround_cache/                  # weight track cache
└── activations/
    ├── layer_statistics.json        # collected stats
    ├── quant_config.json            # W4A8 baseline
    ├── quant_config_multitier.json  # experimental A4/A6/A8
    └── timestep_stats/

quantized_weights/                   # AdaRound output
    ├── config.json
    └── weights/{block_name}.npz
```

## Time Estimates (M4 Max)

| Task | Time |
|------|------|
| Generate 10 images | ~6 min |
| Generate 1000 images | ~11 hours |
| Cache AdaRound data (5 images) | ~30 min |
| Collect activations (5 images) | ~30 min |
| Analyze (either variant) | <1 min |
