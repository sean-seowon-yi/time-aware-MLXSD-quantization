# MLX Project

MLX-based diffusion (SD3-Medium) pipeline with post-training quantization (PTQ) tooling. Implements **TaQ-DiT** adaptive rounding (AdaRound) for W4A8 quantization of the SD3-Medium DiT backbone using Apple's MLX framework.

## Environment

All scripts run inside the `diffusionkit` conda environment:

```bash
conda run -n diffusionkit python -m src.<module> [args]
```

If import errors occur, reinstall DiffusionKit:

```bash
pip install -e DiffusionKit/python
```

## Full Quantization Pipeline

Two tracks run in parallel from the same calibration latents. Steps 2W/3W (weight track) and 2A/3A (activation track) are independent of each other.

```
Step 1: Generate calibration latents
           │
           ├─── WEIGHT TRACK ──────────────────────────────┐
           │    2W. Cache block-level FP16 I/O             │
           │    3W. Optimize AdaRound weights               │
           │                                               │
           └─── ACTIVATION TRACK ──────────────────────────┤
                2A. Collect per-layer activation stats     │
                3A. Analyze → quant config                 │
                                                           ▼
                                               Step 4: Inference
```

### Step 1 — Generate calibration latents

```bash
# ~6 min for 10 images, ~11h for 1000
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 10 --num-steps 50 --calib-dir calibration_data [--resume]
```

Writes one `.npz` per (image, step) under `calibration_data/samples/` and a `manifest.json`.

---

### Weight Track

#### Step 2W — Cache block-level FP16 I/O for AdaRound

```bash
# ~30 min for 5 images
conda run -n diffusionkit python -m src.cache_adaround_data \
    --calib-dir calibration_data --num-images 5 --stride 5 [--force]
```

#### Step 3W — Optimize AdaRound W4A8 weights

```bash
conda run -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data/adaround_cache \
    --output quantized_weights \
    [--iters 20000] [--batch-size 16] [--bits-w 4] [--bits-a 8] [--blocks mm0,mm1]
```

---

### Activation Track

#### Step 2A — Collect per-layer activation statistics

```bash
# ~30 min for 5 images
conda run -n diffusionkit python -m src.collect_layer_activations \
    --calib-dir calibration_data --num-images 5 --stride 2 [--force]
```

#### Step 3A — Analyze statistics and generate quantization config

**Baseline: W4A8 (faithful TaQ-DiT)**

```bash
conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --output calibration_data/activations/quant_config.json
```

Fixed A8 everywhere. Scale from `tensor_absmax` per layer per timestep; post-GELU layers carry per-channel shift vectors for centering before quantization. Use `--use-hist-p999` to clip scale at the 99.9th percentile instead.

**Experimental: multi-tier A4/A6/A8**

```bash
conda run -n diffusionkit python -m src.analyze_activations_multitier \
    --stats calibration_data/activations/layer_statistics.json \
    --output calibration_data/activations/quant_config_multitier.json \
    [--a4-threshold 6.0] [--a6-threshold 10.0] \
    [--shifted-a4-threshold 5.0] [--shifted-a6-threshold 8.0]
```

Dynamic per-timestep bit selection (A4/A6/A8) with SmoothQuant detection. Runs over the same collected data — no re-collection needed. **Not** a faithful TaQ-DiT reproduction; preserved for future experimentation.

#### Optional — Visualize activation statistics

```bash
conda run -n diffusionkit python -m src.visualize_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --output-dir calibration_data/activations/plots \
    [--snapshot-steps 0 12 24 40 48] \
    [--quant-config calibration_data/activations/quant_config.json] \
    [--plot-distributions]
```

---

### Step 4 — Inference with quantized weights

```bash
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output quantized_weights \
    --prompt "a tabby cat on a table" \
    --output-image quant_test.png [--compare] [--diff-stats]
```

---

## Running Tests

```bash
# All tests (~3s, no model loading)
conda run -n diffusionkit python -m pytest tests/ -v

# Single file
conda run -n diffusionkit python -m pytest tests/test_adaround_optimize.py -v
```

Tests use synthetic tensors and mocked DiffusionKit.

## Calibration Data Layout

`calibration_data/` (created by `generate_calibration_data.py`):

```
calibration_data/
├── manifest.json
├── samples/
│   └── {img:04d}_{step:03d}.npz       # keys: x, timestep, sigma, step_index, image_id, is_final
├── adaround_cache/                     # written by cache_adaround_data.py
│   ├── metadata.json
│   └── samples/{img:04d}_{step:03d}.npz
└── activations/                        # written by collect_layer_activations.py
    ├── layer_statistics.json
    ├── quant_config.json               # written by analyze_activations.py
    ├── quant_config_multitier.json     # written by analyze_activations_multitier.py
    ├── layer_temporal_analysis.json
    └── timestep_stats/
        ├── step_{key}.npz             # per-channel avg_min/avg_max/shift/histograms
        └── step_{key}_index.json      # scalar summaries (tensor_absmax, hist_p999, ...)
```
