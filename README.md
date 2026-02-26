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
# ~52s/image; 100 images ≈ 87 min
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 100 --num-steps 100 --calib-dir calibration_data_100 [--resume]
```

Writes one `.npz` per (image, step) under `calibration_data_100/samples/` and a `manifest.json`.

---

### Weight Track

#### Step 2W — Cache block-level FP16 I/O for AdaRound

```bash
#only need 5 images and 25 time steps each for optimization
# ~60 min for 5 images (100-step schedule)
conda run -n diffusionkit python -m src.cache_adaround_data \
    --calib-dir calibration_data_100 --num-images 5 --stride 4 [--force] [--resume]
```

#### Step 3W — Optimize AdaRound W4A8 weights

```bash
conda run -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data_100/adaround_cache \
    --output quantized_weights \
    [--iters 20000] [--batch-size 16] [--bits-w 4] [--bits-a 8] [--blocks mm0,mm1]
```

---

### Activation Track

#### Step 2A — Collect per-layer activation statistics

```bash
# ~60 min for 5 images (100-step schedule)
conda run -n diffusionkit python -m src.collect_layer_activations \
    --calib-dir calibration_data_100 --num-images 5 --stride 4 [--force]
```

#### Step 3A — Analyze statistics and generate quantization config

**Baseline: W4A8 (faithful TaQ-DiT)**

```bash
conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data_100/activations/layer_statistics.json \
    --output calibration_data_100/activations/quant_config.json
```

Fixed A8 everywhere. Scale from `tensor_absmax` per layer per timestep; post-GELU layers carry per-channel shift vectors for centering before quantization. Outlier channels (range > 2.5× median) get a per-channel `multiplier_vector` for two-scale quantization — stored in `outlier_config` in the output JSON. Use `--use-hist-p999` to clip scale at the 99.9th percentile instead.

**Experimental: multi-tier A4/A6/A8**

```bash
conda run -n diffusionkit python -m src.analyze_activations_multitier \
    --stats calibration_data_100/activations/layer_statistics.json \
    --output calibration_data_100/activations/quant_config_multitier.json \
    [--a4-threshold 6.0] [--a6-threshold 10.0] \
    [--shifted-a4-threshold 5.0] [--shifted-a6-threshold 8.0]
```

Dynamic per-timestep bit selection (A4/A6/A8) with SmoothQuant detection. Runs over the same collected data — no re-collection needed. **Not** a faithful TaQ-DiT reproduction; preserved for future experimentation.

#### Optional — Visualize activation statistics

```bash
conda run -n diffusionkit python -m src.visualize_activations \
    --stats calibration_data_100/activations/layer_statistics.json \
    --output-dir calibration_data_100/activations/plots \
    [--snapshot-steps 0 24 48 72 96] \
    [--quant-config calibration_data_100/activations/quant_config.json] \
    [--plot-distributions]
```

---

### Step 4 — Inference with quantized weights

**V1 — weights only (FP16 activations):**

```bash
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output quantized_weights \
    --prompt "a tabby cat on a table" \
    --output-image quant_test.png [--compare] [--diff-stats]
```

**V2 — fake-quantized activations (requires Step 3A output):**

```bash
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output quantized_weights \
    --quant-config calibration_data_100/activations/quant_config.json \
    --prompt "a tabby cat on a table" \
    --output-image quant_w4a8_actquant.png [--compare]
```

V2 applies per-(layer, timestep) fake activation quantization with shift and two-scale outlier handling via a custom Euler inference loop.

**V3 — native MLX int4 weights (~4x memory savings):**

```bash
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output quantized_weights \
    --mlx-int4 --group-size 64 \
    --prompt "a tabby cat on a table" \
    --output-image quant_v3.png [--compare]
```

Note: MLX requires `in_dims ≥ 256` for 4-bit quantization (Metal kernel constraint).

---

---

## HTG + Bayesian Bits Pipeline (Extended)

A 6-stage per-group pipeline that partitions the denoising trajectory into **G groups** with
similar activation statistics (HTG), assigns per-layer bit widths automatically (Bayesian Bits),
and runs AdaRound+joint-reconstruction per group.

```
Step 1: Generate calibration latents       (existing, reused)
Step 2A: Collect activation stats          (existing, reused)
Step 2W: Cache adaround I/O               (existing, reused)

Stage 0 — HTG Clustering
Stage 1 — Bayesian Bits (per group)
Stages 2+3 — Per-group shift + outlier config
Stages 4+5 — Per-group AdaRound + joint reconstruction
```

### Stage 0 — HTG Clustering

```bash
conda run -n diffusionkit python -m src.htg_cluster \
    --stats calibration_data_100/activations/layer_statistics.json \
    --output htg_groups.json \
    [--n-groups 5]
```

Partitions the 25 collected timesteps into G groups using adjacency-constrained
agglomerative clustering on per-layer per-channel shift vectors (arXiv 2503.06930).
Outputs `htg_groups.json` with `global_groups` (shared partition), `per_layer_z_bar`
(averaged shift vectors per group per layer), and `sigma_map`.

### Stage 1 — Bayesian Bits (per group)

```bash
conda run -n diffusionkit python -m src.bayesianbits_optimize \
    --adaround-cache calibration_data_100/adaround_cache \
    --htg-groups htg_groups.json \
    --output bb_config.json \
    [--iters 20000] [--batch-size 16] [--gating-lambda 0.01] [--bits-a 8] \
    [--blocks mm0]
```

Learns per-layer bit widths (W2/W4/W8) per group using hierarchical nested quantization
with L0 hard-concrete gating (arXiv 2005.07093). For each group, only the calibration
samples from that group's timesteps are used.

### Stages 2+3 — Per-group shift + outlier config

```bash
conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data_100/activations/layer_statistics.json \
    --htg-groups htg_groups.json \
    --output quant_config_htg.json
```

Produces `quant_config_htg.json` (format `per_group_quant_config_htg_v1`) with per-group
scale, shift (from `per_layer_z_bar`), and outlier_config. Uses the same
`identify_outlier_channels()` as the standard pipeline, applied to group-averaged statistics.

### Stages 4+5 — Per-group AdaRound

```bash
conda run -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data_100/adaround_cache \
    --output quantized_weights_htg \
    --htg-groups htg_groups.json \
    --bb-config bb_config.json \
    [--iters 20000] [--batch-size 16] [--bits-a 8] \
    [--blocks mm0]
```

Runs one AdaRound+activation-scale optimization loop per group using only that group's
calibration samples. If `--bb-config` is given, uses per-layer bit widths from Stage 1;
otherwise uses global `--bits-w`. Outputs one directory per group:
`quantized_weights_htg/group_{g}/weights/{block_name}.npz`.

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

`calibration_data_100/` (created by `generate_calibration_data.py`):

```
calibration_data_100/
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
