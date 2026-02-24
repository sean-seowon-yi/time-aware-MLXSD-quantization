# TaQ-DiT W4A8 Quantization Pipeline

AdaRound-based PTQ for SD3-Medium on Apple Silicon (MLX).

---

## Two Calibration Tracks

TaQ-DiT requires calibrating weights and activations separately. Both tracks read
the same calibration latents from Step 1 and feed into the final quantized model.

```
                    ┌─────────────────────────────────┐
                    │   Step 1: generate_calibration   │
                    │   (latents + timesteps, ~2-4h)   │
                    └──────────┬──────────────┬────────┘
                               │              │
               ┌───────────────▼───┐    ┌─────▼──────────────────┐
               │  WEIGHT TRACK     │    │  ACTIVATION TRACK      │
               │                   │    │                         │
               │ cache_adaround    │    │ collect_layer_          │
               │ (~30-60 min)      │    │ activations (~30 min)   │
               │        │          │    │         │               │
               │ adaround_optimize │    │ analyze_activations     │
               │ (~1-2h)           │    │ (<1 min)                │
               │        │          │    │         │               │
               │ weights/mm*.npz   │    │ quant_config.json       │
               │ (int8 W4 scales)  │    │ (A8 per layer, shifts,  │
               │                   │    │  outlier_config)        │
               └──────────┬────────┘    └─────────┬───────────────┘
                          │                       │
                    ┌─────▼───────────────────────▼─────┐
                    │    load_adaround_model             │
                    │    (inject weights + apply         │
                    │     activation scales for          │
                    │     true W4A8 inference)           │
                    └────────────────────────────────────┘
```

---

## Step 1 — Generate calibration latents

```bash
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 50 \
    --num-steps 50 \
    --cfg-weight 7.5 \
    --calib-dir /path/to/calibration_data
```

**Why:** Both tracks need real latent trajectories from the FP16 model — not random
noise. This script runs the full 50-step Euler sampler for each prompt, saving the
noisy latent `x` and the corresponding `timestep`/`sigma` at every step.

- **50 images** gives enough diversity for per-channel scale estimates across all 24
  transformer blocks
- **Each image** covers the full denoising trajectory, so early/mid/late noise phases
  are all represented
- Creates a fresh pipeline per image (avoids adaLN weight state corruption)

**Output:**
```
calibration_data/
  manifest.json            # prompts, seeds, cfg, num_steps
  samples/
    0000_000.npz           # x, timestep, sigma at step 0 of image 0
    ...
    0049_049.npz
  images/                  # decoded PNG for sanity-check
  latents/                 # final decoded latent per image
```

---

## Weight Track

### Step 2W — Cache block-level FP16 I/O

```bash
conda run -n diffusionkit python -m src.cache_adaround_data \
    --calib-dir /path/to/calibration_data \
    --output-dir /path/to/calibration_data/adaround_cache \
    --num-images 5 \
    --stride 5 \
    --force
```

**Why:** AdaRound optimizes each block independently by minimizing the difference
between its FP16 output and its quantized output given the same input. It needs those
exact FP16 inputs and outputs pre-cached so it can run the optimization loop without
touching the rest of the model.

All 24 blocks are hooked simultaneously, so only `num_images × (num_steps / stride)`
forward passes are required — not multiplied by the number of blocks.

- **`--stride 5`:** Every 5th step → 10 timesteps per image, spanning early/mid/late noise
- **`--num-images 5`:** 50 total (image, timestep) sample files

**Output:**
```
adaround_cache/
  metadata.json
  samples/
    0000_000.npz    # I/O for all 24 blocks at (image 0, step 0)
    ...
    0040_045.npz
```

---

### Step 3W — Run AdaRound optimization

```bash
conda run -n diffusionkit python -m src.adaround_optimize \
    --cache-dir /path/to/calibration_data/adaround_cache \
    --output-dir /path/to/adaround_output \
    --bits-w 4 \
    --bits-a 8 \
    --iters 1000
```

**Why:** AdaRound (Nagel et al. 2020) learns the optimal rounding direction for each
weight by minimizing per-block reconstruction error. Naive round-to-nearest ignores
cross-weight interactions; AdaRound's learned rounding consistently recovers 0.5–1 dB
PSNR vs RTN at 4-bit with no retraining.

Runs block-by-block: loads cached FP16 inputs, runs the quantized block forward,
computes reconstruction + rounding regularization loss, updates soft-rounding
parameters `alpha` via Adam. The `b` annealing schedule transitions alpha from soft
to hard binary (b: 20→2) after a 20% warmup.

- **`--bits-w 4`:** Per-channel symmetric INT4, stored as int8 + float32 scale
- **`--bits-a 8`:** 8-bit activations during reconstruction (consistent with target inference)
- **`--iters 1000`:** Adam steps per block

**Output:**
```
adaround_output/
  config.json          # bits_w, bits_a, iters, per-block reconstruction error
  weights/
    mm0.npz            # {path}__weight_int (int8), __scale (fp32), __a_scale (fp32)
    ...
    mm23.npz
```

---

## Activation Track

### Step 2A — Collect per-layer activation statistics

```bash
conda run -n diffusionkit python -m src.collect_layer_activations \
    --calib-dir /path/to/calibration_data \
    --num-images 5 \
    --stride 2 \
    --force
```

**Why:** Activation quantization scales must be calibrated from real data. This script
replays the saved latents through the model with hooks on every linear layer, recording
per-channel min/max statistics at each timestep.

TaQ-DiT's key activation quantization techniques, all implemented here:

1. **Per-channel AvgMinMax:** Running average of per-batch per-channel min/max
   (more stable than running max, which is dominated by outliers)
2. **Moving-average shift for post-GELU layers** (`mlp.fc2` inputs): The GELU output
   is asymmetric — its distribution is skewed right. A per-channel shift
   `shift = 0.95 * shift + 0.05 * (min + max) / 2` centers it before quantization,
   recovering significant precision for these layers
3. **Per-timestep statistics:** Activation distributions shift substantially between
   early (high-noise, large activations) and late (low-noise) denoising steps.
   Collecting per-timestep stats enables timestep-adaptive quantization scales —
   the core TaQ-DiT insight
4. **SmoothQuant detection:** Flags layers with a small fraction of extreme-magnitude
   channels (p100/p99 > 5x, <5% channels above p99). These can be handled by
   per-channel weight scaling rather than requiring A8 for the whole layer

- **`--stride 2`:** Every other step → ~25 timesteps, sufficient to capture all phases
- **`--num-images 5`:** 5 images × 25 timesteps = 125 forward passes per layer

**Output:**
```
calibration_data/activations/
  layer_statistics.json        # manifest with sigma_map and step_keys
  timestep_stats/
    step_0.npz                 # avg_min/avg_max/shift/hist arrays per layer
    step_0_index.json          # scalar summary (tensor_absmax, hist_p999, ...)
    step_2.npz
    ...
    step_48.npz
```

---

### Step 3A — Analyze and generate quantization config

```bash
# Baseline: W4A8 (faithful TaQ-DiT)
conda run -n diffusionkit python -m src.analyze_activations \
    --stats /path/to/calibration_data/activations/layer_statistics.json \
    --output /path/to/calibration_data/activations/quant_config.json

# Experimental: multi-tier A4/A6/A8 (not faithful TaQ-DiT)
conda run -n diffusionkit python -m src.analyze_activations_multitier \
    --stats /path/to/calibration_data/activations/layer_statistics.json \
    --output /path/to/calibration_data/activations/quant_config_multitier.json
```

**Why:** Converts raw per-channel statistics into per-layer, per-timestep quantization
scales and shift vectors that will be applied at inference time.

`analyze_activations.py` (faithful TaQ-DiT baseline):
- Fixed **A8** everywhere — `bits = 8` for all layers and timesteps
- Scale: `tensor_absmax` per layer per timestep
- Post-GELU layers (`mlp.fc2`): per-channel `shift` vectors for centering
- **Two-scale outlier handling:** `identify_outlier_channels()` flags channels where
  `range_c > 2.5 × median(range)` and stores a `multiplier_vector` in `outlier_config`

`analyze_activations_multitier.py` (experimental, preserved for future work):
- Dynamic A4/A6/A8 per-timestep bit selection based on scale thresholds
- SmoothQuant detection for isolated spike channels
- Also computes `outlier_config` using the same logic

**Output:**
```
calibration_data/activations/
  quant_config.json              # per_timestep + outlier_config + sigma_map
  layer_temporal_analysis.json   # per-layer scale variability, shift magnitudes
```

---

## Step 4 — Inference

### V1 — Weights only (FP16 activations)

```bash
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output /path/to/adaround_output \
    --prompt "a tabby cat sitting on a wooden table" \
    --output-image quant_test.png \
    --compare \
    --diff-stats
```

Dequantizes INT4 weights to FP16 at load time and runs inference with the standard
`pipeline.generate_image()`. Validates AdaRound rounding quality; activations remain
full-precision.

### V2 — Fake-quantized activations

```bash
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output /path/to/adaround_output \
    --quant-config /path/to/calibration_data/activations/quant_config.json \
    --prompt "a tabby cat sitting on a wooden table" \
    --output-image quant_w4a8_actquant.png \
    --compare
```

Wraps each `nn.Linear` with `_ActQuantLayer` and runs a custom Euler inference loop
that threads the current step_key into proxies before each denoising step.
Per-(layer, timestep) fake activation quantization with:
- Per-channel shift (post-GELU `mlp.fc2` inputs only)
- Two-scale outlier handling via `multiplier_vector` from `outlier_config`
- Fake-quantize: round → clip → dequant (stays float; no memory savings)

This measures the combined W4A8 quality impact and validates that `quant_config.json`
is correct before implementing true INT8 activation kernels.

**V3 target (future):** Custom `QuantizedLinear` holding int8 weights and executing
`(W_int8 * scale).T @ x_quantized` — true memory savings + exact AdaRound rounding.

---

## Quick Reference

| Step | Script | Time estimate | Output |
|------|--------|---------------|--------|
| 1. Calibration latents | `generate_calibration_data.py` | ~2–4h (50 images) | `samples/*.npz` |
| 2W. Block I/O cache | `cache_adaround_data.py` | ~30–60 min | `adaround_cache/` |
| 3W. AdaRound optimize | `adaround_optimize.py` | ~1–2h (24 blocks) | `weights/mm*.npz` |
| 2A. Activation stats | `collect_layer_activations.py` | ~30 min | `timestep_stats/` |
| 3A. Quant config | `analyze_activations.py` | <1 min | `quant_config.json` |
| 4a. Validate (V1, FP16 acts) | `load_adaround_model.py` | ~5 min | comparison PNGs |
| 4b. Validate (V2, fake-quant acts) | `load_adaround_model.py --quant-config` | ~5 min | comparison PNGs |

Steps 2W/3W and 2A/3A can run in parallel (both read from Step 1 output).

---

## Architecture Notes (SD3-Medium)

- **24 multimodal (double-stream) blocks** — image and text processed in parallel
- **0 unified (single-stream) blocks** — SD3-Medium has `depth_unified=0`; "0 unified hooks" is correct
- **Block input shape:** `(batch=2, seq_len, 1, hidden=1536)` — 4D with singleton dim (CFG doubles batch)
- **Post-GELU layers:** `mlp.fc2` inputs in each block — right-skewed, need shift centering
- **adaLN modulation:** Pre-computed for all timesteps, then zeroed; must reload with `load_weights(only_modulation_dict=True)` between images

---

## Tests

```bash
# All pure-logic tests (no model required, ~3s, 212 tests)
conda run -n diffusionkit python -m pytest tests/ -v

# Per-script
conda run -n diffusionkit python -m pytest tests/test_cache_adaround_data.py -v
conda run -n diffusionkit python -m pytest tests/test_adaround_optimize.py -v
conda run -n diffusionkit python -m pytest tests/test_load_adaround_model.py -v
conda run -n diffusionkit python -m pytest tests/test_analyze_activations.py -v
```

---

## Known Issues / Design Decisions

**adaLN reload between images (`cache_adaround_data.py`, `collect_layer_activations.py`):**
`cache_modulation_params` zeroes adaLN weights after the final timestep. Two separate
`try/except` blocks ensure `clear_modulation_params_cache()` failures never prevent
`load_weights(only_modulation_dict=True)` from running.

**BlockHook is not an `nn.Module` (`cache_adaround_data.py`):**
Hooks are installed only for the forward passes — never during `cache_modulation_params`
or `load_weights` calls — so MLX's module-tree walk reaches the real adaLN parameters.

**V2 activation quantization (`load_adaround_model.py --quant-config`):**
`_ActQuantLayer` proxies wrap every `nn.Linear` in the MMDiT. A custom Euler loop
(matching `generate_calibration_data.py`) threads the current step_key into proxies
before each denoising step, enabling per-(layer, timestep) fake quantization.
`apply_act_quant_hooks()` / `remove_act_quant_hooks()` install and restore the proxies;
they do NOT interfere with AdaRound weight injection since weight injection runs first.
