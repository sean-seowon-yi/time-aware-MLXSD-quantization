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
               │ (int8 W4 scales)  │    │ (A4/A6/A8 per layer,    │
               │                   │    │  shifts, SmoothQuant)   │
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
conda run -n diffusionkit python -m src.analyze_activations \
    --stats /path/to/calibration_data/activations/layer_statistics.json \
    --output /path/to/calibration_data/activations/quant_config.json
```

**Why:** Converts raw per-channel statistics into per-layer, per-timestep quantization
decisions (A4/A6/A8) and computes the final activation scales, shift vectors, and
SmoothQuant weight-scaling factors that will be applied at inference time.

Three-tier decision logic:
- **A4** (16 levels): scale < 6.0 (or shifted scale < 5.0 for post-GELU)
- **A6** (64 levels, MLX int6): 6.0–10.0 range
- **A8** (256 levels): scale ≥ 10.0, or high p99/p50 outlier ratio
- **SmoothQuant downgrade:** isolated spike layers get one tier improvement

This produces the per-timestep config that enables TaQ-DiT's adaptive quantization:
early steps (high σ, large activations) get A8, late steps (low σ) can use A4.

**Output:**
```
calibration_data/activations/
  quant_config.json              # per_timestep: {step → layer → {bits, scale, shift}}
  layer_temporal_analysis.json   # variability stats, switcher layers, always-A8 list
```

---

## Step 4 — Validate (current V1: weights only)

```bash
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output /path/to/adaround_output \
    --prompt "a tabby cat sitting on a wooden table" \
    --output-image quant_test.png \
    --compare \
    --diff-stats
```

**V1 limitation:** Currently dequantizes INT4 weights back to FP16 at load time and
runs with FP16 activations. This validates weight quantization quality (AdaRound
rounding decisions are correct) but does not apply the activation quantization scales
from the activation track.

**V2 target:** Apply `quant_config.json` at inference — use static INT8/INT4/INT6
activation scales per-layer per-timestep, apply post-GELU shift, and apply SmoothQuant
weight pre-scaling. This requires a `QuantizedLinear` layer that holds int8 weights and
executes `(W_int8 * scale).T @ x_quantized + bias`.

---

## Quick Reference

| Step | Script | Time estimate | Output |
|------|--------|---------------|--------|
| 1. Calibration latents | `generate_calibration_data.py` | ~2–4h (50 images) | `samples/*.npz` |
| 2W. Block I/O cache | `cache_adaround_data.py` | ~30–60 min | `adaround_cache/` |
| 3W. AdaRound optimize | `adaround_optimize.py` | ~1–2h (24 blocks) | `weights/mm*.npz` |
| 2A. Activation stats | `collect_layer_activations.py` | ~30 min | `timestep_stats/` |
| 3A. Quant config | `analyze_activations.py` | <1 min | `quant_config.json` |
| 4. Validate (V1) | `load_adaround_model.py` | ~5 min | comparison PNGs |

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
# All pure-logic tests (no model required, ~3s)
conda run -n diffusionkit python -m pytest tests/ -v

# Per-script
conda run -n diffusionkit python -m pytest tests/test_cache_adaround_data.py -v
conda run -n diffusionkit python -m pytest tests/test_adaround_optimize.py -v
conda run -n diffusionkit python -m pytest tests/test_load_adaround_model.py -v
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

**V1 activation quantization:**
`load_adaround_model.py` currently runs FP16 activations. The `quant_config.json` from
`analyze_activations.py` is the calibration input for a V2 W4A8 inference path.
