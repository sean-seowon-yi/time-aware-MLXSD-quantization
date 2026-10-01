# SmoothQuant for W4A8 Diffusion Quantization

## What SmoothQuant Does

SmoothQuant migrates outlier magnitude from activations to weights via per-channel scaling,
making the activation distribution more uniform without changing the mathematical result.

**The transform:**
```
y = W * x = (W * diag(s)) * (diag(1/s) * x) = W' * x'
```

- Weight side: `W'[i,j] = W[i,j] * s[j]` — absorbed offline, re-quantized to W4
- Activation side: `x'[j] = x[j] / s[j]` — applied at inference before fake-quant

The product `W' @ x'` is mathematically identical to `W @ x`. The quantization
error changes because each side now has a different distribution to quantize.

---

## The Scale Formula

```
s[c] = s_act[c]^α / s_w[c]^(1-α)
```

Where:
- `s_act[c]` = per-channel activation magnitude (max abs across calibration data)
- `s_w[c]` = per-column weight magnitude (max abs across output rows)
- `α` = migration strength (0 to 1)

### Alpha (α) — Migration Strength

Alpha controls how much outlier magnitude moves from activations to weights:

| Alpha | Effect | Scale becomes |
|-------|--------|---------------|
| 0.0 | No migration — all burden on activations | `s = 1 / s_w` (useless) |
| 0.5 | **Balanced** — split equally between sides | `s = √(s_act / s_w)` |
| 1.0 | Full migration — all burden on weights | `s = s_act` |

**Why α=0.5 is the default:**
- The original SmoothQuant paper recommends 0.5 as a starting point.
- At W4 (16 quantization levels), pushing too much to weights is dangerous —
  amplified weight columns lose precision in fewer quantization bins.
- α=0.5 gives each side an equal share of the outlier handling burden.

**Tuning guidance:**
- `α=0.6`: push slightly more to weights — try if activation quantization is
  still the bottleneck after α=0.5.
- `α=0.3–0.4`: more conservative — useful if W4 weight quality degrades with
  standard α=0.5.
- Higher α helps activations more but hurts weight quantization; lower α
  is the reverse.

### Scale Clip — Safety Cap on Per-Channel Scales

```python
s = np.clip(s, 1.0 / scale_clip, scale_clip)
```

Without clamping, extreme outlier channels can produce `s[c] = 500+`, meaning
that weight column gets multiplied by 500×. At W4 with per-output-channel
quantization scales, that column's values collapse into just a few quantization
levels — destroying precision for an entire row of outputs.

| scale_clip | Max amplification | Use case |
|------------|-------------------|----------|
| 0 (off) | Unlimited | W8 or higher bit-widths that can absorb large scales |
| 16 | 16× | Conservative for W4 — limits worst-case precision loss |
| **32** | **32×** | **Recommended starting point for W4** |
| 64 | 64× | Aggressive — allows more outlier absorption but riskier for W4 |

**Why scale_clip matters at W4:**
- W4 has only 16 quantization levels per output channel.
- If a weight column is amplified 500×, the per-output-channel scale must
  grow to accommodate it. This means all other values in that output row
  are quantized with a much coarser step size.
- `scale_clip=32` caps the worst-case precision loss while still allowing
  substantial outlier migration. Most channels have scales well below 32.

---

## Two Implementation Paths

### Path A: Quick Inference Test (RTN Re-quantization)

Steps: compute scales → absorb into existing AdaRound weights → re-quantize with RTN

```bash
# 1. Compute per-channel scales
python -m src.compute_smoothquant_scales \
    --activations-dir calibration_data_512/activations \
    --weights-dir quantized_weights_w4a8_adaround_poly_p100 \
    --alpha 0.5 --scale-clip 32 \
    --output smoothquant_scales.json

# 2. Absorb into weights (RTN re-quant, NOT AdaRound)
python -m src.apply_smoothquant \
    --weights-dir quantized_weights_w4a8_adaround_poly_p100 \
    --scales smoothquant_scales.json \
    --output-dir quantized_weights_w4a8_smoothquant_absorbed

# 3. Generate smoothed poly schedule
python -m src.generate_poly_schedule \
    --activations-dir calibration_data_512/activations \
    --smoothquant-scales smoothquant_scales.json \
    --output polynomial_clipping_schedule_smoothquant.json

# 4. Inference
python -m src.load_adaround_model \
    --adaround-output quantized_weights_w4a8_smoothquant_absorbed \
    --poly-schedule polynomial_clipping_schedule_smoothquant.json \
    --smoothquant-scales smoothquant_scales.json \
    --prompt "..." --output-image test_sq.png --compare
```

**Limitation:** AdaRound's learned rounding decisions are discarded (the weight
distribution shifted; they're invalid). RTN at W4 is significantly worse than
AdaRound — Path A is only a lower-bound quality estimate.

### Path B: Proper AdaRound from FP16 (Re-training)

Steps: compute scales → generate smoothed poly schedule → run AdaRound with `--smoothquant-scales`

```bash
# Use steps 1 and 3 from Path A, then:
python -m src.adaround_optimize \
    --adaround-cache calibration_data_512/adaround_cache \
    --output quantized_weights_w4a8_adaround_sq_pathb \
    --poly-schedule polynomial_clipping_schedule_smoothquant.json \
    --smoothquant-scales smoothquant_scales.json \
    --iters 20000

# Inference (same as Path A step 4 but with Path B weights)
python -m src.load_adaround_model \
    --adaround-output quantized_weights_w4a8_adaround_sq_pathb \
    --poly-schedule polynomial_clipping_schedule_smoothquant.json \
    --smoothquant-scales smoothquant_scales.json \
    --prompt "..." --output-image test_sq_pathb.png --compare
```

**How it works in `adaround_optimize.py`:**
- FP16 weights are pre-scaled: `W' = W * diag(s)` before AdaRound optimization
- `_QuantProxy` divides activations by `s[c]` before fake-quant during training
- AdaRound learns optimal rounding on the smoothed distribution
- Result: true AdaRound rounding decisions on SQ-smoothed weights

---

## Effect on Polynomial Clipping Schedule

SmoothQuant dramatically flattens activation trajectories across timesteps:

| | Without SQ | With SQ (α=0.5, clip=32) |
|---|---|---|
| Static (degree 0) | 87 layers | **277 layers** |
| Polynomial (degree 1+) | 198 layers | **8 layers** |
| Median α | Large (hundreds for outlier layers) | **1.6** |

**Why this happens:**
- Per-channel outliers were the primary source of σ-dependent activation variation.
- Once outlier magnitude is absorbed into weights, the remaining activation
  distribution is nearly constant across timesteps.
- The 8 remaining polynomial layers are all text MLP (mm2, mm14, mm20-22
  fc1/fc2) — post-GELU layers with the strongest outlier patterns that SQ
  couldn't fully flatten.

**Implication for weighted AdaRound:**
- Before SQ, sigma-weighting and derivative-weighting were fighting outlier-driven
  clipping error — the optimizer was spending effort on extreme clipping scenarios
  that dominated the loss.
- With SQ flattening those ranges, the optimizer has a clean signal. Sigma-weighting
  can now focus on what actually matters: allocating effort to timesteps that
  contribute most to image quality.
- Derivative-weighting has less signal (only 8 polynomial layers remain), but
  sigma-weighting may now show the benefit it couldn't before.
- The tight α distribution (median 1.6) means W4 rounding decisions are less
  stressed overall — there is more headroom for weighting to produce visible gains.

---

## Group Quantization (`--group-size`)

Group quantization is an orthogonal improvement to weight quantization that pairs well
with SmoothQuant. Instead of one scale per output row, the input channels are split into
groups of `group_size` consecutive columns, each with their own scale.

### The Problem with Per-Row Scales

Standard W4 uses a single absmax scale per output row (shape `(out, 1)`). If one column
in that row has a large outlier, the scale inflates to accommodate it, and all other
values in the row are quantized with a coarser step size. With only 16 quantization levels,
this wastes most of the dynamic range on a single outlier.

### How Group Quantization Helps

With `group_size=64` and `d_model=1536`:

| | Per-row (default) | Group (group_size=64) |
|---|---|---|
| Scales per row | 1 | 1536/64 = **24** |
| Scale shape | `(out, 1)` | `(out, 24)` compact |
| Outlier blast radius | Entire row (1536 values) | 64 values in that group |

For MLP layers with `in_features=6144`: 6144/64 = 96 scales per row.

### Why Consecutive Groups (Not Sorted)

Groups are simply consecutive channels — columns 0–63, 64–127, etc. No sorting or
reordering by magnitude.

An alternative would be to sort channels by absmax and group outliers together, giving
tighter quantization bins per group. However, this requires storing a permutation index
per layer and applying a gather operation on every forward pass — adding complexity and
latency.

The consecutive approach is free at inference (just a reshape), and in practice captures
most of the benefit since outlier channels tend to cluster in specific regions rather
than being uniformly spread across all positions.

### Storage

Scales are stored in compact form `(out, n_groups)` in the NPZ files, then expanded
to `(out, in)` at load time via `expand_group_scale()`. The storage overhead is small:
24 float16 values per row instead of 1.

### Usage

```bash
# AdaRound with group quantization
python -m src.adaround_optimize \
    --adaround-cache calibration_data_512/adaround_cache \
    --output quantized_weights_w4a8_group64 \
    --poly-schedule polynomial_clipping_schedule.json \
    --group-size 64

# Can combine with SmoothQuant
python -m src.adaround_optimize \
    --adaround-cache calibration_data_512/adaround_cache \
    --output quantized_weights_w4a8_sq_group64 \
    --poly-schedule polynomial_clipping_schedule_smoothquant.json \
    --smoothquant-scales smoothquant_scales.json \
    --group-size 64
```

---

## Experimental Results (2026-03-27)

### SmoothQuant + W4 Fails; Poly-Only Works

Two test images were compared against the FP16 baseline (fluffy cat prompt):

| Config | Result |
|--------|--------|
| AdaRound + p100 poly clipping (no SQ) | **Coherent image** — recognizable cat, some softness and blocking artifacts |
| AdaRound + p100 poly clipping + SmoothQuant (α=0.5) | **Collapsed** — no structure, severe noise/artifact pattern |

The SmoothQuant result is consistent with **W4 weight quantization collapse**, not
activation quantization failure. The image has no coherent structure at all — this is
what happens when weight scales are so wide that most weights collapse to the same few
quantization levels.

### Why SmoothQuant Hurt W4

SmoothQuant migrates outlier magnitude from activations into weights (`W' = W * diag(s)`).
For layers with large per-channel outliers (e.g. mm22 txt mlp_fc2 with activation shifts
of 254 units), the SmoothQuant scales `s[c]` are large. Multiplying weights by large `s`
values widens the weight distribution, forcing the per-row W4 scale to expand to
accommodate the new range.

With only 16 quantization levels and a much wider range:
- Most weights that were previously well-distributed across ±7 collapse toward zero
- A few extreme columns dominate the scale, wasting precision for the entire row
- **Bucket utilization tanks** — per-row scales are calibrated to the amplified outlier
  columns, not to the bulk of the weight distribution

This is the fundamental incompatibility: SmoothQuant was designed for W8 where 256 levels
can absorb scale migration. At W4, the 16-level budget is too tight.

### The per-row scale problem is visible in activation trajectory plots

The channel spread panels (row 3) in `calibration_data_512/activation_trajectory_plots/`
show that for most layers the per-channel std is small relative to absmax — meaning the
extreme values come from a few outlier channels. SmoothQuant absorbs exactly those
channels into the weights, which is where bucket utilization degrades most severely.

### Next Steps

1. **Measure W4 bucket utilization** directly from quantized weight NPZs — see
   [Bucket Utilization Diagnostic](#bucket-utilization-diagnostic) below.
2. **Group quantization** (`--group-size 64`) without SmoothQuant: finer per-group scales
   reduce the blast radius of within-row outliers without migrating anything into weights.
3. If group quant improves quality: the remaining softness may be addressable with
   sigma-weighting or derivative-weighting (now that activation clipping is solved by p100 poly).

### Bucket Utilization Results (2026-03-27)

Ran `src/analyze_weight_utilization.py` on `quantized_weights_w4a8_adaround_poly_p100`
with simulated RTN group quantization at gs=32, 64, 128:

| Group size | Entropy (bits) | Δ entropy | Fill % | Saturation % |
|---|---|---|---|---|
| per-row (0) | 2.818 | — | 86.0% | 0.07% |
| 128 | 3.178 | +0.360 | 92.4% | 0.96% |
| **64** | **3.265** | **+0.447** | **92.1%** | **1.80%** |
| 32 | 3.303 | +0.485 | 91.6% | 3.30% |

**Worst layers (per-row baseline):**

| Layer | Entropy | Fill % |
|---|---|---|
| mm21 txt mlp_fc1 | 1.866 | 55.9% |
| mm22 txt mlp_fc1 | 2.015 | 62.4% |
| mm20 txt mlp_fc1 | 2.029 | 64.4% |
| mm22 img attn_k_proj | 2.073 | 75.1% |

The txt mlp_fc1 layers in mm20–22 (the extreme-shift blocks) have the worst utilization —
under 2 bits entropy, nearly half the 16 quantization levels going unused.

**gs=64 is the sweet spot.** gs=32 gains only +0.038 bits over gs=64 but saturation
nearly doubles (3.3% vs 1.8%) — groups small enough that single outlier weights start
saturating group boundaries. gs=64 captures the bulk of the benefit cleanly.

**Next experiment:** AdaRound with `--group-size 64` and p100 poly schedule.

---

## Bucket Utilization Diagnostic

To measure how well W4 is using its 16 quantization levels, analyze the `weight_int`
arrays in the quantized NPZ files. Key metrics per layer, per row (or per group):

**Unique levels used:** How many of the 16 possible values [-8..7] appear at all.
A row using only 6 levels has wasted 10/16 of its representational capacity.

**Entropy:** `H = -Σ p_i log2(p_i)` over the 16 bins. Maximum is 4 bits (uniform).
Entropy < 2 bits means the distribution is highly concentrated — poor utilization.

**Saturation rate:** Fraction of weights at ±7 (the extremes). High saturation means
the scale was too small and values are being clipped to the quantization boundary.

Compare these metrics across:
- Current per-row W4 (group_size=0) on the existing quantized weights
- Simulated group_size=64 and group_size=128 (recompute scales from FP16 weights)

Layers with lowest entropy or highest saturation are the ones group quantization
will help most. If those layers correlate with the blocks that look worst in the
trajectory plots (large absmax/p999 ratio, high channel spread), it confirms the
within-row outlier hypothesis.

---

## Implementation Details

### Activation Side (Inference)

In `_ActQuantLayer.__call__()`:
```python
# 0. SmoothQuant: divide by per-channel scale
if self.sq_scale is not None:
    x = x / mx.array(self.sq_scale)

# 1-5. Shift, outlier scaling, fake-quant, restore outlier, un-shift
# ... (standard activation quantization pipeline)

# No SQ restore — absorbed weights W' = W*diag(s) provide the inverse:
# W' @ fake_quant(x/s) = W*diag(s) @ fake_quant(x/s) ≈ W @ x
return self.layer(x)
```

**Critical:** SmoothQuant is NOT un-done after fake-quant. The absorbed weights
already contain `diag(s)`, so `W' @ x'` recovers `W @ x` without restoring `x`.
Un-doing the scale would introduce a spurious `s²` factor per channel.

### Weight Side (AdaRound Training)

In `optimize_block()`:
```python
# Pre-scale FP16 weights before AdaRound optimization
W_fps_np = [np.array(l.weight) for l in linear_layers]
if sq_scales is not None:
    W_fps_np[i] = W[i] * s[np.newaxis, :]  # W' = W * diag(s)

# _QuantProxy divides activations by s[c] during block reconstruction
# AdaRound learns rounding on the smoothed distribution
```
