# Poly-Clipping AdaRound W4A8 — Research Plan

SD3-Medium quantization study comparing polynomial timestep-aware activation clipping
against standard FP16, using MS-COCO val prompts as the evaluation set.

---

## Current State

| Item | Status |
|------|--------|
| Polynomial clipping schedule | ✅ Generated (`polynomial_clipping_schedule.json`, 285 layers) |
| Calibration data (100 images × 25 timesteps) | ✅ Cached (`calibration_data_100/`) |
| Pooled embeddings for calibration images | ✅ Cached (`calibration_data_100/pooled/`) |
| FP16 reference benchmark | ✅ Done (50 images, 28 steps, CFG 1.5, seed 42) |
| MS-COCO prompts CSV | ✅ Ready (`coco_prompts.csv`, 5040 captions) |
| AdaRound optimization — mm0–mm3 | ⚠️ Started but produced NaN (bad `a_scale` init — now fixed) |
| AdaRound optimization — mm4–mm37, uni0–uni23 | ❌ Not started |
| Poly-clipping benchmark run | ❌ Not started |
| σ-weighted AdaRound loss | ❌ Not started |

---

## Phase 1 — Complete AdaRound Optimization

**Goal**: Produce fully optimized W4A8 weights for all 24 MM blocks + 24 Uni blocks (SD3 has 24 MM + 24 Uni = 48 total? check exact count) using polynomial activation clipping.

**Fixes applied (commit 9a22c47)**:
- `a_scale` now initializes from `alpha(σ_median) / 127` instead of 1.0
- Gradient clipping `[-1, 1]` on `a_scale` grads prevents NaN
- Cosine LR annealing now correctly reduces LR (was computing but discarding the scaled grads)

**Command** (restart from scratch, mm0–mm3 weights are suspect from NaN runs):
```bash
conda run -n diffusionkit python -m src.adaround_optimize \
  --adaround-cache calibration_data_100 \
  --output-dir quantized_weights_poly \
  --poly-schedule polynomial_clipping_schedule.json \
  --iters 1000 \
  --batch-size 8
```

Add `--resume` if the run is interrupted partway through.

**Expected**: Initial loss should drop from ~900K to ~1K–10K range. Monitor the
`a_scale init from poly @ σ=...` line — values should be in the range [0.5, 50]
(not 1.0 for all layers). If loss at iter 0 is still >100K, something is wrong.

**Estimated output**: `quantized_weights_poly/weights/{block_name}.npz` for each block,
plus `quantized_weights_poly/quant_config.json`.

---

## Phase 2 — Baseline Benchmark (Poly-Clipping AdaRound vs FP16)

**Goal**: Quantitatively compare the poly-clipping W4A8 model to FP16 on image quality.

### 2a. Run poly-clipping benchmark

```bash
conda run -n diffusionkit python -m src.benchmark_model \
  --config adaround_w4a8_poly \
  --adaround-weights quantized_weights_poly/weights \
  --poly-schedule polynomial_clipping_schedule.json \
  --prompt-csv coco_prompts.csv \
  --num-images 500 \
  --num-steps 28 \
  --cfg-scale 1.5 \
  --seed 42 \
  --output-dir benchmark_results/poly_baseline
```

### 2b. Run matching FP16 benchmark (same prompts/seed)

The existing `fp16_ref` used only 50 images. Re-run with 500 images for statistical parity:
```bash
conda run -n diffusionkit python -m src.benchmark_model \
  --config fp16 \
  --prompt-csv coco_prompts.csv \
  --num-images 500 \
  --num-steps 28 \
  --cfg-scale 1.5 \
  --seed 42 \
  --output-dir benchmark_results/fp16_500
```

### 2c. Compare results

```bash
python -m src.compare_benchmarks \
  benchmark_results/fp16_500 \
  benchmark_results/poly_baseline
```

**Metrics tracked**: sFID, Precision, Recall, latency (mean/p95), peak memory, model size.

**Reference**: TaQ-DiT used CFG 1.5, 100 steps, 10K images for their paper results.
500 images / 28 steps is a reasonable faster proxy — scale up to 10K images / 100 steps
for final paper-quality numbers.

---

## Phase 3 — σ-Weighted AdaRound Loss (Extension)

**Goal**: Weight each calibration sample's reconstruction loss by its importance to
the final image (σ → 0 steps matter more than σ → 1 steps) and measure whether this
improves quantization quality.

**Theory**: The AdaRound loss is currently a uniform average across all timesteps.
In diffusion models, the clean-image prediction steps (low σ) have larger perceptual
impact than the early denoising steps (high σ). Weighting by `1/α(σ)` (the inverse of
the clipping range, from the poly schedule) emphasizes these precision-critical steps.

### Changes needed in `src/adaround_optimize.py`

In the inner loss accumulation loop, multiply each sample's loss by its σ-weight:

```python
# σ-weight: inversely proportional to activation scale (low-σ = small activations = higher weight)
if sample_sigmas is not None:
    sigma_val = float(sample_sigmas[si])
    sigma_weight = 1.0 / (sigma_val + 1e-6)   # or use poly alpha as proxy
    # normalize across batch so total scale doesn't change
    loss_i = loss_i * sigma_weight
```

Then normalize the weights sum to 1.0 across the mini-batch.

**Implementation note**: Re-run AdaRound with `--sigma-weighted` flag after Phase 2
is done. Do NOT re-run Phase 1 — reuse `quantized_weights_poly` as the warm-start
(`--refine` flag).

### Benchmark σ-weighted variant

```bash
# After re-optimizing with sigma weighting:
python -m src.compare_benchmarks \
  benchmark_results/poly_baseline \
  benchmark_results/poly_sigma_weighted
```

---

## Phase 4 — Scale Up (Paper-Quality Numbers)

Once the approach is validated on 500 images:

1. **Regenerate calibration data with MS-COCO prompts** (currently using generic prompts)
   — may improve calibration quality since test-time prompts will be more representative
2. **Re-run benchmark at TaQ-DiT scale**: 10K images, 100 steps, CFG 1.5
3. **Add CLIP score** to `benchmark_model.py` alongside sFID/P/R

---

## File Reference

| File | Purpose |
|------|---------|
| `src/adaround_optimize.py` | Main optimization loop — run this first |
| `src/cache_adaround_data.py` | Cache FP16 block I/O + pooled embeddings |
| `src/load_adaround_model.py` | Load quantized model + poly activation hooks |
| `src/benchmark_model.py` | Generate images + compute metrics |
| `src/compare_benchmarks.py` | Side-by-side metric comparison |
| `src/generate_poly_schedule.py` | Fit polynomials to activation trajectories |
| `src/download_coco_prompts.py` | Download MS-COCO val captions as prompt CSV |
| `polynomial_clipping_schedule.json` | Per-layer poly coefficients (285 layers) |
| `calibration_data_100/` | Cached block I/O for AdaRound (100 images × 25 steps) |
| `quantized_weights_poly/` | Output of AdaRound optimization |
| `benchmark_results/fp16_ref/` | FP16 reference (50 images, 28 steps) |
| `coco_prompts.csv` | 5040 MS-COCO val captions for benchmarking |

---

## Immediate Next Step

**Re-run AdaRound** with the fixed `a_scale` initialization. Watch the first block's
`iter 0` loss — it should be in the thousands, not hundreds of thousands.

```bash
rm -rf quantized_weights_poly/weights/mm*.npz   # clear the NaN-corrupted weights
conda run -n diffusionkit python -m src.adaround_optimize \
  --adaround-cache calibration_data_100 \
  --output-dir quantized_weights_poly \
  --poly-schedule polynomial_clipping_schedule.json \
  --iters 1000 \
  --batch-size 8
```
