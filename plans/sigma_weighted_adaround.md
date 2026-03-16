# σ-Weighted AdaRound Loss + Benchmark Evaluation

## Context

The baseline AdaRound W4A8 optimization with polynomial activation clipping is completing (3000 iters). Currently, all calibration samples are weighted equally in the reconstruction loss regardless of their noise level σ. In diffusion models, low-σ timesteps (near-clean images) have disproportionate perceptual impact — quantization errors at these steps directly affect fine detail. Weighting the AdaRound loss by σ should improve perceptual quality of the quantized model even if the raw (unweighted) reconstruction loss appears higher.

**Goal**: Implement σ-weighted AdaRound loss, run warm-started refinement from baseline weights, and benchmark against the baseline using CMMD/sFID/precision/recall.

---

## Phase 1: Preserve Baseline Weights

Before any code changes, copy the baseline output:

```bash
cp -r quantized_weights_poly quantized_weights_poly_baseline
```

---

## Phase 2: Implement σ-Weighted Loss (DONE)

**File**: `src/adaround_optimize.py` (only file modified)

### Changes made (~50 lines net new/changed):

1. **`_compute_sigma_weights()` helper** (~line 583) — computes `w(σ) = 1/(σ + offset)`, normalized so mean weight = 1.0. Low-σ (clean) samples get higher weight. `offset=1.0` gives ~15x ratio between σ=0.03 and σ=14.6.

2. **Two CLI args**: `--sigma-weighted` (flag) and `--sigma-weight-offset` (default 1.0), with validation requiring `--poly-schedule`.

3. **`optimize_block()` signature** — added `sigma_weighted: bool = False` and `sigma_weight_offset: float = 1.0`.

4. **Weight precomputation** — after sample count, computes and prints weight range at block start.

5. **Inner loop** — applies `w_i` to both loss and gradients per sample; tracks `total_loss_uw` (unweighted) and `weight_sum` alongside existing `total_loss` and `n_valid`.

6. **Grad averaging** — uses `weight_sum` instead of `n_used` in weighted mode (backward compatible: without flag, `weight_sum == n_valid`).

7. **Logging** — shows both `loss(w)` and `loss(uw)` when sigma-weighted is active; normal mode unchanged.

8. **Both call sites** (standard ~line 1447 + HTG ~line 1314) — `sigma_weighted` and `sigma_weight_offset` threaded through.

9. **Config output** — saves `sigma_weighted` and `sigma_weight_offset` in `quant_config.json`.

**Normal adaround is unaffected** — without `--sigma-weighted`, all weights are 1.0 and code paths are identical.

### Why `1/(σ+1)` over alternatives:
- `1/α(σ)` couples weighting to clipping schedule — conflates two concerns
- `exp(-σ)` drops to near-zero too fast for high-σ samples
- SNR-based `α²/σ²` is more principled but harder to tune
- `1/(σ+1)` is simple, interpretable, and `offset` is easy to sweep

---

## Phase 3: Run σ-Weighted Training

Warm-start from baseline weights using `--refine`:

```bash
conda run -n diffusionkit python -m src.adaround_optimize \
  --adaround-cache calibration_data_100 \
  --output quantized_weights_poly_sigma \
  --poly-schedule polynomial_clipping_schedule.json \
  --iters 3000 \
  --batch-size 8 \
  --sigma-weighted \
  --sigma-weight-offset 1.0 \
  --refine quantized_weights_poly_baseline/weights
```

**Note**: The weighted loss values are NOT directly comparable to baseline loss values. A higher weighted loss does not mean worse quality — the weighting redistributes emphasis. Only benchmark metrics (Phase 4) reveal true quality differences.

---

## Phase 4: Benchmark Comparison

### 4a. Generate images — σ-weighted model

```bash
conda run -n diffusionkit python -m src.benchmark_model \
  --config adaround_w4a8_poly_sigma \
  --adaround-weights quantized_weights_poly_sigma/weights \
  --poly-schedule polynomial_clipping_schedule.json \
  --prompt-csv coco_prompts.csv \
  --num-images 500 \
  --num-steps 28 --cfg-scale 1.5 --seed 42 \
  --output-dir benchmark_results/poly_sigma_weighted
```

### 4b. Generate images — baseline model (if not already done)

```bash
conda run -n diffusionkit python -m src.benchmark_model \
  --config adaround_w4a8_poly \
  --adaround-weights quantized_weights_poly_baseline/weights \
  --poly-schedule polynomial_clipping_schedule.json \
  --prompt-csv coco_prompts.csv \
  --num-images 500 \
  --num-steps 28 --cfg-scale 1.5 --seed 42 \
  --output-dir benchmark_results/poly_baseline
```

### 4c. Compare

```bash
python -m src.compare_benchmarks \
  benchmark_results/poly_baseline \
  benchmark_results/poly_sigma_weighted
```

**Metrics** (all already implemented in `benchmark_model.py`):
- **CMMD** — reliable at 100-1K images (vs 10-50K for FID)
- **sFID** — spatial frequency-aware FID
- **Precision/Recall/Density/Coverage** (PRDC)
- **Latency** (mean/p95), peak memory, model size

---

## Critical Files

| File | Role |
|------|------|
| `src/adaround_optimize.py` | **Only file modified** — weight helper, CLI args, loop changes |
| `src/benchmark_model.py` | Generates images + computes CMMD/sFID/PRDC (no changes needed) |
| `src/compare_benchmarks.py` | Side-by-side comparison (no changes needed) |
| `quantized_weights_poly_baseline/` | Preserved baseline weights (copy before modifying) |

---

## Verification

1. Run `--sigma-weighted` on a single block (mm_block_0) for 100 iters — confirm both weighted and unweighted loss are logged and loss decreases
2. Run `--refine` with sigma-weighted — confirm it loads baseline weights correctly and starts from reasonable loss
3. Generate 10 test images with both models — visual sanity check
4. Full 500-image benchmark with CMMD comparison
