# Polynomial Clipping: Technical Reference

Timestep-aware activation quantization for SD3 Medium's MMDiT. Each of the 285 linear layers gets a polynomial `α(σ)` mapping noise level to clipping range, replacing the static per-layer clipping used in standard PTQ.

---

## 1. Why Polynomial Clipping

Diffusion models run the same network 20–50 times per image at different noise levels σ. The activation distribution at each layer changes with σ — a clipping range calibrated at one noise level is wrong at every other.

SD3's dual-stream MMDiT makes this worse: the image and text streams have different scales within shared attention tensors, and these scales diverge as denoising progresses. A single static α per layer is always a bad compromise.

**Key findings from profiling** (Phase 1 EDA, 30 calibration images × 25 σ steps × 285 layers):

| Failure mode | Example |
|-------------|---------|
| Cross-stream scale mismatch | mm3 img/txt ratio: 1.65× at σ=1.0 → 3.58× at σ=0.09 |
| adaLN-induced distribution shift | mm22 txt mlp_fc2: shift of 254 units (typical scale: 1–5) |
| Non-linear drift | mm9 img mlp_fc2: linear R²=0.14, quadratic R²=0.71 (U-shaped) |
| Opposite trajectory directions | Image attention scales rise during denoising; text scales fall |

Plotting any layer's absmax activation magnitude against σ gives a smooth, reproducible curve — a consequence of rectified flow's linear interpolation `x_t = (1−t)·x₀ + t·ε`. Smooth input trajectories produce smooth activation trajectories, which low-degree polynomials capture well.

---

## 2. How It Works

### 2.1 Data collection

Collect per-layer activation statistics across denoising steps using calibration images.

**Pipeline**: `src/build_activation_stats.py` → `calibration_data_100/activations/`

For each calibration image, the denoiser runs all 25 σ steps. At each step, hooks on every linear layer record:
- Per-channel min/max
- Histogram (for percentile extraction)

Output structure:
```
calibration_data_100/activations/
├── layer_statistics.json          # metadata, sigma_map (step_idx → σ value)
└── timestep_stats/
    ├── step_0_index.json          # per-layer {tensor_min, tensor_max, p999, ...}
    ├── step_0.npz                 # per-channel min/max arrays
    ├── step_1_index.json
    └── ...
```

The `layer_statistics.json` contains a `sigma_map` mapping integer step indices to their σ values.

### 2.2 Percentile trajectory extraction

From the per-step statistics, extract the **p100 (absmax) of absolute activation magnitude** for each layer at each σ. This gives one trajectory per layer: a sequence of (σ, absmax) pairs.

The p100 percentile (true maximum) is used so that no activations are clipped — every value fits within the quantization range. This eliminates clipping error entirely, ensuring that weight optimization (GPTQ) sees reconstruction error due only to weight quantization, not confounding clipping artifacts. The cost is a slightly coarser quantization grid (wider α), but the rounding decisions are correct under all inference conditions.

### 2.3 Polynomial fitting

**Script**: `python -m src.generate_poly_schedule`

For each layer's trajectory, fit a polynomial using tiered degree selection:

```
α(σ) = cₙσⁿ + ... + c₁σ + c₀
```

**Degree selection tiers**:

| Condition | Selected degree |
|-----------|----------------|
| Quadratic R² > 0.85 | 2 |
| Cubic R² gain > 0.15 over quadratic | 3 |
| Quartic R² gain > 0.10 over cubic | 4 |
| None of the above | 2 (fallback — still better than static) |

The fitting uses `np.polyfit` for coefficient estimation and R² = 1 − SS_res/SS_tot for quality assessment.

**Result for SD3 Medium**:

| Degree | Count | Fraction | Meaning |
|--------|-------|----------|---------|
| 0 (static) | 227 | 79.6% | Activation scale barely changes with σ |
| 2 (quadratic) | 57 | 20.0% | Smooth parabolic trajectory |
| 3 (cubic) | 1 | 0.4% | Additional curvature needed |

Median R² for quadratic fits: **0.944**. The trajectories are genuinely smooth — the polynomial captures 94.4% of the variance.

### 2.4 Schedule file format

**File**: `polynomial_clipping_schedule.json`

```json
{
  "version": "poly_v1",
  "percentile": "p100_absmax",
  "sigma_range": [0.0934, 1.0],
  "layers": {
    "mm0_img_attn_k_proj": {
      "degree": 0,
      "coeffs": [2.484],
      "r2": 1.0,
      "cv": 0.0936
    },
    "mm0_img_attn_o_proj": {
      "degree": 2,
      "coeffs": [-1.854, 1.235, 2.178],
      "r2": 0.9342,
      "cv": 0.1408
    }
  }
}
```

Fields per layer:
- **`degree`**: polynomial degree (0, 2, 3, or 4)
- **`coeffs`**: coefficients in `np.polyval` convention (highest degree first). For degree 2: `[c₂, c₁, c₀]`
- **`r2`**: goodness of fit (1.0 for degree-0 layers by convention)
- **`cv`**: coefficient of variation of the raw trajectory (std/mean)

Total storage: 402 coefficients for 285 layers.

---

## 3. Runtime Evaluation

At each denoising step with noise level σ:

**Step 1 — Evaluate polynomial**:
```python
alpha = np.polyval(coeffs, sigma)   # e.g. c₂·σ² + c₁·σ + c₀
```
For degree-0 layers, this is just the stored constant.

**Step 2 — Compute quantization scale** (symmetric INT8):
```
scale = α / 127
```

**Step 3 — Fake-quantize activations**:
```
x_q = clip(round(x / scale), −127, 127) × scale
```

This maps the activation range `[−α, +α]` onto the integer grid `[−127, 127]`. Each quantization step covers `α/127` activation units.

**Overhead**: Evaluating a degree-2 polynomial is 2 multiplies and 2 adds per layer per step — negligible next to the matrix multiplications.

---

## 4. Integration with Weight Quantization

The polynomial controls **activation** clipping. Weight quantization (GPTQ or AdaRound) is a separate step that benefits from correct activation clipping in two ways:

### 4.1 During weight optimization

When GPTQ accumulates the Hessian `H = Xᵀ X`, the input activations X should reflect what the quantized model actually sees at inference. With poly-aware Hessian collection:

```python
alpha = np.polyval(poly_coeffs, sigma)   # correct α for this sample's noise level
scale = alpha / 127
x_q = fake_quant(x, scale, bits=8)       # fake-quantize before accumulating
H += x_q.T @ x_q                         # Hessian reflects quantized activations
```

Without this, the Hessian is computed on FP16 activations — a distribution the quantized model never sees, leading to suboptimal weight quantization.

### 4.2 At inference

After weight quantization, the polynomial schedule provides per-step activation clipping during image generation:
1. Load quantized weights (int4/int8) and the poly schedule
2. At each denoising step, evaluate `α(σ)` per layer
3. Fake-quantize activations with the layer-specific scale before each linear

---

## 5. Design Decisions

### Why p100 (absmax)

The schedule uses **p100 (true maximum)** — no activation values are clipped. This means:
- Zero clipping error by construction
- The quantization grid is slightly coarser (wider α → larger step size), but every value is representable
- Weight optimization (GPTQ) sees reconstruction error purely from weight quantization, not confounding clipping artifacts

We previously used p99.9 and found that weights optimized against p99.9-clipped activations produced **worse image quality than round-to-nearest** when tested with FP16 activations — the optimizer had tuned rounding decisions for a clipped distribution that didn't match inference conditions.

For future inference-only A8 quantization (no weight optimization coupling), p99.9 may recover some precision by tightening α, but this is a separate concern from the schedule used during weight quantization.

### Per-stream fitting

Image and text streams get separate polynomials. This is critical because the two streams move in opposite directions during denoising (image scales rise, text scales fall). A single polynomial per joint-attention layer cannot capture opposite trends.

### Sigma range clamping

Polynomials are only valid within the calibration range `[0.0934, 1.0]`. At inference, σ values outside this range should be clamped before evaluation to avoid extrapolation artifacts.

---

## 6. Layer Naming Convention

Schedule keys follow the pattern `mm{block}_{stream}_{module}_{proj}`:

| Component | Values |
|-----------|--------|
| Block | `mm0` through `mm23` |
| Stream | `img`, `txt` |
| Module | `attn`, `mlp` |
| Projection | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `fc1`, `fc2` |

Examples:
- `mm0_img_attn_q_proj` — Block 0, image stream, attention Q projection
- `mm12_txt_mlp_fc2` — Block 12, text stream, FFN second linear
- Block 23 txt has no `o_proj`, `fc1`, `fc2` entries (skip_post_sdpa=True)

---

## 7. Usage

### Generate the schedule

```bash
python -m src.generate_poly_schedule \
    --activations-dir calibration_data_100/activations \
    --percentile p100_absmax \
    --output polynomial_clipping_schedule.json
```

Options:
- `--percentile`: `p99`, `p999`, `mean_absmax`, `p100_absmax`
- `--include-shifts`: also fit shift (center) polynomials for asymmetric quantization

### Use at inference

```bash
python -m src.benchmark_model generate \
    --poly-schedule polynomial_clipping_schedule.json \
    ...
```

### Validate generalization

```bash
python -m src.validate_poly_generalization \
    --activations-dir calibration_data_100/activations \
    --schedule polynomial_clipping_schedule.json
```

---

## 8. Files

| File | Role |
|------|------|
| `src/generate_poly_schedule.py` | Fits polynomials to activation trajectories, outputs schedule JSON |
| `src/validate_poly_generalization.py` | Cross-validates schedule against held-out calibration groups |
| `src/build_activation_stats.py` | Collects per-timestep activation statistics from calibration runs |
| `src/benchmark_model.py` | Loads schedule, applies activation quantization hooks at inference |
| `polynomial_clipping_schedule.json` | The generated schedule (285 layers, 402 coefficients) |
| `calibration_data_100/activations/` | Raw per-timestep statistics feeding into schedule generation |
| `POLYNOMIAL_CLIPPING_EXPLAINER.md` | Extended research narrative with motivation, failure modes, and future work |
