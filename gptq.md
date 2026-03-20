# GPTQ for SD3 MMDiT — Research Plan

**Goal**: Replace AdaRound with GPTQ for weight quantization of Stable Diffusion 3 Medium's MMDiT, retaining the existing polynomial clipping schedule for timestep-aware activation quantization.

---

## 1. Motivation

AdaRound learns per-weight binary rounding decisions via ~1000 gradient iterations per block. With 48 quantization targets (24 MM blocks × 2 streams), this takes **~13 hours** on Apple Silicon.

GPTQ is a one-shot, Hessian-based method: accumulate input statistics, factor the Hessian, and quantize column-by-column with closed-form error compensation. Estimated total time: **~1.5 hours**.

Both methods produce per-channel symmetric quantized weights. The polynomial clipping schedule (`polynomial_clipping_schedule.json`, 285 layers) and calibration data (`calibration_data_100/`) are reusable as-is.

---

## 2. SD3 MMDiT Architecture Mapping

**Model**: `argmaxinc/mlx-stable-diffusion-3-medium`

| Parameter | Value |
|-----------|-------|
| `depth_multimodal` | 24 |
| `depth_unified` | 0 |
| `hidden_size` | 1536 (64 × 24 heads) |
| `mlp_ratio` | 4 (FC1: 1536→6144, FC2: 6144→1536) |

Each `MultiModalTransformerBlock` contains an `image_transformer_block` and a `text_transformer_block`. Each `TransformerBlock` has:

| Linear | Shape | Bias |
|--------|-------|------|
| `q_proj` | 1536→1536 | yes |
| `k_proj` | 1536→1536 | no |
| `v_proj` | 1536→1536 | yes |
| `o_proj` | 1536→1536 | yes |
| `fc1` | 1536→6144 | yes |
| `fc2` | 6144→1536 | yes |
| `adaLN_modulation.layers[1]` | 1536→9216 | yes |

**Exception**: Block 23 txt has `skip_post_sdpa=True` — missing `o_proj`, `fc1`, `fc2`. Only 2 modulation params (β₁, γ₁) instead of 6.

**Total**: ~285 linear layers across all blocks.

**Quantization targets**:
- **Weights**: W4 or W8, configurable via `--bits-w` CLI flag
- **Activations**: Always A8 (8-bit fake quantization via poly schedule)
- **Skip**: adaLN modulation layers remain FP16

---

## 3. GPTQ Algorithm

For a single linear layer with weight matrix W (d_out × d_in):

### 3.1 Hessian accumulation

Collect calibration activations X (n_samples × d_in) at the layer input:

```
H = 2 · Xᵀ X      shape: (d_in, d_in)
```

### 3.2 Damping

```
H += damp_percent · max(diag(H)) · I      damp_percent = 0.01
```

### 3.3 Cholesky factorization

Compute the Cholesky factor of H⁻¹:

```
H_inv_chol = cholesky(H⁻¹)      upper triangular
```

### 3.4 Column-wise quantization with error compensation (block_size B=128)

```
for block [i, i+B):
    for column j in [i, i+B):
        w_q[j] = quantize(W[:, j])                           # per-channel symmetric
        err    = (W[:, j] - w_q[j]) / H_inv_chol[j, j]       # Hessian-weighted error
        W[:, j+1:i+B] -= outer(err, H_inv_chol[j, j+1:i+B])  # compensate within block
    W[:, i+B:] -= E @ H_inv_chol[i:i+B, i+B:]                # compensate remaining blocks
```

Per-channel symmetric quantization: `scale = max_abs / (2^(bits-1) - 1)`.

**Block size rationale**: B=128 balances vectorization with memory. For d_in=1536 (attention projections): 12 blocks. For d_in=6144 (fc2): 48 blocks.

---

## 4. Poly-Aware Hessian Collection

The Hessian should reflect the actual quantized-activation distribution seen at inference time. Before accumulating H, fake-quantize the input activations using the polynomial clipping schedule:

```python
alpha = np.polyval(poly_coeffs, sigma)          # from polynomial_clipping_schedule.json
scale = alpha / 127                              # A8 symmetric
x_q   = fake_quant_int(x, scale, bits=8)
x_2d  = x_q.reshape(-1, x.shape[-1])
H    += x_2d.T @ x_2d
```

The sigma for each calibration sample is extracted from `sample_sigmas` in the cached calibration data. The poly coefficients come from the schedule file, keyed by layer path.

**Memory**: H is (d_in × d_in) float32. For d_in=1536: 9.4 MB. For d_in=6144 (fc2): 150 MB. Peak across all linears in one block: ~330 MB — well within unified memory.

---

## 5. Scaled Quantile α Search

Per-layer search to fine-tune the activation clipping range post-GPTQ, using early-stopping bidirectional search from α_scale=1.0.

**Search range**: `alpha_scale ∈ [0.2, 0.3, ..., 5.0]` (step size 0.1, 49 candidates).

**Algorithm**:
1. Define `effective_alpha(sigma) = alpha_scale · poly_alpha(sigma)`
2. Compute baseline MSE at `alpha_scale = 1.0` (the unscaled polynomial)
3. **Search downward** (0.9, 0.8, ..., 0.2):
   - For each candidate, compute MSE vs FP16 reference output
   - If MSE does not improve over the **current best in this direction** for 2 consecutive steps, stop
   - Track best downward `alpha_scale` and MSE
4. **Search upward** (1.1, 1.2, ..., 5.0):
   - Compare against the **baseline MSE at α_scale=1.0** (not the downward best)
   - If MSE does not improve over the current best in this direction for 2 consecutive steps, stop
   - Track best upward `alpha_scale` and MSE
5. Select the `alpha_scale` with lowest MSE across baseline, downward best, and upward best

**MSE evaluation** (per candidate):
   - Install GPTQ-quantized weights (dequantized to FP16)
   - Apply fake activation quantization with `scale = alpha_scale · poly_alpha(sigma) / 127`
   - Forward all calibration samples through the linear
   - Compute MSE vs FP16 reference output

**Early-stopping rationale**: The MSE curve is approximately U-shaped around the optimum — too small α clips activations, too large α wastes quantization bins. Each side of the minimum is approximately monotonic, so searching outward from 1.0 and stopping after 2 consecutive non-improvements finds the optimum with typically 5–10 evaluations per layer instead of all 49.

**Complexity**: O(n_layers × n_evaluated × n_samples) linear-only forwards — no backward pass. With early stopping, `n_evaluated` is typically ~10 per layer instead of 49.

**Storage**: `alpha_scale` values stored in `config.json` under `"alpha_scales"`, keyed by poly schedule layer name. At inference, wired to `_ActQuantLayer.poly_margin` (existing field, defaults to 1.0).

---

## 6. Proposed Folder Structure

```
src/gptq/
├── __init__.py              # Package init
├── gptq_quantize.py         # Core: Hessian factorization, column-wise quant loop
├── hessian_collector.py     # _HessianCollector proxy replacing nn.Linear during collection
├── alpha_search.py          # Per-layer scaled quantile alpha grid search
├── optimize.py              # CLI entry point: block iteration orchestrator
├── inference.py             # Load GPTQ weights + poly activation hooks for generation
└── utils.py                 # Shared: scale computation, poly key mapping, block enumeration
```

**Reusable logic** from existing codebase:
- `get_block_linears()` → `utils.py`
- `compute_per_channel_scale()` → `utils.py`
- `_path_to_poly_key()` → `utils.py`
- Modulation parameter helpers → `utils.py`
- Block I/O loading pattern → `hessian_collector.py`

**CLI flags**: `--bits-w {4,8}`, `--poly-schedule <path>`, `--alpha-search`, `--damp-percent`, `--block-size`, `--output-dir`

---

## 7. Output Format

```
<output_dir>/
    config.json
    weights/
        mm0.npz
        mm1.npz
        ...
```

**`config.json`**:
```json
{
    "method": "gptq",
    "bits_w": 4,
    "damp_percent": 0.01,
    "block_size": 128,
    "poly_schedule": "polynomial_clipping_schedule.json",
    "alpha_scales": {
        "mm_00_img::q_proj": 0.85,
        "mm_00_img::k_proj": 0.92,
        ...
    }
}
```

**Per-block `.npz`**: Keys are `{safe_path}__weight_int` (int8/int4) and `{safe_path}__scale` (float32).

---

## 8. MLX Considerations

| Concern | Approach |
|---------|----------|
| **Cholesky** | `mx.linalg.cholesky()` available. Compute `H_inv = mx.linalg.inv(H)` then Cholesky of the inverse. Fallback to NumPy if numerical issues arise. |
| **Lazy eval** | Call `mx.eval()` every ~10 samples during Hessian accumulation to keep the computation graph bounded. |
| **Column loop** | GPTQ's column-wise loop is non-differentiable — no computation graph needed. Can use NumPy for the core quantization loop, converting W and H_inv_chol to NumPy arrays. Major simplification vs AdaRound. |
| **Memory** | Hessian matrices (9.4–150 MB each) fit comfortably in Apple Silicon unified memory. Process one block at a time, discard Hessians after quantizing. |
| **adaLN offload** | Same pattern as before: group calibration points by prompt, batch all timesteps per group into one `cache_modulation_params` call, reload weights between groups. |

---

## 9. Verification

| Check | Method |
|-------|--------|
| **Block MSE** | Per-block reconstruction error: `MSE(block_out_fp16, block_out_quantized)` across calibration set |
| **FID / IS** | 500 COCO images, 28 denoising steps, compare vs FP16 baseline |
| **Quantization speed** | Wall-clock time for full pipeline (target: ~1.5 hrs) |
| **Weight loading** | Round-trip: quantize → save → load → generate, verify identical outputs |

---

## 10. Implementation Phases

1. **Core GPTQ** — `gptq_quantize.py` + `optimize.py`: Hessian accumulation (no poly), Cholesky, column-wise quantization, block iteration, weight serialization
2. **Poly-aware Hessian** — `hessian_collector.py` + `--poly-schedule` flag: fake-quantize activations before H accumulation
3. **Alpha search** — `alpha_search.py` + `--alpha-search` flag: per-layer alpha_scale grid search
4. **Benchmark** — FID/IS comparison vs FP16 baseline, speed measurement vs AdaRound
