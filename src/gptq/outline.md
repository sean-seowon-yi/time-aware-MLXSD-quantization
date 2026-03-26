# GPTQ Pipeline Outline

End-to-end quantization pipeline for SD3 MMDiT (24 multimodal transformer blocks, 12 linears per block = 285 total layers). Targets W4A8 or W8A8 quantization with time-aware (polynomial) or static activation clipping.

All estimates assume SD3-Medium dimensions (d=1536, MLP hidden=6144) at latent_size=64, 30 prompts for Phase A, 5 prompts for Phase B, 30 denoising steps.

---

## 0. Setup & Loading

**File:** `optimize.py:86-158`

| Step | What happens |
|------|--------------|
| 0a | Parse CLI args (bits, damping, block size, prompt counts, etc.) |
| 0b | Load prompts from text file, split into Phase A set (all) and Phase B set (first `alpha_prompts`) |
| 0c | Load polynomial clipping schedule JSON (per-layer polynomial coefficients for sigma-dependent alpha) |
| 0d | Optionally compute static alphas: for each layer, evaluate poly(sigma) over 200 sigma points and take the max (`inference.py:73-85`) |
| 0e | Load DiffusionPipeline (SD3-Medium via diffusionkit) with FP16 weights, initialize CFGDenoiser |

**Memory:** ~4-6 GB for the full SD3-Medium model in FP16, plus text encoders (CLIP + T5).
**Speed:** ~15-30s model loading (disk I/O bound).

---

## 1. Phase A — Global Hessian Collection

**File:** `hessian_collector.py:221-286`, called from `optimize.py:166-200`

### 1a. Install Hessian collectors on all blocks

**File:** `hessian_collector.py:151-165`

- For each of the 24 blocks, replace all 12 linear layers with `_HessianCollector` proxy objects.
- Each collector wraps the original `nn.Linear` and intercepts forward calls.
- `cache_io=False` — no input/output caching, only Hessian accumulation.
- If static mode: collectors use a fixed alpha per layer. If poly mode: collectors evaluate `poly(sigma)` at the current timestep.

**Memory:** Negligible overhead for proxy installation.

### 1b. Run all prompts through the denoising loop

**File:** `hessian_collector.py:250-281`

For each prompt (30 by default):
1. Encode text via CLIP + T5 → conditioning tensors.
2. Generate sigma schedule (30 steps) and cache modulation parameters (adaLN).
3. Sample initial noise latent `(1, 64, 64, 16)`.
4. For each denoising step (30 steps):
   - Set current sigma on all 285 collectors.
   - Run full forward pass through CFGDenoiser (2 passes: conditional + unconditional).
   - Each collector's `__call__`:
     - Fake-quantizes input `x` using the poly/static alpha at current sigma.
     - Computes `H += x_2d^T @ x_2d` in float32 (Hessian accumulation).
   - **Crucially**: call `mx.eval()` on all 285 Hessian matrices after every step to prevent the MLX lazy computation graph from exploding.
   - Euler step to advance the latent.
5. Reset modulation cache (reload adaLN weights) after each prompt.

**Memory:**
- 285 Hessian matrices, each `(d_in, d_in)` float32:
  - Q/K/V/O projections: `(1536, 1536)` = 9 MB each → 4 × 2 streams × 24 blocks = 192 matrices × 9 MB = **~1.7 GB**
  - MLP fc1: `(1536, 1536)` = 9 MB, fc2: `(6144, 6144)` = 144 MB each → 2 streams × 24 blocks = 48 fc2 matrices × 144 MB = **~6.9 GB** (fc2 dominates)
  - **Total Hessians: ~8-9 GB**
- Plus model weights (~5 GB) and latent/conditioning tensors (~0.5 GB).
- **Peak: ~14-15 GB**

**Speed:** 30 prompts × 30 steps × 2 CFG passes = 1,800 full MMDiT forward passes.
- Each forward pass: ~0.3-0.5s on M-series GPU.
- **Total: ~10-15 minutes.**

### 1c. Extract Hessians and save checkpoint

**File:** `optimize.py:191-200`

- Call `collector.get_hessian()` → returns `2 * H` as NumPy float32.
- Delete all collectors to free MLX memory.
- Save all Hessians to a single `.npz` file for reuse.

**Memory:** Hessians transition from MLX to NumPy. Briefly double-stored during conversion.
**Speed:** ~10-30s for `.npz` serialization (8-9 GB to disk).

### 1d. (Cached path) Load Hessians from checkpoint

**File:** `optimize.py:170-176`

- If `hessians.npz` exists, skip Phase A entirely and load from disk.

**Speed:** ~10-20s (disk I/O).

---

## 2. Phase B — GPTQ Quantization + Alpha Search

### 2a. GPTQ column-wise quantization (all layers)

**File:** `optimize.py:208-229`, core algorithm in `gptq_quantize.py:13-111`

For each of the 24 blocks, for each of the 12 linears:

1. Extract weight matrix `W` as `(d_out, d_in)` float32.
2. Retrieve the corresponding Hessian `H` from Phase A.
3. Run GPTQ quantization:
   - Compute per-channel scales: `scale[i] = max(|W[i,:]|) / qmax`.
   - Damp the Hessian: `H += damp * I` (prevents singularity).
   - Compute `H_inv` and its Cholesky decomposition (upper triangular).
   - Column-wise quantization in blocks of 128:
     - For each column: quantize, compute error, compensate subsequent columns using `H_inv_chol`.
     - Intra-block compensation: `W[:, j+1:end] -= outer(err, H_inv_chol[j, j+1:end])`.
     - Inter-block compensation: `W[:, end:] -= E @ H_inv_chol[block, end:]`.
   - Compute weight MSE: `||W_orig - dequant(W_q)||^2`.
4. Store `(W_q_int, scales, weight_mse)` in `all_weight_results`.
5. Save each block's weights to `weights/mm{idx}.npz` immediately.

**Memory:**
- All Hessians still in RAM: **~8-9 GB**.
- Per-layer working set: `W` copy + `H_inv` + `H_inv_chol` = 3 × `(d_in, d_in)` float32 per layer, processed sequentially so only one layer at a time.
  - Attention layers: 3 × 9 MB = ~27 MB working set.
  - MLP fc2: 3 × 144 MB = ~432 MB working set.
- `all_weight_results` accumulates and is never freed:
  - 285 layers × ~2.3 MB (int8 weights + scales) = **~640 MB**.
- **Peak: ~10-11 GB** (Hessians + accumulating results + per-layer working set).

**Speed:**
- Cholesky + inverse: O(d_in^3). For d_in=1536: ~0.1s. For d_in=6144 (fc2): ~5-10s.
- Column-wise quantization: O(d_out × d_in × block_size). Typically <1s per layer.
- **Total: ~3-5 minutes** (dominated by the 48 fc2 layers with d_in=6144).

### 2b. Global alpha search (all blocks, one pass)

**File:** `optimize.py:233-241`, implementation in `hessian_collector.py:366-566`

1. **Install `_AlphaAccumulator` proxies** on all 285 linears (`hessian_collector.py:445-485`):
   - Each accumulator stores:
     - `W_q_dequant`: dequantized weight `(d_out, d_in)` float32 = `W_q_int * scales`.
     - `bias`: optional bias vector.
     - 22 MSE accumulators (one per alpha_scale candidate from 0.01 to 100.0).

2. **Run alpha search prompts** through denoising loop (5 prompts × 30 steps = 150 forward passes × 2 CFG):
   - Each accumulator's `__call__`:
     - Subsample input to 256 rows to cap memory.
     - For each of 22 alpha candidates:
       - Fake-quantize input `x` at scaled alpha.
       - Compute `y_q = x_fq @ W_q_dequant.T`.
       - Accumulate SE: `total_se[ci] += sum((y_q - y_ref)^2)`.
     - All computation is NumPy (CPU), not MLX.

3. **Extract best alpha** per layer: `argmin(total_se / total_elements)`.
4. Remove accumulators, restore original linears.

**Memory (bottleneck of entire pipeline):**
- `all_weight_results` still in scope: **~640 MB**.
- 285 `_AlphaAccumulator` instances each holding `W_q_dequant` float32:
  - Attention layers: 1536 × 1536 × 4B = 9 MB × 192 = **~1.7 GB**.
  - MLP fc1: 6144 × 1536 × 4B = 36 MB × 48 = **~1.7 GB**.
  - MLP fc2: 1536 × 6144 × 4B = 36 MB × 48 = **~1.7 GB**.
  - **Total W_q_dequant: ~5.1 GB** (larger than earlier estimate due to asymmetric MLP dims).
- 49-element float64 MSE arrays per layer: negligible.
- Per-call transient: subsampled `x_fq @ W_q_dequant.T` for 49 candidates — short-lived but up to 256 × 6144 × 4B × 49 ≈ 300 MB spikes per layer.
- **Peak: ~11-12 GB** (Hessians freed after 2a, but replaced by W_q_dequant copies).

**Speed:**
- 5 prompts × 30 steps × 2 CFG = 300 MMDiT forward passes.
- Each forward pass: ~0.5s (MLX) + 285 layers × 49 candidates × NumPy matmul overhead.
- The 49× NumPy matmuls per layer per step are the major slowdown — each is (256, d_in) × (d_in, d_out).
- **Total: ~15-30 minutes** (CPU-bound NumPy matmuls dominate).

### 2c. Assemble metrics and save config

**File:** `optimize.py:244-262`

- For each layer: combine `weight_mse` from GPTQ, `best_alpha_scale` and `activation_mse` from alpha search.
- Save `config.json` with all hyperparameters and per-layer metrics.

**Memory:** Negligible.
**Speed:** <1s.

---

## 3. Inference (separate entry point)

**File:** `inference.py:182-332`

Not part of the optimization pipeline but uses its outputs:

1. Load pipeline, load GPTQ weights from `.npz` files.
2. Replace model weights with dequantized GPTQ weights (`W_q_int * scales`).
3. Install `_ActQuantHook` proxies that fake-quantize activations at inference time using the learned `alpha_scale` values.
4. Run denoising loop with sigma-aware hooks → generate image.
5. Optionally generate side-by-side FP16 vs GPTQ comparison.

---

## Summary

| Step | Wall time | Peak RAM (beyond model) | Dominant cost |
|------|-----------|------------------------|---------------|
| 0. Load model | 15-30s | ~5 GB | Disk I/O |
| 1. Phase A: Hessians | 10-15 min | ~9 GB (Hessians) | 1800 fwd passes (GPU) |
| 2a. GPTQ quantize | 3-5 min | ~9 GB (Hessians) + 0.6 GB (results) | Cholesky on fc2 layers (CPU) |
| 2b. Alpha search | 15-30 min | ~5 GB (W_q_dequant) + 0.6 GB (results) | 49× NumPy matmuls per step (CPU) |
| 2c. Save config | <1s | negligible | — |
| **Total** | **~30-50 min** | **~14-15 GB peak** | |

### Key bottlenecks

1. **Memory:** Phase A Hessians (~9 GB for 285 layers, dominated by fc2's 6144×6144 matrices) and Phase B's `W_q_dequant` copies (~5 GB across 285 accumulators). The `all_weight_results` dict (~640 MB) is also never freed.

2. **Speed:** Phase B alpha search is the slowest step — 22 alpha candidates × 285 layers × 300 forward calls, all computed in NumPy on CPU. Phase A is GPU-bound but benefits from batching all blocks in one pass.
