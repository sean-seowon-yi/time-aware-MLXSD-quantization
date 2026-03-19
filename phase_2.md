# Phase 2: Q-Diffusion for SD3 Medium MMDiT

## Context

Apply Q-Diffusion (Li et al., 2023) post-training quantization to SD3 Medium's MMDiT backbone using MLX. Q-Diffusion's key techniques — **timestep-aware calibration** (already collected in Phase 1) and **AdaRound** (learned weight rounding via block-wise reconstruction) — are adapted for MMDiT's dual-stream joint-attention architecture with adaLN modulation.

**Goal**: Quantize the MMDiT noise estimation network to **W4A8** and **W8A8** configurations while maintaining generation quality comparable to FP16, following Algorithm 2 from the paper. The pipeline is parameterized by `weight_bits` (4 or 8) and `activation_bits` (8), enabling direct comparison of both precision targets.

**Input**: Calibration dataset `eda_output/coco_cali_data.npz` (100 prompts × 30 Euler steps = 3,000 (x_t, c, uc, t) tuples at 512×512).

---

## Layers to Quantize

### Quantized (weight + activation)

All `nn.Linear` layers in the MMDiT backbone that perform matrix multiplications:

| Layer | Shape (out, in) | Count | Notes |
|---|---|---|---|
| `q_proj` | (1536, 1536) | 48 (24 blocks × img+txt) | Has bias |
| `k_proj` | (1536, 1536) | 48 | No bias |
| `v_proj` | (1536, 1536) | 48 | Has bias |
| `o_proj` | (1536, 1536) | 47 | Block 23 txt is Identity (skipped) |
| `fc1` | (6144, 1536) | 47 | Block 23 txt has no FFN |
| `fc2` | (1536, 6144) | 47 | Block 23 txt has no FFN |
| **FinalLayer** `linear` | (64, 1536) | 1 | Patch unprojection |
| **Total** | | **~286 linear layers** | |

### NOT Quantized (kept FP16)

- **adaLN modulation Linears** (48 img + 24 txt + 1 FinalLayer = 49 total) — modulation errors amplify across entire feature maps; kept FP16 for stability
- **LayerNorm** (norm1, norm2, norm_final)
- **Activation functions** (SiLU in adaLN Sequential, GELU in FFN)
- **SDPA** (attention matmul q·k^T and attn·v) — kept in FP16
- **Embedders** (x_embedder, context_embedder, t_embedder, y_embedder) — small relative cost
- **QKNorm** (RMSNorm per head, if used)

---

## Quantization Scheme

### Weight Quantization: Configurable (4-bit or 8-bit) with AdaRound

- **Bit widths**: `weight_bits ∈ {4, 8}` (configurable via `QDiffusionConfig`)
  - **W4A8**: 4-bit weights, more aggressive — AdaRound is critical for quality
  - **W8A8**: 8-bit weights, less aggressive — AdaRound still improves over naive rounding but gap is smaller
- **Granularity**: Per-channel (one scale per output channel of each Linear)
- **Type**: Symmetric uniform: `w_hat = s · clamp(round(w/s), -2^(b-1), 2^(b-1)-1)`
- **Rounding**: AdaRound — learn binary rounding decisions (up/down) per weight element
  - Initialize V from fractional part of `w/s`
  - Optimize via rectified sigmoid relaxation: `h(V) = clamp(σ(V)·(ζ+1) - ζ/2, 0, 1)`
  - Parameters: ζ = 1.1, γ = -0.1 (standard AdaRound stretching)
  - Final: hard threshold at 0.5

### Activation Quantization: 8-bit with Statistics-Based Calibration

- **Granularity**: Per-tensor (one clipping range α per activation tensor)
- **Symmetry**: Per-layer, based on activation distribution:
  - **Symmetric** for q_proj, k_proj, v_proj, o_proj, fc1, FinalLayer inputs — activations span positive and negative
    - `x_hat = s · clamp(round(x/s), -2^(b-1), 2^(b-1)-1)` where `s = α / (2^(b-1) - 1)`
  - **Asymmetric** for fc2 inputs — post-GELU activations are non-negative, so symmetric would waste half the representable range on values that don't exist
    - `x_hat = s · clamp(round(x/s - z), 0, 2^b - 1) + z·s` where `s = (α_max - α_min) / (2^b - 1)`
- **Clipping range α**: Set from calibration data, **not learned via gradient descent**
  - Two calibration methods (configurable via `act_calibration_method`):
    - `"percentile"`: `α = percentile(|x|, act_percentile)` — fast, single pass
    - `"mse_search"` (default): Test candidate α values from `act_search_candidates` (e.g., percentiles [99.0, 99.5, 99.9, 99.95, 99.99, 100]), select the α minimizing block reconstruction MSE. This is the BRECQ-inherited approach — percentile is the simplification, not the other way around. Candidates are clustered near the top because diffusion activations are heavy-tailed (large mass near zero, long tails). Cost: one forward pass per candidate, no backward, <2s per block.
  - Symmetric layers: range `[-α, +α]`
  - Asymmetric layers (fc2): `α_min, α_max` from corresponding percentiles
  - Computed per-layer during Step 2 (activation calibration pass)
- **No AdaRound** for activations (they change per input; only weights use AdaRound)
- **No gradient-based tuning** — α is selected from statistics or reconstruction-based search, not optimized via backprop

### Attention Matmul Handling

Per the paper (Section A.1), for W4A8 the attention score matrices (q·k^T, attn·v) use **INT16 mixed precision** to avoid quality degradation. In our MLX implementation, we keep SDPA in FP16 entirely (MLX's `mx.fast.scaled_dot_product_attention` runs at native precision). The q, k, v projection *outputs* are quantized to 8-bit before being fed to SDPA.

---

## Full Algorithm (Following Algorithm 2)

### Overview

Algorithm 2 uses a single model that is progressively quantized in-place:
- **W_θ** starts as the pretrained full-precision model
- FP block outputs are cached in a **single forward pass** before any quantization begins
- Blocks are then replaced with `QuantizedLinear` one at a time (in-place), refined via AdaRound, and frozen
- After all blocks are refined, the FP model is gone — only the quantized model remains

This avoids keeping two full model copies in memory (~4 GB saved vs. two-model approach).

### Pre-compute: Cache FP Block Outputs

Before any quantization, run calibration data through the FP model once:
1. Forward D (subsampled to `n_samples`) through all blocks of W_θ
2. Hook every block to cache its FP output → `fp_targets[block_idx]`
3. These targets are used throughout Step 1 and Step 2 as reconstruction references

### Step 0: Initialize Ŵ_θ (In-Place Naive Quantization)

Replace the FP model in-place with naive weight quantization:
1. Replace every quantizable `nn.Linear` in W_θ with `QuantizedLinear`
2. Compute per-channel weight scales using min/max (or percentile) of FP weights
3. Apply **round-to-nearest** (naive rounding) — this is the V=0.5 initialization
4. Activation quantizers are **created but disabled** — they will be calibrated in Step 2
5. W_θ is now Ŵ_θ — the FP model no longer exists separately (FP outputs preserved in cache)

### Step 1: Block-wise Weight Quantizer Refinement (AdaRound)

### Block Boundaries

Each `MultiModalTransformerBlock` = one reconstruction block. Total: **25 blocks**.
- Blocks 0–23: MultiModalTransformerBlock (img + txt streams jointly)
- Block 24: FinalLayer (img stream only)

### Algorithm (per block, sequential from block 0 to 24)

```
For block_idx = 0 to 24:
  1. COLLECT block inputs from Ŵ_θ:
     - Forward calibration data D through Ŵ_θ blocks 0..block_idx-1
       using their FROZEN AdaRound-refined weights (hard-rounded from prior
       iterations). This ensures block_idx's inputs reflect the actual
       quantization error accumulated from all previously optimized blocks.
     - Collect block_idx's INPUTS from this forward pass.
     - FP reconstruction targets: read from fp_targets[block_idx] cache
       (pre-computed in the single FP forward pass before Step 0).
  2. INIT AdaRound: For block_idx's QuantizedLinear layers (already
     in Ŵ_θ from Step 0), initialize learnable V parameters from the
     fractional part of w/s (replacing naive round-to-nearest).

  3. OPTIMIZE: For 3,000 iterations:
     - Sample mini-batch of batch_size from collected (input, target) pairs
     - Forward block with soft-rounded weights (using V params),
       activations in FP (act quantizers disabled during Step 1)
       (adaLN stays FP16 — cached modulation params are exact)
     - Loss = ReconLoss(quant_output, fp_target) + λ·AdaRound_reg(V, β_t)
       (ReconLoss = Fisher-weighted MSE if use_fisher=True, else plain MSE)
     - β_t anneals from 2 → 20 over warmup (first 20% of iterations)
       (small β early = soft rounding exploration; large β late = push V toward 0/1)
     - Adam update on V params only

  4. LOG: Every iteration, append (iter, recon_loss, reg_loss, total_loss, β_t)
     to a per-block training history list. Print summary every 100 iterations.

  5. FREEZE: Hard-round V → {0, 1}, free cached I/O, log metrics.
```

### Step 2: Activation Quantizer Calibration

After all 25 blocks have refined weight rounding via AdaRound:
```
For block_idx = 0 to 24:
  1. Run calibration data D through the refined Ŵ_θ (AdaRound weights frozen).
     Forward through blocks 0..block_idx, collecting each QuantizedLinear's
     input activations.

  2. For each QuantizedLinear in block_idx, set α using the configured method:

     If act_calibration_method == "percentile":
       - α = percentile(|x|, act_percentile) across collected samples
       - Derive scale: s = α / (2^(b-1) - 1)

     If act_calibration_method == "mse_search":
       - Compute candidate α values from act_search_candidates percentiles of |x|
         Default: [99.0, 99.5, 99.9, 99.95, 99.99, 100.0]
         (clustered near top — diffusion activations are heavy-tailed)
       - For each candidate α_c:
           Set ActivationQuantizer.alpha = α_c
           Forward block (one pass, no backward, small batch)
           Compute MSE(block_out_quant, fp_targets[block_idx])
       - Select α = argmin_α_c(MSE)
       - Cost: ~6 forward passes per layer, <2s per block total

     For asymmetric layers (fc2): apply same method using (α_min, α_max) pairs.
     Store α and derived scale in the ActivationQuantizer.

  3. Enable activation quantizers for this block.
  4. Log per-layer α values and the resulting block reconstruction MSE.
```

### Reconstruction Loss

For `MultiModalTransformerBlock`:
```
L = MSE(img_out_quant, img_out_fp) + MSE(txt_out_quant, txt_out_fp)
```
Block 23: txt_out is None (discarded), so only img term.

For `FinalLayer`:
```
L = MSE(out_quant, out_fp)
```

**Fisher-information weighting** (`use_fisher=False` by default):
When enabled, the AdaRound reconstruction MSE is weighted by a diagonal Hessian approximation (Fisher information), as in BRECQ paper [19]. This gives more weight to output dimensions the model is more sensitive to.
```
# use_fisher=False (default):
L = MSE(quant_output, fp_target)

# use_fisher=True:
F = diag(E[∂L/∂output²])  # estimated from calibration data
L = (quant_output - fp_target)² · F   # element-wise Fisher weighting
```
Fisher weights are computed once per block during the pre-compute step (before quantization) by running a small forward+backward pass to estimate output sensitivity, and cached alongside fp_targets.

### adaLN Modulation During AdaRound

Since adaLN layers are kept in FP16, the cached `_modulation_params[timestep]` are exact during AdaRound weight optimization. No special replay logic is needed — the block forward reads exact FP16 modulation params from the cache, and the reconstruction loss only captures quantization error from the q/k/v/o/fc1/fc2 projections within each block.

---

## Code Structure

```
src/q_diffusion/
├── __init__.py
├── config.py                # QDiffusionConfig dataclass
├── quantizer.py             # Quantization primitives (scale, α, AdaRound math)
├── quant_linear.py          # QuantizedLinear (wraps nn.Linear with AdaRound + act quant)
├── calibration_feeder.py    # Load cali data, collect block I/O via hooks
├── block_reconstruct.py     # AdaRound weight optimization + activation calibration per block
├── training_tracker.py      # Per-iteration loss logging for AdaRound optimization
├── pipeline.py              # End-to-end orchestration (run_q_diffusion)
├── quant_model_io.py        # Save/load quantized model
├── evaluate.py              # Generate images + compute metrics
└── run_quantize.py          # CLI: python -m src.q_diffusion.run_quantize
```

### File Details

#### `config.py`
```python
@dataclass
class QDiffusionConfig:
    weight_bits: int = 4             # 4 for W4A8, 8 for W8A8
    activation_bits: int = 8         # Always 8 for both configs
    weight_symmetric: bool = True
    weight_per_channel: bool = True
    act_symmetric: bool = True       # Default; fc2 inputs override to asymmetric (post-GELU)
    adaround_iters: int = 3000       # Per block (reduced from 10K for faster computation)
    adaround_lr: float = 1e-3
    adaround_beta_start: float = 2.0  # Initial β (soft rounding)
    adaround_beta_end: float = 20.0  # Final β (hard rounding, pushes V → 0/1)
    adaround_warmup: float = 0.2     # Fraction of iters for reg warmup
    adaround_reg_weight: float = 0.01
    batch_size: int = 16             # Mini-batch for AdaRound optimization
    n_samples: int = 512             # Subset of cali data per block
    act_calibration_method: str = "mse_search"  # "mse_search" (recommended) or "percentile"
    act_percentile: float = 99.99    # Used when act_calibration_method == "percentile"
    act_search_candidates: List[float] = (99.0, 99.5, 99.9, 99.95, 99.99, 100.0)  # Percentiles to test (mse_search)
    use_fisher: bool = False         # Fisher-information weighting on reconstruction MSE (default: off)
    calibration_file: str = "eda_output/coco_cali_data.npz"
    output_dir: str = "q_diffusion_output"
    quantize_sdpa: bool = False
    skip_final_layer: bool = False    # Optionally skip FinalLayer quantization
```

**Usage for both configurations:**
```bash
# W4A8 (more aggressive, paper's primary target)
python -m src.q_diffusion.run_quantize --weight-bits 4 --output-dir q_diffusion_output/w4a8

# W8A8 (less aggressive, useful baseline)
python -m src.q_diffusion.run_quantize --weight-bits 8 --output-dir q_diffusion_output/w8a8
```

#### `quantizer.py` — Pure MLX Quantization Math
- `compute_scale(tensor, bits, symmetric, per_channel)` → `scale` (symmetric: `scale = α / (2^(b-1) - 1)`)
- `uniform_quantize(x, scale, bits)` → fake-quantized `x` (STE)
- `adaround_quantize(weight, v, scale, bits, beta)` → soft-rounded fake-quantized weight
- `adaround_reg(v, beta)` → regularization loss pushing V toward 0/1
- `init_v_from_weights(weight, scale)` → initial V parameters
- `class ActivationQuantizer` — clipping range α (via percentile or MSE grid search); derives scale from α, fake-quantizes; can be enabled/disabled

#### `quant_linear.py` — QuantizedLinear Module
```python
class QuantizedLinear(nn.Module):
    # Frozen: weight, bias, weight_scale
    # Trainable: v_param (same shape as weight)
    # Optional: act_quantizer (ActivationQuantizer, statistics-based α)

    @staticmethod
    def from_linear(linear, weight_bits, act_bits=None) -> QuantizedLinear

    def __call__(self, x):
        # Quantize INPUT activation (if act_quantizer enabled):
        #   x_q = act_quantizer(x)          # fake-quantize input
        # Quantize weight via AdaRound:
        #   w_q = adaround_quantize(weight, v, scale, bits, beta)
        # Matmul with both quantized operands:
        #   return x_q @ w_q.T + bias
        # This matches real inference where both matmul operands are quantized.

    def freeze_rounding(self)  # soft → hard rounding
    def trainable_parameters(self) -> dict  # only V params (act quantizers are statistics-based, not trained)
```

#### `calibration_feeder.py` — Block I/O Collection
- `load_calibration_data(path)` → `(xs, ts, prompt_indices, cs, cs_pooled)`
- `group_by_prompt(prompt_indices)` → `Dict[int, List[int]]`
- `class BlockIOCollector`:
  - `collect_all_fp_targets(model, cali_data, n_samples)` → `Dict[int, Tensor]` — single forward pass through FP model (before quantization), caches every block's output
  - `collect_block_inputs(model, cali_data, block_idx, n_samples)` → `Tensor` — forward through blocks 0..block_idx-1 of the (progressively quantized) model, collect block_idx inputs
  - Handles prompt-grouped processing with `cache_modulation_params` + adaLN offload/reload
  - Subsamples to `n_samples` (default 512) with uniform timestep coverage

#### `block_reconstruct.py` — AdaRound Weight Optimization + Activation Calibration
- `replace_linears_in_block(block, config)` → `Dict[str, QuantizedLinear]` — replaces all quantizable nn.Linear in one block
- `calibrate_act_quantizers(block, block_inputs, fp_targets, config)` — set α via percentile or MSE grid search
- `compute_recon_loss(block, inputs, fp_targets, block_idx)` → scalar loss
- `optimize_block_weights(block, block_idx, inputs, fp_targets, mod_inputs, config)` → `BlockTrainingLog` — 3K-iteration AdaRound weight optimization

#### `training_tracker.py` — Loss Tracking for AdaRound
```python
@dataclass
class BlockTrainingLog:
    block_idx: int
    history: List[dict]     # per-iteration: {iter, recon_loss, reg_loss, total_loss, beta}
    mse_before: float       # naive rounding MSE (before AdaRound)
    mse_after: float        # final MSE (after AdaRound freeze)
    improvement_ratio: float  # mse_before / mse_after

class TrainingTracker:
    """Accumulates BlockTrainingLogs across all 25 blocks."""
    logs: List[BlockTrainingLog]

    def add_block(self, log: BlockTrainingLog)
    def print_block_summary(self, block_idx: int)  # final MSE, improvement ratio
    def print_overall_summary(self)                 # table of all blocks
    def save_json(self, path: str)                  # serialize all logs to JSON
    def plot_loss_curves(self, output_dir: str)     # per-block loss vs iteration PNGs
```

#### `pipeline.py` — Orchestration
```python
def run_q_diffusion(config: QDiffusionConfig):
    1. Load FP model W_θ
    2. Load calibration data D

    # Pre-compute: Cache ALL FP block outputs in a single forward pass
    3. Run calibration data D (subsampled to n_samples) through W_θ once.
       Hook every block to cache its FP output → fp_targets[block_idx].

    # Step 0: In-place naive weight quantization (replaces FP model)
    4. Replace ALL quantizable nn.Linear in W_θ with QuantizedLinear (round-to-nearest).
       Activation quantizers created but disabled (calibrated in Step 2).
       W_θ is now Ŵ_θ — the FP model no longer exists separately.
       (FP block outputs are preserved in fp_targets cache from step 3.)

    # Step 1: Block-wise weight refinement (AdaRound)
    5. For block_idx in range(25):
       a. Collect block inputs by forwarding through Ŵ_θ blocks 0..block_idx-1
          (FP targets already cached from step 3)
       b. Initialize AdaRound V params for this block
       c. Optimize V (3K iters AdaRound + reg), log per-iteration losses
       d. Freeze rounding, log block summary (mse_before vs mse_after), cleanup

    # Step 2: Activation quantizer calibration
    6. For block_idx in range(25):
       a. Run calibration data through refined Ŵ_θ
       b. Calibrate α per layer (percentile or mse_search, per config)
       c. Enable activation quantizers for this block

    7. Save quantized model
    8. Save training logs (tracker.save_json + tracker.plot_loss_curves)
    9. Print overall summary table
```

#### `quant_model_io.py` — Serialization
- `save_quantized_model(mmdit, output_dir, config)` — save quantized weights (weight + hard_round_mask + scale per layer), config JSON, activation α per layer
- `load_quantized_model(output_dir)` → `(MMDiT, QDiffusionConfig)` — reconstruct quantized model

#### `evaluate.py` — Evaluation
- `generate_images(pipeline, prompts, output_dir, seed)` — generate with quantized model
- `compute_per_timestep_mse(pipeline_quant, pipeline_fp, cali_data)` — per-step reconstruction error

#### `run_quantize.py` — CLI Entry Point
```
python -m src.q_diffusion.run_quantize [--weight-bits 4] [--activation-bits 8] [--adaround-iters 3000] ...
```

---

## Memory Budget (M5 Pro 32GB Unified)

| Component | Estimated Size |
|---|---|
| Single MMDiT model (progressively quantized in-place) | ~4 GB |
| Cached FP block outputs (25 blocks × 512 samples) | ~3 GB |
| Calibration data: xs (3000 × 64×64×16 float32) | ~0.75 GB |
| Calibration data: cs, cs_pooled, ts | ~0.3 GB |
| Block I/O cache (512 samples × block inputs) | ~1.5 GB |
| AdaRound V params (one block) | ~50 MB |
| Optimizer states (Adam, 2× V params) | ~100 MB |
| Forward/backward intermediates | ~2-4 GB |
| **Peak estimate** | **~12-14 GB** |

Comfortable within 32GB. Single-model in-place approach saves ~4 GB vs. two-model strategy.

**Mitigations**:
- Process one block at a time; free I/O cache between blocks
- Subsample to 512 of 3000 calibration points per block (stratified by timestep)
- Load calibration xs lazily (memory-map the npz)
- Mini-batch of 16 for gradient computation
- `mx.eval()` after each optimizer step to prevent graph accumulation
- `del` + `gc.collect()` + `mx.metal.clear_cache()` after each block
- If memory pressure: reduce `n_samples` to 256 or `batch_size` to 8

---

## Reusable Existing Code

| Existing | Reuse in Phase 2 |
|---|---|
| `src/eda/eda_tracer.py` (lines 244-389) | Monkey-patching pattern for `BlockInputCollector` hooks |
| `src/eda/profile_activations.py` | Prompt-grouped forward loop with adaLN cache/offload pattern |
| `src/eda/weight_profiler.py:_iter_linear_layers` | Enumerate all quantizable Linear layers in MMDiT |
| `src/calibration_sample_generation/calibration_config.py` | Import constants (model version, paths) |
| `DiffusionKit/.../mmdit.py` TransformerBlock/MultiModalTransformerBlock | Hook points, understand block I/O signatures |

---

## Implementation Order

1. **`config.py`** + **`quantizer.py`** — foundation (pure math, testable in isolation)
2. **`quant_linear.py`** — QuantizedLinear module (test: wrap a single Linear, verify forward matches fake-quantized manual computation)
3. **`training_tracker.py`** — loss tracking dataclasses + summary/plot utilities (no model deps, testable standalone)
4. **`calibration_feeder.py`** — load cali data + block I/O collection (test: collect I/O for block 0, verify FP output shapes)
5. **`block_reconstruct.py`** — AdaRound weight optimizer + activation calibration (test: optimize block 0, verify MSE decreases, verify tracker populated)
6. **`pipeline.py`** — end-to-end orchestration
7. **`quant_model_io.py`** — save/load
8. **`evaluate.py`** + **`run_quantize.py`** — CLI + evaluation

---

## Verification

### Per-Block (during quantization, via TrainingTracker)
- Log per-iteration losses (recon_loss, reg_loss, total_loss, β) — printed every 100 iters
- Log MSE before optimization (naive rounding) vs after (AdaRound) for each block
- Verify MSE consistently decreases (expect 2-5× reduction)
- Save per-block loss curve PNGs (`tracker.plot_loss_curves`) and full JSON log (`tracker.save_json`)

### End-to-End (after full quantization — run for both W4A8 and W8A8)
1. **Reconstruction MSE**: Run 50 calibration prompts through quantized model, compute per-timestep MSE of denoiser output vs FP
2. **Visual comparison**: Generate 10 images with same seeds for FP vs W4A8 vs W8A8, side-by-side
3. **Latent trajectory**: Plot latent norm per timestep for FP vs quantized (check error accumulation)
4. **Optional FID**: Generate 1000 images from held-out COCO prompts, compute FID against FP reference set
5. **Comparison table**: Tabulate FP vs W8A8 vs W4A8 on MSE, visual quality, model size reduction

---

## Design Decisions (Resolved)

1. **DiffusionKit**: NO modifications to DiffusionKit source. All code in `src/q_diffusion/` using monkey-patching and wrappers.
2. **Calibration data**: Lazy loading via memory-mapped npz. Load samples on-demand during block I/O collection.
3. **Single-model in-place strategy**: Cache all FP block outputs in one forward pass, then progressively replace blocks with QuantizedLinear in-place. Saves ~4 GB vs. keeping two full models. FP targets are preserved in the cache; the FP model is consumed by quantization.
4. **Target precision**: Both **W4A8** and **W8A8** supported via configurable `weight_bits` (4 or 8). Same pipeline, different bit width. Run each config separately to compare.
5. **Shortcut-splitting quantization (omitted)**: Q-Diffusion Section 3.3.2 introduces split quantization for UNet shortcut layers where deep and shallow feature channels are concatenated, causing activation ranges to differ by up to 200×. MMDiT has **no UNet-style skip connections** — it uses a pure transformer architecture with residual additions (not concatenations), so this technique is inapplicable.
