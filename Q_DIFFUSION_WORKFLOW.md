# Q-Diffusion for SD3 Medium MMDiT: Workflow and Method

This document explains the end-to-end workflow of our Q-Diffusion implementation, how it differs from the original paper, and why each adaptation was made to accommodate the MMDiT architecture.

---

## 1. Background: What Is Q-Diffusion?

Q-Diffusion (Li et al., 2023) is a post-training quantization (PTQ) method designed for diffusion models. It addresses two key problems:

- **Weight rounding error**: Naive round-to-nearest quantization introduces error that accumulates across the iterative denoising process. Q-Diffusion uses **AdaRound** to learn optimal per-weight rounding decisions (round up or round down) that minimize the output distortion of each network block.

- **Activation quantization error**: Activations in diffusion models vary significantly across timesteps. Q-Diffusion uses **timestep-aware calibration** data and statistics-based clipping range selection to set activation quantization parameters that work well across the full denoising trajectory.

The original Q-Diffusion targets **U-Net** architectures under **DDPM** noise schedules. Our work adapts it to **SD3 Medium's MMDiT** (Multi-Modal Diffusion Transformer), which uses a fundamentally different backbone and sampling procedure.

---

## 2. Original Q-Diffusion vs. Our Approach

### 2.1 What We Kept from Q-Diffusion

**AdaRound weight optimization.** The core idea is unchanged: for each weight element, learn a binary rounding decision (up or down) by optimizing a continuous relaxation variable V through a rectified sigmoid. The optimization minimizes a block-level reconstruction loss comparing the quantized block's output against the full-precision reference output. After optimization, V is hard-thresholded at 0.5 to produce the final rounding decision.

**Block-wise reconstruction.** We follow the same sequential block-by-block strategy: optimize one block at a time, freeze it, then move to the next. When collecting inputs for block i, we forward calibration data through blocks 0 through i-1 using their already-optimized (frozen) quantized weights. This ensures each block's optimization accounts for the quantization error accumulated from all preceding blocks.

**Statistics-based activation calibration.** Activation quantization ranges are determined from calibration data statistics, not learned via gradient descent. We support two methods from the literature: simple percentile clipping and MSE-minimizing grid search over candidate clipping ranges (the BRECQ-inherited approach).

**Timestep-aware calibration data.** Our calibration dataset (collected in Phase 1) captures latent states across the full denoising trajectory, matching Q-Diffusion's insight that activation statistics shift substantially across timesteps and must be represented in calibration data.

### 2.2 What We Changed and Why

| Aspect | Original Q-Diffusion | Our Approach | Reason |
|---|---|---|---|
| Target architecture | U-Net (encoder-decoder) | MMDiT (dual-stream transformer) | SD3 uses a pure transformer, not a U-Net |
| Noise schedule | DDPM (1000 steps) | Flow-matching Euler (30 steps) | SD3's sampling procedure |
| Block boundaries | ResBlock / AttnBlock in U-Net | MultiModalTransformerBlock (24) + FinalLayer (1) = 25 blocks | Natural boundaries in the transformer architecture |
| Skip-connection splitting | Split quantization for UNet shortcuts | Omitted entirely | MMDiT has no skip connections (see Section 4.1) |
| Dual-stream handling | N/A (single-stream U-Net) | Joint reconstruction loss over img + txt outputs | Both streams must be preserved (see Section 4.2) |
| adaLN modulation | N/A | Kept in FP16; exact modulation during reconstruction | Modulation errors amplify across feature maps (see Section 4.3) |
| Attention quantization | INT16 mixed precision for q*k^T and attn*v | FP16 SDPA entirely; only projection outputs quantized | MLX runs SDPA at native FP16 precision (see Section 4.4) |
| fc2 activation symmetry | Symmetric for all layers | Asymmetric specifically for fc2 inputs | Post-GELU activations are non-negative (see Section 4.5) |
| Memory strategy | Two models (FP reference + quantized) | Single model, in-place quantization + disk-backed FP target cache | 32 GB constraint requires careful memory management (see Section 5) |
| FP target storage | In-memory cache | Streamed to disk, loaded one block at a time | All-blocks-in-memory exceeded 90 GB (see Section 5.2) |

---

## 3. The Full Pipeline

### Phase 1: Calibration Data Collection (Complete)

Before quantization begins, we need representative data that captures the model's behavior across the denoising trajectory. We run 100 COCO image prompts through SD3's 30-step Euler sampler at 512x512 resolution, recording at each step:

- The noisy latent x_t (the MMDiT's input)
- The timestep t
- The text conditioning embeddings (token-level and pooled)

This yields 3,000 calibration tuples (100 prompts x 30 steps). The timestep diversity is critical: activation distributions at early steps (high noise, t near 1.0) differ substantially from late steps (low noise, t near 0.0), and the quantizer must work well across all of them.

### Phase 2: Q-Diffusion Quantization

The quantization pipeline proceeds in six stages:

**Stage 1 -- Load the full-precision model.** We load the SD3 Medium pipeline including the MMDiT backbone, CLIP and T5 text encoders, and VAE decoder. The MMDiT contains 24 MultiModalTransformerBlocks plus a FinalLayer, totaling approximately 286 quantizable linear layers.

**Stage 2 -- Load and subsample calibration data.** From the 3,000 calibration tuples, we subsample to 256 (configurable) using timestep-stratified sampling. This ensures uniform coverage across the denoising trajectory rather than over-representing any particular noise level.

**Stage 3 -- Cache full-precision block outputs.** Before any quantization is applied, we run all 256 calibration samples through the FP model in a single forward pass. Using monkey-patched hooks on each block, we capture every block's output and stream it to disk. These cached outputs serve as the reconstruction targets throughout the remaining stages. The FP model will be destroyed by in-place quantization, so these targets must be captured first.

**Stage 4 -- In-place naive weight quantization.** Every quantizable nn.Linear layer in the MMDiT is replaced with a QuantizedLinear module that wraps the original weight with:
- A per-channel weight scale (computed from the weight's min/max range)
- An AdaRound V parameter (initialized from the fractional part of w/s, which reproduces round-to-nearest as the starting point)
- An activation quantizer (created but disabled -- it will be calibrated in Stage 6)

After this stage, the original FP model no longer exists. The model is now quantized with naive rounding, which is the worst-case baseline that AdaRound will improve upon.

**Stage 5 -- Block-wise AdaRound weight refinement.** For each of the 25 blocks sequentially:

1. Collect the block's inputs by forwarding calibration data through the model. Blocks 0 through i-1 use their frozen, previously-optimized quantized weights. This means block i's inputs reflect the real quantization error path, not idealized FP inputs.

2. Load block i's FP targets from disk (cached in Stage 3).

3. Run 3,000 iterations of AdaRound optimization:
   - Sample a mini-batch of 16 from the collected (input, target) pairs
   - Forward through the block using soft-rounded weights (V is continuous via rectified sigmoid)
   - Compute the loss: reconstruction MSE between quantized output and FP target, plus an AdaRound regularization term that pushes V toward binary values (0 or 1)
   - The regularization strength beta anneals from 2 to 20 over the first 20% of iterations. Small beta early allows the optimizer to explore rounding configurations freely; large beta late forces V toward hard decisions before freezing.
   - Update only the V parameters via Adam (original weights, scales, and biases are frozen)

4. Freeze: hard-threshold V at 0.5, converting soft rounding decisions to binary. Free the block's cached inputs and targets.

**Stage 6 -- Activation quantizer calibration.** After all blocks have optimized weight rounding, we calibrate the activation clipping ranges:

For each block, re-collect inputs through the now-fully-refined model, then for each QuantizedLinear layer:

- **MSE search (default):** Test 6 candidate clipping ranges (99th, 99.5th, 99.9th, 99.95th, 99.99th percentile, and max). For each candidate, temporarily set the clipping range, forward through the block, and measure the block-level reconstruction MSE. Select the candidate that minimizes MSE. The candidates are clustered near the top of the distribution because diffusion model activations are heavy-tailed (large mass near zero with long tails), so the interesting range tradeoff is between 99% and 100%, not 50% and 100%.

- **Percentile (fast alternative):** Simply set the clipping range to the 99.99th percentile of absolute activation values.

After calibration, each layer's activation quantizer is enabled with its selected clipping range.

**Output.** The quantized model weights, activation parameters, and training logs are saved. Loss curves per block are plotted for diagnostic review.

---

## 4. Architectural Adaptations for MMDiT

### 4.1 No Skip-Connection Splitting

Q-Diffusion Section 3.3.2 introduces a "shortcut-splitting" quantization technique for U-Net skip connections. In a U-Net, encoder features are concatenated with decoder features along the channel dimension. These concatenated channels can have activation ranges differing by up to 200x, which is catastrophic for per-tensor quantization.

MMDiT is a pure transformer with no encoder-decoder structure and no long-range skip connections. Each block applies a residual addition (not concatenation), meaning the dimensions being combined always share the same representational space. The skip-splitting technique is therefore inapplicable, and we omit it entirely.

### 4.2 Dual-Stream Reconstruction

Each MultiModalTransformerBlock processes two token streams simultaneously: image tokens (~1024 tokens at 512x512) and text tokens (~154 tokens from CLIP + T5). The streams are processed through separate projections but attend jointly (concatenated Q/K/V before scaled dot-product attention), then split back into separate streams for output.

Our reconstruction loss treats the block holistically:

    L = MSE(img_out_quant, img_out_fp) + MSE(txt_out_quant, txt_out_fp)

We sum both stream losses rather than treating them independently because the joint attention creates coupling between streams -- the quantization error in one stream's key projections affects the other stream's attention output. Optimizing them jointly allows AdaRound to find rounding decisions that balance error across both streams.

Block 23 is a special case: its text stream has `skip_post_sdpa=True`, meaning it lacks an output projection, FFN, and their associated modulation parameters. Only the image stream contributes to the reconstruction loss for this block, and only the image stream's layers are quantized.

### 4.3 adaLN Modulation Stays in FP16

The adaLN (adaptive Layer Normalization) mechanism is a defining feature of MMDiT. Each block contains modulation linear layers that take the pooled text embedding and timestep as input and produce six modulation parameters (two sets of shift, scale, and gate) that multiplicatively and additively transform the feature maps.

We keep all 49 adaLN modulation linears in FP16 for two reasons:

1. **Error amplification.** Modulation parameters apply element-wise scaling and shifting to entire feature maps. A small quantization error in a scale parameter multiplies across every token in the sequence, producing outsized distortion relative to the parameter count.

2. **Exact reconstruction targets.** During AdaRound optimization, we need the block's non-quantized components to behave identically to the FP reference. If the modulation were also quantized, the reconstruction target would need to account for modulation error interacting with projection error, making the optimization landscape harder and the targets less meaningful.

Since adaLN linears represent only 49 out of 335 total linear layers (~15%), the storage overhead of keeping them in FP16 is minimal.

### 4.4 Attention Matmul in FP16

The original Q-Diffusion paper (Section A.1) notes that for W4A8, the attention score matrices (q*k^T and attention*v) should use INT16 mixed precision to avoid quality degradation. The concern is that quantizing the softmax attention weights introduces compounding error in the weighted value aggregation.

In our MLX implementation, SDPA (scaled dot-product attention) runs entirely in FP16 via MLX's fused kernel (`mx.fast.scaled_dot_product_attention`). The quantization boundary is at the projection outputs: q, k, and v are quantized to 8-bit as they leave their respective projection layers, but the actual attention computation (q*k^T scaling, softmax, weighted sum with v) proceeds in full FP16 precision. This naturally achieves the paper's intent of protecting the attention computation from quantization artifacts.

### 4.5 Asymmetric Quantization for Post-GELU Activations

Most activations in the MMDiT span both positive and negative values, making symmetric quantization (range [-alpha, +alpha]) appropriate. However, the input to fc2 (the second FFN linear) follows a GELU activation, which produces predominantly non-negative outputs.

Using symmetric quantization for post-GELU activations wastes nearly half the representable integer range on negative values that essentially never occur. We use asymmetric quantization specifically for fc2 inputs, with an explicit zero-point that shifts the quantization grid to cover only the actual activation range [alpha_min, alpha_max]. All other layers use symmetric quantization.

---

## 5. Memory Management on 32 GB

### 5.1 Single-Model In-Place Strategy

The textbook approach to block-wise reconstruction keeps two models: a frozen FP reference model and a progressively-quantized model. For SD3 Medium, two copies of the MMDiT would consume approximately 8 GB, plus text encoders and VAE push total model memory to roughly 15 GB -- leaving insufficient headroom for calibration data, cached tensors, and optimization state on a 32 GB machine.

Our approach uses a single model that is modified in-place. Before any quantization, we cache every block's FP outputs (the reconstruction targets). Then we replace the FP model's layers with quantized versions directly. The FP model is consumed by this process, but its outputs survive in the target cache.

### 5.2 Disk-Backed FP Target Cache

The initial implementation cached all FP block outputs in memory. With 256 samples, 25 blocks, and approximately 7 MB per sample per block (image embeddings at shape (2, 1024, 1536) plus text embeddings at (2, 154, 1536) in float16), this totaled roughly 45 GB -- still exceeding available memory.

The solution is streaming FP targets to disk during collection. As each calibration sample is forwarded through the model, all 25 block outputs are captured and immediately written to individual numpy files on disk. In-memory footprint during collection is bounded to a single sample's worth of block outputs (~180 MB), regardless of how many calibration samples are used.

During AdaRound optimization, only the current block's targets are loaded into memory (~1.8 GB for 256 samples), processed, and freed before the next block's targets are loaded. The disk cache is automatically cleaned up after quantization completes.

### 5.3 Additional Memory Controls

- **Timestep-stratified subsampling**: From 3,000 calibration tuples, we subsample to 256 with equal representation per timestep bin, ensuring diverse coverage without excessive memory.
- **Mini-batching**: Each AdaRound iteration processes only 16 samples, keeping gradient computation memory bounded.
- **Aggressive cleanup**: After each block is optimized, we explicitly delete cached inputs and targets, trigger Python garbage collection, and clear the MLX metal memory cache.
- **Prompt-grouped processing**: Calibration samples are grouped by prompt to exploit DiffusionKit's modulation parameter caching, which offloads adaLN weights between prompt groups to reduce peak memory by approximately 1.3 GB.

---

## 6. What We Quantize and What We Don't

### Quantized (weight + activation)

| Layer Type | Count | Notes |
|---|---|---|
| q_proj, k_proj, v_proj | 48 each (24 blocks x 2 streams) | Self-attention projections |
| o_proj | 47 | Output projection (block 23 txt is Identity) |
| fc1 | 47 | FFN first linear (block 23 txt has no FFN) |
| fc2 | 47 | FFN second linear (block 23 txt has no FFN) |
| FinalLayer linear | 1 | Patch un-projection |
| **Total** | **~286** | |

### Kept in FP16

| Component | Count | Reason |
|---|---|---|
| adaLN modulation linears | 49 | Error amplification across feature maps |
| LayerNorm / RMSNorm | All | Negligible compute; quantization destabilizes normalization |
| GELU / SiLU activations | All | Non-linear functions; quantized in/out, not the function itself |
| SDPA attention matmul | All | Runs in FP16 natively in MLX |
| Embedders (x, context, t, y) | All | Small relative cost; quantizing inputs would corrupt signal before it reaches the backbone |

### Supported Configurations

| Config | Weight Bits | Activation Bits | Typical Use |
|---|---|---|---|
| W8A8 | 8 | 8 | Conservative baseline; smaller quality loss |
| W4A8 | 4 | 8 | Aggressive compression; AdaRound is critical for quality |

---

## 7. Evaluation Strategy

Quantization quality is assessed at two levels:

**Per-block diagnostics (during quantization):** Each block logs its reconstruction MSE before optimization (naive rounding baseline) and after (AdaRound-optimized). The expected improvement is 2-5x. Per-iteration loss curves (reconstruction loss, regularization loss, total loss, beta schedule) are saved as plots and JSON for post-hoc analysis. A block that shows no improvement or increasing MSE indicates a problem.

**End-to-end evaluation (after quantization):**
- Per-timestep noise prediction MSE: Compare the quantized model's denoiser output against the FP model for 50 calibration prompts across all timesteps. This reveals whether quantization error is uniform or concentrated at specific noise levels.
- Visual comparison: Generate images from identical prompts and seeds using FP, W8A8, and W4A8 models side by side.
- Latent trajectory analysis: Plot the latent norm at each denoising step for FP vs. quantized to detect error accumulation patterns.

---

## 8. Design Decisions Summary

1. **No DiffusionKit modifications.** All quantization code lives in `src/q_diffusion/` using monkey-patching and wrappers. The DiffusionKit submodule remains unmodified, ensuring clean separation and reproducibility.

2. **Single-model in-place with disk cache.** Avoids the ~4 GB overhead of a second model copy and the ~90 GB overhead of in-memory target caching, at the cost of disk I/O during optimization.

3. **AdaRound only for weights; statistics-only for activations.** Activations change with every input, so learning per-weight rounding decisions does not apply. Instead, we select clipping ranges from calibration data statistics, either via percentile or MSE grid search.

4. **MSE search as default activation calibration.** This is the BRECQ-inherited approach where candidate clipping ranges are evaluated against the block reconstruction objective. The simpler percentile method is available as a faster alternative. Candidates are clustered near the top (99.0 to 100.0) because diffusion activations are heavy-tailed.

5. **Beta annealing from 2 to 20 (not 20 to 2).** Small beta early in optimization allows the rectified sigmoid to produce intermediate values, giving the optimizer freedom to explore. Large beta late forces V toward 0 or 1 before the hard threshold is applied. This follows the AdaRound paper's intended schedule.

6. **3,000 iterations per block (reduced from Q-Diffusion's 10,000).** A practical concession to the M5 Pro's computational budget. With 25 blocks, the full pipeline runs 75,000 total AdaRound iterations.

7. **Fisher-information weighting as an option.** BRECQ proposes weighting the reconstruction MSE by a diagonal Hessian approximation (Fisher information) to prioritize output dimensions the model is most sensitive to. We include this as a configurable option (default off) since the unweighted MSE baseline must be validated first.
