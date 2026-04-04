# Phase 4: GPTQ Weight Quantization with Hessian-Weighted Error Compensation

## 1. Motivation and Scope

Phase 2 produces a W4A8 model with RTN weights and dynamic/static A8; **Phase 3** adds optional polynomial activation clipping. When both are applied:

- **Weights** (Phase 2) are quantized to 4-bit using MLX's built-in `mx.quantize` — per-group affine Round-to-Nearest (RTN).
- **Activations** (Phase 3, when enabled) use polynomial clipping bounds `α(σ) = poly(σ)` instead of dynamic max or static scales.

RTN is the simplest weight quantization strategy: each weight is independently rounded to the nearest representable value. It ignores the **correlations** between weight columns — if column `j` is rounded down, column `j+1` might also benefit from being rounded down (or up) to compensate. RTN also ignores **which columns matter more to the output**, treating all columns equally regardless of how much the input activations exercise them.

GPTQ (Frantar et al., 2023) addresses both limitations:

1. **Hessian-weighted importance.** The Hessian `H = 2·X^T X` (computed from calibration activations) captures how sensitive the layer output is to perturbations in each weight column. Columns receiving large activations contribute more to `H`'s diagonal.
2. **Column-wise error compensation.** After quantizing column `j`, the rounding error is redistributed to not-yet-quantized columns `j+1, ..., d_in`, weighted by the Hessian's off-diagonal entries. This pushes error toward columns that matter less.

**Goal.** Replace Phase 2's RTN weight quantization with GPTQ for the same linear layers Phase 2 quantizes (286 in the full registry). The draft `hessian.py` enumerates **block** linears only (~285 layers); `final_layer.linear` is not in `_get_block_linears` and must be added if GPTQ should cover all 286. Activation quantization (Phase 3 polynomial clipping) is unchanged — GPTQ only modifies weight rounding.

**Why Phase 3 enables better GPTQ.** Without polynomial clipping, the Hessian must be collected under either:
- Dynamic A8: `scale = max(|x|) / 127` — the clipping bound varies unpredictably per input, so the Hessian mixes samples with different effective activation distributions.
- Static A8: a fixed conservative scale — the Hessian is computed on overly-padded activations that don't reflect the per-σ behavior.

With polynomial clipping, each calibration sample arrives with its known σ, and `α(σ) = poly(σ)` gives the correct clipping bound. The Hessian then reflects the true per-σ activation distribution the model will see at inference. Phase 4 can also be run in **raw mode** (no fake-quantization of activations during Hessian collection), which yields an activation-agnostic Hessian independent of clipping decisions.

**Scope.** Phase 4 focuses on the core GPTQ algorithm and Hessian collection. The existing draft files under `src/phase4/` will be refined.

---

## 2. How GPTQ Relates to Prior Phases

```
Phase 1: Collect activation trajectories + weight salience (diagnostics/)
    │
    ▼
Phase 2: SSC calibration → CSB balancing → Absorb into adaLN → RTN W4 + A8
    │
    ▼
Phase 3: Fit polynomial clipping bounds → poly_schedule.json
    │
    ▼
Phase 4: Collect Hessians (using poly A8 or raw) → GPTQ W4 → Save
```

### 2.1 What GPTQ replaces

| Component | Phase 2 (RTN) | **Phase 4 (GPTQ)** |
|-----------|---------------|---------------------|
| Weight quantization method | `mx.quantize` — independent per-value rounding | Column-wise with Hessian-weighted error compensation |
| Activation clipping during calibration | N/A (RTN doesn't need calibration data) | Controlled by Phase 3 poly schedule (or raw mode) |
| Calibration data required | None (RTN is data-free) | Yes — forward passes through the model to accumulate `H = X^T X` |
| Error propagation | None — each weight rounded independently | Cross-column compensation via `H^{-1}` |
| Per-group or per-channel scales | Per-group affine (MLX default) | Per-group symmetric (GPTQ convention) |

### 2.2 What stays the same

- **CSB balancing** — Phase 2's absorption into adaLN and online `b_inv` for o_proj/fc2 are unchanged. GPTQ quantizes the already-balanced weights.
- **Phase 3 polynomial clipping** — The `poly_schedule.json` and `W4A8PolyLinear` module are unchanged. GPTQ only affects the weight values inside `nn.QuantizedLinear`.
- **Target layers** — Same layers as Phase 2 once the runner covers the full registry; the current draft installs collectors only on block linears (~285), not `final_layer.linear`.
- **Activation quantization at inference** — Still `poly(σ) / 127` per Phase 3.

### 2.3 Quantization scheme: symmetric vs affine

Phase 2 (MLX RTN) uses **per-group affine** quantization with zero-point:

$$
W_q = \text{round}\left(\frac{W - \beta}{\text{scale}}\right), \quad \text{scale} = \frac{\max(W_g) - \min(W_g)}{2^b - 1}
$$

Phase 4 (GPTQ) uses **per-group symmetric** quantization (no zero-point):

$$
W_q = \text{clamp}\left(\text{round}\left(\frac{W}{\text{scale}}\right),\; -q_{\max},\; q_{\max}\right), \quad \text{scale} = \frac{\max(|W_g|)}{q_{\max}}
$$

where $q_{\max} = 2^{b-1} - 1 = 7$ for 4-bit. Symmetric quantization simplifies the GPTQ error compensation loop (no zero-point to track) and is standard in GPTQ implementations. The final weights will be repacked into MLX's `nn.QuantizedLinear` format for inference.

---

## 3. Theory

### 3.1 The Hessian approximation

For a linear layer $Y = XW^T$, the squared reconstruction error from quantizing $W$ is:

$$
\mathcal{L} = \|XW^T - X\hat{W}^T\|_F^2 = \sum_{i=1}^{d_{\text{out}}} (w_i - \hat{w}_i)^T \underbrace{X^T X}_{H/2} (w_i - \hat{w}_i)
$$

where $w_i$ is the $i$-th **row** of $W$ (treated as a column vector $\in \mathbb{R}^{d_\text{in}}$). The loss decomposes into independent per-row problems sharing the same Hessian:

$$
H = 2 \cdot X^T X \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}
$$

$H$ is independent of $W$ — it depends only on the input activations $X$. In practice, $X$ is the concatenation of activations from all calibration samples (all prompts × all denoising steps), reshaped to `(N_total, d_in)`. GPTQ exploits the per-row decomposition: each row of $W$ is quantized using the same $H$.

### 3.2 σ-aware Hessian collection

When collecting `H` with Phase 3's polynomial clipping, each activation sample is fake-quantized before accumulation:

$$
X_{\text{fq}} = \text{clamp}\!\left(\text{round}\!\left(\frac{X}{\alpha(\sigma)/127}\right),\; -127,\; 127\right) \cdot \frac{\alpha(\sigma)}{127}
$$

$$
H = 2 \sum_{\text{samples}} X_{\text{fq}}^T X_{\text{fq}}
$$

This makes the Hessian capture the **post-clipping** activation distribution. The GPTQ weight rounding then minimizes error under the actual inference-time activation regime.

In **raw mode**, the Hessian is collected from full-precision activations (no fake-quantization). This yields an activation-agnostic Hessian that doesn't depend on clipping parameters, useful as a baseline or when the poly schedule isn't finalized.

### 3.3 Column-wise quantization with error compensation

Let $C = \text{Chol}_{\text{upper}}(H^{-1})$ — the upper-triangular Cholesky factor of $H^{-1}$. In code this is `H_inv_chol = cholesky(H_inv).T`.

GPTQ processes weights column-by-column in blocks of size $B$ (default 128). For column $j$:

1. **Quantize**: $\hat{w}_j = \text{round}(w_j / s_j) \cdot s_j$, where $s_j$ is the per-group scale.
2. **Error**: $e_j = (w_j - \hat{w}_j) / C_{jj}$.
3. **Compensate**: For all not-yet-quantized columns $k > j$:

$$
w_k \leftarrow w_k - e_j \cdot C_{jk}
$$

The Cholesky factor $C$ provides the cross-column relationships efficiently. Columns with large $C_{jk}$ are strongly correlated with column $j$ in the loss landscape, so they should absorb more of column $j$'s error.

**Block structure.** Columns are processed in blocks of size $B$. Within a block, compensation is applied column-by-column. After each block, the accumulated error is propagated to all remaining columns via a single matrix multiply:

$$
W_{:, j_{\text{end}}:} \leftarrow W_{:, j_{\text{end}}:} - E \cdot C_{i:j_{\text{end}},\; j_{\text{end}}:}
$$

This reduces memory traffic compared to full column-by-column updates.

### 3.4 Numerical conditioning

The Hessian can be ill-conditioned (some columns rarely activated). Safeguards:

1. **Damping**: $H \leftarrow H + \lambda I$, where $\lambda = \text{damp\_percent} \times \max(\text{diag}(H))$ (default 1%).
2. **Cholesky fallback chain**: If Cholesky fails → increase damping 10× → pseudoinverse → diagonal.
3. **NaN/Inf sanitization**: Non-finite Hessian entries are zeroed before factorization.
4. **Minimum diagonal**: $C_{jj} \geq 10^{-6}$ to prevent division by zero in the error formula.

### 3.5 Per-group symmetric scales

For each output row $r$ and each group $g$ of $G$ columns (default 128):

$$
\text{scale}_{r,g} = \frac{\max_{j \in g} |W_{r,j}|}{q_{\max}}, \quad q_{\max} = 2^{b-1} - 1
$$

$$
W_{q,r,j} = \text{clamp}\!\left(\text{round}\!\left(\frac{W_{r,j}}{\text{scale}_{r,g(j)}}\right),\; -q_{\max},\; q_{\max}\right)
$$

The scales array has shape `(d_out, n_groups)` — one scale per (row, group) pair, **not** one per group globally.

Scales are clamped to a minimum of $10^{-10}$ to avoid division by zero.

---

## 4. Data Flow

### 4.1 Offline: Hessian collection and GPTQ quantization

```
Phase 2 checkpoint                Phase 3 schedule
quantized/<tag>/                  quantized/<tag>/
├── mmdit_quantized.safetensors   ├── poly_schedule.json
├── quantize_config.json          │
├── calibration.npz               │
└── calibration_meta.json         │
         │                        │
         ▼                        ▼
    ┌─────────────────────────────────────────────────────┐
    │  Phase 4A: Collect Hessians                         │
    │                                                     │
    │  For each calibration prompt (seed, prompt):        │
    │    1. Encode text → conditioning, pooled            │
    │    2. Cache adaLN modulation params                 │
    │    3. For each Euler step with σ_t:                 │
    │       a. Set σ on all collector proxies              │
    │       b. Forward pass through all 24 blocks         │
    │       c. Each _HessianCollector accumulates:        │
    │          X_fq = fake_quant(X, α(σ))   [poly mode]  │
    │          H += X_fq^T @ X_fq           [float32]    │
    │       d. Periodic mx.eval() every 3 steps          │
    │    4. Reset adaLN cache for next prompt             │
    │                                                     │
    │  Output: {poly_key: H} for all ~285 layers          │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │  GPTQ: Quantize each layer's weights                │
    │                                                     │
    │  For each layer l in 285 layers:                    │
    │    1. W = dequant(phase2_weights)  or  FP16 W       │
    │    2. H_l = collector.get_hessian()  [numpy]        │
    │    3. W_q, scales, mse = gptq_quantize(W, H, ...)  │
    │       - Damp H, Cholesky H^{-1}                    │
    │       - Column-wise quantize in blocks of 128       │
    │       - Error compensation via H^{-1} off-diags    │
    │    4. Store (W_q, scales, mse) per layer            │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
              GPTQ-quantized W4 weights
              + Phase 3 poly_schedule.json (unchanged)
              → Save as new checkpoint
```

### 4.2 Online: inference (unchanged from Phase 3)

Inference is identical to Phase 3. The only difference is that the W4 weights inside `nn.QuantizedLinear` are GPTQ-optimized rather than RTN-quantized. The `W4A8PolyLinear` module and σ hook remain the same.

```
Denoiser loop (Euler sampler, T steps):
│
├── install_sigma_hook(mmdit)
│
├── For each step t with σ_t:
│   │
│   ├── mmdit(timestep=σ_t*1000, ...)  ← hook sets σ register
│   │
│   └── For each W4A8PolyLinear layer l:
│       │
│       ├── 1. [optional] x = x * b_inv       (online CSB, o_proj/fc2)
│       ├── 2. α = poly_eval(coeffs_l, σ_t)   (Phase 3)
│       │      scale = max(α, ε) / 127
│       ├── 3. x_q = clamp(round(x / scale), -128, 127)
│       │      x̂ = x_q * scale
│       └── 4. return qlinear(x̂)              (GPTQ W4 matmul)
```

### 4.3 Hessian collection via proxy layers

The Hessian is collected transparently by replacing each `nn.Linear` (or `W4A8Linear` / `W4A8PolyLinear`) with a `_HessianCollector` proxy. The proxy:

1. Calls the original layer's `__call__` to produce the correct output (model behavior unchanged).
2. Intercepts the input `X` and accumulates `H += X_2d^T @ X_2d` in float32.
3. Delegates all attribute access to the wrapped layer (transparent to the rest of the model).

After collection, the proxy is removed and the original layer restored.

### 4.4 σ propagation during collection

The collector proxies need the current σ to evaluate polynomial clipping bounds. Unlike Phase 3 inference (which uses a module-level σ register + mmdit hook), Hessian collection sets σ explicitly on each collector:

```
For each Euler step with σ_t:
  1. set_sigma_all_blocks(all_collectors, σ_t)   ← explicit per-proxy
  2. denoiser(x, timestep, σ, ...)               ← forward pass fires collectors
```

This avoids coupling to Phase 3's σ register and works regardless of whether the model has `W4A8PolyLinear` modules installed.

---

## 5. Module Design

### 5.1 Files

```
src/phase4/
├── __init__.py              # Package marker
├── utils.py                 # Shared utilities:
│                            #   - load_prompt_file (seed\tprompt pairs)
│                            #   - _get_block_linears (enumerate quantisable layers)
│                            #   - _get_nested / _set_nested (model tree navigation)
│                            #   - full_path_to_poly_key (path → schedule key mapping)
│                            #   - compute_scales / dequantize (NumPy quantization math)
│                            #   - get_poly_alpha (evaluate poly schedule entry)
│                            #   - _reset_modulation_cache (restore adaLN after offload)
├── hessian.py               # Phase A: Hessian collection via _HessianCollector
│                            #   - install_collectors[_all_blocks]
│                            #   - remove_collectors[_all_blocks]
│                            #   - set_sigma_all[_blocks]
│                            #   - eval_hessians[_all_blocks]
│                            #   - collect_hessians_global (full orchestration)
├── gptq_quantize.py         # Core GPTQ algorithm (pure NumPy + SciPy)
│                            #   - gptq_quantize(W, H, bits, damp, block_size, group_size)
└── run_phase4.py            # CLI: prompt file → Hessian collection → GPTQ → save
     (to be implemented)
```

Phase 4 is a **self-contained package**. It imports from Phase 2 (for model loading) and Phase 3 (for the poly schedule), but does not modify any files in those packages.

### 5.2 Layer enumeration

`_get_block_linears` walks each MMDiT block's two transformer sub-blocks (`image_transformer_block`, `text_transformer_block`) and yields the six quantisable linear paths per sub-block:

| Path | Shape | Notes |
|------|-------|-------|
| `attn.q_proj` | (1536, 1536) | Has bias |
| `attn.k_proj` | (1536, 1536) | No bias |
| `attn.v_proj` | (1536, 1536) | Has bias |
| `attn.o_proj` | (1536, 1536) | Block 23 txt is Identity |
| `mlp.fc1` | (6144, 1536) | Block 23 txt has no FFN |
| `mlp.fc2` | (1536, 6144) | Block 23 txt has no FFN |

This gives 12 linears per block × 24 blocks = 288, minus block 23 text's missing o_proj/fc1/fc2 = ~285 layers. The `final_layer.linear` is handled separately if needed.

### 5.3 Poly schedule key mapping

`full_path_to_poly_key(block_idx, full_path)` converts the model tree path to the key format used in `poly_schedule.json`:

```
block_idx=0, "image_transformer_block.attn.q_proj" → "mm0_img_attn_q_proj"
block_idx=5, "text_transformer_block.mlp.fc2"      → "mm5_txt_mlp_fc2"
```

### 5.4 `_HessianCollector` — transparent proxy

```python
class _HessianCollector:
    def __init__(self, wrapped, poly_entry=None,
                 static_alpha=None, raw_hessian=False):
        self._wrapped = wrapped
        self._H = None          # Accumulated X^T X [d_in, d_in]
        self._n_samples = 0
        self._poly_entry = poly_entry
        self._sigma = None
        self._raw_hessian = raw_hessian

    def __call__(self, x):
        out = self._wrapped(x)  # Original forward (unchanged)

        # Choose activation mode for Hessian
        if self._raw_hessian:
            x_fq = x             # Full precision
        elif self._poly_entry and self._sigma:
            alpha = get_poly_alpha(self._poly_entry, self._sigma)
            scale = alpha / 127.0
            x_fq = clip(round(x / scale), -127, 127) * scale
        else:
            x_fq = x

        # Accumulate in float32
        x_2d = x_fq.reshape(-1, x_fq.shape[-1]).astype(float32)
        self._H += x_2d.T @ x_2d
        self._n_samples += x_2d.shape[0]
        return out

    def get_hessian(self):
        """Return 2 * H as NumPy array."""
        return 2.0 * np.array(self._H, dtype=np.float32)
```

The proxy is transparent: `__getattr__` delegates to the wrapped layer, so the rest of the model sees the original module's attributes (weight, bias, etc.).

### 5.5 `gptq_quantize` — core algorithm

```python
def gptq_quantize(W, H, bits=4, damp_percent=0.01,
                  block_size=128, group_size=128):
    """
    Args:
        W: (d_out, d_in) float32 weight matrix.
        H: (d_in, d_in) float32 Hessian (2 * X^T X).
        bits: target bit-width.
        block_size: column block size for error compensation.
        group_size: columns per quantization group.

    Returns:
        W_q_int: (d_out, d_in) int8 quantized weights.
        scales: (d_out, n_groups) or (d_out,) float32.
        weight_mse: float — total ||W - dequant(W_q)||^2.
    """
```

The algorithm:

1. Compute per-(row, group) symmetric scales from the original weights.
2. Damp H, Cholesky-factor to get `C = Chol_upper(H^{-1})`.
3. For each block of `block_size` columns:
   - For each column `j` in the block:
     - Quantize: `w_q = clamp(round(w / scale), -qmax, qmax)`
     - Error: `e = (w - dequant(w_q)) / C[j,j]`
     - Intra-block compensation: `W[:, j+1:j_end] -= outer(e, C[j, j+1:j_end])`
   - Inter-block compensation: `W[:, j_end:] -= E @ C[i:j_end, j_end:]`
4. Compute final MSE: `||W_orig - dequant(W_q)||^2`.

### 5.6 Modulation cache handling

DiffusionKit's `cache_modulation_params` pre-computes adaLN outputs and offloads the MLP weights. After each prompt's denoising loop, `_reset_modulation_cache` reloads the adaLN weights from the pipeline so subsequent prompts work correctly. This mirrors Phase 1/2's pattern (see Phase2.md §13).

---

## 6. Hessian Collection Modes

| Mode | Activation input to `H` | When to use |
|------|------------------------|-------------|
| **poly** | Fake-quantized via `α(σ) = poly(σ)` from poly_schedule | Default — Hessian reflects the actual inference-time distribution |
| **raw** | Full-precision (no fake-quantization) | When poly schedule isn't finalized, or as a baseline |
| **static** | Fake-quantized with a fixed alpha per layer | When using static A8 instead of polynomial |

All three modes produce the same shaped output: one `[d_in, d_in]` Hessian per layer. The mode only affects what distribution of `X` the Hessian captures.

### 6.1 Why poly mode is preferred

In poly mode, the Hessian is conditioned on the actual clipping that will happen at inference:

$$
H_{\text{poly}} = 2 \sum_{\text{samples}} X_{\text{fq}}^T(\sigma) \cdot X_{\text{fq}}(\sigma)
$$

This means GPTQ optimizes weight rounding for the *clipped* activation distribution. If `α(σ)` clips large activations at timestep $\sigma$, those clipped values are what the matmul actually sees — so the Hessian should reflect that.

### 6.2 Why raw mode exists

Raw mode collects the Hessian from unclipped activations:

$$
H_{\text{raw}} = 2 \sum_{\text{samples}} X^T X
$$

This is useful when:
- The poly schedule is being iterated on and isn't final yet.
- You want weight quantization decisions independent of activation clipping.
- Comparing GPTQ quality with and without clipping-aware Hessians.

---

## 7. Memory and Computation Considerations

### 7.1 Hessian storage

Each Hessian is `[d_in, d_in]` in float32:

| Layer type | d_in | Hessian size | Count | Total |
|------------|------|-------------|-------|-------|
| q/k/v/o_proj, fc1 | 1536 | 1536² × 4B ≈ 9 MB | ~238 | ~2.1 GB |
| fc2 | 6144 | 6144² × 4B ≈ 150 MB | ~47 | ~7.1 GB |
| **Total** | | | ~285 | **~9.2 GB** |

`fc1` is `Linear(1536 → 6144)` → **d_in = 1536**. `fc2` is `Linear(6144 → 1536)` → **d_in = 6144**; those Hessians dominate memory.

Collecting all Hessians simultaneously is feasible on machines with ≥16 GB RAM, but the **`fc2`** layers dominate. Strategies to manage this:

- **Process block-by-block**: Collect Hessians for one block at a time, run GPTQ, then discard before moving to the next block. Reduces peak memory to ~12 layers' worth.
- **All-blocks-at-once** (current draft): Collects all ~285 Hessians in one pass over the calibration data. Faster (single pass) but requires more memory.

### 7.2 Lazy graph bounding

MLX uses lazy evaluation. Each `H += X^T @ X` adds to the computation graph without executing. With 285 collectors × 30 steps per prompt, the graph would grow to ~8,550 pending operations before any evaluation — risking OOM.

The current implementation calls `mx.eval()` on all Hessians every 3 denoising steps. At 285 layers × 3 steps = 855 pending `XtX` operations per eval boundary, this keeps the graph manageable.

### 7.3 Float32 accumulation

Hessians are accumulated in float32 despite the model running in float16. This is critical: `X^T X` over ~4096 tokens with float16 values easily exceeds the float16 max of 65504. The explicit `.astype(mx.float32)` before the matmul prevents overflow.

### 7.4 GPTQ computation cost

GPTQ per layer:
- Cholesky of `[d_in, d_in]`: O(d_in³ / 3). For d_in = 6144 (**fc2**): on the order of tens of GFLOP per layer.
- Column-wise quantization: O(d_out × d_in²) per layer.
- Total per layer is dominated by the Cholesky on large **d_in**; full-model CPU time is typically minutes (NumPy, not GPU).

### 7.5 Calibration data volume

The number of calibration prompts affects Hessian quality:
- **Too few** (< 5): Hessian under-represents the activation distribution; GPTQ may overfit to specific inputs.
- **Sweet spot** (8–32): Sufficient coverage of the prompt distribution. Each prompt provides 30 steps × (S_img + S_txt) tokens of data.
- **Diminishing returns** (> 64): Additional prompts add little new information.

---

## 8. Repacking GPTQ Weights into MLX Format

GPTQ produces `W_q_int [d_out, d_in]` (int8, symmetric) and `scales [d_out, n_groups]`. These must be repacked into MLX's `nn.QuantizedLinear` format for efficient inference via `mx.quantized_matmul`.

MLX's format stores:
- `weight`: packed uint32 array of shape `[d_out, d_in * bits / 32]`.
- `scales`: float16 per-group scales.
- `biases`: float16 per-group biases (zero for symmetric, but MLX may still require the field).

The repacking step converts GPTQ's symmetric int8 values to MLX's unsigned representation:

$$
W_{\text{unsigned}} = W_q + q_{\max} \quad (\text{shift from } [-7,7] \text{ to } [0,14])
$$

This conversion, packing into uint32, and creating `nn.QuantizedLinear` from the packed representation will be implemented in the runner script.

---

## 9. Integration Points

### 9.1 Input: Phase 2 checkpoint + Phase 3 schedule

Phase 4 requires:

| Artifact | Source | Used for |
|----------|--------|----------|
| `mmdit_quantized.safetensors` | Phase 2 | Model weights (to extract original FP16 or dequantized RTN weights) |
| `quantize_config.json` | Phase 2 | Layer metadata (d_in, d_out, has_bias, group_size) |
| `calibration.npz` | Phase 2 | CSB balancing vectors (if re-balancing is needed) |
| `poly_schedule.json` | Phase 3 | Polynomial entries for σ-aware fake-quantization during Hessian collection |
| Prompt file | Settings | `(seed, prompt)` pairs for calibration data generation |

### 9.2 Output: GPTQ-quantized checkpoint

Phase 4 produces a new checkpoint in the same directory structure:

```
quantized/<tag>/
├── mmdit_quantized.safetensors    ← GPTQ-quantized weights (replaces Phase 2 RTN)
├── quantize_config.json           ← updated with "weight_quant": "gptq" metadata
├── calibration.npz                ← unchanged
├── calibration_meta.json          ← unchanged
├── poly_schedule.json             ← unchanged
└── gptq_meta.json                 ← GPTQ-specific metadata (per-layer MSE, config)
```

### 9.3 Loading at inference

The inference path is unchanged from Phase 3:

```python
from src.phase3.quantize_poly import load_quantized_model_poly
load_quantized_model_poly(pipeline, quantized_dir)
```

This works because GPTQ only changes the *values* inside `nn.QuantizedLinear`, not the module structure. The safetensors file contains the same keys with GPTQ-optimized values.

---

## 10. CLI Reference (Planned)

### 10.1 Full GPTQ pipeline

```bash
# GPTQ with poly-aware Hessians (default)
python -m src.phase4.run_phase4 \
    --quantized-dir quantized/<tag>/ \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --num-prompts 16 \
    --num-steps 30 \
    --bits 4 \
    --group-size 128 \
    --block-size 128 \
    --damp-percent 0.01

# GPTQ with raw Hessians (activation-agnostic)
python -m src.phase4.run_phase4 \
    --quantized-dir quantized/<tag>/ \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --num-prompts 16 \
    --raw-hessian

# Override group size (smaller = finer-grained, larger = more compression)
python -m src.phase4.run_phase4 \
    --quantized-dir quantized/<tag>/ \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --num-prompts 16 \
    --group-size 64
```

### 10.2 Key CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--quantized-dir` | (required) | Phase 2 checkpoint directory containing `quantize_config.json` and `poly_schedule.json` |
| `--prompts-file` | (required) | Tab-separated `seed\tprompt` file for calibration |
| `--num-prompts` | 16 | Number of prompts to use for Hessian collection |
| `--num-steps` | 30 | Euler denoising steps per prompt |
| `--bits` | 4 | Weight quantization bit-width |
| `--group-size` | 128 | GPTQ group size for per-group scales |
| `--block-size` | 128 | Column block size for error compensation |
| `--damp-percent` | 0.01 | Hessian diagonal damping factor |
| `--raw-hessian` | False | Collect Hessian from full-precision activations |
| `--cfg-weight` | 4.0 | Classifier-free guidance weight |
| `--output-dir` | (same as quantized-dir) | Where to save GPTQ-quantized checkpoint |

---

## 11. Known Issue: Polynomial Coefficient Order

Phase 3 stores polynomial coefficients in **ascending-power order**: `[c₀, c₁, ..., c_d]` (constant term first). Phase 3's `poly_eval.py` uses Horner's method starting from the highest-degree coefficient and working down.

Phase 4's `utils.py` currently evaluates polynomials via `np.polyval(coeffs, sigma)`, which expects **descending-power order**: `[c_d, ..., c₁, c₀]` (highest-degree first).

This mismatch must be resolved before Phase 4 is finalized. Options:
1. **Reverse the coefficients** in `get_poly_alpha`: `np.polyval(coeffs[::-1], sigma)`.
2. **Rewrite** `get_poly_alpha` to use a Horner loop matching Phase 3's convention.
3. **Use `np.polynomial.polynomial.polyval`** which expects ascending order (unlike `np.polyval`).

---

## 12. Implementation Status

### Draft (to be refined)

| File | Status | Notes |
|------|--------|-------|
| `utils.py` | Draft | Core utilities complete; `get_poly_alpha` has coefficient order issue (§11) |
| `hessian.py` | Draft | Hessian collection + full orchestration; needs `__init__.py` |
| `gptq_quantize.py` | Draft | Core algorithm complete; needs integration with MLX repacking |

### To be implemented

| Step | Description |
|------|-------------|
| 1 | Create `__init__.py` |
| 2 | Fix `get_poly_alpha` coefficient order (§11) |
| 3 | Implement `run_phase4.py` CLI runner |
| 4 | Implement weight repacking: GPTQ int8 → MLX `nn.QuantizedLinear` format |
| 5 | Implement checkpoint saving (safetensors + metadata) |
| 6 | End-to-end testing with a small number of prompts |
| 7 | Benchmark GPTQ vs RTN on the evaluation set |

---

## 13. Hyperparameter Guidance

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `num_prompts` | 8–32 | More prompts improve Hessian quality with diminishing returns |
| `group_size` | 128 | Standard GPTQ default; 64 gives finer granularity at slightly larger scale storage |
| `block_size` | 128 | Standard; larger blocks reduce inter-block compensation overhead |
| `damp_percent` | 0.01 | 1% damping; increase to 0.05–0.10 if Cholesky fails frequently |
| `bits` | 4 | Match Phase 2; could explore 3-bit for more aggressive compression |
| `raw_hessian` | False | Default poly mode; use raw for ablation studies |

---

## 14. Expected Quality Improvement over RTN

Based on GPTQ literature and the characteristics of this model:

| Aspect | RTN (Phase 2) | GPTQ (Phase 4) |
|--------|--------------|-----------------|
| Weight MSE | Baseline | Typically 2–5× lower |
| Sensitive columns | Over-quantized (no awareness) | Error redistributed to less-sensitive columns |
| Flat weight regions | Same treatment as important regions | Absorb more error from important columns |
| End-to-end image quality | Good for most layers, degraded for high-salience layers | Expected improvement especially on late-block fc1/fc2 (largest d_in, most room for compensation) |

The improvement is expected to be most pronounced on:
- **fc2 layers** (d_in = 6144 — the most columns for cross-column compensation)
- **Late text-side layers** (blocks 20–22) with extreme weight salience
- **final_layer.linear** with high temporal variation

---

## 15. References

- **Phase 2 documentation:** `src/Phase2.md` (CSB, SSC, W4A8 architecture, checkpoint format).
- **Phase 3 documentation:** `src/PHASE3.md` (polynomial clipping, poly_schedule.json, σ hook).
- **Architecture diagram:** `useful_doc/model.txt`.
- **GPTQ paper:** Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2023).
- **Phase 4 draft implementation:** `src/phase4/` (Hessian collection, GPTQ algorithm, utilities).
