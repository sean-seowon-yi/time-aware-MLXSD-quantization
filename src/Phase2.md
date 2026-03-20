# Phase 2: W4A8 Quantization via CSB + SSC for SD3 Medium

## 1. Scope and Goals

**Objective.** Apply PTQ4DiT's Channel-wise Salience Balancing (CSB) and Spearman's ρ-guided Salience Calibration (SSC) to the MMDiT backbone of Stable Diffusion 3 Medium, producing a W4A8-quantized model (4-bit weights, 8-bit activations).

**Quantization targets.** 286 `nn.Linear` layers inside the MMDiT denoiser:

| Family | Shape (out, in) | Count | Notes |
|---|---|---|---|
| q_proj | (1536, 1536) | 48 | 24 blocks × img + txt. Has bias. |
| k_proj | (1536, 1536) | 48 | No bias. |
| v_proj | (1536, 1536) | 48 | Has bias. |
| o_proj | (1536, 1536) | 47 | Block 23 txt is Identity (skipped). |
| fc1 | (6144, 1536) | 47 | Block 23 txt has no FFN. |
| fc2 | (1536, 6144) | 47 | Block 23 txt has no FFN. |
| final_layer.linear | (64, 1536) | 1 | Patch unprojection. |
| **Total** | | **286** | |

**Excluded from quantization (kept FP16):**

- `context_embedder` — no preceding adaLN for absorption; extreme salience (854) but runs once per prompt, minimal memory contribution.
- adaLN modulation linears (24 img + 24 txt + 1 FinalLayer = 49 total) — modulation errors amplify across entire feature maps.
- LayerNorm (`norm1`, `norm2`, `norm_final`) — normalization precision is critical.
- Activation functions (SiLU in adaLN Sequential, GELU in FFN).
- SDPA (`softmax(Q·K^T/√d) · V`) — kept FP16.
- Embedders (`x_embedder`, `t_embedder`, `y_embedder`) — small relative memory cost.
- QKNorm (RMSNorm per head).

**Key design decision.** CSB and SSC are applied **per-modality**: image-pathway layers use image activation statistics, text-pathway layers use text activation statistics. Phase 1 findings show significant modality asymmetry (Section 4 of `phase1_findings.md`): text-side mean max activation salience is 2× that of image-side, and Spearman ρ differs systematically between pathways.

**Input.** Phase 1 diagnostic data under `diagnostics/`:
- Activation trajectories: `diagnostics/activation_stats/{layer_name}.npz` — per-layer `[num_steps, d_in]` arrays of per-channel max activation magnitudes across sigma steps.
- Weight salience: `diagnostics/weight_stats.npz` — per-layer `[d_in]` arrays of per-channel max weight magnitudes.
- Configuration: `diagnostics/config.json` — sigma schedule, prompts, seeds used during collection.

**Output.** A quantized model checkpoint with:
- Balanced and 4-bit-quantized weights for all 286 target layers.
- Absorbed balancing parameters in adaLN modulation weights.
- Per-layer `b_inv` vectors for layers requiring online balancing.

---

## 2. Theoretical Background

### 2.1 Salience (PTQ4DiT Eq. 4)

For a linear layer with input activation \(X \in \mathbb{R}^{B \times N \times d_{in}}\) and weight \(W \in \mathbb{R}^{d_{out} \times d_{in}}\):

\[
s(X_j^{(\sigma)}) = \max_{b,n} |X_{b,n,j}^{(\sigma)}|, \quad s(W_j) = \max_{i} |W_{i,j}|
\]

- Activation salience is **time-dependent**: it varies across the sigma (denoising) trajectory.
- Weight salience is **time-independent**: weights are fixed.

### 2.2 SSC: Time-Aware Calibration Weighting (PTQ4DiT Eq. 11)

At each sigma step \(\sigma_t\), compute the Spearman rank correlation \(\rho_t\) between the activation salience vector \(s(X^{(\sigma_t)})\) and the weight salience vector \(s(W)\). Then:

\[
\eta_t = \frac{\exp(-\rho_t)}{\sum_{\tau} \exp(-\rho_\tau)}
\]

Timesteps with **lower** \(\rho\) (stronger complementarity) receive **higher** weight. This is a softmax over \(-\rho\).

The SSC-weighted representative activation salience for channel \(j\):

\[
\bar{s}(X_j) = \sum_t \eta_t \cdot s(X_j^{(\sigma_t)})
\]

This replaces the naive time-average with a complementarity-aware average, giving more influence to timesteps where CSB can be most effective.

### 2.3 CSB: Balancing Vector (SmoothQuant-style, PTQ4DiT Eq. 7)

For each target linear layer, compute a per-channel balancing factor:

\[
b_j = \left(\frac{\bar{s}(X_j)}{s(W_j)}\right)^{\alpha}, \quad \alpha \in (0, 1)
\]

Default \(\alpha = 0.5\) (geometric mean, from SmoothQuant). The balancing matrix is \(B = \text{diag}(b_1, \dots, b_{d_{in}})\).

The balanced linear operation:

\[
Y = X W^T = \underbrace{(X \cdot B^{-1})}_{\text{balanced activation}} \cdot \underbrace{(W \cdot B)^T}_{\text{balanced weight}}
\]

In element-wise terms:
- Balanced activation: \(\tilde{X}_j = X_j / b_j\) — reduces dynamic range of high-salience activation channels.
- Balanced weight: \(\tilde{W}_{i,j} = W_{i,j} \cdot b_j\) — increases weight magnitude for those channels (which had low weight salience due to complementarity).

After balancing, both \(\tilde{X}\) and \(\tilde{W}\) have more uniform per-channel dynamic ranges, making uniform quantization significantly more effective.

### 2.4 Numerical Stability

Clamp balancing factors to avoid extreme values:

\[
b_j = \text{clamp}\left(b_j, \; b_{\min}, \; b_{\max}\right)
\]

Defaults: \(b_{\min} = 10^{-5}\), \(b_{\max} = 10^{5}\). Additionally, if \(s(W_j) < \epsilon\) (dead weight channel), set \(b_j = 1\) (no balancing).

---

## 3. Calibration Pipeline

Phase 2 calibration reuses Phase 1 diagnostic data. No new data collection is required.

### 3.1 Load Phase 1 Data

For each target layer \(l\):

1. **Activation trajectory**: Load `diagnostics/activation_stats/{l}.npz` → `act_channel_max` of shape `[T, d_in]` where `T = num_steps` (28 by default). Each entry is the per-channel max activation magnitude at that sigma step, aggregated across all calibration prompts and seeds.
2. **Weight salience**: Load `diagnostics/weight_stats.npz` → `w_channel_max` of shape `[d_in]`. The per-channel max weight magnitude.
3. **Sigma schedule**: Load `diagnostics/config.json` → `sigma_values` for the sigma-to-step mapping (needed only for reference; SSC operates on the ordered step index).

### 3.2 Compute SSC Weights

For each layer \(l\):

```python
rho_trajectory = compute_spearman_trajectory(act_trajectory, wt_salience)
ssc_weights = compute_ssc_weights(rho_trajectory)
```

Both functions are already implemented in `src/phase1/analyze.py` and can be directly reused.

- `rho_trajectory`: shape `[T]`, Spearman ρ at each step.
- `ssc_weights`: shape `[T]`, η_t values summing to 1.

### 3.3 Compute SSC-Weighted Activation Salience

```python
weighted_act_salience = ssc_weights @ act_trajectory  # [T] @ [T, d_in] → [d_in]
```

This produces \(\bar{s}(X_j)\) for every channel.

### 3.4 Compute Balancing Vector

```python
b = (weighted_act_salience / (wt_salience + eps)) ** alpha
b = np.clip(b, b_min, b_max)
```

Where `eps = 1e-12`, `alpha = 0.5`, `b_min = 1e-5`, `b_max = 1e5`.

---

## 4. CSB for Shared-Input Layers (QKV Projections)

Q, K, and V projections within the same TransformerBlock share the same input tensor (`modulated_pre_attention`), which is the output of `affine_transform(x, shift=β₁, scale=γ₁, norm_module=norm1)`. Since the adaLN absorption modifies the shared input, only **one** balancing vector \(b_{\text{qkv}}\) can be applied to it. However, each projection has its own weight matrix with potentially different weight salience profiles.

Two methods are available for computing the shared balancing vector.

### 4.1 Method 1 — Conservative (Max Weight Salience)

Merge the weight salience across Q, K, V by taking the per-channel maximum:

\[
s_{\text{merged}}(W_j) = \max\left(s(W_{q,j}),\; s(W_{k,j}),\; s(W_{v,j})\right)
\]

Then compute the shared balancing vector:

\[
b_{\text{qkv},j} = \left(\frac{\bar{s}(X_j)}{s_{\text{merged}}(W_j)}\right)^{\alpha}
\]

**Rationale.** For each channel \(j\), the projection with the highest weight salience gets the optimal balance. The other two projections receive a slightly conservative balance (their weight channels are scaled up slightly less than their individual optima). This avoids any under-balancing that could cause weight quantization outliers.

**When to prefer.** When Q, K, V weight salience profiles differ significantly at specific channels, or when one projection (e.g., q_proj) has much higher weight salience than the others.

### 4.2 Method 2 — Balanced (Geometric Mean Weight Salience)

Compute each projection's ideal balancing factor, then take the geometric mean:

\[
b_{p,j} = \left(\frac{\bar{s}(X_j)}{s(W_{p,j})}\right)^{\alpha} \quad \text{for } p \in \{q, k, v\}
\]

\[
b_{\text{qkv},j} = \left(b_{q,j} \cdot b_{k,j} \cdot b_{v,j}\right)^{1/3}
\]

Equivalently:

\[
s_{\text{geomean}}(W_j) = \left(s(W_{q,j}) \cdot s(W_{k,j}) \cdot s(W_{v,j})\right)^{1/3}
\]

\[
b_{\text{qkv},j} = \left(\frac{\bar{s}(X_j)}{s_{\text{geomean}}(W_j)}\right)^{\alpha}
\]

**Rationale.** Distributes the approximation error equally across all three projections rather than optimizing for the most extreme one. If the weight salience profiles are fairly similar across Q, K, V, this yields a tighter overall balance.

**When to prefer.** When Q, K, V weight salience profiles are relatively similar, so the geometric mean is a good central estimate.

### 4.3 Weight-Side Balancing (Per-Projection)

Regardless of which method is used for the shared input balancing vector, each projection's weight is balanced independently:

\[
\tilde{W}_p = W_p \cdot \text{diag}(b_{\text{qkv}}) \quad \text{for } p \in \{q, k, v\}
\]

In code (MLX, where `W` has shape `[d_out, d_in]`):
```python
W_p_balanced = W_p * b_qkv[None, :]  # broadcast across rows (d_out)
```

The balanced operation preserves the original output:
\[
\tilde{X} \cdot \tilde{W}_p^T = (X / b_{\text{qkv}}) \cdot (W_p \cdot b_{\text{qkv}})^T = X \cdot W_p^T
\]

The biases of q_proj and v_proj are **unchanged** — bias operates on the output dimension, which is not affected by input-channel balancing.

---

## 5. Re-parameterization (Absorption into adaLN)

The key efficiency of PTQ4DiT is that the \(B^{-1}\) scaling on activations can often be **absorbed** into preceding operations, eliminating runtime overhead. In SD3 Medium, the absorption targets are the `adaLN_modulation` MLP weights.

### 5.1 SD3's adaLN Formulation

DiffusionKit's `affine_transform` applies:

\[
Z = \text{LN}(X) \cdot (1 + \gamma) + \beta
\]

where \(\gamma\) (scale) and \(\beta\) (shift) are produced by the adaLN MLP. The `(1 + γ)` formulation differs from some DiT implementations that use `γ · LN(X) + β` without the additive 1. This requires a bias correction during absorption (derived below).

### 5.2 adaLN MLP Structure

Each TransformerBlock's adaLN MLP:

```
adaLN_modulation = nn.Sequential(SiLU(), nn.Linear(1536, 9216))
```

The output (9216 = 6 × 1536) is split into 6 chunks along the last dimension:

| Index range | Symbol | Role | Downstream layer |
|---|---|---|---|
| `[0:1536]` | β₁ | Pre-attention shift | q/k/v_proj |
| `[1536:3072]` | γ₁ | Pre-attention scale | q/k/v_proj |
| `[3072:4608]` | α₁ | Post-attention gate | Applied after o_proj |
| `[4608:6144]` | β₂ | Pre-FFN shift | fc1 |
| `[6144:7680]` | γ₂ | Pre-FFN scale | fc1 |
| `[7680:9216]` | α₂ | Post-FFN gate | Applied after fc2 |

Let \(W_{\text{mod}} \in \mathbb{R}^{9216 \times 1536}\) and \(b_{\text{mod}} \in \mathbb{R}^{9216}\) denote the adaLN Linear's weight and bias. Given timestep embedding \(e\) (after SiLU), each parameter chunk is:

\[
\beta_1 = e \cdot W_{\text{mod}}[0\!:\!1536]^T + b_{\text{mod}}[0\!:\!1536]
\]

(and similarly for the other chunks).

### 5.3 Absorption Derivation for q/k/v_proj

We want the adaLN output to produce the balanced activation \(\tilde{Z} = Z / b_{\text{qkv}}\):

\[
\tilde{Z}_j = \frac{Z_j}{b_j} = \frac{\text{LN}(X)_j \cdot (1 + \gamma_{1,j}) + \beta_{1,j}}{b_j}
\]

This must equal \(\text{LN}(X)_j \cdot (1 + \tilde{\gamma}_{1,j}) + \tilde{\beta}_{1,j}\) for the modified parameters \(\tilde{\gamma}_1, \tilde{\beta}_1\):

\[
(1 + \tilde{\gamma}_{1,j}) = \frac{1 + \gamma_{1,j}}{b_j}, \qquad \tilde{\beta}_{1,j} = \frac{\beta_{1,j}}{b_j}
\]

Solving for \(\tilde{\gamma}_{1,j}\):

\[
\tilde{\gamma}_{1,j} = \frac{1 + \gamma_{1,j}}{b_j} - 1
\]

Since \(\gamma_{1,j} = e \cdot W_{\text{mod}}[j']^T + b_{\text{mod}}[j']\) where \(j' = j + 1536\):

\[
\tilde{\gamma}_{1,j} = \frac{1 + e \cdot W_{\text{mod}}[j']^T + b_{\text{mod}}[j']}{b_j} - 1
= e \cdot \frac{W_{\text{mod}}[j']^T}{b_j} + \frac{1 + b_{\text{mod}}[j']}{b_j} - 1
\]

This gives the modified adaLN weight and bias:

**Shift (β₁) — rows `[0:1536]`:**

\[
\tilde{W}_{\text{mod}}[j, :] = \frac{W_{\text{mod}}[j, :]}{b_j}, \qquad \tilde{b}_{\text{mod}}[j] = \frac{b_{\text{mod}}[j]}{b_j}
\]

**Scale (γ₁) — rows `[1536:3072]`:**

\[
\tilde{W}_{\text{mod}}[j', :] = \frac{W_{\text{mod}}[j', :]}{b_j}, \qquad \tilde{b}_{\text{mod}}[j'] = \frac{1 + b_{\text{mod}}[j']}{b_j} - 1
\]

where \(j' = j + 1536\) and \(b_j\) is the \(j\)-th element of \(b_{\text{qkv}}\).

**Gate (α₁) — rows `[3072:4608]`:** Unchanged. The gate is applied **after** o_proj, not before it:
```python
residual = residual + attention_out * post_attn_scale
```

### 5.4 Absorption for fc1

Identical procedure using \(b_{\text{fc1}}\) (the balancing vector for fc1) on the MLP portion of the adaLN output:

**Shift (β₂) — rows `[4608:6144]`:**

\[
\tilde{W}_{\text{mod}}[j'', :] = \frac{W_{\text{mod}}[j'', :]}{b_{\text{fc1},j}}, \qquad \tilde{b}_{\text{mod}}[j''] = \frac{b_{\text{mod}}[j'']}{b_{\text{fc1},j}}
\]

**Scale (γ₂) — rows `[6144:7680]`:**

\[
\tilde{W}_{\text{mod}}[j''', :] = \frac{W_{\text{mod}}[j''', :]}{b_{\text{fc1},j}}, \qquad \tilde{b}_{\text{mod}}[j'''] = \frac{1 + b_{\text{mod}}[j''']}{b_{\text{fc1},j}} - 1
\]

where \(j'' = j + 4608\), \(j''' = j + 6144\).

**Gate (α₂) — rows `[7680:9216]`:** Unchanged.

### 5.5 Absorption for final_layer.linear

The FinalLayer's adaLN is:

```
adaLN_modulation = nn.Sequential(SiLU(), nn.Linear(1536, 3072))
```

Output is split into 2 chunks:
- `[0:1536]` → shift (β)
- `[1536:3072]` → scale (γ)

Absorption of \(b_{\text{final}}\) follows the same derivation:

- **Shift rows `[0:1536]`:** weight divided by \(b_j\), bias divided by \(b_j\).
- **Scale rows `[1536:3072]`:** weight divided by \(b_j\), bias → \((1 + b_{\text{old}}) / b_j - 1\).
- **Weight:** `final_layer.linear.weight` balanced as \(W_{\text{new}} = W \cdot \text{diag}(b_{\text{final}})\).

### 5.6 Combined adaLN Modification (Implementation)

For each block's adaLN (image or text), the absorption modifies the single `nn.Linear` layer inside `adaLN_modulation.layers[1]`. The procedure modifies 4 of the 6 output slices (2 shifts + 2 scales) while leaving 2 gates unchanged.

```python
def absorb_into_adaln(
    adaln_linear: nn.Linear,       # adaLN_modulation.layers[1], shape [9216, 1536]
    b_qkv: np.ndarray,             # [1536], balancing vector for q/k/v_proj
    b_fc1: np.ndarray,             # [d_fc1_in], balancing vector for fc1
    hidden_size: int = 1536,
):
    W = np.array(adaln_linear.weight)   # [9216, 1536]
    b = np.array(adaln_linear.bias)     # [9216]

    # --- q/k/v_proj absorption (indices 0:3072) ---
    # Shift β₁: rows [0:H]
    W[0:hidden_size] /= b_qkv[:, None]         # scale each row j by 1/b_qkv_j
    b[0:hidden_size] /= b_qkv

    # Scale γ₁: rows [H:2H]
    W[hidden_size:2*hidden_size] /= b_qkv[:, None]
    b[hidden_size:2*hidden_size] = (1 + b[hidden_size:2*hidden_size]) / b_qkv - 1

    # Gate α₁: rows [2H:3H] — unchanged

    # --- fc1 absorption (indices 4608:7680) ---
    # Shift β₂: rows [3H:4H]
    W[3*hidden_size:4*hidden_size] /= b_fc1[:, None]
    b[3*hidden_size:4*hidden_size] /= b_fc1

    # Scale γ₂: rows [4H:5H]
    W[4*hidden_size:5*hidden_size] /= b_fc1[:, None]
    b[4*hidden_size:5*hidden_size] = (
        (1 + b[4*hidden_size:5*hidden_size]) / b_fc1 - 1
    )

    # Gate α₂: rows [5H:6H] — unchanged

    adaln_linear.weight = mx.array(W, dtype=adaln_linear.weight.dtype)
    adaln_linear.bias = mx.array(b, dtype=adaln_linear.bias.dtype)
```

**Important:** `b_qkv` has shape `[1536]` (hidden size) regardless of which QKV method was chosen. `b_fc1` also has shape `[1536]` (fc1's input dimension equals hidden size). The adaLN output rows for β₂/γ₂ are indexed at `[3H:4H]` and `[4H:5H]` because the output layout is `[β₁, γ₁, α₁, β₂, γ₂, α₂]`.

---

## 6. Online Balancing for Post-Nonlinearity Layers

Two layer families cannot absorb \(B^{-1}\) into a preceding operation:

- **o_proj**: input is the per-modality SDPA output slice. The preceding operation (joint SDPA) is not modified.
- **fc2**: input is `GELU(fc1(x))`. GELU is nonlinear, so \(B^{-1}\) cannot pass through it.

For these layers, \(B^{-1}\) is applied **online** as an element-wise multiply at inference time. This happens per-modality (image o_proj uses \(b_{\text{o\_proj,img}}^{-1}\), text fc2 uses \(b_{\text{fc2,txt}}^{-1}\), etc.).

### 6.1 Data Flow with Online Balancing

**For o_proj** (referencing the architecture in `useful_doc/model.txt`):

```
Joint SDPA output → split into img/txt slices
                        │
                        ▼
              img slice (B, S_img, 1, 1536)
                        │
                  × b_o_img⁻¹   ← online element-wise multiply
                        │
                  fake_quant_A8  ← 8-bit activation quantization
                        │
                  o_proj (W4)    ← 4-bit weight matmul
                        │
                        ▼
```

**For fc2:**

```
fc1 output → GELU
               │
               ▼
        (B, N, 1, 6144)
               │
         × b_fc2⁻¹       ← online element-wise multiply
               │
         fake_quant_A8    ← 8-bit activation quantization
               │
         fc2 (W4)         ← 4-bit weight matmul
               │
               ▼
```

### 6.2 Cost Analysis

The online multiply is an element-wise operation:

| Layer | Online multiply cost | Matmul cost | Ratio |
|---|---|---|---|
| o_proj | O(B·N·1536) | O(B·N·1536²) | 1/1536 ≈ 0.07% |
| fc2 | O(B·N·6144) | O(B·N·6144·1536) | 1/1536 ≈ 0.07% |

Memory: one `float16` vector per layer.
- o_proj: 1536 × 2 bytes = 3 KB per layer, × 47 layers = 141 KB
- fc2: 6144 × 2 bytes = 12 KB per layer, × 47 layers = 564 KB
- **Total: ~705 KB** — negligible.

---

## 7. W4A8 Quantization Scheme

### 7.1 Weight Quantization (W4)

After CSB balancing, quantize the balanced weight matrix to 4-bit using MLX's built-in per-group quantization.

**Per-group affine quantization (MLX default):**

For a weight matrix \(\tilde{W} \in \mathbb{R}^{d_{out} \times d_{in}}\), partition the \(d_{in}\) dimension into groups of size \(G\) (default 64). For each group \(g\), let \(\alpha_g = \max(\tilde{W}_g)\) and \(\beta_g = \min(\tilde{W}_g)\):

\[
\text{scale}_g = \frac{\alpha_g - \beta_g}{2^{b} - 1} = \frac{\alpha_g - \beta_g}{15} \quad \text{(for 4-bit)}
\]

\[
\tilde{W}_{q,g} = \text{round}\left(\frac{\tilde{W}_g - \beta_g}{\text{scale}_g}\right), \quad \text{clamped to } [0, 15]
\]

MLX provides this via:

```python
quantized_w, scales, biases = mx.quantize(W_balanced, group_size=64, bits=4)
```

And the quantized `nn.QuantizedLinear` is created by:

```python
qlinear = nn.QuantizedLinear.from_linear(balanced_linear, group_size=64, bits=4)
```

which internally calls `mx.quantize` and stores the quantized representation. At inference, it uses `mx.quantized_matmul` for efficient 4-bit × 16-bit computation.

### 7.2 Activation Quantization (A8)

Dynamic per-tensor symmetric 8-bit fake quantization applied at runtime:

\[
\text{scale} = \frac{\max(|X|)}{127}, \quad X_q = \text{clamp}\left(\text{round}\left(\frac{X}{\text{scale}}\right),\; -128,\; 127\right), \quad \hat{X} = X_q \cdot \text{scale}
\]

```python
def fake_quantize_a8(x: mx.array) -> mx.array:
    x_abs_max = mx.max(mx.abs(x))
    scale = x_abs_max / 127.0
    scale = mx.maximum(scale, mx.array(1e-8))
    x_q = mx.clip(mx.round(x / scale), -128, 127)
    return x_q * scale
```

This is a "fake quantization" — the output is a float tensor that has been quantized and dequantized. It accurately simulates the precision loss of 8-bit inference. The quantization is **per-tensor** (single scale factor for the entire activation tensor), which is the standard approach for activation quantization.

**Why per-tensor and not per-channel for activations:** Per-channel activation quantization requires knowing the per-channel statistics at runtime, adding overhead. Per-tensor is simpler and, after CSB balancing, the cross-channel dynamic range is already equalized, making per-tensor sufficient.

### 7.3 W4A8Linear Class

A drop-in replacement for `nn.Linear` that combines W4 weight storage, A8 activation quantization, and optional online CSB balancing:

```python
class W4A8Linear(nn.Module):
    """W4A8 quantized linear with optional online CSB balancing."""

    def __init__(
        self,
        quantized_linear: nn.QuantizedLinear,
        b_inv: mx.array | None = None,
    ):
        super().__init__()
        self.qlinear = quantized_linear
        self.b_inv = b_inv  # None for absorbed layers; [d_in] for online layers

    def __call__(self, x: mx.array) -> mx.array:
        # 1. Online CSB balancing (for o_proj, fc2)
        if self.b_inv is not None:
            x = x * self.b_inv

        # 2. A8 fake quantization
        x = fake_quantize_a8(x)

        # 3. W4 quantized matmul
        return self.qlinear(x)
```

For **post-adaLN layers** (q/k/v_proj, fc1, final_layer.linear): `b_inv = None` because the balancing is already absorbed into the adaLN parameters.

For **post-nonlinearity layers** (o_proj, fc2): `b_inv = mx.array(1.0 / b_vector)` stored as a model parameter.

---

## 8. Per-Modality CSB Application

### 8.1 Layer Classification

Each of the 286 target layers belongs to one modality pathway:

| Side | Layers | Use activation stats from |
|---|---|---|
| `image` | `blocks.*.image.attn.{q,k,v,o}_proj`, `blocks.*.image.mlp.{fc1,fc2}` | Image-pathway diagnostic data |
| `text` | `blocks.*.text.attn.{q,k,v,o}_proj`, `blocks.*.text.mlp.{fc1,fc2}` | Text-pathway diagnostic data |
| `image` | `final_layer.linear` | Image-pathway (only image tokens reach final layer) |

The layer registry from Phase 1 (`src/phase1/registry.py`) already tags each layer with its `side` field.

### 8.2 Per-Block CSB Computation

For each block \(i \in [0, 23]\) and each modality side:

1. **Shared activation trajectory**: q/k/v_proj share the same input, so their activation trajectories are identical. Load the activation trajectory from any one of them (e.g., q_proj).

2. **SSC weights**: Although the activation trajectory is shared across q/k/v, the Spearman ρ trajectory **differs** for each projection because ρ is computed between the activation salience vector and the weight salience vector, and each projection has a different weight matrix. Two options: (a) use q_proj's ρ trajectory as the representative (simplest), or (b) compute ρ for each projection separately and average the resulting SSC weights to better account for the different weight salience profiles.

3. **QKV balancing vector**: Apply Method 1 (Max) or Method 2 (Geometric Mean) from Section 4.

4. **fc1 balancing vector**: Compute independently using fc1's own activation trajectory and weight salience.

5. **o_proj balancing vector**: Compute independently using o_proj's activation trajectory (SDPA output slice) and weight salience.

6. **fc2 balancing vector**: Compute independently using fc2's activation trajectory (post-GELU) and weight salience.

### 8.3 Absorption and Weight Modification Order

For each block, the modification order matters because the adaLN MLP is modified in-place:

1. Compute \(b_{\text{qkv}}\) and \(b_{\text{fc1}}\).
2. Absorb both into `adaLN_modulation.layers[1]` (the combined function in Section 5.6 handles both simultaneously).
3. Balance q/k/v_proj weights: `W_p *= b_qkv[None, :]` for each projection.
4. Balance fc1 weight: `W_fc1 *= b_fc1[None, :]`.
5. Balance o_proj weight: `W_o *= b_o[None, :]`.
6. Balance fc2 weight: `W_fc2 *= b_fc2[None, :]`.
7. Store `b_o_inv` and `b_fc2_inv` for online application.

After all balancing is applied, quantize all 286 layers to W4.

---

## 9. Special Layer Handling

### 9.1 final_layer.linear

Phase 1 findings: risk score 0.410 (rank 5/287), ρ = 0.673 (high — CSB less effective), CoV = 0.397 (highest — extreme temporal variation), early-late Jaccard = 0.103 (near-complete top-k turnover).

**Challenge.** High ρ means activation and weight salience are positively correlated, reducing CSB's ability to redistribute dynamic range. The regime shift (early σ vs late σ) means a single calibration point poorly represents the full trajectory.

**Approach.** Apply CSB+SSC as for other layers, but monitor the output MSE closely. The SSC weighting will naturally focus calibration on timesteps with lower ρ (where CSB is more effective). If quantization quality is insufficient at W4A8:
- **Fallback 1:** Keep at W8A8 (8-bit weights, 8-bit activations).
- **Fallback 2:** Keep at FP16.

Since `final_layer.linear` has shape `[64, 1536]` (only 64 output channels), its memory footprint is tiny (192 KB in FP16). Keeping it at higher precision has negligible memory cost.

### 9.2 Early Image Blocks (blocks 0–3)

Phase 1 findings: ρ ranges 0.4–0.7 (higher than mid/late blocks), indicating weaker complementarity. CSB effectiveness is reduced.

**Approach.** Apply CSB+SSC normally. The `alpha` parameter could be reduced for these blocks (e.g., α = 0.3 instead of 0.5) to avoid over-balancing when complementarity is weak. This is a tunable hyperparameter.

### 9.3 Late Text-Side Blocks (blocks 20–23)

Phase 1 findings: extreme activation salience in fc1/fc2 (300–414), moderate-to-high ρ (0.6–0.8 for blocks 21–22). These are the highest-risk quantization targets.

**Approach.** Apply CSB+SSC. The high activation salience means the balancing factors will be large, significantly compressing the activation dynamic range. Verify that the compressed range still has sufficient precision in 8-bit. If not:
- **Fallback:** Per-token activation quantization (separate scale per sequence position) instead of per-tensor.

### 9.4 Block 23 Text

This block has `skip_post_sdpa = True`:
- `o_proj` is `nn.Identity` (no linear layer to quantize).
- No FFN (`fc1`, `fc2` do not exist).
- Only `q_proj`, `k_proj`, `v_proj` are quantized on the text side of block 23.

The adaLN has `num_modulation_params = 2` (only β₁, γ₁ — no gate, no MLP params). So the adaLN Linear has shape `[3072, 1536]` instead of `[9216, 1536]`:
- `[0:1536]` → β₁ (shift for q/k/v_proj)
- `[1536:3072]` → γ₁ (scale for q/k/v_proj)

Absorption uses only `b_qkv` (no fc1 absorption needed).

### 9.5 context_embedder

Excluded from quantization as stated in Section 1. Kept at FP16. If future work includes it:
- Strong anti-correlation (ρ = −0.34) → CSB very effective.
- Zero temporal variation (CoV = 0.0) → no SSC needed, use simple average.
- No preceding adaLN → online balancing required (runs once per prompt, negligible).

---

## 10. Implementation Architecture

### 10.1 Module Structure

```
src/phase2/
├── __init__.py
├── config.py           # Hyperparameters and constants
├── calibrate.py        # Load Phase 1 data, compute SSC weights + weighted salience
├── balance.py          # Compute balancing vectors, absorb into adaLN, balance weights
├── quantize.py         # W4A8Linear class, fake_quantize_a8, model quantization
├── run_quantize.py     # CLI: load model → calibrate → balance → quantize → save
└── run_inference.py    # CLI: load quantized model → generate images
```

### 10.2 config.py

```python
PHASE2_CONFIG = {
    "alpha": 0.5,                # CSB exponent
    "b_min": 1e-5,               # Balancing factor floor
    "b_max": 1e5,                # Balancing factor ceiling
    "w_eps": 1e-12,              # Weight salience floor (avoid div-by-zero)
    "group_size": 64,            # W4 quantization group size
    "bits": 4,                   # Weight quantization bits
    "a_bits": 8,                 # Activation quantization bits
    "qkv_method": "max",         # "max" or "geomean" (Section 4)
    "final_layer_bits": 4,       # Can override to 8 or 16 for final layer
    "exclude_layers": [          # Layers to skip quantization
        "context_embedder",
    ],
}
```

### 10.3 calibrate.py

**Key functions:**

```python
def load_phase1_data(
    layer_name: str,
    diagnostics_dir: Path,
) -> dict:
    """Load activation trajectory and weight salience for one layer.

    Returns:
        {"act_trajectory": ndarray [T, d_in],
         "wt_salience": ndarray [d_in]}
    """

def compute_balancing_data(
    layer_name: str,
    diagnostics_dir: Path,
    alpha: float = 0.5,
    b_min: float = 1e-5,
    b_max: float = 1e5,
) -> np.ndarray:
    """Compute the SSC-weighted balancing vector for a single layer.

    Steps:
      1. Load activation trajectory and weight salience.
      2. Compute Spearman ρ trajectory → SSC weights η_t.
      3. Compute weighted activation salience s̄(X_j).
      4. Compute b_j = (s̄(X_j) / s(W_j))^α, clamped.

    Returns: b_vector of shape [d_in].
    """

def compute_qkv_balancing(
    block_idx: int,
    side: str,
    diagnostics_dir: Path,
    method: str = "max",
    alpha: float = 0.5,
) -> np.ndarray:
    """Compute the shared balancing vector for Q/K/V projections.

    Loads weight salience for q_proj, k_proj, v_proj and merges
    using the specified method ("max" or "geomean").
    Uses the shared activation trajectory (from q_proj).

    Returns: b_qkv of shape [hidden_size].
    """
```

### 10.4 balance.py

**Key functions:**

```python
def absorb_into_adaln(
    adaln_linear: nn.Linear,
    b_qkv: np.ndarray,
    b_fc1: np.ndarray | None,
    hidden_size: int = 1536,
) -> None:
    """Modify adaLN_modulation Linear in-place to absorb B⁻¹ for
    q/k/v_proj (and optionally fc1).

    For block 23 text (num_modulation_params=2), pass b_fc1=None.
    """

def absorb_into_final_adaln(
    adaln_linear: nn.Linear,
    b_final: np.ndarray,
    hidden_size: int = 1536,
) -> None:
    """Modify final_layer.adaLN_modulation Linear in-place."""

def balance_weight(
    linear: nn.Linear,
    b_vector: np.ndarray,
) -> None:
    """Modify linear.weight in-place: W_new = W * b[None, :].

    The bias (if present) is unchanged.
    """

def apply_csb_to_model(
    mmdit,
    registry: list[dict],
    diagnostics_dir: Path,
    config: dict,
) -> dict[str, np.ndarray]:
    """Apply CSB to the entire MMDiT model.

    For each block and modality:
      1. Compute b_qkv, b_fc1, b_o, b_fc2.
      2. Absorb b_qkv and b_fc1 into adaLN.
      3. Balance q/k/v/o_proj and fc1/fc2 weights.
      4. Collect b_inv vectors for o_proj and fc2.

    Returns: dict mapping layer_name → b_inv (only for online-balanced layers).
    """
```

### 10.5 quantize.py

**Key functions and classes:**

```python
def fake_quantize_a8(x: mx.array) -> mx.array:
    """Dynamic per-tensor symmetric 8-bit fake quantization."""

class W4A8Linear(nn.Module):
    """W4A8 quantized linear with optional online CSB balancing.
    See Section 7.3 for the full specification.
    """

def quantize_model(
    mmdit,
    registry: list[dict],
    b_inv_map: dict[str, np.ndarray],
    config: dict,
) -> None:
    """Replace all target nn.Linear layers with W4A8Linear modules.

    For each layer in the registry:
      1. Convert nn.Linear → nn.QuantizedLinear (W4).
      2. Wrap in W4A8Linear with the appropriate b_inv (or None).
      3. Replace the module in the model tree.
    """

def replace_module_in_model(
    model,
    layer_name: str,
    new_module: nn.Module,
) -> None:
    """Navigate the model tree by dotted name and replace the leaf module.

    Example: "blocks.5.image.attn.q_proj" →
      model.multimodal_transformer_blocks[5].image_transformer_block.attn.q_proj
    """
```

### 10.6 run_quantize.py

CLI entry point:

```python
def main():
    """
    Usage: python -m src.phase2.run_quantize [--model-path PATH] [--output-dir DIR]
                                             [--qkv-method max|geomean]
                                             [--alpha 0.5] [--group-size 64]

    Steps:
      1. Load DiffusionPipeline (FP16 model).
      2. Build layer registry (from phase1.registry).
      3. Load Phase 1 diagnostics.
      4. Compute balancing vectors (calibrate.py).
      5. Apply CSB: absorb into adaLN + balance weights (balance.py).
      6. Quantize all target layers to W4A8 (quantize.py).
      7. Save quantized model weights.
    """
```

### 10.7 run_inference.py

CLI entry point for generating images with the quantized model:

```python
def main():
    """
    Usage: python -m src.phase2.run_inference --quantized-dir DIR
                                              --prompt "..." [--seed 42]
                                              [--num-steps 28] [--cfg-weight 5.0]
                                              [--output image.png]

    Steps:
      1. Load DiffusionPipeline.
      2. Load quantized weights (replacing nn.Linear with W4A8Linear).
      3. Run standard inference pipeline (encode_text → denoise → decode).
      4. Save output image.
    """
```

---

## 11. End-to-End Pipeline

The complete quantization process from Phase 1 data to quantized inference:

```
Step 1: Load FP16 model
         │
Step 2: Build layer registry (287 layers, 286 targets)
         │
Step 3: For each target layer, load Phase 1 data:
         ├─ activation trajectory [T, d_in]
         └─ weight salience [d_in]
         │
Step 4: For each target layer, compute:
         ├─ Spearman ρ trajectory [T]
         ├─ SSC weights η_t [T]
         ├─ Weighted activation salience s̄(X) [d_in]
         └─ Balancing vector b [d_in]
         │
Step 5: For shared-input layers (QKV), merge balancing vectors
         using Method 1 (max) or Method 2 (geomean)
         │
Step 6: Apply CSB re-parameterization:
         ├─ Absorb b_qkv, b_fc1 into each block's adaLN weights
         ├─ Absorb b_final into final_layer adaLN weights
         ├─ Balance all target layer weights: W *= b[None, :]
         └─ Store b_inv for o_proj and fc2 (online layers)
         │
Step 7: Quantize all target layers:
         ├─ Convert balanced nn.Linear → nn.QuantizedLinear (W4)
         └─ Wrap in W4A8Linear (adds A8 + optional online B⁻¹)
         │
Step 8: Replace modules in model tree
         │
Step 9: Save quantized checkpoint
         │
Step 10: Inference — standard DiffusionKit pipeline with quantized model
```

---

## 12. Practical Notes

### 12.1 Memory Budget

| Component | FP16 size | W4 size | Savings |
|---|---|---|---|
| q/k/v_proj (144 layers) | 144 × 1536² × 2B = 679 MB | 144 × 1536² × 0.5B = 170 MB | 75% |
| o_proj (47 layers) | 47 × 1536² × 2B = 222 MB | 47 × 1536² × 0.5B = 55 MB | 75% |
| fc1 (47 layers) | 47 × 6144 × 1536 × 2B = 887 MB | 222 MB | 75% |
| fc2 (47 layers) | 47 × 1536 × 6144 × 2B = 887 MB | 222 MB | 75% |
| final_layer.linear | 64 × 1536 × 2B = 192 KB | 48 KB | 75% |
| Online b_inv vectors | — | ~705 KB | — |
| **Total quantized layers** | **~2.67 GB** | **~0.67 GB** | **~2 GB saved** |

Non-quantized components (adaLN, embedders, norms, SDPA, text encoders, VAE) remain at FP16 and are unchanged.

### 12.2 Hyperparameter Sensitivity

- **α (CSB exponent):** α = 0.5 is the SmoothQuant default and a good starting point. Lower α (0.3) applies less balancing, preserving the original weight distribution at the cost of less activation compression. Higher α (0.7) compresses activations more aggressively. Tune based on output quality.
- **group_size (W4):** Smaller groups (32) provide finer-grained quantization but increase scale/bias overhead. Larger groups (128) are more memory-efficient but may lose precision on layers with heterogeneous weight distributions. Default 64 is a standard balance.
- **QKV method:** Test both Method 1 (max) and Method 2 (geomean) on a small validation set and compare output MSE.

### 12.3 Validation Strategy

Before full evaluation, verify correctness at each stage:

1. **After CSB absorption:** Run FP16 inference on the balanced (but unquantized) model. The output should be **numerically very close** to the original FP16 model (CSB is mathematically exact — the balancing cancels out). Small differences (max absolute error on the order of \(10^{-3}\) for FP16) are expected due to floating-point rounding during weight modification. Larger discrepancies indicate a bug in absorption or weight modification.
2. **After W4 quantization (no A8):** Run inference with 4-bit weights but FP16 activations. Compare output to FP16 baseline.
3. **After W4A8:** Run full quantized inference. Compare to FP16 baseline.

For each comparison, compute:
- Per-pixel MSE on the denoised latent.
- Visual inspection on a fixed set of prompts and seeds.

### 12.4 Saving and Loading

The quantized model can be saved as a standard MLX weight file (`.safetensors` or `.npz`). The `W4A8Linear` modules store:
- Quantized weights (`quantized_weight`, `scales`, `biases`) — from `nn.QuantizedLinear`.
- `b_inv` vectors (for online-balanced layers).
- `group_size` and `bits` metadata.

A loading function reconstructs the `W4A8Linear` modules from the saved parameters and replaces the corresponding `nn.Linear` modules in the model tree.

---

## 13. References

- **Phase 1 diagnostics data:** `diagnostics/` (activation trajectories, weight salience, summary table)
- **Phase 1 findings:** `src/phase1_findings.md` (salience patterns, complementarity, modality asymmetry, risk ranking)
- **Phase 1 implementation:** `src/phase1/` (reusable: `analyze.py` for `compute_spearman_trajectory`, `compute_ssc_weights`; `registry.py` for `build_layer_registry`)
- **DiffusionKit architecture:** `DiffusionKit/python/src/diffusionkit/mlx/mmdit.py` (`TransformerBlock.pre_sdpa`, `affine_transform`, `adaLN_modulation`, `FinalLayer`)
- **Architecture diagram:** `useful_doc/model.txt`
- **PTQ4DiT paper:** Eq. 4 (salience), Eq. 7 (balancing matrix), Eq. 11 (SSC weights), Eq. 13 (adaLN), Eq. 20 (re-parameterization)
