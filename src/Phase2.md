# Phase 2: W4A8 Quantization via CSB + SSC for SD3 Medium

## 1. Scope and Goals

**Objective.** Apply PTQ4DiT's Channel-wise Salience Balancing (CSB) and Spearman's ρ-guided Salience Calibration (SSC) to the MMDiT backbone of Stable Diffusion 3 Medium, producing a W4A8-quantized model (4-bit weights, 8-bit activations).

**Quantization targets.** 286 `nn.Linear` layers inside the MMDiT denoiser:

| Family | Shape (out, in) | Count | Notes |
|---|---|---|---|
| `q_proj` | (1536, 1536) | 48 | 24 blocks × img + txt. Has bias. |
| `k_proj` | (1536, 1536) | 48 | No bias. |
| `v_proj` | (1536, 1536) | 48 | Has bias. |
| `o_proj` | (1536, 1536) | 47 | Block 23 txt is Identity (skipped). |
| `fc1` | (6144, 1536) | 47 | Block 23 txt has no FFN. |
| `fc2` | (1536, 6144) | 47 | Block 23 txt has no FFN. |
| `final_layer.linear` | (64, 1536) | 1 | Patch unprojection. |
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
- Configuration: `diagnostics/config.json` — sigma schedule, seed-prompt pairs, layer names used during collection.

**Output.** A quantized model checkpoint with:
- Balanced and 4-bit-quantized weights for all 286 target layers.
- Absorbed balancing parameters in adaLN modulation weights.
- Per-layer `b_inv` vectors for layers requiring online balancing.

---

## 2. Theoretical Background

### 2.1 Salience (PTQ4DiT Eq. 4)

For a linear layer with input activation $X \in \mathbb{R}^{B \times N \times d_{in}}$ and weight $W \in \mathbb{R}^{d_{out} \times d_{in}}$:

$$
s(X_j^{(\sigma)}) = \max_{b,n} |X_{b,n,j}^{(\sigma)}|, \quad s(W_j) = \max_{i} |W_{i,j}|
$$

- Activation salience is **time-dependent**: it varies across the sigma (denoising) trajectory.
- Weight salience is **time-independent**: weights are fixed.

### 2.2 SSC: Time-Aware Calibration Weighting (PTQ4DiT Eq. 11)

At each sigma step $\sigma_t$, compute the Spearman rank correlation $\rho_t$ between the activation salience vector $s(X^{(\sigma_t)})$ and the weight salience vector $s(W)$. Then:

$$
\eta_t = \frac{\exp(-\rho_t)}{\sum_{\tau} \exp(-\rho_\tau)}
$$

Timesteps with **lower** $\rho$ (stronger complementarity) receive **higher** weight. This is a softmax over $-\rho$.

The SSC-weighted representative activation salience for channel $j$:

$$
\bar{s}(X_j) = \sum_t \eta_t \cdot s(X_j^{(\sigma_t)})
$$

This replaces the naive time-average with a complementarity-aware average, giving more influence to timesteps where CSB can be most effective.

### 2.3 CSB: Balancing Vector (SmoothQuant-style, PTQ4DiT Eq. 7)

For each target linear layer, compute a per-channel balancing factor:

$$
b_j = \left(\frac{\bar{s}(X_j)}{s(W_j)}\right)^{\alpha}, \quad \alpha \in (0, 1)
$$

Default $\alpha = 0.5$ (geometric mean, from SmoothQuant). The balancing matrix is $B = \text{diag}(b_1, \dots, b_{d_{in}})$.

The balanced linear operation:

$$
Y = X W^T = \underbrace{(X \cdot B^{-1})}_{\text{balanced activation}} \cdot \underbrace{(W \cdot B)^T}_{\text{balanced weight}}
$$

In element-wise terms:
- Balanced activation: $\tilde{X}_j = X_j / b_j$ — reduces dynamic range of high-salience activation channels.
- Balanced weight: $\tilde{W}_{i,j} = W_{i,j} \cdot b_j$ — increases weight magnitude for those channels (which had low weight salience due to complementarity).

After balancing, both $\tilde{X}$ and $\tilde{W}$ have more uniform per-channel dynamic ranges, making uniform quantization significantly more effective.

### 2.4 Numerical Stability

Clamp balancing factors to avoid extreme values:

$$
b_j = \text{clamp}\left(b_j, \; b_{\min}, \; b_{\max}\right)
$$

Defaults: $b_{\min} = 10^{-2}$, $b_{\max} = 10^{2}$. These bounds ensure $b^{-1}$ stays within float16 range (~65,504) even after online scaling. Additionally, if $s(W_j) < \epsilon$ (dead weight channel), set $b_j = 1$ (no balancing).

---

## 3. Calibration Pipeline

Phase 2 calibration reuses Phase 1 diagnostic data. No new data collection is required.

### 3.1 Load Phase 1 Data

For each target layer $l$:

1. **Activation trajectory**: Load `diagnostics/activation_stats/{l}.npz` → `act_channel_max` of shape `[T, d_in]` where `T = num_steps` (30 by default). Each entry is the per-channel max activation magnitude at that sigma step, aggregated across all calibration prompts and seeds.
2. **Weight salience**: Load `diagnostics/weight_stats.npz` → `w_channel_max` of shape `[d_in]`. The per-channel max weight magnitude.
3. **Sigma schedule**: Load `diagnostics/config.json` → `sigma_values` for the sigma-to-step mapping (needed only for reference; SSC operates on the ordered step index).

### 3.2 Compute SSC Weights

For each layer $l$:

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

This produces $\bar{s}(X_j)$ for every channel.

### 3.4 Compute Balancing Vector

```python
b = (weighted_act_salience / (wt_salience + eps)) ** alpha
b = np.clip(b, b_min, b_max)
```

Where `eps = 1e-12`, `alpha = 0.5`, `b_min = 1e-2`, `b_max = 1e2`.

---

## 4. CSB for Shared-Input Layers (QKV Projections)

Q, K, and V projections within the same TransformerBlock share the same input tensor (`modulated_pre_attention`), which is the output of `affine_transform(x, shift=β₁, scale=γ₁, norm_module=norm1)`. Since the adaLN absorption modifies the shared input, only **one** balancing vector $b_{\text{qkv}}$ can be applied to it. However, each projection has its own weight matrix with potentially different weight salience profiles.

Two methods are available for computing the shared balancing vector.

### 4.1 Method 1 — Conservative (Max Weight Salience)

Merge the weight salience across Q, K, V by taking the per-channel maximum:

$$
s_{\text{merged}}(W_j) = \max\left(s(W_{q,j}),\; s(W_{k,j}),\; s(W_{v,j})\right)
$$

Then compute the shared balancing vector:

$$
b_{\text{qkv},j} = \left(\frac{\bar{s}(X_j)}{s_{\text{merged}}(W_j)}\right)^{\alpha}
$$

**Rationale.** For each channel $j$, the projection with the highest weight salience gets the optimal balance. The other two projections receive a slightly conservative balance (their weight channels are scaled up slightly less than their individual optima). This avoids any under-balancing that could cause weight quantization outliers.

**When to prefer.** When Q, K, V weight salience profiles differ significantly at specific channels, or when one projection (e.g., `q_proj`) has much higher weight salience than the others.

### 4.2 Method 2 — Balanced (Geometric Mean Weight Salience)

Compute each projection's ideal balancing factor, then take the geometric mean:

$$
b_{p,j} = \left(\frac{\bar{s}(X_j)}{s(W_{p,j})}\right)^{\alpha} \quad \text{for } p \in \{q, k, v\}
$$

$$
b_{\text{qkv},j} = \left(b_{q,j} \cdot b_{k,j} \cdot b_{v,j}\right)^{1/3}
$$

Equivalently:

$$
s_{\text{geomean}}(W_j) = \left(s(W_{q,j}) \cdot s(W_{k,j}) \cdot s(W_{v,j})\right)^{1/3}
$$

$$
b_{\text{qkv},j} = \left(\frac{\bar{s}(X_j)}{s_{\text{geomean}}(W_j)}\right)^{\alpha}
$$

**Rationale.** Distributes the approximation error equally across all three projections rather than optimizing for the most extreme one. If the weight salience profiles are fairly similar across Q, K, V, this yields a tighter overall balance.

**When to prefer.** When Q, K, V weight salience profiles are relatively similar, so the geometric mean is a good central estimate.

### 4.3 Weight-Side Balancing (Per-Projection)

Regardless of which method is used for the shared input balancing vector, each projection's weight is balanced independently:

$$
\tilde{W}_p = W_p \cdot \text{diag}(b_{\text{qkv}}) \quad \text{for } p \in \{q, k, v\}
$$

In code (MLX, where `W` has shape `[d_out, d_in]`):
```python
W_p_balanced = W_p * b_qkv[None, :]  # broadcast across rows (d_out)
```

The balanced operation preserves the original output:
$$
\tilde{X} \cdot \tilde{W}_p^T = (X / b_{\text{qkv}}) \cdot (W_p \cdot b_{\text{qkv}})^T = X \cdot W_p^T
$$

The biases of `q_proj` and `v_proj` are **unchanged** — bias operates on the output dimension, which is not affected by input-channel balancing.

---

## 5. Re-parameterization (Absorption into adaLN)

The key efficiency of PTQ4DiT is that the $B^{-1}$ scaling on activations can often be **absorbed** into preceding operations, eliminating runtime overhead. In SD3 Medium, the absorption targets are the `adaLN_modulation` MLP weights.

### 5.1 SD3's adaLN Formulation

DiffusionKit's `affine_transform` applies:

$$
Z = \text{LN}(X) \cdot (1 + \gamma) + \beta
$$

where $\gamma$ (scale) and $\beta$ (shift) are produced by the adaLN MLP. The `(1 + γ)` formulation differs from some DiT implementations that use `γ · LN(X) + β` without the additive 1. This requires a bias correction during absorption (derived below).

### 5.2 adaLN MLP Structure

Each TransformerBlock's adaLN MLP:

```
adaLN_modulation = nn.Sequential(SiLU(), nn.Linear(1536, 9216))
```

The output (9216 = 6 × 1536) is split into 6 chunks along the last dimension:

| Index range | Symbol | Role | Downstream layer |
|---|---|---|---|
| `[0:1536]` | β₁ | Pre-attention shift | `q/k/v_proj` |
| `[1536:3072]` | γ₁ | Pre-attention scale | `q/k/v_proj` |
| `[3072:4608]` | α₁ | Post-attention gate | Applied after `o_proj` |
| `[4608:6144]` | β₂ | Pre-FFN shift | fc1 |
| `[6144:7680]` | γ₂ | Pre-FFN scale | fc1 |
| `[7680:9216]` | α₂ | Post-FFN gate | Applied after fc2 |

Let $W_{\text{mod}} \in \mathbb{R}^{9216 \times 1536}$ and $b_{\text{mod}} \in \mathbb{R}^{9216}$ denote the adaLN Linear's weight and bias. Given timestep embedding $e$ (after SiLU), each parameter chunk is:

$$
\beta_1 = e \cdot W_{\text{mod}}[0\!:\!1536]^T + b_{\text{mod}}[0\!:\!1536]
$$

(and similarly for the other chunks).

### 5.3 Absorption Derivation for q/k/v\_proj

We want the adaLN output to produce the balanced activation $\tilde{Z} = Z / b_{\text{qkv}}$:

$$
\tilde{Z}_j = \frac{Z_j}{b_j} = \frac{\text{LN}(X)_j \cdot (1 + \gamma_{1,j}) + \beta_{1,j}}{b_j}
$$

This must equal $\text{LN}(X)_j \cdot (1 + \tilde{\gamma}_{1,j}) + \tilde{\beta}_{1,j}$ for the modified parameters $\tilde{\gamma}_1, \tilde{\beta}_1$:

$$
(1 + \tilde{\gamma}_{1,j}) = \frac{1 + \gamma_{1,j}}{b_j}, \qquad \tilde{\beta}_{1,j} = \frac{\beta_{1,j}}{b_j}
$$

Solving for $\tilde{\gamma}_{1,j}$:

$$
\tilde{\gamma}_{1,j} = \frac{1 + \gamma_{1,j}}{b_j} - 1
$$

Since $\gamma_{1,j} = e \cdot W_{\text{mod}}[j']^T + b_{\text{mod}}[j']$ where $j' = j + 1536$:

$$
\tilde{\gamma}_{1,j} = \frac{1 + e \cdot W_{\text{mod}}[j']^T + b_{\text{mod}}[j']}{b_j} - 1
= e \cdot \frac{W_{\text{mod}}[j']^T}{b_j} + \frac{1 + b_{\text{mod}}[j']}{b_j} - 1
$$

This gives the modified adaLN weight and bias:

**Shift (β₁) — rows `[0:1536]`:**

$$
\tilde{W}_{\text{mod}}[j, :] = \frac{W_{\text{mod}}[j, :]}{b_j}, \qquad \tilde{b}_{\text{mod}}[j] = \frac{b_{\text{mod}}[j]}{b_j}
$$

**Scale (γ₁) — rows `[1536:3072]`:**

$$
\tilde{W}_{\text{mod}}[j', :] = \frac{W_{\text{mod}}[j', :]}{b_j}, \qquad \tilde{b}_{\text{mod}}[j'] = \frac{1 + b_{\text{mod}}[j']}{b_j} - 1
$$

where $j' = j + 1536$ and $b_j$ is the $j$-th element of $b_{\text{qkv}}$.

**Gate (α₁) — rows `[3072:4608]`:** Unchanged. The gate is applied **after** o_proj, not before it:
```python
residual = residual + attention_out * post_attn_scale
```

### 5.4 Absorption for fc1

Identical procedure using $b_{\text{fc1}}$ (the balancing vector for fc1) on the MLP portion of the adaLN output:

**Shift (β₂) — rows `[4608:6144]`:**

$$
\tilde{W}_{\text{mod}}[j'', :] = \frac{W_{\text{mod}}[j'', :]}{b_{\text{fc1},j}}, \qquad \tilde{b}_{\text{mod}}[j''] = \frac{b_{\text{mod}}[j'']}{b_{\text{fc1},j}}
$$

**Scale (γ₂) — rows `[6144:7680]`:**

$$
\tilde{W}_{\text{mod}}[j''', :] = \frac{W_{\text{mod}}[j''', :]}{b_{\text{fc1},j}}, \qquad \tilde{b}_{\text{mod}}[j'''] = \frac{1 + b_{\text{mod}}[j''']}{b_{\text{fc1},j}} - 1
$$

where $j'' = j + 4608$, $j''' = j + 6144$.

**Gate (α₂) — rows `[7680:9216]`:** Unchanged.

### 5.5 Absorption for final\_layer.linear

The FinalLayer's adaLN is:

```
adaLN_modulation = nn.Sequential(SiLU(), nn.Linear(1536, 3072))
```

Output is split into 2 chunks:
- `[0:1536]` → shift (β)
- `[1536:3072]` → scale (γ)

Absorption of $b_{\text{final}}$ follows the same derivation:

- **Shift rows `[0:1536]`:** weight divided by $b_j$, bias divided by $b_j$.
- **Scale rows `[1536:3072]`:** weight divided by $b_j$, bias → $(1 + b_{\text{old}}) / b_j - 1$.
- **Weight:** `final_layer.linear.weight` balanced as $W_{\text{new}} = W \cdot \text{diag}(b_{\text{final}})$.

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

Two layer families cannot absorb $B^{-1}$ into a preceding operation:

- **`o_proj`**: input is the per-modality SDPA output slice. The preceding operation (joint SDPA) is not modified.
- **`fc2`**: input is `GELU(fc1(x))`. GELU is nonlinear, so $B^{-1}$ cannot pass through it.

For these layers, $B^{-1}$ is applied **online** as an element-wise multiply at inference time. This happens per-modality (image `o_proj` uses $b_{\text{o\_proj,img}}^{-1}$, text `fc2` uses $b_{\text{fc2,txt}}^{-1}$, etc.).

### 6.1 Data Flow with Online Balancing

**For `o_proj`** (referencing the architecture in `useful_doc/model.txt`):

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
| `o_proj` | O(B·N·1536) | O(B·N·1536²) | 1/1536 ≈ 0.07% |
| `fc2` | O(B·N·6144) | O(B·N·6144·1536) | 1/1536 ≈ 0.07% |

Memory: one `float32` vector per layer (stored as float32 to avoid float16 overflow on `1/b` values near 100).
- `o_proj`: 1536 × 4 bytes = 6 KB per layer, × 47 layers = 282 KB
- `fc2`: 6144 × 4 bytes = 24 KB per layer, × 47 layers = 1.1 MB
- **Total: ~1.4 MB** — negligible.

---

## 7. W4A8 Quantization Scheme

### 7.1 Weight Quantization (W4)

After CSB balancing, quantize the balanced weight matrix to 4-bit using MLX's built-in per-group quantization.

**Per-group affine quantization (MLX default):**

For a weight matrix $\tilde{W} \in \mathbb{R}^{d_{out} \times d_{in}}$, partition the $d_{in}$ dimension into groups of size $G$ (default 64). For each group $g$, let $\alpha_g = \max(\tilde{W}_g)$ and $\beta_g = \min(\tilde{W}_g)$:

$$
\text{scale}_g = \frac{\alpha_g - \beta_g}{2^{b} - 1} = \frac{\alpha_g - \beta_g}{15} \quad \text{(for 4-bit)}
$$

$$
\tilde{W}_{q,g} = \text{round}\left(\frac{\tilde{W}_g - \beta_g}{\text{scale}_g}\right), \quad \text{clamped to } [0, 15]
$$

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

$$
\text{scale} = \frac{\max(|X|)}{127}, \quad X_q = \text{clamp}\left(\text{round}\left(\frac{X}{\text{scale}}\right),\; -128,\; 127\right), \quad \hat{X} = X_q \cdot \text{scale}
$$

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
        qlinear: nn.QuantizedLinear,
        b_inv: mx.array | None = None,
    ):
        super().__init__()
        self.qlinear = qlinear
        if b_inv is not None:
            self.b_inv = b_inv  # [d_in], stored as float32

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype

        # 1. Online CSB balancing (for o_proj, fc2)
        if hasattr(self, "b_inv"):
            x = (x * self.b_inv).astype(orig_dtype)

        # 2. A8 fake quantization
        x = fake_quantize_a8(x)

        # 3. W4 quantized matmul
        return self.qlinear(x)
```

**Precision note:** `b_inv` is stored as `mx.float32` because values near `1/b_min = 100` are within float16 range (~65504) but float32 provides a safety margin. The `(x * self.b_inv)` multiplication temporarily promotes to float32, and `.astype(orig_dtype)` casts the result back to fp16 to avoid dtype drift through the network.

For **post-adaLN layers** (`q/k/v_proj`, `fc1`, `final_layer.linear`): `b_inv` is not set (the attribute does not exist) because the balancing is already absorbed into the adaLN parameters.

For **post-nonlinearity layers** (`o_proj`, `fc2`): `b_inv = mx.array(1.0 / b_vector, dtype=mx.float32)` stored as a model parameter.

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

For each block $i \in [0, 23]$ and each modality side:

1. **Shared activation trajectory**: `q/k/v_proj` share the same input, so their activation trajectories are identical. Load the activation trajectory from any one of them (e.g., `q_proj`).

2. **SSC weights**: Although the activation trajectory is shared across q/k/v, the Spearman ρ trajectory **differs** for each projection because ρ is computed between the activation salience vector and the weight salience vector, and each projection has a different weight matrix. Two options: (a) use `q_proj`'s ρ trajectory as the representative (simplest), or (b) compute ρ for each projection separately and average the resulting SSC weights to better account for the different weight salience profiles.

3. **QKV balancing vector**: Apply Method 1 (Max) or Method 2 (Geometric Mean) from Section 4.

4. **fc1 balancing vector**: Compute independently using fc1's own activation trajectory and weight salience.

5. **`o_proj` balancing vector**: Compute independently using `o_proj`'s activation trajectory (SDPA output slice) and weight salience.

6. **`fc2` balancing vector**: Compute independently using `fc2`'s activation trajectory (post-GELU) and weight salience.

### 8.3 Absorption and Weight Modification Order

For each block, the modification order matters because the adaLN MLP is modified in-place:

1. Compute $b_{\text{qkv}}$ and $b_{\text{fc1}}$.
2. Absorb both into `adaLN_modulation.layers[1]` (the combined function in Section 5.6 handles both simultaneously).
3. Balance `q/k/v_proj` weights: `W_p *= b_qkv[None, :]` for each projection.
4. Balance `fc1` weight: `W_fc1 *= b_fc1[None, :]`.
5. Balance `o_proj` weight: `W_o *= b_o[None, :]`.
6. Balance `fc2` weight: `W_fc2 *= b_fc2[None, :]`.
7. Store `b_o_inv` and `b_fc2_inv` for online application.

After all balancing is applied, quantize all 286 layers to W4.

---

## 9. Special Layer Handling

### 9.1 final\_layer.linear

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

### 9.5 context\_embedder

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
├── config.py             # Hyperparameters, constants, pipeline kwargs
├── calibrate.py          # Load Phase 1 data, compute SSC weights + balancing vectors
├── balance.py            # Absorb B⁻¹ into adaLN, balance weights, collect b_inv
├── quantize.py           # W4A8Linear, fake_quantize_a8, quantize/save/load model
├── diagnose.py           # Post-quantization diagnostics (W4A8 vs FP16 comparison)
├── visualize_quant.py    # Post-quantization diagnostic plots
├── run_quantize.py       # CLI: calibrate → balance → quantize → save (Phase 1 data required)
├── run_e2e.py            # CLI: collection → calibrate → balance → quantize → save (all-in-one)
├── run_inference.py      # CLI: load FP16 or W4A8 model → generate images
└── run_diagnose.py       # CLI: collect W4A8 stats → compare vs FP16 → generate plots
```

### 10.2 config.py

```python
PHASE2_CONFIG = {
    "alpha": 0.5,                # CSB exponent (SmoothQuant default)
    "b_min": 1e-2,               # Balancing factor floor (float16-safe)
    "b_max": 1e2,                # Balancing factor ceiling (float16-safe)
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

# Pipeline construction kwargs (mirrors Phase 1 DIAG_CONFIG)
PIPELINE_KWARGS = {
    "w16": True,                 # fp16 model weights
    "shift": 1.0,                # Rectified flow sigma shift
    "use_t5": True,              # Enable T5-XXL text encoder
    "low_memory_mode": False,    # Keep full model in memory
}
```

### 10.3 calibrate.py

```python
def load_phase1_data(layer_name, diagnostics_dir) -> dict:
    """Returns {"act_trajectory": [T, d_in], "wt_salience": [d_in]}."""

def compute_balancing_vector(act_trajectory, wt_salience, alpha, b_min, b_max, w_eps) -> ndarray:
    """Single-layer SSC+CSB: rho → ssc_weights → weighted_act → b_j = (s̄/sW)^α."""

def compute_qkv_balancing(block_idx, side, diagnostics_dir, method, ...) -> ndarray:
    """Shared QKV balancing: merge weight salience via 'max' or 'geomean'."""

def calibrate_all_layers(registry, diagnostics_dir, config) -> dict:
    """Calibrate all target layers.  Returns:
    {"balancing_vectors": dict[name → ndarray], "b_inv_layers": list[str]}"""

def build_lightweight_registry(diagnostics_dir) -> list[dict]:
    """Build registry from config.json (no model load needed for calibration-only)."""

def save_calibration(calibration, output_dir) -> None:
def load_calibration(output_dir) -> dict:
```

### 10.4 balance.py

```python
def absorb_into_adaln(adaln_linear, b_qkv, b_fc1=None, hidden_size=1536) -> None:
    """Absorb B⁻¹ into adaLN_modulation.layers[1].  b_fc1=None for block 23 text."""

def absorb_into_final_adaln(adaln_linear, b_final, hidden_size=1536) -> None:
    """Absorb B⁻¹ into final_layer.adaLN_modulation.layers[1]."""

def balance_weight(linear, b_vector) -> None:
    """W_new = W * b[None, :].  Bias unchanged."""

def apply_csb_to_model(mmdit, registry, calibration, hidden_size=1536) -> dict[str, ndarray]:
    """Full CSB: absorb + balance + collect b_inv.
    Takes pre-computed calibration dict (not diagnostics_dir).
    Returns: dict[layer_name → b_inv] for online layers only."""
```

### 10.5 quantize.py

```python
def fake_quantize_a8(x: mx.array) -> mx.array:
    """Dynamic per-tensor symmetric 8-bit fake quantization."""

class W4A8Linear(nn.Module):
    """Drop-in nn.Linear replacement.  See Section 7.3."""

def quantize_model(mmdit, registry, b_inv_map, config) -> dict[str, dict]:
    """Replace target nn.Linear with W4A8Linear.  Returns layer_meta dict."""

def patch_pipeline_for_quantized_inference(pipeline) -> None:
    """Capture post-CSB adaLN weights and monkey-patch pipeline.load_mmdit
    so DiffusionKit restores absorbed (not original) adaLN weights."""

def save_quantized_model(mmdit, output_dir, config, layer_meta, b_inv_layers) -> None:
    """Save mmdit_quantized.safetensors + quantize_config.json.
    Filters out to_offload.* keys before saving."""

def load_quantized_model(pipeline, output_dir) -> dict:
    """Create W4A8Linear stubs → load weights → patch pipeline."""
```

### 10.6 diagnose.py (post-quantization diagnostics)

```python
def build_quantized_registry(mmdit) -> list[dict]:
    """Like Phase 1 registry but handles W4A8Linear modules.
    Recovers d_in from packed QuantizedLinear weights."""

class QuantizedLinearHook:
    """Monkey-patches W4A8Linear.__call__ to record input activations
    BEFORE b_inv scaling and A8 fake-quantization."""

def run_quantized_collection(pipeline, seed_prompt_pairs, collector, ...) -> None:
    """Euler denoising loop on quantized model with hooks (mirrors Phase 1)."""

def compute_weight_errors(fp16_weight_stats, quantized_registry) -> list[dict]:
    """Compare FP16 weight salience vs dequantized W4 weights.
    Returns per-layer MSE, max error, SNR."""

def compare_activation_trajectories(fp16_act_dir, w4a8_act_dir, layer_names) -> list[dict]:
    """Per-step MSE, SNR, and relative shift between FP16 and W4A8 trajectories."""
```

---

## 11. Calibration Settings

| Parameter | Value | Notes |
|---|---|---|
| Calibration data | 100 MS-COCO prompt-seed pairs | `src/settings/coco_100_calibration_prompts.txt` |
| Evaluation data | 256 prompt-seed pairs | `src/settings/evaluation_set.txt` |
| Prompt-seed format | `<seed>\t<prompt>` per line | Each prompt has a dedicated seed |
| Image resolution | 512×512 | Latent size (64, 64) |
| CFG weight | 4.0 | Classifier-free guidance |
| Denoising steps | 30 | Euler ODE sampler |
| Sampler | Euler (rectified flow) | `shift=1.0` for SD3 Medium |
| Model | `argmaxinc/mlx-stable-diffusion-3-medium` | SD3 2B, fp16 |

The seed-prompt pair format means each calibration/evaluation prompt has its own dedicated seed. The pairs are loaded via `_load_seed_prompt_pairs()` and passed as `list[tuple[int, str]]` throughout the pipeline — no cross-product of prompts × seeds.

---

## 12. Precision Strategy

Precision varies by pipeline stage to balance accuracy and efficiency:

| Stage | Model Execution | Math / Statistics | Stored As |
|---|---|---|---|
| Phase 1: data collection | **fp16** (model forward pass) | **fp32** (all reductions via `.astype(mx.float32)`) | **fp32** numpy `.npz` |
| Phase 2a: calibration | No model (pure computation) | **float64** (numpy default) | **fp64** `.npz` |
| Phase 2b: CSB absorption | N/A (in-place weight modification) | **float64** (numpy arrays) | Written back as **fp16** (`adaln_linear.weight.dtype`) |
| Phase 2b: weight balancing | N/A | **float64** (numpy) | Written back as **fp16** |
| Phase 2c: W4 quantization | N/A | Internal to `mx.quantize` | **4-bit** packed `uint32` + fp16 scales/biases |
| Online `b_inv` vectors | Used at inference time | **fp32** multiplication, cast back to fp16 | **fp32** (in model parameters) |
| Inference: model forward | **fp16** | fp16 (+ fp32 for `b_inv` multiply) | — |
| Inference: A8 fake-quant | Applied per-layer at runtime | fp16 | — |

**Rationale:** High-precision math (fp32/fp64) during calibration and absorption prevents rounding errors from accumulating across hundreds of layers. The final model runs entirely in fp16 (the same precision as the original model), with the single exception of `b_inv` being fp32 for numerical safety.

---

## 13. DiffusionKit Integration: adaLN Cache and `to_offload`

### 13.1 The Modulation Caching Mechanism

DiffusionKit's `cache_modulation_params()` pre-computes all adaLN modulation outputs for all timesteps and stores them in `_modulation_params`. After caching, it **offloads** the adaLN MLP weights (sets them to empty arrays) to save memory, storing the list of offloaded modules in a `to_offload` attribute.

During the denoising loop, the adaLN MLP is not executed — only cached parameters are looked up. `clear_modulation_params_cache()` clears the `_modulation_params` dict but does **not** restore the offloaded weights.

### 13.2 Weight Restoration Pattern

Both Phase 1 collection and Phase 2 diagnostics must handle this correctly for multi-prompt runs. The pattern used throughout:

```python
from mlx.utils import tree_flatten

# Capture adaLN weights BEFORE the denoising loop
_adaln_cache = [
    (k, v) for k, v in tree_flatten(pipeline.mmdit.parameters())
    if "adaLN" in k
]

for seed, prompt in seed_prompt_pairs:
    # ... encode text, create noise ...
    pipeline.mmdit.cache_modulation_params(pooled_conditioning, timesteps)
    # ... run Euler denoising loop (adaLN weights are now empty arrays) ...

    # Restore adaLN weights for the next iteration
    pipeline.mmdit.load_weights(_adaln_cache, strict=False)
    pipeline.mmdit.clear_modulation_params_cache()
```

### 13.3 The `to_offload` Problem

After `cache_modulation_params()`, the model has a `to_offload` attribute containing references to the offloaded adaLN modules. This attribute:

- Causes `ValueError` during `load_weights` if the saved checkpoint contains `to_offload.*` keys but a fresh model instance doesn't have them.
- Must be filtered out when saving: `weights = {k: v for k, v in ... if not k.startswith("to_offload.")}`.
- Must be filtered when loading: `filtered = [(k, v) for k, v in weights.items() if not k.startswith("to_offload.")]`.
- Should be explicitly deleted after restoration: `delattr(pipeline.mmdit, "to_offload")`.

### 13.4 Pipeline Patching for Quantized Inference

After CSB absorption, the adaLN weights contain the modified (absorbed) values. DiffusionKit's `generate_image` internally calls `clear_modulation_params_cache()` which triggers `load_mmdit(only_modulation_dict=True)` — this would reload the **original** un-absorbed adaLN weights from the HuggingFace checkpoint, undoing the CSB absorption.

`patch_pipeline_for_quantized_inference()` prevents this by:

1. Capturing the current (post-CSB) adaLN weights.
2. Monkey-patching `pipeline.load_mmdit` so that when called with `only_modulation_dict=True`, it returns the captured post-CSB weights instead of loading originals from disk.

This patch is applied both during `run_quantize.py` (after CSB, before saving) and inside `load_quantized_model()` (after loading saved weights).

---

## 14. End-to-End Data Flow

The complete pipeline from raw prompts to quantized inference, with precision annotations:

```
src/settings/coco_100_calibration_prompts.txt
│  (tab-separated: <seed>\t<prompt>)
│
▼
Phase 1: Data Collection  [model runs in fp16, stats in fp32]
├── build_layer_registry(mmdit)  →  287 layers (286 targets + context_embedder)
├── compute_weight_salience()    →  s(W_j) = max_i|W_{i,j}|  [fp32]
│                                    saved: diagnostics/weight_stats.npz
├── install_hooks()              →  monkey-patch __call__ on all 287 layers
├── run_diagnostic_collection()  →  Euler loop × 100 prompt-seed pairs
│   ├── For each (seed, prompt):
│   │   ├── encode_text → conditioning [fp16]
│   │   ├── cache_modulation_params → offloads adaLN weights
│   │   ├── 30 Euler steps: hooks fire, collector.record()
│   │   │   └── s(X_j^σ) = max|X[:,:,:,j]|  [computed in fp32, stored in fp32 numpy]
│   │   └── restore adaLN weights from _adaln_cache
│   └── Aggregation: elementwise max across all runs per (layer, step_idx)
│       saved: diagnostics/activation_stats/{layer_name}.npz  →  [T=30, d_in]
└── save_config()  →  diagnostics/config.json  (seed_prompt_pairs, layer_names, etc.)
│
▼
Phase 2a: Calibration  [pure numpy, float64]
├── For each target layer:
│   ├── load_phase1_data()          →  act_trajectory [30, d_in], wt_salience [d_in]
│   ├── compute_spearman_trajectory →  ρ_t [30]  (Spearman ρ per step)
│   ├── compute_ssc_weights         →  η_t [30]  (softmax of -ρ, Eq. 11)
│   ├── weighted_act = η @ act      →  s̄(X_j) [d_in]  (SSC-weighted salience)
│   └── b = (s̄/sW)^α clamped       →  b_j [d_in]  (balancing vector)
├── For QKV groups: merge weight salience (max or geomean), shared b_qkv
├── Mark o_proj and fc2 as b_inv_layers (need online balancing)
└── save_calibration()  →  calibration.npz + calibration_meta.json
│
▼
Phase 2b: CSB Re-parameterization  [numpy float64, written back to fp16]
├── For each block × modality:
│   ├── absorb_into_adaln(adaLN_linear, b_qkv, b_fc1)
│   │   ├── Shift rows: W /= b[:, None],  bias /= b
│   │   └── Scale rows: W /= b[:, None],  bias = (1+bias)/b - 1
│   ├── balance_weight(q_proj, b_qkv)  →  W *= b[None, :]
│   ├── balance_weight(k_proj, b_qkv)
│   ├── balance_weight(v_proj, b_qkv)
│   ├── balance_weight(fc1, b_fc1)
│   ├── balance_weight(o_proj, b_o)    →  + store b_inv = 1/b [fp32]
│   └── balance_weight(fc2, b_fc2)     →  + store b_inv = 1/b [fp32]
├── absorb_into_final_adaln(final_adaln, b_final)
├── balance_weight(final_layer.linear, b_final)
└── patch_pipeline_for_quantized_inference()
│
▼
Phase 2c: W4A8 Quantization
├── For each of 286 target layers:
│   ├── nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
│   │   └── mx.quantize: per-group affine 4-bit (Round-to-Nearest)
│   ├── W4A8Linear(qlinear, b_inv)  (b_inv=None for absorbed layers)
│   └── setattr(parent, attr, w4a8_module)  →  replace in model tree
└── save_quantized_model()
    ├── mmdit_quantized.safetensors  (filters to_offload.* keys)
    └── quantize_config.json         (layer metadata, b_inv_layers, hyperparams)
│
▼
Inference  [fp16 model, fp32 b_inv multiply]
├── load_quantized_model(pipeline, output_dir)
│   ├── Create W4A8Linear stubs from metadata
│   ├── Load safetensors weights (filter to_offload.*)
│   └── patch_pipeline_for_quantized_inference()
└── pipeline.generate_image(prompt, seed=seed, ...)
    └── For each denoising step:
        └── For each W4A8Linear layer:
            ├── x * b_inv [fp32] → .astype(fp16)   (online layers only)
            ├── fake_quantize_a8(x)                  (per-tensor symmetric 8-bit)
            └── qlinear(x)                           (4-bit × fp16 matmul)
```

---

## 15. CLI Reference

### 15.1 End-to-End Pipeline (recommended)

```bash
# Full run: collection + calibration + CSB + quantize + save
python -m src.phase2.run_e2e --output-dir quantized/

# Quick test with 2 prompt-seed pairs
python -m src.phase2.run_e2e --output-dir quantized/ --num-prompts 2

# Skip collection (reuse existing Phase 1 diagnostics)
python -m src.phase2.run_e2e --output-dir quantized/ --skip-collection

# Override hyperparameters
python -m src.phase2.run_e2e --output-dir quantized/ --alpha 0.3 --qkv-method geomean
```

### 15.2 Standalone Quantization (requires prior Phase 1 data)

```bash
# Full: calibrate → balance → quantize → save
python -m src.phase2.run_quantize --output-dir quantized/

# Calibrate only (no model load)
python -m src.phase2.run_quantize --calibrate-only --output-dir quantized/

# Use saved calibration, skip recalculation
python -m src.phase2.run_quantize --from-calibration quantized/ --output-dir quantized/
```

### 15.3 Image Generation

```bash
# FP16 baseline — single prompt
python -m src.phase2.run_inference --mode fp16 --prompt "a cat on a couch" --output-dir results/

# W4A8 quantized — single prompt
python -m src.phase2.run_inference --mode w4a8 --quantized-dir quantized/ \
    --prompt "a cat on a couch" --output-dir results/

# Batch evaluation (seed-prompt pairs from file)
python -m src.phase2.run_inference --mode w4a8 --quantized-dir quantized/ \
    --prompts-file src/settings/evaluation_set.txt --output-dir results/

# Limit to first 5 pairs for quick test
python -m src.phase2.run_inference --mode fp16 \
    --prompts-file src/settings/evaluation_set.txt --num-prompts 5 --output-dir results/
```

Output layout: `results/fp16/000.png, 001.png, ...` and `results/w4a8/000.png, 001.png, ...`

### 15.4 Post-Quantization Diagnostics

```bash
# Full diagnostics: collect W4A8 stats + compare vs FP16 + generate plots
python -m src.phase2.run_diagnose --quantized-dir quantized/ --output-dir post_quant_diagnostics/

# Quick test with 2 prompts
python -m src.phase2.run_diagnose --quantized-dir quantized/ --num-prompts 2

# Skip collection (reuse existing W4A8 stats)
python -m src.phase2.run_diagnose --quantized-dir quantized/ --skip-collection

# Analysis + plots only (no model loading)
python -m src.phase2.run_diagnose --analysis-only --output-dir post_quant_diagnostics/
```

---

## 16. Practical Notes

### 16.1 Memory Budget

| Component | FP16 size | W4 size | Savings |
|---|---|---|---|
| `q/k/v_proj` (144 layers) | 144 × 1536² × 2B = 679 MB | 144 × 1536² × 0.5B = 170 MB | 75% |
| `o_proj` (47 layers) | 47 × 1536² × 2B = 222 MB | 47 × 1536² × 0.5B = 55 MB | 75% |
| `fc1` (47 layers) | 47 × 6144 × 1536 × 2B = 887 MB | 222 MB | 75% |
| `fc2` (47 layers) | 47 × 1536 × 6144 × 2B = 887 MB | 222 MB | 75% |
| `final_layer.linear` | 64 × 1536 × 2B = 192 KB | 48 KB | 75% |
| Online `b_inv` vectors | — | ~1.4 MB | — |
| **Total quantized layers** | **~2.67 GB** | **~0.67 GB** | **~2 GB saved** |

Non-quantized components (adaLN, embedders, norms, SDPA, text encoders, VAE) remain at FP16 and are unchanged.

### 16.2 Hyperparameter Sensitivity

- **α (CSB exponent):** α = 0.5 is the SmoothQuant default and a good starting point. Lower α (0.3) applies less balancing, preserving the original weight distribution at the cost of less activation compression. Higher α (0.7) compresses activations more aggressively. Tune based on output quality.
- **group_size (W4):** Smaller groups (32) provide finer-grained quantization but increase scale/bias overhead. Larger groups (128) are more memory-efficient but may lose precision on layers with heterogeneous weight distributions. Default 64 is a standard balance.
- **QKV method:** Test both Method 1 (max) and Method 2 (geomean) on a small validation set and compare output MSE.

### 16.3 Validation Strategy

Before full evaluation, verify correctness at each stage:

1. **After CSB absorption:** Run FP16 inference on the balanced (but unquantized) model. The output should be **numerically very close** to the original FP16 model (CSB is mathematically exact — the balancing cancels out). Small differences (max absolute error on the order of $10^{-3}$ for FP16) are expected due to floating-point rounding during weight modification. Larger discrepancies indicate a bug in absorption or weight modification.
2. **After W4 quantization (no A8):** Run inference with 4-bit weights but FP16 activations. Compare output to FP16 baseline.
3. **After W4A8:** Run full quantized inference. Compare to FP16 baseline.
4. **Post-quantization diagnostics:** Use `run_diagnose.py` to collect internal activation statistics from the quantized model and compare per-layer trajectories against FP16 Phase 1 baselines (MSE, SNR, relative shift).

For each comparison, compute:
- Per-pixel MSE on the denoised latent.
- Visual inspection on a fixed set of prompts and seeds.

### 16.4 Saving and Loading

**Saved artifacts** (in `--output-dir`, default `quantized/`):

| File | Contents |
|---|---|
| `mmdit_quantized.safetensors` | All MMDiT parameters (quantized + non-quantized). `to_offload.*` keys filtered out. |
| `quantize_config.json` | Hyperparameters, per-layer metadata (`d_in`, `d_out`, `has_bias`, `bits`, `has_b_inv`), `b_inv_layers` list. |
| `calibration.npz` | Per-layer balancing vectors `b` (float64). |
| `calibration_meta.json` | Layer names and `b_inv_layers` list. |

**Loading sequence** (`load_quantized_model`):

1. Read `quantize_config.json` to learn which layers are quantized and their shapes.
2. For each quantized layer, create a `W4A8Linear` stub (with `nn.QuantizedLinear` of correct `d_in`/`d_out`/`bits`/`group_size` and a zero `b_inv` placeholder if needed).
3. Replace the corresponding `nn.Linear` in the model tree.
4. Load `mmdit_quantized.safetensors`, filtering out `to_offload.*` keys.
5. Call `patch_pipeline_for_quantized_inference()` to prevent DiffusionKit from overwriting absorbed adaLN weights.

---

## 17. References

- **Phase 1 diagnostics data:** `diagnostics/` (activation trajectories, weight salience, summary table)
- **Phase 1 findings:** `src/phase1_findings.md` (salience patterns, complementarity, modality asymmetry, risk ranking)
- **Phase 1 implementation:** `src/phase1/` (reusable: `analyze.py` for `compute_spearman_trajectory`, `compute_ssc_weights`; `registry.py` for `build_layer_registry`)
- **DiffusionKit architecture:** `DiffusionKit/python/src/diffusionkit/mlx/mmdit.py` (`TransformerBlock.pre_sdpa`, `affine_transform`, `adaLN_modulation`, `FinalLayer`)
- **Architecture diagram:** `useful_doc/model.txt`
- **PTQ4DiT paper:** Eq. 4 (salience), Eq. 7 (balancing matrix), Eq. 11 (SSC weights), Eq. 13 (adaLN), Eq. 20 (re-parameterization)
