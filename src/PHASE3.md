# Phase 3: Polynomial Clipping for Timestep-Aware A8 Quantization

## 1. Motivation and Scope

Phase 2 provides two activation-quantization strategies for the 286 W4A8 target layers:

| Mode | Scale computation | Strengths | Weaknesses |
|------|-------------------|-----------|------------|
| **Dynamic** | `scale = max(|x|) / 127` every forward | Tracks every input exactly | Per-forward max-reduction cost; no offline optimisation signal |
| **Static** (SSC-weighted / global-max) | One fixed scale per layer from Phase 1 calibration | Zero runtime overhead | Conservative: a single number must cover all σ; under-utilises the INT8 grid at most timesteps |

Both approaches ignore a key empirical observation: **activation magnitudes are smooth, low-degree functions of the denoising noise level σ**. Rectified flow's linear interpolation `x_t = (1−t)·x_0 + t·ε` produces activation trajectories that are well-described by degree-2 or degree-3 polynomials (median R² ≈ 0.94 for non-constant layers; see `useful_doc/POLYNOMIAL_CLIPPING_EXPLAINER.md` §3–4 and Phase 1 findings on temporal CoV).

Phase 3 introduces **polynomial clipping**: instead of a constant scale or a runtime max-reduction, each layer's A8 clipping bound is evaluated as `α(σ) = poly(σ)` — a cheap polynomial of the current noise level. This is a **third activation-quantization mode** that sits alongside `dynamic` and `static` in the existing `--act-quant` CLI.

**Goal.** Replace the `fake_quantize_a8` step inside `W4A8Linear.__call__` with a σ-conditioned variant that:

1. Evaluates a per-layer polynomial at the current σ to obtain the clipping bound α.
2. Computes `scale = α / 127`.
3. Quantises as usual: `x_q = clamp(round(x / scale), −128, 127); x̂ = x_q · scale`.

Because the polynomial is fitted to **post-CSB** activation statistics (i.e. `x / b`), the clip bound automatically accounts for balancing.

**Two granularities.** The schedule supports per-tensor (one polynomial per layer) and per-channel (one polynomial per input channel for high-ρ layers). The granularity decision is driven by the SSC Spearman ρ from Phase 2.

---

## 2. How Polynomial Clipping Relates to Phase 2

Phase 2 performs three sequential transformations on each layer before inference:

```
Phase 1 data ──► Calibration (SSC) ──► CSB (balance) ──► Quantise weights (W4)
```

The activation-quantisation step is **orthogonal** to CSB: CSB changes *what* the quantiser sees (balanced `x/b`), while activation quantisation determines *how tightly* that balanced input is discretised. Phase 3 replaces step "choose A8 scale" with a polynomial schedule, leaving CSB, SSC, and W4 completely unchanged.

```
Phase 2 pipeline (unchanged)                     Phase 3 addition
─────────────────────────────                     ─────────────────
1. Phase 1 collection         ──────────────────► 3a. Load post-CSB trajectories
2a. SSC calibration           ──────────────────►     + mean Spearman ρ per layer
2b. CSB absorption + weight balancing             3b. Fit polynomials:
2c. W4 quantisation                                   - per-tensor for most layers
                                                      - per-channel for high-ρ layers
                                                  3c. Save schedule JSON
                                                  3d. At inference: evaluate poly(σ)
                                                      → per-layer (or per-channel) A8 scale
```

### 2.1 What polynomial clipping replaces

| Component | Dynamic A8 | Static A8 | **Polynomial A8** |
|-----------|------------|-----------|-------------------|
| Scale source | `max(\|x\|)` at runtime | Fixed scalar from calibration | `poly(σ) / 127` |
| σ-awareness | Implicit (input varies) | None | Explicit |
| Runtime cost per layer | max-reduce over full tensor | 0 | 1 polynomial eval (~5 FLOPs) |
| Granularity | per-tensor or per-token | per-tensor or per-channel | **per-tensor or per-channel** |
| Offline cost | 0 | SSC-weighted aggregation | Polynomial fitting |
| INT8 utilisation | Optimal (tracks real data) | Worst-case (conservative) | Near-optimal (tracks trajectory) |

### 2.2 What stays the same

- **CSB balancing** — absorption into adaLN, online `b_inv` multiply for `o_proj`/`fc2`.
- **W4 weight quantisation** — `nn.QuantizedLinear`, per-group affine, group sizes 32/64.
- **`quantize_config.json`** — the saved checkpoint format is extended, not replaced.

---

## 3. Theory

### 3.1 Activation trajectory as a function of σ

Let `a_l(σ)` denote the per-tensor absmax of layer `l`'s input at noise level σ, **after** CSB balancing:

$$
a_l(\sigma) = \max_{b,n,j} \left|\frac{X_{b,n,j}^{(\sigma)}}{b_j}\right|
$$

Phase 1 collects per-channel max magnitudes `act_channel_max[t, j] = max_{b,n} |X_{b,n,j}|` at each denoising step `t` with associated `σ_t`. These are **unsigned** (always ≥ 0), computed as `max(|X_flat|, axis=0)` where `X_flat` is the activation reshaped to `(tokens, d_in)`.

The post-CSB per-tensor absmax trajectory is:

$$
a_l(\sigma_t) = \max_j \frac{\texttt{act\_channel\_max}[t, j]}{b_j}
$$

where `b` is the raw CSB balancing vector from Phase 2 calibration (stored in `calibration.npz`). This is valid because CSB transforms activations to `X/b`, and since `b > 0` (clamped to `[0.01, 100]`), `max(|X|)/b = max(|X/b|)`.

The per-channel post-CSB trajectories retain individual channel information:

$$
a_{l,j}(\sigma_t) = \frac{\texttt{act\_channel\_max}[t, j]}{b_j}
$$

### 3.2 Polynomial fit

Fit a polynomial of degree `d` to the observed `(σ_t, a_l(σ_t))` pairs:

$$
\hat{a}_l(\sigma) = \sum_{k=0}^{d} c_k \, \sigma^k
$$

Degree selection follows a tiered strategy (implemented in `poly_clipping.py`):

| Condition | Degree | Rationale |
|-----------|--------|-----------|
| absmax range < 2 across σ | 0 (constant = max absmax) | Effectively flat — polynomial adds no value |
| absmax range < 5 | 2 max | Low variation — cap at quadratic for stable derivatives |
| Quadratic R² > 0.85 | 2 | Standard parabolic fit sufficient |
| Cubic R² gain > 0.15 over quad | 3 | Captures inflection points (U-shapes) |
| Quartic R² gain > 0.10 over cubic | 4 | Rare; complex trajectories |

Coefficients are stored in **ascending-power order**: `[c₀, c₁, …, c_d]`.

### 3.3 Per-tensor vs per-channel polynomial

The schedule supports two granularities, selected per layer based on the mean Spearman ρ from Phase 2's SSC calibration:

**Per-tensor** (default, ρ ≤ threshold):

$$
\alpha_l(\sigma) = \hat{a}_l(\sigma), \quad \text{scale} = \frac{\alpha_l}{127}
$$

One polynomial per layer, one scale for the entire activation tensor. Works well when CSB has equalised the channels — after dividing by `b`, all channels have similar magnitudes.

**Per-channel** (ρ > threshold):

$$
\alpha_{l,j}(\sigma) = \hat{a}_{l,j}(\sigma), \quad \text{scale}_j = \frac{\alpha_{l,j}}{127}
$$

One polynomial per input channel, yielding a vector `α ∈ ℝ^{d_in}`. Each channel gets its own σ-conditioned bound.

**Why ρ drives the decision:** High ρ means channels maintain a persistent magnitude hierarchy across timesteps — some channels are always large, others always small. CSB can only partially equalise them. A single per-tensor bound must be set to the worst-case channel, wasting precision for all others. Per-channel polynomial gives each channel a tight bound, reclaiming that wasted precision. Low ρ means channel rankings shuffle across σ, so CSB effectively equalises them and a single per-tensor bound suffices.

### 3.4 Per-channel polynomial fitting

For a per-channel layer with `d_in` input channels, all channels are fitted at a **uniform degree** determined by applying `select_degree` to the per-tensor absmax trajectory. This gives a coefficient matrix `C ∈ ℝ^{d_in × (d+1)}`.

The fitting uses a vectorised Vandermonde least-squares solve:

$$
V \cdot C^T = \text{post\_csb}, \quad V = \begin{bmatrix} 1 & \sigma_1 & \sigma_1^2 & \cdots \\ \vdots & & & \\ 1 & \sigma_T & \sigma_T^2 & \cdots \end{bmatrix}
$$

$$
C^T = (V^T V)^{-1} V^T \, \text{post\_csb}
$$

This solves all `d_in` channels simultaneously via `np.linalg.lstsq`. R² is computed per-channel and averaged.

For degree 0, the per-channel constant is the conservative max across all timesteps: `max_t post_csb[t, j]` per channel (not the lstsq mean).

### 3.5 Clipping bound and scale

At inference, given current noise level σ:

$$
\alpha_l(\sigma) = \max\bigl(\hat{a}_l(\sigma),\; \epsilon\bigr), \quad \epsilon = 10^{-8}
$$

$$
\text{scale}_l(\sigma) = \frac{\alpha_l(\sigma)}{127}
$$

The fake-quantisation is:

$$
X_q = \text{clamp}\!\left(\text{round}\!\left(\frac{X}{\text{scale}_l(\sigma)}\right),\; -128,\; 127\right), \quad \hat{X} = X_q \cdot \text{scale}_l(\sigma)
$$

For per-channel, `α` and `scale` are vectors of shape `[d_in]`, and the division `X / scale` broadcasts along the last dimension: `(batch, seq, d_in) / (d_in,)`.

### 3.6 Optional: asymmetric shift polynomial (future)

For layers with large adaLN-induced distribution shift, a second polynomial models the distribution center:

$$
\mu_l(\sigma) = \sum_{k=0}^{d'} c'_k \, \sigma^k
$$

The quantisation window shifts to `[μ − α, μ + α]`:

$$
X_q = \text{clamp}\!\left(\text{round}\!\left(\frac{X - \mu_l(\sigma)}{\text{scale}_l(\sigma)}\right),\; -128,\; 127\right), \quad \hat{X} = X_q \cdot \text{scale}_l(\sigma) + \mu_l(\sigma)
$$

> **Current limitation.** Phase 1 collects `max(|X|)` per channel (unsigned), which does not carry sign information. Computing the distribution center requires signed `tensor_min` / `tensor_max` statistics. The `W4A8PolyLinear` module supports shift coefficients, but `generate_schedule_from_diagnostics` cannot produce them from Phase 1 data. A future Phase 1 enhancement to collect signed statistics would enable this.

---

## 4. Data Flow

### 4.1 Offline: schedule generation

```
diagnostics/                        quantized/<tag>/
├── activation_stats/               ├── calibration.npz        (raw b vectors)
│   ├── blocks.0.image.attn.q_proj.npz     └── calibration_meta.json  (layer_names, mean_rhos)
│   └── ...
└── weight_stats.npz
         │                                    │
         ▼                                    ▼
    ┌──────────────────────────────────────────────┐
    │  Phase 3a: generate_schedule_from_diagnostics│
    │                                              │
    │  Inputs:                                     │
    │    - sigma_values [T] from any layer npz     │
    │    - act_channel_max [T, d_in] per layer     │
    │    - b vectors from calibration.npz          │
    │    - mean_rhos from calibration_meta.json    │
    │                                              │
    │  For each layer l in 286 targets:            │
    │    1. post_csb = act_traj / b     [T, d_in]  │
    │    2. absmax_traj = max(post_csb, axis=1) [T]│
    │    3. if mean_rho > threshold:               │
    │       → _build_per_channel_entry             │
    │         degree = select_degree(absmax_traj)  │
    │         coeffs = lstsq(V, post_csb) [d_in,d+1]│
    │    4. else:                                  │
    │       → _build_per_tensor_entry              │
    │         degree, coeffs = select_degree(...)  │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
              poly_schedule.json
              {
                "version": "poly_v3_csb",
                "sigma_range": [0.001, 1.0],
                "per_channel_rho_threshold": 0.5,
                "n_per_channel": 17,
                "layers": {
                  "blocks.0.image.attn.q_proj": {
                    "degree": 2,
                    "coeffs": [c0, c1, c2],   ← 1D: per-tensor
                    "r2": 0.95, "cv": 0.12
                  },
                  "blocks.12.image.mlp.fc2": {
                    "degree": 2,
                    "coeffs": [[...], ...],    ← 2D: per-channel [d_in, 3]
                    "r2": 0.88, "cv": 0.25,
                    "granularity": "per_channel",
                    "n_channels": 1536
                  }, ...
                }
              }
```

### 4.2 Online: inference with polynomial A8

```
Denoiser loop (Euler sampler, T steps):
│
├── install_sigma_hook(mmdit)          ← wraps mmdit.__call__ (once)
│
├── For each step t with noise level σ_t:
│   │
│   ├── mmdit(timestep=σ_t*1000, ...)  ← hook intercepts, sets σ register
│   │
│   └── For each W4A8PolyLinear layer l:
│       │
│       ├── 1. [optional] x = x * b_inv       (online CSB, o_proj/fc2 only)
│       │
│       ├── 2. α = poly_eval(coeffs_l, σ_t)
│       │      Per-tensor coeffs [d+1]  → α is scalar
│       │      Per-channel coeffs [d_in, d+1] → α is vector [d_in]
│       │      scale = max(α, ε) / 127
│       │
│       ├── 3. x_q = clamp(round(x / scale), -128, 127)
│       │      x̂ = x_q * scale
│       │      (per-channel: broadcasts (..., d_in) / (d_in,))
│       │
│       └── 4. return qlinear(x̂)              (W4 matmul)
```

### 4.3 σ propagation

The denoiser must make the current σ visible to every quantised layer.

**Design constraint.** Both DiffusionKit's `sample_euler` and Phase 1/2's custom loops call `mmdit.cache_modulation_params(pooled, timesteps)` **once** before the loop with all timesteps pre-computed. The per-step σ is then conveyed through `mmdit(timestep=timesteps[i])` at each step.

**Implementation: dynamic-subclass hook on `mmdit.__call__`.** The hook:

1. Intercepts each call to `mmdit(timestep=..., ...)`.
2. Extracts the `timestep` kwarg (or `args[2]` as fallback for positional calls).
3. Converts to σ via `σ = timestep / 1000` (matching the sampler's `timestep(σ) = σ × 1000` convention).
4. Writes to a module-level register via `set_current_sigma(σ)`.
5. Calls the original `mmdit.__call__`.

Each `W4A8PolyLinear.__call__` reads from this register. This requires no DiffusionKit modifications — the hook is installed by `install_sigma_hook(mmdit)` using the same dynamic-subclass pattern as Phase 1's `LinearHook`. A guard prevents double-installation.

---

## 5. Module Design

### 5.1 Files

```
src/phase3/
├── __init__.py
├── poly_clipping.py          # Degree selection, polynomial fitting, schedule generation
│                              #   - select_degree, poly_r2 (per-tensor)
│                              #   - _fit_per_channel_polynomials (vectorised lstsq)
│                              #   - _load_calibration_data (b vectors + mean_rhos)
│                              #   - generate_schedule_from_diagnostics
│                              #   - print_summary
├── poly_eval.py              # Polynomial evaluation primitive (MLX, Horner's method)
│                              #   Handles 1D [d+1] and 2D [d_in, d+1] coefficients
├── quantize_poly.py          # W4A8PolyLinear module, σ register + hook, model loader
│                              #   - set_current_sigma / _get_current_sigma / reset_sigma_register
│                              #   - install_sigma_hook / uninstall_sigma_hook
│                              #   - load_quantized_model_poly
└── generate_schedule.py      # CLI: diagnostics + calibration → poly_schedule.json
```

Phase 3 is a **self-contained package**. It does not modify `src/phase2/` files. The `POLY_SCHEDULE_FILENAME` constant is defined in `poly_clipping.py` and imported by `quantize_poly.py` to ensure consistency. Future integration with Phase 2's `run_e2e.py` / `run_inference.py` is documented in §6 but not implemented here.

### 5.2 `poly_eval.py` — polynomial evaluation

```python
def poly_eval(coeffs: mx.array, sigma: mx.array) -> mx.array:
    """Evaluate polynomial ∑ c_k · σ^k using Horner's method.

    Parameters
    ----------
    coeffs : mx.array
        Coefficients in ascending-power order.
        - Shape [d+1]: per-tensor → returns scalar.
        - Shape [d_in, d+1]: per-channel → returns [d_in] vector.
    sigma  : mx.array, scalar
        Current noise level.
    """
    n = coeffs.shape[-1]
    if n == 0:
        return mx.array(0.0)
    if n == 1:
        return coeffs[..., 0]
    # Horner: ((c_d · σ + c_{d-1}) · σ + ... ) · σ + c_0
    result = coeffs[..., n - 1]
    for k in range(n - 2, -1, -1):
        result = result * sigma + coeffs[..., k]
    return result
```

The `[..., k]` indexing generalises Horner's method: for 1D `[d+1]` coefficients, it selects a scalar element (result is scalar); for 2D `[d_in, d+1]` coefficients, it selects a vector `[d_in]` (result is `[d_in]`). For degree 2 this is 2 multiplies + 2 adds — negligible vs the `quantized_matmul` that follows.

### 5.3 `W4A8PolyLinear` — quantised linear with polynomial A8

```python
class W4A8PolyLinear(nn.Module):
    """W4 weights + polynomial-clipped A8 activations."""

    def __init__(
        self,
        qlinear: nn.QuantizedLinear,
        poly_coeffs: mx.array,            # [d+1] or [d_in, d+1]
        b_inv: mx.array | None = None,
        shift_coeffs: mx.array | None = None,
    ):
        super().__init__()
        self.qlinear = qlinear
        self.poly_coeffs = poly_coeffs
        if b_inv is not None:
            self.b_inv = b_inv
        if shift_coeffs is not None:
            self.shift_coeffs = shift_coeffs

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype

        # 1. Online CSB (o_proj, fc2 only)
        if hasattr(self, "b_inv"):
            x = (x * self.b_inv).astype(orig_dtype)

        # 2. Read current σ from module-level register
        sigma = _get_current_sigma()

        # 3. Evaluate polynomial → clipping bound
        alpha = poly_eval(self.poly_coeffs, sigma)
        #   Per-tensor: alpha is scalar
        #   Per-channel: alpha is [d_in]
        alpha = mx.maximum(alpha, mx.array(1e-8))
        scale = alpha / 127.0

        # 4. Symmetric A8 fake-quantisation (broadcasts for per-channel)
        x_q = mx.clip(mx.round(x / scale), -128, 127)
        x = x_q * scale

        # 5. W4 matmul
        return self.qlinear(x.astype(orig_dtype))
```

For per-channel coefficients `[d_in, d+1]`, `poly_eval` returns `[d_in]`, and the division `x / scale` broadcasts naturally: `(batch, seq, d_in) / (d_in,)` → `(batch, seq, d_in)`. No code branching is needed — the same path handles both granularities.

### 5.4 σ register and hook

```python
_SIGMA_REGISTER: mx.array | None = None

def set_current_sigma(sigma: mx.array):
    global _SIGMA_REGISTER
    _SIGMA_REGISTER = sigma

def _get_current_sigma() -> mx.array:
    if _SIGMA_REGISTER is None:
        raise RuntimeError("σ register not set ...")
    return _SIGMA_REGISTER

def install_sigma_hook(mmdit):
    """Wrap mmdit.__call__ via dynamic subclass to set σ from per-step timestep."""
    if mmdit.__class__.__name__.endswith("_SigmaHooked"):
        return  # guard against double-installation

    original_cls = mmdit.__class__
    original_call = original_cls.__call__

    def _hooked_call(self, *args, **kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) >= 3:
            timestep = args[2]
        if timestep is not None:
            set_current_sigma(timestep.reshape(-1)[0] / 1000.0)
        return original_call(self, *args, **kwargs)

    mmdit.__class__ = type(
        original_cls.__name__ + "_SigmaHooked",
        (original_cls,),
        {"__call__": _hooked_call},
    )
```

The hook fires on every `mmdit(timestep=timesteps[i])` call inside the Euler loop. The conversion `σ = timestep / 1000` matches the sampler convention (`sampler.timestep(σ) = σ * 1000`), used identically by both DiffusionKit and Phase 1. The double-install guard checks the class name suffix.

### 5.5 Schedule generation from Phase 1 + Phase 2 data

The `_load_calibration_data` function reads **both** the raw `b` vectors and the `mean_rhos` from `calibration_meta.json`:

```python
def _load_calibration_data(calibration_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Returns (b_vectors, mean_rhos)."""
```

The main entry point is:

```python
def generate_schedule_from_diagnostics(
    diagnostics_dir: Path,
    calibration_dir: Path,
    *,
    max_degree: int = 4,
    include_shifts: bool = False,
    exclude_layers: list[str] | None = None,
    per_channel_rho_threshold: float | None = None,
) -> dict:
```

For each layer, the function:

1. Loads `act_channel_max [T, d_in]` and `sigma_values [T]`.
2. Loads `b [d_in]` from `calibration.npz`.
3. Computes `post_csb = act_traj / b` → `[T, d_in]`.
4. Computes `absmax_traj = max(post_csb, axis=1)` → `[T]`.
5. Checks `mean_rhos[name] > per_channel_rho_threshold`:
   - **Yes** → `_build_per_channel_entry`: uses `select_degree` on the per-tensor trajectory for uniform degree, then `_fit_per_channel_polynomials` on the full `[T, d_in]` matrix.
   - **No** → `_build_per_tensor_entry`: uses `select_degree` directly.

### 5.6 Loader: Phase 2 → polynomial A8 upgrade

```python
def load_quantized_model_poly(pipeline, quantized_dir: Path) -> dict:
    """Load Phase 2 W4A8 checkpoint and upgrade to polynomial A8."""
```

Steps:
1. Call Phase 2's `load_quantized_model(pipeline, quantized_dir)` — creates `W4A8Linear` modules with correct `b_inv` values (populated via `load_weights` from safetensors).
2. Read `poly_schedule.json`.
3. For each schedule entry, navigate to the `W4A8Linear` module and replace it with `W4A8PolyLinear`, transferring:
   - `module.qlinear` (W4 weights, already loaded).
   - `module.b_inv` if present (for o_proj/fc2).
   - `mx.array(entry["coeffs"])` — auto-detects shape: 1D list → `[d+1]`, 2D list → `[d_in, d+1]`.
4. Call `install_sigma_hook(pipeline.mmdit)`.

---

## 6. Integration with Phase 2 Pipeline

### 6.1 `run_e2e.py` extension (future)

Add `"poly"` as a third `--act-quant` choice. The e2e pipeline gains one additional step after CSB + W4:

```
Step 5 (existing):  W4 quantisation
Step 5b (new):      Generate polynomial schedule from diagnostics + calibration
Step 6 (existing):  Save checkpoint
```

The schedule JSON is saved alongside `mmdit_quantized.safetensors`:

```
quantized/<tag>/
├── mmdit_quantized.safetensors
├── quantize_config.json          # act_quant: "poly"
├── calibration.npz
├── calibration_meta.json
└── poly_schedule.json            # per-layer polynomial coefficients
```

### 6.2 Loading at inference (future)

`run_inference.py` and `benchmark_model.py` already branch on `act_quant` (`dynamic` vs `static`). Add a third branch:

```python
if meta.get("act_quant") == "poly":
    from ..phase3.quantize_poly import load_quantized_model_poly
    load_quantized_model_poly(pipeline, quantized_dir)
```

---

## 7. Per-Tensor vs Per-Channel: Design Rationale

| Granularity | Coefficients per layer | Scale shape | When selected | Benefit |
|-------------|----------------------|-------------|---------------|---------|
| **Per-tensor** (default) | `d+1` (e.g. 3 for degree 2) | scalar | ρ ≤ threshold (or threshold not set) | Minimal schedule size, one scale for all channels |
| **Per-channel** | `d_in × (d+1)` (e.g. 1536×3) | `[d_in]` | ρ > threshold | Each channel gets its own tight bound |

**Why CSB makes per-tensor viable for most layers:** CSB divides activations by `b`, equalising per-channel magnitudes. After CSB, the worst-case channel is much closer to the average, so a single per-tensor bound doesn't waste much precision.

**Why per-channel is needed for high-ρ layers:** High Spearman ρ means channel rankings are persistent across timesteps — CSB can only partially equalise them. The residual spread means a per-tensor bound set to the worst channel wastes 8-bit range for all other channels. Per-channel polynomial gives each channel its own σ-conditioned scale.

**The ρ threshold knob:** `--per-channel-rho-threshold 0.5` is the default. Higher = fewer per-channel layers (smaller schedule, slightly less precise). Lower = more per-channel layers (larger schedule, better precision). Setting it to `None` disables per-channel entirely (all layers are per-tensor).

---

## 8. Relationship to AdaRound / GPTQ Weight Optimisation

Polynomial clipping is not just a better A8 strategy — it **enables** timestep-aware weight optimisation. When a future GPTQ or AdaRound pass optimises weight rounding:

1. **Correct per-σ clipping**: Each calibration sample arrives with its σ. The polynomial evaluates `α(σ)` to give the correct clipping bound for that sample. The reconstruction loss then reflects only weight-rounding error, not clipping error.

2. **σ-weighted loss**: Weight the reconstruction loss by `w(σ) = 1/(σ + offset)` (perceptual) or `w(σ) = |dα/dσ|` (trajectory sensitivity). These strategies only make sense when the clipping is already correct at each σ.

3. **Derivative-based weighting**: The polynomial gives an exact derivative `dα/dσ` for free (e.g. for degree 2: `c₁ + 2c₂σ`). This identifies σ regions where the clipping bound is most sensitive.

---

## 9. Cost Analysis

### 9.1 Schedule size

| Configuration | Coefficients | JSON size |
|---------------|-------------|-----------|
| 286 layers, ~80% degree-0, ~20% degree-2 (all per-tensor) | ~286×1 + ~57×3 ≈ 457 | ~15 KB |
| + per-channel poly for ~17 high-ρ layers (d=2, d_in=1536) | +17×1536×3 ≈ 78K | ~600 KB |
| + shift polynomials for 4 layers (future) | +12 | negligible |

Per-tensor-only schedules are trivially small. Per-channel adds cost but only for a targeted subset (~17 layers with ρ > 0.5).

### 9.2 Runtime cost per denoising step

| Operation | FLOPs per layer | Total (286 layers) |
|-----------|-----------------|--------------------|
| Polynomial eval (degree 2, per-tensor) | 4 | 1,144 |
| Polynomial eval (degree 2, per-channel, d_in=1536) | 4 × 1536 ≈ 6K | ~102K (17 layers) |
| Dynamic A8 max-reduce (per-tensor, d_in=1536) | ~1536 | ~439,296 |
| W4 matmul (e.g. 1536×1536) | ~4.7M | ~1.35B |

Per-tensor polynomial eval is **~400× cheaper** than dynamic max-reduction. Per-channel polynomial eval costs ~6K FLOPs per layer (comparable to the max-reduce) but produces a tighter bound. Both are **negligible** compared to the matmul.

### 9.3 Memory

The polynomial coefficients are loaded from JSON into `mx.array` module attributes. Per-tensor: a few hundred floats total. Per-channel: ~78K floats for 17 layers (~300 KB in float32). This is negligible compared to the model's weight memory.

---

## 10. CLI Reference

### 10.1 Schedule generation (standalone)

```bash
# Generate per-tensor-only polynomial schedule
python -m src.phase3.generate_schedule \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/<tag>/ \
    --output quantized/<tag>/poly_schedule.json

# Hybrid: per-channel poly for layers with ρ > 0.5
python -m src.phase3.generate_schedule \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/<tag>/ \
    --per-channel-rho-threshold 0.5 \
    --output quantized/<tag>/poly_schedule.json

# Cap all layers at degree 0 (equivalent to static A8 with max absmax)
python -m src.phase3.generate_schedule \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/<tag>/ \
    --max-degree 0 \
    --output quantized/<tag>/poly_schedule.json
```

### 10.2 End-to-end with polynomial A8 (future)

```bash
# Full pipeline: collection → CSB → W4 → poly A8 schedule → save
python -m src.phase2.run_e2e --output-dir quantized/ --act-quant poly

# With per-channel for high-ρ layers
python -m src.phase2.run_e2e --output-dir quantized/ --skip-collection \
    --act-quant poly --poly-per-channel-rho-threshold 0.5
```

### 10.3 Inference (future)

```bash
# Inference loads poly_schedule.json automatically when act_quant=poly
python -m src.phase2.run_inference --mode w4a8 \
    --quantized-dir quantized/<tag>/ \
    --prompts-file src/settings/evaluation_set.txt \
    --output-dir results/
```

No special flags — the loader detects `"act_quant": "poly"` in `quantize_config.json` and instantiates `W4A8PolyLinear` modules with the σ hook.

---

## 11. Implementation Status

### Completed

| Step | Description | Files |
|------|-------------|-------|
| 1 | Polynomial fitting + schedule generation (per-tensor and per-channel) | `poly_clipping.py` |
| 2 | Horner's polynomial eval (1D and 2D) | `poly_eval.py` |
| 3 | `W4A8PolyLinear` module + σ register + hook + model loader | `quantize_poly.py` |
| 4 | CLI for schedule generation | `generate_schedule.py` |

### Implementation details

- `poly_clipping.py` inlines `poly_r2` (NumPy `polyfit` + R²), removing the external `src.explore_curve_fits` dependency.
- `_load_calibration_data` returns both `b_vectors` and `mean_rhos` from a single load of `calibration.npz` + `calibration_meta.json`.
- `_fit_per_channel_polynomials` uses a single `np.linalg.lstsq` call on the Vandermonde matrix to fit all channels simultaneously.
- `POLY_SCHEDULE_FILENAME` is defined once in `poly_clipping.py` and imported by `quantize_poly.py`.
- `install_sigma_hook` guards against double-installation by checking the class name suffix.
- `include_shifts` is explicitly set to `False` after the warning log to prevent future code from accidentally using the stale `True` value.

### Future work

| Step | Description |
|------|-------------|
| 5 | Extend `run_e2e.py` with `--act-quant poly` and `--poly-per-channel-rho-threshold` |
| 6 | Extend `run_inference.py` and `benchmark_model.py` to detect `act_quant=poly` and call `load_quantized_model_poly` |
| 7 | Phase 1 enhancement: collect signed `tensor_min` / `tensor_max` to enable shift polynomials |
| 8 | Benchmark comparison: dynamic vs static vs polynomial A8 on the standard evaluation set |

---

## 12. Expected Degree Distribution (from Phase 1 Data)

Based on Phase 1 findings and the explainer analysis:

| Degree | Expected count | Fraction | Typical layers |
|--------|---------------|----------|----------------|
| 0 (constant) | ~220–230 | ~80% | Most `q/k/v_proj`, `fc1`; layers with absmax range < 2 |
| 2 (quadratic) | ~50–60 | ~18% | Image `fc2`, image `o_proj`, some mid-block `q_proj` |
| 3 (cubic) | ~1–5 | ~1% | Layers with U-shaped or inflection-point trajectories |
| 4 (quartic) | 0–1 | <1% | Extremely rare |

~80% of layers get a single constant (identical to static A8 with max absmax), while ~20% benefit from σ-aware clipping. With `--per-channel-rho-threshold 0.5`, approximately 17 of those ~57 dynamic layers also get per-channel granularity.

---

## 13. Comparison of All A8 Modes

| Property | Dynamic | Static | Polynomial (per-tensor) | Polynomial (per-channel) |
|----------|---------|--------|------------------------|-------------------------|
| Scale accuracy | Exact | Conservative | Near-exact (R² ≈ 0.94) | Near-exact per channel |
| Runtime overhead | max-reduce per layer per step | 0 | ~5 FLOPs per layer | ~6K FLOPs per layer (d_in=1536) |
| σ-awareness | Implicit | None | Explicit | Explicit |
| Channel-level precision | No (one scale for all) | No | No (one scale for all) | Yes (one scale per channel) |
| INT8 grid utilisation | Optimal | Under-utilised | Near-optimal | Near-optimal per channel |
| Enables σ-aware AdaRound | No | No | Yes | Yes |
| Checkpoint size increase | 0 | `static_scales.npz` (~KB) | `poly_schedule.json` (~15 KB) | ~600 KB (17 per-channel layers) |
| Handles unseen σ values | Yes (data-driven) | Partially | Yes (extrapolates) | Yes (extrapolates) |

---

## 14. References

- **Phase 1 diagnostics:** `diagnostics/` (activation trajectories, sigma schedule).
- **Phase 2 calibration:** `quantized/<tag>/calibration.npz` (balancing vectors `b`), `calibration_meta.json` (layer names, mean Spearman ρ).
- **Phase 2 documentation:** `src/Phase2.md` (CSB, SSC, W4A8 architecture).
- **Polynomial clipping theory:** `useful_doc/POLYNOMIAL_CLIPPING_EXPLAINER.md`.
- **Phase 3 implementation:** `src/phase3/` (schedule generation, polynomial eval, inference module).
- **PTQ4DiT paper:** Eq. 4 (salience), Eq. 7 (balancing), Eq. 11 (SSC weights).
