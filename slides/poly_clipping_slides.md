# Polynomial Clipping for Time-Aware Quantization of SD3
### Slides

---

## Slide 1: Title

**Polynomial Clipping for Time-Aware Quantization of Stable Diffusion 3**

Improving AdaRound for Dual-Stream Diffusion Transformers

---

## Slide 2: How Diffusion Works

**Core Idea: Learn to Denoise**

- Start with pure Gaussian noise: `x_T ~ N(0, I)`
- Iteratively remove noise over T steps: `x_T → x_{T-1} → ... → x_0`
- A neural network `ε_θ(x_t, t)` predicts the noise to remove at each step

**The Forward Process (training)**
- Corrupt a clean image by adding noise: `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`
- The model learns to reverse this corruption

**The Reverse Process (inference)**
- Run the same network 20–50 times, each at a different noise level σ
- Each step: "how much of what I see is noise?"

**Key consequence for quantization:**
> The same network weights are used at every noise level. But the activations — and the correct clipping range — are different at every step.

---

## Slide 3: Rectified Flow (SD3's Formulation)

**Simpler than DDPM — but same quantization problem**

**SD3 uses Rectified Flow:**
- Linear interpolation between data and noise: `x_t = (1-t) · x_0 + t · ε`
- Noise level σ = t, ranging from 1.0 (pure noise) → 0.0 (clean image)
- Euler ODE solver instead of DDPM's stochastic sampler

**Why this matters:**
- Smooth, linear input trajectory → smooth activation trajectories
- Low-degree polynomials are sufficient to model activation drift
- Simpler physics than cosine DDPM schedules → cooperates with compact polynomial fits

---

## Slide 4: SD3 Architecture — MM-DiT

**Multi-Modal Diffusion Transformer: Two Streams, Joint Attention**

```
Image Tokens  ──┐                          ┌── Image Output
                ├── Joint Q/K/V Attention ──┤
Text Tokens   ──┘                          └── Text Output
```

**What makes SD3 different from DiT-XL / U-ViT:**

| Feature | Single-stream DiT | SD3 MM-DiT |
|---------|-------------------|-----------|
| Token streams | 1 | 2 (image + text) |
| Attention | Self-attention on image only | Joint attention across both streams |
| Conditioning | Learned class embeddings | adaLN conditioned on σ and text |
| Blocks | 28 (DiT-XL) | 24 dual-stream + unified blocks |

**SD3 has 285 quantizable linear layers** across 24 MM-DiT blocks:
- q, k, v, o projections (4 × 2 streams × 24 blocks)
- fc1, fc2 FFN layers (2 × 2 streams × 24 blocks, minus final block text stream)

---

## Slide 5: SD3 — The Dual-Stream Problem

**Image and text streams have systematically different activation scales**

**Within the same joint attention tensor:**
- Block mm3: image stream is **2.2× larger** than text stream
- Block mm13: text stream is dominant

**This mismatch worsens during denoising:**
- mm3 Q-projection ratio: **1.65× at σ=1.0 → 3.58× at σ=0.09**
- Worst exactly when errors are most perceptually visible (fine detail)

**34.8%** of (block, σ) pairs have one stream >1.5× the other

**adaLN compounds this:** SD3 uses adaptive layer normalization to shift activations by noise-level-conditioned amounts:

| Layer | Shift magnitude |
|-------|----------------|
| mm22 txt mlp_fc2 | **254 units** |
| mm21 txt mlp_fc2 | **124 units** |
| mm14 txt mlp_fc2 | **108 units** |
| Typical layer | 1–5 units |

A shift of 254 units means 93% of INT8 buckets are wasted on empty space.

---

## Slide 6: Quantization — The Basics

**Replace 16-bit floats with low-bit integers**

**Why quantize?**
- Model size: 4× smaller (FP16 → INT4)
- Memory bandwidth: fewer bytes to move → faster inference
- SD3 Medium: **4.17 GB** in FP16 → ~1.0 GB in W4

**How it works:**
```
scale = α / qmax                    (α = clipping half-width)
x_q = clip(round(x / scale), -qmax, qmax) × scale
```

**The clipping range α is everything:**
- Too small: activations outside `[-α, +α]` are hard-clipped → large errors
- Too large: integer grid is coarsely spaced → wasted precision

**For a static model (LLM, classifier):** calibrate α once, done.

**For a diffusion model:** the activations at σ=1.0 are completely different from σ=0.09. A single α is wrong at almost every step.

---

## Slide 7: Round-to-Nearest — The Baseline Problem

**The simplest approach: just round each weight to the nearest integer**

```
W_q = round(W / scale) × scale
```

**Why it fails:**

Each weight element has a fractional rounding error between −0.5 and +0.5 quantization steps. For a layer with thousands of weights, these errors accumulate in the output:

```
y_q = W_q · x = (W + ΔW) · x = y + ΔW · x
```

The output error `ΔW · x` is not random noise — it's correlated with the input `x`. For an entire transformer block, errors from all layers compound.

**The rounding decision is binary and irreversible:** once you decide weight `w_ij` rounds up vs. down, that decision is locked in for all inputs.

**The problem is that rounding is done weight-by-weight, independently:**
- Round-to-nearest treats each weight in isolation
- Ignores how rounding errors interact across the entire weight matrix
- A weight that rounds up might compensate for a weight that rounds down — but RTN never considers this

**Result:** Sub-optimal weight rounding, especially at 4-bit where each step is large.

---

## Slide 8: AdaRound — Learned Rounding

**Key Insight: Rounding decisions should minimize *block output* error, not *weight* error**

**Nagel et al., 2020 — "Up or Down? Adaptive Rounding for Post-Training Quantization"**

Instead of rounding each weight to the nearest value, *learn* whether each weight rounds up or down:

```
W_q = s · clip(⌊W/s⌋ + h(α_w), 0, 1)
```

where `h(α_w)` is a soft sigmoid that learns to be 0 (round down) or 1 (round up).

**Objective — block-level reconstruction:**
```
min_α  ||block_fp16(x) - block_quant(α, x)||²  +  λ · Σ(1 - |2h(α_w) - 1|^β)
        block reconstruction loss                    regularization (push to 0 or 1)
```

**β-annealing:** Start with soft decisions (β small), anneal to hard binary choices.

**Block-level is key:** All linears in a transformer block are optimized simultaneously in a single pass. The loss is the *block's* output error — not any individual layer's. This means:
- Layers can compensate for each other — a rounding error in `qkv` can be offset by `o_proj`
- The optimizer finds error-cancelling rounding configurations that per-layer approaches miss
- Each layer still has its own independent `alpha` (rounding) and `a_scale` (clipping range)

**β-annealing:** Start with soft decisions (β small), anneal to hard binary choices.

**AdaRound is now standard** — used in GPTQ, QuIP, and most modern PTQ pipelines.

---

## Slide 9: AdaRound's Limitations for Diffusion Models

**AdaRound was designed for static models. Diffusion models break its core assumption.**

### Limitation 1: Static Activation Clipping
AdaRound needs a fixed clipping range α for activations during optimization. It calibrates this once — at one noise level. The rounding decisions are then optimized against that single α.

At inference, every other noise level uses the wrong α. **Clipping error dominates reconstruction loss**, and AdaRound's rounding optimization is fighting a problem it can't solve.

### Limitation 2: Single-Distribution Calibration
The optimization sees activations from a fixed σ. Rounding decisions are tuned to minimize error at that σ — not across the 20–50 steps of actual inference.

### Limitation 3: No Cross-Stream Awareness
Standard AdaRound treats each layer independently. It doesn't know that the image and text streams in joint attention have different scales, or that they drift in opposite directions.

### Limitation 4: Equal-Weight Loss
All calibration samples contribute equally to the reconstruction loss. But a rounding error at σ=0.1 (fine detail, perceptually visible) is much more damaging than the same error at σ=0.9 (mostly noise).

> **The core problem:** AdaRound optimizes the right objective (output reconstruction) but with the wrong signal (single-σ clipping, single-distribution calibration, uniform loss weighting).

---

## Slide 10: What We Discovered — 5 Failure Modes

**We collected activation statistics across 30 images × 30 noise levels, all 285 layers**

**Failure Mode 1: Cross-Stream Scale Mismatch**
Image and text streams have 1.65–3.58× scale differences within the same joint attention tensor. Worsens during denoising.

**Failure Mode 2: adaLN-Induced Distribution Shift**
Four txt mlp_fc2 layers (mm14, mm20, mm21, mm22) have activation centers at +60 to +254. Symmetric quantization wastes 93% of INT8 buckets on empty space.

**Failure Mode 3: Non-Linear Rectified Flow Drift**

| Layer | Linear R² | Quadratic R² |
|-------|-----------|--------------|
| mm9 img mlp_fc2 | **0.14** | 0.71 |
| mm0 img mlp_fc2 | 0.63 | 0.94 |
| mm18 img mlp_fc2 | 0.71 | 0.91 |

mm9 has a U-shaped trajectory — linear fit is essentially useless.

**Failure Mode 4: Per-Channel Outlier Asymmetry**
Image stream: 2–3 extreme outlier channels then flat. Text stream: more uniform. SmoothQuant applied to the concatenated tensor helps one stream and hurts the other.

**Failure Mode 5: Opposite Trajectory Directions**
Image attention projections **rise** as denoising progresses. Text attention projections **fall**. The cross-stream ratio is not just large — it's accelerating.

---

## Slide 11: Key Observation — Activation Trajectories Are Smooth

**Plot any layer's activation scale vs. σ — it's not noise. It's a smooth curve.**

```
α(σ)
 │                          ● (mm3 img, rising)
 │                    ●─●─●
 │               ●─●─●
 │          ●─●─●
 │     ●─●─●
 │  ●─●
 │●
 └─────────────────────────── σ
  1.0   0.7   0.5   0.3   0.1

vs.

α(σ)
 │●─●
 │    ●─●─●
 │         ●─●─●         (mm3 txt, falling)
 │              ●─●─●
 │                   ●─●─●
 └─────────────────────────── σ
```

**This isn't a coincidence — it's physics.**
Rectified flow defines `x_t = (1-t)x_0 + t·ε`. The network sees a smooth interpolation between data and noise. Smooth inputs → smooth activation trajectories.

**Smooth curves → low-degree polynomials.**

---

## Slide 12: Our Proposal — Polynomial Clipping Schedule

**Fit a polynomial α(σ) to each layer's activation trajectory. Evaluate it at inference time.**

```
α(σ) = c₀ + c₁σ + c₂σ²    (degree 2 example)
```

**Tiered degree selection:**

| Condition | Degree | Meaning |
|-----------|--------|---------|
| CV < 0.10 | 0 (static) | Layer is stable — one α suffices |
| R² > 0.85 | 2 (quadratic) | Smooth parabolic trajectory |
| Cubic gain > 0.15 | 3 (cubic) | Significant additional curvature |

**Results across SD3's 285 layers:**

| Degree | Count | Fraction |
|--------|-------|----------|
| 0 (static) | 227 | **79.6%** |
| 2 (quadratic) | 57 | 20.0% |
| 3 (cubic) | 1 | 0.4% |

**80% of layers are stable — polynomial isn't even needed for most of the network.**

**Fit quality for the 58 dynamic layers:**
- Median R² = **0.944**
- Mean R² = 0.925
- Worst case R² = 0.711 (vs. 0.14 for linear on the same layer)

---

## Slide 13: σ-Aware AdaRound

**What changes with polynomial clipping:**

**Without polynomial:**
- Calibration sample at σ=0.1 arrives
- Its activations are clipped with α calibrated at some average σ
- Reconstruction error = clipping error + rounding error
- AdaRound tries to compensate for clipping error via rounding — it can't
- Rounding decisions are suboptimal

**With polynomial:**
- Same sample arrives
- Each layer evaluates its own α(σ=0.1) → correct clipping range for this layer at this σ
- Reconstruction error = rounding error only
- AdaRound optimizes block reconstruction against the right signal at every layer simultaneously
- Rounding decisions reflect actual quantization difficulty

**Each layer has its own independent polynomial:**
- `mm_00_img_attn_qkv` follows one α(σ) trajectory
- `mm_00_img_mlp_fc1` follows a different α(σ) trajectory
- All are evaluated per-sample and fed into the block-level reconstruction loss together

**Single joint optimization across the full denoising trajectory:**
- One AdaRound pass sees samples from all σ ∈ [0.09, 1.0]
- Each sample has per-layer correct α(σ) from the polynomial
- No buckets, no momentum averaging — continuous σ-aware optimization

**Compare to prior approaches:**

| Method | σ handling | Clipping |
|--------|-----------|---------|
| Standard AdaRound | Single σ | Static α |
| TaQ-DiT | Momentum average | Static (shifted) |
| HTG | 2–4 buckets | Per-bucket static α |
| **Ours** | **Continuous** | **α(σ) polynomial** |

---

## Slide 14: σ-Weighted Loss

**Not all noise levels are equally important**

**Key insight:** Quantization errors at low σ (near-clean image, fine detail) are far more perceptually damaging than errors at high σ (mostly noise).

**Perceptual weighting:**
```
w(σ) = 1 / (σ + 1)
```

σ=0.03 gets ~15× more weight than σ=14.6.

**This only works because clipping is already correct at each σ.**

Without per-σ clipping:
- A sample at σ=0.1 has huge reconstruction error due to wrong clipping
- Weighting it heavily tells AdaRound to focus on a sample it can't fix
- Rounding decisions become distorted trying to compensate for clipping

With polynomial clipping:
- Every sample's reconstruction error is purely rounding error
- Weighting directs optimization toward perceptually critical timesteps
- Loss is focused on the right problem at the right timesteps

**Loss weighting strategies — all implemented:**

| Strategy | Formula | What it targets |
|----------|---------|----------------|
| Perceptual | `1/(σ + ε)` | Low-σ fine detail |
| Trajectory sensitivity | `\|dα/dσ\|` | Blocks where clipping range changes fastest |
| Combined | `\|dα/dσ\| / (σ + ε)` | Both fragile and perceptually visible |

The derivative `dα/dσ = c₁ + 2c₂σ` is exact and free — no finite-difference approximations needed.

**Why derivative weighting converges:**
- The loss is still MSE block reconstruction — same curvature, same landscape
- Weights are normalized to mean=1, so gradient magnitude stays comparable to unweighted training
- Learning rates and β-annealing schedule are unchanged
- For blocks with a flat polynomial (`|dα/dσ| ≈ 0`), weights collapse to uniform — identical to baseline
- Only blocks with steep α(σ) transitions receive amplified gradients, pushing the optimizer to get those right

**Blocks with zero derivative are not penalized** — the function returns uniform weights, so static layers behave identically to unweighted AdaRound.

---

## Slide 15: Why This Is Novel

**Three levels of novelty**

### 1. Continuous σ-dependent activation clipping
Nobody else does this. Prior work either ignores temporal drift (static AdaRound/GPTQ), averages across timesteps via momentum (TaQ-DiT), or discretizes into buckets (HTG). We model the drift as a continuous function and evaluate it on the fly.

### 2. Per-stream polynomial modeling for dual-stream architectures
SD3's image and text streams drift in **opposite directions simultaneously**. Existing quantization methods assume single-stream models. We fit separate polynomials per stream — the first approach that handles opposite-direction trajectories in a coupled attention architecture.

### 3. σ-aware joint AdaRound
A single optimization pass where every calibration sample has its own correct α(σ). Prior work either averages across σ (TaQ-DiT) or runs separate optimizations per bucket (HTG). Ours jointly optimizes rounding across the entire denoising trajectory with correct per-sample clipping.

**What nobody has done:**
> Asked "what is α *as a function of* σ, and can we model it directly?" Everyone else either ignores σ, averages over it, or discretizes it.

**Rectified flow cooperates:** Linear interpolation physics → smooth activation trajectories → degree-2 polynomials capture 94.4% of variance (median R²). DDPM's cosine schedule would produce more complex trajectories. SD3's physics actively helps.

---

## Slide 16: Storage and Runtime Overhead

**The polynomial schedule is essentially free**

**Storage:**
- 402 coefficients for 285 layers
- ~10 KB JSON file (vs. 4.17 GB model weights)
- Negligible

**Runtime:**
- Degree-2 evaluation: 2 multiplies + 2 adds per layer per step
- 285 layers × 4 ops = 1,140 floating-point ops per denoising step
- Compare: one transformer block = hundreds of millions of MACs
- **Runtime overhead: unmeasurable**

**Implementation:**
```python
# At each denoising step, for each layer:
sigma = current_noise_level
alpha = c2 * sigma**2 + c1 * sigma + c0   # polynomial evaluation
scale = alpha / 127                         # INT8 scale
x_q = fake_quant(x, scale)                # quantize activations
```

**Scheduler independence:**
- The polynomial is defined over the continuous range σ ∈ [0.09, 1.0]
- Works at any number of denoising steps: 20, 28, 50, or more
- No recalibration needed when changing the scheduler

---

## Slide 17: Results — Schedule Generalization

**Do the polynomial coefficients generalize across different calibration sets?**

We fitted schedules on two independent calibration groups and compared:

- **285 layers** evaluated
- **Median NRMSE: 0.98%** normalized error across all layers
- **95th percentile NRMSE: 4.43%**
- **98% of layers generalize with < 1% error**

The 4 extreme-shift layers (mm14, mm20, mm21, mm22 txt mlp_fc2) are the exceptions — their 60–254 unit shifts make them inherently harder to fit and they are kept in FP16.

**Takeaway:** The polynomial trajectories are a real property of the model's physics, not overfit to specific calibration images.

---

## Slide 18: What's Still To Do

**Current status and next steps**

| Component | Status |
|-----------|--------|
| Activation statistics collection | ✅ Done (188 layers × 30 timesteps) |
| Polynomial schedule generation | ✅ Done (227 static, 57 quadratic, 1 cubic) |
| σ-aware AdaRound with polynomial | ✅ Implemented |
| σ-weighted loss | ✅ Implemented |
| 4 extreme layers FP16 exclusion | ✅ Identified |
| Derivative-weighted loss (`\|dα/dσ\|`) | ✅ Implemented |
| Asymmetric quantization (shift polynomials) | 🔧 Infrastructure ready, not benchmarked |
| End-to-end FID evaluation | ⏳ In progress |
| Comparison vs. TaQ-DiT / HTG baseline | ⏳ Pending |

**The key open question:**
How much does polynomial clipping improve FID and PSNR vs. standard AdaRound and TaQ-DiT on SD3 W4A8?

---

## Slide 19: Summary

**What we built:**

1. **Characterized** 5 failure modes for quantizing SD3's dual-stream MM-DiT — none fully addressed by prior work

2. **Discovered** that activation scale trajectories follow smooth, low-degree polynomial curves (rectified flow physics cooperates)

3. **Built** a tiered polynomial clipping schedule: 402 coefficients, 10 KB, median R²=0.944, generalizes across calibration sets with 0.98% median error

4. **Extended** AdaRound to be σ-aware: each layer uses its own per-sample α(σ) within a block-level reconstruction loss, joint optimization across the full denoising trajectory, with σ-weighted and derivative-weighted loss (`|dα/dσ|`) focusing gradients on perceptually critical and rapidly-changing timesteps

5. **Modeled** image and text streams independently, the first approach that handles opposite-direction activation trajectories in dual-stream transformers

**The central claim:**
> Polynomial clipping removes activation clipping error from AdaRound's reconstruction loss, allowing rounding decisions to be optimized against the right signal. σ-weighted loss then directs that optimization toward perceptually important timesteps. Together these address the core limitation of applying AdaRound to iterative generative models.
