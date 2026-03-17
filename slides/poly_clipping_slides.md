# Polynomial Clipping for Time-Aware Quantization of SD3
### Slides

---

## Slide 1: Title

**Polynomial Clipping for Time-Aware Quantization of Stable Diffusion 3 Medium-MLX on Apple Silicon**

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

## Slide 3: What Is Quantization?

**Replacing 16-bit floats with low-bit integers to make models smaller and faster**

**Why quantize?**
- SD3 Medium is **4.17 GB** in FP16 — too large for most consumer devices
- 4-bit weights cut that to ~1 GB
- Fewer bytes to move = faster inference, lower memory bandwidth

**How it works — the core idea:**

Think of it like converting a measurement to a coarser unit. If your thermometer reads −12.7°C and you store it as −13°C to save space, that's a tiny rounding error you can live with. Quantization does the same for neural network values — instead of rounding to whole degrees, you round to one of 256 integer "slots."

To do that, you need to know the range of values the layer produces. That range is defined by **α** (alpha) — the largest absolute value you expect to see. If a layer's activations typically fall between −5 and +5, you set α = 5. If they span −100 to +100, α = 100.

**Where α comes from:** You run calibration images through the model and observe what values each layer actually produces. The 99.9th percentile of the absolute values you see — large enough to cover almost everything, but not inflated by rare extreme outliers — becomes α for that layer. This is called *calibration*.

```
scale = α / 127                              (divides the range [-α, +α] into 254 slots)
x_q = clip(round(x / scale), -127, 127)     (map each value to its nearest slot)
x̂   = x_q × scale                           (convert back to approximate float)
```

**Concrete example:** Say α = 5, so scale = 5/127 ≈ 0.039. A value of 3.2 maps to slot round(3.2/0.039) = slot 82. Reconstructed: 82 × 0.039 ≈ 3.20 — nearly exact. A value of 7.0 is outside [−5, +5] and gets hard-clamped to slot 127 → 4.96. That's a clipping error of 2.04 — the entire excess is lost.

Every value in `[-α, +α]` maps to one of 256 integers. Values outside get **clipped** — hard-clamped to ±127. Values inside get **rounded** to the nearest grid point.

**Two sources of quantization error:**
1. **Clipping error** — values beyond ±α get clamped; error equals how far outside they were
2. **Rounding error** — values inside ±α get rounded; error ≤ α/127

**The clipping range α is the critical decision:**
- α too small → activations spill outside and get hard-clipped → large errors
- α too large → 127 grid points spread coarsely → wasted precision
- α just right → tight grid over the actual distribution → minimal error both ways

**For a static model (classifier, LLM):** calibrate α once on representative data, done.

**For a diffusion model:** the same network runs 20–50 times, at different noise levels each time. The activations at σ=1.0 (pure noise) are completely different from σ=0.09 (nearly clean image). A single α is wrong at almost every step.

---

## Slide 4: Quantization — The Basics

**Replace 16-bit floats with low-bit integers**

**Why quantize?**
- Model size: 4× smaller (FP16 → INT4)
- Memory bandwidth: fewer bytes to move → faster inference
- SD3 Medium: **4.17 GB** in FP16 → ~1.0 GB in W4

**How it works:**
```
scale = α / qmax                    (α = the largest value you expect — the edge of the range)
x_q = clip(round(x / scale), -qmax, qmax) × scale
```

**The clipping range α is everything:**
- Too small: activations outside `[-α, +α]` are hard-clipped → large errors
- Too large: integer grid is coarsely spaced → wasted precision

**For a static model (LLM, classifier):** calibrate α once, done.

**For a diffusion model:** the activations at σ=1.0 are completely different from σ=0.09. A single α is wrong at almost every step.

---

## Slide 5: Rectified Flow (SD3's Formulation)

**A cleaner, more direct path from noise to image**

Earlier diffusion models (DDPM) added and removed noise according to a complex cosine-shaped schedule — the amount of noise added at each step followed a non-linear curve, and the removal process used a stochastic sampler that introduced randomness at every step. It worked, but the math was complicated and the denoising path was winding.

**Rectified Flow** (used in SD3) replaces all of that with a straight line.

**The core idea — a weighted average of image and noise:**

Imagine you have a clean photo (call it x₀) and a blob of pure random noise (call it ε). Rectified flow defines every intermediate "noisy image" as simply a blend of the two:

```
x_t = (1 - t) · x₀  +  t · ε
```

- At t = 0: x_t = x₀ — pure clean image
- At t = 0.5: x_t = 0.5·x₀ + 0.5·ε — half image, half noise
- At t = 1.0: x_t = ε — pure noise

The noise level σ = t, so σ runs from 1.0 (pure noise) down to ~0.09 (nearly clean image) during a generation. Each denoising step moves the blend a little further toward the clean image — it's just subtracting a bit of noise each time, following a straight-line trajectory in pixel space.

**Compared to DDPM's cosine schedule:**

DDPM's noise schedule is non-linear — it adds noise rapidly at first, then slows down. This means the "distance" the model has to travel at each step varies a lot. Rectified flow's linear schedule means the model takes equal-sized steps from pure noise to clean image, making the trajectory predictable and smooth.

**Why this cooperates with polynomial clipping:**

Because the input to the network follows a perfectly straight line through noise levels, the network's internal activations also change smoothly and predictably as σ decreases. There are no sudden jumps or inflection points driven by the noise schedule itself — any curvature in the activation trajectories comes from the model's own learned behaviour, not from schedule artifacts. This is why low-degree polynomials (degree 2–3) are enough to fit those trajectories accurately. Under DDPM's cosine schedule, the trajectories would be more complex and harder to model compactly.

**One practical consequence:** the polynomial schedule works at any number of denoising steps — 20, 28, or 50 — without recalibration, because it's a continuous function of σ, not a lookup table tied to specific timestep indices.

---

## Slide 6: SD3 Architecture — MM-DiT

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

## Slide 7: SD3 — The Dual-Stream Problem

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

## Slide 8: Round-to-Nearest — The Baseline Problem

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

## Slide 9: AdaRound — Learned Rounding

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

## Slide 10: AdaRound's Limitations for Diffusion Models

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

## Slide 11: What We Discovered — 5 Failure Modes

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

## Slide 12: Key Observation — Activation Trajectories Are Smooth

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

## Slide 13: Our Proposal — Polynomial Clipping Schedule

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

## Slide 14: σ-Aware AdaRound

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

## Slide 15: σ-Weighted Loss

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

## Slide 16: Why This Is Novel

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

## Slide 17: Storage and Runtime Overhead

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

## Slide 18: Mixed-Precision — Surgically Keeping 4 Layers in FP16

**Not every layer should be quantized. Knowing which ones to skip is part of the method.**

**The problem: adaLN-induced distribution shift**

SD3's adaptive layer normalization injects a timestep-conditioned shift into activations before the FFN. Across all 188 traced activation points, the shift distribution is highly concentrated — 185 layers sit within ±0.5 units of zero, and 3 layers are dramatic outliers:

![Shift across all 188 layers](shift_all_layers.png)

The per-channel shift distribution reveals the true picture — not just the mean across channels but how many channels are severely displaced:

| Layer | Per-channel max | Channels > 100u | INT8 impact |
|-------|----------------|-----------------|-------------|
| mm22 txt mlp_fc2 | **356 units** | 744 / 6144 | Severe — half the grid wasted |
| mm21 txt mlp_fc2 | **144 units** | 26 / 6144 | Significant |
| mm20 txt mlp_fc2 | **46 units** | 0 / 6144 | Moderate |
| mm14 txt mlp_fc2 | **4 units** | 0 / 6144 | ✅ Clean — excluded in error |
| All other 185 layers | < 5 units | — | Negligible |

**mm14 was incorrectly flagged** in the original analysis. Its per-channel max shift is only 4 units — well within symmetric INT8 range. It has been removed from the exclusion list, recovering another ~13.5 MB.

Symmetric INT8 centers its grid at zero. A distribution centered at +254 means nearly all 256 quantization buckets are allocated to values that never appear. Rounding decisions become essentially random.

**Does the polynomial clipping schedule fix this?**

Partially. The polynomial correctly predicts the right α(σ) at each denoising step — so the clipping *width* is no longer wrong. But symmetric quantization always places the grid around zero, so with a distribution centred at +254, half the INT8 buckets still cover the empty negative side regardless of how accurate α(σ) is:

```
Polynomial gives α=254:  [-254 ........... 0 ........... +254]
mm22 activations:                          [+200 .... +308]
→ 127 of 256 buckets still wasted
```

**The complete fix: asymmetric shift polynomial**

We model a second polynomial, shift(σ), capturing how the distribution center drifts with σ due to adaLN. The quantization zero point then tracks the distribution:

```
Asymmetric: [shift(σ) − α(σ) .......... shift(σ) + α(σ)]
mm22:                         [+25 ......... +35]   ✓ all buckets used
```

The shift trajectory is smooth and polynomial-amenable — it rises during mid-denoising and falls near σ→0, following the same rectified flow physics as the clipping range:

![Shift trajectories for extreme layers](shift_trajectories.png)

| Layer | Cubic R² | Residual std | Verdict |
|-------|----------|--------------|---------|
| mm22 txt mlp_fc2 | 0.76 | 1.1 units | Feasible, ~97% correction |
| mm21 txt mlp_fc2 | 0.88 | 0.24 units | Good fit |
| mm20 txt mlp_fc2 | 0.86 | 0.24 units | Good fit |

The shift R² (0.76–0.88) is lower than for the clipping range (median 0.944) because the shift is more sensitive to prompt content than to pure noise-level physics. More calibration data improves it.

This is implemented (infrastructure ready) but not yet benchmarked. **FP16 exclusion is the conservative fallback** — costing only 2.8% of total savings — while asymmetric quant is validated.

Cost of exclusion: **40.5 MB** (3 × 13.5 MB savings forfeited)
Fraction of total potential savings forfeited: **2.1%** (3 out of 285 layers)
The rest of those blocks — attention projections, fc1, image stream — are still fully W4A8.

**Do other methods handle this?**

Yes — mixed-precision is a recognized pattern, though our trigger is more principled than most:

| Method | Approach | Granularity |
|--------|----------|-------------|
| **LLM.int8()** (Dettmers 2022) | Keep outlier *channels* in FP16, rest INT8 | Per-channel |
| **SpQR** (Dettmers 2023) | Keep top-k outlier *weights* in FP16 sparse format | Per-weight |
| **GPTQ sensitivity** | Skip layers whose perplexity degrades past a threshold | Per-layer |
| **Mixed-precision NAS** | Search for per-layer bit width by sensitivity | Per-layer |
| **Ours** | Keep layers where adaLN shift >50 units (interpretable cause) | Per-layer |

**What makes our approach different:**
- The exclusion criterion is *causal* — we know exactly why these layers are unquantizable (adaLN shift displaces the distribution center), not just that they are
- The threshold is diagnostic and interpretable, not a tuned hyperparameter
- Asymmetric quantization is a principled path to re-quantizing them without the 2.8% forfeit

---

## Slide 19: Per-Channel Activation Quantization — When It Matters

**Current design: one activation scale per layer (per-tensor)**

Each layer has a single scalar `a_scale` that controls the INT8 clipping range across all output channels. This is the standard and most hardware-friendly approach.

**The question:** Does channel-to-channel variance within a layer justify per-channel activation scales?

**Metric: channel-to-channel std / within-channel std**

For each layer we collect per-channel statistics across all calibration timesteps. Let `μ_d(t)` be the mean activation value for channel `d` at timestep `t`:

```
within-channel std  =  mean over all channels d of:  std_t[ μ_d(t) ]
                     → how much a single channel's mean drifts across timesteps

channel-to-channel std  =  std over channels d of:  mean_t[ μ_d(t) ]
                          → how different the time-averaged means are across channels
```

Concretely for a layer with hidden dimension D and T calibration timesteps:
1. Compute the D×T matrix of per-channel means
2. **Within-channel std**: average the row-wise standard deviations (each row = one channel over time)
3. **Ch-ch std**: take the std of the column-wise means (each column = one timestep, averaged; then std across channels)
4. **Ratio** = ch-ch std / within-channel std

**Interpretation:**
- **Ratio ≈ 1**: channels behave similarly — per-tensor quantization is fine
- **Ratio >> 1**: channels sit at systematically different offsets — a single scale clips some channels while wasting resolution on others; per-channel scales would help

A high ratio means channels live in systematically different regimes — per-tensor quantization wastes resolution.

![Per-channel vs within-channel variance across all 188 layers](per_channel_variance.png)

**Results across all 188 traced activation points:**

| Layer | ch-ch/within ratio | Max per-channel mean | Per-channel needed? |
|-------|-------------------|---------------------|---------------------|
| mm22 txt fc1_in | **18.8×** | 280 units | Yes — currently quantized |
| mm22 txt fc2_in | **15.0×** | 343 units | Yes — kept FP16 |
| mm21 txt fc1_in | **12.8×** | 122 units | Yes — currently quantized |
| mm21 txt fc2_in | **9.7×** | 135 units | Yes — kept FP16 |
| mm01 txt fc1_in | 6.6× | 18 units | Marginal |
| mm00 txt fc2_in | 6.0× | 26 units | Marginal |
| mm20 txt fc1/fc2_in | 3.1–3.3× | 42–43 units | Moderate |
| All other 182 layers | < 3× | < 9 units | No |

**Key finding:** The extreme inter-channel spread is not a widespread problem — it is almost entirely concentrated in the same mm22/21 adaLN-shift layers already identified for FP16 exclusion. The pattern is consistent: adaLN-induced distribution displacement breaks both the clipping width (fixed by α(σ)) and the assumption of channel homogeneity (the inter-channel spread).

**New concern: mm22/21 fc1_in are currently being quantized**

The fc1_in layers (inputs to the FFN) also show extreme inter-channel spread despite not being excluded. Per-channel max shifts of 280/122 units mean the same INT8 resolution waste applies to the layer *before* the already-excluded fc2. These should be added to the exclusion set.

---

**Critical distinction: two separate problems, two separate fixes**

The adaLN-affected layers suffer from two independent failure modes that are easy to conflate:

**Problem 1 — Temporal shift (intra-channel, over time):**
The whole distribution drifts as σ changes. Channel 744 might be at +320 at σ=1.0 and +356 at σ=0.1.
→ Fix: **asymmetric shift(σ) polynomial** — one time-varying zero-point per layer tracks the distribution center over denoising.

**Problem 2 — Spatial spread (inter-channel, across channels):**
Different channels permanently live at different offsets. Channel 0 has mean ≈ +5, channel 744 has mean ≈ +356 — regardless of timestep.
→ Fix: **per-channel zero-points** — each channel gets its own scale and zero-point.

**A single asymmetric zero-point does not fix Problem 2.** It can only pick one center for all channels. If channels span 0→356 units, the zero-point centers the average but channel 0 and channel 744 are still both poorly served.

| Approach | Fixes temporal shift (Problem 1) | Fixes inter-channel spread (Problem 2) |
|----------|----------------------------------|----------------------------------------|
| Symmetric per-tensor (current) | No | No |
| Asymmetric per-tensor + shift(σ) | Yes | No |
| Per-channel symmetric | No | Yes |
| Per-channel asymmetric | Yes | Yes |
| FP16 exclusion | — (no quant) | — (no quant) |

The asymmetric quant work in Slide 17 addresses Problem 1 for layers we *do* quantize. It will not rescue mm22/21 from Problem 2 — those layers need per-channel treatment or FP16 exclusion.

**Per-channel activation quant is hardware-demanding** — most accelerators (including Apple Neural Engine) only support per-tensor activation scales. The polynomial overhead also scales from 3 coefficients/layer to 3D coefficients/layer. For 5 extreme layers costing 3.7% of total savings, FP16 exclusion is the pragmatic choice.

**Per-tensor vs Per-channel comparison:**

| Property | Per-tensor | Per-channel |
|----------|-----------|-------------|
| Storage overhead | 1 scalar/layer | D scalars/layer (D = hidden dim) |
| Runtime overhead | 1 multiply | channel-wise multiply |
| Hardware support | Universal | Limited (not all accelerators) |
| Polynomial overhead | 3 coefficients/layer | 3×D coefficients/layer |
| Fixes Problem 1 (temporal) | No (needs asymmetric) | No (needs asymmetric) |
| Fixes Problem 2 (inter-channel) | No | Yes |
| Needed for most layers | — | No (ratio < 3 for 182/188 layers) |

**Recommended path:**
- Keep per-tensor for all 182 normal layers (ratio < 3)
- Add mm22/21 fc1_in to the FP16 exclusion set (+2 layers, +27 MB forfeit, total 67.5 MB / 3.7%)
- Apply asymmetric shift(σ) polynomials to mm20 and other layers that remain quantized (fixes Problem 1 only)

---

## Slide 20: Results — Schedule Generalization

**Do the polynomial coefficients generalize across different calibration sets?**

We fitted schedules on two independent calibration groups and compared:

- **285 layers** evaluated
- **Median NRMSE: 0.98%** normalized error across all layers
- **95th percentile NRMSE: 4.43%**
- **98% of layers generalize with < 1% error**

The 4 extreme-shift layers (mm14, mm20, mm21, mm22 txt mlp_fc2) are the exceptions — their 60–254 unit shifts make them inherently harder to fit and they are kept in FP16.

**Takeaway:** The polynomial trajectories are a real property of the model's physics, not overfit to specific calibration images.

---

## Slide 21: What's Still To Do

**Current status and next steps**

| Component | Status |
|-----------|--------|
| Activation statistics collection | ✅ Done (188 layers × 30 timesteps) |
| Polynomial schedule generation | ✅ Done (227 static, 57 quadratic, 1 cubic) |
| σ-aware AdaRound with polynomial | ✅ Implemented |
| σ-weighted loss | ✅ Implemented |
| 3 extreme fc2 layers FP16 exclusion | ✅ Implemented |
| mm22/21 fc1_in per-channel issue | ⚠️ Newly identified — needs exclusion or per-channel scales |
| Derivative-weighted loss (`\|dα/dσ\|`) | ✅ Implemented |
| Asymmetric quant — shift polynomials for adaLN layers | 🔧 Infrastructure ready, not benchmarked (would recover the 2.8%) |
| End-to-end FID evaluation | ⏳ In progress |
| Comparison vs. TaQ-DiT / HTG baseline | ⏳ Pending |

**The key open question:**
How much does polynomial clipping improve FID and PSNR vs. standard AdaRound and TaQ-DiT on SD3 W4A8?

---

## Slide 22: Summary

**What we built:**

1. **Characterized** 5 failure modes for quantizing SD3's dual-stream MM-DiT — none fully addressed by prior work

2. **Discovered** that activation scale trajectories follow smooth, low-degree polynomial curves (rectified flow physics cooperates)

3. **Built** a tiered polynomial clipping schedule: 402 coefficients, 10 KB, median R²=0.944, generalizes across calibration sets with 0.98% median error

4. **Extended** AdaRound to be σ-aware: each layer uses its own per-sample α(σ) within a block-level reconstruction loss, joint optimization across the full denoising trajectory, with σ-weighted and derivative-weighted loss (`|dα/dσ|`) focusing gradients on perceptually critical and rapidly-changing timesteps

5. **Modeled** image and text streams independently, the first approach that handles opposite-direction activation trajectories in dual-stream transformers

**The central claim:**
> Polynomial clipping removes activation clipping error from AdaRound's reconstruction loss, allowing rounding decisions to be optimized against the right signal. σ-weighted loss then directs that optimization toward perceptually important timesteps. Together these address the core limitation of applying AdaRound to iterative generative models.
