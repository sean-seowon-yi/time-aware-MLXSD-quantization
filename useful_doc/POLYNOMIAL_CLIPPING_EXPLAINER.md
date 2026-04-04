# Polynomial Clipping for MM-DiT Quantization

> **Scope:** Standalone notes on polynomial activation clipping for diffusion transformers. The **main quantization pipeline** in this repository is **PTQ4DiT-style CSB + SSC + W4A8** (`src/phase2/`). Polynomial clipping is not the implemented path here.

Quantizing diffusion transformers is hard because the optimal clipping range for each layer changes at every denoising step — and in dual-stream architectures like SD3's MM-DiT, two streams drift in *opposite directions* simultaneously. We discovered that these activation trajectories aren't random: they follow smooth, low-degree polynomial curves dictated by rectified flow physics. By fitting per-layer polynomials to empirical trajectories, we get correct activation clipping at every noise level — and this unlocks everything downstream:

- **Clean AdaRound optimization** — the reconstruction loss reflects only rounding error (not clipping error), so weight rounding decisions are optimized against the right signal
- **σ-weighted loss** — prioritize perceptually important timesteps, which only works when the clipping is already correct at each σ
- **Derivative-based weighting** — the polynomial derivative dα/dσ tells you where the clipping range is most sensitive, focusing optimization effort where it matters most
- **Asymmetric quantization** — fit shift polynomials to distribution centers (not just scale), potentially bringing the 4 extreme-shift FP16 layers back to W4
- **Scheduler independence** — switch from 28 to 50 inference steps or swap schedulers entirely without recalibrating, since the polynomial evaluates at any σ
- **Interpretable diagnostics** — the coefficients themselves reveal each layer's behavior (rising, falling, U-shaped), directly informing mixed-precision decisions without rerunning analysis

---

## 1. The Problem

When you quantize a neural network — replacing 16-bit floating-point weights and activations with 4-bit or 8-bit integers — you need to decide on a **clipping range**: the interval `[-α, +α]` that maps to the integer grid. Values inside the range get quantized with good precision. Values outside get clipped, introducing error.

For a vision classifier or language model, picking `α` is straightforward. You run calibration data through the model, observe the activation ranges, and set `α` once per layer. The ranges don't change at inference time.

**Diffusion models break this assumption.** A diffusion model runs the same network 20–50 times per image, each time at a different noise level `σ`. At `σ = 1.0` (pure noise), the activations have one distribution. At `σ = 0.09` (nearly clean image), they have a completely different distribution. A clipping range calibrated at one noise level is wrong at every other noise level.

SD3's **MM-DiT** (Multi-Modal Diffusion Transformer) makes things worse. It processes image tokens and text tokens through a shared transformer with **joint attention** — the two token streams are concatenated into a single tensor for the Q/K/V projections. If the image stream has scale 3.0 and the text stream has scale 1.0 in the same tensor, any single `α` is a bad compromise: too wide wastes precision on the text tokens, too narrow clips the image tokens.

---

## 2. What Everyone Else Does

### Static calibration (baseline PTQ)
Run calibration images at a representative noise level, collect activation statistics, set one `α` per layer. Simple, but the `α` is wrong at most timesteps. This is what standard AdaRound/GPTQ do.

### TaQ-DiT
The closest prior work to ours. TaQ-DiT collects calibration data across 25 timesteps and runs a single **joint reconstruction** of weights and activations (not separate AdaRound — they found separate reconstruction has convergence issues). Their key insight is that Post-GELU activations are the most quantization-sensitive layers, and they address temporal variability with **static transformations**: momentum-based shifting to recenter activation distributions, and reconstruction-driven migration to handle channel-wise outliers. These are precomputed during calibration and remain fixed at inference — no per-timestep adaptation.

What TaQ-DiT doesn't do: it uses one set of quantization parameters across all timesteps. The static shifting helps, but the shift values are averaged across σ via momentum (β = 0.95). For layers where the shift changes dramatically with σ (up to 254 units for mm22 txt mlp_fc2), a single static shift is still a compromise.

### HTG (Hierarchical Timestep Grouping)
Partitions the denoising schedule into 2–4 "buckets" (e.g., early/mid/late), calibrating separate quantization parameters per bucket. Better than fully static, but:
- **Bucket boundaries are heuristic.** HTG boundaries were derived for DDPM's cosine noise schedule, not rectified flow's linear `σ(t) = 1 - t`. The optimal split points differ.
- **Uniform σ-spacing is suboptimal.** If the activation scale has a U-shaped trajectory, uniform buckets miss the inflection point.
- **Buckets are independent.** No model of how activations evolve between buckets — discontinuous jumps at boundaries.

### PTQ4DiT / Q-DiT
Apply post-training quantization to single-stream DiT models (DiT-XL, U-ViT). They characterize per-layer distributions but have no concept of two coupled streams with different scale ratios. The cross-stream mismatch problem doesn't exist in their architecture.

### The common limitation
No prior work models the *continuous relationship* between noise level and activation statistics. TaQ-DiT averages across timesteps via momentum. HTG discretizes into buckets. PTQ4DiT/Q-DiT ignore the issue on simpler architectures.

---

## 3. What We Found

We collected activation statistics from 30 calibration images across 25 noise levels (`σ = 1.00 → 0.09`), covering all 285 layers across SD3's 24 dual-stream MM-DiT blocks. The analysis revealed **five distinct failure modes** for static quantization:

### Failure Mode 1: Cross-Stream Scale Mismatch
The image and text streams have systematically different activation scales within the same joint attention tensor. Block mm3 has an img/txt ratio of 2.2x; block mm13 is txt-dominant. **This mismatch worsens during denoising** — mm3 goes from 1.65x at `σ=1.0` to 3.58x at `σ=0.09`. The mismatch is worst exactly when quantization errors are most perceptually visible (fine-detail generation at low `σ`).

### Failure Mode 2: adaLN-Induced Distribution Shift
SD3 uses adaptive layer normalization (adaLN) to condition on the noise level and text embeddings. This shifts the *center* of activation distributions by enormous amounts:
- `mm22.txt.mlp.fc2`: shift of **254 units**
- `mm21.txt.mlp.fc2`: shift of **124 units**
- `mm14.txt.mlp.fc2`: shift of **108 units**

For comparison, typical activation scales are 1–5 units. A shift of 254 units means AdaRound's rounding decisions, optimized on the calibration distribution, are invalid at most other timesteps.

### Failure Mode 3: Non-Linear Rectified Flow Drift
Activation scales don't change linearly with noise level. They follow curves with inflection points, U-shapes, and hills. We quantified this by fitting both linear and quadratic models:

| Layer | Linear R² | Quadratic R² |
|-------|-----------|--------------|
| mm0 img mlp_fc2 | 0.63 | 0.94 |
| mm9 img mlp_fc2 | 0.14 | 0.71 |
| mm18 img mlp_fc2 | 0.71 | 0.91 |
| mm0 txt mlp_fc2 | 0.80 | 0.90 |
| mm18 txt mlp_fc2 | 0.57 | 0.85 |

The most striking case: mm9 has a linear R² of just 0.14 (essentially no linear relationship) but a quadratic R² of 0.71 — the trajectory is U-shaped, and a linear model completely misses it.

A key observation: these non-linear trajectories arise even though rectified flow uses a *linear* interpolation `σ(t) = 1-t` — the simplest possible noise schedule. Prior work on DDPM-based DiT hasn't characterized trajectory shapes this way, but DDPM's cosine schedule is itself non-linear, so activation trajectories would likely be even more complex. Rectified flow gives us the simplest possible input trajectory, and we still need degree-2 polynomials. This suggests that polynomial clipping is particularly well-suited to rectified flow models — the physics cooperates with low-degree fits in a way that other noise schedules may not.

### Failure Mode 4: Per-Channel Outlier Asymmetry
Both streams have heavy-tailed channel distributions (a few channels with 10–50x the median absmax), but the *profiles* differ between streams. Block mm18's image stream has 2–3 extreme outlier channels then a flat floor; its text stream is more uniform. A single SmoothQuant scaling vector applied to the concatenated tensor helps one stream while hurting the other.

### Failure Mode 5: Opposite Trajectory Directions
The image attention projections (Q/K/V) **rise** as denoising progresses (scale increases as `σ` decreases). The text attention projections **decline** — they peak at high `σ` and fall toward the clean image. The two streams move in *opposite directions simultaneously*. This means the img/txt scale ratio isn't just varying — it's accelerating apart. Any timestep-aware scheme must account for both streams independently.

---

## 4. Our Approach: Polynomial Clipping

### The Intuition

If you plot any layer's activation scale against noise level `σ`, you get a smooth curve. Not random. Not noisy. A smooth, reproducible trajectory.

This isn't a coincidence — it's physics. Rectified flow defines a linear interpolation path between noise and data: `x_t = (1-t)·x_0 + t·ε`. The network's internal activations are smooth functions of this interpolant. Smooth input trajectories produce smooth activation trajectories. And smooth curves can be described by low-degree polynomials.

Instead of storing a clipping range `α` for every layer at every timestep, we fit a polynomial `α(σ) = c₀ + c₁σ + c₂σ² + ...` to each layer's trajectory. At inference time, we evaluate the polynomial at the current `σ` to get the optimal clipping range — no lookup tables, no bucket boundaries, no discontinuities.

### The Technical Details

**Data collection.** We collect the p99.9 percentile of activation magnitudes for each of the 285 layers at each of the 25 `σ` steps, averaged across 30 calibration images. This gives us 285 trajectories, each with 25 data points.

**Tiered degree selection.** Not every layer needs a polynomial — many are nearly constant. We use a tiered fitting strategy:

| Condition | Degree | Meaning |
|-----------|--------|---------|
| CV < 0.10 | 0 (static) | Activation barely changes — one constant `α` suffices |
| Quadratic R² > 0.85 | 2 | Smooth parabolic trajectory |
| Cubic R² gain > 0.15 over quad | 3 | Cubic captures significant additional structure |
| Quartic R² gain > 0.10 over cubic | 4 | Reserved for rare complex trajectories |

**Per-stream fitting.** Crucially, we fit separate polynomials for the image and text streams. This directly addresses Failure Mode 5 (opposite trajectories) — the image stream's rising polynomial and the text stream's declining polynomial are captured independently.

**How the polynomial connects to AdaRound weight optimization.** This is the part that's easy to miss, because AdaRound optimizes *weights* and the polynomial models *activations*. Here's how they connect:

AdaRound's job is to decide, for each weight element, whether to round up or round down. It makes this decision by minimizing **reconstruction error**: the difference between the layer's FP16 output and its quantized output on calibration data. To compute the quantized output, you need to quantize both the weights *and* the activations in the forward pass. The activation quantization needs a clipping range α.

Here's what happens with and without the polynomial:

- **Without polynomial**: A calibration sample from σ=0.1 arrives. Its activations are large (late denoising), but they get quantized with a static α calibrated at some average σ. The activations are *clipped* — the quantized output is wrong not because of rounding, but because the clipping range was wrong. AdaRound sees a huge reconstruction error and contorts its rounding decisions trying to compensate for a *clipping* problem it fundamentally can't fix. The rounding optimization is fighting the wrong battle.

- **With polynomial**: The same sample arrives. The polynomial evaluates α(0.1), giving the correct clipping range for this σ. The activations are quantized cleanly — no clipping error. Now the reconstruction error that AdaRound sees is *only* due to weight rounding, which is exactly what it can optimize. The rounding decisions are focused on the real problem.

In short: the polynomial doesn't change what AdaRound optimizes (weight rounding). It removes a confounding error source (wrong activation clipping) so that AdaRound can do its actual job effectively.

**σ-weighted loss.** On top of correct per-σ clipping, we weight the AdaRound loss per sample so that not all noise levels contribute equally to the rounding decisions. This only works *because* each sample already has its σ-correct α — without per-σ clipping, any weighting scheme would just emphasize samples whose α is wrong, accomplishing little.

We currently use `w(σ) = 1/(σ + offset)` (perceptual weighting), but the polynomial opens up a family of weighting strategies. Here are the options, roughly ordered by how much additional benefit they might provide:

| Strategy | Formula | What it emphasizes | Polynomial-derived? | Expected benefit |
|----------|---------|-------------------|-------------------|-----------------|
| **Uniform** (baseline) | `w = 1` | All σ equally | No | Baseline — no σ preference |
| **Perceptual** (current) | `w = 1/(σ + offset)` | Low-σ (fine detail) | No | Moderate — prioritizes where errors are visible |
| **Trajectory sensitivity** | `w = \|dα/dσ\|` | Where α changes fastest | Yes — derivative of polynomial | Potentially high — targets fragile regions |
| **SNR-based** | `w = α²/σ²` | High signal-to-noise ratio | Partially — uses α(σ) | Theoretically grounded but harder to tune |
| **Reconstruction-error** | `w = loss_prev(σ)` | Hardest samples | No — adaptive | High, but adds complexity (two-pass) |
| **Perceptual × sensitivity** | `w = \|dα/dσ\| / (σ + offset)` | Both fragile and visible | Yes | Likely highest — compounds both signals |

The two polynomial-derived strategies are worth highlighting:

**Trajectory sensitivity (`|dα/dσ|`).** For a degree-2 polynomial `α(σ) = c₀ + c₁σ + c₂σ²`, the derivative is just `c₁ + 2c₂σ` — trivial to compute. This weight is large where the clipping range is changing rapidly with σ, meaning a small error in the polynomial fit or a small shift in the activation distribution causes a large change in α. These are the timesteps where getting the rounding decisions right matters most, because the quantization grid is most sensitive to perturbation. For the 227 static layers (degree 0), dα/dσ = 0 everywhere, so this naturally assigns uniform weight to layers that don't need special attention.

**Combined perceptual × sensitivity.** Multiply the trajectory derivative by the perceptual weight: `w(σ) = |dα/dσ| / (σ + offset)`. This focuses the optimizer on timesteps that are *both* perceptually important (low σ, fine detail) *and* where the clipping range is fragile (high derivative). A timestep at σ = 0.1 where α is changing fast gets maximum weight; a timestep at σ = 0.9 where α is flat gets minimal weight. This is the strategy most likely to outperform simple perceptual weighting, because it concentrates optimization effort exactly where quantization is both hardest and most consequential.

Both strategies fall directly out of the polynomial representation — a lookup table would require finite-difference approximations for the derivative, which adds noise. The polynomial gives an exact, smooth derivative for free.

**Status:** These weighting strategies are proposed, not yet implemented or benchmarked. We currently use the simple perceptual weighting `1/(σ + offset)`. The derivative-based strategies are a natural next step enabled by the polynomial, but their benefit over perceptual weighting is unproven. To our knowledge, no prior work uses activation trajectory derivatives to weight quantization optimization — this would be novel if it works.

### The Result: Degree Distribution

Of 285 layers in SD3's MM-DiT:

| Degree | Count | Fraction |
|--------|-------|----------|
| 0 (static) | 227 | 79.6% |
| 2 (quadratic) | 57 | 20.0% |
| 3 (cubic) | 1 | 0.4% |

**80% of layers don't need a polynomial at all** — their activations are stable enough that a single constant works. Only 58 layers (20%) have meaningful temporal drift requiring degree-2 or higher fits. This is both a storage advantage and a validation: most of the network behaves well, and our method surgically targets the layers that don't.

### Fit Quality

For the 57 quadratic fits:
- **Median R² = 0.944** — the polynomial captures 94.4% of the variance
- **Mean R² = 0.925**
- **Minimum R² = 0.711** (worst-case still captures the dominant trend)

---

## 5. Why This Is New

### The idea: continuous σ-dependent clipping

The core novelty isn't the polynomial — it's the framing. Prior work treats quantization calibration as a *measurement* problem: observe activation ranges, store them, look them up at inference. We treat it as a *modeling* problem: activation ranges are a predictable function of noise level, so model that function and evaluate it on the fly.

This sounds simple, but nobody does it. Here's why:

| Prior Approach | How it thinks about σ → α |
|----------------|--------------------------|
| **Static calibration** (AdaRound, GPTQ) | "Activations are stationary" — one α, period |
| **TaQ-DiT** | "Average across timesteps with momentum" — static compromise |
| **HTG** | "Which timesteps should share parameters?" — discrete buckets |
| **PTQ4DiT / Q-DiT** | Single-stream DiT with DDPM — drift is milder, no opposite-trajectory problem |

Everyone either ignores the σ-dependence, averages across it, or discretizes it into buckets. Nobody asks: "what is α *as a function of* σ, and can we model it directly?"

Could you get the same per-σ clipping quality with a dense lookup table and interpolation? Yes — the polynomial is one implementation of continuous clipping, not the only one. But the polynomial has a useful property: it acts as a **structural prior**. A lookup table memorizes 25 data points. A degree-2 polynomial with R² = 0.94 tells you the trajectory is a smooth parabola — it filters out calibration noise and makes the layer's behavior legible. You can read the coefficients and immediately see whether a layer's scale rises, falls, or curves.

### σ-aware AdaRound: a new way to optimize quantized weights

Separate from the clipping schedule itself, the trajectory model enables something nobody has done with AdaRound: **varying the clipping range per calibration sample based on its noise level during weight optimization.**

Standard AdaRound (Nagel et al., 2020) assumes stationary activations — one α, one calibration distribution. PTQ4DiT applies AdaRound to diffusion models but keeps the same static-α assumption. TaQ-DiT replaces AdaRound with joint reconstruction and uses momentum-averaged static shifts — better, but the shift values are still a compromise across all σ. HTG runs separate optimizations per timestep bucket, each with its own fixed α.

Our approach: a **single optimization** where every calibration sample carries its σ, and the clipping range is evaluated from the trajectory model at that σ. The optimizer sees the full diversity of noise levels in one pass, with each sample quantized under its own correct α. This means the rounding decisions are jointly optimized across the entire denoising trajectory — not averaged via momentum (TaQ-DiT) or split into independent buckets (HTG).

On top of this, we apply σ-dependent loss weighting (see Section 4 for the full menu of strategies) so that not all noise levels contribute equally to the rounding decisions.

Any weighting strategy — whether perceptual, derivative-based, or combined — has a prerequisite: **the activation clipping must already be correct at each σ.** Here's why. The AdaRound loss for a given sample is the reconstruction error between the FP16 output and the quantized output. If the clipping range α is wrong for that sample's σ, the reconstruction error is dominated by clipping error — the loss is huge regardless of how the weights are rounded. If you then *weight* that sample heavily (because it's at a perceptually important σ), you're telling the optimizer to focus on a sample whose error it can't fix. The optimizer contorts the rounding decisions trying to compensate for a clipping problem, making other samples worse in the process.

With per-σ clipping, the clipping error is removed. The reconstruction error each sample contributes is purely due to rounding — which is what AdaRound controls. Now weighting actually does what you intend: it tells the optimizer which σ regions to prioritize when making rounding trade-offs. Per-σ clipping is the foundation that makes any σ-weighted strategy viable.

### What's specific to MM-DiT

The continuous clipping idea and σ-aware AdaRound are general techniques, but two aspects are unique to dual-stream architectures:

1. **Per-stream modeling is required, not optional.** Image attention projections rise during denoising while text projections fall (Section 3, Failure Mode 5). A single α(σ) per layer can't capture opposite directions — you need separate functions per stream. This problem doesn't exist in single-stream DiT models.

2. **Rectified flow makes the modeling easy.** Rectified flow defines a linear interpolation `x_t = (1-t)·x_0 + t·ε` between data and noise. Smooth input trajectories produce smooth activation trajectories, which means low-degree polynomials (degree 2–3) are sufficient to model the clipping range across the *entire* denoising trajectory — all 20, 28, 50, or however many steps the scheduler uses. Each polynomial is a continuous function over the full σ range [0.09, 1.0]; there are no per-step values, just evaluate at the current σ. Under DDPM's cosine schedule, the trajectories would be more complex and harder to fit compactly. The physics of the sampler is what makes 402 coefficients enough for 285 layers across all steps.

---

## 6. The Evidence

### The trajectories are modelable
- 285 layers fitted with tiered degree selection
- 227 layers (80%) are stable enough that a static α works — continuous clipping isn't even needed for most of the network
- The remaining 58 layers have meaningful σ-dependent drift. Quadratic polynomials fit them with **median R² = 0.944** — the trajectory is genuinely smooth, not noisy
- The worst-case quadratic R² is 0.711 (mm9 img mlp_fc2, a U-shaped trajectory). Compare to linear R² = 0.14 for the same layer — the curvature is real and a linear model misses it entirely

### The numbers from the analysis

**Cross-stream divergence grows during denoising:**
- mm3 Q-projection ratio: 1.65x at `σ=1.0` → 3.58x at `σ=0.09`
- 34.8% of (block, σ) pairs have txt stream > 1.5x larger than img stream

**adaLN shift magnitudes (the extreme layers):**
- 4 layers out of 285 (1.4%) have shifts of 72–254 units (vs. typical scales of 1–5)
- These are specifically txt mlp_fc2 in blocks mm14, mm20, mm21, mm22
- Remaining layers have shifts < 25 — manageable with standard calibration

**Temporal variability is layer-type-specific:**
- mlp_fc1: CV < 0.10 everywhere — safe for static calibration
- Late-block img mlp_fc2: CV up to 0.33 — needs timestep-aware quantization
- The two streams don't track each other: img peaks at end (mm23), txt peaks at mm20

---

## 7. What This Means in Practice

### Better clipping, less quantization error
This is the core benefit. Continuous σ-dependent clipping (whether implemented as a polynomial, a lookup table, or anything else) gives each layer the correct `α` at each `σ`, which directly reduces quantization error in two ways:

1. **Less clipping.** A static `α` calibrated at one noise level is too small at others — activations that exceed it get hard-clipped, introducing large errors. The polynomial tracks the actual scale, so clipping is minimized at every timestep.
2. **Less wasted precision.** A static `α` set conservatively to avoid clipping is too *wide* at quieter timesteps — the 8-bit integer grid covers a range that's half-empty, wasting effective resolution. The polynomial tightens the range when the activations are smaller, recovering precision.

### Better rounding decisions, less weight error
Clipping range is only half the story. The other half is how weights are *rounded* to the integer grid — round up or round down for each weight element. AdaRound optimizes these rounding decisions to minimize reconstruction error on calibration data.

Standard AdaRound optimizes against one activation distribution. If that distribution is from `σ = 0.5` but inference also runs at `σ = 0.1` and `σ = 0.9`, the rounding decisions are tuned for the wrong distribution at most timesteps. Our σ-aware AdaRound sees calibration samples from across the full σ range, each with its correct clipping α, so the rounding decisions minimize error *across the entire denoising trajectory* in a single pass.

The σ-weighted loss goes further: since quantization errors at low σ (near-clean images) are more perceptually damaging than errors at high σ (mostly noise), we weight the loss by `1/(σ + 1)` so the optimizer prioritizes getting the fine-detail timesteps right. The result is rounding decisions that are jointly optimized for correct clipping *and* perceptual quality — not just minimal reconstruction error at an arbitrary calibration point.

### Overhead
The schedule is 402 coefficients for 285 layers. At each denoising step, evaluating a degree-2 polynomial is 2 multiplies and 2 adds per layer — negligible next to the matrix multiplications in the transformer.

### Better integer bucket utilization via asymmetric quantization
Symmetric quantization maps activations to `[-α, +α]` — centered on zero. But for layers with large adaLN-induced shift (up to 254 units in the extreme case), the actual activation distribution isn't centered on zero. It's centered somewhere that moves with σ. A symmetric grid wastes integer buckets on empty space on one side while cramming the real distribution into the other side.

Asymmetric quantization shifts the quantization window to `[center - α, center + α]`, where `center(σ)` is modeled by a second polynomial fitted to the shift trajectory. Now the integer grid is centered on the actual distribution at every σ, and all 256 buckets (for INT8) cover values that actually occur. This directly recovers effective precision — the same 8-bit grid resolves finer differences because it isn't wasting half its range on empty space.

The shift trajectories are already implemented in the schedule generator (`--include-shifts` flag) and follow the same tiered fitting as the scale trajectories. For the 4 extreme-shift layers (mm14, mm20–mm22 txt mlp_fc2) that currently require W8 or FP16 fallback, asymmetric quantization with shift polynomials could bring them back to W4 — eliminating the only mixed-precision exception in the network.
