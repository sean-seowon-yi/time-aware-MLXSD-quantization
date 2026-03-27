# AdaRound W4A8 Research Log

A chronological record of experiments, decisions, and outcomes for SD3 MM-DiT W4A8 quantization.
Deep-dive reasoning lives in the linked explainer docs; this log captures *what happened and why*.

---

## Current State (as of 2026-03-27)

**Weight quantization:** AdaRound W4 per-channel, group_size=64 (currently running).
Group size chosen after bucket utilization analysis showed per-row scales had 1.87–2.03 bits
entropy on mm20–22 txt mlp_fc1 layers (55–64% fill). gs=64 gained +0.447 bits entropy
with acceptable 1.8% saturation rate. SmoothQuant abandoned — incompatible with W4.

**Activation quantization:** Per-tensor A8 LSQ with p100 absmax polynomial clipping schedule
(`polynomial_clipping_schedule_512_p100.json`). Polynomial fitted per-layer to absmax
trajectories across 15 σ steps, degrees 0–4 selected by tiered R² criteria. p999 was the
previous default but caused severe clipping error during AdaRound training.

**Active experiment:** `quantized_weights_w4a8_adaround_poly_p100_group64` — AdaRound with
`--group-size 64` and p100 poly schedule. 3000 iters, batch size 8, calibration_data_512.

**Next:** Evaluate group_size=64 image quality vs per-row baseline. If gap remains, consider
sigma-weighted loss (now that activation clipping is correctly handled, the signal is clean).

---

## Experiment Log

---

### 2026-02-23 — Project start: DiffusionKit baseline

**Hypothesis:** SD3 MM-DiT can be quantized W4A8 with AdaRound following the TaQ-DiT approach.

**Experiment:** Established baseline FP16 inference pipeline for SD3-Medium.
Initial codebase uploaded from DiffusionKit.

**Result:** FP16 baseline working. Identified key difference from TaQ-DiT: they use joint
reconstruction (weights + activations together) with momentum-based shifting. We use separate
AdaRound (weights only) with learned per-layer activation scale (LSQ).

**Decision:** Build on AdaRound + LSQ rather than replicating TaQ-DiT joint reconstruction.
Focus on the activation temporal variability problem TaQ-DiT addressed with momentum averaging.

---

### 2026-03-13 — AdaRound optimizer working; float32 stability fixes

**Hypothesis:** Block-level AdaRound reconstruction with fake-quantized activations will
converge with bfloat16 computation.

**Experiment:** First full AdaRound runs on calibration_data_100. Hit NaN gradients in backward
pass through attention and RMSNorm with bfloat16.

**Result:** NaN gradients in bfloat16 backward pass — Q/K near-zero → RMSNorm grad explosion.
Also hit alpha gradient instability and ~25% NaN rate from fallback-embedding samples.

**Decision:** Force float32 for all block forward+backward in `block_loss_fn`. Add NaN guard
and alpha gradient clipping. Exclude fallback-embedding samples.
See: memory `feedback_float32_backward.md`

---

### 2026-03-13–14 — Module B (FP16 exclusion) and Module C (asymmetric quant)

**Hypothesis:** The 4 extreme adaLN-shift txt mlp_fc2 layers (mm14, mm20, mm21, mm22)
with shifts of 108–254 units are too difficult to quantize; keeping them FP16 will
improve quality at low model size cost.

**Experiment:** Added `--exclude-extreme-shift` flag to skip those 4 layers. Also added
asymmetric activation quantization (`--asymmetric-act`) using σ-dependent shift polynomials.

**Result:** Exclusion works but reduces compression ratio. Asymmetric quant added as Module C.

**Decision:** Keep both as options. p999 polynomial schedule still in use at this point.

---

### 2026-03-16–17 — Polynomial clipping schedule: p999 baseline

**Hypothesis:** Static activation scales (one α per layer) are wrong at most timesteps.
Fitting per-layer polynomials α(σ) to p99.9 percentile trajectories gives correct clipping
at every noise level, enabling AdaRound to optimize rounding without fighting clipping error.

**Experiment:** Built full pipeline: `build_activation_stats.py` → `generate_poly_schedule.py`
(p999 default) → `adaround_optimize.py --poly-schedule`. Fitted degree 0–4 polynomials to
p99.9 trajectories across 25 σ steps from calibration_data_100 (30 images).

**Result:** Polynomial clipping dramatically improved vs static scale. Images recognizable.
But some layers still showed significant quantization error, particularly mm20–22 txt layers.

**Decision:** p999 as initial default. Begin investigating whether p99.9 is the right target.
See: `POLYNOMIAL_CLIPPING_EXPLAINER.md`

---

### 2026-03-18 — p100 absmax schedule; p999 vs p100 analysis

**Hypothesis:** p99.9 clips too aggressively for AdaRound training. The top 0.1% of activations
that p999 discards are present in the forward pass during training, causing reconstruction
error that AdaRound cannot fix through rounding — it's a clipping problem, not a rounding problem.

**Experiment:** Generated p100 absmax polynomial schedule (`polynomial_clipping_schedule_512_p100.json`)
from calibration_data_512 (50 images, 15 σ steps). Compared degree distribution:
p999 → 80% degree-0 layers (flat fits because clipped maxima are stable).
p100 → all 285 layers degree ≥ 2 (true-maximum trajectories have natural curvature).

**Result:** p100 schedule gave noticeably better AdaRound training signal. Activation
trajectory plots (later generated) confirmed: mm20–22 txt layers had absmax/p999 ratio
well above 1.5× — p999 was underestimating the true range by 1.5–3× at low σ.

**Decision:** Switch default from p999 to p100_absmax in `generate_poly_schedule.py`.
See: `POLYNOMIAL_CLIPPING_EXPLAINER.md` § "Choosing the clipping percentile"

---

### 2026-03-19–22 — Sigma-weighted and derivative-weighted AdaRound loss

**Hypothesis:** Not all timesteps contribute equally to perceptual image quality. Weighting
AdaRound reconstruction loss by 1/(σ+ε) prioritizes low-σ (fine detail) timesteps. Weighting
by |dα/dσ| (polynomial derivative) prioritizes timesteps where the clipping range is most
sensitive, focusing rounding optimization where it matters most.

**Experiment:** Added `--sigma-weighted` and sigma-weight offset flags. Implemented
derivative-weighted loss using `np.polyder` on fitted polynomial coefficients. Multiple
runs with different offset values (0.1, 1.0) and iteration counts (2000, 3000).

**Result:** No clear image quality improvement over uniform weighting observed in visual
comparison. Metrics (PSNR, LPIPS, CLIP similarity) added to benchmark pipeline but
differences within noise.

**Decision:** Weighting left as available options but not the primary focus. Hypothesis:
the remaining quality gap is in weight quantization precision, not loss weighting.

---

### 2026-03-25–26 — SmoothQuant investigation

**Hypothesis:** Per-channel activation outliers (especially mm20–22 txt streams with
adaLN-induced shifts of 108–254 units) are causing activation quantization errors even
with p100 poly schedule. SmoothQuant (α=0.5) can migrate outlier magnitude into weights,
flattening activation trajectories and making A8 quantization easier.

**Experiment:** Path A (RTN re-quant) and Path B (full AdaRound re-training) implemented.
Ran SQ with α=0.5, scale_clip=32 on calibration_data_512 activations + p100 weights.
Generated smoothed poly schedule (277/285 layers became degree-0 — nearly flat after SQ).

**Result:** Complete image collapse — no coherent structure in output. Far worse than
per-row W4 without SQ. SQ Path B (re-trained AdaRound from FP16 with SQ-smoothed weights)
also collapsed.

**Decision:** Abandon SmoothQuant for W4. Root cause: SQ multiplies weight columns by
per-channel scale s[c]. For outlier channels, s[c] is large (up to 500×). Per-row W4
scales inflate to accommodate the amplified columns, collapsing bucket utilization for
the entire row. SQ works for W8 (256 levels absorb scale migration); W4 (16 levels) cannot.
See: `slides/smooth_quant_explainer.md` § "Experimental Results"

---

### 2026-03-27 — W4 bucket utilization analysis; group size selection

**Hypothesis:** Per-row W4 scales have poor bucket utilization on layers with within-row
outlier columns. The outlier column sets the scale; all other columns are quantized with
a coarser step than needed. Group quantization (finer-grained scales) can recover precision
without migrating anything into activations.

**Experiment:** `src/analyze_weight_utilization.py` on `quantized_weights_w4a8_adaround_poly_p100`.
Measured entropy, fill%, and saturation per layer. Simulated RTN re-quantization with
group_size = 32, 64, 128 from reconstructed FP16 weights.

**Result:**
| Group size | Entropy (bits) | Δ entropy | Saturation % |
|---|---|---|---|
| per-row (0) | 2.818 | — | 0.07% |
| 128 | 3.178 | +0.360 | 0.96% |
| **64** | **3.265** | **+0.447** | **1.80%** |
| 32 | 3.303 | +0.485 | 3.30% |

Worst layer: mm21 txt mlp_fc1 at 1.866 bits entropy (55.9% fill).
gs=32 marginal entropy gain (+0.038 over gs=64) not worth doubled saturation rate.

**Decision:** Run AdaRound with `--group-size 64`. gs=64 is the sweet spot.

```bash
conda run --no-capture-output -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data_512/adaround_cache \
    --output quantized_weights_w4a8_adaround_poly_p100_group64 \
    --poly-schedule polynomial_clipping_schedule_512_p100.json \
    --group-size 64 \
    --iters 3000 \
    --batch-size 8
```

**Status:** Running. Result pending.
See: `slides/smooth_quant_explainer.md` § "Bucket Utilization Results"

---

*Append new entries above this line as experiments complete.*
