# Research Finding: Does TaQ-DiT Use AdaRound?

## Short Answer

**No. TaQ-DiT does not use AdaRound.** The codebase's `adaround_optimize.py` is mislabeled as
"faithful TaQ-DiT" — it uses AdaRound (Nagel et al. 2020), which is a different paper.

---

## What TaQ-DiT Actually Uses

**Paper:** [arxiv 2411.14172](https://arxiv.org/abs/2411.14172)

### Weight quantization
- Standard **channel-wise uniform quantization** (per-channel scale + zero-point)
- Quantization parameters optimized via **joint reconstruction** — minimizing MSE loss over
  *both* weights and activations together (motivated by QDrop)
- No reference to AdaRound or GPTQ anywhere in the paper

### Activation quantization (the faithful part)
- Per-timestep scales — this IS what the codebase implements correctly
- Post-GELU shift with momentum 0.95 — implemented correctly
- Two-scale outlier handling (§3.3) — implemented correctly

---

## What the Codebase Actually Has

| Component | TaQ-DiT paper | This codebase | Faithful? |
|-----------|--------------|---------------|-----------|
| Weight quant method | Joint reconstruction (W+A together) | AdaRound (weights only, Nagel 2020) | ❌ |
| Activation quant | Per-timestep absmax, A8 | Per-timestep absmax, A8 | ✅ |
| Post-GELU shift | Momentum 0.95 | Momentum 0.95 | ✅ |
| Outlier handling | Two-scale §3.3 | Two-scale §3.3 | ✅ |
| Block reconstruction | Joint W+A MSE loss | Separate: W only (AdaRound), A via a_scale | ❌ |

---

## The Difference Matters

- **AdaRound**: optimizes weight rounding only, using block output reconstruction as the signal
- **TaQ-DiT joint reconstruction**: optimizes weights AND activation scales together in the same
  loop, so they are co-adapted — this is the source of the paper's quality improvement
- The codebase does have a joint element (`a_scales` trained alongside `alphas`), but activation
  scales are per-tensor learned parameters, not the per-timestep scales from the activation track

---

## Options

### Option A: Keep AdaRound, fix the labeling (easy)
Remove "faithful TaQ-DiT" claims from `adaround_optimize.py` docstring and CLAUDE.md.
Describe the system honestly as:
- **Weight track**: AdaRound (Nagel et al. 2020) W4
- **Activation track**: TaQ-DiT per-timestep A8 (this IS faithful)

AdaRound is a legitimate, well-regarded PTQ method. The activation track is genuinely
faithful to TaQ-DiT. The hybrid is a reasonable approach.

### Option B: Implement TaQ-DiT's actual joint reconstruction (significant work)
Replace the separate AdaRound loop with a joint W+A reconstruction loop where:
- Weight rounding AND per-timestep activation scales are optimized together
- The loss is computed over block outputs using the FP16 cache as reference
- This would make the weight track actually faithful to TaQ-DiT

This is a substantial rewrite of `adaround_optimize.py`.

### Option C: Do nothing for now
The system works, produces quantized weights, and the activation quantization IS faithful.
Defer the labeling/implementation question until after you have end-to-end quality metrics.

---

## Recommendation

**Option A** if you want correctness now with minimal effort.
**Option B** if you want to match the paper exactly and are willing to rewrite the weight track.
**Option C** if you want to get to end-to-end results first and revisit later.
