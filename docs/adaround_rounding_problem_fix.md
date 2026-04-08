# AdaRound Optimization Bug: Activation Quantization Bias

## Problem

AdaRound-optimized weights were **1.68x worse** (per-weight MSE vs FP16) than simple round-to-nearest (RTN) across all 95 tested layers — zero layers improved. Generated images were catastrophically broken ("red mess").

## Root Cause

Two issues compounded:

### 1. Loss scale: sum vs mean (commit 7cdfb1d)

The reconstruction loss was changed from `mx.mean((pred - target) ** 2)` to `(pred - target).reshape(-1).square().sum()`. This inflated the loss ~2000x, destroying the relative balance between the reconstruction loss and the `_ROUND_WEIGHT * round_loss` regularizer. The regularizer (which drives alphas toward binary 0/1) became effectively disabled, so alpha binarization stalled and the optimization barely converged.

**Fix:** Reverted to `mx.mean((pred - target) ** 2)`.

### 2. Activation quantization bias in QuantProxy (the real bug)

`_QuantProxy.__call__` applied fake A8 activation quantization during the weight rounding optimization:

```python
x_q = _fake_quant_a8(x)          # quantized activations
w_soft = (floor_w + rect_sigmoid(alpha)) * scales
out = x_q @ w_soft.T              # gradient uses x_q, not x
```

At initialization, `w_soft ~ W_fp` (matching the FP16 weight), so the block reconstruction error is dominated by the activation quantization term `(x_q - x) @ W_fp^T`. The gradient w.r.t. `w_soft` is:

```
dL/dw_soft = x_q^T @ (x_q @ w_soft^T - x_fp @ w_fp^T)
```

This gradient has a systematic bias component `x_q^T @ (x_q - x_fp) @ w_fp^T` that pushes alpha (and therefore the soft weights) to **compensate for activation quantization error through weight adjustments**. The optimizer learns soft fractional rounding values that partially cancel the A8 error.

The problem: this compensation is **lost during hard rounding** (`round_delta = (alpha > 0)`). The binarized weights end up at rounding decisions that were optimized for a compensation effect that no longer exists, making them systematically worse than RTN.

The old convergence parameters (`patience=3, rtol=0.02`) masked this bug by stopping early — alphas barely moved from initialization, producing near-RTN weights. The tightened parameters (`patience=5, rtol=0.005`) let the optimization run longer, allowing more drift.

**Fix:** Remove A8 fake-quantization from the QuantProxy forward pass:

```python
# Before (broken):
x_q = _fake_quant_a8(x)
out = x_q @ w_soft.T

# After (fixed):
out = x @ w_soft.T
```

This ensures the optimizer **only** adjusts rounding to minimize weight quantization error. The activation quantization error is irreducible (same regardless of rounding direction) and should not influence the gradient.

## Diagnostic Evidence

Per-weight MSE comparison (v2 AdaRound vs RTN, both measured against FP16+CSB):

```
Layer                    | RTN MSE    | v2 MSE     | Ratio
b00.attn.q_proj          | 0.000035   | 0.000070   | 2.03x
b00.attn.o_proj          | 0.000029   | 0.000070   | 2.45x
b12.attn.q_proj          | 0.000153   | 0.000249   | 1.63x
b23.attn.q_proj          | 0.000189   | 0.000300   | 1.59x
Average across 15 layers |            |            | 1.68x
```

v2 won 0/95 layers. RTN won 95/95.

## Important: poly clipping is NOT removed from inference

The fix only removes activation quantization from `_QuantProxy.__call__`, which is the
**optimization-time** soft-weight proxy used solely during AdaRound training. It is not
the inference module.

Poly clipping at **inference** lives in `W4A8Linear.__call__` and is unaffected by this
fix. The poly schedule (if provided via `--poly-schedule`) is still applied at inference
time to clip activations before A8 quantization — that is the intended use of poly
clipping. What was wrong was applying it *during weight-rounding optimization*, where it
biased the gradient away from minimizing weight quantization error.

## Files Modified

- `src/phase4/optimize.py`
  - `_QuantProxy.__call__`: removed fake A8 activation quantization (including poly
    clipping) from the optimization forward pass — poly clipping at inference is unaffected
  - `loss_fn`: reverted `square().sum()` back to `mx.mean(... ** 2)`
  - Convergence params: reverted to `patience=3, rtol=0.02`
