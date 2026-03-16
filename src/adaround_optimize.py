"""
AdaRound block-level W4A8 PTQ for SD3-Medium DiT (AdaRound baseline).

This module implements the **AdaRound baseline** for the AdaRound-vs-TaQ-DiT
comparison study.  It differs from TaQ-DiT joint reconstruction (see
``src/taqdit_optimize.py``) in one key respect:

  AdaRound (this file):
    One global activation scale per linear layer — learned once, applied at
    every diffusion timestep regardless of the current noise level.

  TaQ-DiT joint reconstruction (taqdit_optimize.py):
    A separate activation scale per (linear layer, timestep) — captures the
    empirical observation that diffusion activations vary significantly across
    the noise schedule (σ→1 high-noise vs σ→0 low-noise regimes).

Both use the same block-level reconstruction loss structure (rec_loss +
ROUND_WEIGHT * round_loss) and AdaRound weight rounding (learnable alpha).
The distinction is *activation scale granularity*: global-per-layer vs
per-layer-per-timestep.

Details:
  - Per-channel W4 AdaRound: learned alpha initialised from rounding residuals
  - Per-tensor A8 LSQ: one learned activation scale per linear (global)
  - Block-level reconstruction: full transformer block output vs FP16 cache
  - B-annealing: round_loss coefficient b decays 20→2 after 20% warmup
  - Loss: rec_loss + 0.01 * round_loss  (lp_loss with p=2, sum over channels)

Simplifications vs full TaQ-DiT:
  - No input mixing (drop_prob=1.0, always use FP block inputs from cache)
  - adaLN_modulation layers not quantized (attn + MLP only)
  - W4 applied to all blocks (TaQ-DiT uses W8 for first/last few)

Usage:
--no-capture-output flag is what lets tqdm progress bars stream through)
--batch-size 4 recommended to help with out of memory issues
    # Basic training
    conda run --no-capture-output -n diffusionkit python -m src.adaround_optimize \\
        --adaround-cache /path/to/adaround_cache \\
        --output /path/to/quantized_weights_adaround \\
        [--iters 20000] [--batch-size 16] [--bits-w 4] [--bits-a 8]

conda run --no-capture-output -n diffusionkit python -m src.adaround_optimize --adaround-cache calibration_data_100/adaround_cache --output quantized_weights_baseline --iters 600 --batch-size 4 --resume

    # Poly-clipping + σ-weighted loss (Module A)
    conda run --no-capture-output -n diffusionkit python -m src.adaround_optimize \\
        --adaround-cache /path/to/adaround_cache \\
        --output /path/to/quantized_weights_poly \\
        --poly-schedule polynomial_clipping_schedule.json \\
        --sigma-weighted --sigma-weight-offset 1.0

    # Refine from prior run, skipping converged blocks (< 0.5% tail improvement)
    conda run --no-capture-output -n diffusionkit python -m src.adaround_optimize \\
        --adaround-cache /path/to/adaround_cache \\
        --output /path/to/quantized_weights_poly \\
        --poly-schedule polynomial_clipping_schedule.json \\
        --refine --refine-skip-converged 0.5 --iters 500 --batch 8 
        (batch 8 helps keep iterations at 1 per second)

    # FP16 layer exclusion (Module B) — keep extreme-shift layers in FP16
    conda run --no-capture-output -n diffusionkit python -m src.adaround_optimize \\
        --adaround-cache /path/to/adaround_cache \\
        --output /path/to/quantized_weights_exclude \\
        --poly-schedule polynomial_clipping_schedule.json \\
        --exclude-extreme-shift \\
        --refine --refine-skip-converged 0.5

    # Asymmetric activation quant (Module C) — requires poly schedule with shift coeffs
    conda run --no-capture-output -n diffusionkit python -m src.adaround_optimize \\
        --adaround-cache /path/to/adaround_cache \\
        --output /path/to/quantized_weights_asym \\
        --poly-schedule polynomial_clipping_schedule_v2.json \\
        --asymmetric-act \\
        --refine --refine-skip-converged 0.5

    # All modules combined (A + B + C)
    conda run --no-capture-output -n diffusionkit python -m src.adaround_optimize \\
        --adaround-cache /path/to/adaround_cache \\
        --output /path/to/quantized_weights_abc \\
        --poly-schedule polynomial_clipping_schedule_v2.json \\
        --sigma-weighted --exclude-extreme-shift --asymmetric-act \\
        --refine --refine-skip-converged 0.5

Modules:
  A: σ-weighted loss      --sigma-weighted [--sigma-weight-offset ε]
     Weights calibration loss by 1/(σ+ε), emphasising clean-image timesteps.

  B: FP16 layer exclusion --exclude-extreme-shift | --exclude-layers KEY,...
     Keeps specified layers in FP16 (no quant). The 4 extreme adaLN-shift
     txt mlp_fc2 layers (mm14, mm20, mm21, mm22) are the default targets.

  C: Asymmetric act quant --asymmetric-act (requires --poly-schedule with shift_coeffs)
     Replaces symmetric [-α, +α] clipping with asymmetric [center-α, center+α]
     using σ-dependent shift polynomials. Needs poly schedule v2 (generated with
     ``python -m src.generate_poly_schedule --include-shifts``).

Refinement:
  --refine                   Warm-start from existing weights in --output dir.
  --refine-skip-converged P  Skip blocks whose loss improved < P% over the last
                             20% of the prior run's iteration history. Focuses
                             training time on blocks still improving.

Output layout:
    <output>/
        config.json                 quantisation config + per-block metrics
        weights/
            {block_name}.npz        hard-rounded quantised weight + scale per linear
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from src.cache_adaround_data import load_block_data


def load_pooled_embeddings(cache_dir: Path) -> Optional[Dict[int, np.ndarray]]:
    """
    Load per-image pooled text embeddings saved by cache_adaround_data.py.

    Returns dict mapping img_idx -> pooled numpy array, or None if not found.
    """
    pooled_dir = cache_dir / "pooled"
    if not pooled_dir.exists():
        return None
    result = {}
    for p in sorted(pooled_dir.glob("*.npz")):
        img_idx = int(p.stem)
        result[img_idx] = np.load(p)["pooled"]
    return result if result else None


def get_sample_image_indices(sample_files: List[Path]) -> List[int]:
    """Extract image index from each sample filename ({img:04d}_{step:03d}.npz)."""
    return [int(f.stem.split("_")[0]) for f in sample_files]


def set_block_modulation_params(
    block,
    is_mm: bool,
    mmdit,
    pooled_np: np.ndarray,
    timestep_val: float,
    activation_dtype,
):
    """
    Compute and set correct _modulation_params for one block + timestep.

    Uses the block's own adaLN_modulation weights (which must not be offloaded).
    Modulation inputs = y_embed + t_embed, same as cache_modulation_params().
    """
    pooled_mx = mx.array(pooled_np).astype(activation_dtype)
    y_embed = mmdit.y_embedder(pooled_mx)
    ts_mx = mx.array([timestep_val]).astype(activation_dtype)
    t_embed = mmdit.t_embedder(ts_mx)
    modulation_inputs = y_embed[:, None, None, :] + t_embed

    timestep_key = timestep_val
    if is_mm:
        img_tb = block.image_transformer_block
        txt_tb = block.text_transformer_block
        if not hasattr(img_tb, "_modulation_params"):
            img_tb._modulation_params = {}
            txt_tb._modulation_params = {}
        img_tb._modulation_params[timestep_key] = img_tb.adaLN_modulation(modulation_inputs)
        txt_tb._modulation_params[timestep_key] = txt_tb.adaLN_modulation(modulation_inputs)
        mx.eval(img_tb._modulation_params[timestep_key])
        mx.eval(txt_tb._modulation_params[timestep_key])
    else:
        tb = block.transformer_block
        if not hasattr(tb, "_modulation_params"):
            tb._modulation_params = {}
        tb._modulation_params[timestep_key] = tb.adaLN_modulation(modulation_inputs)
        mx.eval(tb._modulation_params[timestep_key])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA: float = -0.1       # lower bound of rectified sigmoid
ZETA: float = 1.1         # upper bound of rectified sigmoid
ROUND_WEIGHT: float = 0.01  # lambda for round_loss in total loss
LP_NORM: float = 2.0       # p for lp_loss


# ---------------------------------------------------------------------------
# B-annealing temperature schedule
# ---------------------------------------------------------------------------

class LinearTempDecay:
    """
    Linearly anneals b from start_b to end_b after warm_up fraction of t_max.

    b(t) = start_b                      for t < warm_up * t_max
           end_b + (start_b - end_b) *  for warm_up*t_max <= t < t_max
               (1 - (t - start_decay) / (t_max - start_decay))
           end_b                         for t >= t_max
    """

    def __init__(
        self,
        t_max: int = 20000,
        warm_up: float = 0.2,
        start_b: float = 20.0,
        end_b: float = 2.0,
    ):
        self.t_max = t_max
        self.start_decay = int(warm_up * t_max)
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t: int) -> float:
        if t < self.start_decay:
            return self.start_b
        if t >= self.t_max:
            return self.end_b
        rel = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + (self.start_b - self.end_b) * max(0.0, 1.0 - rel)


# ---------------------------------------------------------------------------
# Quantisation primitives
# ---------------------------------------------------------------------------

def rectified_sigmoid(alpha: mx.array) -> mx.array:
    """Smooth rounding mask: clamp((zeta-gamma)*sigmoid(alpha)+gamma, 0, 1)."""
    return mx.clip((ZETA - GAMMA) * mx.sigmoid(alpha) + GAMMA, 0.0, 1.0)


def compute_per_channel_scale(W_np: np.ndarray, bits: int = 4) -> np.ndarray:
    """Per-output-channel absmax scale, shape (out_features, 1)."""
    qmax = 2 ** (bits - 1) - 1
    absmax = np.abs(W_np).max(axis=1, keepdims=True)  # (out, 1)
    return np.maximum(absmax / qmax, 1e-8)


def init_alpha(W_np: np.ndarray, scale_np: np.ndarray) -> np.ndarray:
    """
    Initialise AdaRound alpha so rectified_sigmoid(alpha) ≈ fractional part of W/scale.

    alpha = -log((zeta - gamma) / (rest - gamma) - 1)
    where rest = (W / scale) - floor(W / scale)  ∈ [0, 1)
    """
    rest = W_np / scale_np - np.floor(W_np / scale_np)
    rest = np.clip(rest, 1e-6, 1.0 - 1e-6)
    return -np.log((ZETA - GAMMA) / (rest - GAMMA) - 1.0)


def reconstruct_alpha(W_fp_np: np.ndarray, scale_np: np.ndarray,
                      weight_int_np: np.ndarray) -> np.ndarray:
    """
    Reconstruct AdaRound alpha from prior hard-rounded weights.

    The hard rounding decision was: delta = 1 if alpha >= 0 else 0
    W_q = clip(floor(W/s) + delta, qmin, qmax) * s

    To warm-start, we recover delta and set alpha = +large if delta=1, -large if delta=0,
    then perturb slightly so the optimizer has gradient signal.
    """
    W_floor = np.floor(W_fp_np / scale_np)
    # delta = W_int - floor(W/s), should be 0 or 1
    delta = weight_int_np.astype(np.float32) - W_floor
    delta = np.clip(delta, 0.0, 1.0)
    # Convert to alpha: inverse of rectified_sigmoid
    # rectified_sigmoid(alpha) ≈ delta, so alpha = logit((delta - GAMMA) / (ZETA - GAMMA))
    rest = np.clip(delta, 1e-4, 1.0 - 1e-4)
    alpha = -np.log((ZETA - GAMMA) / (rest - GAMMA) - 1.0)
    return alpha


def load_prior_block(
    npz_path: Path,
    linear_paths: List[str],
    W_fps_np: List[np.ndarray],
    bits_w: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Load prior hard-rounded weights and reconstruct alpha + a_scale for warm-start.

    Returns (prior_alphas, prior_a_scales) aligned with linear_paths.
    """
    npz = np.load(npz_path)
    prior_alphas = []
    prior_a_scales = []
    for i, path in enumerate(linear_paths):
        safe = path.replace(".", "_")
        alpha_key = f"{safe}__alpha"
        wi_key = f"{safe}__weight_int"
        sc_key = f"{safe}__scale"
        as_key = f"{safe}__a_scale"

        if alpha_key in npz.files:
            # Prefer saved raw alpha (exact match, no dtype reconstruction issues)
            prior_alphas.append(npz[alpha_key])
            a_scale = float(npz[as_key][0]) if as_key in npz.files else 1.0
            prior_a_scales.append(a_scale)
        elif wi_key in npz.files and sc_key in npz.files:
            # Fallback: reconstruct alpha from hard-rounded weights
            weight_int = npz[wi_key]
            scale = npz[sc_key]
            alpha = reconstruct_alpha(W_fps_np[i], scale, weight_int)
            prior_alphas.append(alpha)
            a_scale = float(npz[as_key][0]) if as_key in npz.files else 1.0
            prior_a_scales.append(a_scale)
        else:
            # No prior data for this linear — init from scratch
            scale_np = compute_per_channel_scale(W_fps_np[i], bits_w)
            prior_alphas.append(init_alpha(W_fps_np[i], scale_np))
            prior_a_scales.append(1.0)

    return prior_alphas, prior_a_scales


def fake_quant_per_tensor(
    x: mx.array, scale: mx.array, qmin: int, qmax: int
) -> mx.array:
    """Symmetric per-tensor fake-quantise (STE straight-through estimator)."""
    s = mx.abs(scale) + 1e-8
    return mx.clip(mx.round(x / s), qmin, qmax) * s


def fake_quant_asymmetric(
    x: mx.array, min_val: float, max_val: float, qmin: int, qmax: int
) -> mx.array:
    """Asymmetric fake quantization: maps [min_val, max_val] to [qmin, qmax]."""
    scale = (max_val - min_val) / (qmax - qmin) + 1e-8
    zero_point = mx.round(mx.array(-min_val / scale))
    x_q = mx.clip(mx.round(x / scale) + zero_point, qmin, qmax)
    return (x_q - zero_point) * scale


# ---------------------------------------------------------------------------
# Block linear-layer enumeration
# ---------------------------------------------------------------------------

# Relative paths within one TransformerBlock (attn + MLP only; not adaLN)
_LOCAL_LINEAR_PATHS = [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "mlp.fc1",
    "mlp.fc2",
]

# Paths of post-GELU inputs (fc2 inputs); these need the shift from calibration
_POST_GELU_LOCAL_PATHS = {"mlp.fc2"}


def _get_nested(obj: Any, path: str) -> Any:
    """Get a nested attribute using dot-separated path (supports list[i] notation)."""
    for part in path.split("."):
        if "[" in part:
            attr, idx_s = part.split("[", 1)
            obj = getattr(obj, attr)[int(idx_s.rstrip("]"))]
        else:
            obj = getattr(obj, part)
    return obj


def _set_nested(obj: Any, path: str, val: Any) -> None:
    """Set a nested attribute using dot-separated path."""
    parts = path.split(".")
    for part in parts[:-1]:
        if "[" in part:
            attr, idx_s = part.split("[", 1)
            obj = getattr(obj, attr)[int(idx_s.rstrip("]"))]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if "[" in last:
        attr, idx_s = last.split("[", 1)
        getattr(obj, attr)[int(idx_s.rstrip("]"))] = val
    else:
        setattr(obj, last, val)


def get_block_linears(block: Any, is_mm: bool) -> List[Tuple[str, Any, bool]]:
    """
    Return (full_attr_path, layer, is_post_gelu) for all quantisable linears.

    For MultiModalTransformerBlock (is_mm=True):
      image_transformer_block.attn.q_proj ... image_transformer_block.mlp.fc2
      text_transformer_block.attn.q_proj  ... text_transformer_block.mlp.fc2

    For UnifiedTransformerBlock (is_mm=False):
      transformer_block.attn.q_proj ... transformer_block.mlp.fc2
    """
    prefixes = (
        ["image_transformer_block", "text_transformer_block"] if is_mm
        else ["transformer_block"]
    )
    results: List[Tuple[str, Any, bool]] = []
    for prefix in prefixes:
        for local in _LOCAL_LINEAR_PATHS:
            full = f"{prefix}.{local}"
            try:
                layer = _get_nested(block, full)
            except AttributeError:
                continue  # sub-module doesn't exist (e.g., no mlp on some blocks)
            # Skip non-Linear layers (e.g., nn.Identity has no weight)
            if not hasattr(layer, "weight"):
                continue
            is_post_gelu = (local in _POST_GELU_LOCAL_PATHS)
            results.append((full, layer, is_post_gelu))
    return results


# ---------------------------------------------------------------------------
# Quantised-layer proxy (used during optimisation only)
# ---------------------------------------------------------------------------

class _QuantProxy:
    """
    Temporarily replaces an nn.Linear in a block during AdaRound optimisation.

    Applies:
      1. Per-tensor LSQ fake-quantisation to the input activation.
         If poly_alpha is provided, it overrides the learnable a_scale with
         a fixed polynomial-derived clipping range.
      2. Soft AdaRound fake-quantisation to the weight.

    Gradients flow to both alpha (via soft weight) and a_scale (via fake-quant input).
    """

    def __init__(
        self,
        orig_layer: Any,
        soft_weight: mx.array,   # derived from alpha — in the computation graph
        a_scale: mx.array,       # learnable activation scale
        qmin_a: int,
        qmax_a: int,
        poly_alpha: Optional[float] = None,  # if set, overrides a_scale
        poly_shift: Optional[float] = None,  # if set, uses asymmetric quant
    ):
        self._orig = orig_layer
        self._soft_weight = soft_weight
        self._a_scale = a_scale
        self._qmin_a = qmin_a
        self._qmax_a = qmax_a
        self._poly_alpha = poly_alpha
        self._poly_shift = poly_shift

    def __call__(self, x: mx.array) -> mx.array:
        if self._poly_alpha is not None and self._poly_shift is not None:
            # Asymmetric path: center ± range → [min_val, max_val]
            center = self._poly_shift
            range_val = self._poly_alpha
            min_val = center - range_val
            max_val = center + range_val
            x_q = fake_quant_asymmetric(x, min_val, max_val, self._qmin_a, self._qmax_a)
        elif self._poly_alpha is not None:
            # Symmetric poly path: α / qmax
            scale = mx.array(self._poly_alpha / max(abs(self._qmax_a), 1))
            x_q = fake_quant_per_tensor(x, scale, self._qmin_a, self._qmax_a)
        else:
            x_q = fake_quant_per_tensor(x, self._a_scale, self._qmin_a, self._qmax_a)
        y = x_q @ self._soft_weight.T
        if hasattr(self._orig, "bias") and self._orig.bias is not None:
            y = y + self._orig.bias
        return y

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute look-ups to the original layer
        return getattr(self._orig, name)


# ---------------------------------------------------------------------------
# AdaRound parameter container (nn.Module so MLX tracks gradients)
# ---------------------------------------------------------------------------

class AdaRoundParams(nn.Module):
    """
    Trainable AdaRound parameters for one transformer block.

    alphas[i]   : AdaRound alpha, shape = W.shape
    a_scales[i] : Per-tensor LSQ activation scale, shape (1,)

    W_fps_np and w_scales_np are kept as numpy so they are not treated
    as model parameters and receive no gradients.
    """

    def __init__(
        self,
        W_fps_np: List[np.ndarray],
        bits_w: int = 4,
        bits_a: int = 8,
        prior_alphas: Optional[List[np.ndarray]] = None,
        prior_a_scales: Optional[List[float]] = None,
    ):
        super().__init__()
        self.bits_w = bits_w
        self.bits_a = bits_a
        self.qmin_w = -(2 ** (bits_w - 1))
        self.qmax_w = 2 ** (bits_w - 1) - 1
        self.qmin_a = -(2 ** (bits_a - 1))
        self.qmax_a = 2 ** (bits_a - 1) - 1

        alphas: List[mx.array] = []
        a_scales: List[mx.array] = []
        for i, W_np in enumerate(W_fps_np):
            if prior_alphas is not None:
                alphas.append(mx.array(prior_alphas[i]))
            else:
                scale_np = compute_per_channel_scale(W_np, bits_w)
                alphas.append(mx.array(init_alpha(W_np, scale_np)))
            if prior_a_scales is not None:
                a_scales.append(mx.array([prior_a_scales[i]]))
            else:
                a_scales.append(mx.array([1.0]))
        self.alphas = alphas
        self.a_scales = a_scales


# ---------------------------------------------------------------------------
# Loss function (called inside nn.value_and_grad)
# ---------------------------------------------------------------------------

def _lp_loss(pred: mx.array, tgt: mx.array, p: float = 2.0) -> mx.array:
    """lp reconstruction loss: mean over batch of sum-over-channels of |pred-tgt|^p."""
    # pred/tgt: (batch, seq, hidden) or (batch, hidden)
    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_tgt = tgt.reshape(tgt.shape[0], -1)
    return ((flat_pred - flat_tgt).abs() ** p).sum(axis=1).mean()


def _round_loss(alphas: List[mx.array], b: float) -> mx.array:
    """Regularisation that pushes rounding decisions to 0 or 1."""
    total = mx.array(0.0)
    for alpha in alphas:
        r = rectified_sigmoid(alpha)
        total = total + (1.0 - ((r - 0.5).abs() * 2.0) ** b).sum()
    return total


def block_loss_fn(
    params: AdaRoundParams,
    block: Any,
    is_mm: bool,
    linear_paths: List[str],
    linear_layers: List[Any],
    W_fps_np: List[np.ndarray],
    w_scales_np: List[np.ndarray],
    sample_inputs: List[mx.array],   # positional args for the block
    sample_kwargs: Dict[str, mx.array],
    fp_outputs: List[mx.array],       # cached FP block outputs
    b_val: float,
    poly_alphas: Optional[List[Optional[float]]] = None,  # per-linear poly α for this sample's σ
    poly_shifts: Optional[List[Optional[float]]] = None,  # per-linear poly shift for asymmetric quant
) -> mx.array:
    """
    Compute block-level reconstruction + round loss for one mini-batch sample.

    Patches each linear in the block with a _QuantProxy (soft AdaRound weight +
    LSQ activation), runs the block forward, restores original layers, then
    computes the loss.  MLX's lazy graph preserves gradients through alpha and
    a_scale even after the Python-level restoration.

    If poly_alphas is provided, each linear's _QuantProxy uses the polynomial-
    derived clipping range instead of the learnable a_scale.
    """
    n = len(linear_paths)

    # Cast inputs/targets to float32 — the block's internal attention and norm
    # backward passes produce NaN gradients in bfloat16 when activations are
    # quantised (Q/K near-zero → RMSNorm grad explosion, attention grad instability).
    # Float32 inputs force the entire forward+backward through the block to use
    # float32 (MLX promotes bfloat16 params to float32 in mixed-precision ops).
    sample_inputs = [inp.astype(mx.float32) for inp in sample_inputs]
    sample_kwargs = {k: v.astype(mx.float32) for k, v in sample_kwargs.items()}
    fp_outputs = [out.astype(mx.float32) for out in fp_outputs]

    # Build soft-quantised weights (in computation graph via params.alphas)
    soft_weights: List[mx.array] = []
    for i in range(n):
        W_fp_mx = mx.array(W_fps_np[i])       # constant — no gradient
        s_mx = mx.array(w_scales_np[i])        # constant — no gradient
        r = rectified_sigmoid(params.alphas[i])
        W_floor = mx.floor(W_fp_mx / s_mx)
        soft_w = mx.clip(W_floor + r, params.qmin_w, params.qmax_w) * s_mx
        soft_weights.append(soft_w)

    # Patch: replace each linear with a _QuantProxy
    orig_layers = []
    for i, path in enumerate(linear_paths):
        orig_layers.append(_get_nested(block, path))
        pa = poly_alphas[i] if poly_alphas else None
        ps = poly_shifts[i] if poly_shifts else None
        proxy = _QuantProxy(
            linear_layers[i],
            soft_weights[i],
            params.a_scales[i],
            params.qmin_a,
            params.qmax_a,
            poly_alpha=pa,
            poly_shift=ps,
        )
        _set_nested(block, path, proxy)

    # Run quantised block forward
    try:
        if sample_kwargs:
            output = block(*sample_inputs, **sample_kwargs)
        else:
            output = block(*sample_inputs)
    finally:
        # Restore originals (Python side-effect; does not break gradient graph)
        for path, orig in zip(linear_paths, orig_layers):
            _set_nested(block, path, orig)

    # Reconstruction loss (both output and target are float32 now)
    if isinstance(output, (list, tuple)):
        rec = sum(_lp_loss(out, tgt, LP_NORM) for out, tgt in zip(output, fp_outputs))
    else:
        rec = _lp_loss(output, fp_outputs[0], LP_NORM)

    # Round loss (annealed regularisation)
    rnd = _round_loss(params.alphas, b_val)

    return rec + ROUND_WEIGHT * rnd


# ---------------------------------------------------------------------------
# Main optimisation loop for one block
# ---------------------------------------------------------------------------

def _extract_sample_sigmas(
    sample_files: List[Path],
    sigma_map: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Extract sigma value for each cached sample.

    First checks for embedded __sigma__ key in NPZ (written by cache_adaround_data.py
    with poly-clipping support). Falls back to filename pattern {img:04d}_{step:03d}.npz
    with sigma_map lookup.

    Returns array of shape (n_samples,) with sigma for each sample.
    """
    sigmas = []
    for f in sample_files:
        # Try embedded sigma first
        try:
            npz = np.load(f, allow_pickle=False)
            if "__sigma__" in npz.files:
                sigmas.append(float(npz["__sigma__"][0]))
                continue
        except Exception:
            pass
        # Fallback: derive from filename + sigma_map
        parts = f.stem.split("_")
        step_key = parts[1] if len(parts) >= 2 else "0"
        if sigma_map and step_key in sigma_map:
            sigmas.append(sigma_map[step_key])
        else:
            sigmas.append(0.0)
    return np.array(sigmas)


def _compute_sigma_weights(
    sample_sigmas: np.ndarray,
    offset: float = 1.0,
) -> np.ndarray:
    """Compute per-sample importance weights: w(σ) = 1/(σ + offset).

    Low-σ (clean) samples get higher weight. offset prevents division
    by zero and controls how aggressively low-σ is emphasized.
    offset=1.0 gives ~15× ratio between σ=0.03 and σ=14.6.
    """
    raw = 1.0 / (sample_sigmas + offset)
    return raw / raw.mean()  # Normalize so mean weight = 1.0


def _compute_derivative_weights(
    poly_schedule: Dict,
    block_name: str,
    is_mm: bool,
    linear_paths: List[str],
    sample_sigmas: np.ndarray,
) -> np.ndarray:
    """Compute per-sample |dα/dσ| weights averaged across linear layers in a block.

    For each sample i, evaluates |dα/dσ|(σ_i) for every linear layer in the block
    using the polynomial derivative, then averages across layers.

    Degree-0 (static) layers contribute zero derivative — blocks where all layers
    are static return uniform weights (no amplification, no suppression).

    The returned weights are normalised so mean = 1.0, making them a multiplicative
    modifier that can be combined with σ-weights:
        combined_w = sigma_w * deriv_w
    """
    poly_layers = poly_schedule.get("layers", {})
    sigma_range = poly_schedule.get("sigma_range")

    # Collect |dα/dσ| per layer per sample
    layer_derivs: List[np.ndarray] = []
    for path in linear_paths:
        poly_key = _path_to_poly_key(block_name, is_mm, path)
        layer_cfg = poly_layers.get(poly_key)
        if layer_cfg is None:
            continue
        coeffs = np.array(layer_cfg["coeffs"], dtype=np.float64)
        degree = len(coeffs) - 1
        if degree == 0:
            # Static layer — zero derivative everywhere
            layer_derivs.append(np.zeros_like(sample_sigmas, dtype=np.float64))
            continue

        # Derivative coefficients: d/dσ of np.polyval(coeffs, σ)
        # np.polyval uses descending order: coeffs[0]*σ^n + ... + coeffs[-1]
        deriv_coeffs = np.polyder(coeffs)

        sigmas = sample_sigmas.astype(np.float64)
        if sigma_range:
            sigmas = np.clip(sigmas, sigma_range[0], sigma_range[1])

        layer_derivs.append(np.abs(np.polyval(deriv_coeffs, sigmas)))

    if not layer_derivs:
        return np.ones_like(sample_sigmas, dtype=np.float64)

    # Average |dα/dσ| across layers
    mean_deriv = np.mean(np.stack(layer_derivs, axis=0), axis=0)

    # If all layers are static, return uniform weights
    if mean_deriv.max() == 0.0:
        return np.ones_like(sample_sigmas, dtype=np.float64)

    return mean_deriv / mean_deriv.mean()  # Normalize so mean weight = 1.0


def _init_a_scales_from_poly(
    poly_schedule: Dict,
    block_name: str,
    is_mm: bool,
    linear_paths: List[str],
    sigma: float,
    qmax_a: int = 127,
) -> List[float]:
    """
    Compute initial a_scale values from poly schedule at a representative sigma.

    Returns per-linear list of floats: alpha(sigma) / qmax_a.
    Falls back to 1.0 for layers not in the poly schedule.
    """
    alphas = _compute_poly_alphas_for_sample(
        poly_schedule, block_name, is_mm, linear_paths, sigma
    )
    if alphas is None:
        return [1.0] * len(linear_paths)
    return [(alpha / qmax_a) if alpha is not None else 1.0 for alpha in alphas]


def _path_to_poly_key(block_name: str, is_mm: bool, path: str) -> str:
    """Convert block-internal path to poly schedule key (underscore-separated).

    e.g. block_name="mm14", path="text_transformer_block.mlp.fc2" → "mm14_txt_mlp_fc2"
    """
    if is_mm:
        if path.startswith("image_transformer_block."):
            stream = "img"
            local = path[len("image_transformer_block."):]
        elif path.startswith("text_transformer_block."):
            stream = "txt"
            local = path[len("text_transformer_block."):]
        else:
            return f"{block_name}_{path.replace('.', '_')}"
        return f"{block_name}_{stream}_{local.replace('.', '_')}"
    else:
        if path.startswith("transformer_block."):
            local = path[len("transformer_block."):]
        else:
            local = path
        return f"{block_name}_{local.replace('.', '_')}"


def _compute_poly_alphas_for_sample(
    poly_schedule: Optional[Dict],
    block_name: str,
    is_mm: bool,
    linear_paths: List[str],
    sigma: float,
) -> Optional[List[Optional[float]]]:
    """
    Compute per-linear polynomial α values for a given sigma.

    Returns None if no poly schedule, or a list of Optional[float] per linear.
    """
    if poly_schedule is None:
        return None

    poly_layers = poly_schedule.get("layers", {})
    alphas = []

    for path in linear_paths:
        poly_key = _path_to_poly_key(block_name, is_mm, path)
        layer_cfg = poly_layers.get(poly_key)
        if layer_cfg is not None:
            sigma_range = poly_schedule.get("sigma_range")
            clamped_sigma = sigma
            if sigma_range:
                clamped_sigma = max(sigma_range[0], min(sigma, sigma_range[1]))
            alpha = float(np.polyval(layer_cfg["coeffs"], clamped_sigma))
            alphas.append(max(alpha, 1e-8))  # clamp to positive
        else:
            alphas.append(None)

    return alphas


def _compute_poly_shifts_for_sample(
    poly_schedule: Optional[Dict],
    block_name: str,
    is_mm: bool,
    linear_paths: List[str],
    sigma: float,
) -> Optional[List[Optional[float]]]:
    """
    Compute per-linear polynomial shift values for a given sigma.

    Returns None if no poly schedule or no shift coefficients present,
    or a list of Optional[float] per linear (None for layers without shift).
    """
    if poly_schedule is None:
        return None

    poly_layers = poly_schedule.get("layers", {})
    shifts = []
    any_shift = False

    for path in linear_paths:
        poly_key = _path_to_poly_key(block_name, is_mm, path)
        layer_cfg = poly_layers.get(poly_key)
        if layer_cfg is not None and "shift_coeffs" in layer_cfg:
            sigma_range = poly_schedule.get("sigma_range")
            clamped_sigma = sigma
            if sigma_range:
                clamped_sigma = max(sigma_range[0], min(sigma, sigma_range[1]))
            shift = float(np.polyval(layer_cfg["shift_coeffs"], clamped_sigma))
            shifts.append(shift)
            any_shift = True
        else:
            shifts.append(None)

    return shifts if any_shift else None


def optimize_block(
    block: Any,
    block_name: str,
    is_mm: bool,
    block_data: Dict[str, np.ndarray],  # from load_block_data
    iters: int = 20000,
    batch_size: int = 16,
    bits_w: int = 4,
    bits_a: int = 8,
    w_lr: float = 1e-3,
    a_lr: float = 4e-5,
    poly_schedule: Optional[Dict] = None,
    sample_sigmas: Optional[np.ndarray] = None,
    prior_alphas: Optional[List[np.ndarray]] = None,
    prior_a_scales: Optional[List[float]] = None,
    mmdit: Optional[Any] = None,
    sample_pooled: Optional[np.ndarray] = None,  # shape (n_samples, *pooled_shape)
    activation_dtype=None,
    sigma_weighted: bool = False,
    sigma_weight_offset: float = 1.0,
    derivative_weighted: bool = False,
    exclude_keys: Optional[set] = None,
    asymmetric_act: bool = False,
) -> Tuple[AdaRoundParams, Dict]:
    """
    Run AdaRound optimisation for a single transformer block.

    Parameters
    ----------
    poly_schedule : optional dict
        Polynomial clipping schedule from generate_poly_schedule.py.
        When provided, A8 fake quant uses poly-derived α(σ) per sample.
    sample_sigmas : optional array of shape (n_samples,)
        Sigma value for each calibration sample. Required if poly_schedule is set.

    Returns
    -------
    params : AdaRoundParams
        Optimised alpha and a_scale (use finalize_block to extract hard weights).
    metrics : dict
        loss_history, final_rec_loss, etc.
    """
    linears = get_block_linears(block, is_mm)

    # Filter out excluded layers (they stay FP16, no alpha/a_scale optimized)
    excluded_indices: set = set()
    if exclude_keys:
        for i, (path, _, _) in enumerate(linears):
            poly_key = _path_to_poly_key(block_name, is_mm, path)
            if poly_key in exclude_keys:
                excluded_indices.add(i)
                print(f"    EXCLUDE {poly_key} (FP16)", flush=True)
    linear_paths = [p for i, (p, _, _) in enumerate(linears) if i not in excluded_indices]
    linear_layers = [l for i, (_, l, _) in enumerate(linears) if i not in excluded_indices]
    W_fps_np = [np.array(l.weight) for l in linear_layers]
    w_scales_np = [compute_per_channel_scale(W, bits_w) for W in W_fps_np]

    # Initialize a_scale from poly schedule when no prior is provided.
    # Using 1.0 for activations in ±hundreds causes catastrophic clipping → NaN.
    init_a_scales = prior_a_scales
    if init_a_scales is None and poly_schedule is not None:
        qmax_a = 2 ** (bits_a - 1) - 1
        # Use median sigma if available and non-degenerate. If sample_sigmas are all
        # zero (older cache without embedded __sigma__ key), fall back to σ=1.0 so
        # that a_scale is initialized at the worst-case (highest-noise) clipping range,
        # which is conservative and avoids catastrophic under-clipping at init.
        if sample_sigmas is not None and float(np.max(sample_sigmas)) > 0.01:
            repr_sigma = float(np.median(sample_sigmas))
        else:
            repr_sigma = 1.0  # conservative fallback: max noise level
        init_a_scales = _init_a_scales_from_poly(
            poly_schedule, block_name, is_mm, linear_paths, repr_sigma, qmax_a=qmax_a
        )
        print(
            f"  a_scale init from poly @ σ={repr_sigma:.3f}: "
            f"[{min(init_a_scales):.4f}, {max(init_a_scales):.4f}]",
            flush=True,
        )

    params = AdaRoundParams(
        W_fps_np, bits_w=bits_w, bits_a=bits_a,
        prior_alphas=prior_alphas, prior_a_scales=init_a_scales,
    )

    # Two optimisers: Adam for alpha (w_lr), Adam+cosine for a_scale (a_lr)
    w_opt = optim.Adam(learning_rate=w_lr)
    a_opt = optim.Adam(learning_rate=a_lr)

    temp_decay = LinearTempDecay(t_max=iters, warm_up=0.2, start_b=20.0, end_b=2.0)

    # Unpack all calibration samples
    n_samples = block_data["arg0"].shape[0]

    # Precompute per-sample sigma weights
    sample_weights = None
    if sigma_weighted and sample_sigmas is not None:
        sample_weights = _compute_sigma_weights(sample_sigmas, sigma_weight_offset)
        print(
            f"  σ-weighted loss enabled (offset={sigma_weight_offset}): "
            f"weight range [{sample_weights.min():.3f}, {sample_weights.max():.3f}]",
            flush=True,
        )

    if derivative_weighted and poly_schedule is not None and sample_sigmas is not None:
        deriv_w = _compute_derivative_weights(
            poly_schedule, block_name, is_mm, linear_paths, sample_sigmas
        )
        if sample_weights is not None:
            sample_weights = sample_weights * deriv_w
            sample_weights = sample_weights / sample_weights.mean()
        else:
            sample_weights = deriv_w
        print(
            f"  derivative-weighted loss enabled: "
            f"|dα/dσ| weight range [{sample_weights.min():.3f}, {sample_weights.max():.3f}]",
            flush=True,
        )

    # Build grad function w.r.t. params
    loss_and_grad_fn = nn.value_and_grad(params, block_loss_fn)

    # Separate params module into alpha-group and a_scale-group for two optimisers
    # We manually split the grads after each step.
    class _AlphaOnly(nn.Module):
        pass

    alpha_mod = _AlphaOnly()
    alpha_mod.alphas = params.alphas  # shared reference

    a_scale_mod = _AlphaOnly()
    a_scale_mod.a_scales = params.a_scales  # shared reference

    loss_history: List[float] = []
    pbar = tqdm(total=iters, desc=f"  {block_name}", unit="it", leave=True,
                mininterval=1.0, file=sys.stdout,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    for step in range(iters):
        b_val = temp_decay(step)

        # Sample mini-batch
        idx = np.random.randint(0, n_samples, min(batch_size, n_samples))

        # Average loss over sampled indices
        total_loss = mx.array(0.0)
        total_loss_uw = mx.array(0.0)  # unweighted loss (for logging)
        total_grads = None
        n_valid = 0
        weight_sum = 0.0

        for si in idx:
            # Build input tensors for this sample
            s_inputs: List[mx.array] = []
            s_kwargs: Dict[str, mx.array] = {}
            s_outputs: List[mx.array] = []

            # Positional args: arg0, arg1, arg2, ...
            for key in sorted(k for k in block_data if k.startswith("arg")):
                s_inputs.append(mx.array(block_data[key][si]))

            # Keyword args
            for key in (k for k in block_data if k.startswith("kw_")):
                kw_name = key[3:]  # strip "kw_" prefix
                s_kwargs[kw_name] = mx.array(block_data[key][si])

            # Target outputs
            for key in sorted(k for k in block_data if k.startswith("out")):
                s_outputs.append(mx.array(block_data[key][si]))

            # Set correct per-sample modulation params if pooled embeddings available
            if sample_pooled is not None and mmdit is not None and activation_dtype is not None:
                # timestep is arg1 for Uni, arg2 for MM
                ts_key = "arg2" if is_mm else "arg1"
                timestep_val = float(block_data[ts_key][si].flat[0])
                set_block_modulation_params(
                    block, is_mm, mmdit,
                    sample_pooled[si], timestep_val, activation_dtype,
                )

            # Compute poly alphas (and shifts for asymmetric mode) for this sample's sigma
            poly_alphas = None
            poly_shifts = None
            if poly_schedule is not None and sample_sigmas is not None:
                sigma_val = float(sample_sigmas[si])
                poly_alphas = _compute_poly_alphas_for_sample(
                    poly_schedule, block_name, is_mm, linear_paths, sigma_val
                )
                if asymmetric_act:
                    poly_shifts = _compute_poly_shifts_for_sample(
                        poly_schedule, block_name, is_mm, linear_paths, sigma_val
                    )

            loss_i, grads_i = loss_and_grad_fn(
                params, block, is_mm,
                linear_paths, linear_layers,
                W_fps_np, w_scales_np,
                s_inputs, s_kwargs, s_outputs,
                b_val, poly_alphas, poly_shifts,
            )
            # NO mx.eval here — keep everything lazy.
            # Accumulate loss and grads in the computation graph so MLX can
            # fuse all 16 forward+backward passes into one evaluation.

            w_i = float(sample_weights[si]) if sample_weights is not None else 1.0
            total_loss = total_loss + w_i * loss_i
            total_loss_uw = total_loss_uw + loss_i
            weight_sum += w_i
            n_valid += 1

            if total_grads is None:
                if w_i != 1.0:
                    total_grads = {
                        "alphas": [g * w_i for g in grads_i["alphas"]],
                        "a_scales": [g * w_i for g in grads_i["a_scales"]],
                    }
                else:
                    total_grads = grads_i
            else:
                for i in range(len(params.alphas)):
                    total_grads["alphas"][i] = (
                        total_grads["alphas"][i] + w_i * grads_i["alphas"][i]
                    )
                    total_grads["a_scales"][i] = (
                        total_grads["a_scales"][i] + w_i * grads_i["a_scales"][i]
                    )

        n_used = n_valid
        if total_grads is None:
            loss_val = float("nan")
            loss_history.append(loss_val)
            if step % 50 == 0:
                pbar.set_postfix(loss="nan", b=f"{b_val:.1f}")
            if step % 200 == 0:
                print(f"  {block_name} iter {step}/{iters}  loss=nan  b={b_val:.1f}", flush=True)
            pbar.update(1)
            continue

        # Average grads over valid samples (use weight_sum for σ-weighted mode)
        grad_denom = weight_sum if sample_weights is not None else n_used
        for i in range(len(params.alphas)):
            total_grads["alphas"][i] = total_grads["alphas"][i] / grad_denom
            total_grads["a_scales"][i] = total_grads["a_scales"][i] / grad_denom

        # Single eval for the entire batch: loss + all gradients together.
        # This lets MLX fuse all 16 forward+backward passes into one GPU dispatch
        # instead of 16 separate sync points.
        all_grad_arrays = total_grads["alphas"] + total_grads["a_scales"]
        mx.eval(total_loss, *all_grad_arrays)

        # NaN check on the batch total (if any sample was NaN, total is NaN)
        loss_denom = weight_sum if sample_weights is not None else n_used
        loss_val = float(total_loss) / loss_denom
        if math.isnan(loss_val) or math.isinf(loss_val):
            if step < 10 or step % 100 == 0:
                print(
                    f"  WARNING: NaN/Inf loss at step {step} (b={b_val:.1f}), "
                    f"skipping optimizer update",
                    flush=True,
                )
            loss_history.append(loss_val)
            pbar.update(1)
            continue

        has_nan_grad = any(mx.isnan(g).any().item() for g in all_grad_arrays)
        if has_nan_grad:
            if step < 10 or step % 100 == 0:
                print(
                    f"  WARNING: NaN in gradients at step {step} (b={b_val:.1f}), "
                    f"skipping optimizer update",
                    flush=True,
                )
            loss_history.append(loss_val)
            pbar.update(1)
            continue

        # Clip gradients: a_scale needs tight clipping (small param, sensitive to explosion)
        # alpha uses a looser clip — needs enough gradient to make rounding decisions
        ALPHA_GRAD_CLIP = 0.1
        A_SCALE_GRAD_CLIP = 1.0
        clipped_alpha_grads = [
            mx.clip(g, -ALPHA_GRAD_CLIP, ALPHA_GRAD_CLIP) for g in total_grads["alphas"]
        ]
        clipped_a_grads = [
            mx.clip(g, -A_SCALE_GRAD_CLIP, A_SCALE_GRAD_CLIP) for g in total_grads["a_scales"]
        ]

        # Cosine LR schedule for a_scale: anneal effective LR from a_lr → 0
        # Multiply gradients by cos_factor so step size = a_lr * cos_factor (decreasing)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * step / iters))
        scaled_a_grads = [g * cos_factor for g in clipped_a_grads]

        # Split and apply
        alpha_grads = {"alphas": clipped_alpha_grads}
        a_scale_grads = {"a_scales": scaled_a_grads}

        w_opt.update(params, alpha_grads)
        a_opt.update(params, a_scale_grads)

        mx.eval(params.parameters(), w_opt.state, a_opt.state)

        loss_denom = weight_sum if sample_weights is not None else n_used
        loss_val = float(total_loss) / loss_denom
        loss_history.append(loss_val)

        if step % 50 == 0:
            if sample_weights is not None:
                loss_uw_val = float(total_loss_uw) / n_used
                pbar.set_postfix(loss_w=f"{loss_val:.4f}", loss_uw=f"{loss_uw_val:.4f}", b=f"{b_val:.1f}")
            else:
                pbar.set_postfix(loss=f"{loss_val:.4f}", b=f"{b_val:.1f}")
        if step % 200 == 0:
            if sample_weights is not None:
                loss_uw_val = float(total_loss_uw) / n_used
                print(f"  {block_name} iter {step}/{iters}  loss(w)={loss_val:.4f}  loss(uw)={loss_uw_val:.4f}  b={b_val:.1f}", flush=True)
            else:
                print(f"  {block_name} iter {step}/{iters}  loss={loss_val:.4f}  b={b_val:.1f}", flush=True)
        pbar.update(1)

    pbar.close()

    initial_loss = loss_history[0] if loss_history else None
    final_loss = loss_history[-1] if loss_history else None
    min_loss = min(loss_history) if loss_history else None

    metrics = {
        "block_name": block_name,
        "iters": iters,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "loss_history_stride100": loss_history[::100],
        "refined": prior_alphas is not None,
    }
    return params, metrics


# ---------------------------------------------------------------------------
# Finalise: apply hard rounding and extract quantised weights
# ---------------------------------------------------------------------------

def finalize_block(
    params: AdaRoundParams,
    W_fps_np: List[np.ndarray],
    w_scales_np: List[np.ndarray],
    linear_paths: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Commit hard rounding decisions (alpha >= 0) and return quantised weights.

    Returns
    -------
    dict mapping linear_path -> {'weight_int': np.ndarray, 'scale': np.ndarray,
                                  'a_scale': float}
    """
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for i, path in enumerate(linear_paths):
        alpha = params.alphas[i]
        W_fp_mx = mx.array(W_fps_np[i])
        s_mx = mx.array(w_scales_np[i])

        W_floor = mx.floor(W_fp_mx / s_mx)
        hard_delta = (alpha >= 0).astype(mx.float32)
        W_q_float = mx.clip(W_floor + hard_delta, params.qmin_w, params.qmax_w)
        mx.eval(W_q_float)

        result[path] = {
            "weight_int": np.array(W_q_float).astype(np.int8),
            "scale": np.array(s_mx),         # (out, 1)
            "a_scale": float(np.array(mx.abs(params.a_scales[i]))[0]),
            "bits_w": params.bits_w,
            "bits_a": params.bits_a,
        }
    return result


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AdaRound block-level W4A8 quantisation for SD3-Medium DiT"
    )
    parser.add_argument("--adaround-cache", type=Path, required=True,
                        help="Path to adaround_cache/ dir from cache_adaround_data.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for quantised weights")
    parser.add_argument("--iters", type=int, default=20000,
                        help="AdaRound iterations per block (default 20000)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Calibration samples per optimisation step (default 16)")
    parser.add_argument("--bits-w", type=int, default=4, choices=[4, 8],
                        help="Weight quantisation bits (default 4)")
    parser.add_argument("--bits-a", type=int, default=8,
                        help="Activation quantisation bits (default 8)")
    parser.add_argument("--w-lr", type=float, default=1e-3,
                        help="Adam lr for AdaRound alpha (default 1e-3)")
    parser.add_argument("--a-lr", type=float, default=4e-5,
                        help="Adam lr for activation scales (default 4e-5)")
    parser.add_argument("--blocks", type=str, default=None,
                        help="Comma-separated block names to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output")
    parser.add_argument("--resume", action="store_true",
                        help="Skip blocks whose .npz already exists in the output weights dir")
    parser.add_argument("--refine", action="store_true",
                        help="Warm-start from existing weights (polish previous results with more iters)")
    parser.add_argument("--htg-groups", type=Path, default=None,
                        help="htg_groups.json from src.htg_cluster; enables per-group optimization")
    parser.add_argument("--bb-config", type=Path, default=None,
                        help="bb_config.json from src.bayesianbits_optimize; uses per-layer bit widths")
    parser.add_argument("--poly-schedule", type=Path, default=None,
                        help="polynomial_clipping_schedule.json; enables σ-aware A8 fake quant")
    parser.add_argument("--sigma-map", type=Path, default=None,
                        help="Path to layer_statistics.json containing sigma_map "
                             "(default: auto-detect from activations dir)")
    parser.add_argument("--sigma-weighted", action="store_true",
                        help="Weight per-sample loss by 1/(σ+offset), emphasizing clean timesteps")
    parser.add_argument("--sigma-weight-offset", type=float, default=1.0,
                        help="Offset ε in w(σ)=1/(σ+ε). Lower = more aggressive (default: 1.0)")
    parser.add_argument("--derivative-weighted", action="store_true",
                        help="Also weight per-sample loss by |dα/dσ|, amplifying blocks where "
                             "clipping range is most sensitive to σ. Requires --poly-schedule.")
    parser.add_argument("--refine-skip-converged", type=float, default=None, metavar="PCT",
                        help="With --refine, skip blocks whose loss improved less than PCT%% "
                             "over the last 20%% of the prior run (e.g., 0.5 = skip if <0.5%% improvement)")
    parser.add_argument("--exclude-layers", type=str, default=None,
                        help="Comma-separated layer keys to keep in FP16 "
                             "(e.g., mm14_txt_mlp_fc2,mm20_txt_mlp_fc2)")
    parser.add_argument("--exclude-extreme-shift", action="store_true",
                        help="Shortcut: exclude mm14/20/21/22 txt mlp_fc2 (extreme adaLN shift layers)")
    parser.add_argument("--asymmetric-act", action="store_true",
                        help="Use asymmetric activation quantization with σ-dependent shift from poly schedule")
    args = parser.parse_args()

    if args.sigma_weighted and args.poly_schedule is None:
        parser.error("--sigma-weighted requires --poly-schedule (need per-sample sigmas)")
    if args.derivative_weighted and args.poly_schedule is None:
        parser.error("--derivative-weighted requires --poly-schedule")
    if args.asymmetric_act and args.poly_schedule is None:
        parser.error("--asymmetric-act requires --poly-schedule")

    # Build exclude set from CLI flags
    exclude_set: set = set()
    if args.exclude_extreme_shift:
        exclude_set.update([
            "mm14_txt_mlp_fc2", "mm20_txt_mlp_fc2",
            "mm21_txt_mlp_fc2", "mm22_txt_mlp_fc2",
        ])
    if args.exclude_layers:
        exclude_set.update(k.strip() for k in args.exclude_layers.split(","))
    if exclude_set:
        print(f"  FP16 exclusions: {sorted(exclude_set)}")

    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"

    if config_path.exists() and not args.force and not args.resume and not args.refine:
        print(f"Output already exists at {args.output}. Use --force to overwrite, --resume to continue, or --refine to polish.")
        return

    # ------------------------------------------------------------------
    # Load cache metadata
    # ------------------------------------------------------------------
    meta_path = args.adaround_cache / "metadata.json"
    if not meta_path.exists():
        print(f"Error: metadata.json not found at {meta_path}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    block_names: List[str] = meta["block_names"]
    n_mm: int = meta["n_mm_blocks"]
    all_sample_files = sorted((args.adaround_cache / "samples").glob("*.npz"))

    if not all_sample_files:
        print("Error: no sample files found in cache")
        return

    if args.blocks:
        selected_blocks = set(args.blocks.split(","))
        block_names = [b for b in block_names if b in selected_blocks]

    # Load optional polynomial schedule and build per-sample sigma array
    poly_schedule: Optional[Dict] = None
    sample_sigmas: Optional[np.ndarray] = None
    if args.poly_schedule is not None:
        with open(args.poly_schedule) as f:
            poly_schedule = json.load(f)
        # Build sigma map from layer_statistics.json or --sigma-map
        sigma_map_path = args.sigma_map
        if sigma_map_path is None:
            # Try auto-detect from calibration dir
            for candidate in [
                args.adaround_cache.parent / "activations" / "layer_statistics.json",
                args.adaround_cache / "layer_statistics.json",
            ]:
                if candidate.exists():
                    sigma_map_path = candidate
                    break
        sigma_map: Optional[Dict[str, float]] = None
        if sigma_map_path and sigma_map_path.exists():
            with open(sigma_map_path) as f:
                stats = json.load(f)
            sigma_map = {k: float(v) for k, v in stats.get("sigma_map", {}).items()}
        sample_sigmas = _extract_sample_sigmas(all_sample_files, sigma_map)
        print(f"  Poly schedule: {args.poly_schedule} ({len(poly_schedule.get('layers', {}))} layers)")
        if sigma_map:
            print(f"  Sigma range: [{sample_sigmas.min():.3f}, {sample_sigmas.max():.3f}]")

    # Load optional HTG groups and BB config
    htg_groups: Optional[Dict] = None
    bb_config: Optional[Dict] = None
    if args.htg_groups is not None:
        with open(args.htg_groups) as f:
            htg_groups = json.load(f)
    if args.bb_config is not None:
        with open(args.bb_config) as f:
            bb_config = json.load(f)

    htg_mode = htg_groups is not None
    print("=== AdaRound Optimisation ===")
    print(f"  Cache:   {args.adaround_cache}  ({len(all_sample_files)} samples)")
    print(f"  Blocks:  {len(block_names)}")
    print(f"  Iters:   {args.iters}  batch={args.batch_size}")
    print(f"  W{args.bits_w}A{args.bits_a}  w_lr={args.w_lr}  a_lr={args.a_lr}")
    if htg_mode:
        print(f"  HTG mode: {htg_groups['n_groups']} groups")
    if bb_config is not None:
        print(f"  BB config: per-layer bit widths from {args.bb_config}")
    print(f"  Output:  {args.output}\n")

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    from diffusionkit.mlx import DiffusionPipeline

    print("=== Loading Pipeline ===")
    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    print("✓ Pipeline loaded\n")

    mmdit = pipeline.mmdit

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Load per-image pooled text embeddings for correct modulation params.
    # These are saved by cache_adaround_data.py as pooled/{img_idx:04d}.npz.
    # Each calibration sample uses its image's pooled embedding so that
    # adaLN modulation params match what was used during data collection.
    # ------------------------------------------------------------------
    img_pooled_map = load_pooled_embeddings(args.adaround_cache)
    if img_pooled_map:
        print(f"=== Modulation Params ===")
        print(f"  Loaded pooled embeddings for {len(img_pooled_map)} images: {sorted(img_pooled_map.keys())}")
        # Build per-sample pooled array aligned with all_sample_files
        sample_image_indices = get_sample_image_indices(all_sample_files)
        unique_sample_ids = sorted(set(sample_image_indices))
        print(f"  Sample image indices found: {unique_sample_ids}")
        # Build valid_sample_mask — exclude images whose pooled embeddings are missing.
        # Using fallback (wrong) embeddings gives wrong adaLN modulation → NaN gradients.
        missing = [i for i in unique_sample_ids if i not in img_pooled_map]
        if missing:
            print(f"  WARNING: {len(missing)} image(s) missing pooled embeddings {missing}")
            print(f"  These samples will be EXCLUDED from optimization to prevent NaN gradients.")
            print(f"  Run cache_adaround_data with --resume to fill missing pooled files.")
        global_valid_sample_mask = np.array([
            img_id in img_pooled_map for img_id in sample_image_indices
        ])
        n_excluded = int((~global_valid_sample_mask).sum())
        if n_excluded:
            print(f"  Excluding {n_excluded}/{len(sample_image_indices)} samples with missing embeddings")
        global_sample_pooled = np.stack([
            img_pooled_map[idx] for idx in sample_image_indices
            if idx in img_pooled_map
        ])
        # Filter all_sample_files and sample_sigmas to valid (non-fallback) samples only
        all_sample_files = [sf for sf, ok in zip(all_sample_files, global_valid_sample_mask) if ok]
        sample_image_indices = [idx for idx, ok in zip(sample_image_indices, global_valid_sample_mask) if ok]
        if sample_sigmas is not None:
            sample_sigmas = sample_sigmas[global_valid_sample_mask]
        print(f"  Using {len(all_sample_files)} samples from {sorted(set(sample_image_indices))} images")
        print("✓ Per-sample modulation will be computed on the fly\n")
    else:
        print("WARNING: No pooled embeddings found in cache. Run cache_adaround_data.py --resume to generate them.")
        print("  Modulation params will be computed from a dummy prompt (may cause high initial loss).\n")
        # Fall back to dummy prompt modulation
        unique_ts = set()
        for sf in all_sample_files[:50]:
            npz = np.load(sf)
            for key in npz.files:
                if key.endswith("__arg2") and key.startswith("mm"):
                    unique_ts.add(float(npz[key].flat[0]))
                    break
        unique_ts = sorted(unique_ts)
        cfg_weight = meta.get("cfg_weight", 5.0)
        _, dummy_pooled = pipeline.encode_text("a photo", cfg_weight, "")
        mx.eval(dummy_pooled)
        ts_mx = mx.array(unique_ts).astype(pipeline.activation_dtype)
        mmdit.cache_modulation_params(dummy_pooled, ts_mx)
        global_sample_pooled = None

    # ------------------------------------------------------------------
    # Helper to save one block's quantised weights
    # ------------------------------------------------------------------
    def _save_block_weights(
        weights_dir: Path,
        block_name: str,
        block: Any,
        is_mm: bool,
        params: "AdaRoundParams",
        bits_w: int,
        block_exclude_keys: Optional[set] = None,
    ) -> List[str]:
        linears = get_block_linears(block, is_mm)
        # Filter out excluded layers (same logic as optimize_block)
        if block_exclude_keys:
            linears = [
                (p, l, g) for p, l, g in linears
                if _path_to_poly_key(block_name, is_mm, p) not in block_exclude_keys
            ]
        linear_paths = [p for p, _, _ in linears]
        linear_layers = [l for _, l, _ in linears]
        W_fps_np = [np.array(l.weight) for l in linear_layers]
        w_scales_np = [compute_per_channel_scale(W, bits_w) for W in W_fps_np]
        quant_weights = finalize_block(params, W_fps_np, w_scales_np, linear_paths)
        out_npz = weights_dir / f"{block_name}.npz"
        save_dict: Dict[str, np.ndarray] = {}
        for i, (path, data) in enumerate(quant_weights.items()):
            safe = path.replace(".", "_")
            save_dict[f"{safe}__weight_int"] = data["weight_int"]
            save_dict[f"{safe}__scale"] = data["scale"]
            save_dict[f"{safe}__a_scale"] = np.array([data["a_scale"]])
            # Save raw alpha for exact warm-start (avoids dtype mismatch in reconstruct)
            save_dict[f"{safe}__alpha"] = np.array(params.alphas[i])
        np.savez_compressed(out_npz, **save_dict)
        return linear_paths

    # ------------------------------------------------------------------
    # HTG mode: one optimization pass per group
    # ------------------------------------------------------------------
    if htg_mode:
        global_groups = htg_groups["global_groups"]
        all_group_metrics: Dict[str, List[Dict]] = {}

        for group_id, group_info in tqdm(global_groups.items(), desc="Groups"):
            group_step_indices = set(group_info["timestep_indices"])
            # Filter sample files: step index encoded as f.stem.split("_")[1]
            group_sample_files = [
                f for f in all_sample_files
                if int(f.stem.split("_")[1]) in group_step_indices
            ]

            tqdm.write(f"\n=== Group {group_id}: {len(group_sample_files)} samples ===")

            # Output directory for this group
            group_out_dir = args.output / f"group_{group_id}"
            group_out_dir.mkdir(parents=True, exist_ok=True)
            group_weights_dir = group_out_dir / "weights"
            group_weights_dir.mkdir(exist_ok=True)

            group_metrics: List[Dict] = []

            for block_name in tqdm(block_names, desc=f"  Blocks (group {group_id})", leave=False):
                is_mm = block_name.startswith("mm")
                idx = int(block_name[2:] if is_mm else block_name[3:])
                block = (
                    mmdit.multimodal_transformer_blocks[idx]
                    if is_mm
                    else mmdit.unified_transformer_blocks[idx]
                )

                block_data = load_block_data(block_name, group_sample_files)
                if not block_data or "arg0" not in block_data:
                    tqdm.write(f"  SKIP {block_name} (group {group_id}): no data")
                    continue

                # Determine bit width: prefer BB config if available
                effective_bits_w = args.bits_w
                if bb_config is not None:
                    group_key = f"group_{group_id}"
                    group_bb = bb_config.get(group_key, {})
                    # Block-level median bit-width from per-layer BB decisions
                    block_bits_list = [
                        v for k, v in group_bb.items()
                        if k.startswith(f"{block_name}.")
                    ]
                    if block_bits_list:
                        from statistics import median
                        effective_bits_w = int(round(median(block_bits_list)))
                        effective_bits_w = max(4, min(8, effective_bits_w))

                n_samples = block_data["arg0"].shape[0]
                tqdm.write(f"\n  [{block_name}] group={group_id}  samples={n_samples}  W{effective_bits_w}")

                # Build group-specific sample_sigmas if poly schedule is active
                group_sample_sigmas = None
                if poly_schedule is not None:
                    group_sample_sigmas = _extract_sample_sigmas(
                        group_sample_files, sigma_map
                    )

                params, metrics = optimize_block(
                    block=block,
                    block_name=block_name,
                    is_mm=is_mm,
                    block_data=block_data,
                    iters=args.iters,
                    batch_size=args.batch_size,
                    bits_w=effective_bits_w,
                    bits_a=args.bits_a,
                    w_lr=args.w_lr,
                    a_lr=args.a_lr,
                    poly_schedule=poly_schedule,
                    sample_sigmas=group_sample_sigmas,
                    sigma_weighted=args.sigma_weighted,
                    sigma_weight_offset=args.sigma_weight_offset,
                    derivative_weighted=args.derivative_weighted,
                    exclude_keys=exclude_set if exclude_set else None,
                    asymmetric_act=args.asymmetric_act,
                )
                tqdm.write(f"  {block_name} (group {group_id}) done — "
                           f"final_loss={metrics['final_loss']:.4f}")

                linear_paths = _save_block_weights(
                    group_weights_dir, block_name, block, is_mm, params, effective_bits_w,
                    block_exclude_keys=exclude_set if exclude_set else None,
                )
                metrics["quant_paths"] = linear_paths
                metrics["group_id"] = group_id
                metrics["bits_w"] = effective_bits_w
                group_metrics.append(metrics)

                del block_data

            # Save per-group config
            group_config = {
                "format": "adaround_v1",
                "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
                "group_id": group_id,
                "bits_w": args.bits_w,
                "bits_a": args.bits_a,
                "iters": args.iters,
                "batch_size": args.batch_size,
                "n_blocks_quantised": len(group_metrics),
                "block_metrics": group_metrics,
            }
            with open(group_out_dir / "config.json", "w") as f:
                json.dump(group_config, f, indent=2)

            all_group_metrics[group_id] = group_metrics

        # Save top-level config
        total_blocks = sum(len(v) for v in all_group_metrics.values())
        top_config = {
            "format": "adaround_htg_v1",
            "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
            "n_groups": htg_groups["n_groups"],
            "bits_w": args.bits_w,
            "bits_a": args.bits_a,
            "iters": args.iters,
            "batch_size": args.batch_size,
            "n_blocks_quantised_total": total_blocks,
            "group_metrics": all_group_metrics,
        }
        with open(config_path, "w") as f:
            json.dump(top_config, f, indent=2)

        print(f"\n=== Done (HTG) ===")
        print(f"  {total_blocks} block×group optimizations")
        print(f"  Config:  {config_path}")
        print(f"  Weights: {args.output}/group_*/weights/")
        return

    # ------------------------------------------------------------------
    # Standard (non-HTG) mode: single pass over all sample files
    # ------------------------------------------------------------------
    weights_dir = args.output / "weights"
    weights_dir.mkdir(exist_ok=True)

    # Load existing metrics if resuming or refining
    prev_block_metrics: Dict[str, Dict] = {}
    if (args.resume or args.refine) and config_path.exists():
        with open(config_path) as f:
            prev_config = json.load(f)
        prev_metrics_list = prev_config.get("block_metrics", [])
        for m in prev_metrics_list:
            prev_block_metrics[m["block_name"]] = m
        if args.resume:
            print(f"  Resuming: {len(prev_metrics_list)} blocks already done")
    if args.resume:
        all_metrics_by_name = dict(prev_block_metrics)
    elif args.refine:
        # Preserve prior metrics so interrupted refine runs don't lose history
        all_metrics_by_name = dict(prev_block_metrics)
    else:
        all_metrics_by_name = {}

    n_total = len(block_names)
    n_skipped_converged = 0
    for block_idx, block_name in enumerate(block_names, 1):
        # Skip blocks already completed
        if args.resume and (weights_dir / f"{block_name}.npz").exists():
            print(f"[{block_idx}/{n_total}] SKIP {block_name}: already done (resume mode)")
            continue

        # Skip converged blocks in refine mode
        if args.refine and args.refine_skip_converged is not None:
            prior_m = prev_block_metrics.get(block_name)
            if prior_m is not None:
                hist = prior_m.get("loss_history_stride100", [])
                if len(hist) >= 5:
                    # Check improvement over last 20% of history
                    tail_start = max(0, len(hist) - len(hist) // 5)
                    tail = hist[tail_start:]
                    tail_first = tail[0]
                    tail_last = tail[-1]
                    if tail_first > 0:
                        tail_improvement_pct = 100.0 * (tail_first - tail_last) / tail_first
                    else:
                        tail_improvement_pct = 0.0
                    if tail_improvement_pct < args.refine_skip_converged:
                        print(
                            f"[{block_idx}/{n_total}] SKIP {block_name}: "
                            f"converged (tail improvement {tail_improvement_pct:.2f}% "
                            f"< {args.refine_skip_converged}%)"
                        )
                        n_skipped_converged += 1
                        continue

        # Determine block object and type
        is_mm = block_name.startswith("mm")
        idx = int(block_name[2:] if is_mm else block_name[3:])
        block = (
            mmdit.multimodal_transformer_blocks[idx]
            if is_mm
            else mmdit.unified_transformer_blocks[idx]
        )

        # Load calibration data for this block
        block_data = load_block_data(block_name, all_sample_files)
        if not block_data or "arg0" not in block_data:
            tqdm.write(f"  SKIP {block_name}: no calibration data")
            continue

        n_samples = block_data["arg0"].shape[0]

        # Load prior weights for warm-start if refining
        block_prior_alphas = None
        block_prior_a_scales = None
        prior_npz = weights_dir / f"{block_name}.npz"
        if args.refine and prior_npz.exists():
            linears = get_block_linears(block, is_mm)
            if exclude_set:
                linears = [
                    (p, l, g) for p, l, g in linears
                    if _path_to_poly_key(block_name, is_mm, p) not in exclude_set
                ]
            lp = [p for p, _, _ in linears]
            ll = [l for _, l, _ in linears]
            wfp = [np.array(l.weight) for l in ll]
            block_prior_alphas, block_prior_a_scales = load_prior_block(
                prior_npz, lp, wfp, args.bits_w
            )
            print(f"\n[{block_idx}/{n_total}] {block_name}  is_mm={is_mm}  samples={n_samples}  (refine)", flush=True)
        else:
            print(f"\n[{block_idx}/{n_total}] {block_name}  is_mm={is_mm}  samples={n_samples}", flush=True)

        # Build per-sample pooled array for this block (filter to samples that had data)
        block_sample_pooled = None
        if global_sample_pooled is not None:
            safe = block_name.replace(".", "_")
            prefix = safe + "__arg"
            valid_mask = [
                any(k.startswith(prefix) for k in np.load(sf).files)
                for sf in all_sample_files
            ]
            block_sample_pooled = global_sample_pooled[valid_mask]

        # Optimise
        params, metrics = optimize_block(
            block=block,
            block_name=block_name,
            is_mm=is_mm,
            block_data=block_data,
            iters=args.iters,
            batch_size=args.batch_size,
            bits_w=args.bits_w,
            bits_a=args.bits_a,
            w_lr=args.w_lr,
            a_lr=args.a_lr,
            poly_schedule=poly_schedule,
            sample_sigmas=sample_sigmas,
            prior_alphas=block_prior_alphas,
            prior_a_scales=block_prior_a_scales,
            mmdit=mmdit,
            sample_pooled=block_sample_pooled,
            activation_dtype=pipeline.activation_dtype,
            sigma_weighted=args.sigma_weighted,
            sigma_weight_offset=args.sigma_weight_offset,
            derivative_weighted=args.derivative_weighted,
            exclude_keys=exclude_set if exclude_set else None,
            asymmetric_act=args.asymmetric_act,
        )
        if metrics.get("refined") and metrics["initial_loss"] is not None:
            improvement = metrics["initial_loss"] - metrics["final_loss"]
            pct = 100 * improvement / metrics["initial_loss"] if metrics["initial_loss"] > 0 else 0
            print(
                f"  {block_name} done — loss: {metrics['initial_loss']:.1f} → {metrics['final_loss']:.1f} "
                f"({pct:+.1f}%){' (converged)' if abs(pct) < 0.5 else ''}"
            )
        else:
            print(f"  {block_name} done — final_loss={metrics['final_loss']:.4f}")

        linear_paths = _save_block_weights(
            weights_dir, block_name, block, is_mm, params, args.bits_w,
            block_exclude_keys=exclude_set if exclude_set else None,
        )
        metrics["quant_paths"] = linear_paths
        all_metrics_by_name[block_name] = metrics

        # Free calibration data
        del block_data

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    # Emit metrics in block order (preserves prior metrics for untouched blocks)
    all_metrics = [
        all_metrics_by_name[bn] for bn in block_names
        if bn in all_metrics_by_name
    ]
    config = {
        "format": "adaround_v1",
        "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
        "bits_w": args.bits_w,
        "bits_a": args.bits_a,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "w_lr": args.w_lr,
        "a_lr": args.a_lr,
        "sigma_weighted": args.sigma_weighted,
        "sigma_weight_offset": args.sigma_weight_offset,
        "derivative_weighted": args.derivative_weighted,
        "exclude_layers": sorted(exclude_set) if exclude_set else [],
        "asymmetric_act": args.asymmetric_act,
        "n_blocks_quantised": len(all_metrics),
        "block_metrics": all_metrics,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== Done ===")
    print(f"  {len(all_metrics)} blocks quantised")
    if n_skipped_converged > 0:
        print(f"  {n_skipped_converged} blocks skipped (converged, threshold={args.refine_skip_converged}%)")
    print(f"  Config:  {config_path}")
    print(f"  Weights: {weights_dir}")

    # Refine summary: show which blocks improved and whether more runs would help
    refined_metrics = [m for m in all_metrics if m.get("refined")]
    if refined_metrics:
        print(f"\n=== Refine Summary ===")
        converged = 0
        for m in refined_metrics:
            init = m["initial_loss"]
            final = m["final_loss"]
            if init and init > 0:
                pct = 100 * (init - final) / init
                status = "converged" if abs(pct) < 0.5 else f"{pct:+.1f}%"
                if abs(pct) < 0.5:
                    converged += 1
            else:
                status = "n/a"
            print(f"  {m['block_name']:>8}  {init:.1f} → {final:.1f}  ({status})")
        if converged == len(refined_metrics):
            print(f"\n  All {converged} blocks converged — further refine runs unlikely to help.")
        else:
            improving = len(refined_metrics) - converged
            print(f"\n  {improving}/{len(refined_metrics)} blocks still improving — another --refine run may help.")


if __name__ == "__main__":
    main()
