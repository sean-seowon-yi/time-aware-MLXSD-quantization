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
    conda run -n diffusionkit python -m src.adaround_optimize \\
        --adaround-cache /path/to/adaround_cache \\
        --output /path/to/quantized_weights_adaround \\
        [--iters 20000] [--batch-size 16] [--bits-w 4] [--bits-a 8]

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


def fake_quant_per_tensor(
    x: mx.array, scale: mx.array, qmin: int, qmax: int
) -> mx.array:
    """Symmetric per-tensor fake-quantise (STE straight-through estimator)."""
    s = mx.abs(scale) + 1e-8
    return mx.clip(mx.round(x / s), qmin, qmax) * s


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
            layer = _get_nested(block, full)
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
    ):
        self._orig = orig_layer
        self._soft_weight = soft_weight
        self._a_scale = a_scale
        self._qmin_a = qmin_a
        self._qmax_a = qmax_a

    def __call__(self, x: mx.array) -> mx.array:
        x_q = fake_quant_per_tensor(x, self._a_scale, self._qmin_a, self._qmax_a)
        y = x_q @ self._soft_weight.T
        if self._orig.bias is not None:
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
        for W_np in W_fps_np:
            scale_np = compute_per_channel_scale(W_np, bits_w)
            alphas.append(mx.array(init_alpha(W_np, scale_np)))
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
) -> mx.array:
    """
    Compute block-level reconstruction + round loss for one mini-batch sample.

    Patches each linear in the block with a _QuantProxy (soft AdaRound weight +
    LSQ activation), runs the block forward, restores original layers, then
    computes the loss.  MLX's lazy graph preserves gradients through alpha and
    a_scale even after the Python-level restoration.
    """
    n = len(linear_paths)

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
        proxy = _QuantProxy(
            linear_layers[i],
            soft_weights[i],
            params.a_scales[i],
            params.qmin_a,
            params.qmax_a,
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

    # Reconstruction loss
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
) -> Tuple[AdaRoundParams, Dict]:
    """
    Run AdaRound optimisation for a single transformer block.

    Returns
    -------
    params : AdaRoundParams
        Optimised alpha and a_scale (use finalize_block to extract hard weights).
    metrics : dict
        loss_history, final_rec_loss, etc.
    """
    linears = get_block_linears(block, is_mm)
    linear_paths = [p for p, _, _ in linears]
    linear_layers = [l for _, l, _ in linears]
    W_fps_np = [np.array(l.weight) for l in linear_layers]
    w_scales_np = [compute_per_channel_scale(W, bits_w) for W in W_fps_np]

    params = AdaRoundParams(W_fps_np, bits_w=bits_w, bits_a=bits_a)

    # Two optimisers: Adam for alpha (w_lr), Adam+cosine for a_scale (a_lr)
    w_opt = optim.Adam(learning_rate=w_lr)
    a_opt = optim.Adam(learning_rate=a_lr)

    temp_decay = LinearTempDecay(t_max=iters, warm_up=0.2, start_b=20.0, end_b=2.0)

    # Unpack all calibration samples
    n_samples = block_data["arg0"].shape[0]

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
    pbar = tqdm(total=iters, desc=f"  {block_name}", unit="it", leave=False)

    for step in range(iters):
        b_val = temp_decay(step)

        # Sample mini-batch
        idx = np.random.randint(0, n_samples, min(batch_size, n_samples))

        # Average loss over sampled indices
        total_loss = mx.array(0.0)
        total_grads = None

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

            loss_i, grads_i = loss_and_grad_fn(
                params, block, is_mm,
                linear_paths, linear_layers,
                W_fps_np, w_scales_np,
                s_inputs, s_kwargs, s_outputs,
                b_val,
            )
            total_loss = total_loss + loss_i

            if total_grads is None:
                total_grads = grads_i
            else:
                # Accumulate grads (same structure as params)
                for i in range(len(params.alphas)):
                    total_grads["alphas"][i] = (
                        total_grads["alphas"][i] + grads_i["alphas"][i]
                    )
                    total_grads["a_scales"][i] = (
                        total_grads["a_scales"][i] + grads_i["a_scales"][i]
                    )

        n_used = len(idx)
        # Average grads
        for i in range(len(params.alphas)):
            total_grads["alphas"][i] = total_grads["alphas"][i] / n_used
            total_grads["a_scales"][i] = total_grads["a_scales"][i] / n_used

        # Cosine LR schedule for a_scale (anneal from a_lr to 0)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * step / iters))
        scaled_a_grads_list = [
            g * (1.0 / max(cos_factor, 1e-8))  # scale grad so effective LR = a_lr*cos
            for g in total_grads["a_scales"]
        ]
        # Note: we apply w_opt to alphas and a_opt (with cosine) to a_scales separately

        # Split and apply
        alpha_grads = {"alphas": total_grads["alphas"]}
        a_scale_grads = {"a_scales": total_grads["a_scales"]}

        w_opt.update(params, alpha_grads)
        a_opt.update(params, a_scale_grads)

        mx.eval(params.parameters(), w_opt.state, a_opt.state)

        loss_val = float(total_loss) / n_used
        loss_history.append(loss_val)

        if step % 500 == 0:
            pbar.set_postfix(loss=f"{loss_val:.4f}", b=f"{b_val:.1f}")
        pbar.update(1)

    pbar.close()

    metrics = {
        "block_name": block_name,
        "iters": iters,
        "final_loss": loss_history[-1] if loss_history else None,
        "min_loss": min(loss_history) if loss_history else None,
        "loss_history_stride100": loss_history[::100],
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
    parser.add_argument("--htg-groups", type=Path, default=None,
                        help="htg_groups.json from src.htg_cluster; enables per-group optimization")
    parser.add_argument("--bb-config", type=Path, default=None,
                        help="bb_config.json from src.bayesianbits_optimize; uses per-layer bit widths")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"

    if config_path.exists() and not args.force:
        print(f"Output already exists at {args.output}. Use --force to overwrite.")
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
    # Helper to save one block's quantised weights
    # ------------------------------------------------------------------
    def _save_block_weights(
        weights_dir: Path,
        block_name: str,
        block: Any,
        is_mm: bool,
        params: "AdaRoundParams",
        bits_w: int,
    ) -> List[str]:
        linears = get_block_linears(block, is_mm)
        linear_paths = [p for p, _, _ in linears]
        linear_layers = [l for _, l, _ in linears]
        W_fps_np = [np.array(l.weight) for l in linear_layers]
        w_scales_np = [compute_per_channel_scale(W, bits_w) for W in W_fps_np]
        quant_weights = finalize_block(params, W_fps_np, w_scales_np, linear_paths)
        out_npz = weights_dir / f"{block_name}.npz"
        save_dict: Dict[str, np.ndarray] = {}
        for path, data in quant_weights.items():
            safe = path.replace(".", "_")
            save_dict[f"{safe}__weight_int"] = data["weight_int"]
            save_dict[f"{safe}__scale"] = data["scale"]
            save_dict[f"{safe}__a_scale"] = np.array([data["a_scale"]])
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
                )
                tqdm.write(f"  {block_name} (group {group_id}) done — "
                           f"final_loss={metrics['final_loss']:.4f}")

                linear_paths = _save_block_weights(
                    group_weights_dir, block_name, block, is_mm, params, effective_bits_w
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

    all_metrics: List[Dict] = []

    for block_name in tqdm(block_names, desc="Blocks", unit="block"):
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
        tqdm.write(f"\n[{block_name}]  is_mm={is_mm}  samples={n_samples}")

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
        )
        tqdm.write(
            f"  {block_name} done — final_loss={metrics['final_loss']:.4f}"
        )

        linear_paths = _save_block_weights(
            weights_dir, block_name, block, is_mm, params, args.bits_w
        )
        metrics["quant_paths"] = linear_paths
        all_metrics.append(metrics)

        # Free calibration data
        del block_data

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    config = {
        "format": "adaround_v1",
        "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
        "bits_w": args.bits_w,
        "bits_a": args.bits_a,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "w_lr": args.w_lr,
        "a_lr": args.a_lr,
        "n_blocks_quantised": len(all_metrics),
        "block_metrics": all_metrics,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== Done ===")
    print(f"  {len(all_metrics)} blocks quantised")
    print(f"  Config:  {config_path}")
    print(f"  Weights: {weights_dir}")


if __name__ == "__main__":
    main()
