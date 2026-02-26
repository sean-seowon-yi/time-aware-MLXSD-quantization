"""
Bayesian Bits (BB) mixed-precision weight quantization for SD3-Medium DiT.

Implements hierarchical nested quantization with L0 hard-concrete gating to
automatically select per-layer bit-widths (W2, W4, or W8) per HTG group
(arXiv 2005.07093 — Bayesian Bits).

Core math:
  Nested scales (per-output-channel):
    s_2 = absmax / (2^(2-1) - 1)   = absmax / 1
    s_4 = s_2 / 3                   (4-bit residual scale)
    s_8 = s_4 / 9                   (8-bit residual scale)

  Recursive hierarchical quantization:
    x_q2 = s_2 * round_ste(W / s_2)
    x_q4 = s_4 * round_ste((W - x_q2) / s_4)
    x_q8 = s_8 * round_ste((W - x_q2 - x_q4) / s_8)
    W_q  = x_q2 + gate_4 * (x_q4 + gate_8 * x_q8)

  L0 regulariser (expected bit cost beyond base-2):
    reg = gating_lambda * (p4 * 2 + p4 * p8 * 4)

  Loss = block_reconstruction_L2 + reg

Usage:
    conda run -n diffusionkit python -m src.bayesianbits_optimize \\
        --adaround-cache calibration_data_100/adaround_cache \\
        --htg-groups htg_groups.json \\
        --output bb_config.json \\
        [--iters 20000] [--batch-size 16] [--gating-lambda 0.01] [--bits-a 8]

Output: bb_config.json
  {
    "group_0": {"mm0.image_transformer_block.attn.q_proj": 4, ...},
    "group_1": {...},
    ...
  }
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

from src.adaround_optimize import (
    fake_quant_per_tensor,
    get_block_linears,
    _get_nested,
    _set_nested,
)
from src.cache_adaround_data import load_block_data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BB_ZETA: float = 1.1
BB_GAMMA: float = -0.1
BB_BETA: float = 2.0 / 3.0

# Nested scale factors (Bayesian Bits §3.2)
SCALE_FACTOR_4: float = 3.0   # s_4 = s_2 / 3
SCALE_FACTOR_8: float = 9.0   # s_8 = s_4 / 9 = s_2 / 27


# ---------------------------------------------------------------------------
# L0 hard-concrete primitives
# ---------------------------------------------------------------------------

def hc_prob_pos(log_alpha: mx.array) -> mx.array:
    """P(gate > 0) = sigmoid(log_alpha - beta * log(-gamma/zeta))."""
    bias = BB_BETA * math.log(-BB_GAMMA / BB_ZETA)
    return mx.sigmoid(log_alpha - bias)


def sample_gate(log_alpha: mx.array) -> mx.array:
    """Hard-concrete sample stretched to [0, 1]."""
    u = mx.random.uniform(shape=log_alpha.shape)
    # Avoid log(0): clamp u away from 0 and 1
    u = mx.clip(u, 1e-6, 1.0 - 1e-6)
    s = mx.sigmoid((mx.log(u) - mx.log(1.0 - u) + log_alpha) / BB_BETA)
    return mx.clip(s * (BB_ZETA - BB_GAMMA) + BB_GAMMA, 0.0, 1.0)


def deterministic_gate(log_alpha: mx.array) -> mx.array:
    """Deterministic gate: sigmoid(log_alpha / beta) stretched to [0,1], then threshold 0.5."""
    s = mx.sigmoid(log_alpha / BB_BETA)
    stretched = mx.clip(s * (BB_ZETA - BB_GAMMA) + BB_GAMMA, 0.0, 1.0)
    return (stretched > 0.5).astype(mx.float32)


# ---------------------------------------------------------------------------
# STE round
# ---------------------------------------------------------------------------

def round_ste(x: mx.array) -> mx.array:
    """Straight-through estimator for rounding."""
    return mx.stop_gradient(mx.round(x) - x) + x


# ---------------------------------------------------------------------------
# Hierarchical quantization
# ---------------------------------------------------------------------------

def compute_bb_scales(W_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute nested BB scales for a weight matrix.

    s_2 : absmax / 1  (per-output-channel, shape (out, 1))
    s_4 : s_2 / SCALE_FACTOR_4
    s_8 : s_4 / SCALE_FACTOR_8
    """
    absmax = np.abs(W_np).max(axis=1, keepdims=True)  # (out, 1)
    absmax = np.maximum(absmax, 1e-8)
    s_2 = absmax / 1.0
    s_4 = s_2 / SCALE_FACTOR_4
    s_8 = s_4 / SCALE_FACTOR_8
    return s_2, s_4, s_8


def hierarchical_quant(
    W: mx.array,
    s_2: mx.array,
    s_4: mx.array,
    s_8: mx.array,
    gate_4: mx.array,
    gate_8: mx.array,
) -> mx.array:
    """
    Hierarchical Bayesian Bits quantization.

    W_q = x_q2 + gate_4 * (x_q4 + gate_8 * x_q8)
    """
    x_q2 = s_2 * round_ste(W / s_2)
    x_q4 = s_4 * round_ste((W - x_q2) / s_4)
    x_q8 = s_8 * round_ste((W - x_q2 - x_q4) / s_8)
    return x_q2 + gate_4 * (x_q4 + gate_8 * x_q8)


# ---------------------------------------------------------------------------
# BB parameter container
# ---------------------------------------------------------------------------

class BBParams(nn.Module):
    """
    Trainable Bayesian Bits parameters for one transformer block.

    log_alphas_4[i] : L0 gate logits for 4-bit activation, shape = W.shape
    log_alphas_8[i] : L0 gate logits for 8-bit activation, shape = W.shape
    a_scales[i]     : Per-tensor activation scale, shape (1,)
    """

    def __init__(
        self,
        W_fps_np: List[np.ndarray],
        bits_a: int = 8,
        init_log_alpha: float = 0.0,
    ):
        super().__init__()
        self.bits_a = bits_a
        self.qmin_a = -(2 ** (bits_a - 1))
        self.qmax_a = 2 ** (bits_a - 1) - 1

        log_alphas_4: List[mx.array] = []
        log_alphas_8: List[mx.array] = []
        a_scales: List[mx.array] = []

        for W_np in W_fps_np:
            log_alphas_4.append(mx.full(W_np.shape, init_log_alpha))
            log_alphas_8.append(mx.full(W_np.shape, init_log_alpha))
            a_scales.append(mx.array([1.0]))

        self.log_alphas_4 = log_alphas_4
        self.log_alphas_8 = log_alphas_8
        self.a_scales = a_scales


# ---------------------------------------------------------------------------
# BB proxy layer (replaces nn.Linear during optimization)
# ---------------------------------------------------------------------------

class _BBProxy:
    """
    Replaces nn.Linear during BBParams optimization.

    Applies:
      1. Per-tensor LSQ fake-quantisation to input activation.
      2. Hierarchical BB soft-quantisation to weight.
    """

    def __init__(
        self,
        orig_layer: Any,
        soft_weight: mx.array,
        a_scale: mx.array,
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
        bias = getattr(self._orig, "bias", None)
        if bias is not None:
            y = y + bias
        return y

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


# ---------------------------------------------------------------------------
# Loss function for BB block reconstruction
# ---------------------------------------------------------------------------

def _lp_loss(pred: mx.array, tgt: mx.array, p: float = 2.0) -> mx.array:
    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_tgt = tgt.reshape(tgt.shape[0], -1)
    return ((flat_pred - flat_tgt).abs() ** p).sum(axis=1).mean()


def bb_block_loss_fn(
    params: BBParams,
    block: Any,
    is_mm: bool,
    linear_paths: List[str],
    linear_layers: List[Any],
    W_fps_np: List[np.ndarray],
    bb_scales: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    sample_inputs: List[mx.array],
    sample_kwargs: Dict[str, mx.array],
    fp_outputs: List[mx.array],
    gating_lambda: float,
) -> mx.array:
    """
    BB block-level reconstruction + L0 regularization loss.
    """
    n = len(linear_paths)

    soft_weights: List[mx.array] = []
    for i in range(n):
        W_fp_mx = mx.array(W_fps_np[i])
        s_2_mx = mx.array(bb_scales[i][0])
        s_4_mx = mx.array(bb_scales[i][1])
        s_8_mx = mx.array(bb_scales[i][2])

        # Sample stochastic gates during training
        gate_4 = sample_gate(params.log_alphas_4[i])
        gate_8 = sample_gate(params.log_alphas_8[i])

        soft_w = hierarchical_quant(W_fp_mx, s_2_mx, s_4_mx, s_8_mx, gate_4, gate_8)
        soft_weights.append(soft_w)

    # Patch linears with BB proxies
    orig_layers = []
    for i, path in enumerate(linear_paths):
        orig_layers.append(_get_nested(block, path))
        proxy = _BBProxy(
            linear_layers[i],
            soft_weights[i],
            params.a_scales[i],
            params.qmin_a,
            params.qmax_a,
        )
        _set_nested(block, path, proxy)

    try:
        if sample_kwargs:
            output = block(*sample_inputs, **sample_kwargs)
        else:
            output = block(*sample_inputs)
    finally:
        for path, orig in zip(linear_paths, orig_layers):
            _set_nested(block, path, orig)

    # Reconstruction loss
    if isinstance(output, (list, tuple)):
        rec = sum(_lp_loss(out, tgt) for out, tgt in zip(output, fp_outputs))
    else:
        rec = _lp_loss(output, fp_outputs[0])

    # L0 regularization: expected bit cost
    reg = mx.array(0.0)
    for i in range(n):
        p4 = hc_prob_pos(params.log_alphas_4[i]).mean()
        p8 = hc_prob_pos(params.log_alphas_8[i]).mean()
        reg = reg + (p4 * 2.0 + p4 * p8 * 4.0)
    reg = reg * gating_lambda

    return rec + reg


# ---------------------------------------------------------------------------
# Finalize: extract effective bit-width per layer
# ---------------------------------------------------------------------------

def finalize_bb_bits(params: BBParams, linear_paths: List[str]) -> Dict[str, int]:
    """
    Evaluate gates deterministically and return effective bit-width per layer.

    gate_4=0 → W2; gate_4=1, gate_8=0 → W4; gate_4=1, gate_8=1 → W8
    """
    result: Dict[str, int] = {}
    for i, path in enumerate(linear_paths):
        det_gate_4 = deterministic_gate(params.log_alphas_4[i])
        det_gate_8 = deterministic_gate(params.log_alphas_8[i])

        # Fraction of weights where gate is active
        frac_4 = float(det_gate_4.mean())
        frac_8 = float(det_gate_8.mean())

        if frac_4 < 0.5:
            bits = 2
        elif frac_8 < 0.5:
            bits = 4
        else:
            bits = 8

        result[path] = bits
    return result


# ---------------------------------------------------------------------------
# Per-group optimization
# ---------------------------------------------------------------------------

def optimize_block_bb(
    block: Any,
    block_name: str,
    is_mm: bool,
    block_data: Dict[str, np.ndarray],
    iters: int = 20000,
    batch_size: int = 16,
    bits_a: int = 8,
    w_lr: float = 1e-3,
    a_lr: float = 4e-5,
    gating_lambda: float = 0.01,
) -> Tuple[BBParams, Dict]:
    """
    Run BB optimization for a single transformer block on the given calibration data.
    """
    linears = get_block_linears(block, is_mm)
    linear_paths = [p for p, _, _ in linears]
    linear_layers = [l for _, l, _ in linears]
    W_fps_np = [np.array(l.weight) for l in linear_layers]
    bb_scales = [compute_bb_scales(W) for W in W_fps_np]

    params = BBParams(W_fps_np, bits_a=bits_a)

    w_opt = optim.Adam(learning_rate=w_lr)
    a_opt = optim.Adam(learning_rate=a_lr)

    n_samples = block_data["arg0"].shape[0]
    loss_and_grad_fn = nn.value_and_grad(params, bb_block_loss_fn)

    loss_history: List[float] = []
    pbar = tqdm(total=iters, desc=f"  BB {block_name}", unit="it", leave=False)

    for step in range(iters):
        idx = np.random.randint(0, n_samples, min(batch_size, n_samples))

        total_loss = mx.array(0.0)
        total_grads = None

        for si in idx:
            s_inputs: List[mx.array] = []
            s_kwargs: Dict[str, mx.array] = {}
            s_outputs: List[mx.array] = []

            for key in sorted(k for k in block_data if k.startswith("arg")):
                s_inputs.append(mx.array(block_data[key][si]))
            for key in (k for k in block_data if k.startswith("kw_")):
                kw_name = key[3:]
                s_kwargs[kw_name] = mx.array(block_data[key][si])
            for key in sorted(k for k in block_data if k.startswith("out")):
                s_outputs.append(mx.array(block_data[key][si]))

            loss_i, grads_i = loss_and_grad_fn(
                params, block, is_mm,
                linear_paths, linear_layers,
                W_fps_np, bb_scales,
                s_inputs, s_kwargs, s_outputs,
                gating_lambda,
            )
            total_loss = total_loss + loss_i

            if total_grads is None:
                total_grads = grads_i
            else:
                for i in range(len(params.log_alphas_4)):
                    total_grads["log_alphas_4"][i] = (
                        total_grads["log_alphas_4"][i] + grads_i["log_alphas_4"][i]
                    )
                    total_grads["log_alphas_8"][i] = (
                        total_grads["log_alphas_8"][i] + grads_i["log_alphas_8"][i]
                    )
                    total_grads["a_scales"][i] = (
                        total_grads["a_scales"][i] + grads_i["a_scales"][i]
                    )

        n_used = len(idx)
        for i in range(len(params.log_alphas_4)):
            total_grads["log_alphas_4"][i] = total_grads["log_alphas_4"][i] / n_used
            total_grads["log_alphas_8"][i] = total_grads["log_alphas_8"][i] / n_used
            total_grads["a_scales"][i] = total_grads["a_scales"][i] / n_used

        # Cosine LR for a_scale
        cos_factor = max(0.5 * (1.0 + math.cos(math.pi * step / iters)), 1e-8)

        gate_grads = {
            "log_alphas_4": total_grads["log_alphas_4"],
            "log_alphas_8": total_grads["log_alphas_8"],
        }
        a_scale_grads = {"a_scales": total_grads["a_scales"]}

        w_opt.update(params, gate_grads)
        a_opt.update(params, a_scale_grads)
        mx.eval(params.parameters(), w_opt.state, a_opt.state)

        loss_val = float(total_loss) / n_used
        loss_history.append(loss_val)

        if step % 500 == 0:
            pbar.set_postfix(loss=f"{loss_val:.4f}")
        pbar.update(1)

    pbar.close()

    metrics = {
        "block_name": block_name,
        "iters": iters,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history_stride100": loss_history[::100],
    }
    return params, metrics


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian Bits mixed-precision per-group optimization for SD3-Medium DiT"
    )
    parser.add_argument("--adaround-cache", type=Path, required=True,
                        help="Path to adaround_cache/ dir from cache_adaround_data.py")
    parser.add_argument("--htg-groups", type=Path, required=True,
                        help="htg_groups.json from src.htg_cluster")
    parser.add_argument("--output", type=Path, default=Path("bb_config.json"),
                        help="Output bb_config.json (default: bb_config.json)")
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bits-a", type=int, default=8)
    parser.add_argument("--w-lr", type=float, default=1e-3)
    parser.add_argument("--a-lr", type=float, default=4e-5)
    parser.add_argument("--gating-lambda", type=float, default=0.01,
                        help="L0 regularization strength (default: 0.01)")
    parser.add_argument("--blocks", type=str, default=None,
                        help="Comma-separated block names (default: all)")
    args = parser.parse_args()

    with open(args.htg_groups) as f:
        htg_groups = json.load(f)

    global_groups = htg_groups["global_groups"]
    n_groups = htg_groups["n_groups"]

    meta_path = args.adaround_cache / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    block_names: List[str] = meta["block_names"]
    n_mm: int = meta["n_mm_blocks"]
    all_sample_files = sorted((args.adaround_cache / "samples").glob("*.npz"))

    if args.blocks:
        selected = set(args.blocks.split(","))
        block_names = [b for b in block_names if b in selected]

    print("=== Bayesian Bits Optimization ===")
    print(f"  Cache:   {args.adaround_cache}  ({len(all_sample_files)} samples)")
    print(f"  Groups:  {n_groups}")
    print(f"  Blocks:  {len(block_names)}")
    print(f"  Iters:   {args.iters}  batch={args.batch_size}")
    print(f"  lambda_gate={args.gating_lambda}  A{args.bits_a}")
    print(f"  Output:  {args.output}\n")

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
    bb_config: Dict[str, Dict[str, int]] = {}

    for group_id, group_info in tqdm(global_groups.items(), desc="Groups"):
        group_step_indices = set(group_info["timestep_indices"])
        # Filter samples: step index is encoded as f.stem.split("_")[1]
        group_sample_files = [
            f for f in all_sample_files
            if int(f.stem.split("_")[1]) in group_step_indices
        ]

        tqdm.write(f"\n=== Group {group_id}: {len(group_sample_files)} samples ===")
        group_key = f"group_{group_id}"
        bb_config[group_key] = {}

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

            params, metrics = optimize_block_bb(
                block=block,
                block_name=block_name,
                is_mm=is_mm,
                block_data=block_data,
                iters=args.iters,
                batch_size=args.batch_size,
                bits_a=args.bits_a,
                w_lr=args.w_lr,
                a_lr=args.a_lr,
                gating_lambda=args.gating_lambda,
            )

            linears = get_block_linears(block, is_mm)
            linear_paths = [p for p, _, _ in linears]
            bits_per_path = finalize_bb_bits(params, linear_paths)

            for path, bits in bits_per_path.items():
                full_path = f"{block_name}.{path}"
                bb_config[group_key][full_path] = bits

            del block_data

    with open(args.output, "w") as f:
        json.dump(bb_config, f, indent=2)
    print(f"\n✓ BB config -> {args.output}")


if __name__ == "__main__":
    main()
