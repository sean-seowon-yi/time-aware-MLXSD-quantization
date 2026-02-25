"""
TaQ-DiT Joint W+A Block-Level Reconstruction for SD3-Medium DiT.

Implements TaQ-DiT (Zhao et al. 2024, arxiv 2411.14172) joint optimization:
  - Per-channel W4 AdaRound: learned alpha initialized from rounding residuals
  - Per-layer, PER-TIMESTEP A8 scales: separate learned scale for each (layer, timestep)
  - Block-level reconstruction: full transformer block output vs FP16 cache
  - B-annealing: round_loss coefficient b decays 20→2 after 20% warmup

Key difference from adaround_optimize.py (AdaRound baseline):
  AdaRound:  one global activation scale per linear layer — timestep-agnostic
  TaQ-DiT:   a_scales matrix (n_layers × n_timesteps) — captures that diffusion
             activations vary significantly across the noise schedule (σ→1 high-noise
             vs σ→0 low-noise regimes require different quantization granularity)

Output layout:
    <output>/
        config.json                  quantization config (format: taqdit_v1)
        weights/
            {block_name}.npz         hard-rounded int weights + per-channel scale
        taqdit_act_config.json       per-timestep per-layer activation scales
                                     (per_timestep_quant_config_v4, compatible with
                                      load_adaround_model.py --quant-config)

Usage:
    conda run -n diffusionkit python -m src.taqdit_optimize \\
        --adaround-cache calibration_data/adaround_cache \\
        --output quantized_weights_taqdit \\
        [--iters 20000] [--batch-size 16] [--bits-w 4] [--bits-a 8] [--blocks mm0,mm1]

    # Inference with learned per-timestep activation scales:
    conda run -n diffusionkit python -m src.load_adaround_model \\
        --adaround-output quantized_weights_taqdit \\
        --quant-config quantized_weights_taqdit/taqdit_act_config.json \\
        --prompt "a tabby cat on a table" --output-image taqdit_out.png
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
    GAMMA,
    ZETA,
    ROUND_WEIGHT,
    LP_NORM,
    LinearTempDecay,
    rectified_sigmoid,
    compute_per_channel_scale,
    init_alpha,
    fake_quant_per_tensor,
    get_block_linears,
    _get_nested,
    _set_nested,
    _lp_loss,
    _round_loss,
)


# ---------------------------------------------------------------------------
# Data loading with timestep tracking
# ---------------------------------------------------------------------------

def load_block_data_with_ts(
    block_name: str,
    sample_files: List[Path],
    step_indices: List[int],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load block I/O stacked across all samples, retaining per-sample timestep index.

    Sample filenames follow the pattern ``{img:04d}_{step:03d}.npz``.  The step
    index (the last ``_``-separated integer before ``.npz``) is matched against
    ``step_indices`` to produce a per-sample index into that list.

    Parameters
    ----------
    block_name   : e.g. ``"mm3"``
    sample_files : sorted list of .npz paths from the adaround_cache/samples dir
    step_indices : step indices used during cache collection (from metadata.json
                   ``key_timesteps``), e.g. [0, 4, 8, ..., 96]

    Returns
    -------
    block_data    : ``{'arg0': (N, ...), 'out0': (N, ...), ...}`` — same format as
                    :func:`~src.cache_adaround_data.load_block_data`
    sample_ts_idx : int32 array of shape (N,); ``sample_ts_idx[i]`` is the index
                    into ``step_indices`` that sample i belongs to
    """
    safe = block_name.replace(".", "_")
    prefix = safe + "__"
    step_to_ts_idx = {si: idx for idx, si in enumerate(step_indices)}

    per_sample: List[Dict[str, np.ndarray]] = []
    per_sample_ts_idx: List[int] = []

    for path in sample_files:
        # Filename: "{img:04d}_{step:03d}.npz"  — last "_"-field is step index
        stem = path.stem
        parts = stem.split("_")
        try:
            step_idx = int(parts[-1])
        except (ValueError, IndexError):
            continue

        if step_idx not in step_to_ts_idx:
            continue

        npz = np.load(path)
        sample = {
            k[len(prefix):]: npz[k]
            for k in npz.files
            if k.startswith(prefix)
        }
        if sample:
            per_sample.append(sample)
            per_sample_ts_idx.append(step_to_ts_idx[step_idx])

    if not per_sample:
        return {}, np.array([], dtype=np.int32)

    common_keys = set(per_sample[0].keys())
    for s in per_sample[1:]:
        common_keys &= set(s.keys())

    block_data = {k: np.stack([s[k] for s in per_sample]) for k in sorted(common_keys)}
    sample_ts_idx = np.array(per_sample_ts_idx, dtype=np.int32)
    return block_data, sample_ts_idx


# ---------------------------------------------------------------------------
# TaQ-DiT parameter container
# ---------------------------------------------------------------------------

class TaqDitParams(nn.Module):
    """
    Trainable parameters for TaQ-DiT joint W+A block-level reconstruction.

    alphas[i]                  : AdaRound alpha for layer i, shape = W.shape
    a_scales                   : activation scales, shape (n_layers, n_timesteps),
                                 a_scales[i, t] = scale for (layer i, timestep t)

    W_fps_np and w_scales_np are kept as numpy so they receive no gradients.
    """

    def __init__(
        self,
        W_fps_np: List[np.ndarray],
        n_timesteps: int,
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
        self.n_layers = len(W_fps_np)
        self.n_timesteps = n_timesteps

        # Weight rounding (same as AdaRoundParams)
        alphas: List[mx.array] = []
        for W_np in W_fps_np:
            scale_np = compute_per_channel_scale(W_np, bits_w)
            alphas.append(mx.array(init_alpha(W_np, scale_np)))
        self.alphas = alphas

        # Per-timestep activation scales: 2D matrix (n_layers, n_timesteps)
        # Initialized to 1.0 everywhere
        self.a_scales = mx.ones((len(W_fps_np), n_timesteps))


# ---------------------------------------------------------------------------
# Per-timestep quantization proxy
# ---------------------------------------------------------------------------

class _TaqDitQuantProxy:
    """
    Like _QuantProxy in adaround_optimize.py but takes a 0-d activation scale
    (per timestep) rather than a per-layer global scale.
    """

    def __init__(
        self,
        orig_layer: Any,
        soft_weight: mx.array,
        a_scale: mx.array,   # 0-d or (1,) — scale for this specific timestep
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
        return getattr(self._orig, name)


# ---------------------------------------------------------------------------
# TaQ-DiT block loss function
# ---------------------------------------------------------------------------

def taqdit_loss_fn(
    params: TaqDitParams,
    block: Any,
    is_mm: bool,
    linear_paths: List[str],
    linear_layers: List[Any],
    W_fps_np: List[np.ndarray],
    w_scales_np: List[np.ndarray],
    sample_inputs: List[mx.array],
    sample_kwargs: Dict[str, mx.array],
    fp_outputs: List[mx.array],
    b_val: float,
    ts_idx: int,
) -> mx.array:
    """
    Block-level reconstruction loss with per-timestep activation scales.

    Identical in structure to ``block_loss_fn`` in adaround_optimize.py, except
    ``params.a_scales[i, ts_idx]`` is used as the activation scale for layer i
    at the given timestep index, enabling the optimizer to learn per-timestep
    activation quantization granularity.
    """
    n = len(linear_paths)

    # Build soft-quantised weights (gradient flows through params.alphas)
    soft_weights: List[mx.array] = []
    for i in range(n):
        W_fp_mx = mx.array(W_fps_np[i])
        s_mx = mx.array(w_scales_np[i])
        r = rectified_sigmoid(params.alphas[i])
        W_floor = mx.floor(W_fp_mx / s_mx)
        soft_w = mx.clip(W_floor + r, params.qmin_w, params.qmax_w) * s_mx
        soft_weights.append(soft_w)

    # Patch each linear with a timestep-specific proxy
    orig_layers = []
    for i, path in enumerate(linear_paths):
        orig_layers.append(_get_nested(block, path))
        proxy = _TaqDitQuantProxy(
            linear_layers[i],
            soft_weights[i],
            params.a_scales[i, ts_idx],  # 0-d scalar in the computation graph
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

    if isinstance(output, (list, tuple)):
        rec = sum(_lp_loss(out, tgt, LP_NORM) for out, tgt in zip(output, fp_outputs))
    else:
        rec = _lp_loss(output, fp_outputs[0], LP_NORM)

    rnd = _round_loss(params.alphas, b_val)
    return rec + ROUND_WEIGHT * rnd


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def optimize_block_taqdit(
    block: Any,
    block_name: str,
    is_mm: bool,
    block_data: Dict[str, np.ndarray],
    sample_ts_idx: np.ndarray,
    n_timesteps: int,
    iters: int = 20000,
    batch_size: int = 16,
    bits_w: int = 4,
    bits_a: int = 8,
    w_lr: float = 1e-3,
    a_lr: float = 4e-5,
) -> Tuple[TaqDitParams, Dict]:
    """
    Run TaQ-DiT joint W+A optimization for a single transformer block.

    Returns
    -------
    params  : TaqDitParams with optimized alphas and per-timestep a_scales matrix
    metrics : dict with loss_history, final_loss, n_timesteps, etc.
    """
    linears = get_block_linears(block, is_mm)
    linear_paths = [p for p, _, _ in linears]
    linear_layers = [l for _, l, _ in linears]
    W_fps_np = [np.array(l.weight) for l in linear_layers]
    w_scales_np = [compute_per_channel_scale(W, bits_w) for W in W_fps_np]

    params = TaqDitParams(W_fps_np, n_timesteps=n_timesteps, bits_w=bits_w, bits_a=bits_a)

    # Two optimizers: Adam for alpha (w_lr), Adam for per-timestep a_scales (a_lr)
    w_opt = optim.Adam(learning_rate=w_lr)
    a_opt = optim.Adam(learning_rate=a_lr)

    temp_decay = LinearTempDecay(t_max=iters, warm_up=0.2, start_b=20.0, end_b=2.0)

    n_samples = block_data["arg0"].shape[0]
    loss_and_grad_fn = nn.value_and_grad(params, taqdit_loss_fn)

    n_layers = len(linear_paths)
    loss_history: List[float] = []
    pbar = tqdm(total=iters, desc=f"  {block_name}", unit="it", leave=False)

    for step in range(iters):
        b_val = temp_decay(step)

        # Cosine LR annealing for a_scales: reduce a_opt lr over training
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * step / iters))
        a_opt.learning_rate = max(a_lr * cos_factor, 1e-8)

        idx = np.random.randint(0, n_samples, min(batch_size, n_samples))

        total_loss = mx.array(0.0)
        total_grads: Optional[Dict] = None

        for si in idx:
            ts_idx = int(sample_ts_idx[si])

            s_inputs: List[mx.array] = []
            s_kwargs: Dict[str, mx.array] = {}
            s_outputs: List[mx.array] = []

            for key in sorted(k for k in block_data if k.startswith("arg")):
                s_inputs.append(mx.array(block_data[key][si]))
            for key in (k for k in block_data if k.startswith("kw_")):
                s_kwargs[key[3:]] = mx.array(block_data[key][si])
            for key in sorted(k for k in block_data if k.startswith("out")):
                s_outputs.append(mx.array(block_data[key][si]))

            loss_i, grads_i = loss_and_grad_fn(
                params, block, is_mm,
                linear_paths, linear_layers,
                W_fps_np, w_scales_np,
                s_inputs, s_kwargs, s_outputs,
                b_val, ts_idx,
            )
            total_loss = total_loss + loss_i

            if total_grads is None:
                total_grads = grads_i
            else:
                for i in range(n_layers):
                    total_grads["alphas"][i] = (
                        total_grads["alphas"][i] + grads_i["alphas"][i]
                    )
                total_grads["a_scales"] = total_grads["a_scales"] + grads_i["a_scales"]

        n_used = len(idx)
        for i in range(n_layers):
            total_grads["alphas"][i] = total_grads["alphas"][i] / n_used
        total_grads["a_scales"] = total_grads["a_scales"] / n_used

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
        "n_timesteps": n_timesteps,
        "final_loss": loss_history[-1] if loss_history else None,
        "min_loss": min(loss_history) if loss_history else None,
        "loss_history_stride100": loss_history[::100],
    }
    return params, metrics


# ---------------------------------------------------------------------------
# Finalize: hard-round weights and extract per-timestep activation scales
# ---------------------------------------------------------------------------

def finalize_block_taqdit(
    params: TaqDitParams,
    W_fps_np: List[np.ndarray],
    w_scales_np: List[np.ndarray],
    linear_paths: List[str],
) -> Dict[str, Dict]:
    """
    Commit hard rounding decisions (alpha >= 0) and extract quantized weights.

    a_scale stored per linear = mean across all timesteps (for V1 load compatibility).
    Per-timestep a_scales are exported separately via :func:`build_act_config`.

    Returns
    -------
    dict mapping linear_path -> {weight_int, scale, a_scale, bits_w, bits_a}
    """
    result: Dict[str, Dict] = {}
    a_scales_np = np.array(params.a_scales)   # (n_layers, n_timesteps)

    for i, path in enumerate(linear_paths):
        alpha = params.alphas[i]
        W_fp_mx = mx.array(W_fps_np[i])
        s_mx = mx.array(w_scales_np[i])

        W_floor = mx.floor(W_fp_mx / s_mx)
        hard_delta = (alpha >= 0).astype(mx.float32)
        W_q_float = mx.clip(W_floor + hard_delta, params.qmin_w, params.qmax_w)
        mx.eval(W_q_float)

        # Mean a_scale across timesteps for backwards compatibility with V1 load
        mean_a_scale = float(np.abs(a_scales_np[i]).mean())

        result[path] = {
            "weight_int": np.array(W_q_float).astype(np.int8),
            "scale": np.array(s_mx),        # (out, 1)
            "a_scale": mean_a_scale,
            "bits_w": params.bits_w,
            "bits_a": params.bits_a,
        }
    return result


# ---------------------------------------------------------------------------
# Build activation config from learned per-timestep a_scales
# ---------------------------------------------------------------------------

def _block_path_to_act_name(block_name: str, is_mm: bool, local_path: str) -> str:
    """
    Convert a block-internal linear path to the activation-config layer naming
    convention used by ``load_adaround_model.apply_act_quant_hooks``.

    Examples
    --------
    >>> _block_path_to_act_name("mm3", True, "image_transformer_block.attn.q_proj")
    'mm3.img.attn.q_proj'
    >>> _block_path_to_act_name("mm3", True, "text_transformer_block.mlp.fc2")
    'mm3.txt.mlp.fc2'
    >>> _block_path_to_act_name("uni0", False, "transformer_block.attn.o_proj")
    'uni0.attn.o_proj'
    """
    if is_mm:
        if local_path.startswith("image_transformer_block."):
            return f"{block_name}.img.{local_path[len('image_transformer_block.'):]}"
        if local_path.startswith("text_transformer_block."):
            return f"{block_name}.txt.{local_path[len('text_transformer_block.'):]}"
    else:
        if local_path.startswith("transformer_block."):
            return f"{block_name}.{local_path[len('transformer_block.'):]}"
    # Fallback (should not happen with standard DiffusionKit blocks)
    return f"{block_name}.{local_path}"


def build_act_config(
    params: TaqDitParams,
    block_name: str,
    is_mm: bool,
    linear_paths: List[str],
    step_indices: List[int],
    bits_a: int = 8,
) -> Dict[int, Dict[str, Dict]]:
    """
    Produce per-timestep activation config entries for this block.

    Returns
    -------
    dict: ``{step_idx: {layer_name: {"bits": bits_a, "scale": float}}}``
    """
    a_scales_np = np.abs(np.array(params.a_scales))   # (n_layers, n_timesteps)

    per_ts: Dict[int, Dict[str, Dict]] = {}
    for ts_idx, step_idx in enumerate(step_indices):
        per_ts[step_idx] = {}
        for layer_idx, local_path in enumerate(linear_paths):
            act_name = _block_path_to_act_name(block_name, is_mm, local_path)
            scale_val = float(a_scales_np[layer_idx, ts_idx])
            per_ts[step_idx][act_name] = {
                "bits": bits_a,
                "scale": max(scale_val, 1e-8),
            }
    return per_ts


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TaQ-DiT joint W+A block-level reconstruction for SD3-Medium DiT"
    )
    parser.add_argument("--adaround-cache", type=Path, required=True,
                        help="Path to adaround_cache/ dir from cache_adaround_data.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for quantized weights + activation config")
    parser.add_argument("--iters", type=int, default=20000,
                        help="Optimization iterations per block (default 20000)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Calibration samples per optimization step (default 16)")
    parser.add_argument("--bits-w", type=int, default=4, choices=[4, 8],
                        help="Weight quantization bits (default 4)")
    parser.add_argument("--bits-a", type=int, default=8,
                        help="Activation quantization bits (default 8)")
    parser.add_argument("--w-lr", type=float, default=1e-3,
                        help="Adam lr for AdaRound alpha (default 1e-3)")
    parser.add_argument("--a-lr", type=float, default=4e-5,
                        help="Adam lr for per-timestep activation scales (default 4e-5)")
    parser.add_argument("--blocks", type=str, default=None,
                        help="Comma-separated block names to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    weights_dir = args.output / "weights"
    weights_dir.mkdir(exist_ok=True)
    config_path = args.output / "config.json"

    if config_path.exists() and not args.force:
        print(f"Output already exists at {args.output}. Use --force to overwrite.")
        return

    meta_path = args.adaround_cache / "metadata.json"
    if not meta_path.exists():
        print(f"Error: metadata.json not found at {meta_path}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    block_names: List[str] = meta["block_names"]
    n_mm: int = meta["n_mm_blocks"]
    step_indices: List[int] = meta["key_timesteps"]
    n_timesteps = len(step_indices)
    sample_files = sorted((args.adaround_cache / "samples").glob("*.npz"))

    if not sample_files:
        print("Error: no sample files found in cache")
        return

    if args.blocks:
        selected_blocks = set(args.blocks.split(","))
        block_names = [b for b in block_names if b in selected_blocks]

    print("=== TaQ-DiT Joint W+A Optimization ===")
    print(f"  Cache:      {args.adaround_cache}  ({len(sample_files)} samples)")
    print(f"  Blocks:     {len(block_names)}")
    ts_preview = step_indices[:6]
    print(f"  Timesteps:  {n_timesteps} (indices: {ts_preview}"
          f"{'...' if n_timesteps > 6 else ''})")
    print(f"  Iters:      {args.iters}  batch={args.batch_size}")
    print(f"  W{args.bits_w}A{args.bits_a}  w_lr={args.w_lr}  a_lr={args.a_lr}")
    print(f"  Output:     {args.output}\n")

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
    all_metrics: List[Dict] = []

    # Accumulate per-timestep activation config across all blocks
    global_per_ts: Dict[int, Dict[str, Dict]] = {si: {} for si in step_indices}

    # ------------------------------------------------------------------
    # Process blocks sequentially
    # ------------------------------------------------------------------
    for block_name in tqdm(block_names, desc="Blocks", unit="block"):
        is_mm = block_name.startswith("mm")
        idx = int(block_name[2:] if is_mm else block_name[3:])
        block = (
            mmdit.multimodal_transformer_blocks[idx]
            if is_mm
            else mmdit.unified_transformer_blocks[idx]
        )

        block_data, sample_ts_idx = load_block_data_with_ts(
            block_name, sample_files, step_indices
        )
        if not block_data or "arg0" not in block_data:
            tqdm.write(f"  SKIP {block_name}: no calibration data")
            continue

        n_samples = block_data["arg0"].shape[0]
        ts_present = len(set(sample_ts_idx.tolist()))
        tqdm.write(
            f"\n[{block_name}]  is_mm={is_mm}  samples={n_samples}"
            f"  timesteps_present={ts_present}/{n_timesteps}"
        )

        params, metrics = optimize_block_taqdit(
            block=block,
            block_name=block_name,
            is_mm=is_mm,
            block_data=block_data,
            sample_ts_idx=sample_ts_idx,
            n_timesteps=n_timesteps,
            iters=args.iters,
            batch_size=args.batch_size,
            bits_w=args.bits_w,
            bits_a=args.bits_a,
            w_lr=args.w_lr,
            a_lr=args.a_lr,
        )
        tqdm.write(f"  {block_name} done — final_loss={metrics['final_loss']:.4f}")

        # Finalize weights
        linears = get_block_linears(block, is_mm)
        linear_paths = [p for p, _, _ in linears]
        linear_layers = [l for _, l, _ in linears]
        W_fps_np = [np.array(l.weight) for l in linear_layers]
        w_scales_np = [compute_per_channel_scale(W, args.bits_w) for W in W_fps_np]

        quant_weights = finalize_block_taqdit(params, W_fps_np, w_scales_np, linear_paths)

        out_npz = weights_dir / f"{block_name}.npz"
        save_dict: Dict[str, np.ndarray] = {}
        for path, data in quant_weights.items():
            safe = path.replace(".", "_")
            save_dict[f"{safe}__weight_int"] = data["weight_int"]
            save_dict[f"{safe}__scale"] = data["scale"]
            save_dict[f"{safe}__a_scale"] = np.array([data["a_scale"]])
        np.savez_compressed(out_npz, **save_dict)

        # Accumulate per-timestep activation config
        block_per_ts = build_act_config(
            params, block_name, is_mm, linear_paths, step_indices, args.bits_a
        )
        for step_idx, layer_cfgs in block_per_ts.items():
            global_per_ts[step_idx].update(layer_cfgs)

        metrics["quant_paths"] = linear_paths
        all_metrics.append(metrics)

        del block_data

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    config = {
        "format": "taqdit_v1",
        "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
        "bits_w": args.bits_w,
        "bits_a": args.bits_a,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "w_lr": args.w_lr,
        "a_lr": args.a_lr,
        "n_timesteps": n_timesteps,
        "step_indices": step_indices,
        "n_blocks_quantised": len(all_metrics),
        "block_metrics": all_metrics,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Save activation config (per_timestep_quant_config_v4 format,
    # compatible with load_adaround_model.py --quant-config)
    # ------------------------------------------------------------------
    act_config = {
        "format": "per_timestep_quant_config_v4",
        "per_timestep": {str(k): v for k, v in global_per_ts.items()},
        "sigma_map": {},          # no sigma info at block-cache level
        "outlier_config": {},     # two-scale handling not used in joint opt
        "summary": {
            "total_timesteps": n_timesteps,
            "total_layers": len({
                lname
                for ts_layers in global_per_ts.values()
                for lname in ts_layers
            }),
            "total_decisions": sum(len(v) for v in global_per_ts.values()),
            "activation_bits": args.bits_a,
            "scale_source": "learned_taqdit_joint",
        },
        "metadata": {
            "bits_w": args.bits_w,
            "bits_a": args.bits_a,
            "n_blocks_quantised": len(all_metrics),
        },
    }
    act_config_path = args.output / "taqdit_act_config.json"
    with open(act_config_path, "w") as f:
        json.dump(act_config, f, indent=2)

    print(f"\n=== Done ===")
    print(f"  {len(all_metrics)} blocks quantized")
    print(f"  Config:     {config_path}")
    print(f"  Weights:    {weights_dir}")
    print(f"  Act config: {act_config_path}")
    print()
    print("  To run W4A8 inference with learned per-timestep activation scales:")
    print(f"    conda run -n diffusionkit python -m src.load_adaround_model \\")
    print(f"      --adaround-output {args.output} \\")
    print(f"      --quant-config {act_config_path} \\")
    print(f"      --prompt 'a tabby cat on a table' --output-image taqdit_out.png")


if __name__ == "__main__":
    main()
