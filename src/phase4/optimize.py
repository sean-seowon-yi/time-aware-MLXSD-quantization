"""Phase 4 AdaRound optimisation: block-wise weight rounding.

For each transformer block:
  1. Load cached (img_in, txt_in, vec, pe, img_out, txt_out) from collection.
  2. Replace every target nn.Linear with a _QuantProxy that holds learnable
     rounding offsets (alpha) and uses soft-quantized weights in the forward.
  3. Freeze the block, unfreeze only proxy alphas.
  4. Minimise block-level reconstruction MSE:
         L = ||block_fp_output - block_proxy_output||²
     aggregated over a mini-batch of cached samples.
  5. Hard-binarise: round_delta = (alpha > 0).astype(int8).
  6. Build nn.QuantizedLinear via repack.build_quantized_linear + W4A8Linear.
  7. Inject into pipeline.mmdit and checkpoint after every block.

Running in float32 throughout to avoid NaN in attention backward on
high-magnitude activations.  With CSB applied, activation ranges should
be well-balanced and gradients stable.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ..phase2.quantize import W4A8Linear
from .repack import build_quantized_linear, unpack_from_quantized_linear

logger = logging.getLogger(__name__)

# Linear paths inside one transformer half-block
_LINEAR_PATHS = [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "mlp.fc1",
    "mlp.fc2",
]


# ---------------------------------------------------------------------------
# Helpers: nested get/set
# ---------------------------------------------------------------------------

def _get_nested(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_nested(obj: Any, path: str, val: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], val)


# ---------------------------------------------------------------------------
# Scale / floor helpers
# ---------------------------------------------------------------------------

def _compute_quant_params(
    W: np.ndarray,
    bits: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric per-group scale, floor(W/s), and scale expanded to weight shape."""
    d_out, d_in = W.shape
    n_groups = d_in // group_size
    qmax = 2 ** (bits - 1) - 1

    W_g = W.reshape(d_out, n_groups, group_size)
    scales = np.maximum(np.max(np.abs(W_g), axis=-1) / qmax, 1e-8).astype(np.float32)
    scales_exp = np.repeat(scales, group_size, axis=1)           # (d_out, d_in)
    floor_w = np.floor(W / scales_exp).astype(np.float32)
    return scales, floor_w, scales_exp


# ---------------------------------------------------------------------------
# Soft-quantization helpers (used inside loss graph)
# ---------------------------------------------------------------------------

def _fake_quant_a8(x: mx.array) -> mx.array:
    """Dynamic per-tensor symmetric A8 fake-quantisation."""
    scale = mx.maximum(mx.max(mx.abs(x)) / 127.0, mx.array(1e-8))
    return mx.clip(mx.round(x / scale), -128, 127) * scale


# ---------------------------------------------------------------------------
# _QuantProxy: drop-in replacement during AdaRound block optimisation
# ---------------------------------------------------------------------------

class _QuantProxy(nn.Module):
    """Soft-quantized linear for AdaRound block reconstruction.

    Holds the FP16 balanced weights plus learnable rounding offsets (alpha).
    Activation quantisation uses dynamic A8; b_inv is applied for o_proj/fc2.
    """

    def __init__(
        self,
        W_fp: np.ndarray,
        scales_exp: np.ndarray,
        floor_w: np.ndarray,
        bias: np.ndarray | None,
        b_inv: np.ndarray | None,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.alpha = mx.full(W_fp.shape, alpha_init, dtype=mx.float32)  # learnable
        self._W_fp      = mx.array(W_fp,       dtype=mx.float32)
        self._scales    = mx.array(scales_exp, dtype=mx.float32)
        self._floor_w   = mx.array(floor_w,    dtype=mx.float32)
        self._bias      = mx.array(bias,       dtype=mx.float32) if bias is not None else None
        self._b_inv     = mx.array(b_inv,      dtype=mx.float32) if b_inv is not None else None

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        if self._b_inv is not None:
            x = x * self._b_inv
        x_q = _fake_quant_a8(x)
        w_soft = (self._floor_w + mx.sigmoid(self.alpha)) * self._scales
        out = x_q @ w_soft.T
        if self._bias is not None:
            out = out + self._bias
        return out.astype(orig_dtype)


# ---------------------------------------------------------------------------
# Block-wise optimisation
# ---------------------------------------------------------------------------

def _get_tb_linears(
    tb: Any,
    b_inv_store: dict[str, np.ndarray],
    layer_prefix: str,
    bits: int,
    group_size: int,
) -> list[tuple[str, _QuantProxy, np.ndarray, np.ndarray]]:
    """Build _QuantProxy for each target linear in a transformer half-block.

    Returns list of (dot_path, proxy, scales, floor_w) tuples.
    """
    result = []
    for rel_path in _LINEAR_PATHS:
        try:
            linear = _get_nested(tb, rel_path)
        except AttributeError:
            continue
        if not isinstance(linear, nn.Linear):
            continue

        W_fp = np.array(linear.weight, dtype=np.float32)
        bias = np.array(linear.bias) if getattr(linear, "bias", None) is not None else None
        layer_name = f"{layer_prefix}.{rel_path}"
        b_inv = b_inv_store.get(layer_name)

        scales, floor_w, scales_exp = _compute_quant_params(W_fp, bits, group_size)
        proxy = _QuantProxy(W_fp, scales_exp, floor_w, bias, b_inv)
        result.append((rel_path, proxy, scales, floor_w))

    return result


def _get_tb_linears_from_checkpoint(
    tb: Any,
    b_inv_store: dict[str, np.ndarray],
    layer_prefix: str,
    bits: int,
    group_size: int,
) -> list[tuple[str, _QuantProxy, np.ndarray, np.ndarray]]:
    """Build _QuantProxy from an already-quantized W4A8Linear (for --refine).

    Unpacks the existing quantized weights and uses them as floor_w.
    """
    result = []
    for rel_path in _LINEAR_PATHS:
        try:
            layer = _get_nested(tb, rel_path)
        except AttributeError:
            continue
        if not hasattr(layer, "qlinear"):
            continue

        w_int, scales = unpack_from_quantized_linear(layer.qlinear, bits, group_size)
        d_out, d_in = w_int.shape
        n_groups = scales.shape[1]
        scales_exp = np.repeat(scales, group_size, axis=1)[:, :d_in]

        floor_w = w_int.astype(np.float32)
        W_fp = floor_w * scales_exp  # dequantized approximation (placeholder)

        bias = np.array(layer.qlinear.bias) if getattr(layer.qlinear, "bias", None) is not None else None
        layer_name = f"{layer_prefix}.{rel_path}"
        b_inv = None
        if hasattr(layer, "b_inv"):
            b_inv = np.array(layer.b_inv, dtype=np.float32)
        elif layer_name in b_inv_store:
            b_inv = b_inv_store[layer_name]

        # alpha_init=-5 so sigmoid(-5)≈0.007 — initial weights match checkpoint
        proxy = _QuantProxy(W_fp, scales_exp, floor_w, bias, b_inv, alpha_init=-5.0)
        result.append((rel_path, proxy, scales, floor_w))

    return result


def _optimize_block(
    block_idx: int,
    block: Any,
    block_data_dir: Path,
    b_inv_store: dict[str, np.ndarray],
    phase2_meta: dict,
    config: dict,
    refine: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Run AdaRound for all linears in one multimodal transformer block.

    Returns dict: {full_layer_name → (w_int, scales, bias)} for both image
    and text half-blocks.
    """
    bits       = phase2_meta["bits"]
    group_size = phase2_meta["group_size"]
    n_iters    = config["n_iters"]
    lr         = config["lr"]
    batch_size = config["batch_size"]
    qmax       = 2 ** (bits - 1) - 1

    # --- Load samples ---
    sample_files = sorted(block_data_dir.glob("*.npz"))
    if not sample_files:
        logger.warning("Block %02d: no samples found in %s — skipping", block_idx, block_data_dir)
        return {}

    samples = [np.load(str(f)) for f in sample_files]
    has_modulation = "img_mod" in samples[0] if samples else False
    logger.info("Block %02d: %d samples (modulation=%s)", block_idx, len(samples),
                "saved" if has_modulation else "missing")

    img_tb = block.image_transformer_block
    txt_tb = block.text_transformer_block

    img_prefix = f"blocks.{block_idx}.image"
    txt_prefix = f"blocks.{block_idx}.text"

    if refine:
        img_linears = _get_tb_linears_from_checkpoint(img_tb, b_inv_store, img_prefix, bits, group_size)
        txt_linears = _get_tb_linears_from_checkpoint(txt_tb, b_inv_store, txt_prefix, bits, group_size)
    else:
        img_linears = _get_tb_linears(img_tb, b_inv_store, img_prefix, bits, group_size)
        txt_linears = _get_tb_linears(txt_tb, b_inv_store, txt_prefix, bits, group_size)

    if not img_linears and not txt_linears:
        logger.warning("Block %02d: no quantisable linears found", block_idx)
        return {}

    # --- Install proxies ---
    for rel_path, proxy, _, _ in img_linears:
        _set_nested(img_tb, rel_path, proxy)
    for rel_path, proxy, _, _ in txt_linears:
        _set_nested(txt_tb, rel_path, proxy)

    # Freeze block; only proxy alphas are trainable
    block.freeze()
    for _, proxy, _, _ in img_linears + txt_linears:
        proxy.unfreeze()

    # --- Optimiser ---
    optimizer = optim.Adam(learning_rate=lr)

    # Pre-initialise _modulation_params dicts on the transformer sub-blocks.
    # cache_modulation_params (which normally fills these) is not called during
    # optimisation — we set the per-sample modulation saved during collection.
    if not has_modulation:
        raise RuntimeError(
            f"Block {block_idx:02d}: calibration samples missing 'img_mod'/'txt_mod'. "
            f"Re-run collection (remove --skip-collection) to capture modulation params."
        )

    if not hasattr(img_tb, "_modulation_params"):
        img_tb._modulation_params = {}
    if not hasattr(txt_tb, "_modulation_params"):
        txt_tb._modulation_params = {}

    def loss_fn(block_ref: Any) -> mx.array:
        batch_idx = np.random.randint(0, len(samples), size=batch_size)
        total_loss = mx.array(0.0)
        for si in batch_idx:
            s = samples[si]
            img_in  = mx.array(s["img_in"],  dtype=mx.float32)
            txt_in  = mx.array(s["txt_in"],  dtype=mx.float32)
            vec     = mx.array(s["vec"],     dtype=mx.float32)
            pe      = mx.array(s["pe"],      dtype=mx.float32)
            img_tgt = mx.array(s["img_out"], dtype=mx.float32)
            txt_tgt = mx.array(s["txt_out"], dtype=mx.float32) if "txt_out" in s else None

            # Set modulation params for this sample so pre_sdpa can look them up
            ts_key = float(vec[0]) if vec.size > 1 else float(vec)
            img_tb._modulation_params[ts_key] = mx.array(s["img_mod"], dtype=mx.float32)
            txt_tb._modulation_params[ts_key] = mx.array(s["txt_mod"], dtype=mx.float32)

            img_pred, txt_pred = block_ref(img_in, txt_in, vec, pe)
            img_pred = img_pred.astype(mx.float32)
            total_loss = total_loss + mx.mean((img_pred - img_tgt) ** 2)

            if txt_pred is not None and txt_tgt is not None:
                txt_pred = txt_pred.astype(mx.float32)
                total_loss = total_loss + mx.mean((txt_pred - txt_tgt) ** 2)
        return total_loss / batch_size

    loss_and_grad = nn.value_and_grad(block, loss_fn)

    from tqdm import tqdm

    n_nan = 0
    loss_val = float("inf")
    all_losses: list[float] = []
    avg_window = 50           # steps to average over
    converge_patience = 3     # consecutive flat averages before early stop
    converge_rtol = 0.02      # 2% relative improvement threshold
    flat_count = 0
    prev_avg = None
    t0 = time.time()
    pbar = tqdm(range(1, n_iters + 1), desc=f"  Block {block_idx:02d}", unit="it", leave=True)
    for step in pbar:
        loss, grads = loss_and_grad(block)
        loss_val = float(loss.item())

        if not np.isfinite(loss_val):
            n_nan += 1
            old_lr = float(optimizer.learning_rate.item()) if hasattr(optimizer.learning_rate, 'item') else float(optimizer.learning_rate)
            new_lr = old_lr * 0.5
            pbar.set_postfix_str(f"NaN! lr={new_lr:.1e}")
            optimizer = optim.Adam(learning_rate=new_lr)
            mx.eval(optimizer.state)
            continue

        optimizer.update(block, grads)
        mx.eval(block, optimizer.state)
        all_losses.append(loss_val)

        if step % 50 == 0 or step == 1:
            cur_avg = np.mean(all_losses[-avg_window:])
            pbar.set_postfix_str(f"loss={loss_val:.5f} avg={cur_avg:.5f}")
            logger.info("  block %02d  step %4d/%d  loss=%.5f  avg=%.5f",
                        block_idx, step, n_iters, loss_val, cur_avg)
            if prev_avg is not None:
                rel_improvement = (prev_avg - cur_avg) / max(abs(prev_avg), 1e-8)
                if rel_improvement < converge_rtol:
                    flat_count += 1
                    if flat_count >= converge_patience:
                        logger.info("  block %02d  converged at step %d (avg stable for %d checks)",
                                    block_idx, step, converge_patience)
                        break
                else:
                    flat_count = 0
            prev_avg = cur_avg

    if n_nan > 0:
        logger.warning("  block %02d  finished with %d NaN steps", block_idx, n_nan)
    logger.info("  block %02d  done in %.1f s  final_loss=%.5f", block_idx, time.time() - t0, loss_val)

    # --- Extract results and restore original linears ---
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}

    for prefix, tb, linears_list in [
        (img_prefix, img_tb, img_linears),
        (txt_prefix, txt_tb, txt_linears),
    ]:
        for rel_path, proxy, scales, floor_w in linears_list:
            alpha_np = np.array(proxy.alpha)
            round_delta = (alpha_np > 0).astype(np.int8)
            w_int = np.array(floor_w, dtype=np.int8) + round_delta
            w_int = np.clip(w_int, -qmax, qmax).astype(np.int8)
            bias = np.array(proxy._bias) if proxy._bias is not None else None
            layer_name = f"{prefix}.{rel_path}"
            results[layer_name] = (w_int, scales, bias)

            # Restore original linear
            _set_nested(tb, rel_path, proxy._W_fp)   # placeholder; real restore below

    # Actually restore the original nn.Linear modules (stored in proxy._W_fp is wrong)
    # We need to reload them — done by the caller after inject
    block.unfreeze()
    return results


# ---------------------------------------------------------------------------
# Model-wide optimisation
# ---------------------------------------------------------------------------

def optimize_all_blocks(
    pipeline,
    registry: list[dict],
    calibration: dict,
    data_dir: Path,
    phase2_meta: dict,
    config: dict,
    output_dir: Path,
    block_subset: set[int] | None = None,
    refine: bool = False,
) -> None:
    """Run AdaRound for every block and save a phase2-compatible checkpoint."""
    from ..phase2.quantize import (
        _navigate_to_parent,
        patch_pipeline_for_quantized_inference,
        save_quantized_model,
    )

    bits        = phase2_meta["bits"]
    group_size  = phase2_meta["group_size"]
    qmax        = 2 ** (bits - 1) - 1
    b_inv_set   = set(calibration["b_inv_layers"])
    b_inv_store = {
        name: (1.0 / calibration["balancing_vectors"][name]).astype(np.float32)
        for name in b_inv_set
    }
    mean_rhos      = calibration.get("mean_rhos", {})
    rho_threshold  = phase2_meta.get("per_token_rho_threshold", 0.5)
    exclude        = set(phase2_meta.get("exclude_layers", ["context_embedder"]))

    layer_meta: dict[str, dict] = {}
    n_adaround = 0
    n_rtn      = 0

    from tqdm import tqdm

    n_blocks = len(pipeline.mmdit.multimodal_transformer_blocks)

    for block_idx in tqdm(range(n_blocks), desc="Blocks", unit="block"):
        block     = pipeline.mmdit.multimodal_transformer_blocks[block_idx]
        block_dir = data_dir / f"block_{block_idx:02d}"

        if block_subset is not None and block_idx not in block_subset:
            logger.info("Block %02d: skipped (not in --blocks)", block_idx)
            block_results = {}
        elif not block_dir.exists():
            logger.warning("Block %02d: data dir not found (%s) — using RTN", block_idx, block_dir)
            block_results = {}
        else:
            block_results = _optimize_block(
                block_idx, block, block_dir, b_inv_store, phase2_meta, config,
                refine=refine,
            )

        # --- Build W4A8Linear for each layer in this block and inject ---
        for entry in registry:
            name = entry["name"]
            if name in exclude:
                continue
            family  = entry.get("family", "")
            layer_bits = (
                phase2_meta.get("final_layer_bits", bits)
                if family == "final_linear" else bits
            )
            if layer_bits >= 16:
                continue
            # Only process layers belonging to this block
            parts = name.split(".")
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            if int(parts[1]) != block_idx:
                continue

            if name in block_results:
                w_int, scales, bias_np = block_results[name]
                b_inv  = b_inv_store.get(name)
                per_token = mean_rhos.get(name, 0.0) > rho_threshold
                d_out, d_in = w_int.shape

                qlinear = build_quantized_linear(w_int, scales, bias_np, layer_bits, group_size)
                b_inv_mx = mx.array(b_inv, dtype=mx.float32) if b_inv is not None else None
                w4a8 = W4A8Linear(qlinear, b_inv=b_inv_mx, per_token=per_token)

                parent, attr = _navigate_to_parent(pipeline.mmdit, name)
                setattr(parent, attr, w4a8)

                layer_meta[name] = {
                    "d_in": d_in,
                    "d_out": d_out,
                    "has_bias": bias_np is not None,
                    "bits": layer_bits,
                    "has_b_inv": b_inv is not None,
                    "per_token": per_token,
                }
                n_adaround += 1
            elif refine:
                # In refine mode, layer already has W4A8Linear from checkpoint
                parent, attr = _navigate_to_parent(pipeline.mmdit, name)
                existing = getattr(parent, attr)
                if hasattr(existing, "qlinear"):
                    ql = existing.qlinear
                    d_in = ql.weight.shape[1] * (32 // layer_bits)
                    d_out = ql.weight.shape[0]
                    layer_meta[name] = {
                        "d_in": d_in,
                        "d_out": d_out,
                        "has_bias": getattr(ql, "bias", None) is not None,
                        "bits": layer_bits,
                        "has_b_inv": hasattr(existing, "b_inv"),
                        "per_token": getattr(existing, "_per_token", False),
                    }
                n_rtn += 1
            else:
                linear = entry.get("module")
                if linear is None:
                    continue

                W_fp   = np.array(linear.weight, dtype=np.float32)
                d_out, d_in = W_fp.shape
                bias   = np.array(linear.bias) if getattr(linear, "bias", None) is not None else None
                b_inv  = b_inv_store.get(name)
                per_token = mean_rhos.get(name, 0.0) > rho_threshold

                # RTN fallback
                scales, _, scales_exp = _compute_quant_params(W_fp, layer_bits, group_size)
                w_int = np.clip(
                    np.round(W_fp / np.repeat(scales, group_size, axis=1)),
                    -qmax, qmax,
                ).astype(np.int8)

                qlinear = build_quantized_linear(w_int, scales, bias, layer_bits, group_size)
                b_inv_mx = mx.array(b_inv, dtype=mx.float32) if b_inv is not None else None
                w4a8 = W4A8Linear(qlinear, b_inv=b_inv_mx, per_token=per_token)

                parent, attr = _navigate_to_parent(pipeline.mmdit, name)
                setattr(parent, attr, w4a8)

                layer_meta[name] = {
                    "d_in": d_in,
                    "d_out": d_out,
                    "has_bias": bias is not None,
                    "bits": layer_bits,
                    "has_b_inv": b_inv is not None,
                    "per_token": per_token,
                }
                n_rtn += 1

        mx.eval(pipeline.mmdit.parameters())
        logger.info("Block %02d injected (%d AdaRound, %d RTN so far)", block_idx, n_adaround, n_rtn)

        # Checkpoint after every block
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(pipeline, output_dir, phase2_meta, layer_meta, list(b_inv_set), bits, group_size, rho_threshold, exclude)

    logger.info("Optimisation complete: %d AdaRound, %d RTN fallback", n_adaround, n_rtn)
    patch_pipeline_for_quantized_inference(pipeline)


def _save_checkpoint(
    pipeline,
    output_dir: Path,
    phase2_meta: dict,
    layer_meta: dict,
    b_inv_layers: list,
    bits: int,
    group_size: int,
    rho_threshold: float,
    exclude: set,
) -> None:
    from ..phase2.quantize import save_quantized_model
    cfg = {
        "model_version":          phase2_meta["model_version"],
        "group_size":             group_size,
        "bits":                   bits,
        "a_bits":                 phase2_meta.get("a_bits", 8),
        "final_layer_bits":       phase2_meta.get("final_layer_bits", bits),
        "alpha":                  phase2_meta["alpha"],
        "qkv_method":             phase2_meta["qkv_method"],
        "ssc_tau":                phase2_meta.get("ssc_tau", 1.0),
        "per_token_rho_threshold": rho_threshold,
        "exclude_layers":         list(exclude),
        "weight_quant":           "adaround",
    }
    save_quantized_model(pipeline.mmdit, output_dir, cfg, layer_meta, b_inv_layers)
