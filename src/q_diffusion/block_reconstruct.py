"""AdaRound weight optimization and activation calibration per block."""

from __future__ import annotations

import gc
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import QDiffusionConfig
from .quantizer import (
    adaround_reg,
    compute_beta,
    compute_weight_scale,
)
from .quant_linear import QuantizedLinear
from .training_tracker import BlockTrainingLog


def _is_quantizable_linear(name: str, module) -> bool:
    """Check if a module is a quantizable nn.Linear (not adaLN, not Identity)."""
    if not isinstance(module, nn.Linear):
        return False
    # Skip adaLN modulation linears
    if "adaLN" in name or "modulation" in name:
        return False
    return True


def _iter_named_linears(block, stream: str = "") -> list[tuple[str, nn.Linear]]:
    """Enumerate all quantizable nn.Linear layers in a transformer block.

    For MultiModalTransformerBlock: iterates both img and txt streams.
    For FinalLayer: iterates the single linear.
    """
    results = []

    # Check if this is a MultiModalTransformerBlock
    if hasattr(block, "image_transformer_block"):
        for stream_name, tb in [("img", block.image_transformer_block),
                                 ("txt", block.text_transformer_block)]:
            attn = tb.attn
            # q, k, v projections (always present)
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                proj = getattr(attn, proj_name, None)
                if proj is not None and isinstance(proj, nn.Linear):
                    results.append((f"{stream_name}_{proj_name}", proj))

            # o_proj (skip if Identity, i.e., skip_post_sdpa)
            if not getattr(tb, "skip_post_sdpa", False):
                o_proj = getattr(attn, "o_proj", None)
                if o_proj is not None and isinstance(o_proj, nn.Linear):
                    results.append((f"{stream_name}_o_proj", o_proj))

            # FFN (skip if skip_post_sdpa)
            if not getattr(tb, "skip_post_sdpa", False) and hasattr(tb, "mlp"):
                results.append((f"{stream_name}_fc1", tb.mlp.fc1))
                results.append((f"{stream_name}_fc2", tb.mlp.fc2))

    # FinalLayer
    elif hasattr(block, "linear"):
        if isinstance(block.linear, nn.Linear):
            results.append(("linear", block.linear))

    return results


def replace_linears_in_block(
    block,
    config: QDiffusionConfig,
) -> Dict[str, QuantizedLinear]:
    """Replace all quantizable nn.Linear layers in a block with QuantizedLinear.

    Returns dict of name -> QuantizedLinear for reference.
    """
    replaced = {}

    if hasattr(block, "image_transformer_block"):
        for stream_name, tb in [("img", block.image_transformer_block),
                                 ("txt", block.text_transformer_block)]:
            attn = tb.attn
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                linear = getattr(attn, proj_name, None)
                if linear is not None and isinstance(linear, nn.Linear):
                    is_fc2 = False
                    act_sym = True  # symmetric for q/k/v
                    ql = QuantizedLinear.from_linear(
                        linear, config.weight_bits,
                        act_bits=config.activation_bits,
                        act_symmetric=act_sym,
                        per_channel=config.weight_per_channel,
                    )
                    setattr(attn, proj_name, ql)
                    name = f"{stream_name}_{proj_name}"
                    replaced[name] = ql

            if not getattr(tb, "skip_post_sdpa", False):
                o_proj = getattr(attn, "o_proj", None)
                if o_proj is not None and isinstance(o_proj, nn.Linear):
                    ql = QuantizedLinear.from_linear(
                        o_proj, config.weight_bits,
                        act_bits=config.activation_bits,
                        act_symmetric=True,
                        per_channel=config.weight_per_channel,
                    )
                    attn.o_proj = ql
                    replaced[f"{stream_name}_o_proj"] = ql

            if not getattr(tb, "skip_post_sdpa", False) and hasattr(tb, "mlp"):
                # fc1: symmetric activation
                fc1 = tb.mlp.fc1
                if isinstance(fc1, nn.Linear):
                    ql = QuantizedLinear.from_linear(
                        fc1, config.weight_bits,
                        act_bits=config.activation_bits,
                        act_symmetric=True,
                        per_channel=config.weight_per_channel,
                    )
                    tb.mlp.fc1 = ql
                    replaced[f"{stream_name}_fc1"] = ql

                # fc2: ASYMMETRIC activation (post-GELU is non-negative)
                fc2 = tb.mlp.fc2
                if isinstance(fc2, nn.Linear):
                    ql = QuantizedLinear.from_linear(
                        fc2, config.weight_bits,
                        act_bits=config.activation_bits,
                        act_symmetric=False,  # Asymmetric for post-GELU
                        per_channel=config.weight_per_channel,
                    )
                    tb.mlp.fc2 = ql
                    replaced[f"{stream_name}_fc2"] = ql

    elif hasattr(block, "linear"):
        linear = block.linear
        if isinstance(linear, nn.Linear):
            ql = QuantizedLinear.from_linear(
                linear, config.weight_bits,
                act_bits=config.activation_bits,
                act_symmetric=True,
                per_channel=config.weight_per_channel,
            )
            block.linear = ql
            replaced["linear"] = ql

    return replaced


def _get_all_v_params(quant_linears: Dict[str, QuantizedLinear]) -> list:
    """Collect all V parameters from quantized linears for the optimizer."""
    params = []
    for name, ql in quant_linears.items():
        params.append(ql.v_param)
    return params


def _group_by_timestep(block_inputs: list, fp_targets: list) -> dict:
    """Group samples by timestep value — enables one batched forward pass per group."""
    groups: dict = defaultdict(list)
    for inp, tgt in zip(block_inputs, fp_targets):
        t_val = float(inp[2][0].item())   # inp[2] is (2,) timestep, both values equal
        groups[t_val].append((inp, tgt))
    return dict(groups)


def _run_batched_group(block_model, group_pairs: list) -> mx.array:
    """Stack all samples in a timestep group and run one batched forward pass.

    Returns unnormalized sum-of-MSEs (caller divides by total sample count).
    """
    N = len(group_pairs)
    imgs = mx.concatenate([p[0][0] for p in group_pairs], axis=0)   # (2N, seq, dim)
    txts = mx.concatenate([p[0][1] for p in group_pairs], axis=0)
    t_single = group_pairs[0][0][2]   # (2,), same value for all in group
    t_val = t_single[0].item()

    img_targets = mx.concatenate([p[1][0] for p in group_pairs], axis=0)
    txt_targets_raw = [p[1][1] for p in group_pairs]

    # The modulation cache stores params with shape (2, 1, 1, D) — one row per uncond/cond.
    # With N stacked samples the input is (2N, ...), so we tile the cache to (2N, 1, 1, D)
    # before the forward pass and restore it afterwards.
    orig_img_mod = orig_txt_mod = None
    if N > 1 and hasattr(block_model, "image_transformer_block"):
        img_tb = block_model.image_transformer_block
        txt_tb = block_model.text_transformer_block
        orig_img_mod = img_tb._modulation_params[t_val]
        orig_txt_mod = txt_tb._modulation_params[t_val]
        img_tb._modulation_params[t_val] = mx.repeat(orig_img_mod, N, axis=0)
        txt_tb._modulation_params[t_val] = mx.repeat(orig_txt_mod, N, axis=0)

    out = block_model(
        latent_image_embeddings=imgs,
        token_level_text_embeddings=txts,
        timestep=t_single,
    )

    # Restore original cached params
    if orig_img_mod is not None:
        block_model.image_transformer_block._modulation_params[t_val] = orig_img_mod
        block_model.text_transformer_block._modulation_params[t_val] = orig_txt_mod

    img_out, txt_out = out[0], out[1]

    # multiply by N so caller can do: sum(group_mses) / total_N  →  per-sample mean
    group_mse = mx.mean((img_out - img_targets) ** 2) * N
    if txt_out is not None and all(t is not None for t in txt_targets_raw):
        txt_targets = mx.concatenate(txt_targets_raw, axis=0)
        group_mse = group_mse + mx.mean((txt_out - txt_targets) ** 2) * N
    return group_mse


def _compute_block_mse(
    block,
    block_inputs: list,
    fp_targets: list,
    max_samples: int = 64,
    timestep_groups: dict | None = None,
) -> float:
    """Compute block reconstruction MSE over a subset of samples."""
    is_multimodal = isinstance(block_inputs[0], tuple) and len(block_inputs[0]) >= 3

    if is_multimodal and timestep_groups is not None:
        total_mse = 0.0
        total_n = 0
        for t_val, pairs in timestep_groups.items():
            remaining = max_samples - total_n
            if remaining <= 0:
                break
            sub_pairs = pairs[:remaining]
            group_mse = _run_batched_group(block, sub_pairs)
            mx.eval(group_mse)
            total_mse += group_mse.item()
            total_n += len(sub_pairs)
        return total_mse / max(total_n, 1)

    n = min(len(block_inputs), len(fp_targets), max_samples)
    total_mse = 0.0

    for i in range(n):
        inp = block_inputs[i]
        target = fp_targets[i]

        if is_multimodal:
            # MultiModalTransformerBlock: (img_input, txt_input, timestep)
            out = block(
                latent_image_embeddings=inp[0],
                token_level_text_embeddings=inp[1],
                timestep=inp[2],
            )
            img_out, txt_out = out[0], out[1]
            img_target, txt_target = target[0], target[1]

            mse = mx.mean((img_out - img_target) ** 2)
            if txt_out is not None and txt_target is not None:
                mse = mse + mx.mean((txt_out - txt_target) ** 2)
        else:
            # FinalLayer
            out = block(*inp) if isinstance(inp, tuple) else block(inp)
            mse = mx.mean((out - target) ** 2)

        mx.eval(mse)
        total_mse += mse.item()

    return total_mse / max(n, 1)


def optimize_block_weights(
    block,
    block_idx: int,
    block_inputs: list,
    fp_targets: list,
    quant_linears: Dict[str, QuantizedLinear],
    config: QDiffusionConfig,
) -> BlockTrainingLog:
    """Run AdaRound optimization for one block.

    Optimizes V parameters to minimize block reconstruction MSE + AdaRound regularization.
    Activation quantizers are disabled during this step.

    Returns BlockTrainingLog with per-iteration loss history.
    """
    log = BlockTrainingLog(block_idx=block_idx)

    qls = list(quant_linears.values())
    n_samples = len(block_inputs)
    n_v_params = sum(v.size for v in (ql.v_param for ql in qls))
    layer_names = list(quant_linears.keys())
    print(f"  Layers ({len(qls)}): {', '.join(layer_names)}")
    print(f"  V params: {n_v_params:,}  |  samples: {n_samples}  |  batch: {config.batch_size}  |  lr={config.adaround_lr}")

    is_multimodal = isinstance(block_inputs[0], tuple) and len(block_inputs[0]) >= 3

    # Pre-group multimodal samples by timestep for batched forwards
    timestep_groups: dict | None = None
    group_keys: list | None = None
    if is_multimodal:
        timestep_groups = _group_by_timestep(block_inputs, fp_targets)
        group_keys = list(timestep_groups.keys())
        n_groups = len(group_keys)
        print(f"  Timestep groups: {n_groups}  (~{n_samples/n_groups:.1f} samples/group)")

    # Compute MSE before optimization (naive rounding baseline)
    mse_before = _compute_block_mse(block, block_inputs, fp_targets, max_samples=16, timestep_groups=timestep_groups)
    print(f"  Naive-round MSE: {mse_before:.6e}")

    # Freeze all block params; selectively unfreeze only V params for AdaRound.
    # nn.value_and_grad requires trainable_parameters() to identify what to differentiate —
    # mx.value_and_grad with a plain list of arrays causes bad_variant_access in MLX 0.17.
    block.freeze()
    for ql in qls:
        ql.unfreeze(keys=["v_param"])

    # Mutable closure state: beta/iter change each iteration, aux captured for logging
    _beta = [config.adaround_beta_start]
    _iter = [0]
    _aux = [mx.array(0.0), mx.array(0.0)]  # [recon_loss, reg_loss]

    # Pre-sample all group index choices to eliminate np.random.choice from the hot path
    n_groups_to_sample = (
        min(config.adaround_batch_groups, len(group_keys))
        if is_multimodal and group_keys else 0
    )
    _presampled = (
        [np.random.choice(len(group_keys), size=n_groups_to_sample, replace=False).tolist()
         for _ in range(config.adaround_iters)]
        if is_multimodal and group_keys else None
    )

    def loss_fn(block_model):
        """AdaRound loss: block reconstruction MSE + V-param regularization."""
        beta = _beta[0]
        recon_loss = mx.array(0.0)

        if is_multimodal and _presampled is not None:
            sampled_gis = _presampled[_iter[0]]
            total_n = 0
            for gi in sampled_gis:
                pairs = timestep_groups[group_keys[gi]]
                recon_loss = recon_loss + _run_batched_group(block_model, pairs)
                total_n += len(pairs)
            recon_loss = recon_loss / max(total_n, 1)
        else:
            batch_indices = np.random.choice(n_samples, size=min(config.batch_size, n_samples), replace=False)
            for idx in batch_indices:
                inp = block_inputs[idx]
                target = fp_targets[idx]
                out = block_model(*inp) if isinstance(inp, tuple) else block_model(inp)
                recon_loss = recon_loss + mx.mean((out - target) ** 2)
            recon_loss = recon_loss / len(batch_indices)

        # AdaRound regularization — differentiates through ql.v_param (unfrozen above)
        reg_loss = mx.array(0.0)
        for ql in qls:
            reg_loss = reg_loss + adaround_reg(ql.v_param, beta)
        reg_loss = reg_loss / len(qls)

        total_loss = recon_loss + config.adaround_reg_weight * reg_loss

        # Side-effect capture for logging (evaluated after mx.eval each iteration)
        _aux[0] = recon_loss
        _aux[1] = reg_loss

        return total_loss

    optimizer = optim.Adam(learning_rate=config.adaround_lr)
    loss_and_grad = nn.value_and_grad(block, loss_fn)

    pbar = tqdm(range(config.adaround_iters), desc=f"  AdaRound", ncols=100, leave=True)
    for iteration in pbar:
        _iter[0] = iteration
        _beta[0] = compute_beta(
            iteration, config.adaround_iters,
            config.adaround_warmup,
            config.adaround_beta_start, config.adaround_beta_end,
        )

        total_loss, grads = loss_and_grad(block)
        optimizer.update(block, grads)
        v_params = [ql.v_param for ql in qls]
        mx.eval(total_loss, _aux[0], _aux[1], *v_params)

        beta = _beta[0]
        recon_val = _aux[0].item()
        reg_val = _aux[1].item()
        total_val = total_loss.item()

        log.append(
            iteration=iteration,
            recon_loss=recon_val,
            reg_loss=reg_val,
            total_loss=total_val,
            beta=beta,
        )

        pbar.set_postfix(recon=f"{recon_val:.3e}", reg=f"{reg_val:.3e}", beta=f"{beta:.1f}")

    # Restore full trainable state (freeze was only for AdaRound gradient scoping)
    block.unfreeze()

    # Freeze rounding: soft V → hard 0/1
    print(f"  Freezing V params (threshold at 0.5)...")
    for name, ql in quant_linears.items():
        ql.freeze_rounding()

    # Compute MSE after optimization
    mse_after = _compute_block_mse(block, block_inputs, fp_targets, max_samples=16, timestep_groups=timestep_groups)
    log.finalize(mse_before, mse_after)
    direction = "▼" if mse_after < mse_before else "▲"
    print(f"  {direction} MSE: {mse_before:.6e} → {mse_after:.6e}  "
          f"({log.improvement_ratio:.2f}× improvement)")

    return log


def calibrate_act_quantizers(
    block,
    block_idx: int,
    block_inputs: list,
    fp_targets: list,
    quant_linears: Dict[str, QuantizedLinear],
    config: QDiffusionConfig,
):
    """Calibrate activation quantizer clipping ranges for a block.

    Method depends on config.act_calibration_method:
    - "percentile": α = percentile(|x|, act_percentile)
    - "mse_search": Grid search over candidates, select α minimizing block MSE
    """
    # First, collect activation statistics by running block forwards
    # We temporarily install hooks to capture each QuantizedLinear's input
    act_stats: Dict[str, list] = {name: [] for name in quant_linears}

    # Monkey-patch each QuantizedLinear to capture inputs
    orig_calls = {}
    for name, ql in quant_linears.items():
        orig_call = ql.__call__

        def make_capturing_call(ql_ref, name_ref, orig_fn):
            def capturing_call(x):
                act_stats[name_ref].append(mx.abs(x))
                return orig_fn(x)
            return capturing_call

        ql.__call__ = make_capturing_call(ql, name, orig_call)
        orig_calls[name] = orig_call

    # Run a subset of samples to collect activation stats
    n_stat_samples = min(64, len(block_inputs))
    is_multimodal = isinstance(block_inputs[0], tuple) and len(block_inputs[0]) >= 3
    n_layers_with_act = sum(1 for ql in quant_linears.values() if ql.act_quantizer is not None)
    print(f"    Collecting activation stats ({n_stat_samples} samples, {n_layers_with_act} quantizable layers)...")

    for i in range(n_stat_samples):
        inp = block_inputs[i]
        if is_multimodal:
            out = block(
                latent_image_embeddings=inp[0],
                token_level_text_embeddings=inp[1],
                timestep=inp[2],
            )
        else:
            out = block(*inp) if isinstance(inp, tuple) else block(inp)
        mx.eval(out)
        if (i + 1) % 16 == 0:
            print(f"    ... {i + 1}/{n_stat_samples} samples collected")

    # Restore original calls
    for name, ql in quant_linears.items():
        ql.__call__ = orig_calls[name]

    print(f"    Calibrating clipping ranges ({config.act_calibration_method})...")

    # Now calibrate each QuantizedLinear's activation quantizer
    for name, ql in quant_linears.items():
        if ql.act_quantizer is None:
            continue

        collected = act_stats[name]
        if not collected:
            continue

        # Stack all collected |activations|
        all_abs = mx.concatenate([a.reshape(-1) for a in collected])
        mx.eval(all_abs)

        if config.act_calibration_method == "percentile":
            p = config.act_percentile / 100.0
            alpha = mx.quantile(all_abs, mx.array(p))
            mx.eval(alpha)

            if ql.act_quantizer.symmetric:
                ql.act_quantizer.set_alpha(alpha)
            else:
                # For asymmetric (fc2): get actual min/max from non-abs values
                all_vals = mx.concatenate([
                    block_inputs[i][0].reshape(-1) if is_multimodal
                    else block_inputs[i].reshape(-1)
                    for i in range(n_stat_samples)
                ])
                alpha_min = mx.quantile(all_vals, mx.array(1.0 - p))
                alpha_max = mx.quantile(all_vals, mx.array(p))
                mx.eval(alpha_min, alpha_max)
                ql.act_quantizer.set_alpha(alpha_max, alpha_min)

        elif config.act_calibration_method == "mse_search":
            # Grid search: test candidate alphas, select best MSE
            candidates = []
            for pct in config.act_search_candidates:
                p = pct / 100.0
                if p >= 1.0:
                    alpha_c = mx.max(all_abs)
                else:
                    alpha_c = mx.quantile(all_abs, mx.array(p))
                mx.eval(alpha_c)
                candidates.append((pct, alpha_c))

            best_alpha = None
            best_mse = float("inf")

            for pct, alpha_c in candidates:
                # Temporarily set this alpha
                if ql.act_quantizer.symmetric:
                    ql.act_quantizer.set_alpha(alpha_c)
                else:
                    # For asymmetric: scale alpha_min proportionally
                    ratio = alpha_c / mx.max(all_abs)
                    all_vals = mx.concatenate([
                        block_inputs[i][0].reshape(-1) if is_multimodal
                        else block_inputs[i].reshape(-1)
                        for i in range(min(8, n_stat_samples))
                    ])
                    alpha_min = mx.min(all_vals) * ratio
                    mx.eval(alpha_min)
                    ql.act_quantizer.set_alpha(alpha_c, alpha_min)

                # Compute block MSE with this alpha
                mse = _compute_block_mse(
                    block, block_inputs[:16], fp_targets[:16], max_samples=16
                )

                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha_c

            # Set the best alpha
            if ql.act_quantizer.symmetric:
                ql.act_quantizer.set_alpha(best_alpha)
            else:
                ratio = best_alpha / mx.max(all_abs)
                all_vals = mx.concatenate([
                    block_inputs[i][0].reshape(-1) if is_multimodal
                    else block_inputs[i].reshape(-1)
                    for i in range(min(8, n_stat_samples))
                ])
                alpha_min = mx.min(all_vals) * ratio
                mx.eval(alpha_min)
                ql.act_quantizer.set_alpha(best_alpha, alpha_min)

        print(f"    {name}: α={ql.act_quantizer.alpha.item():.4f}, "
              f"scale={ql.act_quantizer.scale.item():.6e}")

    # Clean up collected stats
    del act_stats
    gc.collect()
