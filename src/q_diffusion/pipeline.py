"""End-to-end Q-Diffusion orchestration."""

from __future__ import annotations

import gc
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np

from .config import QDiffusionConfig
from .calibration_feeder import (
    BlockIOCollector, BlockInputCache, FPTargetCache, load_calibration_data,
)
from .block_reconstruct import (
    replace_linears_in_block,
    optimize_block_weights,
    calibrate_act_quantizers,
)
from .training_tracker import TrainingTracker
from .quant_model_io import save_quantized_model


_ROOT = Path(__file__).resolve().parents[2]


def _fmt(seconds: float) -> str:
    """Format elapsed seconds as mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _ensure_diffusionkit() -> None:
    try:
        import diffusionkit.mlx  # noqa: F401
        return
    except ImportError:
        pass
    dk_src = _ROOT / "DiffusionKit" / "python" / "src"
    if dk_src.is_dir() and str(dk_src) not in sys.path:
        sys.path.insert(0, str(dk_src))
    import diffusionkit.mlx  # noqa: F401


def _prime_modulation_cache(pipeline, mmdit, cali_data):
    """Ensure _modulation_params is populated on all TransformerBlocks.

    In a normal run this is a side-effect of _run_forward_collecting (stage 4.5).
    When resuming and skipping that stage, we must prime it explicitly so that
    block.pre_sdpa can look up pre-computed adaLN values during AdaRound.

    Uses the first prompt group's pooled embeddings and all unique timesteps.
    Reloads adaLN weights immediately after (cache_modulation_params offloads them).
    """
    first_block = mmdit.multimodal_transformer_blocks[0]
    if hasattr(first_block.image_transformer_block, "_modulation_params"):
        return  # Already populated by a previous stage

    ts_vals = cali_data["ts"]
    cs_pooled = cali_data["cs_pooled"]
    prompt_indices = cali_data["prompt_indices"]

    first_prompt_idx = int(prompt_indices[0])
    pooled_mx = mx.array(cs_pooled[first_prompt_idx]).astype(pipeline.activation_dtype)
    unique_ts = sorted(set(float(t) for t in ts_vals))
    ts_mx = mx.array(unique_ts, dtype=pipeline.activation_dtype)

    print(f"  Priming modulation cache ({len(unique_ts)} timesteps)...")
    mmdit.cache_modulation_params(pooled_text_embeddings=pooled_mx, timesteps=ts_mx)
    # Restore adaLN weights that cache_modulation_params offloaded
    mmdit.load_weights(pipeline.load_mmdit(only_modulation_dict=True), strict=False)
    print(f"  Done — adaLN weights restored.")


def _load_pipeline(model_version: str, low_memory_mode: bool = True):
    _ensure_diffusionkit()
    import logging
    for _name in list(logging.Logger.manager.loggerDict):
        if "diffusionkit" in _name or "argmaxtools" in _name:
            logging.getLogger(_name).setLevel(logging.WARNING)
    from diffusionkit.mlx import DiffusionPipeline
    return DiffusionPipeline(
        w16=True, shift=3.0, use_t5=True,
        model_version=model_version,
        low_memory_mode=low_memory_mode,
        a16=True, local_ckpt=None,
    )


def run_q_diffusion(config: QDiffusionConfig):
    """Full Q-Diffusion quantization pipeline.

    1. Load FP model
    2. Pre-compute: stream all FP block outputs to disk (single forward pass)
    3. Step 0: In-place naive weight quantization
    4. Step 1: Block-wise AdaRound weight refinement (loads targets per block from disk)
    5. Step 2: Activation quantizer calibration
    6. Save quantized model + training logs
    """
    from src.calibration_sample_generation.calibration_config import MODEL_VERSION

    pipeline_start = time.time()
    step_times: Dict[str, float] = {}

    print("=" * 70)
    print(f"  Q-Diffusion: W{config.weight_bits}A{config.activation_bits}")
    print(f"  Model:       {MODEL_VERSION}")
    print(f"  AdaRound:    {config.adaround_iters} iters/block  |  batch_size={config.batch_size}  |  lr={config.adaround_lr}")
    print(f"  Beta:        {config.adaround_beta_start} → {config.adaround_beta_end}  (warmup={config.adaround_warmup:.0%})")
    print(f"  Act calib:   {config.act_calibration_method}  |  n_samples={config.n_samples}")
    print(f"  Output:      {config.output_dir}")
    print(f"  Resume:      {'yes — will reuse existing caches if valid' if config.resume else 'no — fresh run'}")
    print("=" * 70)

    # If not resuming, wipe any stale caches so they do not get accidentally reused.
    if not config.resume:
        for _stale in (".fp_target_cache", ".naive_input_cache", ".adaround_checkpoints"):
            _stale_path = Path(config.output_dir) / _stale
            if _stale_path.exists():
                print(f"  [restart] Removing stale cache: {_stale_path}")
                shutil.rmtree(_stale_path)

    # 1. Load FP model
    print("\n[1/6] Loading FP model...")
    t0 = time.time()
    pipeline = _load_pipeline(MODEL_VERSION)
    mmdit = pipeline.mmdit
    n_mm_blocks = len(mmdit.multimodal_transformer_blocks)
    step_times["1_load_model"] = time.time() - t0
    print(f"  Loaded MMDiT: {n_mm_blocks} multimodal blocks + FinalLayer  [{_fmt(step_times['1_load_model'])}]")

    # 2. Load calibration data
    print("\n[2/6] Loading calibration data...")
    t0 = time.time()
    cali_data = load_calibration_data(config.calibration_file)
    n_total = len(cali_data["xs"])
    ts_vals = cali_data["ts"]
    n_prompts = len(set(cali_data["prompt_indices"].tolist()))
    step_times["2_load_cali"] = time.time() - t0
    print(f"  {n_total} tuples  |  {n_prompts} prompts  |  "
          f"timestep range [{float(ts_vals.min()):.3f}, {float(ts_vals.max()):.3f}]  "
          f"[{_fmt(step_times['2_load_cali'])}]")

    # 3. Pre-compute: Stream FP block outputs to disk
    fp_cache_dir = Path(config.output_dir) / ".fp_target_cache"
    collector = BlockIOCollector(pipeline, cali_data, n_samples=config.n_samples)

    _fp_probe = FPTargetCache.from_existing_dir(fp_cache_dir, n_mm_blocks)
    if config.resume and _fp_probe.is_complete(len(collector.sample_indices)):
        print(f"\n[3/6] Reusing existing FP target cache ({_fp_probe.disk_usage_mb():.0f} MB)  [skipped]")
        fp_cache = _fp_probe
        step_times["3_fp_cache"] = 0.0
    else:
        print("\n[3/6] Caching FP block outputs to disk (single forward pass)...")
        t0 = time.time()
        fp_cache = collector.collect_all_fp_targets(cache_dir=str(fp_cache_dir))
        step_times["3_fp_cache"] = time.time() - t0
        print(f"  FP target cache complete  [{_fmt(step_times['3_fp_cache'])}]")

    n_blocks = len(mmdit.multimodal_transformer_blocks)
    total_blocks = n_blocks + (0 if config.skip_final_layer else 1)

    # 4. Step 0: In-place naive weight quantization
    print(f"\n[4/6] Step 0: In-place W{config.weight_bits} quantization ({total_blocks} blocks)...")
    t0 = time.time()
    all_quant_linears: Dict[int, dict] = {}

    for block_idx in range(n_blocks):
        block = mmdit.multimodal_transformer_blocks[block_idx]
        ql_dict = replace_linears_in_block(block, config)
        all_quant_linears[block_idx] = ql_dict
        print(f"  block {block_idx:02d}: {len(ql_dict):2d} layers  [{', '.join(ql_dict.keys())}]")

    if not config.skip_final_layer and hasattr(mmdit, "final_layer"):
        block_idx = n_blocks
        ql_dict = replace_linears_in_block(mmdit.final_layer, config)
        all_quant_linears[block_idx] = ql_dict
        print(f"  FinalLayer:  {len(ql_dict):2d} layers  [{', '.join(ql_dict.keys())}]")

    total_layers = sum(len(d) for d in all_quant_linears.values())
    step_times["4_naive_quant"] = time.time() - t0
    print(f"  Done: {total_layers} QuantizedLinear layers created (adaLN kept FP16)  [{_fmt(step_times['4_naive_quant'])}]")

    # 4.5. Collect all block inputs on naive-quantized model (single sweep)
    naive_cache_dir = Path(config.output_dir) / ".naive_input_cache"
    _naive_probe = BlockInputCache.from_existing_dir(naive_cache_dir, n_blocks)
    if config.resume and _naive_probe.is_complete(len(collector.sample_indices)):
        print(f"\n[4.5/6] Reusing existing naive-input cache ({_naive_probe.disk_usage_mb():.0f} MB)  [skipped]")
        naive_input_cache = _naive_probe
        step_times["4b_naive_inputs"] = 0.0
    else:
        print(f"\n[4.5/6] Collecting naive-quantized block inputs (single sweep)...")
        t0 = time.time()
        naive_input_cache = collector.collect_all_block_inputs(cache_dir=str(naive_cache_dir))
        step_times["4b_naive_inputs"] = time.time() - t0
        print(f"  Done  [{_fmt(step_times['4b_naive_inputs'])}]")

    # Ensure _modulation_params is cached before any block is called directly.
    # Normally a side-effect of stage 4.5; must be explicit when resuming.
    _prime_modulation_cache(pipeline, mmdit, cali_data)

    # 5. Step 1: Block-wise AdaRound weight refinement
    total_iters = total_blocks * config.adaround_iters
    print(f"\n[5/6] Step 1: AdaRound weight refinement")
    print(f"  {total_blocks} blocks × {config.adaround_iters} iters = {total_iters:,} total iterations")
    print(f"  β anneals {config.adaround_beta_start}→{config.adaround_beta_end} over first "
          f"{int(config.adaround_warmup * config.adaround_iters)} iters per block")
    tracker = TrainingTracker()

    # Per-block checkpointing
    ckpt_dir = Path(config.output_dir) / ".adaround_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt_complete(bidx: int) -> bool:
        return (ckpt_dir / f"block_{bidx:02d}" / "done").exists()

    def _save_ckpt(bidx: int, qlinears: dict):
        d = ckpt_dir / f"block_{bidx:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for name, ql in qlinears.items():
            np.save(d / f"{name}_v.npy", np.array(ql.v_param, copy=False))
        (d / "done").touch()

    def _load_ckpt(bidx: int, qlinears: dict):
        d = ckpt_dir / f"block_{bidx:02d}"
        for name, ql in qlinears.items():
            ql.v_param = mx.array(np.load(d / f"{name}_v.npy"))
            ql.freeze_rounding()

    adaround_start = time.time()
    block_times_adaround = []

    for block_idx in range(total_blocks):
        label = f"block {block_idx:02d}/{total_blocks - 1}"
        if block_idx == n_blocks:
            label = f"FinalLayer ({label})"
        print(f"\n{'─' * 60}")
        print(f"  AdaRound [{label}]")
        print(f"{'─' * 60}")

        if block_idx < n_blocks:
            block = mmdit.multimodal_transformer_blocks[block_idx]
        else:
            block = mmdit.final_layer

        quant_linears = all_quant_linears[block_idx]

        # Resume: restore from checkpoint if this block was already completed
        if config.resume and _ckpt_complete(block_idx):
            _load_ckpt(block_idx, quant_linears)
            print(f"  Restored from checkpoint  [skipped]")
            continue

        t0 = time.time()

        block_inputs = naive_input_cache.load_block(block_idx, n_blocks)
        print(f"  {len(block_inputs)} input samples loaded (from cache)")

        block_fp_targets = fp_cache.load_block(block_idx)
        print(f"  {len(block_fp_targets)} FP targets loaded from disk")

        log = optimize_block_weights(
            block=block,
            block_idx=block_idx,
            block_inputs=block_inputs,
            fp_targets=block_fp_targets,
            quant_linears=quant_linears,
            config=config,
        )
        tracker.add_block(log)
        tracker.print_block_summary(log)

        # Save checkpoint immediately after successful optimization
        _save_ckpt(block_idx, quant_linears)

        block_elapsed = time.time() - t0
        block_times_adaround.append(block_elapsed)
        blocks_done = block_idx + 1
        blocks_left = total_blocks - blocks_done
        avg_block = sum(block_times_adaround) / len(block_times_adaround)
        eta = avg_block * blocks_left
        print(f"  [{blocks_done}/{total_blocks} blocks  |  block time: {_fmt(block_elapsed)}  |  "
              f"ETA: {_fmt(eta)}]")

        del block_inputs, block_fp_targets
        gc.collect()
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

    step_times["5_adaround"] = time.time() - adaround_start
    print(f"\n  AdaRound complete  [{_fmt(step_times['5_adaround'])}]")

    # Cleanup naive input cache
    naive_input_cache.cleanup()
    del naive_input_cache
    gc.collect()

    # 5.5. Collect all block inputs on AdaRound-refined model (single sweep)
    print(f"\n[5.5/6] Collecting AdaRound-refined block inputs (single sweep)...")
    t0 = time.time()
    refined_input_cache = collector.collect_all_block_inputs(
        cache_dir=str(Path(config.output_dir) / ".refined_input_cache")
    )
    step_times["5b_refined_inputs"] = time.time() - t0
    print(f"  Done  [{_fmt(step_times['5b_refined_inputs'])}]")

    # 6. Step 2: Activation quantizer calibration
    print(f"\n[6/6] Step 2: Activation calibration ({config.act_calibration_method})")
    print(f"  Candidates: {config.act_search_candidates}" if config.act_calibration_method == "mse_search"
          else f"  Percentile: {config.act_percentile}%")

    act_calib_start = time.time()
    block_times_calib = []

    for block_idx in range(total_blocks):
        label = f"block {block_idx:02d}/{total_blocks - 1}"
        if block_idx == n_blocks:
            label = f"FinalLayer ({label})"
        print(f"\n  Calibrating [{label}]  ({len(all_quant_linears[block_idx])} layers)...")

        if block_idx < n_blocks:
            block = mmdit.multimodal_transformer_blocks[block_idx]
        else:
            block = mmdit.final_layer

        t0 = time.time()

        block_inputs = refined_input_cache.load_block(block_idx, n_blocks)
        block_fp_targets = fp_cache.load_block(block_idx)
        quant_linears = all_quant_linears[block_idx]

        calibrate_act_quantizers(
            block=block,
            block_idx=block_idx,
            block_inputs=block_inputs,
            fp_targets=block_fp_targets,
            quant_linears=quant_linears,
            config=config,
        )

        block_elapsed = time.time() - t0
        block_times_calib.append(block_elapsed)
        blocks_done = block_idx + 1
        blocks_left = total_blocks - blocks_done
        avg_block = sum(block_times_calib) / len(block_times_calib)
        eta = avg_block * blocks_left
        print(f"  [block done: {_fmt(block_elapsed)}  |  ETA: {_fmt(eta)}]")

        del block_inputs, block_fp_targets
        gc.collect()

    step_times["6_act_calib"] = time.time() - act_calib_start
    print(f"\n  Activation calibration complete  [{_fmt(step_times['6_act_calib'])}]")

    # Save
    print(f"\n{'=' * 70}")
    print("Saving quantized model...")
    t0 = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_quantized_model(mmdit, str(output_dir), config)
    tracker.save_json(str(output_dir / "training_log.json"))
    tracker.plot_loss_curves(str(output_dir / "loss_curves"))
    step_times["7_save"] = time.time() - t0
    print(f"  Saved  [{_fmt(step_times['7_save'])}]")

    tracker.print_overall_summary()

    # Cleanup
    refined_input_cache.cleanup()
    del refined_input_cache
    fp_cache.cleanup()
    del fp_cache, collector
    gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception:
        pass

    total_elapsed = time.time() - pipeline_start

    # Final timing summary
    print(f"\n{'=' * 70}")
    print("  Timing Summary")
    print(f"{'─' * 70}")
    labels = {
        "1_load_model":    "[1/6]   Load FP model",
        "2_load_cali":     "[2/6]   Load calibration data",
        "3_fp_cache":      "[3/6]   Cache FP targets",
        "4_naive_quant":   "[4/6]   Naive quantization",
        "4b_naive_inputs": "[4.5/6] Collect naive inputs",
        "5_adaround":      "[5/6]   AdaRound",
        "5b_refined_inputs": "[5.5/6] Collect refined inputs",
        "6_act_calib":     "[6/6]   Activation calibration",
        "7_save":          "        Save outputs",
    }
    for key, label in labels.items():
        t = step_times.get(key, 0)
        pct = 100 * t / total_elapsed if total_elapsed > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {label:<32}  {_fmt(t):>10}  ({pct:4.1f}%)  {bar}")
    print(f"{'─' * 70}")
    print(f"  {'Total':>32}  {_fmt(total_elapsed):>10}")
    print(f"{'=' * 70}")
    print(f"\nQ-Diffusion complete! Output: {output_dir}")
