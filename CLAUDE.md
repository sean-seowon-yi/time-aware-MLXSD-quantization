# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX-based diffusion (SD3-Medium) pipeline with post-training quantization (PTQ) tooling. The main focus is implementing **TaQ-DiT** adaptive rounding (AdaRound) for W4A8 quantization of the SD3-Medium DiT backbone using Apple's MLX framework.

Extended pipeline adds **HTG + Bayesian Bits** (arXiv 2503.06930, 2005.07093): agglomerative timestep clustering partitions the denoising trajectory into G groups; Bayesian Bits learns per-layer W2/W4/W8 bit-widths per group via L0 hard-concrete gating; AdaRound + activation-scale optimization runs independently per group.

## Environment

Scripts must run inside the `diffusionkit` conda environment:

```bash
conda run -n diffusionkit python -m src.<module> [args]
```

DiffusionKit is installed as an editable local package. If import errors occur, reinstall with:
```bash
pip install -e DiffusionKit/python
```

## Running Tests

```bash
# All tests
conda run -n diffusionkit python -m pytest tests/ -v

# Single test file
conda run -n diffusionkit python -m pytest tests/test_adaround_optimize.py -v

# Single test
conda run -n diffusionkit python -m pytest tests/test_adaround_optimize.py::TestRectifiedSigmoid -v
```

Tests use synthetic small tensors and mock DiffusionKit — they do not load the full model.

## Full Quantization Pipeline

Two tracks run in parallel from the same calibration latents. Steps 2W/3W and 2A/3A are independent.

```bash
# 1. Generate calibration latents (~11h for 1000 images, ~12min for 10)
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 10 --num-steps 100 --calib-dir calibration_data [--resume]

# --- WEIGHT TRACK ---

# 2W. Cache block-level FP16 I/O for AdaRound (~60min for 5 images)
conda run -n diffusionkit python -m src.cache_adaround_data \
    --calib-dir calibration_data --num-images 5 --stride 4 [--force]

# 3W. Optimize AdaRound W4A8 weights
conda run -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data/adaround_cache \
    --output quantized_weights \
    [--iters 20000] [--batch-size 16] [--bits-w 4] [--bits-a 8] [--blocks mm0,mm1]

# --- ACTIVATION TRACK ---

# 2A. Collect per-layer activation statistics (~60min for 5 images)
conda run -n diffusionkit python -m src.collect_layer_activations \
    --calib-dir calibration_data --num-images 5 --stride 4 [--force]

# 3A. Analyze statistics — W4A8 baseline (faithful TaQ-DiT, <1 min)
conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --output calibration_data/activations/quant_config.json

# 3A-alt. Experimental multi-tier A4/A6/A8 config (same collected data, different output)
conda run -n diffusionkit python -m src.analyze_activations_multitier \
    --stats calibration_data/activations/layer_statistics.json \
    --output calibration_data/activations/quant_config_multitier.json \
    [--a4-threshold 6.0] [--a6-threshold 10.0] \
    [--shifted-a4-threshold 5.0] [--shifted-a6-threshold 8.0]

# Optional: visualize activation statistics
conda run -n diffusionkit python -m src.visualize_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --output-dir calibration_data/activations/plots \
    [--snapshot-steps 0 12 24 40 48] \
    [--quant-config calibration_data/activations/quant_config.json] \
    [--plot-distributions]

# --- INFERENCE ---

# 4a. V1: dequantize weights to FP16, run standard inference
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output quantized_weights \
    --prompt "a tabby cat on a table" \
    --output-image quant_test.png [--compare] [--diff-stats]

# 4b. V2: fake-quantized activations (requires quant_config.json from Step 3A)
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output quantized_weights \
    --quant-config calibration_data/activations/quant_config.json \
    --prompt "a tabby cat on a table" \
    --output-image quant_w4a8_actquant.png [--compare]
```

## HTG + Bayesian Bits Extended Pipeline

```bash
# Stage 0 — HTG Clustering
conda run -n diffusionkit python -m src.htg_cluster \
    --stats calibration_data_100/activations/layer_statistics.json \
    --output htg_groups.json [--n-groups 5]

# Stage 1 — Bayesian Bits (per group)
conda run -n diffusionkit python -m src.bayesianbits_optimize \
    --adaround-cache calibration_data_100/adaround_cache \
    --htg-groups htg_groups.json --output bb_config.json \
    [--iters 20000] [--gating-lambda 0.01] [--blocks mm0]

# Stages 2+3 — Per-group shift + outlier config
conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data_100/activations/layer_statistics.json \
    --htg-groups htg_groups.json --output quant_config_htg.json

# Stages 4+5 — Per-group AdaRound + joint reconstruction
conda run -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data_100/adaround_cache \
    --output quantized_weights_htg \
    --htg-groups htg_groups.json --bb-config bb_config.json \
    [--iters 20000] [--blocks mm0]
```

### HTG Data Flow

```
collect_layer_activations.py + cache_adaround_data.py  (existing, reused)

htg_cluster.py  (adjacent agglomerative on per-layer shift vectors)
    → htg_groups.json
      global_groups: {g: {timestep_indices, step_keys, sigma_range}}
      per_layer_z_bar: {layer: {g: [z̄_c, ...]}}

bayesianbits_optimize.py  (hierarchical quant + L0 hard-concrete gating, per group)
    → bb_config.json
      {group_g: {block.linear_path: bits (2|4|8)}}

analyze_activations.py --htg-groups  (per-group scale/shift/outlier)
    → quant_config_htg.json  (format: per_group_quant_config_htg_v1)

adaround_optimize.py --htg-groups --bb-config  (one pass per group)
    → quantized_weights_htg/group_{g}/weights/{block_name}.npz
    → quantized_weights_htg/group_{g}/config.json
    → quantized_weights_htg/config.json
```

### Key HTG Implementation Details

**adjacent_agglomerative()** (`htg_cluster.py`): O(T²) greedy merge; only adjacent cluster
pairs are eligible. Centroid distance = L2 norm between group averages of shift vectors.
Produces monotone (non-decreasing) group labels.

**Consensus partition** (`htg_cluster.py`): Each layer produces its own partition boundaries.
Global boundaries = median boundary per slot across all layers, used by Stages 1 and 4+5.
Per-layer z̄ is computed from per-layer partitions for Stages 2+3 (faithful to paper).

**BBParams** (`bayesianbits_optimize.py`): holds `log_alphas_4`, `log_alphas_8` (per layer,
shape=W.shape), `a_scales` (per layer, shape=(1,)). Nested scales: s_2=absmax, s_4=s_2/3,
s_8=s_4/9. Gate finalization: frac_4<0.5→W2; frac_4≥0.5,frac_8<0.5→W4; both≥0.5→W8.

**HTG mode in adaround_optimize.py**: `--htg-groups` activates per-group sample filtering
(by `global_groups[g]["timestep_indices"]`); `--bb-config` sets block-level effective bits
as the median of per-layer bits from bb_config[group_id].

## Architecture

### MMDiT Block Taxonomy

SD3-Medium's transformer (`pipeline.mmdit`) has two block types:
- `multimodal_transformer_blocks[i]` → named `mm{i}` — process image + text jointly
- `unified_transformer_blocks[i]` → named `uni{i}` — process unified hidden states

Each block contains two sub-blocks (`image_transformer_block`, `text_transformer_block` for MM; `transformer_block` for unified), each with `attn.{q,k,v,o}_proj` and `mlp.{fc1,fc2}`. The `adaLN_modulation` layers are intentionally **not** quantized.

### Quantization Pipeline Data Flow

```
generate_calibration_data.py
    → calibration_data/samples/{img:04d}_{step:03d}.npz   (latent x per step)
    → calibration_data/manifest.json

── WEIGHT TRACK ──────────────────────────────────────────────────────────────

cache_adaround_data.py  (installs BlockHook proxies, runs forward passes)
    → calibration_data/adaround_cache/samples/{img:04d}_{step:03d}.npz
      keys: {block_name}__arg{i}, {block_name}__kw_{name}, {block_name}__out{i}
    → calibration_data/adaround_cache/metadata.json

adaround_optimize.py    (loads block_data via load_block_data, trains AdaRoundParams)
    → quantized_weights/weights/{block_name}.npz
      keys: {safe_linear_path}__weight_int (int8), __scale (float32), __a_scale (float32)
    → quantized_weights/config.json

load_adaround_model.py  (dequantizes: W_fp16 = weight_int * scale, injects into model)
    → V1: inference with FP16 activations (standard pipeline.generate_image)
    → V2: inference with fake-quantized activations (--quant-config, custom Euler loop)

── ACTIVATION TRACK ──────────────────────────────────────────────────────────

collect_layer_activations.py  (hooks _HookedLayer proxies on every nn.Linear)
    → calibration_data/activations/layer_statistics.json   (manifest + sigma_map + step_keys)
    → calibration_data/activations/timestep_stats/step_{key}.npz
      keys per layer: avg_min, avg_max, shift, hist_counts, hist_edges  (all per-channel)
    → calibration_data/activations/timestep_stats/step_{key}_index.json
      (scalar summary per layer: tensor_absmax, hist_p999, ...)

analyze_activations.py  (W4A8 baseline — faithful TaQ-DiT)
    → calibration_data/activations/quant_config.json         (per_timestep_quant_config_v4)
      top-level: per_timestep, sigma_map, outlier_config (per-layer multiplier vectors)
      per-timestep: step → layer → {bits: 8, scale, shift[]}  (shift on post-GELU only)
      outlier_config: layer → {outlier_indices, multiplier_vector, scale_normal, scale_outlier}
    → calibration_data/activations/layer_temporal_analysis.json
      (per-layer mean/min/max scale across timesteps, shift magnitude summary)

analyze_activations_multitier.py  (experimental A4/A6/A8 — NOT faithful TaQ-DiT)
    → same output paths as above, also includes outlier_config (run separately)
```

### Key Implementation Details

**Euler sampling** (`generate_calibration_data.py`): Must use the Karras ODE formula — `d = (x - denoised) / append_dims(sigma, x.ndim)`, `x = x + d * dt`. Modulation params are cached once before the loop via `CFGDenoiser.cache_modulation_params(pooled, timesteps)`.

**Block hooking** (`cache_adaround_data.py`): `BlockHook` proxy objects replace blocks in the model's list. After each forward pass, `flush_hooks()` does a single `mx.eval(*pending)` over all hooked tensors before converting to numpy. Hooks must be removed (`remove_block_hooks`) before `clear_cache` / `load_weights` calls to avoid corrupting adaLN weights across images.

**AdaRound** (`adaround_optimize.py`): `AdaRoundParams` holds learnable `alphas` (weight rounding) and `a_scales` (activation scaling) as MLX arrays. `_QuantProxy` temporarily replaces linear layers during the forward pass. Two separate Adam optimizers are used: `w_lr=1e-3` for alphas, `a_lr=4e-5` with cosine schedule for activation scales. B-annealing decays from 20→2 after 20% warmup.

**Weight injection** (`load_adaround_model.py`): V1 dequantizes int8 → float16 and assigns to `layer.weight`. This validates rounding quality with zero memory savings. The `quant_paths` list in `config.json` is authoritative for reversing the safe-encoded NPZ key names (dots → underscores). V2 (enabled by `--quant-config`) wraps each `nn.Linear` with `_ActQuantLayer` and runs a custom Euler loop (same pattern as `generate_calibration_data.py`) that threads the current step_key into proxies before each denoising step, enabling per-(layer, timestep) fake activation quantization with shift + two-scale outlier handling.

**Activation collection** (`collect_layer_activations.py`): `_HookedLayer` proxies replace `nn.Linear` layers (same deferred-eval pattern as `BlockHook`). `ChannelStats` accumulates per-channel min/max as a **running average** (AvgMinMax), not running max — outliers do not dominate. Post-GELU shift momentum: `shift = 0.95 * shift + 0.05 * (min + max) / 2` applied only to `mlp.fc2` inputs. 256-bin histograms: edges are fixed after batch 1 using the observed range; batch 0 is retroactively re-histogrammed. adaLN reload between images uses the same pattern as `cache_adaround_data.py`.

**W4A8 activation quantization** (`analyze_activations.py`): Fixed A8 everywhere — faithful TaQ-DiT baseline. Scale is `tensor_absmax` per layer per timestep (or `hist_p999` with `--use-hist-p999`). Post-GELU layers (`mlp.fc2`) carry per-channel `shift` vectors (momentum 0.95 from collection) for centering before quantization. `identify_outlier_channels()` detects channels where `range_c > 2.5 × median(range)` and computes a per-channel `multiplier_vector` for two-scale quantization (TaQ-DiT §3.3); stored in `outlier_config` keyed by layer name. Output format `per_timestep_quant_config_v4`. The `quant_config.json` is consumed by `load_adaround_model.py --quant-config` for V2 fake-quantized inference. For experimental A4/A6/A8 dynamic switching, see `analyze_activations_multitier.py`.

**Visualization** (`visualize_activations.py`): Optional analysis tool, not in the critical path. Reads `layer_statistics.json` + per-timestep NPZ + optional `quant_config.json`. Six plot types: snapshot bar chart (all layers at one timestep), temporal line plots, heatmap (layers × timesteps), variability scatter, per-channel distribution, and histogram with shift/quantization overlays. The σ axis is inverted (1.0 → 0.0, high noise → clean image).

### adaLN Reload Bug (fixed)

After `cache_modulation_params` runs, adaLN weights in the model are overwritten. Before processing the next image, reload them with:
```python
pipeline.mmdit.load_weights(pipeline.load_mmdit(only_modulation_dict=True), strict=False)
```

### Calibration Data NPZ Keys

`generate_calibration_data.py` saves: `x`, `timestep`, `sigma`, `step_index`, `image_id`, `is_final`.

`cache_adaround_data.py` saves one file per `(image, timestep)` containing all blocks: `{block_name}__arg0`, `{block_name}__arg1`, `{block_name}__arg2`, `{block_name}__kw_positional_encodings`, `{block_name}__out0`, `{block_name}__out1` (MM blocks have two outputs; unified blocks have one).

## Critical Lesson

**Always read DiffusionKit source before implementing anything that touches the pipeline.** Guessing at the Euler formula, sigma broadcasting, or modulation caching causes subtle bugs that are hard to diagnose. See `LESSONS_READ_SOURCE_FIRST.md` for details.
