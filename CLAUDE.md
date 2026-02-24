# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX-based diffusion (SD3-Medium) pipeline with post-training quantization (PTQ) tooling. The main focus is implementing **TaQ-DiT** adaptive rounding (AdaRound) for W4A8 quantization of the SD3-Medium DiT backbone using Apple's MLX framework.

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

Run these four scripts in order:

```bash
# 1. Generate calibration latents (~11h for 1000 images, ~6min for 10)
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 10 --num-steps 50 --calib-dir calibration_data [--resume]

# 2. Cache block-level FP16 I/O for AdaRound (~30min for 100 images)
conda run -n diffusionkit python -m src.cache_adaround_data \
    --calib-dir calibration_data --num-images 5 --stride 5 [--force]

# 3. Optimize AdaRound W4A8 weights
conda run -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data/adaround_cache \
    --output quantized_weights \
    [--iters 20000] [--batch-size 16] [--bits-w 4] [--bits-a 8] [--blocks mm0,mm1]

# 4. Load quantized model and run inference
conda run -n diffusionkit python -m src.load_adaround_model \
    --adaround-output quantized_weights \
    --prompt "a tabby cat on a table" \
    --output-image quant_test.png [--compare] [--diff-stats]
```

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

cache_adaround_data.py  (installs BlockHook proxies, runs forward passes)
    → calibration_data/adaround_cache/samples/{img:04d}_{step:03d}.npz
      keys: {block_name}__arg{i}, {block_name}__kw_{name}, {block_name}__out{i}
    → calibration_data/adaround_cache/metadata.json

adaround_optimize.py    (loads block_data via load_block_data, trains AdaRoundParams)
    → quantized_weights/weights/{block_name}.npz
      keys: {safe_linear_path}__weight_int (int8), __scale (float32), __a_scale (float32)
    → quantized_weights/config.json

load_adaround_model.py  (dequantizes: W_fp16 = weight_int * scale, injects into model)
    → inference with quantized weights
```

### Key Implementation Details

**Euler sampling** (`generate_calibration_data.py`): Must use the Karras ODE formula — `d = (x - denoised) / append_dims(sigma, x.ndim)`, `x = x + d * dt`. Modulation params are cached once before the loop via `CFGDenoiser.cache_modulation_params(pooled, timesteps)`.

**Block hooking** (`cache_adaround_data.py`): `BlockHook` proxy objects replace blocks in the model's list. After each forward pass, `flush_hooks()` does a single `mx.eval(*pending)` over all hooked tensors before converting to numpy. Hooks must be removed (`remove_block_hooks`) before `clear_cache` / `load_weights` calls to avoid corrupting adaLN weights across images.

**AdaRound** (`adaround_optimize.py`): `AdaRoundParams` holds learnable `alphas` (weight rounding) and `a_scales` (activation scaling) as MLX arrays. `_QuantProxy` temporarily replaces linear layers during the forward pass. Two separate Adam optimizers are used: `w_lr=1e-3` for alphas, `a_lr=4e-5` with cosine schedule for activation scales. B-annealing decays from 20→2 after 20% warmup.

**Weight injection** (`load_adaround_model.py`): V1 dequantizes int8 → float16 and assigns to `layer.weight`. This validates rounding quality with zero memory savings. The `quant_paths` list in `config.json` is authoritative for reversing the safe-encoded NPZ key names (dots → underscores).

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
