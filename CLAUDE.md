# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Hardware**: Mac M5 (Apple Silicon); MLX is the compute backend — no CUDA.
- **Conda env**: `mlxsd` (Python 3.11, MLX 0.17.3, DiffusionKit 0.5.2)
  ```bash
  conda activate mlxsd
  ```
- **DiffusionKit**: checked out as submodule at `DiffusionKit/` and installed in editable mode:
  ```bash
  pip install -e DiffusionKit/   # from repo root, NOT DiffusionKit/python/src/
  ```
- **All commands run from repo root** using `python -m src.<module>` (never `cd src/`).
- **Model**: `argmaxinc/mlx-stable-diffusion-3-medium` (SD3 Medium, hidden_size=1536).

## Common Commands

### Phase 1 — Calibration data generation
```bash
# Full run (256 trajectories × 100 steps → 6,400 calibration points)
python -m src.calibration_sample_generation.sample_cali_data --output DiT_cali_data.npz

# Quick dry run for functional testing
python -m src.calibration_sample_generation.sample_cali_data \
    --num-fid-samples 8 --num-sampling-steps 20 --num-selected-steps 5 \
    --output dry_run_cali.npz
```

### Phase 2 — Activation diagnostics
```bash
python -m src.activation_diagnostics.profile_postgelu \
    --calibration-file DiT_cali_data.npz --num-samples 512 \
    --output activation_stats_postgelu.npz

python -m src.activation_diagnostics.visualize_postgelu \
    --stats-file activation_stats_postgelu.npz --output-dir activation_plots/
```

### Phase 3 — HTG quantization calibration
```bash
# Full pipeline (profiling → param computation → reparameterization)
python -m src.htg_quantization.apply_htg \
    --calibration-file DiT_cali_data.npz --output-dir htg_output/

# Skip expensive stages if intermediate files exist
python -m src.htg_quantization.apply_htg --calibration-file DiT_cali_data.npz \
    --output-dir htg_output/ --skip-profile --skip-compute
```

### Phase 4 — Inference
```bash
# FP16 baseline (no quantization)
python -m src.inference.run_inference --prompt "..." --output baseline.png

# Full HTG corrections (no integer quantization)
python -m src.inference.run_inference \
    --htg-corrections htg_output/htg_corrections.npz \
    --prompt "..." --output htg.png

# W8A8 (paper-aligned)
python -m src.inference.run_inference \
    --htg-corrections htg_output/htg_corrections.npz \
    --htg-quantize-weights --htg-bits 8 \
    --htg-quantize-activations --htg-activation-ranges htg_output/htg_activation_ranges.npz \
    --prompt "..." --output htg_w8a8.png

# Ablation: weight rescaling only
python -m src.inference.run_inference \
    --htg-corrections htg_output/htg_corrections.npz \
    --no-htg-qkv --no-htg-fc1 --no-htg-oproj \
    --prompt "..." --output weight_only.png
```

## Architecture Overview

This repo implements the HTG (Hierarchical Timestep Grouping) PTQ method from arXiv:2503.06930 on SD3 Medium via DiffusionKit (MLX). It is structured as four sequential phases:

### Data flow across phases
```
Phase 1: sample_cali_data.py  →  DiT_cali_data.npz
Phase 2: profile_postgelu.py  →  activation_stats_postgelu.npz  →  visualize_postgelu.py (plots)
Phase 3: apply_htg.py         →  htg_output/{htg_corrections.npz, htg_params.npz, htg_activation_ranges.npz}
Phase 4: run_inference.py     ←  htg_output/htg_corrections.npz
```

### Inference pipeline pattern (`src/inference/`)

The inference side uses a **Strategy + Chain-of-Responsibility** pattern:

- `base.py` defines `InferenceTransform` (ABC with 4 hook methods) and `QuantizedInferencePipeline` (composes a list of transforms).
- `htg_transform.py` implements `HTGTransform(InferenceTransform)`.
- `run_inference.py` is the CLI that builds `QuantizedInferencePipeline(pipeline, transforms=[...])`.

The four hooks, called in this order per inference run:
1. `apply_weight_modifications(mmdit)` — once at load; rescales weights (`Ŵ = W * s`), optionally quantizes to `nn.QuantizedLinear`, optionally wraps with `FakeQuantizedLinear` for activation simulation.
2. `wrap_cache_modulation_params(mmdit, fn)` — wraps `MMDiT.cache_modulation_params`; applies in-place adaLN corrections to the cached modulation tensors after each call.
3. `wrap_pre_sdpa(fn)` — class-level patch on `TransformerBlock.pre_sdpa`; injects `_htg_t_key` into the intermediates dict and updates the activation quantization context.
4. `wrap_post_sdpa(fn)` — class-level patch on `TransformerBlock.post_sdpa`; applies the oproj shift `(sdpa_output - z_g) / s`.

To add a new PTQ strategy, create `src/inference/<name>_transform.py`, subclass `InferenceTransform`, override only the hooks you need, then pass it to `QuantizedInferencePipeline(..., transforms=[..., NewTransform(...)])`.

### HTG corrections format (`htg_corrections.npz`)

Keys per layer: `{layer_id}::z_g` (G×D), `{layer_id}::s` (D,), `{layer_id}::group_assignments` (T,).
Top-level: `timesteps_sorted` (T,), `num_groups`, `ema_alpha`.

Layer ID scheme: `mm_{idx:02d}_img_{qkv|fc1|oproj}`, `mm_{idx:02d}_txt_{...}`, `uni_{idx:02d}_{...}`.

> **Critical**: `htg_mmdit_weights.npz` keys use Python `id(linear)` (session-specific memory addresses) — do not use them across sessions. Always re-derive weight rescaling from `s` vectors in `htg_corrections.npz` at runtime (this is what `HTGTransform.apply_weight_modifications` does).

### SD3 MMDiT specifics

- `adaLN_modulation` outputs a 6-chunk packed tensor: `[β₁, γ₁, α₁, β₂, γ₂, α₂]` × hidden_size.
- DiffusionKit uses the `(1 + γ)` convention for scale, so the corrected scale is `γ̂ = (1 + γ) / s - 1`.
- `txt` blocks with `skip_post_sdpa=True` have `num_modulation_params=2` (not 6) — the fc1 correction must guard on `num_mod == 6`.
- `parallel_mlp=True` blocks share qkv/fc1 modulation — also skip fc1 correction.
- After `cache_modulation_params`, adaLN linear weights are offloaded. To reload: `load_weights(only_modulation_dict=True)`.
