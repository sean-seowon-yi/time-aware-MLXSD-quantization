# Calibration Data Collection System — Summary

## What This System Does

Generates calibration data and collects layer activation statistics for W4A8 quantization of SD3-Medium's DiT backbone using TaQ-DiT adaptive rounding (AdaRound).

## Key Design Decisions

### 1. Two-Track Pipeline

Both tracks read from the same Step 1 calibration latents and run independently.

**Weight Track** (AdaRound):
- Cache block-level FP16 inputs/outputs per image per timestep
- Optimize per-layer rounding `alphas` and activation scales via AdaRound
- Storage: ~few GB depending on images/blocks

**Activation Track** (quantization config):
- Collect per-channel activation statistics across timesteps
- Analyze and output per-layer scale + shift vectors
- Storage: ~few hundred MB

### 2. What Is and Isn't Saved

**Calibration latents** (`generate_calibration_data.py`):
- Saved: per-step latents (`x`, `timestep`, `sigma`, `step_index`, `image_id`, `is_final`)
- Not saved: text conditioning (19 MB per step) — regenerated from prompts as needed
- Result: 99% storage reduction vs naive approach

**Activation statistics** (`collect_layer_activations.py`):
- Per-channel `avg_min`, `avg_max`, `shift` (running averages, not running max)
- 256-bin histograms with fixed edges
- Post-GELU shift vector: momentum 0.95 applied to `mlp.fc2` inputs only

### 3. Storage Optimization

Initial naive implementation: ~10 GB for 10 images (storing full conditioning)
After optimization: ~200 MB for 10 images

**Solutions**:
1. Don't store conditioning — regenerate from prompts
2. Subsample images for activation collection (5–100 instead of full set)
3. Subsample timesteps using `--stride`

### 4. DiffusionKit Issues and Fixes

**adaLN overwrite between images**:
- Problem: `cache_modulation_params()` overwrites adaLN weights in the live model
- Fix: Reload only modulation weights after each image (milliseconds, not ~10s):
  ```python
  pipeline.mmdit.load_weights(
      pipeline.load_mmdit(only_modulation_dict=True), strict=False
  )
  ```
- Important: remove hooks *before* this reload — see TROUBLESHOOTING_GUIDE.md Issue 11

**No PyTorch-style hooks in MLX**:
- Fix: Proxy objects (`BlockHook`, `_HookedLayer`) replace blocks/layers in the model list
- Deferred eval: batch all `mx.array` tensors into a single `mx.eval(*pending)` after each forward pass

**Euler sampling bugs (all fixed)**:
- Proper `append_dims` for sigma broadcasting
- Correct `to_d` derivative formula
- Modulation params cached once before the loop, not per step

## Output Files

```
calibration_data/
├── manifest.json
├── samples/
│   └── {img:04d}_{step:03d}.npz        # keys: x, timestep, sigma, step_index, image_id, is_final
├── adaround_cache/                      # weight track
│   ├── metadata.json
│   └── samples/{img:04d}_{step:03d}.npz
└── activations/                         # activation track
    ├── layer_statistics.json            # manifest + sigma_map + step_keys
    ├── quant_config.json                # W4A8 baseline (analyze_activations.py)
    ├── quant_config_multitier.json      # experimental A4/A6/A8 (analyze_activations_multitier.py)
    ├── layer_temporal_analysis.json
    └── timestep_stats/
        ├── step_{key}.npz              # per-channel avg_min/avg_max/shift/histograms
        └── step_{key}_index.json       # scalar summaries (tensor_absmax, hist_p999, ...)
```

## analyze_activations.py: W4A8 Baseline

`analyze_activations.py` produces a **fixed W4A8 config** — faithful to TaQ-DiT:
- All layers: `bits = 8`
- Scale: `tensor_absmax` per layer per timestep (or `hist_p999` with `--use-hist-p999`)
- Post-GELU layers (`mlp.fc2`): per-channel `shift` vectors passed through for centering
- Output format: `per_timestep_quant_config_v4`

`analyze_activations_multitier.py` is the experimental A4/A6/A8 dynamic-switching variant (not faithful TaQ-DiT). It reads the same collected data and outputs a separate config.

## Time and Storage Estimates (M4 Max)

| Task | Images | Time |
|------|--------|------|
| Generate calibration latents | 10 | ~6 min |
| Generate calibration latents | 1000 | ~11 hours |
| Cache AdaRound data | 5 | ~30 min |
| Collect activation stats | 5 | ~30 min |
| Analyze (either variant) | — | <1 min |

| Dataset | Storage |
|---------|---------|
| 10 calibration images | ~200 MB |
| 1000 calibration images | ~3.3 GB |
| AdaRound cache (5 images) | ~few GB |
| Activation stats (5 images) | ~few hundred MB |

## Advantages

1. **Two independent tracks** — weight and activation calibration can run in parallel
2. **Resume-able** — `--resume` flag for generation; `--force` to overwrite
3. **No DiffusionKit source modification** — proxy objects intercept without patching
4. **Re-analyzable** — run `analyze_activations.py` or the multitier variant any time over the same collected NPZ files
