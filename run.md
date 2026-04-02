# CLI Reference

## `src.generate_poly_schedule`

Fits polynomial clipping schedules from collected activation statistics.

```bash
python -m src.generate_poly_schedule [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--activations-dir` | Path | `calibration_data_100/activations` | Directory containing activation stats from `collect_activation_stats` |
| `--output` | Path | `polynomial_clipping_schedule.json` | Output JSON path |
| `--percentile` | Choice | `p100_absmax` | Statistic to fit. Choices: `p99`, `p999`, `mean_absmax`, `p100_absmax` |
| `--include-shifts` | Flag | off | Fit shift (center) trajectories for asymmetric activation quantization |

---

## `src.gptq.optimize`

Two-phase GPTQ pipeline: Hessian collection (Phase A) then quantization + alpha search (Phase B).

```bash
python -m src.gptq.optimize --prompts <FILE> --poly-schedule <FILE> [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompts` | Path | **required** | Tab-separated file: `seed<TAB>prompt` per line |
| `--poly-schedule` | Path | **required** | Path to `polynomial_clipping_schedule.json` |
| `--bits-w` | int | `4` | Weight quantization bits. Choices: `4`, `8` |
| `--output-dir` | Path | `gptq_output` | Directory for quantized weights and config |
| `--num-steps` | int | `30` | Number of denoising steps |
| `--cfg-weight` | float | `4.0` | Classifier-free guidance weight |
| `--latent-size` | int | `64` | Spatial size of latent (64 = 512px) |
| `--damp-percent` | float | `0.01` | Hessian diagonal damping factor |
| `--block-size` | int | `128` | Column block size for GPTQ internal error compensation (compute knob, does not affect saved weights) |
| `--group-size` | int | `128` | Number of weights sharing one scale factor. Use `0` for per-channel (no grouping) |
| `--max-prompts` | int | all | Limit number of prompts for Phase A (Hessian collection) |
| `--alpha-prompts` | int | `5` | Number of prompts for Phase B alpha search |
| `--subsample-rows` | int | `128` | Max activation rows sampled per forward call in Phase B. Lower = faster but noisier MSE |
| `--hessian-cache` | Path | `<output-dir>/hessians.npz` | Path to save/load Hessian checkpoint. If file exists, Phase A is skipped |
| `--static-act-quant` | Flag | off | Use static (timestep-agnostic) activation clipping instead of polynomial during alpha search |
| `--skip-gptq` | Flag | off | Skip Phase A and B.1. Loads existing weights from `output-dir/weights/` and reruns alpha search only |
| `--weight-mode` | Choice | `gptq` | Weight quantization method. Choices: `gptq` (full GPTQ), `rtn` (round-to-nearest per-channel), `fp16` (no weight quant, activation-only alpha search) |
| `--raw-hessian` | Flag | off | Use full-precision activations (not fake-quantized) when accumulating Hessians in Phase A. Avoids conditioning weight quantization on activation parameters not yet finalized |

---

## `src.gptq.inference`

Generate images using GPTQ-quantized weights, optionally compared to FP16 baseline.

```bash
python -m src.gptq.inference --prompt "..." [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt` | str | **required** | Text prompt for image generation |
| `--gptq-dir` | Path | `gptq_output` | Directory containing GPTQ weights and config |
| `--poly-schedule` | Path | `polynomial_clipping_schedule.json` | Path to polynomial clipping schedule |
| `--output-dir` | Path | `gptq_comparison` | Output directory; images saved under `<output-dir>/<prompt-slug>/` |
| `--num-steps` | int | `30` | Number of denoising steps |
| `--cfg-weight` | float | `4.0` | Classifier-free guidance weight |
| `--seed` | int | `42` | Random seed (uses NumPy RNG, matching `pipeline.generate_image()`) |
| `--latent-size` | int | `64` | Spatial size of latent (64 = 512px) |
| `--no-act-quant` | Flag | off | Disable activation quantization — W4A16 weight-only mode |
| `--no-alpha-scale` | Flag | off | Use poly clipping with `alpha_scale=1.0` for all layers (ignores learned scales from config) |
| `--poly-only` | Flag | off | W16A8 mode: poly activation clipping only, no weight quantization |
| `--skip-blocks` | int list | none | Block indices to leave at FP16 (e.g. `--skip-blocks 22 23`) |
| `--no-baseline` | Flag | off | Skip FP16 baseline generation; only produce quantized image |

---

## `src.benchmark_model`

End-to-end benchmark: image generation + latency/memory profiling + FID/IS/KID/LPIPS/PSNR metrics.

```bash
python -m src.benchmark_model --config <CONFIG> [OPTIONS]
```

### Config values

| `--config` | Description |
|------------|-------------|
| `fp16` | FP16 baseline (no quantization) |
| `naive_int8` | Naive per-tensor INT8 activation quantization |
| `gptq` | GPTQ W4 weights + poly/static/dynamic INT8 activations (requires `--gptq-dir`) |
| `gptq_only` | GPTQ W4 weights + static calibrated INT8 activations, standard pipeline path (requires `--gptq-dir` and `--poly-schedule`) |

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | str | `fp16` | Quantization config label (see table above) |
| `--gptq-dir` | Path | none | GPTQ output dir (from `src.gptq.optimize`). Required for `gptq` and `gptq_only` configs. Mutually exclusive with `--adaround-output` |
| `--dynamic-act-quant` | Flag | off | Per-layer dynamic INT8 activation quantization after GPTQ weight patching (scale = max(\|x\|)/127). Mutually exclusive with `--poly-schedule` |
| `--adaround-output` | Path | none | AdaRound weights dir |
| `--adaround-act-config` | Path | none | Activation quant config JSON |
| `--poly-schedule` | Path | none | Polynomial clipping schedule JSON (from `generate_poly_schedule.py`) |
| `--lut-schedule` | Path | none | LUT clipping schedule JSON |
| `--poly-margin` | float | `1.0` | Multiplier applied to poly/LUT clipping bounds |
| `--prompt-csv` | Path | none | Prompt file — CSV with `prompt` column, or tab-separated `seed<TAB>prompt` |
| `--num-images` | int | `150` | Number of images to generate |
| `--num-steps` | int | `30` | Denoising steps per image |
| `--cfg-scale` | float | `4.0` | CFG guidance weight |
| `--seed` | int | `42` | Base seed; image i uses seed+i (ignored when prompt file provides per-prompt seeds) |
| `--output-dir` | Path | `benchmark_results` | Root output directory |
| `--reference-dir` | Path | none | Reference image dir for FID/IS/KID (omit to skip distribution metrics) |
| `--baseline-dir` | Path | none | FP16 baseline image dir for paired metrics (PSNR + LPIPS) |
| `--generated-dir` | Path | none | Override generated image dir for metrics-only runs |
| `--skip-generation` | Flag | off | Skip image generation; compute metrics on existing images only |
| `--skip-metrics` | Flag | off | Skip all metrics; only generate images + record latency/memory |
| `--skip-clip-metrics` | Flag | off | Skip CLIP-based metrics (PRDC + CMMD) |
| `--skip-paired-metrics` | Flag | off | Skip paired metrics (PSNR + LPIPS) even if `--baseline-dir` is set |
| `--warmup` | int | `2` | Warmup images excluded from latency statistics |
| `--resume` | Flag | off | Skip images whose PNG already exists in `output-dir/images/` |
| `--reload-n` | int | none | Reload pipeline for first N images (memory profiling), then persist for remainder. Default: reload every image |
| `--eval-interval` | int | `0` | Compute cumulative metrics every N images and save to `benchmark_checkpoints.json` (0 = disabled) |
| `--mlx-int4` | Flag | off | Inject AdaRound weights as native MLX int4 |
| `--group-size` | int | `64` | Group size for MLX int4 |
