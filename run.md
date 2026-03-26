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
| `--block-size` | int | `128` | Column block size for GPTQ error compensation |
| `--group-size` | int | `128` | Group size for per-group weight quantization. Use `0` for per-channel (no grouping) |
| `--max-prompts` | int | all | Limit number of prompts for Phase A (Hessian collection) |
| `--alpha-prompts` | int | `5` | Number of prompts for Phase B (alpha search) |
| `--hessian-cache` | Path | `<output-dir>/hessians.npz` | Path to save/load Hessian checkpoint. If file exists, Phase A is skipped |
| `--static-act-quant` | Flag | off | Use static (timestep-agnostic) activation clipping instead of polynomial |
| `--skip-gptq` | Flag | off | Skip Phase A and B.1. Loads existing weights from `output-dir/weights/` and reruns alpha search only |
| `--weight-mode` | Choice | `gptq` | Weight quantization method. Choices: `gptq` (full GPTQ), `rtn` (round-to-nearest per-channel), `fp16` (no weight quant, activation-only alpha search) |

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
| `--output-dir` | Path | `gptq_comparison` | Output directory for generated images |
| `--num-steps` | int | `30` | Number of denoising steps |
| `--cfg-weight` | float | `4.0` | Classifier-free guidance weight |
| `--seed` | int | `42` | Random seed |
| `--latent-size` | int | `64` | Spatial size of latent (64 = 512px) |
| `--no-act-quant` | Flag | off | Disable activation quantization (weight-only mode) |
| `--no-alpha-scale` | Flag | off | Use poly clipping with `alpha_scale=1.0` for all layers (ignores learned scales) |
| `--poly-only` | Flag | off | W16A8 mode: poly activation clipping only, no weight quantization |
| `--skip-blocks` | int list | none | Block indices to leave at FP16 (e.g. `--skip-blocks 22 23`) |
| `--no-baseline` | Flag | off | Skip FP16 baseline generation (only produce quantized image) |
