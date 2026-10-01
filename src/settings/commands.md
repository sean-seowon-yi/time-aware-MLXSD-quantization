# CLI Command Reference

All commands assume the working directory is the project root.

---

## Table of Contents

1. [End-to-End Pipeline (`run_poly_alpha_pipeline`)](#1-end-to-end-pipeline)
2. [Phase 1 -- Diagnostic Collection (`run_collection`)](#2-phase-1--diagnostic-collection)
3. [Phase 1 -- Analysis & Plots (`run_analysis`)](#3-phase-1--analysis--plots)
4. [Phase 2 -- End-to-End Quantization (`run_e2e`)](#4-phase-2--end-to-end-quantization)
5. [Phase 2 -- Standalone Quantize (`run_quantize`)](#5-phase-2--standalone-quantize)
6. [Phase 2 -- Inference (`run_inference`)](#6-phase-2--inference)
7. [Phase 2 -- Post-Quantization Diagnostics (`run_diagnose`)](#7-phase-2--post-quantization-diagnostics)
8. [Phase 3 -- Polynomial Schedule (`generate_schedule`)](#8-phase-3--polynomial-schedule)
9. [Phase 4.1 -- Alpha Search (`alpha_search`)](#9-phase-41--alpha-search)
10. [Benchmark (GT comparison)](#10-benchmark-gt-comparison-pipeline)
11. [Recommended Recipes](#11-recommended-recipes)

---

## 1. End-to-End Pipeline

Orchestrates Phase 1+2 -> 3 -> 4.1 as subprocesses (RTN W4A8 + poly clipping + alpha search).

```bash
python -m src.run_poly_alpha_pipeline [FLAGS]
```

### Global I/O

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompts-file` | str | `src/settings/coco_100_calibration_prompts.txt` | Tab-separated `seed<TAB>prompt` file |
| `--output-dir` | str | `quantized` | Root output directory for Phase 2 checkpoint |
| `--diagnostics-dir` | str | `diagnostics` | Directory for Phase 1 diagnostics |
| `--quantized-dir` | str | None | Explicit Phase 2 checkpoint dir (auto-detected if not set) |

### Phase Control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--start-from` | choice | `e2e` | Start from this step. Choices: `e2e`, `poly`, `alpha` |
| `--skip-calibration` | flag | -- | Skip Phase 1 collection (use existing `--diagnostics-dir`) |

### Phase 1/2 Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-prompts-collection` | int | None (all) | Number of prompts for Phase 1 collection |
| `--alpha` | float | None (0.5) | CSB exponent |
| `--qkv-method` | choice | None (max) | Choices: `max`, `geomean`, `l2` |
| `--group-size` | int | None (64) | W4 group size |
| `--bits` | int | None (4) | Weight bit-width |
| `--final-layer-bits` | int | None (4) | Final layer bit-width |
| `--ssc-tau` | float | None (1.0) | SSC temperature; < 1 sharpens time-weighting |
| `--static-mode` | choice | `ssc_weighted` | Static scale aggregation. Choices: `ssc_weighted`, `global_max` |
| `--static-granularity` | choice | `per_tensor` | Static scale granularity. Choices: `per_tensor`, `per_channel` |

### Phase 3 Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-degree` | int | 4 | Maximum polynomial degree |
| `--per-channel-rho-threshold` | float | None | Per-channel poly for layers with mean rho above threshold |
| `--exclude-layers` | str (space-sep) | None | Layer keys to exclude from poly schedule |
| `--include-shifts` | flag | -- | Enable shift polynomial fitting |

### Phase 4.1 Alpha Search Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--alpha-prompts-file` | str | None | Prompts for alpha search (default: same as `--prompts-file`) |
| `--alpha-num-prompts` | int | 16 | Calibration prompts for alpha search |
| `--alpha-num-steps` | int | 30 | Denoising steps per prompt |
| `--alpha-cfg-weight` | float | 4.0 | CFG weight during alpha search |
| `--alpha-latent-size` | int | 64 | Latent spatial size |
| `--alpha-subsample-rows` | int | 128 | Row subsampling for NumPy matmul |
| `--no-merge-poly-schedule` | flag | -- | Do not merge `alpha_multiplier` into `poly_schedule.json` |
| `--no-poly-schedule-backup` | flag | -- | Do not backup existing `poly_schedule.json` before merge |
| `--poly-schedule` | str | None | Override path to `poly_schedule.json` |
| `--alpha-checkpoint` | str | None | Checkpoint JSON for resume |
| `--alpha-resume` | flag | -- | Use default checkpoint under quantized dir |
| `--keep-alpha-checkpoint` | flag | -- | Do not delete checkpoint after full run |
| `--alpha-reference-fp32` | flag | -- | Use FP32 reference matmul instead of default FP16 |

### Miscellaneous

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dry-run` | flag | -- | Print commands without executing |

---

## 2. Phase 1 -- Diagnostic Collection

Standalone Phase 1: run calibration prompts through the denoiser with hooks to collect activation trajectories and weight salience.

```bash
python -m src.phase1.run_collection [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--pilot` | flag | -- | Quick pilot run: 2 prompts, verify hooks fire correctly |
| `--num-prompts` | int | None (all 100) | Limit number of prompts |

Output: `diagnostics/` (activation stats, weight stats, adaLN stats, config)

---

## 3. Phase 1 -- Analysis & Plots

Generate all diagnostic plots from Phase 1 data. No arguments.

```bash
python -m src.phase1.run_analysis
```

Expects data in `diagnostics/` from a prior `run_collection` run.
Output: `diagnostics/plots/`

---

## 4. Phase 2 -- End-to-End Quantization

Combines Phase 1 collection + Phase 2 calibration + CSB + W4A8 quantization in a single script (loads model once).

```bash
python -m src.phase2.run_e2e [FLAGS]
```

### Output

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | str | `quantized` | Root output directory |
| `--diagnostics-dir` | str | `diagnostics` | Phase 1 diagnostics directory |

### Phase 1 -- Data Collection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--skip-collection` | flag | -- | Skip Phase 1; use existing data in `--diagnostics-dir` |
| `--num-prompts` | int | None (all 100) | Number of calibration prompts |
| `--num-steps` | int | None (30) | Denoising steps |
| `--cfg-weight` | float | None (4.0) | CFG scale |

### Phase 2 -- Quantization

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--alpha` | float | None (0.5) | CSB exponent |
| `--qkv-method` | choice | None (max) | Choices: `max`, `geomean`, `l2` |
| `--group-size` | int | None (64) | W4 group size |
| `--bits` | int | None (4) | Weight bit-width |
| `--final-layer-bits` | int | None (4) | Final layer bit-width |
| `--ssc-tau` | float | None (1.0) | SSC temperature |
| `--static-mode` | choice | `ssc_weighted` | Choices: `ssc_weighted`, `global_max` |
| `--static-granularity` | choice | `per_tensor` | Choices: `per_tensor`, `per_channel` |

Output: `quantized/<tag>/` (e.g. `quantized/w4a8_l2_a0.50_gs32_static/`)

---

## 5. Phase 2 -- Standalone Quantize

Run Phase 2 quantization only (without Phase 1 collection). Requires prior Phase 1 data.

```bash
python -m src.phase2.run_quantize [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | str | `quantized` | Output directory |
| `--diagnostics-dir` | str | `diagnostics` | Phase 1 diagnostics directory |
| `--calibrate-only` | flag | -- | Only compute and save calibration data (no model load) |
| `--from-calibration` | str | None | Load calibration from this dir instead of recomputing |
| `--alpha` | float | None (0.5) | CSB exponent |
| `--qkv-method` | choice | None (max) | Choices: `max`, `geomean`, `l2` |
| `--group-size` | int | None (64) | W4 group size |
| `--bits` | int | None (4) | Weight bit-width |
| `--final-layer-bits` | int | None (4) | Final layer bit-width |
| `--ssc-tau` | float | None (1.0) | SSC temperature |
| `--static-mode` | choice | `ssc_weighted` | Choices: `ssc_weighted`, `global_max` |
| `--static-granularity` | choice | `per_tensor` | Choices: `per_tensor`, `per_channel` |

---

## 6. Phase 2 -- Inference

Generate images with FP16 or W4A8-quantized model.

```bash
python -m src.phase2.run_inference [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | choice | (required) | Choices: `fp16`, `w4a8` |
| `--quantized-dir` | str | None | Quantized model directory (required for `--mode w4a8`) |
| `--prompt` | str | -- | Single text prompt (**one of** `--prompt` or `--prompts-file` is required) |
| `--prompts-file` | str | -- | Path to text file with one prompt per line |
| `--seed` | int | 42 | Seed for single `--prompt` mode |
| `--num-steps` | int | 30 | Denoising steps |
| `--cfg-weight` | float | 4.0 | CFG guidance weight |
| `--latent-size` | int int | 64 64 | Latent size H W |
| `--num-prompts` | int | None (all) | Limit prompt-seed pairs from `--prompts-file` |
| `--eval-indices-file` | str | None | JSON file listing prompt indices to generate |
| `--output-dir` | str | `results` | Output directory |

---

## 7. Phase 2 -- Post-Quantization Diagnostics

Compare W4A8 vs FP16 activations/weights and generate diagnostic plots.

```bash
python -m src.phase2.run_diagnose [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--quantized-dir` | str | `quantized` | Quantized model directory |
| `--diagnostics-dir` | str | `diagnostics` | Phase 1 FP16 diagnostics directory |
| `--output-dir` | str | `post_quant_diagnostics` | Output directory |
| `--skip-collection` | flag | -- | Skip W4A8 activation collection |
| `--analysis-only` | flag | -- | Only run analysis + plots (no model loading) |
| `--num-prompts` | int | None (all) | Number of prompt-seed pairs |
| `--num-steps` | int | 30 | Denoising steps |
| `--cfg-weight` | float | 4.0 | CFG scale |

---

## 8. Phase 3 -- Polynomial Schedule

Generate polynomial clipping schedule from Phase 1/2 artifacts.

```bash
python -m src.phase3.generate_schedule [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--diagnostics-dir` | Path | `diagnostics` | Phase 1 diagnostics directory |
| `--calibration-dir` | Path | (required) | Phase 2 output directory with `calibration.npz` + `calibration_meta.json` |
| `--output` | Path | `<calibration-dir>/poly_schedule.json` | Output path for schedule |
| `--max-degree` | int | 4 | Maximum polynomial degree (0 = static constant) |
| `--include-shifts` | flag | -- | (Future) Fit shift polynomials for asymmetric quantization |
| `--per-channel-rho-threshold` | float | None | Per-channel poly for layers with mean rho above threshold |
| `--exclude-layers` | str... | None | Layer names to exclude (e.g. `context_embedder`) |

Output: `poly_schedule.json` in the calibration directory

### Phase 3 -- Visualize Poly Fits

Plot polynomial clipping fits against pre/post-CSB activation trajectories.

```bash
python -m src.phase3.visualize_poly [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--diagnostics-dir` | Path | `diagnostics` | Phase 1 diagnostics directory |
| `--calibration-dir` | Path | (required) | Phase 2 output with `calibration.npz` + `poly_schedule.json` |
| `--output-dir` | Path | `plots/poly_clipping` | Output directory for PNGs |
| `--layers` | str... | None (auto) | Specific layer names to plot |
| `--non-static-only` | flag | -- | Only plot layers with polynomial degree > 0 |
| `--max-layers` | int | 50 | Max individual layer plots |

Output: Per-layer 2-panel plots, summary grid, int8 utilization comparison chart.

---

## 9. Phase 4.1 -- Alpha Search

Per-layer activation alpha multiplier search: tunes poly clipping multipliers by minimizing MSE between FP16 reference linear and poly-style int8 proxy (with RTN dequant weights).

```bash
python -m src.phase4_1.alpha_search [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--quantized-dir` | Path | (required) | Phase 2 checkpoint (quantize_config.json + weights) |
| `--prompts-file` | Path | None | seed<TAB>prompt file (default: `src/settings/coco_100_calibration_prompts.txt`) |
| `--poly-schedule` | Path | None | Override poly_schedule.json path (default: `<quantized-dir>/poly_schedule.json`) |
| `--num-prompts` | int | 16 | Number of calibration prompts |
| `--num-steps` | int | 30 | Denoising steps per prompt |
| `--cfg-weight` | float | 4.0 | CFG guidance weight |
| `--latent-size` | int | 64 | Latent spatial size |
| `--subsample-rows` | int | 128 | Row subsampling for NumPy matmul |
| `--output-json` | Path | None | Output path (default: `<quantized-dir>/alpha_search_results.json`) |
| `--no-merge-poly-schedule` | flag | -- | Do not write `alpha_multiplier` into `poly_schedule.json` |
| `--no-poly-schedule-backup` | flag | -- | Do not backup existing `poly_schedule.json` before merge |
| `--checkpoint` | Path | None | Save/load resume state after each prompt |
| `--resume` | flag | -- | Use `<quantized-dir>/alpha_search_checkpoint.json` as checkpoint |
| `--keep-checkpoint` | flag | -- | Keep checkpoint file after a full successful run |
| `--reference-fp32` | flag | -- | Use FP32 matmul for the MSE reference (default: FP16 matmul) |

Output: `alpha_search_results.json`; updated `poly_schedule.json` with per-layer `alpha_multiplier` (default). Deploy via `load_quantized_model_poly`.

---

## 10. Benchmark (GT comparison pipeline)

Generate W4A8 images if needed, then compute FID, CMMD, CLIP image-text scores, and LPIPS against ground truth and FP16 baselines. Modular metrics live under `src/benchmark/`.

```bash
python -m src.benchmark.gt_comparison_pipeline [FLAGS]
```

### Required

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ground-truth-dir` | Path | (required) | Ground-truth images directory |
| `--fp16-images-dir` | Path | (required) | Directory of FP16-generated PNGs |
| `--quantized-dir` | Path | (required) | Phase 2 W4A8 checkpoint |
| `--output-dir` | Path | (required) | Run root; W4A8 images written to `output_dir/images/` |

### Generation and prompts

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt-file` | Path | None | If omitted: `src/settings/evaluation_set.txt` |
| `--config` | str | `w4a8` | Must be `w4a8`, `w4a8_static`, or `w4a8_poly` |
| `--num-images` | int | None | Cap on images |
| `--img-digits` | int | None | PNG zero-padding |
| `--num-steps` | int | `30` | Denoising steps |
| `--cfg-scale` | float | `4.0` | CFG scale |
| `--seed` | int | `42` | Base seed |
| `--warmup` | int | `0` | Warmup images |
| `--no-resume` | flag | off | Regenerate every W4A8 PNG |
| `--reload-n` | int | `1` | Pipeline reload strategy |
| `--force-w4a8-gen` | flag | off | Always run the generation step |
| `--poly-schedule` | Path | None | JSON schedule for `w4a8_poly` |
| `--group-size` | int | `64` | MLX group size |

### Metric knobs

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--kid-subset-max` | int | `1000` | torch-fidelity KID subset cap |
| `--fidelity-cuda` | flag | off | torch-fidelity on CUDA |
| `--clip-arch` | str | `ViT-L-14-336` | open_clip architecture |
| `--clip-pretrained` | str | `openai` | open_clip pretrained tag |
| `--clip-batch-size` | int | `16` | CLIP embedding batch size |
| `--lpips-net` | str | `alex` | LPIPS backbone |
| `--lpips-resize` | int | `256` | LPIPS resize |

### Metrics computed

- **FID** FP16 vs GT, W4A8 vs GT (via torch-fidelity)
- **CMMD** FP16 vs GT, W4A8 vs GT (via CLIP embeddings + RBF-MMD)
- **CLIP score** FP16 vs prompts, W4A8 vs prompts (mean image-text cosine)
- **LPIPS** W4A8 vs FP16 (paired by filename)

### Outputs

- `gt_comparison_results.json` under resolved `--output-dir`.

---

## 11. Recommended recipes

**Authoritative flag lists:** S1 (`run_poly_alpha_pipeline`), S2-9 (phase CLIs), S10 (`gt_comparison_pipeline`). Recipes below are copy-paste-safe shell continuations.

### A. Full Pipeline (Phase 1+2 -> 3 -> 4.1)

Run everything from scratch:

```bash
python -m src.run_poly_alpha_pipeline \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --output-dir quantized \
    --diagnostics-dir diagnostics \
    --qkv-method l2 \
    --alpha 0.5 \
    --group-size 32 \
    --bits 4 \
    --static-mode ssc_weighted \
    --static-granularity per_tensor \
    --ssc-tau 1.0 \
    --num-prompts-collection 100 \
    --max-degree 4 \
    --per-channel-rho-threshold 0.5 \
    --alpha-num-prompts 16 \
    --alpha-num-steps 30 \
    --alpha-cfg-weight 4.0
```

### B. Skip Phase 1 (reuse diagnostics)

```bash
python -m src.run_poly_alpha_pipeline \
    --skip-calibration \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --output-dir quantized \
    --diagnostics-dir diagnostics \
    --qkv-method l2 \
    --alpha 0.5 \
    --group-size 32 \
    --bits 4 \
    --static-mode ssc_weighted \
    --static-granularity per_tensor \
    --max-degree 4 \
    --alpha-num-prompts 16
```

### C. Poly + Alpha only (existing Phase 2 checkpoint)

```bash
python -m src.run_poly_alpha_pipeline \
    --start-from poly \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --diagnostics-dir diagnostics \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --max-degree 4 \
    --alpha-num-prompts 16
```

### D. Alpha search only (existing poly schedule)

```bash
python -m src.run_poly_alpha_pipeline \
    --start-from alpha \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --alpha-num-prompts 16 \
    --alpha-resume
```

### E. Phase 3 Only (Generate Poly Schedule)

```bash
python -m src.phase3.generate_schedule \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \
    --output quantized/w4a8_l2_a0.50_gs32_static/poly_schedule.json \
    --max-degree 4 \
    --per-channel-rho-threshold 0.5 \
    --exclude-layers context_embedder
```

### F. Phase 2 E2E (Collection + Quantization)

```bash
python -m src.phase2.run_e2e \
    --output-dir quantized \
    --diagnostics-dir diagnostics \
    --skip-collection \
    --num-prompts 100 \
    --num-steps 30 \
    --cfg-weight 4.0 \
    --alpha 0.5 \
    --qkv-method l2 \
    --group-size 32 \
    --bits 4 \
    --final-layer-bits 4 \
    --ssc-tau 1.0 \
    --static-mode ssc_weighted \
    --static-granularity per_tensor
```

### G. Phase 4.1 Alpha Search (standalone)

```bash
python -m src.phase4_1.alpha_search \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --num-prompts 16 \
    --num-steps 30 \
    --cfg-weight 4.0 \
    --resume
```

### H. Benchmark -- GT comparison (FP16 vs W4A8 vs GT)

```bash
python -m src.benchmark.gt_comparison_pipeline \
    --ground-truth-dir /path/to/gt_images \
    --fp16-images-dir benchmark_results/fp16_p2/images \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --output-dir benchmark_results/w4a8_gt_eval \
    --config w4a8_poly \
    --prompt-file src/settings/evaluation_set.txt \
    --num-steps 30 \
    --cfg-scale 4.0 \
    --seed 42
```

### J. Phase 1 -- Quick Pilot Run

```bash
python -m src.phase1.run_collection --pilot
```

### K. Post-Quantization Diagnostics

```bash
python -m src.phase2.run_diagnose \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --diagnostics-dir diagnostics \
    --output-dir post_quant_diagnostics \
    --num-prompts 100 \
    --num-steps 30 \
    --cfg-weight 4.0 \
    --skip-collection
```

---

## Prompt Files

| File | Purpose |
|------|---------|
| `src/settings/coco_100_calibration_prompts.txt` | 100 calibration prompts (Phase 1/2 collection, Phase 4.1 alpha search) |
| `src/settings/evaluation_set.txt` | Evaluation prompts for benchmark image generation |

---

## Default Configuration (Phase 2)

Defined in `src/phase2/config.py`:

| Parameter | Default |
|-----------|---------|
| `alpha` | 0.5 |
| `group_size` | 64 |
| `bits` | 4 |
| `a_bits` | 8 |
| `qkv_method` | `max` |
| `ssc_tau` | 1.0 |
| `final_layer_bits` | 4 |
| `exclude_layers` | `["context_embedder"]` |

The output directory tag format is: `w{bits}a{a_bits}_{qkv}_a{alpha:.2f}_gs{gs}[_t{tau}]_static` or with `_staticpc` for per-channel.

Example: `w4a8_l2_a0.50_gs32_static`
