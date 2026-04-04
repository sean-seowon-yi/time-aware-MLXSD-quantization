# CLI Command Reference

All commands assume the working directory is the project root.

---

## Table of Contents

1. [End-to-End Pipeline (`run_pipeline`)](#1-end-to-end-pipeline)
2. [Phase 1 — Diagnostic Collection (`run_collection`)](#2-phase-1--diagnostic-collection)
3. [Phase 1 — Analysis & Plots (`run_analysis`)](#3-phase-1--analysis--plots)
4. [Phase 2 — End-to-End Quantization (`run_e2e`)](#4-phase-2--end-to-end-quantization)
5. [Phase 2 — Standalone Quantize (`run_quantize`)](#5-phase-2--standalone-quantize)
6. [Phase 2 — Inference (`run_inference`)](#6-phase-2--inference)
7. [Phase 2 — Post-Quantization Diagnostics (`run_diagnose`)](#7-phase-2--post-quantization-diagnostics)
8. [Phase 3 — Polynomial Schedule (`generate_schedule`)](#8-phase-3--polynomial-schedule)
9. [Phase 4 — GPTQ (`run_phase4`)](#9-phase-4--gptq)
10. [Benchmark (`benchmark_model`)](#10-benchmark)
11. [Recommended Recipes](#11-recommended-recipes)

---

## 1. End-to-End Pipeline

Orchestrates Phase 1 → 2 → 3 → 4 as subprocesses.

```bash
python -m src.run_pipeline [FLAGS]
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
| `--start-from` | choice | `phase1` | Start from this phase. Choices: `phase1`, `phase2`, `phase3`, `phase4` |
| `--stop-after` | choice | `phase4` | Stop after this phase. Choices: `phase1`, `phase2`, `phase3`, `phase4` |
| `--skip-phase1` | flag | — | Shorthand for `--start-from phase2` |

### Phase 1/2 Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-prompts-collection` | int | None (all) | Number of prompts for Phase 1 collection |
| `--alpha` | float | None (0.5) | CSB exponent |
| `--qkv-method` | choice | None (max) | Choices: `max`, `geomean`, `l2` |
| `--group-size` | int | None (64) | W4 group size |
| `--bits` | int | None (4) | Weight bit-width |
| `--ssc-tau` | float | None (1.0) | SSC temperature; < 1 sharpens time-weighting |
| `--per-token-rho-threshold` | float | None (0.5) | Layers above this threshold use per-token A8 |
| `--act-quant` | choice | `dynamic` | A8 mode. Choices: `dynamic`, `static` |
| `--static-mode` | choice | `ssc_weighted` | Static scale aggregation. Choices: `ssc_weighted`, `global_max` |
| `--static-granularity` | choice | `per_tensor` | Static scale granularity. Choices: `per_tensor`, `per_channel` |

### Phase 3 Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-degree` | int | 4 | Maximum polynomial degree |
| `--per-channel-rho-threshold` | float | None | Per-channel poly for layers with mean rho above threshold |

### Phase 4 Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-prompts-gptq` | int | 16 | Number of prompts for Hessian collection |
| `--gptq-block-size` | int | 128 | GPTQ column block size |
| `--gptq-damp-percent` | float | 0.01 | Hessian diagonal damping |
| `--raw-hessian` | flag | — | Collect Hessians without poly fake-quant |

### Miscellaneous

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dry-run` | flag | — | Print commands without executing |

---

## 2. Phase 1 — Diagnostic Collection

Standalone Phase 1: run calibration prompts through the denoiser with hooks to collect activation trajectories and weight salience.

```bash
python -m src.phase1.run_collection [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--pilot` | flag | — | Quick pilot run: 2 prompts, verify hooks fire correctly |
| `--num-prompts` | int | None (all 100) | Limit number of prompts |

Output: `diagnostics/` (activation stats, weight stats, adaLN stats, config)

---

## 3. Phase 1 — Analysis & Plots

Generate all diagnostic plots from Phase 1 data. No arguments.

```bash
python -m src.phase1.run_analysis
```

Expects data in `diagnostics/` from a prior `run_collection` run.
Output: `diagnostics/plots/`

---

## 4. Phase 2 — End-to-End Quantization

Combines Phase 1 collection + Phase 2 calibration + CSB + W4A8 quantization in a single script (loads model once).

```bash
python -m src.phase2.run_e2e [FLAGS]
```

### Output

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | str | `quantized` | Root output directory |
| `--diagnostics-dir` | str | `diagnostics` | Phase 1 diagnostics directory |

### Phase 1 — Data Collection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--skip-collection` | flag | — | Skip Phase 1; use existing data in `--diagnostics-dir` |
| `--num-prompts` | int | None (all 100) | Number of calibration prompts |
| `--num-steps` | int | None (30) | Denoising steps |
| `--cfg-weight` | float | None (4.0) | CFG scale |

### Phase 2 — Quantization

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--alpha` | float | None (0.5) | CSB exponent |
| `--qkv-method` | choice | None (max) | Choices: `max`, `geomean`, `l2` |
| `--group-size` | int | None (64) | W4 group size |
| `--bits` | int | None (4) | Weight bit-width |
| `--final-layer-bits` | int | None (4) | Final layer bit-width |
| `--ssc-tau` | float | None (1.0) | SSC temperature |
| `--per-token-rho-threshold` | float | None (0.5) | Per-token A8 rho threshold |

### Activation Quantization Mode

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--act-quant` | choice | `dynamic` | Choices: `dynamic`, `static` |
| `--static-mode` | choice | `ssc_weighted` | Choices: `ssc_weighted`, `global_max` |
| `--static-granularity` | choice | `per_tensor` | Choices: `per_tensor`, `per_channel` |

Output: `quantized/<tag>/` (e.g. `quantized/w4a8_l2_a0.50_gs32_static/`)

---

## 5. Phase 2 — Standalone Quantize

Run Phase 2 quantization only (without Phase 1 collection). Requires prior Phase 1 data.

```bash
python -m src.phase2.run_quantize [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | str | `quantized` | Output directory |
| `--diagnostics-dir` | str | `diagnostics` | Phase 1 diagnostics directory |
| `--calibrate-only` | flag | — | Only compute and save calibration data (no model load) |
| `--from-calibration` | str | None | Load calibration from this dir instead of recomputing |
| `--alpha` | float | None (0.5) | CSB exponent |
| `--qkv-method` | choice | None (max) | Choices: `max`, `geomean`, `l2` |
| `--group-size` | int | None (64) | W4 group size |
| `--bits` | int | None (4) | Weight bit-width |
| `--final-layer-bits` | int | None (4) | Final layer bit-width |
| `--ssc-tau` | float | None (1.0) | SSC temperature |
| `--per-token-rho-threshold` | float | None (0.5) | Per-token A8 rho threshold |

---

## 6. Phase 2 — Inference

Generate images with FP16 or W4A8-quantized model.

```bash
python -m src.phase2.run_inference [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | choice | (required) | Choices: `fp16`, `w4a8` |
| `--quantized-dir` | str | None | Quantized model directory (required for `--mode w4a8`) |
| `--prompt` | str | — | Single text prompt (mutually exclusive with `--prompts-file`) |
| `--prompts-file` | str | — | Path to text file with one prompt per line |
| `--seed` | int | 42 | Seed for single `--prompt` mode |
| `--num-steps` | int | 30 | Denoising steps |
| `--cfg-weight` | float | 4.0 | CFG guidance weight |
| `--latent-size` | int int | 64 64 | Latent size H W (→ 512x512) |
| `--num-prompts` | int | None (all) | Limit prompt-seed pairs from `--prompts-file` |
| `--eval-indices-file` | str | None | JSON file listing prompt indices to generate |
| `--output-dir` | str | `results` | Output directory |

---

## 7. Phase 2 — Post-Quantization Diagnostics

Compare W4A8 vs FP16 activations/weights and generate diagnostic plots.

```bash
python -m src.phase2.run_diagnose [FLAGS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--quantized-dir` | str | `quantized` | Quantized model directory |
| `--diagnostics-dir` | str | `diagnostics` | Phase 1 FP16 diagnostics directory |
| `--output-dir` | str | `post_quant_diagnostics` | Output directory |
| `--skip-collection` | flag | — | Skip W4A8 activation collection |
| `--analysis-only` | flag | — | Only run analysis + plots (no model loading) |
| `--num-prompts` | int | None (all) | Number of prompt-seed pairs |
| `--num-steps` | int | 30 | Denoising steps |
| `--cfg-weight` | float | 4.0 | CFG scale |

---

## 8. Phase 3 — Polynomial Schedule

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
| `--include-shifts` | flag | — | (Future) Fit shift polynomials for asymmetric quantization |
| `--per-channel-rho-threshold` | float | None | Per-channel poly for layers with mean rho above threshold |
| `--exclude-layers` | str... | None | Layer names to exclude (e.g. `context_embedder`) |

Output: `poly_schedule.json` in the calibration directory

---

## 9. Phase 4 — GPTQ

Hessian-weighted error compensation for weight quantization.

```bash
python -m src.phase4.run_phase4 [FLAGS]
```

### I/O

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--quantized-dir` | Path | (required) | Phase 2 checkpoint directory (must contain `quantize_config.json`) |
| `--prompts-file` | Path | None | Tab-separated `seed<TAB>prompt` file for Hessian calibration |
| `--output-dir` | Path | `--quantized-dir` | Output directory (defaults to in-place overwrite) |

### Hessian Collection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--skip-collection` | flag | — | Skip Hessian collection; reuse saved Hessians from `<output-dir>/hessians/` |
| `--num-prompts` | int | 16 | Number of prompts for Hessian collection |
| `--num-steps` | int | 30 | Denoising steps per prompt |
| `--cfg-weight` | float | 4.0 | CFG guidance weight |
| `--raw-hessian` | flag | — | Collect Hessians from full-precision activations (no poly fake-quant) |

### GPTQ

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--bits` | int | 4 | Quantization bit-width (must match Phase 2) |
| `--group-size` | int | None (from Phase 2) | Group size (must match Phase 2; auto-read from checkpoint) |
| `--block-size` | int | 128 | GPTQ column block size for error compensation |
| `--damp-percent` | float | 0.01 | Hessian diagonal damping factor |

### Skip Steps

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--skip-gptq` | flag | — | Only collect + save Hessians (skip GPTQ optimization) |

Output: Overwrites `mmdit_quantized.safetensors` + `quantize_config.json` in the output directory. Hessians saved under `<output-dir>/hessians/`.

---

## 10. Benchmark

Generate images and compute quality metrics (FID, IS, KID, PSNR, LPIPS, etc.).

```bash
python -m src.benchmark.benchmark_model [FLAGS]
```

### Model Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | str | `fp16` | Quantization config. Choices: `fp16`, `fp16_p2`, `naive_int8`, `gptq`, `w4a8`, `w4a8_static`, `w4a8_poly` |
| `--quantized-dir` | Path | None | Phase 2 W4A8 quantized model directory (required for `w4a8`/`w4a8_static`/`w4a8_poly`) |
| `--gptq-dir` | Path | None | Legacy GPTQ output dir (use with `--config gptq`) |
| `--adaround-output` | Path | None | AdaRound weights dir |
| `--adaround-act-config` | Path | None | Activation quant config JSON |
| `--poly-schedule` | Path | None | Legacy polynomial clipping schedule JSON |
| `--lut-schedule` | Path | None | LUT clipping schedule JSON |
| `--poly-margin` | float | 1.0 | Multiplier on poly/LUT clipping bounds |
| `--taqdit-output` | Path | None | TaQ-DiT weights dir (reserved) |
| `--taqdit-act-config` | Path | None | TaQ-DiT act config JSON (reserved) |
| `--mlx-int4` | flag | — | Inject AdaRound weights as native MLX int4 |
| `--group-size` | int | 64 | Group size for MLX int4 |

### Image Generation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt-csv` | Path | None | Prompt file: CSV with `prompt` column, or tab-separated `seed<TAB>prompt` .txt. Default: `src/settings/evaluation_set.txt` |
| `--num-images` | int | 150 | Number of images to generate |
| `--num-steps` | int | 30 | Denoising steps per image |
| `--cfg-scale` | float | 4.0 | CFG guidance weight |
| `--seed` | int | 42 | Base seed; image i uses `seed + i` |

### Output & Metrics

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | Path | `benchmark_results` | Root output directory |
| `--reference-dir` | Path | None | Reference image dir for FID/IS/KID |
| `--generated-dir` | Path | None | Override generated image dir for metrics-only mode |
| `--baseline-dir` | Path | None | FP16 baseline image dir for paired metrics (PSNR + LPIPS) |
| `--ground-truth-dir` | Path | None | Ground truth image dir for distributional metrics |
| `--fp16-dir` | Path | None | FP16 image dir for ground truth comparison (used with `--ground-truth-dir`) |

### Phase Control & Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--skip-generation` | flag | — | Skip image generation; compute metrics only |
| `--skip-metrics` | flag | — | Skip FID/IS/KID; only generate + latency/memory |
| `--skip-clip-metrics` | flag | — | Skip CLIP-based metrics (PRDC + CMMD) |
| `--skip-paired-metrics` | flag | — | Skip paired metrics (PSNR + LPIPS) even if `--baseline-dir` is set |
| `--warmup` | int | 2 | Warmup images excluded from latency stats |
| `--resume` | flag | — | Skip images whose PNG already exists |
| `--reload-n` | int | None (reload every) | Reload pipeline for first N images (memory profiling), then persist |
| `--eval-interval` | int | 0 | Compute cumulative metrics every N images (0 = disabled) |

---

## 11. Recommended Recipes

Every recipe below shows **all available flags** for the command. Flags marked `# optional` use the default value shown — omit them if the default is acceptable.

### A. Full Pipeline (Phase 1 → 2 → 3 → 4) — All Flags

Run everything from scratch:

```bash
python -m src.run_pipeline \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --output-dir quantized \
    --diagnostics-dir diagnostics \
    --qkv-method l2 \
    --alpha 0.5 \
    --group-size 32 \
    --bits 4                              # optional, default 4 \
    --act-quant static \
    --static-mode ssc_weighted            # optional, default ssc_weighted \
    --static-granularity per_tensor       # optional, default per_tensor \
    --ssc-tau 1.0                         # optional, default 1.0 \
    --per-token-rho-threshold 0.5         # optional, default 0.5 \
    --num-prompts-collection 100          # optional, default all \
    --max-degree 4                        # optional, Phase 3 poly degree \
    --per-channel-rho-threshold 0.5       # optional, omit for per-tensor only \
    --num-prompts-gptq 16                 # optional, Phase 4 Hessian prompts \
    --gptq-block-size 128                 # optional, GPTQ column block size \
    --gptq-damp-percent 0.01             # optional, Hessian damping \
    --dry-run                             # optional, print commands only
```

### B. Phase 2 → 3 → 4 (Skip Phase 1 Collection) — All Flags

Reuse existing Phase 1 diagnostics:

```bash
python -m src.run_pipeline \
    --skip-phase1 \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --output-dir quantized \
    --diagnostics-dir diagnostics \
    --qkv-method l2 \
    --alpha 0.5 \
    --group-size 32 \
    --bits 4                              # optional \
    --act-quant static \
    --static-mode ssc_weighted            # optional \
    --static-granularity per_tensor       # optional \
    --ssc-tau 1.0                         # optional \
    --per-token-rho-threshold 0.5         # optional \
    --max-degree 4                        # optional \
    --per-channel-rho-threshold 0.5       # optional \
    --num-prompts-gptq 16                 # optional \
    --gptq-block-size 128                 # optional \
    --gptq-damp-percent 0.01             # optional \
    --raw-hessian                         # optional, skip poly fake-quant in Hessian
```

### C. Phase 3 → 4 Only (Poly Schedule + GPTQ on Existing Phase 2) — All Flags

```bash
python -m src.run_pipeline \
    --start-from phase3 \
    --stop-after phase4 \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --diagnostics-dir diagnostics \
    --max-degree 4                        # optional \
    --per-channel-rho-threshold 0.5       # optional \
    --num-prompts-gptq 16                 # optional \
    --gptq-block-size 128                 # optional \
    --gptq-damp-percent 0.01             # optional \
    --bits 4                              # optional \
    --raw-hessian                         # optional
```

### D. Phase 4 Only (GPTQ on Existing Checkpoint) — All Flags

Via orchestrator:

```bash
python -m src.run_pipeline \
    --start-from phase4 \
    --stop-after phase4 \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --num-prompts-gptq 16                 # optional \
    --gptq-block-size 128                 # optional \
    --gptq-damp-percent 0.01             # optional \
    --bits 4                              # optional \
    --raw-hessian                         # optional
```

Or standalone with Phase 4's own CLI (all flags):

```bash
python -m src.phase4.run_phase4 \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --prompts-file src/settings/coco_100_calibration_prompts.txt \
    --output-dir quantized/w4a8_l2_a0.50_gs32_static  # optional, defaults to --quantized-dir \
    --num-prompts 16                      # optional \
    --num-steps 30                        # optional \
    --cfg-weight 4.0                      # optional \
    --bits 4                              # optional, must match Phase 2 \
    --group-size 32                       # optional, auto-read from Phase 2 \
    --block-size 128                      # optional \
    --damp-percent 0.01                   # optional \
    --raw-hessian                         # optional \
    --skip-collection                     # optional, reuse saved Hessians \
    --skip-gptq                           # optional, only collect Hessians
```

### E. Phase 3 Only (Generate Poly Schedule) — All Flags

```bash
python -m src.phase3.generate_schedule \
    --diagnostics-dir diagnostics \
    --calibration-dir quantized/w4a8_l2_a0.50_gs32_static \
    --output quantized/w4a8_l2_a0.50_gs32_static/poly_schedule.json  # optional \
    --max-degree 4                        # optional \
    --per-channel-rho-threshold 0.5       # optional \
    --exclude-layers context_embedder     # optional, space-separated names \
    --include-shifts                      # optional, future use
```

### F. Phase 2 E2E (Collection + Quantization) — All Flags

```bash
python -m src.phase2.run_e2e \
    --output-dir quantized \
    --diagnostics-dir diagnostics \
    --skip-collection                     # optional, reuse Phase 1 data \
    --num-prompts 100                     # optional \
    --num-steps 30                        # optional \
    --cfg-weight 4.0                      # optional \
    --alpha 0.5 \
    --qkv-method l2 \
    --group-size 32 \
    --bits 4                              # optional \
    --final-layer-bits 4                  # optional \
    --ssc-tau 1.0                         # optional \
    --per-token-rho-threshold 0.5         # optional \
    --act-quant static \
    --static-mode ssc_weighted            # optional \
    --static-granularity per_tensor       # optional
```

### G. Phase 2 E2E — Static A8 Per-Channel

```bash
python -m src.phase2.run_e2e \
    --output-dir quantized \
    --skip-collection \
    --alpha 0.5 \
    --qkv-method l2 \
    --group-size 32 \
    --act-quant static \
    --static-granularity per_channel \
    --static-mode ssc_weighted
```

### H. Phase 2 Standalone Quantize — All Flags

```bash
python -m src.phase2.run_quantize \
    --output-dir quantized \
    --diagnostics-dir diagnostics \
    --calibrate-only                      # optional, only save calibration \
    --from-calibration quantized/w4a8_l2_a0.50_gs32_static  # optional \
    --alpha 0.5                           # optional \
    --qkv-method l2                       # optional \
    --group-size 32                       # optional \
    --bits 4                              # optional \
    --final-layer-bits 4                  # optional \
    --ssc-tau 1.0                         # optional \
    --per-token-rho-threshold 0.5         # optional
```

Note: `--calibrate-only` and `--from-calibration` are mutually exclusive.

### I. Phase 2 Inference — All Flags

```bash
python -m src.phase2.run_inference \
    --mode w4a8 \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --prompts-file src/settings/evaluation_set.txt \
    --seed 42                             # optional, for --prompt mode \
    --num-steps 30                        # optional \
    --cfg-weight 4.0                      # optional \
    --latent-size 64 64                   # optional \
    --num-prompts 10                      # optional \
    --eval-indices-file indices.json      # optional \
    --output-dir results
```

Note: `--prompt "a cat"` and `--prompts-file` are mutually exclusive (one required).

### J. Benchmark — FP16 Baseline — All Flags

```bash
python -m src.benchmark.benchmark_model \
    --config fp16_p2 \
    --prompt-csv src/settings/evaluation_set.txt  # optional, auto-detected \
    --num-images 150 \
    --num-steps 30                        # optional \
    --cfg-scale 4.0                       # optional \
    --seed 42                             # optional \
    --output-dir benchmark_results/fp16_p2 \
    --warmup 2                            # optional \
    --resume                              # optional \
    --reload-n 1                          # optional \
    --eval-interval 50                    # optional, checkpoints every 50 images \
    --skip-metrics                        # optional, skip FID/IS if no reference
```

### K. Benchmark — W4A8 + Poly + GPTQ (Full Generation + All Metrics) — All Flags

```bash
python -m src.benchmark.benchmark_model \
    --config w4a8_poly \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --prompt-csv src/settings/evaluation_set.txt  # optional \
    --num-images 150 \
    --num-steps 30                        # optional \
    --cfg-scale 4.0                       # optional \
    --seed 42                             # optional \
    --output-dir benchmark_results/w4a8_poly_gptq \
    --baseline-dir benchmark_results/fp16_p2/images  # optional, for PSNR/LPIPS \
    --reference-dir benchmark_results/fp16_p2/images  # optional, for FID/IS/KID \
    --warmup 2                            # optional \
    --resume                              # optional \
    --reload-n 1                          # optional \
    --eval-interval 50                    # optional \
    --skip-clip-metrics                   # optional, skip PRDC/CMMD \
    --skip-paired-metrics                 # optional, skip PSNR/LPIPS
```

### L. Benchmark — Ground Truth Comparison (FID/IS for W4A8 + FP16 vs GT) — All Flags

Metrics-only mode (images already generated):

```bash
python -m src.benchmark.benchmark_model \
    --skip-generation \
    --generated-dir benchmark_results/w4a8_poly_gptq/images \
    --ground-truth-dir /path/to/coco_gt_images \
    --fp16-dir benchmark_results/fp16_p2/images \
    --output-dir benchmark_results/w4a8_poly_gptq \
    --skip-clip-metrics                   # optional \
    --skip-paired-metrics                 # optional
```

### M. Benchmark — Metrics Only on Existing Images — All Flags

```bash
python -m src.benchmark.benchmark_model \
    --skip-generation \
    --generated-dir benchmark_results/w4a8_poly_gptq/images \
    --baseline-dir benchmark_results/fp16_p2/images  # optional, for PSNR/LPIPS \
    --reference-dir benchmark_results/fp16_p2/images  # optional, for FID/IS/KID \
    --ground-truth-dir /path/to/coco_gt_images        # optional \
    --fp16-dir benchmark_results/fp16_p2/images       # optional \
    --output-dir benchmark_results/w4a8_poly_gptq \
    --skip-clip-metrics                   # optional \
    --skip-paired-metrics                 # optional
```

### N. Phase 1 — Quick Pilot Run

```bash
python -m src.phase1.run_collection --pilot
```

### O. Phase 1 — Full Collection

```bash
python -m src.phase1.run_collection --num-prompts 100
```

### P. Post-Quantization Diagnostics — All Flags

```bash
python -m src.phase2.run_diagnose \
    --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
    --diagnostics-dir diagnostics \
    --output-dir post_quant_diagnostics \
    --num-prompts 100                     # optional \
    --num-steps 30                        # optional \
    --cfg-weight 4.0                      # optional \
    --skip-collection                     # optional, reuse W4A8 stats \
    --analysis-only                       # optional, no model loading
```

Note: `--skip-collection` and `--analysis-only` serve different purposes — `--skip-collection` still loads the model for analysis, while `--analysis-only` skips model loading entirely.

---

## Prompt Files

| File | Purpose |
|------|---------|
| `src/settings/coco_100_calibration_prompts.txt` | 100 calibration prompts (Phase 1/2/4 Hessian collection) |
| `src/settings/evaluation_set.txt` | Evaluation prompts for benchmark image generation |
| `src/settings/hyperparameters_4_metrics.txt` | Hyperparameter search reference |

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
| `per_token_rho_threshold` | 0.5 |
| `final_layer_bits` | 4 |
| `exclude_layers` | `["context_embedder"]` |

The output directory tag format is: `w{bits}a{a_bits}_{qkv}_a{alpha:.2f}_gs{gs}[_t{tau}][_static|_staticpc]`

Example: `w4a8_l2_a0.50_gs32_static`
