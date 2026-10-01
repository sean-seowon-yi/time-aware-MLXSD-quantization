# PQ4DiT: Time-Aware Polynomial Clipping for Post-Training Quantization of Stable Diffusion 3

**David Holt, A. Michael Tjhin, Seo Won Yi**

W4A8 post-training quantization of **Stable Diffusion 3 Medium** (MM-DiT backbone), implemented end-to-end in **MLX on Apple Silicon** (no CUDA anywhere in the pipeline). **Paper: [PDF](paper/PQ4DiT%20-%20Time-Aware%20Polynomial%20Clipping%20for%20Post-Training%20Quantization%20of%20Stable%20Diffusion%203.pdf)**

## Summary

Diffusion transformers are hard to quantize for two reasons: per-layer activation ranges drift across the denoising schedule, and a few salient channels dominate each layer's dynamic range. We study a lightweight, time-aware activation clipping scheme on top of PTQ4DiT:

1. **Polynomial clipping (Poly)** — each layer's A8 clipping range is a low-degree polynomial in the noise level σ, fit by least squares to per-timestep activation absmax (degree 0 when CV < 0.10, otherwise 2–4).
2. **PTQ4DiT transforms (P4D)** — Channel-wise Salience Balancing (CSB) and Spearman-guided Salience Calibration (SSC), applied independently to SD3's image and text sub-blocks *before* the polynomial fit.
3. **α-search** — a per-layer scalar multiplier α on the polynomial range, chosen by grid search to minimize per-layer reconstruction MSE on calibration data.

**Main finding (negative result):** the full pipeline P4D + Poly + α matches, but does not beat, the PTQ4DiT baseline within evaluation noise. Adding polynomial clipping to P4D degrades CMMD, and α-search only recovers that loss. Once CSB/SSC is applied, a static per-tensor 8-bit activation scale is already competitive on this architecture.

**Contributions:** to our knowledge, the first public W4A8 PTQ benchmark on SD3 MM-DiT (12 quantized configurations), plus a self-contained MLX/Apple Silicon reference implementation.

## Results

SD3-medium, W4A8 (per-group W4, group size 32; per-tensor A8), 512×512, 30 Euler steps, CFG 4.0, 512 held-out MS-COCO prompts ([`src/settings/evaluation_set.txt`](src/settings/evaluation_set.txt)). FID/CMMD are against ground-truth images; LPIPS pairs each image with its matched-seed FP16 image. CMMD is the primary metric.

| Pipeline | FID ↓ | CMMD ↓ | LPIPS ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|
| FP16 (reference) | 105.87 | 0.507 | – | 0.269 (0.032) |
| *Static-calibration baselines* | | | | |
| Vanilla W4A8 | 250.85 | 2.696 | 0.667 (0.075) | 0.160 (0.034) |
| GPTQ | 106.53 | 0.684 | 0.406 (0.105) | 0.265 (0.032) |
| *PTQ4DiT baseline* | | | | |
| P4D | **103.80** | **0.553** | 0.384 (0.114) | 0.266 (0.033) |
| *Polynomial clipping without P4D* | | | | |
| Poly + GPTQ | 357.92 | 3.457 | 0.729 (0.056) | 0.119 (0.024) |
| Poly + AdaRound | 142.64 | 1.498 | 0.558 (0.077) | 0.238 (0.036) |
| Poly + MW AdaRound | 145.57 | 1.436 | 0.565 (0.083) | 0.238 (0.036) |
| Poly + α | 109.75 | 0.858 | 0.476 (0.097) | 0.259 (0.033) |
| Poly + α + GPTQ | 104.83 | 0.660 | 0.406 (0.109) | 0.266 (0.033) |
| *Polynomial clipping with P4D (ours)* | | | | |
| P4D + Poly | 106.32 | 0.662 | 0.394 (0.109) | 0.265 (0.032) |
| P4D + Poly + AdaRound | 107.12 | 0.697 | 0.440 (0.095) | **0.268** (0.032) |
| P4D + Poly + MW AdaRound | 107.90 | 0.730 | 0.466 (0.095) | 0.266 (0.031) |
| **P4D + Poly + α (ours)** | 104.63 | 0.556 | **0.383** (0.116) | **0.268** (0.031) |

Naming: RTN is the implicit default for both the W4 rounder and the A8 quantizer. "MW AdaRound" reweights AdaRound's per-timestep loss by the mean absolute derivative of the polynomial schedule.

Raw metric outputs plus the quantization config, polynomial schedule, and α-search results for the P4D, P4D + Poly, and P4D + Poly + α rows are in [`paper/benchmarks/`](paper/benchmarks/). Qualitative comparisons and method figures are in the [paper](paper/PQ4DiT%20-%20Time-Aware%20Polynomial%20Clipping%20for%20Post-Training%20Quantization%20of%20Stable%20Diffusion%203.pdf).

## Where each configuration lives

| Configurations | Code |
|---|---|
| P4D, P4D + Poly, P4D + Poly + α | `src/phase1/` → `src/phase2/` → `src/phase3/` → `src/phase4_1/`, orchestrated by `src/run_poly_alpha_pipeline.py` (CLI reference: [`src/settings/commands.md`](src/settings/commands.md)) |
| Poly + AdaRound, Poly + MW AdaRound | `src/generate_poly_schedule.py`, `src/cache_adaround_data.py`, `src/adaround_optimize.py --poly-schedule ... [--derivative-weighted --deriv-agg mean]` (see [`docs/POLYNOMIAL_CLIPPING_EXPLAINER.md`](docs/POLYNOMIAL_CLIPPING_EXPLAINER.md), [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md)) |
| P4D + Poly + AdaRound, P4D + Poly + MW AdaRound | branch `ptq4dit-polynomial-adaround` (`src/phase4/`) |
| GPTQ, Poly + GPTQ, Poly + α, Poly + α + GPTQ | branch `gptq` (`src/gptq/`) |
| FID / CMMD / LPIPS / CLIP evaluation | `src/benchmark/gt_comparison_pipeline.py` |

## Setup

macOS on Apple Silicon (tested on M1, M4, M5), Python 3.10+.

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/DiffusionKit/python/src:$PYTHONPATH"
```

## Reproducing P4D + Poly + α

Calibration uses 100 MS-COCO prompt/seed pairs ([`src/settings/coco_100_calibration_prompts.txt`](src/settings/coco_100_calibration_prompts.txt)); CSB β = 0.5, SSC τ = 1.0. The settings below are the ones recorded in [`paper/benchmarks/p4d_poly_alpha/`](paper/benchmarks/p4d_poly_alpha/).

**1. Calibrate, quantize (P4D), fit the polynomial schedule, and run α-search** (α-search on 50 prompts took ~4 h on Apple Silicon):

```bash
python -m src.run_poly_alpha_pipeline --prompts-file src/settings/coco_100_calibration_prompts.txt --output-dir quantized --diagnostics-dir diagnostics --qkv-method l2 --alpha 0.5 --group-size 32 --bits 4 --static-mode ssc_weighted --static-granularity per_tensor --ssc-tau 1.0 --max-degree 4 --alpha-num-prompts 50
```

This writes `quantized/w4a8_l2_a0.50_gs32_static/`, containing the P4D checkpoint and a `poly_schedule.json` with the per-layer α merged in. The schedule before α-search is kept as `poly_schedule.json.pre_alpha_search.bak` (the P4D + Poly row).

**2. Benchmark against ground truth and FP16:**

```bash
python -m src.benchmark.gt_comparison_pipeline --ground-truth-dir results/gt/images --fp16-images-dir results/fp16/images --quantized-dir quantized/w4a8_l2_a0.50_gs32_static --output-dir benchmark_results/p4d_poly_alpha --config w4a8_poly --poly-schedule quantized/w4a8_l2_a0.50_gs32_static/poly_schedule.json --group-size 32 --prompt-file src/settings/evaluation_set.txt
```

For the P4D baseline, use `--config w4a8_static` and omit `--poly-schedule`.

## Repository layout

```
paper/                        Paper (PDF) and benchmark outputs for the P4D rows
src/
  phase1/ … phase4_1/         P4D → Poly → α-search pipeline (see docs/pipeline/)
  run_poly_alpha_pipeline.py  End-to-end driver for the pipeline above
  benchmark/                  FID / CMMD / LPIPS / CLIP evaluation
  settings/                   Calibration + evaluation prompts, CLI reference (commands.md)
  *.py                        Polynomial schedule, AdaRound / MW AdaRound, SmoothQuant, benchmarking
  calibration_sample_generation/, activation_diagnostics/
                              Early TaQ-DiT-style calibration and post-GELU profiling
data/
  prompts/                    MS-COCO prompt sets used by the src/*.py scripts
  schedules/                  Fitted polynomial / LUT clipping schedules
  generalization_results/     Polynomial-schedule generalization study
docs/
  pipeline/                   Design docs for each phase (PHASE1–PHASE4_1) and Phase 1 findings
  plans/, slides/, figures/   Design plans, explainer slides, and their figures
  RESEARCH_LOG.md, PLAN.md, POLYNOMIAL_CLIPPING_EXPLAINER.md
scripts/                      Small standalone utilities and plotting scripts
tests/                        pytest suite (tests/integration needs local calibration data)
DiffusionKit/                 Vendored DiffusionKit (MLX SD3)
```

## Tests

```bash
pytest
```

`tests/integration/` is skipped by default because it needs local calibration data and AdaRound weights that are not in the repo. Run it explicitly with `pytest tests/integration`.

## Citation

```bibtex
@misc{holt2026pq4dit,
  title  = {PQ4DiT: Time-Aware Polynomial Clipping for Post-Training Quantization of Stable Diffusion 3},
  author = {Holt, David and Tjhin, A. Michael and Yi, Seo Won},
  year   = {2026}
}
```
