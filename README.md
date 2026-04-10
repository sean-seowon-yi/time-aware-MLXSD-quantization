# time-aware-MLXSD-quantization

Time-aware post-training quantization for **Stable Diffusion 3 Medium (MMDiT)** on **Apple Silicon**, using **DiffusionKit (MLX)**. The implementation follows **PTQ4DiT**-style **Channel-wise Salience Balancing (CSB)** and **Spearman-based Sigma Calibration (SSC)** for **W4A8** (4-bit weights, 8-bit activations), with optional **static** activation scales derived from Phase 1 calibration data.

---

## What this repository contains

| Area | Role |
|------|------|
| **`src/phase1/`** | Diagnostic collection: hooks on target `nn.Linear` layers, activation stats per σ-step, weight salience. Outputs under `diagnostics/`. |
| **`src/phase2/`** | Calibration (`calibrate.py`), CSB (`balance.py`), dynamic W4A8 (`quantize.py`), **static** W4A8 (`quantize_static.py`), end-to-end CLI (`run_e2e.py`), inference (`run_inference.py`), optional post-quant diagnostics (`run_diagnose.py`), and **visualization scripts** (`plot_post_csb.py`, `plot_weight_profile.py`, `plot_quantized_weight.py`, `plot_mse_vs_block.py`). |
| **`src/benchmark/`** | `gt_comparison_pipeline.py` — GT comparison: generates W4A8 images if needed, then computes FID, CMMD, CLIP scores, and LPIPS against ground truth and FP16 baselines. |
| **`src/settings/`** | `coco_100_calibration_prompts.txt` (100 tab-separated seed/prompt pairs for Phase 1) and `evaluation_set.txt` (larger set for benchmarks/sweeps). |
| **`DiffusionKit/`** | Vendored DiffusionKit Python sources (see `DiffusionKit/README.md`). |

**Documentation** (in `src/`):

- `PHASE1.md` -- design and architecture notes for Phase 1 diagnostics.
- `PHASE2.md` -- CSB/SSC/W4A8 pipeline, data flow, CLI reference.
- `phase1_findings.md` -- summarized empirical findings from collected diagnostics.

There is **no** `src/calibration_sample_generation/` or `src/activation_diagnostics/` tree; those paths referred to an older scaffold and are **not** present in this codebase.

---

## Environment and dependencies

- **Platform:** macOS with Apple Silicon (MLX).
- **Python:** 3.10+ (use a conda env with MLX + DiffusionKit stack, or a venv).

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Key packages include: `mlx`, `torch`, `safetensors`, `transformers`, `pillow`, `numpy`, `scipy`, `matplotlib`, and (for benchmarks) `torch-fidelity`, `open-clip-torch`, `lpips`, `psutil`.

Ensure DiffusionKit is importable (this repo expects `DiffusionKit/python/src` on `PYTHONPATH`, or the Phase 2 scripts add it when run as modules):

```bash
export PYTHONPATH="$PWD/DiffusionKit/python/src:$PYTHONPATH"
```

---

## Phase 1 — Diagnostic collection

Collects per-layer activation trajectories (per-channel max over tokens at each denoising step) and weight salience, using **100** COCO-caption–style seed/prompt pairs by default (`src/settings/coco_100_calibration_prompts.txt`), **30** Euler steps, **CFG 4.0**, latent **64×64** (512×512 pixels).

**Entry points**

- **Collection** (writes `diagnostics/activation_stats/`, `weight_stats.npz`, `config.json`, adaLN stats; no plots):  
  `python -m src.phase1.run_collection`  
  Optional: `--pilot` (2 prompts), `--num-prompts N`.

- **Analysis + plots** (requires existing `diagnostics/`):  
  `python -m src.phase1.run_analysis`

See `src/PHASE1.md` and `src/phase1_findings.md` for methodology and results.

---

## Phase 2 — W4A8 quantization (PTQ4DiT-style)

**End-to-end (collection → calibration → CSB → quantize → save):**

```bash
python -m src.phase2.run_e2e --output-dir quantized/
```

**Reuse existing diagnostics** (skip Phase 1 collection):

```bash
python -m src.phase2.run_e2e --output-dir quantized/ --skip-collection
```

**Static activation quantization** (scales from calibration; separate from dynamic fake-quant per forward):

```bash
python -m src.phase2.run_e2e --output-dir quantized/ --skip-collection \
  --act-quant static --static-mode ssc_weighted --static-granularity per_tensor
```

**Standalone quantize** (no static path; dynamic W4A8 only): `python -m src.phase2.run_quantize` — see `--calibrate-only` and `--from-calibration`.

**Inference**

```bash
python -m src.phase2.run_inference --mode fp16 --prompts-file src/settings/evaluation_set.txt --output-dir results/
python -m src.phase2.run_inference --mode w4a8 --quantized-dir quantized/<tag>/ --prompts-file src/settings/evaluation_set.txt --output-dir results/
```

Images are written under **`results/fp16/`** (fp16 mode) or **`results/<config_tag>/`** (w4a8 mode, where `<config_tag>` comes from `quantize_config.json` via `config_tag_from_meta`, e.g. `w4a8_l2_a0.50_gs32`). Filenames use **3-digit** indices (`000.png`, `001.png`, …) aligned with prompt order and per-line seeds in tab-separated prompt files.

**Benchmark (GT comparison)**

```bash
python -m src.benchmark.gt_comparison_pipeline \
  --ground-truth-dir /path/to/gt_images \
  --fp16-images-dir benchmark_results/fp16_p2/images \
  --quantized-dir quantized/<tag>/ \
  --output-dir benchmark_results/<tag>_gt_eval \
  --config w4a8_poly
```

Computes FID, CMMD, CLIP image-text scores, and LPIPS. Generates W4A8 images automatically if not already present. See `gt_comparison_pipeline.py` docstring.

**Visualization helpers** (several require **`--calibration-dir`** on a quantized output tree with `calibration.npz`):

- `python -m src.phase2.plot_post_csb --calibration-dir quantized/<tag>/ [--diagnostics-dir diagnostics]` — activation absmax vs σ (pre/post CSB).
- `python -m src.phase2.plot_weight_profile --calibration-dir quantized/<tag>/` — per-channel weight absmax (pre/post CSB).
- `python -m src.phase2.plot_quantized_weight --quantized-dir quantized/<tag>/` — per-group FP vs dequantized W4 (**requires MLX**).
- `python -m src.phase2.plot_mse_vs_block --quantized-dir quantized/<tag>/` — analytical W4 / dynamic-A8 MSE vs block (**requires MLX**).

Full detail, theory, and artifact tables: **`src/PHASE2.md`** (all-caps `PHASE2`; on case-sensitive systems this is the only path -- not `Phase2.md`).

---

## Current status

- **Phase 1:** Implemented (`src/phase1/`); outputs in `diagnostics/` when collection is run.
- **Phase 2:** W4A8 with CSB + SSC, dynamic and static activation quantization, inference, benchmark integration, sweep, and plotting utilities.
- **Default Phase 2 hyperparameters** are in `src/phase2/config.py` (`PHASE2_CONFIG`: e.g. `alpha=0.5`, `group_size=64`, `qkv_method=max`, `ssc_tau=1.0`, `per_token_rho_threshold=0.5`).

---

## References

- **PTQ4DiT** (methodology): *Post-training Quantization for Diffusion Transformers*.
- This repo implements **PTQ4DiT-aligned CSB/SSC**, polynomial σ-aware activation clipping, and alpha search on SD3/MMDiT via DiffusionKit.
