# time-aware-MLXSD-quantization

Time-aware quantization experiments on **Stable Diffusion 3 Medium (MMDiT)** using **DiffusionKit (MLX)**, inspired by the TaQ-DiT paper (*Time-aware Quantization for Diffusion Transformers*, arXiv:2411.14172).

This repo focuses on:

- **Phase 1 – Calibration data generation** for SD3/MMDiT.
- **Phase 2 – Diagnostic profiling of post-GELU FFN activations**, including TaQ-DiT-style visualizations.

Later phases (actual quantization schemes and ablations) can build directly on these components.

---

## Project structure (relevant parts)

- `src/calibration_sample_generation/`
  - `calibration_config.py` – Paper-aligned constants (100 steps, 256 trajectories, 25 selected timesteps, CFG=1.5, SD3 model ID).
  - `calibration_collector.py` – Euler sampler wrapper that collects `(x_t, t)` at each denoising step using `CFGDenoiser`.
  - `sample_cali_data.py` – Main entrypoint for Phase 1 calibration dataset generation.
  - `sample_prompts.txt` – 20 diverse text prompts used for calibration.

- `src/activation_diagnostics/`
  - `activation_tracer.py` – Monkey-patches SD3’s MMDiT to trace **post-GELU FFN** activations per layer and timestep.
  - `profile_postgelu.py` – Main entrypoint for Phase 2 diagnostic profiling (reads Phase 1 `.npz`, records stats/histograms).
  - `visualize_postgelu.py` – Generates TaQ-DiT-style plots and a summary table from the profiling output.

- `src/PHASE1.md` – Detailed Phase 1 design doc: how it maps the TaQ-DiT spec to SD3, plus usage.
- `src/PHASE2.md` – Detailed Phase 2 design doc: alignment/differences vs. TaQ-DiT, plus usage.

---

## Environment & dependencies

High level:

- macOS with Apple Silicon (MLX target).
- Python 3.10+ recommended.
- [DiffusionKit](https://github.com/argmaxinc/DiffusionKit) checked out under `DiffusionKit/` (already part of this repo layout).

Typical setup (from repo root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install MLX + basics
pip install mlx mlx-core matplotlib numpy tqdm

# Ensure DiffusionKit is importable (from this repo layout)
export PYTHONPATH="$PWD/DiffusionKit/python/src:$PYTHONPATH"
```

> Note: the Phase 1 and Phase 2 scripts try to add `DiffusionKit/python/src` to `sys.path` automatically when needed, but setting `PYTHONPATH` explicitly is convenient when working in notebooks or ad-hoc scripts.

---

## Phase 1 – Calibration data generation (SD3 / MMDiT)

Implements the TaQ-DiT calibration recipe for SD3, with SD3-specific adaptations:

- **Paper alignment**
  - 100 sampling steps per trajectory.
  - 256 trajectories total.
  - Uniform selection of 25 timesteps from the 100.
  - CFG scale = 1.5.
  - Result: **6,400** `(x_t, t)` calibration points.

- **SD3-specific adaptations**
  - Uses **Euler sampler** (ODE) instead of DDPM, which is more appropriate for SD3’s flow-matching formulation.
  - Text conditioning instead of class labels, using 20 diverse prompts.
  - Stores **token-level** and **pooled** text embeddings once per prompt, plus a `prompt_indices` array to keep `.npz` size reasonable.
  - Enforces **batch=1** for calibration sampling to respect `CFGDenoiser`’s internal batch-doubling (CFG).

### CLI – full calibration run

From repo root:

```bash
python -m src.calibration_sample_generation.sample_cali_data \
    --output DiT_cali_data.npz
```

This will:

- Load SD3 Medium via DiffusionKit.
- Run 256 Euler trajectories × 100 steps.
- Uniformly pick 25 timesteps.
- Flatten to 6,400 calibration points and shuffle.
- Save everything to `DiT_cali_data.npz` (`xs`, `ts`, `prompt_indices`, `cs`, `cs_pooled`, `prompts`, `cfg_scale`).

### CLI – quick dry run

```bash
python -m src.calibration_sample_generation.sample_cali_data \
    --num-fid-samples 8 \
    --num-sampling-steps 20 \
    --num-selected-steps 5 \
    --output dry_run_cali.npz
```

This produces a small calibration set suitable for functional testing.

For detailed shapes, rationale, and exact mapping to the paper, see `src/PHASE1.md`.

---

## Phase 2 – Activation diagnostics (post-GELU FFN)

Phase 2 verifies whether SD3’s **post-GELU FFN activations** exhibit the same issues TaQ-DiT observed for DiT’s post-GELU activations (asymmetry, temporal drift, channel outliers).

### Profiling (statistics + histograms)

```bash
python -m src.activation_diagnostics.profile_postgelu \
    --calibration-file DiT_cali_data.npz \
    --num-samples 512 \
    --output activation_stats_postgelu.npz
```

What this does:

- Loads the Phase 1 `.npz` calibration dataset.
- Samples a subset of calibration points (default: 512) uniformly without replacement.
- Groups points by prompt, batches all timesteps per prompt into a single `cache_modulation_params` call (to avoid adaLN weight offloading issues).
- Doubles `x_t` to batch size 2, matching CFG conditioning shape.
- For each `(layer, timestep)` pair, accumulates:
  - Per-channel: count, mean, std, min, max.
  - Global: 512-bin histogram over a fixed range [-8, 8].

The result is written to `activation_stats_postgelu.npz`.

### Visualization (TaQ-DiT-style plots)

```bash
python -m src.activation_diagnostics.visualize_postgelu \
    --stats-file activation_stats_postgelu.npz \
    --output-dir activation_plots/
```

This produces:

- `range_vs_timestep_all.png` (+ image/text-only variants) – activation range vs. timestep per layer.
- `histogram_<layer_id>.png` – per-timestep histograms for selected layers.
- `channel_ranges_<layer_id>.png` – per-channel dynamic ranges with top 2% outliers highlighted.
- `heatmap_mean.png`, `heatmap_std.png` – temporal drift heatmaps.
- `negative_fraction_heatmap.png` – fraction of activations < 0 per (layer, timestep).
- A console summary table with aggregated stats per layer.

You can also restrict visualizations to specific layers:

```bash
python -m src.activation_diagnostics.visualize_postgelu \
    --stats-file activation_stats_postgelu.npz \
    --output-dir activation_plots/ \
    --layers mm_00_img mm_12_img mm_23_img
```

For design details and how this maps to TaQ-DiT’s figures/tables, see `src/PHASE2.md`.

---

## Current status

- ✅ **Phase 1** (Calibration data generation) implemented and tested with both full and dry runs.
- ✅ **Phase 2** (Activation diagnostics + visualizations) implemented and tested on dry-run data.
- 🔜 **Next phases** (joint reconstruction, momentum-based shifting, reconstruction-driven migration, and evaluation on SD3) are not yet implemented in this repo, but the calibration + diagnostic infrastructure is ready for them.

