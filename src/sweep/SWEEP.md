# Hyperparameter Sweep Pipeline

Systematic comparison of W4A8 quantization configurations for SD3 Medium MMDiT.
The sweep varies calibration hyperparameters — **`qkv_method`**, **`alpha`**,
**`group_size`**, and optionally **`ssc_tau`** — in a two-phase design: a broad
baseline grid (gs64) followed by a targeted refinement (gs32 + tau probes).
All configs use **per-token A8** quantization for layers with mean Spearman ρ > 0.5
(approximately 17 layers, automatically selected).

All four steps are independent scripts, orchestrated by a unified CLI that runs
a **staged, randomized** evaluation pipeline.

---

## Overview

```
Step 1: Quantize         Step 2: Inference        Step 3: Metrics         Step 4: Summarize
┌───────────────┐       ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ run_quantize  │──────▶│ run_inference  │──────▶│  run_metrics  │──────▶│  summarize    │
│ _sweep.py     │       │ _sweep.py     │       │    .py        │       │    .py        │
└───────────────┘       └───────────────┘       └───────────────┘       └───────────────┘
diagnostics/ ──▶ quantized/<tag>/  ──▶ results/<tag>/  ──▶ metrics/<tag>/  ──▶ stdout
  (Phase 1         (calibration +       (generated            (per_image.json     (ranked
   reused)          quantized model)      PNG images)           aggregate.json)     table)
```

The staged CLI (`python -m src.sweep.cli`) automates the full flow:
quantize all configs once, then progressively evaluate with **randomized
prompt subsets** of increasing size, eliminating weak candidates at each stage.

---

## Sweep Matrix

The grid sweeps over **27 configurations** in two phases:

- **Phase 1 (indices 0–11)**: Broad baseline at `group_size=64` — 9 configs
  (3 methods × 3 alpha) + 3 with SSC temperature τ=0.5.
- **Phase 2 (indices 12–26)**: Targeted refinement based on Phase 1 winners —
  `group_size=32` alpha bracket + τ=0.8 probes at both group sizes.

### Phase 1 — Baseline (gs64)

| Index | Tag | `qkv_method` | `alpha` | `gs` | `τ` |
|-------|-----|-------------|---------|------|-----|
| 0 | `w4a8_max_a0.30_gs64` | `max` | 0.30 | 64 | 1.0 |
| 1 | `w4a8_max_a0.50_gs64` | `max` | 0.50 | 64 | 1.0 |
| 2 | `w4a8_max_a0.70_gs64` | `max` | 0.70 | 64 | 1.0 |
| 3 | `w4a8_geomean_a0.30_gs64` | `geomean` | 0.30 | 64 | 1.0 |
| 4 | `w4a8_geomean_a0.50_gs64` | `geomean` | 0.50 | 64 | 1.0 |
| 5 | `w4a8_geomean_a0.70_gs64` | `geomean` | 0.70 | 64 | 1.0 |
| 6 | `w4a8_l2_a0.30_gs64` | `l2` | 0.30 | 64 | 1.0 |
| 7 | `w4a8_l2_a0.50_gs64` | `l2` | 0.50 | 64 | 1.0 |
| 8 | `w4a8_l2_a0.70_gs64` | `l2` | 0.70 | 64 | 1.0 |
| 9 | `w4a8_max_a0.50_gs64_t0.5` | `max` | 0.50 | 64 | 0.5 |
| 10 | `w4a8_geomean_a0.50_gs64_t0.5` | `geomean` | 0.50 | 64 | 0.5 |
| 11 | `w4a8_l2_a0.50_gs64_t0.5` | `l2` | 0.50 | 64 | 0.5 |

### Phase 2 — Refinement (gs32 + tau probes)

| Index | Tag | `qkv_method` | `alpha` | `gs` | `τ` | Purpose |
|-------|-----|-------------|---------|------|-----|---------|
| 12 | `w4a8_max_a0.40_gs32` | `max` | 0.40 | 32 | 1.0 | gs32 alpha bracket |
| 13 | `w4a8_max_a0.50_gs32` | `max` | 0.50 | 32 | 1.0 | gs32 alpha bracket |
| 14 | `w4a8_max_a0.60_gs32` | `max` | 0.60 | 32 | 1.0 | gs32 alpha bracket |
| 15 | `w4a8_geomean_a0.40_gs32` | `geomean` | 0.40 | 32 | 1.0 | gs32 alpha bracket |
| 16 | `w4a8_geomean_a0.50_gs32` | `geomean` | 0.50 | 32 | 1.0 | gs32 alpha bracket |
| 17 | `w4a8_geomean_a0.60_gs32` | `geomean` | 0.60 | 32 | 1.0 | gs32 alpha bracket |
| 18 | `w4a8_l2_a0.40_gs32` | `l2` | 0.40 | 32 | 1.0 | gs32 alpha bracket |
| 19 | `w4a8_l2_a0.50_gs32` | `l2` | 0.50 | 32 | 1.0 | gs32 alpha bracket |
| 20 | `w4a8_l2_a0.60_gs32` | `l2` | 0.60 | 32 | 1.0 | gs32 alpha bracket |
| 21 | `w4a8_max_a0.50_gs32_t0.8` | `max` | 0.50 | 32 | 0.8 | gs32 + gentle τ |
| 22 | `w4a8_geomean_a0.50_gs32_t0.8` | `geomean` | 0.50 | 32 | 0.8 | gs32 + gentle τ |
| 23 | `w4a8_l2_a0.50_gs32_t0.8` | `l2` | 0.50 | 32 | 0.8 | gs32 + gentle τ |
| 24 | `w4a8_max_a0.50_gs64_t0.8` | `max` | 0.50 | 64 | 0.8 | isolate τ=0.8 effect |
| 25 | `w4a8_geomean_a0.50_gs64_t0.8` | `geomean` | 0.50 | 64 | 0.8 | isolate τ=0.8 effect |
| 26 | `w4a8_l2_a0.50_gs64_t0.8` | `l2` | 0.50 | 64 | 0.8 | isolate τ=0.8 effect |

**Fixed parameters** (not swept): `bits=4`, `a_bits=8`, `per_token_rho_threshold=0.5`.

### What the hyperparameters control

- **`qkv_method`**: How weight salience from Q, K, V projections is merged into
  a shared balancing vector (since SD3's MMDiT shares adaLN modulation across Q/K/V).
  The same merged salience is used for both the SSC rho trajectory and the balancing
  formula, ensuring internal consistency.
  - `"max"` — element-wise maximum across all three projections (conservative;
    protects any projection's high-salience channels).
  - `"geomean"` — geometric mean (balanced; respects all projections equally).
  - `"l2"` — RMS (root mean square); a middle ground between max and geomean.
    By the power-mean inequality: geomean ≤ l2 ≤ max.

- **`alpha`**: The exponent in the balancing formula `b_j = (s_act_j / s_wt_j)^alpha`.
  Higher alpha means more aggressive channel re-balancing between activations and
  weights. `alpha=0` disables balancing entirely; `alpha=1.0` fully equalises the
  salience ratio.

- **`group_size`**: Number of weight elements sharing one quantization scale factor.
  Smaller groups adapt better to local outliers within each weight row, reducing
  quantization error at the cost of slightly more metadata. `64` is the baseline;
  `32` is tested in Phase 2 as a low-cost quality improvement.

- **`ssc_tau`** (SSC temperature): Controls the sharpness of time-aware calibration
  weighting. `τ=1.0` is the original PTQ4DiT formula. `τ<1` sharpens the softmax,
  giving more weight to timesteps where activation-weight complementarity is strongest.
  Phase 1 findings show that with τ=1.0, SSC is near-uniform for ~87% of layers;
  lower τ amplifies the small ρ differences for the ~13% of layers where SSC matters.
  Phase 2 tests `τ=0.8` (gentle) in addition to the original `τ=0.5` (aggressive).

- **Per-token A8** (automatic, not swept): Layers with mean Spearman ρ > 0.5 use
  per-token (one scale per sequence position) instead of per-tensor activation
  quantization. This is a fixed policy applied to all configs, targeting the ~17
  high-ρ layers where CSB incompletely equalises dynamic range.

The sweep matrix is defined in `sweep_config.py` and can be modified there to add
new configurations.

---

## Staged Evaluation

The default CLI runs a **three-stage progressive evaluation** with randomized
prompt selection. Defaults match `src/sweep/cli.py`: `--stage-prompts 16,32,64`,
`--stage-topk 4,2` (promote 4 configs after stage 1, then 2 after stage 2).

Example with stricter promotion (`--stage-topk 6,3`):

| Stage | Images | Configs evaluated | New generations (cold) | Top-k kept |
|-------|--------|-------------------|------------------------|------------|
| 1 | 16 (random) | 27 (all) | 27 × 16 = 432 | top 6 |
| 2 | 32 (superset of stage 1) | 6 | 6 × 16 = 96 | top 3 |
| 3 | 64 (superset of stage 2) | 3 | 3 × 32 = 96 | final |
| **Total** | | | **~624** (cold) | |

**Incremental run** (Phase 1 configs 0–11 already have results): only the 15
new configs (12–26) need quantize + inference + metrics. Existing images and
cached metric scores are reused automatically — see [Reusing Previous Results](#reusing-previous-results).

Stricter promotion example:
```bash
python -m src.sweep.cli --stage-topk 6,3
```

### Randomized prompt selection

Instead of always evaluating the first N prompts from the evaluation set, each
stage evaluates a **random but reproducible** subset:

1. A fixed-seed permutation of all prompt indices is generated once (`--sweep-seed`,
   default: 2026).
2. Stage 1 takes the first 16 indices from the permutation, stage 2 the first 32,
   stage 3 the first 64.
3. Each stage is a **strict superset** of the previous — so cached images and
   metrics carry over, and only newly added indices require generation.
4. FP16 baseline images (all 256, pre-generated) are reused as-is.
5. The permutation and per-stage index files are saved under
   `metrics/staged_runs/<run_id>/` for reproducibility.

### Image and metric caching

- **Image reuse**: `run_inference.py` skips generation when `{idx:03d}.png`
  already exists on disk. Stage 2 reuses all images from stage 1.
- **Metric caching**: `run_metrics.py` stores per-image scores in
  `per_image.json` with a `scorer_config.json` checksum. Scores computed in
  earlier stages are reused if the scorer config hasn't changed.
- **Cache preservation**: Entries from wider stages are not dropped when a
  narrower stage re-runs.

---

## Reusing Previous Results

The pipeline is fully incremental at every step:

- **Quantize**: Skips any config where `quantize_config.json` already exists.
- **Inference**: Skips any image where `{idx:03d}.png` already exists on disk.
- **Metrics**: Reuses cached per-image scores from `per_image.json` when the
  scorer configuration (LPIPS backbone, CLIP model) hasn't changed.

This means **running the full 27-config CLI after Phase 1 (12 configs) has
already completed will only quantize, generate images, and score the 15 new
configs**. The original 12 configs participate in ranking at zero cost.

```bash
# Incremental: reuses all Phase 1 results, only runs Phase 2 configs
python -m src.sweep.cli --stage-topk 6,3

# Only run the new configs explicitly (skip old ones entirely)
python -m src.sweep.cli --configs 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 --stage-topk 6,3
```

---

## Quick Start

```bash
# Full staged pipeline (quantize → 3-stage evaluation → summarize)
python -m src.sweep.cli --stage-topk 6,3

# Dry run (prints commands without executing)
python -m src.sweep.cli --stage-topk 6,3 --dry-run

# Custom stage sizes and promotion counts
python -m src.sweep.cli --stage-prompts 8,24,48 --stage-topk 6,3

# Skip staged selection, run all steps sequentially for all configs
python -m src.sweep.cli --disable-staged-selection

# Quick profile (32 images, no staging)
python -m src.sweep.cli --profile quick
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--stage-prompts` | `16,32,64` | Comma-separated image counts per stage |
| `--stage-topk` | `4,2` | Comma-separated: how many configs to keep after each stage (length = stages − 1). Use e.g. `6,3` for stricter filtering. |
| `--sweep-seed` | `2026` | Seed for randomized prompt selection |
| `--configs` | all | Indices into SWEEP_MATRIX to evaluate |
| `--profile` | `full` | `full` (staged) or `quick` (32 images, flat) |
| `--disable-staged-selection` | off | Run steps directly without staging |
| `--dry-run` | off | Print commands without executing |
| `--fp16-dir` | `results/fp16` | Override FP16 reference image directory |
| `--clip-model` | `openai/clip-vit-base-patch32` | CLIP model for scoring |
| `--lpips-net` | `alex` | LPIPS backbone (`alex`, `vgg`, `squeeze`) |

---

## Directory Layout

Every script uses a consistent tag-based naming convention.

```
project_root/
├── diagnostics/                         ← Phase 1 data (shared, not duplicated)
│   ├── activation_stats/
│   ├── weight_stats.npz
│   ├── adaln_stats.npz
│   └── config.json
│
├── quantized/                           ← Step 1 output
│   ├── w4a8_max_a0.30_gs64/
│   │   ├── mmdit_quantized.safetensors
│   │   ├── quantize_config.json
│   │   ├── calibration.npz
│   │   └── calibration_meta.json
│   ├── w4a8_max_a0.50_gs64/
│   ├── w4a8_l2_a0.50_gs32/          ← Phase 2 refinement
│   ├── w4a8_l2_a0.50_gs64_t0.8/     ← tau probe
│   └── ...  (27 subdirectories total)
│
├── results/                             ← Step 2 output + FP16 reference
│   ├── fp16/                            ← FP16 baseline (pre-generated)
│   │   ├── 000.png ... 255.png
│   ├── w4a8_max_a0.30_gs64/
│   │   ├── 002.png, 017.png, ...       ← sparse: only eval indices
│   │   └── run_meta.json
│   └── ...
│
├── metrics/                             ← Step 3 output
│   ├── w4a8_max_a0.30_gs64/
│   │   ├── per_image.json
│   │   ├── aggregate.json
│   │   └── scorer_config.json
│   ├── ...
│   ├── summary.json                     ← combined aggregates
│   └── staged_runs/                     ← staged pipeline metadata
│       ├── 20260321T120000Z/
│       │   ├── permutation.json
│       │   ├── stage_1_indices.json
│       │   ├── stage_2_indices.json
│       │   └── stage_3_indices.json
│       ├── 20260321T120000Z.json        ← run metadata + rankings
│       └── latest.json                  ← symlink to most recent run
│
└── src/sweep/
    ├── __init__.py
    ├── sweep_config.py                  ← matrix definition & path helpers
    ├── run_quantize_sweep.py            ← Step 1
    ├── run_inference_sweep.py           ← Step 2
    ├── run_metrics.py                   ← Step 3
    ├── summarize.py                     ← Step 4
    ├── cli.py                           ← unified orchestrator
    └── SWEEP.md                         ← this file
```

---

## Step-by-Step Guide (Manual)

Each step can also be run independently for debugging or partial re-runs.

### Prerequisites

1. **Phase 1 diagnostic data** must exist in `diagnostics/`.
2. **FP16 reference images** must exist in `results/fp16/` (all 256).
3. Install sweep-specific dependencies:
   ```bash
   pip install lpips torchvision
   ```

### Step 1 — Quantize All Configs

```bash
python -m src.sweep.run_quantize_sweep
python -m src.sweep.run_quantize_sweep --configs 0 3      # subset
python -m src.sweep.run_quantize_sweep --dry-run           # preview
```

Skips configs where `quantize_config.json` already exists.
Runtime: ~5-10 min per config.

### Step 2 — Generate W4A8 Images

```bash
python -m src.sweep.run_inference_sweep
python -m src.sweep.run_inference_sweep --num-prompts 32
python -m src.sweep.run_inference_sweep --eval-indices-file indices.json
```

Supports `--eval-indices-file` (JSON list of prompt indices) for randomized
selection, or `--num-prompts` for contiguous first-N.
Runtime: ~30-90s per image.

### Step 3 — Compute Metrics

```bash
python -m src.sweep.run_metrics --all
python -m src.sweep.run_metrics --all --num-images 32
python -m src.sweep.run_metrics --all --eval-indices-file indices.json
```

Loads LPIPS (AlexNet) and CLIP (ViT-B/32) once, then processes all configs.
Incrementally caches per-image scores.

### Step 4 — Summarize and Rank

```bash
python -m src.sweep.summarize
python -m src.sweep.summarize --top-k-worst 10
```

Prints ranked table, LPIPS distributions, worst samples, and recommendation.

---

## Evaluation Metrics

### LPIPS (Learned Perceptual Image Patch Similarity)

- **Reference**: Zhang et al., "The Unreasonable Effectiveness of Deep Features
  as a Perceptual Metric" (CVPR 2018).
- **Backbone**: AlexNet (default). Alternatives: VGG, SqueezeNet.
- **Interpretation**: Perceptual distance between FP16 and W4A8. **Lower is better.**
  0.0 = identical; above ~0.3 = noticeable degradation.

### CLIPScore

- **Model**: `openai/clip-vit-base-patch32` (default).
- **Interpretation**: Text-image alignment. **Higher is better.** Typical range: 25-35.
- **Formula**: `CLIPScore = 100 × cos(image_embed, text_embed)`.
- **Reported values**: `clip_fp16` (baseline), `clip_w4a8` (candidate),
  `clip_delta` (w4a8 - fp16; near-zero = no regression).

### Ranking criteria

Primary: LPIPS mean (lower is better). Tie-breakers:
1. LPIPS p90 (lower)
2. |clip_delta| (closer to zero)
3. CLIPScore w4a8 (higher)

---

## Modifying the Sweep

### Adding configurations

Edit `SWEEP_MATRIX` in `sweep_config.py`:

```python
SWEEP_MATRIX: list[dict] = [
    {"qkv_method": "max",     "alpha": 0.3},
    # ...existing entries...
    {"qkv_method": "l2",      "alpha": 0.4},  # new
]
```

### Sweeping additional hyperparameters

Add the key to the sweep entry dict.  Any key not specified falls back to the
default from `PHASE2_CONFIG`:

```python
{"qkv_method": "max", "alpha": 0.5, "group_size": 128},
{"qkv_method": "l2",  "alpha": 0.5, "ssc_tau": 0.3},
```

The `config_tag` function encodes `group_size` and `ssc_tau` in the tag
(e.g., `w4a8_max_a0.50_gs128`, `w4a8_l2_a0.50_gs64_t0.3`).

### Changing the shuffle seed

```bash
python -m src.sweep.cli --sweep-seed 12345
```

Different seeds produce different random prompt selections. The seed is recorded
in the staged run metadata for reproducibility.

---

## Dependencies

Core project dependencies (already installed):
- `torch`, `transformers`, `numpy`, `pillow`, `safetensors`, `mlx`

Additional for sweep metrics (Step 3):
- `lpips`, `torchvision`

```bash
pip install lpips torchvision
```
