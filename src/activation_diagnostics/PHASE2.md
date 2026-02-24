# Phase 2: Diagnostic Profiling of Post-GELU FFN Activations

**Paper**: TaQ-DiT — *Time-aware Quantization for Diffusion Transformers* (arXiv:2411.14172v2)  
**Target model**: Stable Diffusion 3 Medium (MMDiT) via [DiffusionKit](https://github.com/argmaxinc/DiffusionKit) on Apple MLX

---

## Purpose

Phase 2 answers a critical question before applying the paper's quantization techniques:

> **Do the post-activation (post-GELU) FFN outputs in SD3's MMDiT exhibit the same pathological properties that TaQ-DiT identifies in DiT?**

Specifically, we measure:
1. **Asymmetry** — Is the activation distribution heavily skewed toward zero/negative values?
2. **Temporal shift** — Do activation mean and variance change significantly across denoising timesteps?
3. **Channel outliers** — Are there specific channels with disproportionately large dynamic ranges?

These diagnostics directly determine whether the paper's proposed Time-Grouping Quantization (TGQ) and Saliency-Weighted Quantization (SWQ) techniques are applicable to SD3.

---

## Paper specification (Section III-B, Figures 2–3)

The paper demonstrates on DiT-XL/2 (ImageNet class-conditional):

- **Post-SwiGLU activations show extreme asymmetry**: ~79.7% of values are concentrated in a single small bin near zero (Fig 3a).
- **Heavy-tailed positive channel**: A few channels have values spanning thousands while most are near zero.
- **Temporal drift**: Activation distributions shift substantially between early timesteps (high noise) and late timesteps (low noise), meaning a single static quantization grid is suboptimal.
- **Channel-wise outliers**: The top ~2% of channels by dynamic range dominate the quantization error.

| Diagnostic | Paper's DiT finding |
|-----------|-------------------|
| Negative fraction | ~79.7% of post-SwiGLU values < 0 |
| Activation range | Varies 10–1000× across timesteps |
| Channel outliers | Top 2% dominate quantization error |
| Temporal shift | Mean and variance drift significantly |

---

## Our implementation

### Alignment with the paper

| Diagnostic | Paper | Ours | Matches? |
|-----------|-------|------|----------|
| Activation location | Post-SwiGLU in FFN | Post-GELU in FFN | See note below |
| Per-layer statistics | mean, std, min, max | mean, std, min, max per channel | Yes |
| Histograms | Distribution over activation values | 512-bin histogram in [-8, 8] | Yes |
| Per-timestep analysis | Statistics at each denoising step | Keyed by exact timestep value | Yes |
| Channel-wise ranges | max − min per channel | Computed and visualized | Yes |
| Negative fraction | Reported as percentage | Computed from histogram | Yes |
| Outlier identification | Top 2% channels by range | Highlighted in red in plots | Yes |

**Note on GELU vs. SwiGLU**: The paper targets DiT which uses SwiGLU activations. DiffusionKit's SD3 MMDiT uses `nn.GELU()` in its FFN blocks (verified by inspecting `DiffusionKit/python/src/diffusionkit/mlx/mmdit.py`). The diagnostic approach is identical — we capture the output of the activation function between `fc1` and `fc2` — but we refer to it as "post-GELU" rather than "post-SwiGLU" throughout.

### Deliberate adaptations for SD3 / MLX

| Aspect | Paper (DiT / PyTorch) | Our adaptation (SD3 / MLX) | Reasoning |
|--------|----------------------|---------------------------|-----------|
| **Hook mechanism** | PyTorch `register_forward_hook` | Monkey-patching `TransformerBlock.pre_sdpa`, `post_sdpa`, and `FFN.__call__` at the class level | MLX `nn.Module` does not support forward hooks. Monkey-patching is non-invasive (restores originals on cleanup) and avoids modifying DiffusionKit source. |
| **Dual stream** | Single transformer block type | Separate image and text FFN blocks per layer | SD3's MMDiT has `image_transformer_block` and `text_transformer_block` in each multimodal layer. We trace both, assigning IDs like `mm_05_img` and `mm_05_txt`. The last text block (`skip_post_sdpa=True`) is correctly skipped since it has no FFN. |
| **Unified blocks** | Not applicable | Supports `unified_transformer_blocks` with IDs like `uni_00` | SD3 Medium has `depth_unified=0` so none exist, but the code handles architectures that do. |
| **CFG handling** | Not discussed for diagnostics | Input `x` is doubled to batch=2 to match cfg_batch=2 conditioning | Replicates `CFGDenoiser`'s internal batching so the MMDiT sees realistic input shapes. |
| **Modulation caching** | Not applicable | Group calibration points by prompt; batch all timesteps per group into one `cache_modulation_params` call; reload adaLN weights between groups | DiffusionKit's `cache_modulation_params` offloads the adaLN linear weights after caching. Calling it per-point would destroy weights. Grouping by prompt + batching timesteps avoids this. |
| **Timestep dict keys** | Integer class labels / simple indices | String keys via `f"{val:.6f}"` | Avoids float32 → float64 precision drift when using timestep values as dictionary keys. |
| **Statistics storage** | Not specified | Running sums in float64 NumPy (not MLX) | Prevents holding device memory across thousands of calibration points. Float64 accumulation avoids numerical drift. |

---

## Files

| File | Role |
|------|------|
| `activation_tracer.py` | Core instrumentation: `ActivationTracer` dataclass, `PerLayerTimeStats`, monkey-patching machinery (`install_tracing` / `remove_tracing`). Records count, sum, sq_sum, min, max, and histogram per (layer, timestep) pair. |
| `profile_postgelu.py` | Main profiling script: loads Phase 1 calibration data, instantiates the pipeline, groups points by prompt, runs forward passes with tracing active, serializes results to `.npz`. |
| `visualize_postgelu.py` | Visualization: loads the `.npz` and generates all diagnostic plots + a console summary table. |
| `__init__.py` | Package marker |

---

## Output format

`profile_postgelu.py` produces an `.npz` file with flat keys:

| Key pattern | Shape | Description |
|------------|-------|-------------|
| `{layer_id}::t={timestep}::mean` | `(D_hidden,)` | Per-channel mean |
| `{layer_id}::t={timestep}::std` | `(D_hidden,)` | Per-channel standard deviation |
| `{layer_id}::t={timestep}::min` | `(D_hidden,)` | Per-channel minimum |
| `{layer_id}::t={timestep}::max` | `(D_hidden,)` | Per-channel maximum |
| `{layer_id}::t={timestep}::count` | scalar | Total activation elements seen |
| `{layer_id}::t={timestep}::histogram` | `(512,)` | Binned activation counts |
| `timesteps_unique` | `(N_ts,)` | All unique timesteps observed |
| `histogram_bin_edges` | `(513,)` | Bin edges for the histograms ([-8, 8], 512 bins) |

Layer IDs follow the pattern:
- `mm_00_img` through `mm_23_img` — image-stream FFN in multimodal block 0–23
- `mm_00_txt` through `mm_22_txt` — text-stream FFN in multimodal blocks 0–22 (block 23 has `skip_post_sdpa=True`, no FFN)
- `uni_XX` — unified blocks (none in SD3 Medium, but supported)

---

## Visualizations

`visualize_postgelu.py` produces 5 categories of diagnostic output:

| # | Plot | Paper reference | What it shows |
|---|------|----------------|--------------|
| 1 | `range_vs_timestep_{all,image,text}.png` | Fig 2(d) | Median channel activation range (max−min) at each timestep for every layer. Reveals temporal drift. |
| 2 | `histogram_{layer_id}.png` | Fig 3(a) | Per-timestep histogram of all activation values for selected layers. Shows asymmetry and mass concentration near zero. |
| 3 | `channel_ranges_{layer_id}.png` | Fig 3(d) | Per-channel dynamic range at each timestep. Top 2% outlier channels highlighted in red. |
| 4 | `heatmap_mean.png`, `heatmap_std.png` | — | Heatmaps of median channel mean/std across all layers × timesteps. Global view of temporal drift. |
| 5 | `negative_fraction_heatmap.png` | Section III-B | Fraction of activations < 0 per (layer, timestep). Paper reports ~80% for DiT. |
| — | Console summary table | — | Aggregated statistics per layer: median mean, std, range, max range, negative fraction, outlier channel count. |

---

## Usage

### Step 1: Run profiling

```bash
cd /path/to/time-aware-MLXSD-quantization

# Full run (requires Phase 1 output)
python -m src.activation_diagnostics.profile_postgelu \
    --calibration-file DiT_cali_data.npz \
    --num-samples 512 \
    --output activation_stats_postgelu.npz
```

### Step 2: Generate visualizations

```bash
python -m src.activation_diagnostics.visualize_postgelu \
    --stats-file activation_stats_postgelu.npz \
    --output-dir activation_plots/
```

### Dry run (quick validation)

```bash
# Profile only 16 calibration points
python -m src.activation_diagnostics.profile_postgelu \
    --calibration-file dry_run_cali.npz \
    --num-samples 16 \
    --output dry_run_activation_stats.npz

# Visualize
python -m src.activation_diagnostics.visualize_postgelu \
    --stats-file dry_run_activation_stats.npz \
    --output-dir dry_run_activation_plots/
```

### Targeting specific layers for visualization

```bash
python -m src.activation_diagnostics.visualize_postgelu \
    --stats-file activation_stats_postgelu.npz \
    --output-dir activation_plots/ \
    --layers mm_00_img mm_12_img mm_23_img mm_05_txt
```

### All CLI options — profile_postgelu

```
--calibration-file    Path to Phase 1 .npz (default: DiT_cali_data.npz)
--model-version       DiffusionKit model key (default: argmaxinc/mlx-stable-diffusion-3-medium)
--num-samples         Number of calibration points to profile (default: 512)
--seed                Random seed for point selection (default: 0)
--low-memory-mode     Enable low-memory mode (default: on)
--no-low-memory-mode  Disable low-memory mode
--local-ckpt          Path to local MMDiT checkpoint
--output / -o         Output .npz path (default: activation_stats_postgelu.npz)
```

### All CLI options — visualize_postgelu

```
--stats-file    Path to profiling .npz (default: activation_stats_postgelu.npz)
--output-dir    Directory for plot PNGs (default: activation_plots/)
--layers        Specific layer IDs for per-layer plots (default: auto-select a spread)
```

---

## Key implementation details

### Monkey-patching strategy

Since MLX does not provide `register_forward_hook`, we patch three methods at the class level:

1. **`TransformerBlock.pre_sdpa`** — Intercepts the timestep scalar and converts it to a stable string key (`_timestep_to_key`), passing it downstream via the intermediates dict.
2. **`TransformerBlock.post_sdpa`** — Reads the timestep key, pushes a `(layer_id, timestep_key)` context onto the tracer's stack, runs the original `post_sdpa` (which calls the FFN), then pops the context.
3. **`FFN.__call__`** — Reproduces `fc1 → GELU → fc2` but captures the post-GELU tensor between activation and `fc2`, recording statistics into the tracer.

All patches are cleanly removed by `remove_tracing()`, restoring original method references.

### Statistics accumulation

For each `(layer, timestep)` pair, we maintain running aggregates in float64 NumPy:
- **Per-channel**: sum, sum-of-squares, min, max — enabling exact reconstruction of mean, variance, and std.
- **Global histogram**: 512 bins over [-8, 8] — capturing the full distribution shape for asymmetry analysis.

This approach never stores raw activations, keeping memory bounded regardless of how many calibration points are profiled.

### adaLN weight reload

DiffusionKit's `cache_modulation_params` pre-computes modulation parameters for a set of timesteps and then **offloads the adaLN linear layer weights** to save memory. This means:

1. After calling `cache_modulation_params` for one prompt group, those weights are zeroed.
2. Before processing the next prompt group, we must call `pipeline.mmdit.load_weights(pipeline.load_mmdit(only_modulation_dict=True), strict=False)` to restore them.
3. Within a single prompt group, all timesteps are batched into one `cache_modulation_params` call, so the cache covers all needed timesteps without re-caching.

### How results connect to later phases

If the diagnostics confirm:
- **High asymmetry** → justifies the paper's ASymQ (Asymmetry-aware Quantization) approach for post-GELU outputs.
- **Temporal drift** → justifies Time-Grouping Quantization (TGQ), where quantization parameters vary by timestep group.
- **Channel outliers** → justifies Saliency-Weighted Quantization (SWQ), protecting high-range channels from aggressive quantization.
