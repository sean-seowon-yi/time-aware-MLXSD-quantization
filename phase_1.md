# Phase 1: EDA — Activation and Weight Profiling

## Overview

Before implementing Q-Diffusion on SD3 Medium's MMDiT, this EDA phase answers two questions:

1. **RQ1 (Block Reconstruction Viability)**: Do the per-layer weight and activation distributions satisfy Q-Diffusion's assumptions — near-Gaussian weights (for AdaRound initialization) and a well-defined block-local reconstruction loss (for BRECQ)?
2. **RQ2 (MMDiT Transferability)**: Do Q-Diffusion's block reconstruction and rounding strategies transfer to MMDiT's novel features (dual-stream attention, flow-matching, adaLN modulation) without structural modification?

The EDA produces empirical evidence to guide Phase 2 implementation decisions. All analysis outputs are in `eda_output/`; all decisions are tabulated in Section 7.

---

## Section 1: Research Questions

### RQ1 — Block Reconstruction Viability

**AdaRound assumption**: Weight distributions should be approximately Gaussian so that rounding variables can be initialized from the weight distribution. Heavy-tailed or multi-modal weight distributions require adjusted initialization.

**BRECQ assumption**: Block-local reconstruction loss should be well-conditioned — i.e., quantizing one block's weights should not dramatically change the input distribution seen by the next block. In U-Net architectures this is enforced by skip connections; in MMDiT it depends on the residual stream remaining stable.

**adaLN complication**: Each block's output is scaled by a timestep-dependent adaLN affine transform. BRECQ's reconstruction target is `W_q * x` vs `W * x`; when adaLN rescales `x` differently per timestep, the reconstruction loss must either (a) be computed after adaLN or (b) use calibration data that averages over the adaLN scale distribution.

**EDA evidence required**: A1/A2 (weight distribution shape and kurtosis — do distributions satisfy near-Gaussian assumption for AdaRound?); A3/A4 (activation drift across timesteps — how much does the reconstruction target move with timestep, affecting BRECQ calibration?).

### RQ2 — MMDiT Transferability

**Block boundary definition**: Q-Diffusion's original BRECQ uses U-Net encoder/decoder blocks as reconstruction units. For MMDiT, the natural unit is a `MultiModalTransformerBlock` (img + txt streams, shared attention). The question is whether the block output is a clean enough reconstruction target when adaLN modulation is involved.

**Dual-stream attention**: img (≈1024 tokens) and txt (≈77 tokens) streams are concatenated for joint SDPA. A per-channel outlier in txt inflates the joint Q/K/V range disproportionate to txt's sequence-length share (≈7%). The reconstruction loss must be computed on the joint output, and the quantization ranges must account for the asymmetric stream contributions.

**Flow-matching schedule**: Q-Diffusion calibrated on DDPM 1000-step schedule; MMDiT uses 25-step Euler flow-matching. The calibration data already collected (68 COCO × 25 steps) is appropriate. The question is whether the reconstruction loss needs to be weighted by timestep importance.

**EDA evidence required**: A6 img/txt asymmetry (range-ratio — is stream-specific quantization needed?); A10 QKV range per stream; A12 cross-stream outlier contamination; A7 negative fraction (post-GELU distribution shape — affects AdaRound rounding threshold initialization).

---

## Section 2: Calibration Data — MS-COCO Sampling

### Rationale

MS-COCO 2017 val captions are used as the prompt source (same justification as Q-Diffusion): 80 object categories, diverse real-world contexts, no class-label bias. This produces more representative activation distributions for quantization than a fixed prompt set.

### Sampling Procedure

- **Source**: MS-COCO 2017 val captions — `captions_val2017.json` (via `pycocotools` or HuggingFace `datasets` `HuggingFaceM4/COCO`)
- **Count**: 68 captions drawn uniformly at random
- **Seed**: Fixed (e.g., `seed=42`) for reproducibility
- **Filter**: Keep only captions with word count in [5, 30]
- **Deduplication**: Lowercase the caption, strip punctuation, hash — discard exact duplicates
- **Output**: list of 68 unique captions

### Trajectory Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Trajectories | 68 | One per COCO prompt |
| Steps per trajectory | 25 (all) | Flow-matching Euler schedule; no subsampling |
| Timesteps collected | All 25 | Dense coverage for temporal drift analysis (A3, A4, A11) |
| Total calibration points | 68 × 25 = **1,700** | 68 samples per timestep |
| Latent size | (32, 32) → 256×256 px | Same as existing calibration config |
| CFG scale | 1.5 | Matches `DEFAULT_CFG_WEIGHT` in `calibration_config.py` |

**Important**: Run the Euler sampler end-to-end for all 25 steps. Do **not** subsample timesteps (unlike the original `sample_cali_data.py` which uses 25 of 100 steps). This gives uniform coverage across the full flow-matching trajectory and avoids gaps in A11's temporal drift heatmap.

### Implementation Notes

`src/calibration_sample_generation/sample_cali_data.py` is the starting point. Three changes are needed:

1. Replace `load_prompts()` with a COCO sampler function
2. Set `NUM_CALIBRATION_SAMPLES=68`, `NUM_SAMPLING_STEPS=25`, `NUM_SELECTED_TIMESTEPS=25`
3. Update `--output` default to `eda_output/coco_cali_data.npz`

All other logic (Euler loop, encoding, save format) is reused unchanged.

### Output Schema

`eda_output/coco_cali_data.npz` — same schema as `DiT_cali_data.npz`:

```
xs              (1700, H, W, C)                   noisy latent inputs
ts              (1700,)                            timesteps
prompt_indices  (1700,)                            maps each point → prompt
cs              (68, cfg_batch, seq, dim)          token-level text embeddings
cs_pooled       (68, cfg_batch, dim_p)             pooled text embeddings
prompts         (68,)                              the caption strings
cfg_scale       scalar
```

---

## Section 3: Data Collection Specification

### Tensor Families

Six tensor families are profiled. The attention projections receive extra emphasis because SD3 MMDiT concatenates Q, K, V from img and txt streams before SDPA, making cross-stream outlier contamination the key transferability question for CSB.

| Family | Location in forward pass | Shape | Hook point |
|---|---|---|---|
| Pre-attn activations | After adaLN affine, before q/k/v proj | `(B, S_img or S_txt, 1536)` | `pre_sdpa` patch |
| Q/K/V projections (per stream) | Output of q_proj, k_proj, v_proj — img and txt separately, before concatenation | `(B, S, 1536)` each | patch `Attention.__call__` |
| Concatenated QKV (joint) | After img+txt concatenation, before SDPA | `(B, S_img+S_txt, 1536)` | patch SDPA input |
| Post-GELU FFN | After fc1+GELU, before fc2 | `(B, S, 6144)` | existing `FFN.__call__` patch |
| Post-SDPA pre-FFN residual | After o_proj+residual, before norm2+FFN | `(B, S, 1536)` | `post_sdpa` patch |
| Weight distributions | Linear layer weight matrices | `(out, in)` | one-time at load |

**Attention concatenation detail**: In `MultiModalTransformerBlock.__call__`, the img and txt Q/K/V projections are computed independently, then concatenated:

```python
q = mx.concatenate([q_img, q_txt], axis=2)   # axis=2 = sequence dim after reshape
k = mx.concatenate([k_img, k_txt], axis=2)
v = mx.concatenate([v_img, v_txt], axis=2)
```

Profiling Q/K/V both before (per-stream) and after (joint) this concatenation directly motivates whether CSB must be applied per-stream or on the joint sequence.

### Per-Tensor Statistics

For each activation tensor at each `(layer_id, timestep_index)`:

- Per-channel: min, max, mean, std
- 512-bin histogram over range `[-8, 8]` (clipped)
- Outlier channel count: channels where robust z-score > 5 (MAD-based)

For weight tensors (one-time):

- Per-output-channel range (max − min)
- Global 512-bin histogram over range `[-0.5, 0.5]` (clipped)
- Kurtosis (to check Gaussian assumption for AdaRound initialization)
- Per-layer kurtosis distribution

### Layer Enumeration

All 24 multimodal transformer blocks × img and txt streams × the following linear layers:

```
{stream} ∈ {img, txt}
{proj}   ∈ {q_proj, k_proj, v_proj, o_proj, fc1, fc2, adaLN_modulation}
```

Layer ID format (consistent with `htg_corrections.npz`): `mm_{idx:02d}_{stream}_{proj}`

Special cases:
- Block mm\_23 (last txt block): `skip_post_sdpa=True` — **exclude** txt FFN (fc1, fc2) for mm\_23
- Blocks with `parallel_mlp=True`: share qkv/fc1 modulation — **guard** before fc1 correction
- Joint concatenated QKV: one entry per block (no stream suffix): `mm_{idx:02d}_joint_{q|k|v}`

Total profiling points: 24 blocks × (7 img + 6 txt + 3 joint) × 25 timesteps = **9,600 profiling cells** (excluding mm\_23 txt FFN).

### Hook Implementation

Extend `src/activation_diagnostics/activation_tracer.py`:

- **Pre-attn hook**: already exists as `pre_sdpa` patch in the tracer. Extend to also record per-stream Q/K/V immediately after projection.
- **Post-SDPA hook**: already exists as `post_sdpa` patch. Confirm it captures the post-o_proj+residual tensor, not just the raw SDPA output.
- **QKV joint hook**: patch the concatenation point in `MultiModalTransformerBlock.__call__` to intercept the joint `q`, `k`, `v` tensors before they enter SDPA.
- **AdaLN offload guard**: follow the pattern in `src/activation_diagnostics/profile_postgelu.py` — call `load_weights(only_modulation_dict=True)` between prompt groups after `cache_modulation_params` offloads the adaLN linear weights.

---

## Section 4: Analysis Catalog

Twelve analyses, each mapped to one or both research questions.

### A1 — Weight Range Heatmap

**Plot**: Heatmap, rows = layer depth (0–23), columns = output channel index, color = log-scale per-channel weight range.

**Purpose**: Identify layers with extreme per-channel weight variation, which would violate the near-Gaussian assumption for AdaRound initialization (RQ2).

**RQ**: RQ2

---

### A2 — Weight Histogram Grid

**Plot**: 2×4 grid of histograms (rows = {img, txt} streams; columns = {q_proj, k_proj, v_proj, fc1} layer types). One histogram per cell, aggregated over all 24 depths.

**Purpose**: Check whether weight distributions are approximately Gaussian across layer types (RQ2 for AdaRound/BRECQ).

**RQ**: RQ2

---

### A3 — Activation Range vs. Timestep

**Plot**: 4-panel line plot (one panel per tensor family: pre-attn, post-GELU, post-SDPA, and joint QKV). X-axis = timestep index (0–24), Y-axis = median per-channel range across all layers. Also compute **TVC** (temporal variation coefficient) = std\_t / mean\_t of median range.

**Purpose**: Determine whether activations drift significantly across timesteps, and whether SSB grouping is warranted (TVC > 0.2 threshold). Primary evidence for RQ1.

**RQ**: RQ1 + RQ2

---

### A4 — Temporal Drift Heatmaps

**Plot**: 8 heatmaps (4 tensor families × 2 streams). Rows = layer depth, columns = timestep index (0–24). Color = median per-channel activation range at that `(layer, timestep)`.

**Purpose**: Identify which layers drive the temporal drift seen in A3. Layers with high temporal drift are SSB candidates.

**RQ**: RQ1

---

### A5a — Outlier Concentration Bar Chart

**Plot**: Grouped bar chart. X-axis = layer index (all 24 depths). Two bars per depth: img stream and txt stream. Y-axis = fraction of channels classified as outliers (robust z-score > 5). Reference horizontal line at the DiT baseline value (from PTQ4DiT paper if available; otherwise at 0.05).

**Purpose**: Quantify per-stream outlier concentration. High outlier concentration motivates CSB; comparison to the DiT baseline validates (or refutes) the PTQ4DiT transfer assumption.

**RQ**: RQ1

---

### A5b — Outlier vs. TVC Scatter (Key RQ1 Evidence)

**Plot**: Scatter plot. X-axis = outlier concentration ratio (fraction of outlier channels per layer). Y-axis = TVC (temporal variation coefficient per layer). Each point = one layer (colored by type: q/k/v\_proj, o\_proj, fc1, fc2). Quadrant lines at X=0.05 and Y=0.2.

**Purpose**: This is the primary evidence for RQ1. Four quadrants:
- **High outlier + High TVC** → SSB and CSB must be applied together (correlated failure modes)
- **High outlier + Low TVC** → CSB only (static per-channel correction is sufficient)
- **Low outlier + High TVC** → SSB only (temporal drift dominates, no per-channel outliers)
- **Low outlier + Low TVC** → neither method needed; AdaRound/BRECQ alone may suffice

If points separate cleanly into at most two non-overlapping quadrants, SSB/CSB and AdaRound/BRECQ compose independently (RQ1 confirmed). If all layers cluster in high-outlier + high-TVC, the methods target the same failure mode and their interaction must be explicitly modeled.

**RQ**: RQ1 (key evidence)

---

### A6 — Img/Txt Stream Asymmetry (Key RQ2 Evidence)

**Plot**: Three subplots:
1. Range-ratio heatmap: rows = layer depth, columns = {q, k, v, o, fc1, fc2}, color = `max(img_range, txt_range) / min(img_range, txt_range)`. Red = asymmetric.
2. Depth profile: line plot of mean range-ratio vs. layer depth.
3. Negative fraction difference: `neg_frac_img - neg_frac_txt` per layer (bar chart), showing whether the two streams have systematically different activation sign distributions.

**Decision threshold**: If `range_ratio > 2` for any (depth, projection) cell, a uniform PTQ4DiT grouping scheme applied to the joint sequence will be suboptimal — stream-specific scale parameters are required in Phase 2.

**RQ**: RQ2 (key evidence)

---

### A7 — Negative Fraction All Families

**Plot**: 3-panel heatmap. Rows = layer depth, columns = timestep or channel statistic. One panel each for: pre-attn activations, post-GELU FFN activations, post-SDPA residual. Color = fraction of negative-valued channels.

**Purpose**: Post-GELU activations should be non-negative (ReLU-like behavior). Significant negative fraction post-GELU indicates the GELU is not saturating and the activation distribution is more symmetric than assumed. This affects the AdaRound rounding initialization (asymmetric distributions require adjusted rounding thresholds).

**RQ**: RQ1

---

### A8 — Activation-Weight Interaction Score

**Plot**: Heatmap. Rows = layer depth, columns = timestep index. Color = `activation_range × weight_range` (product of median channel range for activation and weight at that layer). Normalized per row.

**Purpose**: Identify layers where both activation AND weight variability are high — these are the highest-priority candidates for combined PTQ4DiT + Q-Diffusion treatment. Layers where only one is high may need only one method.

**RQ**: RQ1 + RQ2 (joint evidence)

---

### A9 — Calibration Set Coverage

**Plot**: Two subplots:
1. Timestep histogram: counts of calibration points per timestep index (should be uniform: 68 per step).
2. Prompt similarity matrix: 68×68 heatmap of pairwise cosine similarity between COCO caption TF-IDF vectors. Off-diagonal high similarity = redundant prompts.

**Purpose**: Verify data quality — uniform timestep coverage and prompt diversity. If prompts cluster (similarity > 0.8 for any pair), the COCO sampling failed to provide diverse conditioning.

**RQ**: Data quality (not RQ1/RQ2 directly)

---

### A10 — QKV Range: Img vs. Txt vs. Joint (Key Attention Evidence)

**Plot**: For each of the 24 blocks, a 3-panel bar chart (Q, K, V). Each panel shows three bars: img-stream median channel range, txt-stream median channel range, and joint-sequence median channel range. Aggregated over all 25 timesteps.

**Purpose**: Directly quantify the scale difference between the two streams at each projection type and depth. If the joint range is dominated by one stream's outlier channels rather than being the arithmetic mean of both, that stream's outliers are disproportionately biasing the joint attention distribution.

**RQ**: RQ2 (key attention evidence)

---

### A11 — QKV Temporal Drift Across All 25 Timesteps

**Plot**: Three heatmaps (one per projection: Q, K, V). Rows = block depth (0–23), columns = timestep index (0–24). Color = median Q/K/V channel range at that `(depth, timestep)`. All 25 timesteps shown (no subsampling).

**Purpose**: Determine whether QKV projections exhibit temporal drift comparable to the FFN activations in A4. If Q/K/V drift is low (TVC < 0.2) but FFN drift is high, SSB should be scoped to FFN only. If both drift, SSB must wrap the entire transformer block (pre-attn normalization + post-SDPA FFN together).

**RQ**: RQ1 + RQ2

---

### A12 — Cross-Stream QKV Outlier Contamination (Key Attention Evidence)

**Plot**: Per-block stacked bar chart (24 bars on X-axis). For each block and each projection (Q, K, V), the bar shows what fraction of the joint-sequence outlier channels originate from the img stream vs. the txt stream. Stacked: img fraction (bottom) + txt fraction (top) = 1.0. Three groups of 24 bars (one group per projection), side by side.

**Metric**: For each joint QKV tensor at each block, identify outlier channels (robust z-score > 5). Then trace each outlier channel back to whether it originated from the img or txt subrange of the concatenated sequence. Fraction = outlier\_count\_from\_stream / total\_joint\_outlier\_count.

**Expected txt baseline**: txt contributes ≈77/(77+1024) ≈ 7% of tokens. If txt's outlier fraction substantially exceeds 7%, txt-stream outliers are disproportionately contaminating the joint attention.

**Decision**: If txt stream contributes outlier channels disproportionate to its sequence-length fraction (fraction > 2× baseline), CSB must be applied independently per stream before concatenation, not on the joint sequence. The threshold `2×` is conservative; use the Wilcoxon signed-rank test (Section 5) to confirm statistical significance.

**RQ**: RQ2 (key attention evidence)

---

## Section 5: Statistical Procedures

### TVC (Temporal Variation Coefficient)

```
TVC(layer) = std_over_timesteps(median_channel_range(layer, t)) / mean_over_timesteps(median_channel_range(layer, t))
```

- **Threshold**: TVC > 0.2 → layer is an SSB candidate
- Computed separately for each tensor family and each stream
- Reported in `eda_output/tables/tvc_ranking.csv`

### SSB Group Proposal

For layers with TVC > 0.2: apply K=4 k-means clustering on the 25-timestep activation range vector. Report:
- Within-group variance fraction (should be < 0.1 for clean grouping)
- Group boundary timestep indices

Output: `eda_output/tables/ssb_group_assignment.csv`

### Outlier Detection (Robust Z-Score)

```
z_robust(x) = |x - median(x)| / (1.4826 * MAD(x))
```

- **Threshold**: z\_robust > 5
- MAD-based (median absolute deviation), robust to the scale variation expected across layers
- Applied per-layer per-timestep (not globally normalized)

### Cross-Stream Significance Test

For A12: test whether the img-stream outlier fraction and txt-stream outlier fraction differ significantly from their sequence-length baseline fractions.

- **Test**: Wilcoxon signed-rank test, comparing per-block txt outlier fraction against the null baseline (77 / (77 + 1024) ≈ 0.070)
- **Correction**: Bonferroni correction for 23 comparisons (24 blocks minus block 23 which has no txt FFN) → threshold α = 0.05 / 23 ≈ 0.0022
- Output: `eda_output/tables/cross_stream_stats.csv`

---

## Section 6: Output Artifacts

```
eda_output/
├── coco_cali_data.npz              ← calibration latents and embeddings (Section 2)
├── weight_stats.npz                ← per-channel weight stats for all linear layers
├── activation_stats_full.npz       ← all 6 tensor families, all timesteps, all layers
│                                      includes Q/K/V per-stream and joint stats
├── plots/
│   ├── A1_weight_range_heatmap.png
│   ├── A2_weight_histogram_grid.png
│   ├── A3_activation_range_vs_timestep.png
│   ├── A4_temporal_drift_heatmaps.png
│   ├── A5a_outlier_concentration_bar.png
│   ├── A5b_outlier_vs_tvc_scatter.png
│   ├── A6_stream_asymmetry.png
│   ├── A7_negative_fraction_heatmaps.png
│   ├── A8_activation_weight_interaction.png
│   ├── A9_calibration_coverage.png
│   ├── A10_qkv_range_per_block.png
│   ├── A11_qkv_temporal_drift.png
│   └── A12_cross_stream_outlier_contamination.png
└── tables/
    ├── tvc_ranking.csv             ← all layers ranked by TVC, with SSB candidacy flag
    ├── outlier_channel_count.csv   ← per-layer outlier counts (img vs. txt)
    ├── ssb_group_assignment.csv    ← K=4 group boundaries for SSB-candidate layers
    ├── cross_stream_stats.csv      ← Wilcoxon test results for A12
    └── qkv_outlier_source.csv      ← per-block fraction of joint outliers from img vs. txt
```

---

## Section 7: Interpretation Summary — Phase 2 Decision Map

Each analysis produces evidence that maps to a specific Phase 2 Q-Diffusion implementation decision.

| Analysis | Evidence | Decision for Phase 2 (Q-Diffusion) |
|---|---|---|
| A1 (weight range heatmap) | Per-channel weight range variation per layer | Flag layers with high per-channel variation → AdaRound initialization must use layer-specific rounding threshold, not global Gaussian assumption |
| A2 (weight histogram grid) | Distribution shape and kurtosis per layer type | Kurtosis >> 0 (heavy tails) → AdaRound needs adjusted initialization; confirms or rejects near-Gaussian assumption per projection type |
| A3 (activation range vs. timestep) | TVC per tensor family | High TVC → BRECQ calibration must average over multiple timesteps, not a single representative; determines minimum calibration set size per block |
| A4 (temporal drift heatmaps) | Per-layer TVC at each depth | Identifies which blocks have high activation variability → those blocks need timestep-stratified calibration samples for BRECQ reconstruction loss |
| A5a (outlier bar chart) | Per-layer outlier fraction, img vs. txt | High outlier fraction → block reconstruction loss may be dominated by outlier channels; may need channel-wise weighting in BRECQ loss |
| A5b (outlier vs. TVC scatter) | Outlier fraction vs. TVC per layer | Layers with both high outlier and high TVC are hardest for BRECQ — reconstruction loss is simultaneously unstable and dominated by outliers |
| A6 (stream asymmetry) | img vs. txt range-ratio per block | **RQ2 key**: if range\_ratio > 2 → joint quantization range is stream-dominated; per-stream quantization ranges needed before concatenation |
| A7 (negative fraction) | Post-GELU negative fraction | Non-zero post-GELU negative fraction → GELU is not behaving as a pure rectifier; AdaRound rounding direction initialization should not assume non-negative activations |
| A8 (interaction score) | Activation IQR × weight IQR product | Identifies highest-priority blocks for BRECQ (where both activation and weight variability are high) vs. blocks where AdaRound alone may suffice |
| A9 (calibration coverage) | Timestep uniformity + prompt diversity | Confirms calibration data quality; non-uniform timestep coverage would bias BRECQ reconstruction loss toward overrepresented timesteps |
| A10 (QKV range per stream) | Absolute img vs. txt range at each depth | **Attention RQ2**: if txt-stream QKV range >> img-stream → joint BRECQ reconstruction loss is txt-dominated; stream-aware loss weighting needed |
| A11 (QKV temporal drift) | TVC for Q, K, V across all timesteps | **BRECQ scope**: high QKV TVC → reconstruction loss must be evaluated across the full timestep range, not a single timestep sample |
| A12 (cross-stream outlier contamination) | Fraction of joint outliers from img vs. txt | **RQ2 attention key**: if txt fraction > 2× baseline → txt outliers dominate joint attention quantization error; per-stream quantization boundary must be set before concatenation |

---

## Critical File References

| File | Purpose in Phase 1 |
|---|---|
| `src/calibration_sample_generation/sample_cali_data.py` | Reuse for COCO; replace `load_prompts()` only |
| `src/calibration_sample_generation/calibration_config.py` | Constants to update: `NUM_CALIBRATION_SAMPLES=68`, `NUM_SAMPLING_STEPS=25`, `NUM_SELECTED_TIMESTEPS=25` |
| `src/activation_diagnostics/activation_tracer.py` | Extend with per-stream Q/K/V hooks and joint QKV hook |
| `src/activation_diagnostics/visualize_postgelu.py` | Reuse `_layer_sort_key`, heatmap and histogram utilities for A1–A12 |
| `src/activation_diagnostics/profile_postgelu.py` | Reuse prompt-group forward-pass loop + adaLN offload guard pattern |
| `DiffusionKit/python/src/diffusionkit/mlx/mmdit.py` | Layer enumeration; locate exact hook points for `pre_sdpa`, `post_sdpa`, and QKV concatenation |
