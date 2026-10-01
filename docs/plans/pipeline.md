# HTG + Bayesian Bits + AdaRound: Full 6-Stage Per-Group Quantization Pipeline

## Context

The existing pipeline applies quantization uniformly across all denoising timesteps.
The 6-stage pipeline below structures all calibration around **HTG (Hierarchical Timestep
Grouping)** — which partitions the trajectory into G groups with similar activation statistics
— so that every subsequent stage operates per-group rather than globally. This gives each
noise regime its own bit-width assignment (Bayesian Bits), shift/outlier config, and
AdaRound+scale calibration.

**Method sources:**
- **HTG**: arXiv 2503.06930 — agglomerative clustering on per-channel shift vectors
- **Bayesian Bits**: arXiv 2005.07093 — L0-gated hierarchical mixed-precision
- **Momentum Shifting / RDM / AdaRound**: existing codebase (`analyze_activations.py`,
  `adaround_optimize.py`) — already implemented globally, needs per-group adaptation
- **Joint Reconstruction**: TaQ-DiT `recon.py` — already implemented in `adaround_optimize.py`
  (alpha + a_scale optimized jointly); **Stages 4+5 share one optimization loop**

---

## Pipeline Overview

```
[Existing output]                          [Needed input]
collect_layer_activations.py
  → layer_statistics.json                  ──► Stage 0 (HTG)
  → timestep_stats/step_*.npz

cache_adaround_data.py
  → adaround_cache/samples/*.npz            ──► Stages 1, 4+5

Stage 0 — HTG Clustering
  → htg_groups.json                         ──► All subsequent stages

Stage 1 — Bayesian Bits (per group)
  → bb_config.json                          ──► Stage 4+5 (bit widths)

Stages 2+3 — Shift + Outlier (per group)
  → quant_config_htg.json                   ──► Stage 4+5 (activation config)

Stages 4+5 — AdaRound + Joint Reconstruction (per group)
  → quantized_weights_htg/group_{g}/        ──► load_adaround_model.py
```

---

## Calibration Data Warning

With 25 selected timesteps and G = ⌊25/10⌋ = **2 groups**, each group gets
~12 timesteps × 5 images = 60 samples — workable but thin for BB (20k iters × batch 16).

**Recommendation**: use G = 3–5 groups (configurable) or increase images to 10–25.
The plan defaults to `--n-groups 5` (i.e., ~5 timesteps per group × 5 images = 25 samples/group).

---

## Stage 0 — HTG Clustering

### New file: `src/htg_cluster.py`

**Input:** `layer_statistics.json` + `timestep_stats/step_*.npz` (from `collect_layer_activations.py`)

**Algorithm** (from arXiv 2503.06930 Algorithm 1) — **per-layer independent clustering:**

The paper runs Algorithm 1 independently for each linear layer. Each layer produces its own
partition boundaries based on its own T shift vectors. The shift vectors are NOT concatenated
across layers.

1. For each layer ℓ, for each selected timestep t, compute per-channel shift vector:
   `z_t_ℓ[c] = (avg_max[c] + avg_min[c]) / 2`  (shape: C_in, from existing `avg_min`/`avg_max`)
2. For each layer independently: agglomerative clustering with **adjacency constraint**
   (only adjacent timestep pairs merge), distance = L2 norm between adjacent group centroids
   (average-linkage, centroid-linkage, or Ward's — paper leaves this open)
3. Merge until G groups remain → per-layer partition boundaries `{τ₁, …, τ_{G-1}}_ℓ`
4. For each group g, compute grouped shift vector per layer:
   `z̄_g_ℓ = mean(z_t_ℓ for t in group g)` — this is the output used at inference

**Practical reconciliation for downstream stages (BB, AdaRound):**
Different layers may produce different partition boundaries. To have one shared timestep
partition for block-level optimization (BB and AdaRound operate per-block, not per-layer):
- Derive a **global consensus partition** from per-layer boundaries: for each potential boundary
  position, take the median boundary index across all layers (rounding to nearest timestep).
- This global partition is used by Stages 1 and 4+5 for sample selection.
- Stages 2+3 use per-layer boundaries to compute per-layer z̄_g (faithful to paper).

**Output:** `htg_groups.json`
```json
{
  "n_groups": 5,
  "global_groups": {
    "0": {"timestep_indices": [0, 4, 8], "sigma_range": [0.8, 1.0]},
    "1": {"timestep_indices": [12, 16, 20], "sigma_range": [0.6, 0.8]},
    ...
  },
  "per_layer_z_bar": {
    "mm0.image_transformer_block.attn.q_proj": {
      "0": [0.12, -0.05, ...],
      "1": [0.08, -0.03, ...],
      ...
    },
    ...
  },
  "sigma_map": {...}
}
```

`global_groups` is used by Stages 1 and 4+5. `per_layer_z_bar` carries the averaged shift
vectors per group per layer, consumed by Stages 2+3.

**CLI:**
```bash
conda run -n diffusionkit python -m src.htg_cluster \
    --stats calibration_data_100/activations/layer_statistics.json \
    --output htg_groups.json \
    [--n-groups 5]
```

**Reuse:** `sigma_map` from `layer_statistics.json` for sigma-to-group mapping;
`avg_min`/`avg_max` arrays already collected per-layer per-timestep by
`collect_layer_activations.py`.

---

## Stage 1 — Bayesian Bits (per HTG group)

### New file: `src/bayesianbits_optimize.py`

**Input:** `adaround_cache/samples/` (filtered by group's timestep indices), `htg_groups.json`

**Core math** (from BayesianBits arXiv 2005.07093):

```python
# Nested scales (per-channel):
s_2 = absmax / (2**(2-1) - 1)
s_4 = s_2 / 3            # 4-bit residual scale
s_8 = s_4 / 9            # 8-bit residual scale

# Straight-through round:
def round_ste(x): return mx.stop_gradient(mx.round(x) - x) + x

# Recursive hierarchical quantization:
x_q2 = s_2 * round_ste(W / s_2)
x_q4 = s_4 * round_ste((W - x_q2) / s_4)
x_q8 = s_8 * round_ste((W - x_q2 - x_q4) / s_8)
W_q  = x_q2 + gate_4 * (x_q4 + gate_8 * x_q8)

# L0 hard concrete gate (training):
u = mx.random.uniform(shape=log_alpha.shape)
gate = mx.clip(mx.sigmoid((mx.log(u/(1-u)) + log_alpha) / beta) * (zeta-gamma) + gamma, 0, 1)

# L0 regularizer (expected bit cost):
p4 = hc_prob_pos(log_alpha_4)      # P(gate_4 active)
p8 = hc_prob_pos(log_alpha_8)      # P(gate_8 active)
reg = gating_lambda * (p4 * 2 + p4 * p8 * 4)  # Expected extra bits beyond base-2

# Loss = block_reconstruction_L2 + reg
```

**BBParams(nn.Module):** `log_alphas_4`, `log_alphas_8` (per layer, shape=W.shape),
`a_scales` (per layer, shape=(1,)), `s_2_np` (constant from absmax).

**Per-group loop:** For each group g, load only the calibration samples whose `step_idx`
is in `htg_groups["global_groups"][str(g)]["timestep_indices"]`. Run `optimize_block_bb()` over those samples.

**Finalize:** Evaluate gates deterministically. Effective bits per layer:
- gate_4=0 → W2; gate_4=1, gate_8=0 → W4; gate_4=1, gate_8=1 → W8

**Output:** `bb_config.json`
```json
{
  "group_0": {"mm0.image_transformer_block.attn.q_proj": 4, "mm0.mlp.fc2": 8, ...},
  "group_1": {...},
  ...
}
```

**CLI:**
```bash
conda run -n diffusionkit python -m src.bayesianbits_optimize \
    --adaround-cache calibration_data_100/adaround_cache \
    --htg-groups htg_groups.json \
    --output bb_config.json \
    [--iters 20000] [--batch-size 16] [--gating-lambda 0.01] [--bits-a 8]
```

**Tests:** `tests/test_bayesianbits_optimize.py` — 8 test classes (see BayesianBits plan).

---

## Stages 2+3 — Per-Group Shift + Outlier Config

### Modified: `src/analyze_activations.py`

Add `--htg-groups htg_groups.json` flag. When present, Stage 0 has already computed per-layer
per-group averaged shift vectors (`per_layer_z_bar` in `htg_groups.json`), so this stage
does NOT need to re-cluster. It simply:

- For each group g and each layer, use `htg_groups["per_layer_z_bar"][layer][str(g)]` as the
  shift vector (instead of per-timestep shift from `layer_statistics.json`)
- Run existing `identify_outlier_channels()` on the per-group averaged `avg_min`/`avg_max`
  (averaged across timesteps in each group from `timestep_stats/step_*.npz`)
- Produce one `scale` per group per layer (max of tensor_absmax across timesteps in group)

**Output:** `quant_config_htg.json` — same structure as `quant_config.json` but keyed by
`group_id` instead of `step_key`. Outlier_config also per-group (using group-averaged stats).

```bash
conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data_100/activations/layer_statistics.json \
    --htg-groups htg_groups.json \
    --output quant_config_htg.json
```

**Reuse:** `identify_outlier_channels()` (line ~220 in `analyze_activations.py`) unchanged.
`load_stats_v2()` reused to load per-timestep NPZ data for group averaging.

---

## Stages 4+5 — Per-Group AdaRound + Joint Reconstruction

### Modified: `src/adaround_optimize.py`

Add two new flags:
- `--htg-groups htg_groups.json` — when set, run one optimization pass per group
  using only that group's calibration samples
- `--bb-config bb_config.json` — when set, use per-layer bit widths from Stage 1
  instead of global `--bits-w`

**Per-group loop** (uses `global_groups` from `htg_groups.json` for sample selection):
```python
global_groups = htg_groups["global_groups"]
for group_id, group_info in global_groups.items():
    group_step_indices = set(group_info["timestep_indices"])
    group_samples = [f for f in sample_files
                     if int(f.stem.split("_")[1]) in group_step_indices]
    # Run existing optimize_block() over group_samples
    # Save to quantized_weights_htg/group_{group_id}/weights/{block_name}.npz
```

**Bit-width selection:** If `--bb-config` given, look up `bits_w` per linear path from
`bb_config[f"group_{group_id}"]`. Otherwise use global `--bits-w`.

**Note:** Stages 4 and 5 are one optimization loop (alpha + a_scale jointly), exactly as
in TaQ-DiT `recon.py` and the existing `adaround_optimize.py`. Stage 5 is not a separate
script — it's the a_scale optimization half of Stage 4's Adam pair.

**Output:** `quantized_weights_htg/group_{g}/` (one directory per group, same layout as
existing `quantized_weights/`).

```bash
conda run -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data_100/adaround_cache \
    --output quantized_weights_htg \
    --htg-groups htg_groups.json \
    --bb-config bb_config.json \
    [--iters 20000] [--batch-size 16] [--bits-a 8]
```

---

## Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| `src/htg_cluster.py` | **New** | HTG agglomerative clustering |
| `src/bayesianbits_optimize.py` | **New** | BB mixed-precision per group |
| `src/analyze_activations.py` | **Modify** | Add `--htg-groups` for per-group shift+outlier |
| `src/adaround_optimize.py` | **Modify** | Add `--htg-groups`, `--bb-config`, per-group loop |
| `tests/test_htg_cluster.py` | **New** | Tests for clustering logic |
| `tests/test_bayesianbits_optimize.py` | **New** | Tests for BB primitives (8 classes) |
| `README.md` | **Modify** | Add HTG pipeline section |
| `CLAUDE.md` | **Modify** | Update architecture section |

**Unchanged (reused):**
- `collect_layer_activations.py` — provides Stage 0 input data as-is
- `cache_adaround_data.py` — provides Stage 1/4 input data as-is
- `load_adaround_model.py` — can load per-group weights (each group's dir = one quantized model)

---

---

## Implementation Context

All code needed to implement without re-reading source files.

### Key File Paths

| Role | Path |
|------|------|
| Read collect output | `src/analyze_activations.py` → `load_stats_v2()` (line 39) |
| Outlier detection | `src/analyze_activations.py` → `identify_outlier_channels()` (line 92) |
| Block optimization | `src/adaround_optimize.py` → `optimize_block()` (line 394) |
| Block data loader | `src/cache_adaround_data.py` → `load_block_data()` (imported at adaround_optimize.py:65) |
| AdaRound params | `src/adaround_optimize.py` → `AdaRoundParams` (line 265) |
| Block linears | `src/adaround_optimize.py` → `get_block_linears()` |

### `layer_statistics.json` manifest format

```json
{
  "format": "per_timestep_npz_v2",
  "hist_bins": 256,
  "timestep_dir": "<absolute path to timestep_stats/>",
  "sigma_map": {"0": 0.97, "4": 0.85, ...},    // step_key (str) → float
  "step_keys": ["0", "4", "8", "12", ...],       // sorted list of step indices as strings
  "metadata": {
    "key_timesteps": [...],
    "selected_image_ids": [...],
    "num_images": 5,
    "num_timesteps": 25
  }
}
```

### NPZ keys in `timestep_stats/step_{key}.npz`

For each linear layer (layer names safe-encoded: `.` → `_`):
```
{safe}__avg_min      → np.ndarray shape (C_in,)  per-channel running avg of min
{safe}__avg_max      → np.ndarray shape (C_in,)  per-channel running avg of max
{safe}__shift        → np.ndarray shape (C_in,)  only for post-GELU layers (*.mlp.fc2)
{safe}__hist_counts  → np.ndarray  256-bin histogram counts
{safe}__hist_edges   → np.ndarray  histogram edges
```

`timestep_stats/step_{key}_index.json` (scalar summary per layer):
```json
{
  "layer_name": {
    "tensor_absmax": float,
    "hist_p999": float,
    "has_shift": bool
  }
}
```

### `load_stats_v2()` return values (reuse directly)

```python
# In analyze_activations.py, line 39
timesteps, per_step_full, layer_names, metadata, sigma_map = load_stats_v2(stats_path)
# timesteps[step_key][layer_name] = {tensor_absmax, hist_p999, ...}  (scalars)
# per_step_full[step_key][layer_name] = {avg_min, avg_max, shift, tensor_absmax, ...}
# layer_names: sorted list of all layer name strings
# sigma_map: {int(step_key): float(sigma)}
```

### `identify_outlier_channels()` signature (reuse unchanged)

```python
# analyze_activations.py line 92
def identify_outlier_channels(
    avg_min: np.ndarray,   # shape (C_in,)
    avg_max: np.ndarray,   # shape (C_in,)
    threshold_multiplier: float = 2.5,
    bits: int = 8,
) -> Dict:
    # Returns: {outlier_indices, multiplier_vector, scale_normal, scale_outlier}
    # Returns {} if no outliers
```

### `AdaRoundParams` structure (adaround_optimize.py line 265)

```python
params = AdaRoundParams(W_fps_np, bits_w=bits_w, bits_a=bits_a)
# params.alphas  : List[mx.array]  — one per linear, shape=W.shape
# params.a_scales: List[mx.array]  — one per linear, shape=(1,)
# No __call__; params are read directly in _QuantProxy
```

### `optimize_block()` signature (adaround_optimize.py line 394)

```python
params, metrics = optimize_block(
    block=block,          # nn.Module block (e.g. mmdit.multimodal_transformer_blocks[0])
    block_name=block_name,
    is_mm=is_mm,          # True for mm{i}, False for uni{i}
    block_data=block_data,# dict from load_block_data(): keys arg0, arg1, arg2, kw_positional_encodings, out0, out1
    iters=20000,
    batch_size=16,
    bits_w=4,
    bits_a=8,
)
```

`block_data` keys (populated by `load_block_data(block_name, sample_files)`):
```
arg0                 → np.ndarray (N, ...)  image hidden states
arg1                 → np.ndarray (N, ...)  text hidden states (MM only)
arg2                 → np.ndarray (N, ...)  (MM only)
kw_positional_encodings → np.ndarray (N, ...)
out0                 → np.ndarray (N, ...)  reference output (image)
out1                 → np.ndarray (N, ...)  reference output (text, MM only)
```

### adaround_optimize.py existing argparse flags

```
--adaround-cache PATH  (required)
--output PATH          (required)
--iters INT            default 20000
--batch-size INT       default 16
--bits-w INT           default 4  choices=[4, 8]
--bits-a INT           default 8
--w-lr FLOAT           default 1e-3
--a-lr FLOAT           default 4e-5
--blocks STR           comma-separated block names (default: all)
--force                overwrite existing output
```

### adaround_cache NPZ keys per sample file

Each `{img:04d}_{step:03d}.npz` in `adaround_cache/samples/` contains:
```
{block_name}__arg0                    → image hidden states
{block_name}__arg1                    → text hidden states (MM blocks)
{block_name}__arg2                    → (MM blocks)
{block_name}__kw_positional_encodings → positional encodings
{block_name}__out0                    → image output
{block_name}__out1                    → text output (MM blocks)
```

Step index is encoded in the filename: `f.stem.split("_")[1]` → step index as string.

### `quant_config.json` output format (per_timestep_quant_config_v4)

```json
{
  "format": "per_timestep_quant_config_v4",
  "per_timestep": {
    "0": {"layer_name": {"bits": 8, "scale": 0.42, "shift": [...]}, ...},
    "4": {...},
    ...
  },
  "sigma_map": {"0": 0.97, "4": 0.85, ...},
  "outlier_config": {
    "layer_name": {
      "outlier_indices": [...],
      "multiplier_vector": [...],
      "scale_normal": 0.003,
      "scale_outlier": 0.021
    }
  }
}
```

For `quant_config_htg.json` (HTG variant): replace `per_timestep` with `per_group`:
```json
{
  "format": "per_group_quant_config_htg_v1",
  "per_group": {
    "0": {"layer_name": {"bits": 8, "scale": 0.42, "shift": [...]}, ...},
    "1": {...}
  },
  "outlier_config": {...per-group...},
  "n_groups": 5
}
```

### scipy availability

scipy is available in the `diffusionkit` conda environment.
Use `scipy.cluster.hierarchy.linkage` and `fcluster` for agglomerative clustering:

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Per-layer clustering (adjacency-constrained):
# z_layer: shape (T, C_in) — shift vectors for one layer across all timesteps
# Agglomerative clustering with adjacency constraint is NOT built into scipy.
# Must implement manually:

def adjacent_agglomerative(z: np.ndarray, n_groups: int) -> np.ndarray:
    """
    z: (T, D) shift vectors ordered by timestep.
    Returns: (T,) integer group assignments (0..n_groups-1), contiguous.
    Uses average-linkage distance between group centroids.
    """
    T = len(z)
    # Start: each timestep is its own cluster (stored as list of lists of indices)
    clusters = [[i] for i in range(T)]

    while len(clusters) > n_groups:
        # Find adjacent pair with minimum centroid distance
        best_i, best_dist = 0, float("inf")
        for i in range(len(clusters) - 1):
            c1 = z[clusters[i]].mean(axis=0)
            c2 = z[clusters[i + 1]].mean(axis=0)
            d = float(np.linalg.norm(c1 - c2))
            if d < best_dist:
                best_dist = d
                best_i = i
        # Merge clusters[best_i] and clusters[best_i + 1]
        clusters[best_i] = clusters[best_i] + clusters[best_i + 1]
        del clusters[best_i + 1]

    assignments = np.zeros(T, dtype=int)
    for g, indices in enumerate(clusters):
        for idx in indices:
            assignments[idx] = g
    return assignments
```

### Consensus partition derivation

Each layer gives per-layer boundary positions. To get `global_groups`:
```python
# per_layer_boundaries[layer] = sorted list of boundary timestep indices (between groups)
# e.g. for G=3, n_groups-1=2 boundaries

all_boundaries = np.array([boundaries[layer] for layer in layers])  # shape (n_layers, G-1)
# Take median boundary per slot
consensus_boundaries = np.round(np.median(all_boundaries, axis=0)).astype(int)
# Convert to global_groups dict with timestep_indices per group
```

### MLX STE (needed for BayesianBits)

```python
def round_ste(x: mx.array) -> mx.array:
    return mx.stop_gradient(mx.round(x) - x) + x
```

### BayesianBits L0 hard concrete parameters

```python
ZETA = 1.1    # stretch right
GAMMA = -0.1  # stretch left
BETA = 2/3    # temperature

def hc_prob_pos(log_alpha: mx.array) -> mx.array:
    """P(gate > 0) under hard concrete = sigmoid(log_alpha - beta * log(-gamma/zeta))"""
    return mx.sigmoid(log_alpha - BETA * math.log(-GAMMA / ZETA))

def sample_gate(log_alpha: mx.array) -> mx.array:
    """Hard concrete sample in [0, 1]."""
    u = mx.random.uniform(shape=log_alpha.shape)
    s = mx.sigmoid((mx.log(u) - mx.log(1 - u) + log_alpha) / BETA)
    return mx.clip(s * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)
```

---

## Phased Implementation Order

1. **`src/htg_cluster.py` + tests** — unblocks everything else; fast to implement (~1 day)
2. **`src/bayesianbits_optimize.py` + tests** — core new algorithm (~2–3 days)
3. **`analyze_activations.py` --htg-groups** — straightforward adaptation (~0.5 day)
4. **`adaround_optimize.py` --htg-groups + --bb-config** — straightforward adaptation (~1 day)

---

## Verification

```bash
# Stage 0: HTG clustering
conda run -n diffusionkit python -m src.htg_cluster \
    --stats calibration_data_100/activations/layer_statistics.json \
    --output htg_groups.json --n-groups 5
# Confirm: htg_groups.json has global_groups (5 groups), per_layer_z_bar (one entry per linear layer)

# Stage 1: BB on 1 block, 1 group, 500 iters (smoke test)
conda run -n diffusionkit python -m src.bayesianbits_optimize \
    --adaround-cache calibration_data_100/adaround_cache \
    --htg-groups htg_groups.json --output bb_config.json \
    --blocks mm0 --iters 500
# Confirm: bb_config.json shows per-layer bit widths in {2, 4, 8}

# Stages 2+3: per-group quant config
conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data_100/activations/layer_statistics.json \
    --htg-groups htg_groups.json --output quant_config_htg.json
# Confirm: 5 group entries, each with shift[] and outlier_config

# Stages 4+5: per-group AdaRound on 1 block, 500 iters
conda run -n diffusionkit python -m src.adaround_optimize \
    --adaround-cache calibration_data_100/adaround_cache \
    --output quantized_weights_htg --htg-groups htg_groups.json \
    --bb-config bb_config.json --blocks mm0 --iters 500
# Confirm: quantized_weights_htg/group_0/weights/mm0.npz exists

# Tests
conda run -n diffusionkit python -m pytest tests/test_htg_cluster.py tests/test_bayesianbits_optimize.py -v
conda run -n diffusionkit python -m pytest tests/ -q`
