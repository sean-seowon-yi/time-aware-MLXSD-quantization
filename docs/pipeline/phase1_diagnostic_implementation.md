# Phase 1 Diagnostic Implementation Guide

**Code layout:** The running implementation is under `src/phase1/` (`collect.py`, `hooks.py`, `run_collection.py`, `registry.py`, `analyze.py`, `visualize.py`). Outputs default to `diagnostics/`. This guide describes design and behavior; where naming differs from early drafts, trust the `src/phase1/` sources.

## 1. Objectives

Instrument the SD3 Medium MMDiT denoiser and collect per-channel activation and weight statistics across the full Euler sampling trajectory. No quantization is performed. The deliverables are data and visualizations that answer:

1. **Salient activation channels**: Do some channels carry disproportionately large activations, and are they consistent across tokens and prompts?
2. **Salient weight channels**: Do some input channels of weight matrices have disproportionately large magnitudes?
3. **Complementarity**: Within a given layer, are the channels that dominate activation salience different from those that dominate weight salience?
4. **Temporal variation**: Does the set or magnitude of salient activation channels change across the sigma trajectory (σ ≈ 1.0 → σ ≈ 0.0)?
5. **Risk ranking**: Which layers and submodule families are the hardest quantization targets, and which modality branch (image vs text) is more sensitive?

These map directly to the phenomena exploited by PTQ4DiT's Channel-wise Salience Balancing (CSB) and Spearman's ρ-guided Salience Calibration (SSC).

---

## 2. Scope

### In scope

The MMDiT denoiser backbone only — all `nn.Linear` layers inside:
- 24 `MultiModalTransformerBlock`s (image-side and text-side `TransformerBlock`s)
- `context_embedder` (standalone linear projection)
- `final_layer` (adaLN + linear projection)

### Out of scope

- VAE encoder / decoder
- CLIP-L, CLIP-G, T5-XXL text encoders (frozen, not quantization targets)
- Any quantization implementation or balancing matrix computation

---

## 3. Layer Selection

### 3.1 Layer registry

Every linear layer to analyze must be enumerated in a registry. Each entry records a canonical name, the module path for hook attachment, the modality branch, and the weight shape.

Build the registry by walking the model tree:

```python
def build_layer_registry(mmdit):
    registry = []
    for bidx, block in enumerate(mmdit.multimodal_transformer_blocks):
        skip_text_post = block.text_transformer_block.skip_post_sdpa
        for side, tb in [("image", block.image_transformer_block),
                         ("text",  block.text_transformer_block)]:
            for name in ["q_proj", "k_proj", "v_proj"]:
                layer = getattr(tb.attn, name)
                registry.append({
                    "name": f"blocks.{bidx}.{side}.attn.{name}",
                    "module": layer,
                    "block": bidx,
                    "family": name,
                    "side": side,
                    "d_in": layer.weight.shape[1],
                })
            if not (side == "text" and skip_text_post):
                for sub in ["o_proj"]:
                    layer = getattr(tb.attn, sub)
                    if not isinstance(layer, nn.Identity):
                        registry.append({
                            "name": f"blocks.{bidx}.{side}.attn.{sub}",
                            "module": layer,
                            "block": bidx,
                            "family": sub,
                            "side": side,
                            "d_in": layer.weight.shape[1],
                        })
                for sub in ["fc1", "fc2"]:
                    layer = getattr(tb.mlp, sub)
                    registry.append({
                        "name": f"blocks.{bidx}.{side}.mlp.{sub}",
                        "module": layer,
                        "block": bidx,
                        "family": sub,
                        "side": side,
                        "d_in": layer.weight.shape[1],
                    })
    # Singletons
    registry.append({
        "name": "context_embedder",
        "module": mmdit.context_embedder,
        "block": -1,
        "family": "context_embedder",
        "side": "shared",
        "d_in": mmdit.context_embedder.weight.shape[1],
    })
    registry.append({
        "name": "final_layer.linear",
        "module": mmdit.final_layer.linear,
        "block": -1,
        "family": "final_linear",
        "side": "image",
        "d_in": mmdit.final_layer.linear.weight.shape[1],
    })
    return registry
```

### 3.2 Expected layer count

| Group | Per block | Total |
|---|---|---|
| Image attn (q/k/v_proj) | 3 | 72 |
| Text attn (q/k/v_proj) | 3 | 72 |
| Image o_proj | 1 | 24 |
| Text o_proj | 1 | 23 |
| Image FFN (fc1, fc2) | 2 | 48 |
| Text FFN (fc1, fc2) | 2 | 46 |
| context_embedder | — | 1 |
| final_layer.linear | — | 1 |
| **Total** | | **287** |

adaLN modulation layers (48 + 1 = 49 total) are analyzed separately via the modulation cache, not via forward hooks during the denoising loop. Including those gives ~336 layers total.

---

## 4. Hook Strategy

### 4.1 Why hooks

MLX does not have PyTorch-style `register_forward_hook`. Instead, intercept the `__call__` method of each target `nn.Linear` to capture its input tensor before the linear transform executes. The weight tensor is accessible as `module.weight` at any time (it is static).

### 4.2 Hook wrapper

```python
import mlx.core as mx
import numpy as np

class LinearHook:
    def __init__(self, module, name, collector):
        self.name = name
        self.collector = collector
        self._original_call = module.__class__.__call__

        outer = self
        original = self._original_call

        def hooked_call(self_module, x):
            outer.collector.record(outer.name, x, self_module.weight)
            return original(self_module, x)

        module.__class__ = type(
            module.__class__.__name__ + "_Hooked",
            (module.__class__,),
            {"__call__": hooked_call},
        )
        self.module = module
```

**Critical: MLX lazy evaluation.** The activation tensor `x` is a lazy graph node. Any reduction must be materialized immediately with `mx.eval()` before the graph is released. The `collector.record` method must do this.

### 4.3 What each hook captures

On every call, the hook receives the input activation `X` and has access to the weight `W`. It should compute and store per-channel summary statistics (not the raw tensor):

```python
class ChannelStatsCollector:
    def __init__(self):
        self.records = []
        self._step_idx = 0
        self._sigma = 0.0
        self._prompt_id = ""
        self._seed = 0

    def set_context(self, step_idx, sigma, prompt_id, seed):
        self._step_idx = step_idx
        self._sigma = sigma
        self._prompt_id = prompt_id
        self._seed = seed

    def record(self, layer_name, X, W):
        X_fp32 = X.astype(mx.float32)

        # Flatten all non-channel dims: X is [B, N, 1, d] for all layers in SD3
        if X_fp32.ndim == 4:
            X_flat = X_fp32.reshape(-1, X_fp32.shape[-1])
        elif X_fp32.ndim == 3:
            X_flat = X_fp32.reshape(-1, X_fp32.shape[-1])
        else:
            X_flat = X_fp32

        act_abs = mx.abs(X_flat)
        act_max = mx.max(act_abs, axis=0)          # [d_in]
        act_mean = mx.mean(act_abs, axis=0)         # [d_in]
        act_var = mx.var(act_abs, axis=0)            # [d_in]

        mx.eval(act_max, act_mean, act_var)

        self.records.append({
            "layer": layer_name,
            "step_idx": self._step_idx,
            "sigma": self._sigma,
            "prompt_id": self._prompt_id,
            "seed": self._seed,
            "act_channel_max": np.array(act_max),    # [d_in]
            "act_channel_mean": np.array(act_mean),
            "act_channel_var": np.array(act_var),
            "n_tokens": X_flat.shape[0],
        })
```

Weight statistics are time-independent and computed once:

```python
def compute_weight_salience(registry):
    weight_stats = {}
    for entry in registry:
        W = entry["module"].weight.astype(mx.float32)
        # W shape: [d_out, d_in] — channel axis is dim=1
        w_abs = mx.abs(W)
        w_max = mx.max(w_abs, axis=0)       # [d_in]: max over output dim
        w_mean = mx.mean(w_abs, axis=0)
        mx.eval(w_max, w_mean)
        weight_stats[entry["name"]] = {
            "w_channel_max": np.array(w_max),
            "w_channel_mean": np.array(w_mean),
        }
    return weight_stats
```

### 4.4 adaLN modulation layers

These are NOT executed during the denoising loop (they are pre-cached). To collect their statistics, extract the cached values directly after `cache_modulation_params()` runs:

```python
def collect_adaln_stats(mmdit, timesteps):
    adaln_records = {}
    for bidx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for side, tb in [("image", block.image_transformer_block),
                         ("text",  block.text_transformer_block)]:
            name = f"blocks.{bidx}.{side}.adaLN"
            adaln_records[name] = {}
            for ts_key, params in tb._modulation_params.items():
                params_fp32 = params.astype(mx.float32)
                abs_params = mx.abs(params_fp32)
                channel_max = mx.max(abs_params.reshape(-1, abs_params.shape[-1]), axis=0)
                mx.eval(channel_max)
                adaln_records[name][ts_key] = np.array(channel_max)
    return adaln_records
```

---

## 5. Channel Definition

A "channel" is one feature dimension of the input to a linear layer, indexed along the last axis of the activation tensor and along `dim=1` of the weight matrix.

### 5.1 Axis conventions

| Layer | Activation shape | Channel axis | Weight shape | Channel axis |
|---|---|---|---|---|
| `attn.q_proj` | `[B, N, 1, 1536]` | dim 3 | `[1536, 1536]` | dim 1 |
| `attn.k_proj` | `[B, N, 1, 1536]` | dim 3 | `[1536, 1536]` | dim 1 |
| `attn.v_proj` | `[B, N, 1, 1536]` | dim 3 | `[1536, 1536]` | dim 1 |
| `attn.o_proj` | `[B, N, 1, 1536]` | dim 3 | `[1536, 1536]` | dim 1 |
| `mlp.fc1` | `[B, N, 1, 1536]` | dim 3 | `[6144, 1536]` | dim 1 |
| `mlp.fc2` | `[B, N, 1, 6144]` | dim 3 | `[1536, 6144]` | dim 1 |
| `context_embedder` | `[B, N_txt, 1, 4096]` | dim 3 | `[1536, 4096]` | dim 1 |
| `final_layer.linear` | `[B, N_img, 1, 1536]` | dim 3 | `[64, 1536]` | dim 1 |

The `d_in` for each layer determines the number of channels: 1536 for most layers, 6144 for fc2, 4096 for context_embedder.

### 5.2 Shared input observation

Within one `TransformerBlock`, `q_proj`, `k_proj`, and `v_proj` receive the **same** input tensor (the `modulated_pre_attention` output). Their activation channel statistics are therefore identical. Recording all three is useful for validation but redundant for analysis. Weight statistics differ across the three projections.

---

## 6. Statistics to Collect

### 6.1 Per-channel activation statistics (per layer, per sigma step, per prompt+seed)

| Statistic | Formula | Shape | Purpose |
|---|---|---|---|
| Channel max | `max_tokens(|X[:, :, :, j]|)` | `[d_in]` | **Primary salience** (PTQ4DiT Eq. 4) |
| Channel mean | `mean_tokens(|X[:, :, :, j]|)` | `[d_in]` | Distinguishes persistent vs spike outliers |
| Channel variance | `var_tokens(|X[:, :, :, j]|)` | `[d_in]` | Token-level spread |
| Token count | `N_tokens` | scalar | Needed for cross-prompt aggregation |

### 6.2 Per-channel weight statistics (per layer, computed once)

| Statistic | Formula | Shape | Purpose |
|---|---|---|---|
| Channel max | `max_output_dim(|W[:, j]|)` | `[d_in]` | **Primary salience** (PTQ4DiT Eq. 4) |
| Channel mean | `mean_output_dim(|W[:, j]|)` | `[d_in]` | Weight distribution shape |

### 6.3 Derived statistics (computed during analysis, not during collection)

Computed from the stored per-channel vectors after all data is collected:

| Statistic | Computed from | Purpose |
|---|---|---|
| Top-1 / median ratio | `max(s) / median(s)` | Outlier severity |
| Top-k indices | `argsort(s)[-k:]` | Identity of salient channels |
| Gini coefficient | Standard Gini on `s` | Salience concentration |
| Spearman ρ(s_act, s_wt) | Rank correlation | Complementarity |
| Jaccard(top-k act, top-k wt) | Set overlap | Complementarity (binary) |
| CoV across sigma | `std_σ(s_j) / mean_σ(s_j)` per channel | Temporal stability |
| Top-k overlap across sigma | Jaccard between step pairs | Channel identity stability |

---

## 7. How to Measure Salience

### 7.1 Primary salience definition

Following PTQ4DiT (Eq. 4), the salience of the j-th channel is the maximum absolute value among all elements in that channel:

```
s(X_j) = max(|X_j|)           — activation salience at one sigma step
s(W_j) = max(|W_j|)           — weight salience (time-independent)
```

where `X_j` is the j-th column of the flattened activation matrix `[n_tokens, d_in]` and `W_j` is the j-th column of the weight matrix `[d_out, d_in]`.

### 7.2 Aggregation across prompts

For a given (layer, sigma_step), multiple prompts and seeds each yield an `act_channel_max` vector. Aggregate by taking the elementwise maximum across all prompts at that sigma step:

```python
# For layer L at sigma step s:
all_maxes = [record["act_channel_max"]
             for record in records
             if record["layer"] == L and record["step_idx"] == s]
s_act = np.max(np.stack(all_maxes), axis=0)  # [d_in]
```

This follows the PTQ4DiT convention — salience is a worst-case measure.

### 7.3 Concentration metrics

For each layer and sigma step, compute from the salience vector `s` of length `d_in`:

```python
def salience_concentration(s):
    sorted_s = np.sort(s)
    return {
        "top1_over_median": s.max() / np.median(s),
        "top1_over_top10_mean": s.max() / np.mean(np.sort(s)[-10:]),
        "gini": gini_coefficient(s),
        "top32_mass_fraction": np.sum(np.sort(s)[-32:]) / np.sum(s),
    }
```

---

## 8. How to Measure Activation–Weight Correlation

### 8.1 Spearman correlation

For each layer `L` at each sigma step `s`:

```python
from scipy.stats import spearmanr

s_act = aggregate_act_salience(L, s)  # [d_in]
s_wt  = weight_salience[L]            # [d_in]
rho, p_value = spearmanr(s_act, s_wt)
```

- `rho` near 0: activation and weight salient channels do not overlap → strong complementarity → CSB will be effective.
- `rho` near +1: salient channels coincide → CSB will have limited benefit because both extremes concentrate on the same channels.
- `rho` near -1: anti-correlation — channels with high activation salience have low weight salience and vice versa. This is the ideal case for CSB, enabling maximum redistribution of extremes.

### 8.2 Top-k overlap

```python
k = 32  # or 5% of d_in
top_k_act = set(np.argsort(s_act)[-k:])
top_k_wt  = set(np.argsort(s_wt)[-k:])
jaccard = len(top_k_act & top_k_wt) / len(top_k_act | top_k_wt)
```

Low Jaccard (< 0.1) indicates strong complementarity. High Jaccard (> 0.5) indicates weak complementarity.

### 8.3 Temporal aggregation for SSC relevance

PTQ4DiT's SSC (Eq. 11) weights each timestep by `exp(-ρ_t) / Σ exp(-ρ_τ)`, giving more weight to steps where complementarity is stronger. Compute the ρ trajectory for each layer:

```python
rho_trajectory = []
for step_idx in range(num_steps):
    s_act = aggregate_act_salience(L, step_idx)
    rho, _ = spearmanr(s_act, s_wt)
    rho_trajectory.append(rho)
```

If `rho_trajectory` is nearly constant, SSC adds little value over a single calibration point. If it varies substantially, SSC is justified.

---

## 9. How to Analyze Timestep Variation

### 9.1 Per-channel salience trajectory

For each layer and each channel `j`, form the trajectory:

```python
# trajectory[j] = [s_act_j(σ_0), s_act_j(σ_1), ..., s_act_j(σ_{N-1})]
trajectory = np.stack([
    aggregate_act_salience(L, step_idx)
    for step_idx in range(num_steps)
])  # shape: [num_steps, d_in]
```

### 9.2 Coefficient of variation per channel

```python
cov_per_channel = np.std(trajectory, axis=0) / (np.mean(trajectory, axis=0) + 1e-10)
```

High CoV (> 0.3) indicates that channel salience changes substantially across the trajectory. This directly tests Hypothesis H3 from `Phase1.md`.

### 9.3 Top-k identity stability

```python
top_k_sets = [set(np.argsort(trajectory[t])[-k:]) for t in range(num_steps)]

# Early-to-late overlap
early_late_jaccard = len(top_k_sets[0] & top_k_sets[-1]) / len(top_k_sets[0] | top_k_sets[-1])

# Consecutive step overlap
consecutive_jaccards = [
    len(top_k_sets[t] & top_k_sets[t+1]) / len(top_k_sets[t] | top_k_sets[t+1])
    for t in range(num_steps - 1)
]
```

If consecutive Jaccard is consistently high (> 0.8), the identity of salient channels is stable and temporal weighting may not be necessary. If early-late Jaccard is low (< 0.3), there is a regime shift.

### 9.4 Trajectory shape classification

For each layer, classify the temporal behavior:
- **Stable**: CoV < 0.1 and early-late Jaccard > 0.8
- **Monotonic drift**: salience increases or decreases consistently
- **Regime shift**: abrupt change at a specific sigma
- **Oscillatory**: non-monotonic with multiple peaks

---

## 10. Plots and Tables to Produce

This section specifies **every** visualization to produce. Plots 10.1–10.3 are direct SD3 Medium adaptations of the three diagnostic figures in the PTQ4DiT paper (Figures 3, 4, and 1-Left). Plots 10.4–10.17 are SD3-specific extensions that cover modality asymmetry, block-depth effects, and additional correlation analyses. The final subsection defines the summary diagnostic table.

All plots should use a consistent visual language:
- Image-side layers: **blue** palette
- Text-side layers: **orange** palette
- Shared / global layers: **green** palette
- Salient channels (top-k): **red** highlight or marker
- Use `matplotlib` with `constrained_layout=True`; save both `.pdf` (vector) and `.png` (300 dpi) versions.

---

### Paper reproduction plots

---

### 10.1 Per-channel magnitude + quantization error bar chart — PTQ4DiT Figure 3 reproduction (tests H1)

**What the paper shows.** Figure 3 displays, for a single DiT linear layer, two side-by-side bar plots:
- **Left panel — Activation**: x-axis = channel index `j` (0 … d_in−1), primary y-axis (bars) = `max(|X_j|)` (the per-channel max absolute activation magnitude), secondary y-axis (line/dots) = per-channel quantization error (MSE between full-precision and naively quantized output for that channel).
- **Right panel — Weight**: same layout but using `max(|W_j|)`.

The key visual takeaway is that channels with greater max absolute values incur larger quantization errors.

**SD3 adaptation.**

Produce this figure for **at least 6 representative layers** (one from each major family):

| Layer example | Family | Side |
|---|---|---|
| `blocks.0.image.attn.q_proj` | q_proj | image |
| `blocks.12.text.attn.q_proj` | q_proj | text |
| `blocks.12.image.attn.o_proj` | o_proj | image |
| `blocks.12.image.mlp.fc1` | fc1 | image |
| `blocks.12.image.mlp.fc2` | fc2 | image |
| `context_embedder` | context | shared |
| `final_layer.linear` | final | shared |

For each representative layer, produce a **two-panel figure** (activation left, weight right) with:
- **Bars**: `s(X_j) = max(|X_j|)` or `s(W_j) = max(|W_j|)` for all channels `j`, sorted by channel index.
- **Overlaid line (secondary y-axis)**: Per-channel quantization error from naive W8A8 uniform quantization. For the activation panel, compute `mse_act_j = mean_tokens((X_j - Q(X)_j)^2)` — the MSE of channel j's values before vs after tensor-wise quantization. For the weight panel, compute `mse_wt_j = mean_out((W_j - Q(W)_j)^2)` — the MSE of channel j's weight column before vs after channel-wise quantization. `Q` denotes round-to-nearest uniform quantization with min-max scale.
- **Highlight**: Color the top-k (k = 32 or top 2%) salient channels in red to make them visually pop.
- **Title**: Include layer name, d_in, sigma step used (choose the median sigma, σ ≈ 0.5).

```python
def plot_figure3_reproduction(layer_name, act_channel_max, wt_channel_max,
                               act_channel_mse, wt_channel_mse, k=32):
    fig, (ax_act, ax_wt) = plt.subplots(1, 2, figsize=(16, 5))
    d_in = len(act_channel_max)
    channels = np.arange(d_in)

    top_k_act = set(np.argsort(act_channel_max)[-k:])
    colors_act = ["#e74c3c" if j in top_k_act else "#3498db" for j in channels]
    ax_act.bar(channels, act_channel_max, color=colors_act, width=1.0, edgecolor="none")
    ax_act.set_ylabel("max|activation|", color="#3498db")
    ax_act.set_xlabel("Channel index")
    ax_act_mse = ax_act.twinx()
    ax_act_mse.plot(channels, act_channel_mse, color="#2ecc71", linewidth=0.8, alpha=0.7)
    ax_act_mse.set_ylabel("Channel quant. MSE", color="#2ecc71")
    ax_act.set_title(f"{layer_name} — Activation channels")

    top_k_wt = set(np.argsort(wt_channel_max)[-k:])
    colors_wt = ["#e74c3c" if j in top_k_wt else "#e67e22" for j in channels]
    ax_wt.bar(channels, wt_channel_max, color=colors_wt, width=1.0, edgecolor="none")
    ax_wt.set_ylabel("max|weight|", color="#e67e22")
    ax_wt.set_xlabel("Channel index")
    ax_wt_mse = ax_wt.twinx()
    ax_wt_mse.plot(channels, wt_channel_mse, color="#2ecc71", linewidth=0.8, alpha=0.7)
    ax_wt_mse.set_ylabel("Channel quant. MSE", color="#2ecc71")
    ax_wt.set_title(f"{layer_name} — Weight channels")

    fig.suptitle(f"Per-channel salience & quantization error (PTQ4DiT Fig.3 analog)")
    return fig
```

**What to look for:**
- Do a small number of channels have magnitudes 10–100× larger than the median? (confirms salient channels exist in SD3)
- Does the MSE line track the magnitude bars? (confirms salient channels cause disproportionate error)
- Is the spike pattern different between image-side and text-side layers?

---

### 10.2 Temporal boxplot of activation channel magnitudes — PTQ4DiT Figure 4 reproduction (tests H3)

**What the paper shows.** Figure 4 displays a single boxplot chart for one linear layer:
- x-axis = timestep (t₁, t₂, …, t_T)
- y-axis = max absolute activation magnitude across channels
- Each box-and-whisker represents the distribution of `{s(X_j^t) : j = 1…d_in}` at timestep `t`.

The key visual takeaway is that the distribution of channel salience varies significantly across timesteps.

**SD3 adaptation.**

Produce this figure for the same representative layers from 10.1.

For each layer, create a **boxplot** with:
- x-axis: sigma step index (0 … N−1), labeled with the actual σ value (e.g., 0.97, 0.81, …, 0.03). Use all collected sigma steps (recommended: 25 uniformly spaced).
- y-axis: per-channel max absolute activation magnitude `s(X_j^σ)`.
- Each box at position `σ_i` summarizes the distribution over all `d_in` channels.
- Overlay the individual top-k channel points as scatter (semi-transparent red) to show outlier channels.
- Optionally add a second y-axis or annotation line showing the **median** salience at each step.

```python
def plot_figure4_reproduction(layer_name, trajectory, sigma_values, k=32):
    """
    trajectory: np.ndarray of shape [num_steps, d_in]
                trajectory[t, j] = s(X_j) at sigma step t
    sigma_values: list of float sigma values
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    num_steps, d_in = trajectory.shape

    bp = ax.boxplot(
        [trajectory[t] for t in range(num_steps)],
        positions=range(num_steps),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#3498db", alpha=0.4),
        medianprops=dict(color="#e74c3c", linewidth=1.5),
    )

    for t in range(num_steps):
        top_k_indices = np.argsort(trajectory[t])[-k:]
        ax.scatter(
            [t] * k, trajectory[t, top_k_indices],
            color="#e74c3c", alpha=0.3, s=8, zorder=3,
        )

    ax.set_xticks(range(num_steps))
    ax.set_xticklabels([f"{s:.2f}" for s in sigma_values], rotation=45, fontsize=7)
    ax.set_xlabel("σ value (sigma step)")
    ax.set_ylabel("max|activation| per channel")
    ax.set_title(f"{layer_name} — Temporal variation of activation channel salience (PTQ4DiT Fig.4 analog)")
    return fig
```

**What to look for:**
- Do the boxes shift vertically (median changes)? → distribution-level temporal variation
- Do the whiskers/outliers change positions? → the identity of salient channels changes
- Is the variation monotonic (smooth drift from high σ to low σ) or non-monotonic (regime shifts)?
- Compare image-side vs text-side: does one modality have more temporal variation?

---

### 10.3 Complementarity bar chart — PTQ4DiT Figure 1-Left reproduction (tests H2)

**What the paper shows.** Figure 1 (Left) shows, for a single linear layer, two grouped bar charts side by side:
- **Activation panel**: Per-channel magnitudes at three different timesteps (t₁, t₂, t₃), with salient channels highlighted. This shows that salient activation channels vary over timesteps.
- **Weight panel**: Per-channel magnitudes (fixed across timesteps). The crucial visual point is that channels that are salient in activation are NOT the same channels that are salient in weight — demonstrating complementarity.

**SD3 adaptation.**

For 3 representative layers, produce a **3-row × 2-column figure**:

- **Columns**: Left = Activation, Right = Weight
- **Rows**: 3 sigma steps — early (σ ≈ 0.95), middle (σ ≈ 0.50), late (σ ≈ 0.05)
- **Bars**: Per-channel magnitude, sorted by channel index. Color salient channels (top-k) in red.
- **Weight column**: Same data repeated in all 3 rows (weights don't change), but highlight different top-k weight channels in orange to contrast with the red activation channels.
- **Annotation**: For each row, annotate the Spearman ρ between the activation salience and weight salience vectors, plus the Jaccard overlap of their top-k sets.

```python
def plot_figure1_left_reproduction(layer_name, trajectory, wt_salience,
                                    sigma_values, step_indices, k=32):
    """
    trajectory: [num_steps, d_in]
    wt_salience: [d_in]
    step_indices: list of 3 ints (early, mid, late)
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    d_in = len(wt_salience)
    channels = np.arange(d_in)
    top_k_wt = set(np.argsort(wt_salience)[-k:])

    for row, step_idx in enumerate(step_indices):
        sigma = sigma_values[step_idx]
        act_s = trajectory[step_idx]
        top_k_act = set(np.argsort(act_s)[-k:])

        rho_val, _ = spearmanr(act_s, wt_salience)
        jaccard = len(top_k_act & top_k_wt) / len(top_k_act | top_k_wt)

        colors_act = ["#e74c3c" if j in top_k_act else "#3498db" for j in channels]
        axes[row, 0].bar(channels, act_s, color=colors_act, width=1.0, edgecolor="none")
        axes[row, 0].set_ylabel(f"σ={sigma:.2f}\nmax|act|")
        axes[row, 0].annotate(f"ρ={rho_val:.3f}  J={jaccard:.3f}",
                               xy=(0.98, 0.92), xycoords="axes fraction",
                               ha="right", fontsize=9, bbox=dict(boxstyle="round", fc="wheat"))

        colors_wt = ["#e74c3c" if j in top_k_wt else "#e67e22" for j in channels]
        axes[row, 1].bar(channels, wt_salience, color=colors_wt, width=1.0, edgecolor="none")
        axes[row, 1].set_ylabel("max|weight|")

    axes[0, 0].set_title("Activation channels (vary with σ)")
    axes[0, 1].set_title("Weight channels (fixed)")
    axes[-1, 0].set_xlabel("Channel index")
    axes[-1, 1].set_xlabel("Channel index")
    fig.suptitle(f"{layer_name} — Complementarity across timesteps (PTQ4DiT Fig.1-Left analog)")
    return fig
```

**What to look for:**
- Do the red-highlighted activation channels shift position across the 3 rows? → temporal variability confirmed
- Do the red activation channels overlap with the orange weight channels? If NO → complementarity holds in SD3
- Is ρ consistently low (< 0.3) across all 3 sigma steps? → CSB is viable
- Does Jaccard stay low (< 0.1)? → binary complementarity confirmed

---

### Extended SD3-specific diagnostic plots

---

### 10.4 Per-layer salience histogram (tests H1)

For 3–5 representative layers (one from each family), plot a **histogram / KDE** of per-channel salience values. Show both activation salience and weight salience in the same figure as overlapping distributions (different colors, semi-transparent). Mark the top-k channels with vertical dashed lines.

This complements 10.1 by showing the full *distribution shape* rather than individual bars. Use log-scale x-axis if the distribution spans multiple orders of magnitude.

---

### 10.5 Heatmap: channel salience × sigma step (tests H3)

For each representative layer, create a **2D heatmap** with:
- x-axis: channel index (sorted by mean salience across all sigma steps, most salient on the right)
- y-axis: sigma step (top row = σ ≈ 1.0 / early denoising, bottom row = σ ≈ 0.0 / late denoising)
- Color: `log₁₀(s(X_j^σ))`, using a sequential colormap (e.g., `viridis` or `inferno`)

Draw a dashed horizontal line at any sigma step where a "regime shift" is detected (abrupt change in the distribution). Annotate the colorbar with the global min and max log-salience.

This is the most information-dense temporal plot — it shows at a glance whether salience is confined to a stable set of channels (vertical bright stripes) or whether it migrates (diagonal or shifting bright patches).

Produce one heatmap per representative layer. Additionally, produce a **small-multiples grid** (4×6) showing all 24 blocks' image-side `q_proj` heatmaps to enable quick visual comparison across block depth.

---

### 10.6 Layerwise Spearman ρ bar plot (tests H2)

**Single wide bar chart** with one bar per linear layer (~287 layers), sorted left-to-right by block index and grouped by family:
- y-axis: mean Spearman ρ(s_act, s_wt) averaged across all sigma steps
- Color-code by family: `q_proj` blue, `k_proj` teal, `v_proj` purple, `o_proj` grey, `fc1` orange, `fc2` red, `context` green, `final` black
- Separate the image-side layers (top sub-plot) from text-side layers (bottom sub-plot) for direct comparison

Draw a horizontal reference line at ρ = 0 (ideal complementarity) and ρ = 0.5 (borderline). Layers with ρ > 0.5 are candidates where CSB may be less effective.

---

### 10.7 Temporal ρ trajectory per layer (tests H3 + H2)

For each representative layer, plot **ρ(σ) as a line** (x-axis = sigma step, y-axis = Spearman ρ) on a primary axis. On the secondary axis, overlay the implied SSC weight `η_t = exp(−ρ_t) / Σ exp(−ρ_τ)` as a filled area or bar.

Produce a **small-multiples grid** (6×4) covering all 24 blocks for a single family (e.g., `q_proj` image-side). This reveals whether the ρ trajectory shape is consistent across depth or changes (e.g., early blocks might have flat ρ, late blocks oscillatory).

---

### 10.8 Image-side vs text-side paired scatter (tests H5)

For each block index `b` (0–23) and each family `f`, extract a scalar summary metric from both the image-side and text-side layer. Produce **three scatter plots** (one per metric):

1. **Mean ρ**: `mean_ρ_image(b, f)` vs `mean_ρ_text(b, f)` — complementarity symmetry
2. **Mean CoV**: `mean_CoV_image(b, f)` vs `mean_CoV_text(b, f)` — temporal stability symmetry
3. **Max salience**: `max_salience_image(b, f)` vs `max_salience_text(b, f)` — magnitude symmetry

Each scatter plot has x = image metric, y = text metric, one point per (block, family), colored by block depth (cool→warm colormap). Draw the y = x diagonal. Points far from the diagonal indicate strong modality asymmetry. Annotate outlier blocks.

---

### 10.9 Submodule family violin plot (tests H4)

**Three-panel violin plot** (one per metric: salience, ρ, CoV), each panel with violins grouped by family `{q_proj, k_proj, v_proj, o_proj, fc1, fc2}`:
- Within each violin, split by modality (left half = image, right half = text)
- Overlay individual data points as strip jitter
- Mark median and quartiles

This shows at a glance which submodule families are the most problematic. E.g., if `fc1` consistently has the highest salience and lowest ρ across all blocks, it is the top quantization risk regardless of block depth.

---

### 10.10 Block depth vs salience profile (tests H4)

**Line plot** with x-axis = block index (0–23), y-axis = metric. Plot separate lines for:
- Image-side mean salience (solid blue)
- Text-side mean salience (solid orange)
- Image-side max salience (dashed blue)
- Text-side max salience (dashed orange)

Add a secondary y-axis for Spearman ρ (dotted lines). Produce one figure per family or an overlaid multi-family version.

This reveals depth-dependent risk: e.g., late blocks may have higher salience due to accumulated signal, or early blocks may have more temporal variation.

---

### 10.11 Top-k overlap heatmap across sigma steps (tests H3)

For a representative layer, produce a **symmetric `num_steps × num_steps` heatmap** where cell `(i, j)` contains the Jaccard overlap between top-k activation channel sets at sigma steps `i` and `j`:

```
J(i,j) = |top_k(σ_i) ∩ top_k(σ_j)| / |top_k(σ_i) ∪ top_k(σ_j)|
```

Use a diverging colormap (`RdYlGn`): green = high overlap (stable channels), red = low overlap (shifting channels). Annotate the diagonal blocks to show whether temporal stability clusters into "phases" (e.g., early σ steps form one cluster, late σ steps another).

Produce this for at least 3 layers (one from q_proj, fc1, final_layer) to compare stability patterns across families.

---

### 10.12 Activation salience vs weight salience scatter (tests H2)

For a representative layer at a representative sigma step (σ ≈ 0.5), produce a **scatter plot** with:
- x-axis: `s(X_j)` (activation channel salience)
- y-axis: `s(W_j)` (weight channel salience)
- One point per channel `j`, colored by channel index (sequential colormap)
- Mark the top-k salient activation channels with a red ring and top-k salient weight channels with an orange ring

If complementarity holds, the plot should show an **L-shaped** or dispersed pattern — channels with high activation salience tend to have low weight salience and vice versa. If it's clustered along the diagonal, complementarity is weak.

Log-scale both axes for better visibility. Annotate the Spearman ρ in the corner.

Produce one scatter per representative layer, and optionally a **4-panel grid** showing the same layer at 4 different sigma steps to reveal temporal changes in the scatter pattern.

---

### 10.13 Salience rank stability ribbon plot (tests H3)

For a representative layer, select the top-k = 16 salient channels at the first sigma step. Track their **rank** across all sigma steps. Produce a **ribbon/spaghetti plot**:
- x-axis: sigma step
- y-axis: rank (1 = most salient, d_in = least salient), inverted axis
- One line per tracked channel, colored distinctly

If lines remain near the top throughout → stable salient channels. If lines cross and diverge → salient identity shifts over time. This is a more granular version of the Jaccard heatmap (10.11).

---

### 10.14 Quantization sensitivity ranking (tests H4 + risk ranking)

Produce a **horizontal bar chart** ranking all linear layers by a composite quantization risk score:

```python
risk_score = (
    0.4 * normalized_max_salience_act
    + 0.2 * normalized_max_salience_wt
    + 0.2 * normalized_mean_rho           # higher ρ = more risk (CSB less effective)
    + 0.2 * normalized_cov_temporal
)
```

Sort bars from highest risk (top) to lowest risk (bottom). Color by family. Annotate the top 20 highest-risk layers with their full names.

This produces the **final risk ranking** used to prioritize quantization attention in Phase 2.

---

### 10.15 CFG conditioning regime comparison (tests H6 / velocity prediction)

If CFG data is collected (both conditioned and unconditioned forward passes), produce:

1. **Paired boxplot**: For a representative layer, show side-by-side boxes for the conditioned batch half vs the unconditioned batch half at 5 sigma steps. y-axis = distribution of per-channel salience. This reveals whether conditioning shifts the activation distribution substantially.

2. **Ratio scatter**: x-axis = `s_conditioned(X_j)`, y-axis = `s_unconditioned(X_j)`. One point per channel. If the ratio is consistently ~1.0 (points on diagonal), CFG doesn't split the distribution. Large deviations indicate that separate calibration may be needed.

---

### 10.16 Final layer analysis (tests H6 / velocity prediction)

Dedicated figure for `final_layer.linear`, since SD3 predicts velocity `v = noise - image` rather than noise `ε`:

1. **Per-channel output magnitude profile**: Bar chart of per-output-channel max absolute values from the final linear layer. Compare this to a mid-block layer to show whether the final layer has a different salience shape.
2. **Temporal boxplot** (same as 10.2) specifically for the final layer, to check if velocity prediction creates different temporal dynamics than noise prediction would.
3. **Activation distribution histogram**: Overlay the activation distribution at 3 sigma steps (early, mid, late) as KDE curves. At high σ, the velocity target is dominated by noise; at low σ, it is dominated by −image. Check whether this creates bimodal or shifting distributions in the activations feeding the final layer.

---

### 10.17 Global summary dashboard (single-page overview)

A **single-page figure** with a 2×3 grid of subplots providing a high-level overview of all findings:

| Position | Content |
|---|---|
| (0,0) | Block-depth profile of mean activation salience (lines, image vs text) |
| (0,1) | Block-depth profile of mean Spearman ρ (lines, image vs text) |
| (0,2) | Block-depth profile of mean temporal CoV (lines, image vs text) |
| (1,0) | Family violin of activation salience |
| (1,1) | Family violin of Spearman ρ |
| (1,2) | Histogram of risk scores across all layers |

This is the "executive summary" figure that appears at the top of the report.

---

### 10.18 Summary diagnostic table

One row per layer, exported as both CSV and a rendered table in the notebook/report:

| Column | Description |
|---|---|
| `layer_name` | e.g. `blocks.12.image.attn.q_proj` |
| `family` | q_proj / k_proj / v_proj / o_proj / fc1 / fc2 / context / final |
| `side` | image / text / shared |
| `block` | 0–23 or -1 |
| `d_in` | Number of input channels |
| `mean_act_salience` | Mean of `s(X_j)` over channels, averaged across sigma steps |
| `max_act_salience` | Max of `s(X_j)` over channels, worst across sigma steps |
| `mean_wt_salience` | Mean of `s(W_j)` over channels |
| `max_wt_salience` | Max of `s(W_j)` over channels |
| `top1_median_ratio_act` | Max channel / median channel (activation), averaged across sigma steps |
| `top1_median_ratio_wt` | Max channel / median channel (weight) |
| `gini_act` | Gini coefficient of activation salience, averaged across sigma steps |
| `gini_wt` | Gini coefficient of weight salience |
| `mean_spearman_rho` | Mean Spearman ρ(s_act, s_wt) across sigma steps |
| `std_spearman_rho` | Std dev of ρ across sigma steps |
| `min_spearman_rho` | Minimum ρ across sigma steps (best complementarity) |
| `max_spearman_rho` | Maximum ρ across sigma steps (worst complementarity) |
| `mean_jaccard_topk` | Mean Jaccard overlap of top-k act vs wt channels |
| `cov_temporal` | Mean CoV of per-channel salience across sigma |
| `early_late_topk_jaccard` | Jaccard of top-k channels at first vs last sigma step |
| `risk_score` | Composite quantization risk score (see 10.14) |

Sort the table by `risk_score` descending in the report. Additionally produce a **"Top 20 hardest layers"** extract for quick reference.

---

## 11. Practical Implementation Notes

### 11.1 Inference configuration

```python
DIAG_CONFIG = {
    "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
    "w16": True,
    "shift": 1.0,
    "use_t5": True,
    "low_memory_mode": False,  # need model to stay in memory for hooks
    "num_steps": 28,
    "cfg_weight": 0.0,         # disable CFG for clean single-batch statistics
    "latent_size": (128, 128), # 1024×1024 output
    "seed_range": range(42, 50),
}
```

**`low_memory_mode` must be False** for diagnostics. In low memory mode, DiffusionKit deletes the MMDiT after denoising, and unloads text encoders after encoding. We need the model to persist for hook attachment and multiple runs.

### 11.2 Prompt corpus

```python
DIAGNOSTIC_PROMPTS = [
    "a red cube on a white table",
    "a Victorian library with dust motes in golden afternoon light, leather-bound books, and a sleeping cat on a velvet armchair",
    "a neon sign reading OPEN 24 HOURS against a dark alley wall",
    "three blue spheres and two yellow cones arranged on a checkerboard floor",
    "an oil painting in the style of Vermeer depicting a woman reading a letter",
    "a bustling Tokyo intersection at night with crowds, taxis, and neon signs",
    "portrait of an elderly woman smiling, soft studio lighting",
    "entropy and order in visual tension, abstract geometric composition",
    "a single photorealistic water droplet on a leaf, macro photography",
    "a medieval castle on a cliff overlooking a stormy sea at sunset",
    "a child's drawing of a house with a sun and clouds, crayon on paper",
    "an astronaut riding a horse on the surface of Mars, cinematic",
    "a bowl of ramen with steam rising, overhead view, food photography",
    "blueprint technical drawing of a spacecraft with annotations",
    "a field of sunflowers under dramatic cumulonimbus clouds",
    "two cats sitting symmetrically on a windowsill, silhouette against sunset",
    "a dense jungle with a hidden ancient temple, volumetric light rays",
    "minimalist flat vector illustration of a coffee cup",
    "a crowded bookshelf seen through a magnifying glass, tilt-shift effect",
    "the word HELLO written in fire against a black background",
]
```

### 11.3 Running the collection loop

```python
def run_diagnostic_collection(pipeline, prompts, seeds, collector):
    for prompt_id, prompt in enumerate(prompts):
        conditioning, pooled_conditioning = pipeline.encode_text(
            prompt, cfg_weight=0.0
        )
        mx.eval(conditioning, pooled_conditioning)

        for seed in seeds:
            mx.random.seed(seed)
            x_T = pipeline.get_empty_latent(128, 128)
            noise = pipeline.get_noise(seed, x_T)
            sigmas = pipeline.get_sigmas(pipeline.sampler, DIAG_CONFIG["num_steps"])
            noise_scaled = pipeline.sampler.noise_scaling(
                sigmas[0], noise, x_T, True
            )

            timesteps = pipeline.sampler.timestep(sigmas).astype(
                pipeline.activation_dtype
            )

            # Cache modulation params (needed for forward pass)
            pipeline.mmdit.cache_modulation_params(
                pooled_conditioning, timesteps
            )

            x = noise_scaled.astype(pipeline.activation_dtype)
            mmdit = pipeline.mmdit

            for i in range(len(sigmas) - 1):
                sigma_val = float(sigmas[i].item())
                collector.set_context(
                    step_idx=i, sigma=sigma_val,
                    prompt_id=str(prompt_id), seed=seed,
                )

                token_text = mx.expand_dims(conditioning, 2)
                ts = mx.broadcast_to(timesteps[i], [1])

                mmdit_output = mmdit(
                    latent_image_embeddings=x,
                    token_level_text_embeddings=token_text,
                    timestep=ts,
                )

                denoised = pipeline.sampler.calculate_denoised(
                    sigmas[i], mmdit_output, x
                )
                d = (x - denoised) / sigmas[i]
                x = x + d * (sigmas[i + 1] - sigmas[i])
                mx.eval(x)

            pipeline.mmdit.clear_modulation_params_cache()
```

### 11.4 Storage format

Save collected data as compressed numpy archives:

```
diagnostics/
├── config.json                    # inference settings, prompt list, seeds
├── weight_stats.npz               # {layer_name: {w_channel_max, w_channel_mean}}
├── activation_stats/
│   ├── blocks.0.image.attn.q_proj.npz
│   ├── blocks.0.image.attn.k_proj.npz
│   ├── ...
│   └── final_layer.linear.npz
└── adaln_stats.npz                # modulation parameter statistics
```

Each activation stats file contains:
```python
np.savez_compressed(path,
    sigma_values=np.array([...]),           # [num_steps]
    # shape [num_prompts * num_seeds, num_steps, d_in]:
    act_channel_max=stacked_maxes,
    act_channel_mean=stacked_means,
    prompt_ids=np.array([...]),
    seeds=np.array([...]),
)
```

### 11.5 Memory management

- The full SD3 2b model uses ~4–5 GB in fp16.
- Per-channel statistics for one layer at one step: ~6 KB (1536 channels × 4 bytes).
- For 287 layers × 28 steps × 20 prompts × 8 seeds = ~287 × 28 × 160 = ~1.3M records.
- At ~6 KB each = ~7.5 GB raw. With compression this drops to ~1–2 GB.
- To reduce, aggregate across prompts on-the-fly (keep elementwise max and running mean/variance) rather than storing per-prompt data for every layer. This reduces storage by 160×.

### 11.6 Pilot run

Before the full sweep, run a pilot with:
- 2 prompts, 1 seed, all 28 steps
- Verify all expected layers fire (count hook calls = 287 layers × 28 steps × 2 prompts = 16,072 calls)
- Verify shapes match expectations (check `d_in` values)
- Verify `q_proj`, `k_proj`, `v_proj` produce identical `act_channel_max` within a block
- Produce one salience histogram to sanity-check magnitudes

---

## 12. Expected Deliverables

### 12.1 Data artifacts

1. **`weight_stats.npz`** — per-channel weight salience for all ~287 layers
2. **`activation_stats/`** — per-channel activation salience per layer per sigma step, aggregated across prompts
3. **`adaln_stats.npz`** — per-channel modulation parameter magnitudes across sigma steps
4. **`config.json`** — full reproducibility metadata

### 12.2 Analysis notebook

A Jupyter notebook (`src/phase1/analyze.ipynb`) that loads the collected data and produces all plots from Section 10.

### 12.3 Summary report

A markdown file (`src/phase1/report.md`) containing:

1. **Per-hypothesis verdict** for each of H1–H6 (accept / reject / partial, with evidence)
2. **Risk-ranked layer table** (Section 10.18) sorted by quantization difficulty
3. **Submodule family summary** — which families are hard, which are safe
4. **Modality comparison** — image vs text branch findings
5. **Temporal dynamics summary** — does the rectified flow trajectory show the same temporal variation patterns as DDPM in DiT-XL?
6. **Recommendation** — go / no-go for PTQ4DiT-style CSB+SSC, with any necessary adaptations identified
