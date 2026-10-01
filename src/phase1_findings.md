# Phase 1 Diagnostic Findings — SD3 Medium MMDiT

**Collection setup:** 100 COCO-style prompt–seed pairs (`src/settings/coco_100_calibration_prompts.txt`), 30 Euler steps, CFG 4.0, fp16 weights, fp32 activation statistics. 512×512 images (latent 64×64). **287** `nn.Linear` layers hooked for activation/weight stats; **adaLN** modulation statistics are collected separately (49 adaLN records: 24 image + 24 text blocks + final layer). Sigma schedule: 1.0 → 0.001 (monotonically decreasing, rectified flow). **Pipeline:** `python -m src.phase1.run_collection` → `diagnostics/`.

---

## 1. Salient Channels Exist in SD3 Medium (H1 — Confirmed)

**PTQ4DiT claim:** A small number of channels carry disproportionately large magnitudes, and these cause most of the quantization error.

**SD3 finding: Confirmed.** Salient channels are present in every layer family.

- The `context_embedder` has by far the highest max activation salience at **854.0** — about 2× the next-highest layer (`blocks.22.text.mlp.fc2` at 414) and roughly 40× the mean max activation salience across all layers (21.1). Its top-1/median ratio is 832 and its Gini coefficient is 0.62, confirming extreme concentration.
- Text-side late-block MLP layers show the next tier of extreme salience: `blocks.22.text.mlp.fc1` (349), `blocks.14.text.mlp.fc2` (252), `blocks.21.text.mlp.fc1` (247), `blocks.21.text.mlp.fc2` (184), `blocks.20.text.mlp.fc2` (179). These are all 8–17× the mean.
- In **fig3_\*.png** plots, per-channel max|activation| bars show blue for most channels and red for the top-32. A green MSE line on the secondary axis spikes at exactly the same salient channels, directly confirming that salient channels drive disproportionate quantization error.
- The **family_violins.png** (left panel) shows that text `fc2` layers have extreme outliers reaching 400+, while attention projections (q/k/v/o_proj) are more moderate (mostly under 30). The median max salience across all layers is 10.6.

**What differs from PTQ4DiT (DiT-XL):** SD3 has the `context_embedder` (T5-XXL 4096→1536 projection) with an extraordinarily high activation salience that has no analogue in DiT-XL.

---

## 2. Activation–Weight Complementarity Is Strong (H2 — Confirmed)

**PTQ4DiT claim:** Channels with high activation salience tend to have low weight salience, and vice versa (low Spearman ρ). This complementarity is what makes Channel-wise Salience Balancing (CSB) effective.

**SD3 finding: Confirmed, and in many layers even stronger than in DiT-XL.**

- **80.5% of layers** (231/287) have mean Spearman ρ < 0.3, indicating strong complementarity.
- **30.3% of layers** (87/287) have mean ρ < 0.0, meaning activation and weight salience are anti-correlated — the ideal case for CSB.
- Only **5.9% of layers** (17/287) have mean ρ > 0.5, where CSB may be less effective.
- Mean ρ across all layers: **0.121** (std 0.214). Median: **0.098**.

**Per-family ρ:**

| Family | Layers | Mean ρ | Best for CSB? |
|--------|--------|--------|---------------|
| o_proj | 47 | 0.033 | Strongest complementarity |
| fc2 | 47 | 0.081 | Strong |
| q_proj | 48 | 0.112 | Strong |
| k_proj | 48 | 0.148 | Strong |
| fc1 | 47 | 0.160 | Moderate |
| v_proj | 48 | 0.191 | Moderate |
| final_linear | 1 | 0.673 | Weak — CSB less effective |
| context_embedder | 1 | −0.342 | Ideal (anti-correlated) |

**Depth-dependent ρ profile:**

| Block range | Layers | Mean ρ |
|-------------|--------|--------|
| 0–3 | 48 | 0.334 |
| 4–7 | 48 | 0.125 |
| 8–11 | 48 | 0.041 |
| 12–15 | 48 | 0.040 |
| 16–19 | 48 | 0.100 |
| 20–23 | 45 | 0.083 |

The **layerwise_rho.png** bar chart shows this clearly: early blocks (0–3) have higher ρ (0.3–0.7), while mid blocks (8–15) hover near zero or go negative. Late blocks (20–23) are bimodal — text-side MLP layers have very high ρ (0.6–0.8) while image-side layers remain low.

The **scatter_\*.png** plots illustrate complementarity visually. For mid/late blocks with low ρ, top-k activation channels cluster at the right (high activation, low weight) while top-k weight channels cluster at the top — the classic L-shaped separation. In contrast, early blocks like `blocks.0.image.attn.q_proj` (ρ = 0.53) show overlap.

**What differs from PTQ4DiT:** The depth-dependent ρ profile is a new finding. PTQ4DiT reports generally low ρ across DiT-XL layers without highlighting this early-high/late-low gradient.

---

## 3. Temporal Variation of Activation Salience (H3 — Confirmed with Nuances)

**PTQ4DiT claim:** The distribution and identity of salient activation channels changes across timesteps, motivating time-aware calibration (SSC).

**SD3 finding: Confirmed.** Temporal variation is present but heterogeneous — moderate in most layers, extreme in a subset.

- The mean temporal CoV across all layers is **0.171** (std 0.054, median 0.149). Most channels' salience values fluctuate by roughly 17% of their mean across the σ trajectory.
- The mean early-late top-32 Jaccard is **0.255** (std 0.184, median 0.231), meaning only about 25% of the top-32 salient channels at σ=1.0 are still in the top-32 at σ=0.001.
- **11 layers have Jaccard = 0.0** (complete top-k turnover between early and late σ). These are predominantly image-side `fc2` and `o_proj` layers.
- **fig4_\*.png** temporal boxplots show salience generally increasing as σ decreases (noise is removed).
- The **heatmap_grid_image_q_proj.png** (4×6 grid) shows vertical bright stripes (persistent salient channels) with some intensity variation across σ rows. Blocks 0–3 show more pronounced changes.

**The final_layer.linear is a critical outlier:**

- CoV = **0.397** (highest of all 287 layers), more than 2× the mean.
- Early-late Jaccard = **0.103** (one of the lowest) — near-complete turnover of salient channels.
- ρ = **0.673** — high positive correlation means CSB is less effective here.
- The **topk_overlap_final_layer.linear.png** Jaccard heatmap shows a striking block-diagonal pattern: early σ steps form one cluster, late σ steps form another, with cross-regime Jaccard near 0.

**Key layer temporal comparisons:**

| Layer | CoV | Early-late Jaccard | Pattern |
|-------|-----|-------------------|---------|
| `final_layer.linear` | 0.397 | 0.103 | Regime shift |
| `blocks.2.image.mlp.fc1` | 0.303 | 0.049 | High turnover |
| `blocks.0.text.attn.q_proj` | 0.286 | 0.280 | Moderate |
| `blocks.12.image.attn.q_proj` | 0.127 | 0.641 | Mostly stable |
| `context_embedder` | 0.000 | 1.000 | Perfectly stable |

**What differs from PTQ4DiT:** The final layer's extreme temporal variation is likely unique to SD3's velocity prediction (v = noise − image) rather than DiT-XL's noise prediction (ε). At high σ, the velocity target is noise-dominated; at low σ, it is image-dominated. Additionally, 11 layers having Jaccard = 0.0 (complete turnover) suggests stronger regime shifts in SD3's MMDiT than DiT-XL's single-stream architecture.

---

## 4. Modality Asymmetry Is Significant (H5 — New Finding for SD3)

**PTQ4DiT context:** DiT-XL has no modality split. SD3's MMDiT processes image and text tokens through separate pathways.

**SD3 finding: Image and text pathways behave differently.**

| Metric | Image (145 layers) | Text (141 layers) |
|--------|-------------------|-------------------|
| Mean ρ | 0.084 | 0.163 |
| Mean CoV | 0.174 | 0.168 |
| Mean max salience | 12.1 | 24.4 |
| Median max salience | 10.4 | 10.6 |

- **Salience magnitude:** Text-side layers have 2× higher mean max activation salience, driven entirely by late-block MLP outliers. Median values are comparable (10.4 vs 10.6), showing the difference is concentrated in a few extreme layers.
- **Complementarity:** Text-side has higher mean ρ (0.163 vs 0.084), meaning weaker complementarity. This is driven by text-side fc1 (ρ = 0.325) vs image-side fc1 (ρ = 0.002), the largest modality gap of any family.
- **Temporal variation:** Image-side CoV (0.174) is slightly higher than text-side (0.168) — approximately comparable.
- The **summary_dashboard.png** top-left panel shows text-side mean act salience spiking in blocks 20–23 while image-side remains flat.
- The **family_violins.png** middle panel shows image-side attention projections (q/k/v) have higher spread in ρ (spanning −0.4 to +0.7) compared to text-side equivalents.

**Per-family modality split:**

| Family | Image mean ρ | Text mean ρ | Image mean sal | Text mean sal |
|--------|-------------|------------|----------------|---------------|
| q_proj | 0.116 | 0.107 | 16.4 | 9.4 |
| k_proj | 0.181 | 0.115 | 16.4 | 9.4 |
| v_proj | 0.157 | 0.224 | 16.4 | 9.4 |
| o_proj | −0.028 | 0.096 | 9.4 | 9.3 |
| fc1 | 0.002 | 0.325 | 7.3 | 48.9 |
| fc2 | 0.050 | 0.113 | 6.3 | 62.2 |

The most striking asymmetry is in MLP layers: text-side fc1 and fc2 have far higher salience (~7–10× image-side) and substantially higher ρ (especially fc1: 0.325 vs 0.002).

**Implication:** CSB effectiveness varies by modality. Image-side layers are generally more amenable to CSB (lower ρ) than text-side layers.

---

## 5. Submodule Family Risk Ranking (H4)

**SD3 finding:** Risk is concentrated in specific families and specific blocks. The composite risk score combines normalized max activation salience (40%), max weight salience (20%), mean Spearman ρ (20%), and temporal CoV (20%).

Top 10 highest-risk layers by composite risk score:

| Rank | Layer | Risk | Max Sal. | ρ | CoV |
|------|-------|------|----------|------|------|
| 1 | `blocks.22.text.mlp.fc2` | 0.535 | 414.4 | 0.728 | 0.247 |
| 2 | `blocks.22.text.mlp.fc1` | 0.463 | 349.0 | 0.791 | 0.098 |
| 3 | `context_embedder` | 0.453 | 854.0 | −0.342 | 0.000 |
| 4 | `blocks.1.image.attn.q_proj` | 0.412 | 12.7 | 0.704 | 0.282 |
| 5 | `final_layer.linear` | 0.410 | 23.6 | 0.673 | 0.397 |
| 6 | `blocks.21.text.mlp.fc2` | 0.410 | 183.7 | 0.611 | 0.274 |
| 7 | `blocks.14.text.mlp.fc2` | 0.407 | 252.2 | 0.108 | 0.161 |
| 8 | `blocks.21.text.mlp.fc1` | 0.387 | 246.9 | 0.645 | 0.113 |
| 9 | `blocks.1.image.attn.v_proj` | 0.386 | 12.7 | 0.680 | 0.282 |
| 10 | `blocks.14.image.attn.k_proj` | 0.386 | 21.5 | 0.253 | 0.133 |

**Key patterns from family_violins.png:**

- **fc2** has the widest spread and highest outliers in activation salience (text-side reaching 400+). It also has the highest mean temporal CoV.
- **o_proj** has the lowest mean ρ (0.033) — strongest complementarity overall.
- **Attention projections** (q/k/v) share the same input within a block — identical activation salience but different weight salience, so ρ differs. Note: `blocks.1.image.attn.{q,v}_proj` both appear in the top 10 because early blocks have high ρ combined with moderate salience.

**Layers with strongest complementarity (best CSB targets):**

| Layer | ρ | Max Sal. |
|-------|------|----------|
| `blocks.22.image.attn.o_proj` | −0.403 | 15.3 |
| `context_embedder` | −0.342 | 854.0 |
| `blocks.11.image.mlp.fc1` | −0.324 | 10.3 |
| `blocks.22.text.attn.o_proj` | −0.306 | 9.7 |
| `blocks.10.image.mlp.fc1` | −0.298 | 8.4 |

---

## 6. Velocity Prediction and the Final Layer (H6)

**PTQ4DiT context:** DiT-XL predicts noise ε. SD3 predicts velocity v = noise − image.

**SD3 finding: The final layer is one of the highest-risk quantization targets.**

- `final_layer.linear` maps 1536 dims → 64 (VAE latent channels).
- ρ = 0.673 (one of the highest) — activation and weight salience are positively correlated, meaning CSB is less effective.
- CoV = 0.397 (the highest) — extreme temporal variation.
- Early-late Jaccard = 0.103 — near-complete turnover of salient channels.
- Gini (activation) = 0.69, Gini (weight) = 0.57 — both highly concentrated.
- ρ trajectory spans 0.58–0.81 (std 0.045) — consistently high across all σ steps.

**Implication:** The final layer may require special treatment — potentially higher bit-width or a dedicated per-timestep calibration scheme.

---

## 7. Context Embedder Is an Extreme Outlier

The `context_embedder` (projecting 4096-dim T5-XXL embeddings to 1536-dim hidden) has unique properties:

- **Max activation salience = 854.0** — by far the highest (next is 414 for text fc2).
- **ρ = −0.342** — strong anti-correlation. Ideal for CSB.
- **CoV = 0.000 and early-late Jaccard = 1.000** — text embeddings are fixed per prompt, so there is zero temporal variation. The ρ trajectory has std = 0 (every σ step produces the same rank correlation). The **heatmap_context_embedder.png** shows perfectly horizontal color bands.
- **Gini = 0.62** — high concentration of salience in few channels.

**Implication:** A single calibration point (no time-awareness needed) with aggressive CSB should suffice. Currently excluded from quantization (`exclude_layers: ["context_embedder"]`).

---

## 8. SSC Weight Non-Uniformity: Layer-Dependent, Not Universal

**Critical finding:** The SSC (Spearman-based Sigma Calibration) time-aware weighting is **not uniformly near-flat** across all layers. Its effectiveness depends on the layer's ρ trajectory span — which varies by over an order of magnitude across the network.

### The mechanism

SSC weights are computed as η_t = exp(−ρ_t) / Σ exp(−ρ_τ). Timesteps where activation and weight salience are anti-correlated (low/negative ρ) receive higher weight, since those are the steps where the activation statistics are most informative for calibration.

### The ρ trajectory spans

| Metric | Value |
|--------|-------|
| Median ρ range (max−min) | 0.125 |
| Mean ρ range | 0.150 |
| 75th percentile range | 0.187 |
| Max ρ range | 0.592 |
| Layers with range > 0.3 | 24 / 287 (8.4%) |
| Layers with range > 0.2 | 63 / 287 (22.0%) |
| Layers with range < 0.1 | 102 / 287 (35.5%) |

### What this means for SSC weights

The softmax over small ρ differences produces mild rather than dramatic weighting. Even in the worst case, the effective number of timesteps remains close to 30:

| Layer | ρ range | max η / min η | eff_T / 30 | Character |
|-------|---------|---------------|-----------|-----------|
| `blocks.12.image.mlp.fc1` | 0.586 | 1.80 | 29.2 / 30 | Strongest differentiation |
| `blocks.8.image.attn.q_proj` | 0.539 | 1.71 | 29.4 / 30 | Significant |
| `blocks.12.text.attn.q_proj` | 0.278 | 1.32 | 29.8 / 30 | Moderate |
| `final_layer.linear` | 0.221 | 1.25 | 29.9 / 30 | Mild |
| `blocks.0.image.attn.q_proj` | 0.104 | 1.11 | 30.0 / 30 | Near-uniform |
| `context_embedder` | 0.000 | 1.00 | 30.0 / 30 | Perfectly uniform |

### Distribution across the network

- **30% of layers** (87/287): near-uniform, ratio < 1.1. SSC adds essentially nothing over simple averaging.
- **57% of layers** (164/287): mild variation, ratio 1.1–1.3.
- **9% of layers** (26/287): moderate, ratio 1.3–1.5.
- **3.5% of layers** (10/287): significant, ratio ≥ 1.5. All are image-side MLPs and some image-side q_proj.

### By layer family (mean max η / min η ratio)

| Family | Mean ratio | Max ratio | Most affected |
|--------|-----------|-----------|---------------|
| fc1 | 1.22 | 1.80 | `blocks.12.image.mlp.fc1` |
| fc2 | 1.18 | 1.59 | `blocks.3.image.mlp.fc2` |
| final_linear | 1.25 | 1.25 | Only layer |
| q_proj | 1.18 | 1.71 | `blocks.8.image.attn.q_proj` |
| o_proj | 1.16 | 1.46 | `blocks.6.image.attn.o_proj` |
| k_proj | 1.14 | 1.49 | `blocks.14.image.attn.k_proj` |
| v_proj | 1.15 | 1.45 | `blocks.23.image.attn.v_proj` |
| context_embedder | 1.00 | 1.00 | N/A |

### Detailed example: blocks.12.image.mlp.fc1 (worst case)

This layer has ρ going from −0.30 (at σ ≈ 0.8, high noise) to +0.29 (at σ ≈ 0.001, low noise). SSC upweights the high-noise steps by ~14% above uniform and downweights the low-noise steps by ~37% below uniform:

- Heaviest 3 η values: 0.038, 0.038, 0.038 (σ ≈ 0.83, 0.86, 0.76) — 1.14× uniform
- Lightest 3 η values: 0.021, 0.021, 0.022 (σ ≈ 0.001, 0.035, 0.070) — 0.64× uniform
- 50% of total weight is carried by the top 14/30 timesteps (vs 15/30 for uniform)

### Root cause

Spearman ρ between activation and weight salience is bounded to [−1, +1] and in SD3 Medium rarely exceeds ±0.4 within any single layer's trajectory. The `exp(−ρ)` softmax with inputs in this narrow range cannot produce a peaked distribution. For SSC to create truly non-uniform weights, ρ would need to span several units — but rank correlations are inherently constrained.

### Implications

SSC does real work for **~13% of layers** (image-side MLPs and some q_proj in mid-blocks) where it meaningfully upweights high-noise timesteps with η ratios of 1.3–1.8×. For the majority of the network it produces near-uniform weights — but this is itself a principled conclusion: those layers have stable activation-weight relationships across σ, so uniform averaging is the correct calibration strategy.

A sharper weighting function (e.g., temperature-scaled softmax `exp(−ρ/τ)` with τ < 1) could amplify the small ρ differences for the ~13% of layers where differentiation exists.

---

## 9. Summary: Similarities and Differences vs PTQ4DiT (DiT-XL)

### What is similar (PTQ4DiT observations that hold in SD3)

1. **Salient channels exist** — confirmed across all families. A small number of channels carry disproportionately large magnitudes (Gini up to 0.71).
2. **Complementarity holds** — 80.5% of layers have ρ < 0.3, supporting CSB applicability.
3. **Temporal variation exists** — top-k channel identity shifts across σ steps (mean early-late Jaccard = 0.255; 11 layers have complete turnover).
4. **MSE tracks salience** — quantization error per channel correlates with channel magnitude (confirmed in fig3 plots).

### What differs (SD3-specific observations not present in DiT-XL)

1. **Modality asymmetry** — SD3's dual image/text pathways differ in salience (text MLP ~7–10× image MLP), complementarity (text fc1 ρ = 0.325 vs image fc1 ρ = 0.002), and risk profile.
2. **Depth-dependent ρ profile** — early blocks (0–3): mean ρ = 0.334; mid blocks (8–15): mean ρ = 0.040. PTQ4DiT does not report a strong depth gradient.
3. **Final layer regime shift** — velocity-prediction final layer shows near-complete top-k turnover (Jaccard = 0.103) with the highest CoV (0.397) and high ρ (0.673).
4. **Context embedder extreme outlier** — T5-XXL projection has 854.0 max salience (~40× the layer average), with zero temporal variation and strong anti-correlation (ρ = −0.342).
5. **Text-side MLP extreme salience** — late-block text fc1/fc2 layers reach 127–414 max salience. Most are paired with high ρ (0.6–0.8), making CSB less effective, though `blocks.20.text.mlp.fc2` is a notable exception (ρ = 0.13).
6. **SSC non-uniformity is layer-dependent** — SSC provides meaningful (~1.3–1.8× ratio) differentiation for ~13% of layers (image-side MLPs, mid-block q_proj) but is near-uniform for the remaining ~87%.
7. **Complete top-k turnover in 11 layers** — where Jaccard = 0.0, suggesting stronger regime shifts than DiT-XL's architecture.

### Implications for Phase 2

- **CSB is broadly applicable** but effectiveness varies by depth (early blocks weaker), family (o_proj strongest), and modality (image-side more amenable).
- **SSC provides meaningful time-awareness for ~13% of layers** (image-side MLPs, some mid-block q_proj) where ρ trajectories span > 0.3. For the majority, it correctly produces near-uniform weights. A temperature parameter could amplify differentiation where it exists.
- **The final layer** (ρ = 0.673, CoV = 0.397, regime shift) should use higher bit-width (8 or 16).
- **Late text-side blocks** (20–23, fc1/fc2 with 127–414 salience, most with ρ > 0.6) are the highest-risk quantization targets where CSB may be insufficient.
- **The context embedder** should remain excluded or use dedicated single-point calibration.

---

*Data: `diagnostics/activation_stats/` (287 layers × 30 σ steps, per-channel max and mean aggregated over 100 prompts), `diagnostics/weight_stats.npz` (287 layers, per-channel max), `diagnostics/adaln_stats.npz` (49 adaLN layers × 31 timesteps). Summary: `diagnostics/summary_table.csv` (287 rows × 21 diagnostic columns). Plots: `diagnostics/plots/` (128 files). Collection config: `diagnostics/config.json` (100 seed-prompt pairs, 30 steps, CFG 4.0).*
