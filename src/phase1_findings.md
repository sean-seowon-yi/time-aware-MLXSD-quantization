# Phase 1 Diagnostic Findings — SD3 Medium MMDiT

**Collection setup:** 10 prompts × 4 seeds = 40 runs, 28 Euler steps each, CFG disabled (cfg_weight=0.0), fp16 weights, fp32 activations. 287 linear layers + 49 adaLN layers instrumented.

---

## 1. Salient Channels Exist in SD3 Medium (H1 — Confirmed)

**PTQ4DiT claim:** A small number of channels carry disproportionately large magnitudes, and these cause most of the quantization error.

**SD3 finding: Confirmed.** Salient channels are present in every layer family.

- Across all layers, the mean top-1/median activation ratio is high, indicating that the most extreme channel is typically many times larger than the typical channel. For example, `blocks.22.text.mlp.fc1` shows a top-1/median ratio of ~90, and `final_layer.linear` shows ~77 (summary_table.csv).
- The `context_embedder` has by far the highest max activation salience at **854.0** — about 2× the next-highest single layer (`blocks.22.text.mlp.fc2` at 414) and roughly 70× the average max activation salience across all block layers (~12). This extreme outlier is visible as the third-highest bar in `risk_ranking.png`.
- In **fig3_\*.png** plots (e.g. `fig3_blocks.0.image.attn.q_proj.png`), each activation panel shows per-channel max|activation| as bars — blue for most channels, red (#e74c3c) for the top-32. A green line on the secondary axis traces the per-channel quantization MSE, which spikes at exactly the same channels that have high magnitude. The weight panel uses the same layout (orange bars, red for top-32, green MSE line). This directly confirms the PTQ4DiT observation that salient channels drive disproportionate quantization error.
- The **family_violins.png** (left panel, "Max activation salience") shows that `fc2` text-side layers have extreme outliers reaching 400+, while attention projections (q/k/v/o_proj) are more moderate (mostly under 30).

**What differs from PTQ4DiT (DiT-XL):** SD3 has the `context_embedder` with an extraordinarily high activation salience (854.0) that has no analogue in DiT-XL. This layer projects 4096-dimensional T5-XXL text embeddings into the 1536-dimensional hidden space and contains a single extreme channel that dominates all others (Gini = 0.62).

---

## 2. Activation–Weight Complementarity Is Strong (H2 — Confirmed)

**PTQ4DiT claim:** Channels with high activation salience tend to have low weight salience, and vice versa (low Spearman ρ). This complementarity is what makes Channel-wise Salience Balancing (CSB) effective.

**SD3 finding: Confirmed, and in many layers even stronger than in DiT-XL.**

- **80.5% of layers** (231 of 287) have mean Spearman ρ < 0.3, indicating strong complementarity.
- **30.3% of layers** (87 of 287) have mean ρ < 0.0, meaning activation and weight salience are **anti-correlated** — the ideal case for CSB.
- Only **5.9% of layers** (17 of 287) have mean ρ > 0.5, where CSB may be less effective.
- The **layerwise_rho.png** bar chart shows a clear trend: early blocks (0–3) have higher ρ (0.4–0.7), while mid-to-late blocks (8–20) have ρ near zero or negative. This "ρ decay with depth" pattern is visible in the image-side subplot.
- The **scatter_\*.png** plots illustrate complementarity visually. For mid/late blocks with low ρ, such as `scatter_blocks.12.image.attn.o_proj.png` (ρ = −0.193), the plot shows a dispersed cloud where top-k activation channels (red circles) cluster at the right side (high activation, low-to-moderate weight) while top-k weight channels (orange circles) cluster at the top (high weight, low-to-moderate activation) — the classic L-shaped separation. In contrast, early blocks like `scatter_blocks.0.image.attn.q_proj.png` (ρ = 0.550) show much more overlap between top-k act and top-k wt circles in the upper-right region, confirming the depth-dependent ρ gradient.

**What differs from PTQ4DiT:** The depth-dependent ρ profile is a new finding. PTQ4DiT reports generally low ρ across DiT-XL layers without highlighting this early-high/late-low gradient. In SD3, the first ~4 blocks have notably higher ρ (0.4–0.7 range), while blocks 8–20 often have negative ρ. This suggests CSB effectiveness will vary by block depth, which is not a concern flagged for DiT-XL.

---

## 3. Temporal Variation of Activation Salience (H3 — Confirmed with Nuances)

**PTQ4DiT claim:** The distribution and identity of salient activation channels changes across timesteps, motivating time-aware calibration (SSC).

**SD3 finding: Confirmed.** Temporal variation is present but moderate in most layers, with the notable exception of `final_layer.linear`.

- The mean temporal CoV across all layers is **0.171** (std 0.054). This indicates moderate but not extreme variation — most channels' salience values fluctuate by roughly 17% of their mean across the σ trajectory.
- The mean early-late top-k Jaccard is **0.255** (std 0.184), meaning only about 25% of the top-32 salient channels at the first σ step are still in the top-32 at the last step. This confirms that channel identity does shift over time.
- **fig4_\*.png** (temporal boxplots) show the boxes shifting vertically across σ steps. For most block layers, salience increases as σ decreases (noise is removed). The red top-k dots move between steps, confirming identity shifts.
- The **heatmap_grid_image_q_proj.png** (4×6 grid of all 24 blocks' image q_proj heatmaps) is particularly revealing. Most blocks show vertical bright stripes (persistent salient channels) with some variation in intensity across σ rows. Blocks 0–3 show more pronounced stripe changes, while mid blocks (8–15) have more uniform heatmaps.
- The **rho_grid_image_q_proj.png** (ρ trajectory across all 24 blocks) shows that most blocks have relatively flat ρ trajectories — ρ does not change dramatically over σ within a single layer. Blocks 0–3 show a slight downward drift from early to late σ, while blocks 8+ are nearly flat near zero.

**The final_layer.linear is a critical outlier:**
- CoV = **0.397** (highest of all 287 layers), more than 2× the mean.
- Early-late Jaccard = **0.103** (one of the lowest), meaning the salient channels at σ ≈ 1.0 are almost entirely different from those at σ ≈ 0.0.
- The **topk_overlap_final_layer.linear.png** Jaccard heatmap shows a striking block-diagonal pattern with two distinct "regimes": early σ steps (high noise) form one green cluster, late σ steps (low noise) form another, and the cross-regime cells are red/orange (Jaccard ≈ 0.0–0.2). This is the clearest evidence of a regime shift.
- The **final_layer_analysis.png** shows: (1) the weight profile has a few dominant channels among 1536 channels, (2) the temporal boxplot shows a clear upward trend in activation salience as σ decreases (boxes grow taller toward late σ), and (3) the distribution shift panel shows the activation distribution broadening and shifting rightward at low σ.

**Contrast: regular blocks vs final layer (top-k overlap heatmaps):**
- The **topk_overlap_blocks.12.image.attn.q_proj.png** shows a mostly green matrix (Jaccard ≈ 0.4–0.8 throughout), indicating that the top-32 salient channels at any σ step have substantial overlap with those at any other σ step. There is a slight fade toward the off-diagonal corners, but no sharp regime boundary.
- The **topk_overlap_final_layer.linear.png** shows a starkly different pattern: two distinct green clusters (upper-left for early σ, lower-right for late σ) separated by a yellow-to-red transition band (Jaccard ≈ 0.0–0.2), indicating near-complete turnover of salient channels between early and late denoising.

**What differs from PTQ4DiT:** The final layer's extreme temporal variation (CoV = 0.40, regime shift in top-k identity) is likely unique to SD3's velocity prediction (v = noise − image) rather than DiT-XL's noise prediction (ε). At high σ, the velocity target is noise-dominated; at low σ, it is image-dominated. This change in prediction target creates a fundamental shift in what the final layer must output, explaining the regime shift. PTQ4DiT does not report such extreme final-layer behavior for DiT-XL.

---

## 4. Modality Asymmetry Is Significant (H5 — New Finding for SD3)

**PTQ4DiT context:** DiT-XL has no modality split — it processes a single token stream. SD3's MMDiT processes image and text tokens through separate TransformerBlock pathways within each MultiModalTransformerBlock, which is architecturally novel.

**SD3 finding: Image and text pathways behave differently.**

- **Salience magnitude:** Text-side layers have higher mean max activation salience (24.4) than image-side layers (12.1). The extreme outliers (fc2 with 400+ salience) are exclusively on the text side (summary_table.csv).
- **Complementarity:** Text-side layers have higher mean ρ (0.163) than image-side (0.084), meaning slightly weaker complementarity on the text pathway. The **modality_scatter_mean_spearman_rho.png** scatter plot shows many points above the diagonal (text ρ > image ρ), especially for late blocks (red-colored points in the upper-left region).
- **Temporal variation:** Image and text CoV are comparable in aggregate (image 0.174 vs text 0.168), but the **modality_scatter_cov_temporal.png** shows substantial scatter away from the diagonal, with many points above it (text higher CoV) particularly at early blocks (dark blue points).
- The **summary_dashboard.png** top-left panel ("Mean act. salience" vs block) shows text-side salience (orange line) spiking dramatically at late blocks (20–23) while image-side (blue) remains relatively flat. The top-center panel ("Mean ρ" vs block) shows image-side ρ decreasing with depth while text-side ρ is more erratic.

**Implication:** Any CSB/SSC implementation for SD3 should potentially use **separate calibration parameters** (or at least separate calibration sets) for image and text pathways, since they exhibit different salience patterns, different complementarity levels, and different temporal profiles. This is a fundamental difference from DiT-XL where a single calibration regime suffices.

---

## 5. Submodule Family Risk Ranking (H4)

**SD3 finding:** Risk is concentrated in specific families and specific blocks.

From the **risk_ranking.png** and summary_table.csv, the top-10 highest-risk layers are:

| Rank | Layer | Family | Risk | Key issue |
|------|-------|--------|------|-----------|
| 1 | blocks.22.text.mlp.fc2 | fc2 | 0.535 | Extreme salience (414), high ρ (0.73) means CSB less effective |
| 2 | blocks.22.text.mlp.fc1 | fc1 | 0.463 | Extreme salience (349), high ρ (0.79) means CSB less effective |
| 3 | context_embedder | context | 0.453 | Extreme salience (854), but strong complementarity (ρ = −0.34) helps CSB |
| 4 | blocks.1.image.attn.q_proj | q_proj | 0.412 | Moderate salience (13), high CoV (0.28), moderate ρ (0.70) |
| 5 | final_layer.linear | final | 0.410 | Highest CoV (0.40), high ρ (0.67), regime shift in top-k identity |
| 6 | blocks.21.text.mlp.fc2 | fc2 | 0.410 | High salience (184), high ρ (0.61) |
| 7 | blocks.14.text.mlp.fc2 | fc2 | 0.407 | High salience (252), low ρ (0.11) — CSB effective here |
| 8 | blocks.21.text.mlp.fc1 | fc1 | 0.387 | High salience (247), moderate ρ (0.64) |
| 9 | blocks.1.image.attn.v_proj | v_proj | 0.386 | Same input as q_proj block 1, high CoV (0.28) |
| 10 | blocks.14.image.attn.k_proj | k_proj | 0.386 | Moderate salience (22), moderate ρ (0.25) |

**Key patterns from the family_violins.png:**
- **fc2** has the widest spread and highest outliers in activation salience. It also has the highest temporal CoV (mean 0.215).
- **fc1** has extreme outliers in text-side salience (300+ range) but tighter distributions on the image side.
- **o_proj** has the lowest mean ρ (0.033) of any family — strongest complementarity overall.
- **Attention projections** (q/k/v_proj) share the same input within a block. Their activation salience is identical (same input tensor), but weight salience differs, so ρ differs across q/k/v.

**Depth effects (from depth_profile_\*.png):**
- The **depth_profile_q_proj.png** shows image-side max salience varying between 5–25 across blocks with no strong trend, while text-side max salience shows a mild increase with depth (from ~8 at block 0 to ~17 at block 23). The Spearman ρ (dotted lines) decreases with depth for both sides.
- The **depth_profile_fc1.png** reveals a different pattern: text-side max salience is extremely high at early blocks (250–300 at blocks 0–1), dips to ~150 in mid blocks, then rises again to ~350 at blocks 22–23 — a U-shaped profile. Image-side fc1 stays comparatively low (<100 throughout).
- The **depth_profile_o_proj.png** shows o_proj salience is more uniform across depth (max ≈ 6–16 for both sides), but with ρ (dotted lines) fluctuating around zero across all blocks, confirming that o_proj consistently exhibits strong complementarity regardless of depth.
- Overall, late text-side blocks (20–23) are consistently among the highest-risk targets, driven by extreme salience in fc1/fc2 combined with moderate-to-high ρ.

---

## 6. Velocity Prediction and the Final Layer (H6)

**PTQ4DiT context:** DiT-XL predicts noise ε. SD3 predicts velocity v = noise − image. This changes the target distribution at the final layer.

**SD3 finding: The final layer is one of the highest-risk quantization targets.**

- `final_layer.linear` has risk score 0.410 (rank 5 of 287).
- Its ρ = 0.673 is one of the highest in the model, meaning activation and weight salience are **positively correlated** — CSB would be less effective here.
- Its CoV = 0.397 is the highest — extreme temporal variation.
- Its early-late Jaccard = 0.103 — almost complete turnover of salient channels.
- The **final_layer_analysis.png** confirms:
  - Panel 1 (weight profile): A few channels in the 1536-dim space have notably higher max|weight| than the rest.
  - Panel 2 (temporal boxplot): Clear upward trend in activation salience from early to late σ. At σ ≈ 0.0, the boxes are approximately 10× taller than at σ ≈ 1.0.
  - Panel 3 (distribution shift): The activation distribution shifts rightward and broadens dramatically from σ = 1.0 (narrow, near zero) to σ = 0.0 (broader, higher mean).

**Implication:** The final layer may require special treatment in Phase 2 — potentially a dedicated per-timestep quantization scheme or higher bit-width, since CSB is less effective (high ρ) and standard calibration will be undermined by the regime shift.

---

## 7. Context Embedder Is an Extreme Outlier

The `context_embedder` (projecting 4096-dim T5-XXL embeddings to 1536-dim hidden) has unique properties:

- **Max activation salience = 854.0** — by far the highest in the model (next highest is ~414 for text fc2).
- **Spearman ρ = −0.342** — strong anti-correlation (activation and weight salience are anti-aligned). This is actually the ideal scenario for CSB.
- **CoV = 0.000 and early-late Jaccard = 1.000** — the text embeddings are fixed per prompt (not time-dependent), so there is zero temporal variation. The same channels are salient at every σ step. This is directly visible in **heatmap_context_embedder.png**: every row (σ step) has identical coloring, producing perfectly horizontal color bands. The heatmap also shows a bimodal structure — roughly half the 4096 channels have very low salience (purple), and the other half have high salience (orange-yellow), with a sharp boundary near channel ~2000 (when sorted by mean salience).
- **Gini = 0.624** — high concentration; salience is very uneven.
- The **fig3_context_embedder.png** activation panel shows salience values reaching 800+, with a distinct spatial clustering: channels in the first ~2000 indices (sorted) have near-zero activation, while the remaining ~2000 channels show extremely high and variable magnitudes.

**Implication:** The context embedder's extreme salience makes it a high-risk layer, but its perfect temporal stability and strong anti-correlation mean CSB should work well here. A single calibration point (no time-awareness needed) with aggressive channel balancing should suffice.

---

## 8. Summary: Similarities and Differences vs PTQ4DiT (DiT-XL)

### What is similar (PTQ4DiT observations that hold in SD3)

1. **Salient channels exist** — confirmed across all families. A small number of channels carry disproportionately large magnitudes and drive most quantization error.
2. **Complementarity holds** — 80.5% of layers have ρ < 0.3, supporting CSB applicability.
3. **Temporal variation exists** — top-k channel identity shifts across σ steps (mean early-late Jaccard = 0.255), supporting SSC's time-aware approach.
4. **MSE tracks salience** — quantization error per channel correlates with channel magnitude, just as in PTQ4DiT Figure 3.

### What differs (SD3-specific observations not present in DiT-XL)

1. **Modality asymmetry** — SD3's dual image/text pathways behave differently in salience magnitude, complementarity, and temporal variation. DiT-XL has no modality split. This may require separate calibration for each pathway.
2. **Depth-dependent ρ profile** — early blocks (0–3) have notably higher ρ than mid/late blocks. PTQ4DiT does not report a strong depth gradient in ρ for DiT-XL.
3. **Final layer regime shift** — the velocity-prediction final layer shows a clear two-regime temporal pattern (early vs late σ) with near-complete turnover of salient channels. This is likely absent in DiT-XL's noise-prediction final layer.
4. **Context embedder extreme outlier** — the T5-XXL projection layer has activation salience ~70× higher than the average block layer. DiT-XL has no analogous layer.
5. **Text-side fc2 extreme salience** — late-block text fc2 layers reach 400+ max activation salience, far exceeding anything on the image side or in attention projections.
6. **ρ varies by family within SD3** — o_proj has the lowest mean ρ (0.033), while fc1 has higher mean ρ (0.160). In DiT-XL, PTQ4DiT presents ρ as generally low across all layer types.

### Implications for Phase 2

- **CSB is broadly applicable** to SD3, but may need per-block or per-family tuning rather than a single global strategy.
- **SSC (time-aware calibration) is essential** for the final layer and beneficial for most layers. The final layer may need a dedicated quantization strategy.
- **Modality-specific calibration** should be explored — image and text pathways may benefit from separate balancing parameters.
- **The context embedder** can use CSB with a single calibration point (no time-awareness needed).
- **Late text-side blocks** (especially fc1/fc2 at blocks 20–23) and **early image-side blocks** (blocks 0–3) should be prioritized as high-risk targets.

---

*Data: `diagnostics/summary_table.csv` (287 layers). Plots: `diagnostics/plots/`. Collection config: `diagnostics/config.json`.*
