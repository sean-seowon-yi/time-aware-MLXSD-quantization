## Phase 1: Diagnostic Analysis of Activation and Weight Salience for PTQ on Stable Diffusion 3 Medium

### Goal of Phase 1

Phase 1 is a pure diagnosis and measurement stage. The objective is not to quantize Stable Diffusion 3 Medium yet, but to determine whether the main assumptions behind PTQ4DiT also hold for SD3 Medium, and if they do, where they hold most strongly.

For SD3 Medium, Phase 1 should answer the following questions:

1. Do salient activation channels exist in the SD3 Medium denoiser?
2. Do salient weight channels exist in the same layers?
3. Does larger channel magnitude actually correspond to larger post-quantization error, at least in a proxy experiment?
4. Are activation-salient and weight-salient channels often weakly aligned or complementary?
5. How much do activation salience patterns change over the sampling trajectory?
6. Which submodules and which layers are the hardest quantization targets?

The final output of Phase 1 should be a ranked diagnostic report that tells you whether PTQ4DiT is likely to transfer directly to SD3 Medium, and if not, which architectural components will require modification.

---

### Architectural comparison: DiT-XL/2 vs SD3 Medium MMDiT

PTQ4DiT was designed and validated exclusively on class-conditional DiT-XL/2. Before running any diagnostics, we need to understand exactly how SD3 Medium differs, because each difference creates a potential point where PTQ4DiT's assumptions may not transfer.

#### Side-by-side summary

| Property | DiT-XL/2 (PTQ4DiT target) | SD3 Medium (our target) |
|---|---|---|
| **Architecture family** | DiT (homogeneous transformer stack) | MMDiT (multimodal diffusion transformer) |
| **Block type** | Single `DiTBlock` with MHSA + PF | `MultiModalTransformerBlock` with paired image/text `TransformerBlock`s sharing a joint SDPA |
| **Depth** | 28 blocks | 24 multimodal blocks (`depth_multimodal=24`), 0 unified blocks |
| **Hidden size** | 1152 | 1536 (`64 * 24`) |
| **Attention heads** | 16 | 24 |
| **Per-head dim** | 72 | 64 |
| **MLP ratio** | 4 | 4 |
| **Attention type** | Self-attention (image tokens only) | Joint attention (image + text tokens concatenated for Q/K/V, then split post-SDPA) |
| **Conditioning** | Class label → adaLN via MLP | **Pooled**: CLIP-L pooled (768d) ⊕ CLIP-G pooled (1280d) = 2048d → `y_embedder` MLP → combined with timestep embedding → adaLN. **Token-level**: CLIP-L hidden (77 tok × 768d) ⊕ₐ CLIP-G hidden (77 tok × 1280d) = 77 tok × 2048d, zero-padded to 4096d, then concatenated along sequence dim with T5-XXL (up to 512 tok × 4096d) → `context_embedder` Linear(4096, 1536) → text tokens |
| **Text encoders** | None (class-conditional) | CLIP-L/14 (768d), CLIP-G/14 (1280d), T5-XXL (4096d) |
| **Token-level text token count** | N/A | 77 (CLIP) + up to 512 (T5) = up to 589 tokens, all projected to hidden_size by `context_embedder` |
| **Noise schedule / sampler** | DDPM with discrete timesteps (250 steps) | Rectified flow with continuous sigma; Euler ODE sampler |
| **Prediction target** | Noise ε (model predicts noise added to clean image) | Velocity v = noise − image (model predicts the velocity field; denoised estimate: `x − v · σ`) |
| **Timestep representation** | Discrete integer `t ∈ {1, ..., T}` | Continuous `σ ∈ [0, 1]`, mapped to sinusoidal embedding of `σ * 1000` |
| **Positional encoding** | Learned absolute position embedding | Learned absolute position embedding (SD3 2b); QK-norm + learned embedding (SD3 8b) |
| **Normalization** | LayerNorm (no elementwise affine) | LayerNorm (no elementwise affine), weight=None, bias=None |
| **FFN activation** | GELU | GELU |
| **adaLN modulation params per block** | 6 (shift, scale, gate for pre-attn and pre-MLP) | 6 per image `TransformerBlock` + 6 per text `TransformerBlock` (except last text block: 2) + 2 for `FinalLayer` |
| **Re-parameterization target for CSB** | adaLN MLP → single block | adaLN MLP → separate image-side and text-side blocks |
| **Weight matrix convention** | `nn.Linear(d_in, d_out)` stored as `[d_out, d_in]` | MLX `nn.Linear` stored as `[d_out, d_in]`; `k_proj` has no bias |
| **VAE latent dim** | 4 (SD1/2 style) | 16 (SD3 VAE) |
| **Patch size** | 2 | 2 |

#### Implications for PTQ4DiT transfer

Each difference above maps to a concrete risk for the PTQ4DiT method:

**1. Joint attention mixes modality statistics.**
In DiT-XL/2, the Q/K/V projections and the o_proj all operate on a single stream of image tokens. The activation statistics for any given layer come from one modality.

In SD3 Medium, each `MultiModalTransformerBlock` computes Q/K/V separately for image and text via their own `TransformerBlock.attn.{q,k,v}_proj` (lines 471–473 of `mlx/mmdit.py`). These are concatenated before SDPA:

```
q = concat([image_q, text_q])
k = concat([image_k, text_k])
v = concat([image_v, text_v])
sdpa_output = scaled_dot_product_attention(q, k, v)
```

After SDPA, the output is split back and routed through separate `o_proj` layers. This means:
- **Pre-SDPA projections** (q_proj, k_proj, v_proj) see only their own modality's tokens. Their activation statistics are modality-pure.
- **Post-SDPA projection** (o_proj) sees the SDPA output, which already contains cross-modal attention information. Its activation statistics reflect both modalities.
- **FFN layers** (fc1, fc2) also see post-attention residual stream tokens that carry cross-modal information.

The diagnostic must track image-side and text-side projections separately and determine whether they exhibit different salience patterns.

**2. Two sets of adaLN modulation per block complicate re-parameterization.**
In DiT-XL/2, there is one `adaLN_modulation` MLP per block that produces 6 parameters. The re-parameterization absorbs \(B_\rho^X\) into this MLP's weights and biases (Eq. 20 in the paper).

In SD3 Medium, each `MultiModalTransformerBlock` contains:
- `image_transformer_block.adaLN_modulation` → SiLU + Linear(1536, 9216) producing 6 × 1536 params
- `text_transformer_block.adaLN_modulation` → SiLU + Linear(1536, 9216) producing 6 × 1536 params (or 2 × 1536 for the last block)

Re-parameterization is still structurally feasible because the adaLN integration (Eq. 13) is per-linear-layer, and SD3 Medium has the same `adaLN(Z) = LN(Z) * (1 + γ) + β` structure via `affine_transform()`. But each side needs its own \(B_\rho^X\), and the image-side and text-side balancing matrices may differ. Phase 1 must determine whether complementarity holds independently on each side.

**3. Rectified flow changes the meaning of "timestep" and the sampling trajectory.**
DiT-XL/2 uses DDPM with discrete timesteps `t ∈ {1, ..., 1000}`, sampled uniformly during training. The denoising trajectory uses 250 steps (or fewer). PTQ4DiT selects 25 calibration timesteps uniformly from these 250 steps.

SD3 Medium uses rectified flow where \(\sigma \in [0, 1]\) parameterizes a straight-line interpolation:
```
x_σ = σ * noise + (1 - σ) * image
```

The `ModelSamplingDiscreteFlow` (in `mlx/sampler.py`) maps an internal timestep index `t ∈ {1, ..., 1000}` to sigma via:
```
σ = shift * (t/1000) / (1 + (shift - 1) * (t/1000))
```
where `shift=1.0` for SD3 (so `σ = t/1000`, a linear mapping).

The Euler sampler steps through a decreasing sequence of sigma values. This is an ODE solver, not the ancestral DDPM sampler PTQ4DiT was validated with. The activation dynamics over the trajectory may differ qualitatively:
- **DDPM**: noisy at high `t`, progressively denoised. The noise schedule is nonlinear (cosine or linear beta schedule).
- **Rectified flow**: linear interpolation trajectory. The model predicts velocity \(v = \text{noise} - \text{image}\) rather than noise \(\epsilon\).

Phase 1 must characterize activation salience as a function of sigma (or equivalently the step index in the Euler schedule) to determine whether temporal variation is comparable to what PTQ4DiT observed.

**4. Text-conditioned generation creates prompt-dependent activation regimes.**
DiT-XL/2 is class-conditional: the conditioning input `c` is a class embedding (one of 1000 classes). The activation statistics are relatively homogeneous across classes at a given timestep.

SD3 Medium conditions on rich text embeddings from three encoders. Different prompts produce dramatically different token-level text embeddings (varying in sequence length content, and magnitude) and pooled text embeddings. This means:
- Activation statistics at the same sigma may vary more across prompts than across classes in DiT-XL/2.
- Salient channel identity may be prompt-dependent, not just timestep-dependent.
- The calibration corpus must be deliberately diverse (not just "random ImageNet classes").

**5. Token sequence length is variable and asymmetric.**
In DiT-XL/2, all tokens are image tokens. The sequence length is fixed for a given resolution (e.g., 256 patches for 256×256 with patch_size=2).

In SD3 Medium, the joint attention sequence is `[image_tokens, text_tokens]` (for `depth_unified=0`, the code concatenates image-first: `mx.concatenate([image_q, text_q], axis=1)`), where:
- Image tokens: `(H/patch_size) * (W/patch_size)` = 4096 tokens for 1024×1024 input (latent 128×128, patched to 64×64), or 1024 for 512×512
- Text tokens: 77 (CLIP-L + CLIP-G hidden states concatenated along the **feature** axis, not the sequence axis) + up to 512 (T5-XXL) = up to 589 tokens total

The text construction is frequently misunderstood. In `DiffusionPipeline.encode_text()`:
```
CLIP-L hidden_states[-2]: [B, 77, 768]
CLIP-G hidden_states[-2]: [B, 77, 1280]
→ concat along feature dim: [B, 77, 2048]
→ zero-pad to 4096:         [B, 77, 4096]
→ concat T5 along seq dim:  [B, 77 + T5_len, 4096]
→ context_embedder:          [B, 77 + T5_len, 1536]
```

At standard resolution (1024×1024), the ratio is approximately 4096 image tokens to ~589 text tokens (7:1). The text pathway has substantially fewer tokens contributing to each channel's per-channel statistics. This asymmetry has two consequences:
- Text-side salience estimates are noisier (fewer tokens in the reduction).
- The joint SDPA attention matrix is rectangular in the modality dimension, and cross-modal attention patterns may introduce modality-dependent activation distributions in the post-SDPA output.

**6. The `context_embedder` is a standalone linear projection.**
SD3 Medium has a `context_embedder = nn.Linear(4096, 1536)` that projects T5-XXL token embeddings to hidden_size before they enter the transformer blocks. This layer does not exist in DiT-XL/2 and should be included in the diagnostic because it processes raw text encoder outputs (which may have their own outlier channels).

**7. The `FinalLayer` has its own adaLN modulation.**
The `FinalLayer` contains `adaLN_modulation` (SiLU + Linear(1536, 3072), producing shift and scale) and a `linear` projection (Linear(1536, 64)) that maps back to VAE latent space. This layer is a quantization target and needs separate analysis.

**8. The model predicts velocity, not noise — the output distribution differs.**
DiT-XL/2 predicts noise \(\epsilon\). The model output has the statistical properties of Gaussian noise at high timesteps and residual correction at low timesteps. SD3 Medium predicts the velocity field \(v = \text{noise} - \text{image}\). The `calculate_denoised` method in `ModelSamplingDiscreteFlow` computes:
```
denoised = model_input - model_output * sigma
```
This means the model output \(v\) has a different magnitude profile than \(\epsilon\):
- At \(\sigma \approx 1\) (pure noise), \(v \approx \text{noise} - \text{image}\), so the output magnitude is on the order of the data range.
- At \(\sigma \approx 0\) (near-clean), \(v\) is still \(\text{noise} - \text{image}\), but the model input is nearly clean, so the prediction must be precise.

The final layer's output statistics may therefore differ qualitatively from DiT-XL/2. Phase 1 should include the `final_layer.linear` in the diagnostic and compare its activation dynamics against the interior blocks.

**9. Classifier-Free Guidance doubles the effective batch and mixes conditioning regimes.**
When CFG is enabled (`cfg_weight > 0`), the `CFGDenoiser` concatenates the latent twice along the batch dimension:
```python
x_t_mmdit = mx.concatenate([x_t] * 2, axis=0)  # [2, H, W, C]
```
The conditioning tensor already has two entries: `conditioning[0]` for the positive prompt and `conditioning[1]` for the negative (empty) prompt. So the model processes both conditioned and unconditioned inputs in a single forward pass with batch size 2.

This has implications for activation statistics collection:
- Per-channel statistics computed across the batch dimension conflate conditioned and unconditioned activations, which may have different distributions.
- The text-side pathway sees rich prompt embeddings for the positive sample and near-zero embeddings for the negative sample simultaneously.
- Phase 1 hooks should either record batch elements separately or at minimum record whether CFG was active, so we can determine whether CFG-induced distribution mixing affects salience estimates.

For Phase 1 diagnostics, running with CFG disabled (cfg_weight=0, single batch element) simplifies the analysis and isolates the model's intrinsic activation statistics. A follow-up ablation with CFG enabled can determine whether the batch-mixed regime changes salience patterns.

**10. The modulation caching mechanism changes the execution pattern.**
The DiffusionKit implementation pre-computes all `adaLN_modulation` outputs for all timesteps via `cache_modulation_params()` and offloads the modulation MLP weights to save memory. The cache key is `timestep.item()` (the sigma value × 1000). This means during actual inference, the adaLN MLP is not executed — only the cached parameters are looked up via `self._modulation_params[timestep.item()]`. For Phase 1 diagnostics, this is important because:
- Hooks on `adaLN_modulation` linear layers will only fire during the caching phase, not during the main forward pass.
- The forward hooks for activation collection should target the layers that execute during the main forward pass: `attn.{q,k,v,o}_proj`, `mlp.{fc1,fc2}`, `context_embedder`, and `final_layer.linear`.
- To collect activation statistics for adaLN layers, hooks must be registered before calling `cache_modulation_params()`, or alternatively, the modulation outputs can be extracted from the `_modulation_params` dict directly after caching.

---

### Why Phase 1 is necessary for SD3 Medium

Given the ten architectural and operational differences above, Phase 1 should not assume that PTQ4DiT's observations transfer unchanged. The differences are not cosmetic — they affect the statistical properties of activations, the shape of the temporal trajectory, the prediction target, and the feasibility of the re-parameterization scheme.

Phase 1 should be framed as: measure first, adapt second.

### Scope of Phase 1

Phase 1 should focus primarily on the SD3 Medium denoiser backbone, not the full end-to-end pipeline.

The highest-priority target is the MMDiT denoiser, because PTQ4DiT's theory is about quantization difficulty in transformer linear layers, especially around attention and feedforward submodules.

**Analyze in Phase 1**

The following linear layers exist per `MultiModalTransformerBlock` (24 blocks total for SD3 2b):

| Layer | Image-side | Text-side | Shape (weight) | Notes |
|---|---|---|---|---|
| `attn.q_proj` | ✓ | ✓ | `[1536, 1536]` | Has bias |
| `attn.k_proj` | ✓ | ✓ | `[1536, 1536]` | No bias (softmax invariance) |
| `attn.v_proj` | ✓ | ✓ | `[1536, 1536]` | Has bias |
| `attn.o_proj` | ✓ | ✓ (except block 23) | `[1536, 1536]` | Has bias; last text block skips this |
| `mlp.fc1` | ✓ | ✓ (except block 23) | `[6144, 1536]` | GELU follows |
| `mlp.fc2` | ✓ | ✓ (except block 23) | `[1536, 6144]` | |
| `adaLN_modulation.1` | ✓ | ✓ | `[9216, 1536]` or `[3072, 1536]` | SiLU precedes; 6 or 2 output groups |

Plus these singleton layers:

| Layer | Shape (weight) | Notes |
|---|---|---|
| `context_embedder` | `[1536, 4096]` | Projects T5 text embeddings |
| `x_embedder.proj` | Conv2d, `[1536, 16, 2, 2]` | Patchify + project latent image |
| `y_embedder.mlp.0` | `[1536, 2048]` | Pooled text embedding adapter |
| `y_embedder.mlp.2` | `[1536, 1536]` | |
| `t_embedder.mlp.0` | `[1536, 256]` | Timestep sinusoidal → hidden |
| `t_embedder.mlp.2` | `[1536, 1536]` | |
| `final_layer.linear` | `[64, 1536]` | Unpatchify projection |
| `final_layer.adaLN_modulation.1` | `[3072, 1536]` | Final adaLN |

This gives a total of approximately **24 × 12 + 8 = 296 linear layers** to analyze (excluding the last text block's missing o_proj/mlp).

**De-prioritize for Phase 1**
- VAE encoder/decoder
- CLIP and T5 text encoders as standalone modules
- Safety checker or ancillary pipeline modules

---

### Core hypotheses to test

Phase 1 should be explicitly organized around the following hypotheses.

**H1. Salient channel hypothesis.**
Some channels in SD3 Medium linear layers will have much larger absolute magnitude than others, for both activations and weights, and those channels will dominate range-sensitive quantization error.

**H2. Complementarity hypothesis.**
In at least a meaningful subset of layers, the channels that are highly salient in activation will not exactly coincide with the channels that are highly salient in weights. This complementarity is the foundation of CSB.

**H3. Temporal variation hypothesis.**
Activation salience in SD3 Medium will vary meaningfully across the sampling trajectory (parameterized by sigma), so a single-timestep estimate of activation scale will be biased.

**H4. Architecture-localization hypothesis.**
The strongest salience and temporal-variation effects will not be uniform across all layers. Some submodules (e.g., image-side attention vs text-side FFN vs adaLN modulation) will be much harder PTQ targets than others.

**H5. Modality-asymmetry hypothesis.**
Image-side and text-side pathways within the same block may exhibit different salience profiles, complementarity strength, and temporal stability, because they process tokens of different modality and sequence length.

**H6. Velocity-prediction hypothesis.**
Because SD3 Medium predicts velocity \(v = \text{noise} - \text{image}\) rather than noise \(\epsilon\), the output layer (`final_layer.linear`) and late-block activations may have a different salience profile than in noise-predicting DiT-XL/2. The velocity target does not vanish as the sample becomes clean (unlike residual noise), so late-trajectory activations may remain high-magnitude rather than shrinking.

Phase 1 is successful if it produces enough evidence to accept, reject, or refine these six hypotheses layer by layer.

---

### High-level implementation plan

#### 1. Freeze a reference inference configuration

Choose one fixed reference inference setup and keep it unchanged during Phase 1.

Fix:
- **Model**: SD3 Medium 2B (`SD3_2b` config: `depth_multimodal=24`, `num_heads=24`, `hidden_size=1536`)
- **Model weights**: `stabilityai/stable-diffusion-3-medium` (specific revision)
- **Precision for analysis collection**: fp16 model inference (SD3_2b config uses `mx.float16`; note: FLUX uses `mx.bfloat16`), fp32 for all diagnostic reductions
- **Image resolution**: 1024×1024 (latent 128×128, yielding 4096 image tokens after patch_size=2)
- **Sampler**: Euler ODE (from `ModelSamplingDiscreteFlow` with `shift=1.0`)
- **Number of inference steps**: 28 (SD3 Medium default) or 50 (for finer trajectory coverage)
- **CFG scale**: 0.0 (disabled) for initial diagnostics to isolate intrinsic activation statistics; ablate with 5.0 or 7.0 later to assess CFG impact
- **Prompt format**: raw text string, tokenized by CLIP-L, CLIP-G, and T5-XXL
- **Negative prompt**: empty string (standard SD3 practice)
- **Seed handling**: deterministic per-run; multiple seeds per prompt

This matters because activation statistics depend on the inference regime. The rectified flow trajectory with 28 steps produces a different sigma schedule than 50 steps. Start with one setting, then optionally ablate.

#### 2. Define the calibration corpus for diagnosis

Construct a diagnostic calibration corpus. It should be small enough to be feasible but diverse enough to expose different activation regimes.

The prompt set should deliberately include:
- short object prompts ("a red cube on a table")
- long compositional prompts ("a Victorian library with dust motes in golden afternoon light, leather-bound books, and a sleeping cat on a velvet armchair")
- prompts with typography or quoted text ("a neon sign reading OPEN 24 HOURS")
- counting / attribute-binding prompts ("three blue spheres and two yellow cones")
- style-heavy prompts ("an oil painting in the style of Vermeer")
- scene prompts with many entities ("a bustling Tokyo intersection at night with crowds, taxis, and neon signs")
- human or face prompts ("portrait of an elderly woman smiling")
- abstract prompts ("entropy and order in visual tension")

Use 20–50 prompts with 4–8 seeds each, yielding 80–400 trajectories total.

The purpose is not statistical representativeness but exposure of diverse activation regimes, especially for the text pathway where prompt content directly affects token-level embeddings.

#### 3. Decide which timesteps to observe

SD3 Medium's Euler sampler steps through a decreasing sigma sequence. The exact schedule is constructed in `DiffusionPipeline.get_sigmas()`:

```python
start = sampler.timestep(sampler.sigma_max).item()  # 1000.0
end = sampler.timestep(sampler.sigma_min).item()     # 1.0
timesteps = mx.linspace(start, end, num_steps)       # N evenly spaced values in [1000, 1]
sigmas = [sampler.sigma(ts) for ts in timesteps]      # sigma = ts / 1000 (for shift=1.0)
sigmas.append(0.0)                                     # terminal sigma
# Result: N+1 sigma values from ~1.0 down to 0.0
```

For `shift=1.0` (SD3 Medium default), `sigma(t) = t/1000`, so sigma values are linearly spaced. With `N=28` steps (SD3 default), the sigma values are approximately `[1.0, 0.964, 0.929, ..., 0.036, 0.001, 0.0]`.

The Euler ODE step (`sample_euler`) then iterates:
```python
denoised = model(x, timesteps[i], sigmas[i], ...)
d = (x - denoised) / sigmas[i]      # Karras ODE derivative
x = x + d * (sigmas[i+1] - sigmas[i])  # Euler step
```

Choose a subset of step indices spanning the full trajectory. A good Phase 1 strategy:
- Observe all N steps if feasible (N=28 is manageable), or at least 10–15 evenly spaced steps
- Ensure explicit coverage of early (σ ≈ 1.0, near-pure noise), middle (σ ≈ 0.5), and late (σ ≈ 0.0, near-clean image) regions
- Log the exact sigma value at each observed step for reproducibility

The rectified flow trajectory is more linear than DDPM's cosine/linear beta schedule. If activation salience varies primarily at trajectory endpoints, fewer interior points may suffice. But do not assume this — measure it.

#### 4. Instrument the model with forward hooks

Add hooks to all candidate linear layers inside the SD3 Medium denoiser. The hooks should capture the input activation tensor to the linear layer and associate it with the fixed weight tensor.

**Tensor layout in DiffusionKit.** DiffusionKit uses a 4D layout `[B, N, 1, d]` for sequence tensors inside the transformer blocks (the trailing singleton dimension is an artifact of the reshape used for SDPA compatibility). This affects how channel indices map:
- Activation input to a linear layer: shape `[B, N, 1, d_in]`, channel axis is `dim=3` (last)
- After `q_proj`/`k_proj`/`v_proj`: shape `[B, N, 1, d]`, then reshaped to `[B, N, num_heads, per_head_dim]` and transposed to `[B, num_heads, N, per_head_dim]` for SDPA
- After SDPA and before `o_proj`: reshaped back to `[B, N, 1, d]`

The `pre_sdpa` method additionally reshapes Q/K/V through head-splitting and back-flattening for the `depth_unified=0` case:
```python
q = q.reshape(batch, -1, num_heads, per_head_dim).transpose(0, 2, 1, 3)
# Then immediately back:
q = q.transpose(0, 2, 1, 3).reshape(batch, -1, 1, hidden_size)
```
This round-trip is a no-op on values but changes the memory layout. Hooks on `q_proj`/`k_proj`/`v_proj` capture the input before any reshaping, which is the correct point for salience measurement.

**Hooking strategy for the DiffusionKit MLX implementation:**

Because DiffusionKit pre-caches adaLN modulation parameters, the layers that execute during the main denoising loop are:

```
For each MultiModalTransformerBlock (blocks 0–23):
  image_transformer_block.attn.q_proj     ← hook input X [B, N_img, 1, 1536]
  image_transformer_block.attn.k_proj     ← hook input X [B, N_img, 1, 1536]  (same input as q_proj)
  image_transformer_block.attn.v_proj     ← hook input X [B, N_img, 1, 1536]  (same input as q_proj)
  image_transformer_block.attn.o_proj     ← hook input X [B, N_img, 1, 1536]  (SDPA output, image slice)
  image_transformer_block.mlp.fc1         ← hook input X [B, N_img, 1, 1536]  (post-adaLN residual)
  image_transformer_block.mlp.fc2         ← hook input X [B, N_img, 1, 6144]  (post-GELU)
  text_transformer_block.attn.q_proj      ← hook input X [B, N_txt, 1, 1536]
  text_transformer_block.attn.k_proj      ← hook input X [B, N_txt, 1, 1536]
  text_transformer_block.attn.v_proj      ← hook input X [B, N_txt, 1, 1536]
  text_transformer_block.attn.o_proj      ← hook input X [B, N_txt, 1, 1536]  (SDPA output, text slice)
  text_transformer_block.mlp.fc1          ← hook input X [B, N_txt, 1, 1536]
  text_transformer_block.mlp.fc2          ← hook input X [B, N_txt, 1, 6144]
  (block 23: text-side o_proj, fc1, fc2 are skipped — skip_text_post_sdpa=True)

Singleton layers:
  context_embedder                        ← hook input X [B, N_txt, 4096]     (raw T5+CLIP text embeddings)
  final_layer.linear                      ← hook input X [B, N_img, 1, 1536]  (post-adaLN)
```

Note: `q_proj`, `k_proj`, `v_proj` within the same `TransformerBlock` receive the **same** input tensor (the `modulated_pre_attention` output of `affine_transform`). Hooking all three is still useful for validation, but the activation statistics will be identical. The weight salience will differ.

For each hooked layer, log:
- layer name (e.g., `mm_blocks.12.image.attn.q_proj`)
- block index
- submodule type (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `fc1`, `fc2`)
- modality branch (`image`, `text`, `shared`)
- sigma value (continuous) and step index
- prompt id and seed
- activation input shape

**MLX-specific considerations:**
MLX uses lazy evaluation. Forward hooks that compute reductions (like `max(abs(X))`) should call `mx.eval()` on the result to materialize it before the tensor graph is garbage-collected. Alternatively, accumulate statistics in numpy arrays after converting small summary tensors.

#### 5. Standardize the notion of channel

Before collecting any statistics, define one consistent internal convention:

- For activations, a channel is one feature dimension across all tokens. The DiffusionKit layout uses `[B, N, 1, d]` for most activations inside transformer blocks (the trailing singleton is an artifact of reshape/SDPA compatibility). The channel axis is `dim=3` (the last dimension). For `context_embedder` input, the shape is `[B, N_txt, 4096]` (no singleton), so the channel axis is `dim=2`.
- For weights, map to the same logical input-channel index. MLX `nn.Linear` stores weights as `[d_out, d_in]` and computes `x @ W.T + b`. The input-channel axis of the weight matrix is `dim=1`.

Verify this mapping once on a small example before running the full sweep:
```python
# For a Linear(d_in=1536, d_out=1536), e.g. attn.q_proj:
# W.shape = [1536, 1536], W[:, j] is the j-th input channel column
# X.shape = [B, N, 1, 1536], X[:, :, :, j] is the j-th activation channel
# s(X_j) = max(|X[:, :, :, j]|)  — reduce over B, N, and the singleton dim
# s(W_j) = max(|W[:, j]|)         — reduce over the output dimension
#
# For fc2 with d_in=6144:
# W.shape = [1536, 6144], X.shape = [B, N, 1, 6144]
# Channel j ranges over 0..6143
#
# For context_embedder with d_in=4096:
# W.shape = [1536, 4096], X.shape = [B, N_txt, 4096]  (no singleton)
# s(X_j) = max(|X[:, :, j]|)
```

#### 6. Compute the primary salience statistics

For each layer \(l\) and sigma step \(\sigma\), compute the salience measure from PTQ4DiT:

\[
s(X_j^{(\sigma)}) = \max |X_j^{(\sigma)}|, \qquad s(W_j) = \max |W_j|
\]

For every layer, store:
- per-channel max absolute activation (vector of length \(d_{in}\))
- per-channel max absolute weight (vector of length \(d_{in}\))
- overall max across channels
- top-k salient activation channel indices (k=32 or 5% of channels)
- top-k salient weight channel indices

In addition, log supporting robust summaries per channel:
- mean absolute value
- standard deviation
- p99 and p99.9 absolute value

These distinguish true structural outlier channels from channels that spike on a single token.

#### 7. Quantify how concentrated salience is

For each layer and sigma step, compute:
- ratio of largest channel salience to median channel salience
- ratio of top-1 to top-10 average
- fraction of total maximum-range mass carried by top-k channels
- Gini coefficient or entropy over per-channel salience

This tells you whether a layer has broad moderate scaling or a small number of severe outlier channels. CSB is most compelling when the latter pattern appears.

#### 8. Test the link between salience and quantization error

Run proxy quantization experiments on captured activations and weights:
- Tensor-wise activation quantization (asymmetric uniform, 8-bit)
- Channel-wise weight quantization (symmetric uniform, 8-bit and 4-bit)

For each layer and sigma step, compute:
- per-channel activation MSE after quantization
- per-channel weight MSE after quantization
- Spearman correlation between channel salience and channel MSE
- scatter plots of channel salience vs channel MSE

If large salience does not predict large proxy quantization error in a certain SD3 submodule, then that submodule may not benefit from PTQ4DiT-style treatment.

#### 9. Measure activation–weight complementarity

This is the most important structural test — it determines whether CSB can work.

For each layer and sigma step, compare the activation salience vector \(s(X^{(\sigma)})\) and the weight salience vector \(s(W)\).

Compute:
- Spearman correlation between \(s(X^{(\sigma)})\) and \(s(W)\)
- Pearson correlation as a secondary metric
- top-k overlap rate between salient activation channels and salient weight channels
- Jaccard overlap of top-k sets
- rank displacement statistics

**Key deliverable:** a layerwise map showing:
- layers where complementarity is strong (low Spearman ρ between activation and weight salience),
- layers where it is weak (high ρ, meaning salient channels coincide),
- whether image-side and text-side branches behave differently,
- whether the attention projections and MLP layers behave differently.

In PTQ4DiT, the SSC weighting (Eq. 11) gives higher weight to timesteps where ρ is lower (more complementarity). If SD3 Medium shows uniformly high ρ for some layer families, CSB will have limited effect there.

#### 10. Measure temporal variation explicitly

For each layer and each channel, track salience across all sampled sigma steps.

Compute:
- per-channel salience trajectory over sigma
- mean, variance, and coefficient of variation across sigma steps
- top-k salient set at each sigma step
- overlap of top-k salient channels between early/mid/late steps
- maximum rank change across sigma steps
- layerwise temporal drift score

The practical question is not just "does salience vary" but:
- Do the same few channels stay salient across the trajectory, or does the identity of salient channels change?
- Is the variation monotonic (salience grows or shrinks consistently from σ=1 to σ=0) or non-monotonic?
- Does the rectified flow's more linear trajectory produce smoother salience curves than DDPM?

If channel identity is stable, a simpler fixed-calibration method may suffice. If it shifts, SSC-style temporal weighting is needed.

#### 11. Separate analysis by submodule type

Keep the analysis broken down by submodule family:

| Group | Layers per block | Total layers (24 blocks) |
|---|---|---|
| Image-side attention (q/k/v_proj) | 3 | 72 |
| Text-side attention (q/k/v_proj) | 3 | 72 |
| Image-side o_proj | 1 | 24 |
| Text-side o_proj | 1 | 23 (last block skips) |
| Image-side FFN (fc1, fc2) | 2 | 48 |
| Text-side FFN (fc1, fc2) | 2 | 46 (last block skips) |
| Image-side adaLN modulation | 1 | 24 |
| Text-side adaLN modulation | 1 | 24 |
| context_embedder | — | 1 |
| final_layer (linear + adaLN) | — | 2 |

For each group, report:
- prevalence of salient channels (what fraction of layers show outlier channels)
- strength of complementarity (distribution of Spearman ρ)
- temporal instability (distribution of coefficient of variation)
- proxy quantization sensitivity (mean and max MSE)

#### 12. Separate analysis by modality interaction regime

Compare activation statistics across prompt types:
- very short prompts (< 10 tokens)
- long descriptive prompts (> 50 tokens)
- typography-heavy prompts
- prompts with many entities and relations

Also compare:
- early blocks (0–7) vs middle blocks (8–15) vs late blocks (16–23)
- the effect of prompt content on text-side vs image-side salience

The multimodal coupling may cause certain channels to become salient only under certain prompt regimes. A layer that looks benign on short prompts may become highly unstable for complex compositions.

#### 13. Assess re-parameterization feasibility for SD3 Medium

PTQ4DiT's re-parameterization absorbs \(B_\rho^X\) into adjacent operations to avoid runtime overhead. The paper (Appendix D, Eq. 20) shows that for a linear layer following adaLN, the absorption modifies the MLP that regresses \(\gamma\) and \(\beta\):

\[
\tilde{W}_\gamma = W_\gamma B_\rho^X, \quad \tilde{W}_\beta = W_\beta B_\rho^X, \quad \tilde{b}_\gamma = b_\gamma B_\rho^X, \quad \tilde{b}_\beta = b_\beta B_\rho^X
\]

This yields \(\tilde{X} = X B_\rho^X\) without any extra multiply at inference time. We need to verify that each absorption path exists in SD3 Medium.

**Post-adaLN (q/k/v_proj, fc1):** The `adaLN_modulation` MLP produces `(γ, β)` that modulate the input before q/k/v_proj and fc1. In SD3 Medium, the adaLN integration matches PTQ4DiT's Eq. 13 exactly via `affine_transform()`:
```python
def affine_transform(x, shift, residual_scale, norm_module):
    return norm_module(x) * (1.0 + residual_scale) + shift
    # Equivalent to: LN(Z) * (1 + γ) + β
```

The adaLN modulation MLP is:
```python
adaLN_modulation = nn.Sequential(SiLU(), nn.Linear(1536, 9216))
# 9216 = 6 × 1536: [shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp]
```

Since `nn.Linear` in MLX stores weights as `[d_out, d_in]` and computes `x @ W.T + b`, the absorption becomes:
```
W_new = B_rho_X @ W_old      (right-multiply the transposed weight)
b_new = b_old @ B_rho_X.T    (for the relevant output slice)
```
This works identically to DiT-XL/2 because the adaLN structure is the same. **However**: image-side and text-side adaLN MLPs are separate, so each needs its own \(B_\rho^X\). The `shift` and `scale` outputs for q/k/v_proj come from the first two chunks (indices 0, 1) of the 6-way split, while those for fc1 come from chunks 3, 4. Each pair needs its own balancing matrix from the corresponding downstream linear layer.

**Post-SDPA (o_proj):** The SDPA output goes through `o_proj`. The \(B_\rho^X\) for o_proj must be absorbed into the SDPA de-quantization step (or equivalently, into the value projection's output scaling), matching PTQ4DiT's "Post-Matrix-Multiplication" case. In SD3 Medium, the joint SDPA output is split before being fed to separate image/text o_proj layers:
```python
image_sdpa_output = sdpa_outputs[:, :img_seq_len, :, :]
text_sdpa_output = sdpa_outputs[:, -txt_seq_len:, :, :]
```
The absorption must happen per-modality after the split. This is structurally feasible because the split does not mix channels — it only partitions along the sequence dimension.

**Gate parameters.** SD3 Medium applies a gate (modulation parameter index 2 for attention, index 5 for MLP) as a multiplicative scale on the output:
```python
residual = residual + attention_out * post_attn_scale
```
The gate is applied **after** `o_proj`, not before it, so it does not interfere with the \(B_\rho^X\) absorption for o_proj. Similarly, the MLP gate is applied after `fc2`, not before.

**FC2:** The input to fc2 is the GELU output of fc1. The \(B_\rho^X\) for fc2 cannot be absorbed into fc1's weight matrix because GELU is nonlinear (i.e., \(\text{GELU}(x \cdot B) \neq \text{GELU}(x) \cdot B\)). PTQ4DiT treats fc2's activation balancing as a Post-Matrix-Multiplication case, absorbing it into fc1's output (equivalently, the de-quantization step). The same approach applies to SD3 Medium without modification.

**context_embedder:** This is a standalone `nn.Linear(4096, 1536)` with no preceding adaLN. Its input comes from the concatenated CLIP + T5 text encoder output. Re-parameterization requires either:
- Absorbing \(B_\rho^X\) into the text encoders' final layers (frozen and out of scope), or
- Treating this layer differently (e.g., keep at higher precision, or apply online balancing at negligible cost since it runs only once per inference)

Since `context_embedder` runs once per forward pass (not per denoising step), any runtime overhead from online balancing is amortized and negligible.

**final_layer.linear:** This layer follows `final_layer.adaLN_modulation`, which has the same `nn.Sequential(SiLU(), nn.Linear(1536, 3072))` structure producing 2 parameters (shift, scale). Absorption into this adaLN MLP works identically to the block-level case.

**Summary of re-parameterization feasibility:**

| Layer family | Absorption path | Feasible? | Notes |
|---|---|---|---|
| q/k/v_proj (image) | Image adaLN MLP (chunks 0, 1) | Yes | Standard post-adaLN |
| q/k/v_proj (text) | Text adaLN MLP (chunks 0, 1) | Yes | Standard post-adaLN |
| fc1 (image) | Image adaLN MLP (chunks 3, 4) | Yes | Standard post-adaLN |
| fc1 (text) | Text adaLN MLP (chunks 3, 4) | Yes | Standard post-adaLN |
| o_proj (image) | Post-SDPA (image slice) | Yes | Post-matrix-multiplication |
| o_proj (text) | Post-SDPA (text slice) | Yes | Post-matrix-multiplication |
| fc2 (image/text) | Post-GELU (fc1 output) | Yes | Post-matrix-multiplication |
| context_embedder | No preceding adaLN | No | Use online balancing or higher precision |
| final_layer.linear | final_layer adaLN MLP | Yes | Standard post-adaLN |

Phase 1 should confirm this analysis by verifying that the gate parameters do not introduce unexpected interactions, and by checking that the `affine_transform` fast path (batch_size=1 optimization using `mx.fast.layer_norm` with fused scale/shift) is mathematically equivalent to the standard path for re-parameterization purposes.

---

### Recommended outputs of Phase 1

#### A. Layerwise diagnostic table

Each row should correspond to one linear layer and include:
- layer id (e.g., `mm_blocks.12.image.attn.q_proj`)
- submodule family (q_proj, k_proj, v_proj, o_proj, fc1, fc2, adaLN, context, final)
- modality branch (image, text, shared)
- block index
- mean activation salience (averaged over sigma steps)
- max activation salience
- mean weight salience
- max weight salience
- mean Spearman correlation over sigma steps
- minimum Spearman correlation over sigma steps
- temporal variation score (coefficient of variation of per-channel salience over sigma)
- proxy W8A8 activation MSE
- proxy W4A8 weight MSE
- top-k overlap metric (Jaccard between top-k activation and weight channels)
- final risk score for quantization difficulty (composite metric)

#### B. Plots

At minimum:
- per-layer salience histograms (all channels, with top-k highlighted)
- activation salience vs quantization error scatter plots
- weight salience vs quantization error scatter plots
- sigma-vs-channel salience heatmaps (one per representative layer)
- layerwise Spearman correlation bar plots
- top-k channel overlap across sigma steps
- grouped comparisons by submodule family
- image-side vs text-side comparison plots
- salience trajectory plots for the most extreme layers (salience vs sigma)

#### C. Ranked conclusions

A short written summary answering:
- Which layers are hardest?
- Is complementarity present strongly enough to justify CSB?
- Is temporal variation strong enough to justify SSC?
- Are image-side layers and text-side layers different?
- Is one branch safe to quantize more aggressively than the other?
- Does the conditioning path (`context_embedder`, `adaLN`) require separate treatment?
- Are there layers where re-parameterization is infeasible?
- Does the rectified flow trajectory produce qualitatively different temporal dynamics than DDPM?
- Does velocity prediction (vs noise prediction) alter the activation salience landscape, especially in the `final_layer`?
- Does CFG batch doubling affect per-channel statistics in a way that requires separate calibration?
- How does the 4D tensor layout `[B, N, 1, d]` interact with quantization granularity choices?

---

### Practical implementation guidance

**Online accumulation.** Do not dump full activation tensors. For each hook call, compute per-channel statistics and discard the raw tensor. Use Welford's online algorithm for mean/variance, and track running max for salience.

**Storage format.** Store statistics in a structured format:
- one row per (layer, sigma_step, prompt_id, seed)
- separate arrays for per-channel summaries
- a metadata file recording all inference settings

**Precision.** Run all diagnostic reductions in fp32, even if the model runs in fp16 (SD3 2b) or bf16 (FLUX). Cast activation tensors to fp32 inside the hook before computing statistics to avoid precision loss in reductions like `max(abs(X))`.

**Eval mode.** Run the model in eval mode with deterministic seeds. No dropout, no stochastic settings beyond the latent seed.

**Pilot sweep.** Start with 2–4 prompts, 1 seed each, 10 sigma steps, to validate:
- hook coverage (all expected layers fire)
- channel-axis correctness (check shapes and indexing)
- manageable storage (estimate full run size)
- sensible plots (do basic salience histograms look reasonable?)

Only after validating the pilot should you launch the full Phase 1 data collection.

**Approximate compute budget:** For SD3 2b on Apple Silicon (M-series), one full inference trajectory (28 steps) takes ~30–60 seconds. With 200 trajectories and 15 observed sigma steps per trajectory, the main cost is the inference itself (200 × 60s ≈ 3.3 hours). Hook overhead for online statistics is negligible compared to model forward pass time.

---

### Decision criteria for moving to Phase 2

Phase 1 should end with a clear go / no-go decision.

**Move to Phase 2** (implement CSB/SSC for SD3 Medium) if most of the following hold:
- Many important denoiser layers exhibit concentrated salient channels (high Gini, top-1/median ratio > 10×).
- Salience is positively associated with proxy quantization error (Spearman ρ(salience, MSE) > 0.5).
- Activation and weight salience are often weakly correlated in at least some major layer families (Spearman ρ(s(X), s(W)) < 0.3 in many layers).
- Activation salience changes materially over the sigma trajectory (coefficient of variation > 0.2 in many layers).
- A small subset of layers accounts for most quantization difficulty (Pareto-like distribution of sensitivity).

**Do not move to Phase 2 directly** if Phase 1 shows:
- No meaningful channel outliers (salience is broadly uniform).
- No clear salience–error relationship.
- Strong activation–weight alignment everywhere (high ρ), meaning CSB's complementarity assumption fails.
- Negligible temporal variation (stable salience across sigma), meaning SSC adds no value over midpoint calibration.
- Completely different behavior between image-side and text-side that invalidates a unified treatment.

In that case, the right next step is not "implement PTQ4DiT anyway" but "redesign the method around the actual SD3 Medium statistics." Possible alternative directions include:
- SmoothQuant-style migration (if complementarity is weak but outlier channels are consistent)
- Mixed-precision assignment (if only certain layer families are hard)
- Per-modality calibration strategies (if image and text sides need different treatment)
- Asymmetric quantization or log2 quantizers for non-standard distributions

---

### Copy-pasteable project description

**Phase 1: Statistical Diagnosis of Quantization Difficulty in Stable Diffusion 3 Medium**

The purpose of Phase 1 is to analyze whether the key assumptions behind PTQ4DiT (NeurIPS 2024) transfer to Stable Diffusion 3 Medium before implementing any quantization-specific intervention. PTQ4DiT identifies two primary quantization challenges in Diffusion Transformers: salient channels with extreme magnitudes and temporal variation in salient activations across the sampling trajectory. It relies on an empirical complementarity between activation salience and weight salience within the same layer, and a re-parameterization scheme that absorbs salience balancing matrices into adaLN modulation MLPs.

SD3 Medium uses a multimodal diffusion transformer (MMDiT) architecture that differs from DiT-XL/2 in ten ways relevant to PTQ4DiT: (1) dual image/text transformer blocks per layer with joint attention rather than single-modality self-attention; (2) rectified flow with an Euler ODE sampler rather than DDPM; (3) rich text conditioning from three encoders (CLIP-L, CLIP-G, T5-XXL) rather than class labels; (4) separate adaLN modulation MLPs per modality branch; (5) variable-length text token sequences (up to 589 tokens) creating asymmetric statistics; (6) a standalone `context_embedder` with no preceding adaLN; (7) a dedicated `FinalLayer` adaLN; (8) velocity prediction (\(v = \text{noise} - \text{image}\)) instead of noise prediction; (9) Classifier-Free Guidance batch doubling that mixes conditioned and unconditioned activations; and (10) modulation caching that alters the execution pattern during inference. These differences mean PTQ4DiT's observations should be verified directly rather than inherited from DiT-XL/2.

Phase 1 will instrument the SD3 Medium denoiser (DiffusionKit MLX implementation) and collect per-channel activation and weight statistics from all ~296 linear layers across the full Euler sampling trajectory under a fixed inference configuration. The analysis will cover image-side attention, text-side attention, feedforward, conditioning/modulation, and singleton projection layers. For each layer and sigma step, per-channel salience will be measured as maximum absolute value (consistent with PTQ4DiT Eq. 4), complementarity between activation and weight salience will be quantified via Spearman correlation, and proxy quantization experiments will test whether salience predicts quantization error.

The deliverables will include: (1) a layerwise diagnostic table with salience, complementarity, temporal variation, and proxy MSE metrics; (2) a visualization suite including salience heatmaps, temporal trajectories, and complementarity maps; (3) a ranked assessment identifying the hardest quantization targets and whether CSB/SSC are justified; (4) a feasibility analysis of the re-parameterization scheme, with a layer-by-layer feasibility table; and (5) explicit answers to whether the rectified-flow trajectory produces qualitatively different temporal dynamics than DDPM and whether velocity prediction alters the salience landscape.
