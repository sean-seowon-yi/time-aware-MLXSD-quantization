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

### Why Phase 1 is necessary for SD3 Medium

PTQ4DiT was designed and validated on class-conditional DiT-XL/2, where the relevant structure is a stack of transformer blocks with MHSA and feedforward modules, both around adaLN-conditioned token processing.

Stable Diffusion 3 Medium is not that exact architecture. It is an MMDiT text-to-image model built on the DiT family but using a multimodal diffusion transformer with separate sets of weights for image and language representations.

Because of that, Phase 1 should not assume that PTQ4DiT’s observations transfer unchanged. In particular, SD3 Medium may differ from DiT-XL/2 in at least four ways:

- the denoiser is multimodal rather than image-only,
- image and text pathways do not necessarily share identical salience behavior,
- the interaction between text and image tokens may change activation statistics,
- the exact locations of the hardest linear layers may shift relative to DiT-XL/2.

So Phase 1 should be framed as: measure first, adapt second.

### Scope of Phase 1

Phase 1 should focus primarily on the SD3 Medium denoiser backbone, not the full end-to-end pipeline.

That means the highest-priority target is the MMDiT denoiser, because PTQ4DiT’s theory is about quantization difficulty in transformer linear layers, especially around attention and feedforward submodules.

At the start of the project, the following components should be treated as follows:

**Analyze in Phase 1**
- image-side linear projections inside MMDiT blocks
- text-side linear projections inside MMDiT blocks
- joint attention-related projections if present in the implementation
- feedforward layers
- modulation / conditioning linear layers that play a role analogous to adaLN modulation

**De-prioritize for Phase 1**
- VAE encoder/decoder
- CLIP and T5 text encoders as standalone modules
- safety checker or ancillary pipeline modules

This is not because those components are unimportant, but because your first question is whether the PTQ4DiT salience story appears inside the SD3 Medium denoiser. If that story does not show up there, extending analysis to the rest of the pipeline is premature.

### Core hypotheses to test

Phase 1 should be explicitly organized around the following hypotheses.

**H1. Salient channel hypothesis.**  
Some channels in SD3 Medium linear layers will have much larger absolute magnitude than others, for both activations and weights, and those channels will dominate range-sensitive quantization error.

**H2. Complementarity hypothesis.**  
In at least a meaningful subset of layers, the channels that are highly salient in activation will not exactly coincide with the channels that are highly salient in weights.

**H3. Temporal variation hypothesis.**  
Activation salience in SD3 Medium will vary meaningfully across the sampling trajectory, so a one-timestep estimate of activation scale will be biased.

**H4. Architecture-localization hypothesis.**  
The strongest salience and temporal-variation effects will not be uniform across all layers. Some submodules will be much harder PTQ targets than others.

Phase 1 is successful if it produces enough evidence to accept, reject, or refine these four hypotheses layer by layer.

### High-level implementation plan

#### 1. Freeze a reference inference configuration

Choose one fixed reference inference setup and keep it unchanged during Phase 1. The purpose is to make the statistics interpretable.

Fix:
- model weights and revision
- precision used for analysis collection
- image resolution
- scheduler / sampler
- number of inference steps
- CFG setting
- prompt format
- seed handling
- whether negative prompts are enabled and how they are defined

This matters because activation statistics are inference-setting dependent. If you change scheduler, steps, guidance, or resolution while collecting diagnostics, you will not know whether differences come from architecture or from the inference regime.

Use one main setting first, then optionally one or two ablation settings later. For example, Phase 1 can start with a single production-relevant SD3 Medium setup, then later compare low-CFG vs high-CFG or fewer-step vs more-step runs.

#### 2. Define the calibration corpus for diagnosis

Construct a diagnostic calibration corpus rather than a training dataset. It should be small enough to be feasible, but diverse enough to expose different activation regimes.

The prompt set should deliberately include:
- short object prompts
- long compositional prompts
- prompts with typography or quoted text
- counting / attribute-binding prompts
- style-heavy prompts
- scene prompts with many entities
- human or face prompts
- abstract prompts

The point is not dataset representativeness in the statistical learning sense. The point is to expose a broad range of activation regimes.

Use multiple random seeds per prompt so the diagnostic results are not just artifacts of one latent trajectory.

#### 3. Decide which timesteps to observe

Choose a fixed subset of timesteps or continuous-time indices spanning the whole sampling trajectory. The simplest approach is to use uniform coverage across the trajectory, with extra attention to:
- early noisy region
- middle transition region
- late refinement region

A good Phase 1 rule is:
- use enough timesteps to see shape, not just endpoints,
- include the beginning, middle, and end explicitly,
- log exact scheduler indices or sigma values so your later plots are reproducible.

Do not only inspect the midpoint.

#### 4. Instrument the model with forward hooks

Add hooks to all candidate linear layers inside the SD3 Medium denoiser. The hooks should capture the input activation tensor to the linear layer and associate it with the fixed weight tensor of that layer.

For each hooked layer, log:
- layer name
- block index
- submodule type
- modality branch if identifiable
- timestep index
- prompt id
- seed
- batch id
- activation input shape
- weight shape

You do not want to save full activations for every run unless your setup is tiny. That will become infeasible very quickly. Instead, Phase 1 should compute statistics online inside the hook and store summaries.

At minimum, every hook should reduce the activation tensor to per-channel diagnostics.

If the activation input to a linear layer is \(X \in \mathbb{R}^{B \times N \times d}\), flatten across batch and token dimensions so that the channel axis remains the final feature dimension. Then compute per-channel summaries over the flattened \(BN\) elements.

#### 5. Standardize the notion of channel

Before collecting any statistics, define one consistent internal convention:

- for activations, a channel is one feature dimension across all tokens,
- for weights, the channel axis must correspond to the same input-channel index used by the activation side.

This sounds trivial but is easy to get wrong in implementation because frameworks may store weight matrices as \([d_{out}, d_{in}]\) while the paper writes them as \(W \in \mathbb{R}^{d_{in} \times d_{out}}\).

So Phase 1 should explicitly include a layer-normalization step for statistics:
- define the activation channel axis,
- map the weight tensor to the same logical channel index,
- verify this mapping once on a toy example before running the large diagnostic sweep.

#### 6. Compute the primary salience statistics

For each layer \(l\) and timestep \(t\), compute the exact salience measure used in PTQ4DiT:

\[
s(X_j^{(t)}) = \max |X_j^{(t)}|, \qquad s(W_j) = \max |W_j|
\]

For every layer, store:
- per-channel max absolute activation
- per-channel max absolute weight
- overall max across channels
- top-k salient activation channel indices
- top-k salient weight channel indices

In addition to the exact salience definition, Phase 1 should also log supporting robust summaries:
- mean absolute value
- standard deviation
- p99 absolute value
- p99.9 absolute value

These are not replacements for the salience metric. They are sanity checks that help you distinguish a true structural outlier channel from a channel that only spikes once.

#### 7. Quantify how concentrated salience is

Do not stop at plotting raw max values. Phase 1 should measure how concentrated the salience is.

For each layer and timestep, compute:
- ratio of largest channel salience to median channel salience
- ratio of top-1 to top-10 average
- fraction of total maximum-range mass carried by top-k channels
- Gini-like or entropy-like concentration measure over per-channel salience

This tells you whether a layer has:
- broad moderate scaling, or
- a small number of severe outlier channels

CSB is most compelling when the latter pattern appears.

#### 8. Test the link between salience and quantization error

Run proxy quantization experiments on captured activations and weights:
- tensor-wise activation quantization candidate
- channel-wise weight quantization candidate
- candidate bit-widths such as W8A8 and W4A8
- uniform quantizers first

For each layer and timestep, compute:
- per-channel activation MSE after quantization
- per-channel weight MSE after quantization
- correlation between salience and quantization error
- scatter plots of channel salience vs channel MSE

If large salience does not predict large proxy quantization error in a certain SD3 submodule, then that submodule may not benefit from PTQ4DiT-style treatment in the same way.

#### 9. Measure activation–weight complementarity

This is the most important structural test.

For each layer and timestep, compare the activation salience vector and the weight salience vector.

Compute:
- Spearman correlation between \(s(X^{(t)})\) and \(s(W)\)
- Pearson correlation as a secondary metric
- top-k overlap rate between salient activation channels and salient weight channels
- Jaccard overlap of top-k sets
- rank displacement statistics
- optional heatmaps of activation-rank vs weight-rank alignment

Your key deliverable here is not a single average number. It is a layerwise map showing:
- layers where complementarity is strong,
- layers where it is weak,
- whether image-side and text-side branches behave differently,
- whether the attention projections and MLP layers behave differently.

This is the part of Phase 1 that will tell you whether CSB is likely to transfer.

#### 10. Measure temporal variation explicitly

For each layer and each channel, track salience across all sampled timesteps.

Phase 1 should compute:
- per-channel salience trajectory over timestep
- mean, variance, and coefficient of variation across time
- top-k salient set at each timestep
- overlap of top-k salient channels between early/mid/late steps
- maximum rank change across time
- salience stability index for each channel
- layerwise temporal drift score

The practical question is not just does salience vary, but:
- do the same few channels stay salient, or
- does the identity of the salient channels move over time?

If the identity changes substantially, then midpoint-only calibration will likely be inadequate. If the identity is stable, then a simpler method may suffice.

#### 11. Separate analysis by submodule type

Because SD3 Medium is MMDiT rather than vanilla DiT-XL/2, Phase 1 should not treat all linear layers as one pool. Keep the analysis broken down by submodule family.

Recommended grouping:
- image-side attention projections
- text-side attention projections
- attention output projections
- feedforward first linear layer
- feedforward second linear layer
- conditioning / modulation projections
- any special multimodal projection layers in the implementation

For each group, report:
- prevalence of salient channels
- strength of complementarity
- temporal instability
- proxy quantization sensitivity

For SD3 Medium, you need to know whether the same priority ordering still holds.

#### 12. Separate analysis by modality interaction regime

Because SD3 Medium performs multimodal processing with separate image and language weights that interact through the denoiser, Phase 1 should also compare runs across prompt regimes.

At minimum compare:
- very short prompts
- long descriptive prompts
- typography-heavy prompts
- prompts with many entities and relations

The reason is that multimodal coupling may change which channels become salient. A layer that looks benign on short object prompts may become highly unstable for typography or complex compositional prompts.

This matters if your end goal is real deployment rather than paper-style synthetic calibration.

### Recommended outputs of Phase 1

By the end of Phase 1, you should produce the following artifacts.

#### A. Layerwise diagnostic table
Each row should correspond to one linear layer and include:
- layer id
- submodule family
- modality branch
- mean activation salience
- max activation salience
- mean weight salience
- max weight salience
- mean Spearman correlation over timesteps
- minimum Spearman correlation over timesteps
- temporal variation score
- proxy W8A8 activation MSE
- proxy W4A8 weight MSE
- top-k overlap metric
- a final risk score for quantization difficulty

#### B. Plots
At minimum:
- per-layer salience histograms
- activation salience vs quantization error scatter plots
- weight salience vs quantization error scatter plots
- timestep-vs-channel salience heatmaps
- layerwise Spearman correlation plots
- top-k channel overlap across time
- grouped comparisons by submodule family

#### C. Ranked conclusions
A short written summary answering:
- Which layers are hardest?
- Is complementarity present strongly enough to justify CSB?
- Is temporal variation strong enough to justify SSC?
- Are image-side layers and text-side layers different?
- Is one branch safe to quantize earlier than the other?
- Does the conditioning path require separate treatment?

### Practical implementation guidance

Use online accumulation, not full activation dumps. For each hook call, update running statistics and discard the raw tensor.

Store statistics in a structured format such as:
- one row per layer, timestep, prompt, seed
- separate arrays for per-channel summaries if needed
- a reproducible metadata file containing all inference settings

Do all diagnostic computations in at least fp32, even if the model itself runs in fp16 or bf16. Otherwise your measurement stage will be contaminated by reduced-precision analysis.

Run the model in eval mode with deterministic seeds. Keep dropout off and avoid any stochastic settings beyond the latent seed.

Start with one small pilot sweep to validate:
- hook coverage,
- channel-axis correctness,
- manageable storage,
- sensible plots.

Only after that should you launch the full Phase 1 data collection.

### Decision criteria for moving to Phase 2

Phase 1 should end with a clear go / no-go decision.

Move to Phase 2 if most of the following are true:
- many important denoiser layers exhibit concentrated salient channels,
- salience is positively associated with proxy quantization error,
- activation and weight salience are often weakly correlated in at least some major layer families,
- activation salience changes materially over the sampling trajectory,
- a small subset of layers accounts for most quantization difficulty.

Do not move directly to PTQ4DiT-style balancing if Phase 1 shows:
- no meaningful channel outliers,
- no clear salience–error relationship,
- strong activation–weight alignment everywhere,
- negligible temporal variation,
- or completely different behavior between SD3 Medium submodules that invalidates a unified treatment.

In that case, the right next step is not implement PTQ4DiT anyway, but redesign the method around the actual SD3 Medium statistics.

### Copy-pasteable project description

**Phase 1: Statistical Diagnosis of Quantization Difficulty in Stable Diffusion 3 Medium**

The purpose of Phase 1 is to analyze whether the key assumptions behind PTQ4DiT transfer to Stable Diffusion 3 Medium before implementing any quantization-specific intervention. PTQ4DiT identifies two primary quantization challenges in Diffusion Transformers: salient channels with extreme magnitudes and temporal variation in salient activations across the sampling trajectory. It also relies on an empirical complementarity between activation salience and weight salience within the same layer.

Since Stable Diffusion 3 Medium uses a multimodal diffusion transformer architecture with separate weights for image and language representations and multiple fixed text encoders, these assumptions should be verified directly rather than inherited from DiT-XL/2.

Phase 1 will therefore instrument the SD3 Medium denoiser and collect activation and weight statistics from linear layers across the full sampling trajectory under a fixed inference configuration. The analysis will focus on image-side, text-side, attention, feedforward, and conditioning-related linear layers. For each layer and timestep, per-channel activation salience and weight salience will be measured using the maximum absolute value, consistent with PTQ4DiT. Additional proxy quantization experiments will be used to test whether large channel magnitude is associated with increased quantization error.

The main objectives of Phase 1 are:
(1) to determine whether salient channels are present in SD3 Medium linear layers;
(2) to quantify how strongly salience predicts quantization error;
(3) to measure the correlation or complementarity between activation and weight salience in each layer;
(4) to characterize how activation salience evolves over timesteps; and
(5) to identify which layers and submodules are the highest-risk PTQ targets.

The deliverables of Phase 1 will include a layerwise diagnostic table, salience and quantization-error visualizations, temporal salience trajectory plots, correlation analyses between activation and weight salience, and a ranked assessment of which SD3 Medium components are most suitable for PTQ4DiT-style balancing and which may require architecture-specific adaptation.
