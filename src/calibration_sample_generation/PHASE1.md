# Phase 1: Calibration Data Generation

**Paper**: TaQ-DiT — *Time-aware Quantization for Diffusion Transformers* (arXiv:2411.14172v2)  
**Target model**: Stable Diffusion 3 Medium (MMDiT) via [DiffusionKit](https://github.com/argmaxinc/DiffusionKit) on Apple MLX

---

## Purpose

Phase 1 constructs the **calibration dataset** that all subsequent phases (activation profiling, quantization parameter tuning) depend on.
The dataset consists of `(x_t, t)` pairs — noisy latent inputs and their corresponding model timesteps — collected from realistic denoising trajectories, along with the text conditioning used to generate them.

---

## Paper specification (Section IV-A)

> *"We set the sampling steps to 100 and use a Classifier-Free Guidance (CFG) score of 1.50.
> To construct our calibration dataset, we uniformly select 25 steps from the total steps
> and generate 256 samples at each selected step, then randomly shuffle them across the chosen steps."*

| Parameter | Paper value |
|-----------|------------|
| Sampling steps per trajectory | 100 |
| Number of trajectories (samples) | 256 |
| Uniformly selected timesteps | 25 of 100 |
| CFG scale | 1.50 |
| Image resolution | 256 × 256 |
| Sampler | DDPM |
| Conditioning | ImageNet class labels |
| Total calibration points | 25 × 256 = **6,400** |

---

## Our implementation

### Alignment with the paper

| Parameter | Our value | Matches? | Notes |
|-----------|----------|----------|-------|
| Sampling steps | 100 | Yes | `NUM_SAMPLING_STEPS = 100` |
| Trajectories | 256 | Yes | `NUM_CALIBRATION_SAMPLES = 256` |
| Selected timesteps | 25 | Yes | `NUM_SELECTED_TIMESTEPS = 25`, via `np.linspace(0, 99, 25)` |
| CFG scale | 1.5 | Yes | `DEFAULT_CFG_WEIGHT = 1.5` |
| Resolution | 256 × 256 | Yes | `DEFAULT_LATENT_SIZE = (32, 32)` → 32 × 8 = 256 pixels |
| Calibration points | 6,400 | Yes | 25 × 256, shuffled with deterministic seed |

### Deliberate adaptations for SD3

| Aspect | Paper (DiT) | Our adaptation (SD3/MMDiT) | Reasoning |
|--------|-------------|---------------------------|-----------|
| **Sampler** | DDPM | Euler (deterministic ODE) | SD3 uses a flow-matching formulation where Euler is the native sampler. DDPM is incompatible with the noise schedule. |
| **Conditioning** | ImageNet class labels (single integer) | Text prompts (20 diverse prompts, cycled round-robin) | SD3 is text-conditioned, not class-conditioned. Multiple prompts ensure the calibration set covers varied conditioning, producing more representative activation distributions. |
| **Conditioning storage** | Not applicable (class labels are trivial) | Token-level embeddings (`cs`) + pooled embeddings (`cs_pooled`), stored **once per unique prompt** with a `prompt_indices` lookup array | SD3 requires both `conditioning` (token-level, for cross-attention) and `pooled_conditioning` (for adaLN timestep modulation). Storing per-prompt instead of per-calibration-point reduces the `.npz` from ~8 GB to ~200 MB. |
| **Batch size** | Not discussed | Fixed to 1 per trajectory | DiffusionKit's `CFGDenoiser` doubles the batch internally (positive + negative prompt). Enforcing batch=1 ensures the doubled batch=2 matches the cfg_batch=2 conditioning shape. |

---

## Files

| File | Role |
|------|------|
| `calibration_config.py` | Central constants: steps, samples, timesteps, model version, latent size, CFG weight, prompt file path |
| `calibration_collector.py` | Euler sampling loop with per-step `(x_t, t)` collection. Uses `CFGDenoiser` + `cache_modulation_params` for efficient modulation. |
| `sample_cali_data.py` | Main orchestration script: loads pipeline, encodes prompts, runs 256 trajectories, selects 25 steps, flattens, shuffles, saves `.npz` |
| `sample_prompts.txt` | 20 diverse text prompts covering different visual domains |
| `__init__.py` | Package marker |

---

## Output format

The output `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `xs` | `(6400, 32, 32, 16)` | Noisy latent inputs (model inputs before each denoising step) |
| `ts` | `(6400,)` | Model timesteps (output of `sampler.timestep(sigmas)`, float16) |
| `prompt_indices` | `(6400,)` | Maps each calibration point to its prompt index in `cs`/`cs_pooled` |
| `cs` | `(20, 2, 589, 4096)` | Token-level text embeddings. Dim 1 = cfg_batch [positive, negative] |
| `cs_pooled` | `(20, 2, 2048)` | Pooled text embeddings for adaLN modulation |
| `prompts` | `(20,)` | The prompt strings (object array) |
| `cfg_scale` | scalar | CFG weight used (1.5) |

---

## Usage

### Full run (paper-scale)

```bash
cd /path/to/time-aware-MLXSD-quantization

python -m src.calibration_sample_generation.sample_cali_data \
    --output DiT_cali_data.npz
```

This generates 256 trajectories × 100 steps, selects 25, and produces 6,400 calibration points. Expect ~4–8 hours on an M-series Mac depending on model.

### Dry run (quick validation)

```bash
python -m src.calibration_sample_generation.sample_cali_data \
    --num-fid-samples 8 \
    --num-sampling-steps 20 \
    --num-selected-steps 5 \
    --output dry_run_cali.npz
```

Produces 8 × 5 = 40 calibration points in a few minutes.

### All CLI options

```
--model-version       DiffusionKit model key (default: argmaxinc/mlx-stable-diffusion-3-medium)
--num-fid-samples     Total trajectories (default: 256)
--num-sampling-steps  Denoising steps per trajectory (default: 100)
--num-selected-steps  Uniformly selected timesteps (default: 25)
--latent-size H W     Latent dimensions (default: 32 32)
--cfg-scale           Classifier-free guidance weight (default: 1.5)
--prompt-file         Path to text file with prompts (default: sample_prompts.txt)
--seed                Global random seed (default: 0)
--output / -o         Output .npz path (default: DiT_cali_data.npz)
--low-memory-mode     Enable DiffusionKit low-memory mode (default: on)
--no-low-memory-mode  Disable low-memory mode
--local-ckpt          Path to a local MMDiT checkpoint
```

---

## Key implementation details

### x_t collection point

The noisy latent `x_t` is captured **before** the model processes it at each step:

```python
for i in trange(n_steps):
    x_cali_list.append(x)        # collect BEFORE model call
    t_cali_list.append(timesteps[i])
    denoised = model(x, ...)     # model processes x
    x = x + d * dt               # Euler step
```

This is correct for calibration — we want the `(input, timestep)` pairs the model encounters during inference.

### Uniform step selection

```python
selected_indices = np.linspace(0, total_steps - 1, num_selected_steps, dtype=int)
```

For 100 steps and 25 selections, this picks indices `[0, 4, 8, ..., 96, 99]`, evenly spanning the full denoising trajectory.

### Shuffle

All 6,400 points are shuffled with a deterministic `numpy.random.default_rng(seed)` permutation, applied consistently to `xs`, `ts`, and `prompt_indices`.
