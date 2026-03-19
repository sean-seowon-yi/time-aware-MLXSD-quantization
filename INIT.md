# time-aware-MLXSD-quantization

## Goal

Apply **Q-Diffusion** (AdaRound + BRECQ block reconstruction) to **SD3 Medium's MMDiT** architecture as a post-training quantization (PTQ) method.

**Research question**: Do AdaRound and BRECQ transfer to MMDiT's novel architectural features (dual-stream attention, flow-matching timestep schedule, adaLN modulation), and what modifications are required to make block-wise reconstruction work correctly for MMDiT transformer blocks?

---

## Architecture Snapshot: SD3 Medium MMDiT

| Property | Value |
|---|---|
| Architecture | MMDiT (Multi-Modal Diffusion Transformer) |
| Backbone | Dual-stream transformer (img + txt tokens) |
| Hidden size | 1536 |
| MLP ratio | 4 → FFN hidden = 6144 |
| Number of blocks | 24 multimodal transformer blocks (mm\_00 … mm\_23) |
| Attention | Joint QKV (img+txt concatenated before SDPA) |
| Modulation | adaLN: 6-chunk packed tensor `[β₁, γ₁, α₁, β₂, γ₂, α₂]` × 1536 |
| Conditioning | Flow-matching (not DDPM); Euler sampler |
| Text encoder | T5 + CLIP dual; text seq ≈ 77 tokens, image tokens ≈ 1024 |
| Special cases | Last txt block (mm\_23): `skip_post_sdpa=True`, `num_modulation_params=2` (no FFN); blocks with `parallel_mlp=True` share qkv/fc1 modulation |
| Inference dtype | float16 (MLX) |
| Model ID | `argmaxinc/mlx-stable-diffusion-3-medium` |

---

## Reference Method: Q-Diffusion

| Property | Q-Diffusion |
|---|---|
| Original target | U-Net (encoder–decoder with skip connections) |
| Diffusion process | DDPM |
| Core strategies | AdaRound (per-weight rounding), BRECQ (block-wise reconstruction) |
| Failure mode addressed | Weight rounding error; activation quantization error |
| Assumptions | Near-Gaussian weight distributions; block-local reconstruction loss; DDPM timestep schedule |

### MMDiT Transferability Challenges

Q-Diffusion was not designed for SD3 MMDiT. Key transfer challenges to address in Phase 2:

- **No skip connections**: BRECQ's block reconstruction loss was designed for U-Net blocks where skip connections define natural boundaries. MMDiT transformer blocks have residual connections but no long-range skips — block boundaries must be redefined around the TransformerBlock unit.
- **adaLN modulation**: Each block's output is scaled and shifted by a timestep/conditioning-dependent adaLN affine transform. The reconstruction target for BRECQ must account for this data-dependent scale, or the loss will be computed against an incorrect reference.
- **Dual-stream attention**: MMDiT concatenates img (≈1024 tokens) and txt (≈77 tokens) Q/K/V before SDPA. A per-channel scale difference between streams affects the joint attention distribution — Q-Diffusion's reconstruction loss must be evaluated on the joint output, not per-stream.
- **Flow-matching schedule**: Q-Diffusion's calibration was designed for DDPM's 1000-step noise schedule. MMDiT uses 25-step flow-matching Euler — the calibration data already collected (68 COCO × 25 steps) is appropriate.

Phase 1 EDA provides evidence for how significant each challenge is before Phase 2 implementation.

---

## Phase Map

| Phase | Name | Status | Output |
|---|---|---|---|
| Phase 1 | EDA — Activation & Weight Profiling | **COMPLETE** | `eda_output/` |
| Phase 2 | Q-Diffusion Implementation for MMDiT | Not started — defined after EDA | TBD |

Phase 2 decisions (which block boundaries to use for BRECQ, how to handle adaLN in the reconstruction loss, whether per-stream vs. joint quantization is needed) will be driven by the Phase 1 EDA results. See `phase_1.md` Section 7 for the mapping.

---

## Repository Layout

```
time-aware-MLXSD-quantization/
│
├── INIT.md                          ← this file
├── phase_1.md                       ← Phase 1 EDA methodology
├── README.md                        ← original project overview
│
├── DiffusionKit/                    ← git submodule (editable install)
│   └── python/src/diffusionkit/mlx/
│       └── mmdit.py                 ← MMDiT forward pass; hook points documented in phase_1.md
│
├── src/
│   ├── calibration_sample_generation/
│   │   ├── sample_cali_data.py      ← Phase 1 data generator (reused for COCO variant)
│   │   ├── calibration_config.py    ← constants (NUM_CALIBRATION_SAMPLES, NUM_SAMPLING_STEPS, …)
│   │   └── calibration_collector.py ← sample_euler_with_calibration(); Euler loop with x_t capture
│   │
│   ├── activation_diagnostics/
│   │   ├── activation_tracer.py     ← monkey-patches pre_sdpa / post_sdpa hooks
│   │   ├── profile_postgelu.py      ← forward-pass loop with adaLN offload guard
│   │   └── visualize_postgelu.py    ← heatmap/histogram plotting utilities
│   │
│   ├── htg_quantization/            ← Phase 3 HTG implementation (complete)
│   │   ├── apply_htg.py
│   │   ├── compute_htg_params.py
│   │   ├── htg_reparameterize.py
│   │   ├── input_activation_tracer.py
│   │   └── htg_config.py
│   │
│   └── inference/                   ← Phase 4 modular inference pipeline (complete)
│       ├── base.py                  ← InferenceTransform ABC + QuantizedInferencePipeline
│       ├── htg_transform.py         ← HTGTransform with ablation flags
│       └── run_inference.py         ← CLI entry point
│
├── htg_output/                      ← Phase 3 artifacts
│   ├── htg_corrections.npz          ← PRIMARY inference input (z_g, s, group_assignments)
│   ├── htg_params.npz
│   ├── htg_input_activation_stats.npz
│   └── htg_activation_ranges.npz
│
└── eda_output/                      ← Phase 1 output (to be created)
    ├── coco_cali_data.npz
    ├── weight_stats.npz
    ├── activation_stats_full.npz
    ├── plots/                       ← A1–A12 PNGs
    └── tables/                      ← CSV analysis outputs
```

---

## Environment Quickstart

```bash
# Activate environment
conda activate mlxsd
# Python 3.11.14, MLX 0.17.3, DiffusionKit 0.5.2

# Install DiffusionKit in editable mode (run from repo root)
pip install -e DiffusionKit/

# Run any module from repo root
python -m src.<module_path>

# Re-generate EDA plots (stats already collected)
python -m src.eda.run_eda --skip-profile

# Full EDA pipeline (re-profiles activations — slow)
python -m src.eda.run_eda
```

**Note**: Always run from repo root. The editable install path is `DiffusionKit/` (not `DiffusionKit/python/src/`). If you get `AttributeError: type object 'object' has no attribute 'pre_sdpa'`, DiffusionKit is not installed — the `except Exception` in `input_activation_tracer.py:43` silently downgrades `TransformerBlock = object`.
