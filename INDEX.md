# Documentation Index

MLX-based SD3-Medium diffusion pipeline with TaQ-DiT W4A8 PTQ tooling.

---

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Project overview and full pipeline commands |
| [QUICKSTART_CALIBRATION.md](QUICKSTART_CALIBRATION.md) | Command reference for the full pipeline |
| [CALIBRATION_SYSTEM_SUMMARY.md](CALIBRATION_SYSTEM_SUMMARY.md) | Design decisions, data layout, time/storage estimates |
| [LESSONS_READ_SOURCE_FIRST.md](LESSONS_READ_SOURCE_FIRST.md) | Why reading DiffusionKit source first is critical |
| [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md) | All issues encountered and their fixes |
| [CLAUDE.md](CLAUDE.md) | Claude Code guidance (authoritative reference) |

Archived/historical docs: `docs/archived/`

---

## Scripts

### Step 1 — Calibration Data

| Script | Purpose | Time |
|--------|---------|------|
| `generate_calibration_data.py` | Generate per-step latents for N images | ~6 min/10 images |

### Weight Track (Steps 2W–3W)

| Script | Purpose | Time |
|--------|---------|------|
| `cache_adaround_data.py` | Cache block-level FP16 I/O for AdaRound | ~30 min/5 images |
| `adaround_optimize.py` | Optimize AdaRound W4A8 weights | varies |
| `load_adaround_model.py` | Load quantized weights and run inference | — |

### Activation Track (Steps 2A–3A)

| Script | Purpose | Time |
|--------|---------|------|
| `collect_layer_activations.py` | Collect per-channel activation stats | ~30 min/5 images |
| `analyze_activations.py` | W4A8 baseline config (faithful TaQ-DiT) | <1 min |
| `analyze_activations_multitier.py` | Experimental A4/A6/A8 config | <1 min |
| `visualize_activations.py` | Optional: plot activation statistics | — |

---

## Quick Commands

```bash
# Step 1
conda run -n diffusionkit python -m src.generate_calibration_data \
    --num-images 10 --num-steps 50 --calib-dir calibration_data

# Step 2A
conda run -n diffusionkit python -m src.collect_layer_activations \
    --calib-dir calibration_data --num-images 5 --stride 2

# Step 3A — W4A8 baseline
conda run -n diffusionkit python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --output calibration_data/activations/quant_config.json

# Step 3A-alt — experimental multitier
conda run -n diffusionkit python -m src.analyze_activations_multitier \
    --stats calibration_data/activations/layer_statistics.json \
    --output calibration_data/activations/quant_config_multitier.json
```

See [QUICKSTART_CALIBRATION.md](QUICKSTART_CALIBRATION.md) for the full two-track pipeline.

---

## Output Layout

```
calibration_data/
├── manifest.json
├── samples/                         # per-step latents
├── adaround_cache/                  # weight track
└── activations/
    ├── layer_statistics.json
    ├── quant_config.json            # W4A8 baseline
    ├── quant_config_multitier.json  # experimental
    └── timestep_stats/

quantized_weights/                   # AdaRound output
    ├── config.json
    └── weights/{block_name}.npz
```

---

## Troubleshooting

See [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md) for the full list. Key issues:

- **adaLN dimension mismatch** → Issue 4 (targeted modulation reload)
- **Hooks corrupt weights on reload** → Issue 11 (remove hooks before load_weights)
- **Images are noise** → Issues 6, 7, 8 (Euler formula, append_dims)
- **Out of memory** → Issue 10 (reduce --num-images, clear cache)
