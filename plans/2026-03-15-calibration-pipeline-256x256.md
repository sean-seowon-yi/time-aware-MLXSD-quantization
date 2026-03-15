# Plan: Fresh 256×256 Calibration Pipeline

## Context

The existing `calibration_data_100` was captured at 512×512 (64×64 latents) using HuggingFace
prompts for activation statistics and COCO prompts for the adaround cache. This resolution and
prompt mismatch makes the poly schedule unreliable for 256×256 inference (32×32 latents, 256
image tokens vs 1024). A clean rebuild at 256×256 with matched data throughout is required
before starting adaround from scratch.

**Target settings:** 256×256 images, 32×32 latents, cfg=1.5, seed=42, 25 denoising steps,
100 COCO prompts from `coco_prompts.csv` (10k prompts available).

---

## Pipeline Overview

```
coco_prompts.csv
       │
       ├─► generate_calibration_data.py ──► calibration_data_256/
       │       (images + trajectories)         manifest.json
       │                                        samples/{img}_{step}.npz
       │                                        images/, latents/
       │
       ├─► sample_cali_data.py ──► DiT_cali_data_256.npz
       │       (latents + text embeds)
       │
       │       ┌──────────────────────────────────┐
       │       │ (future: poly schedule pipeline) │
       │       └──────────────────────────────────┘
       │
       ├─► cache_adaround_data.py ──► calibration_data_256/adaround_cache/
       │       (block FP16 I/O)
       │
       └─► adaround_optimize.py ──► quantized_weights_256/
               (W4A8 optimization)
```

---

## Step 1: Restore and patch `generate_calibration_data.py`

The script was deleted in commit 3252ca7 but exists in commit e251a18. Restore it and add
`--image-size` support (the old script had no size control; SD3 defaults to 1024×1024).

```bash
git show e251a18:src/generate_calibration_data.py > src/generate_calibration_data.py
```

**Code change required** — add to argparse in the restored script:
```python
parser.add_argument("--image-size", type=int, default=256,
                    help="Output image size in pixels (default 256). "
                         "Latent will be image_size // 8.")
```
And pass `height=args.image_size, width=args.image_size` to the pipeline generate call
(DiffusionKit's `pipeline.generate()` accepts `height` and `width` kwargs).

**File to modify:** `src/generate_calibration_data.py`

---

## Step 2: Generate calibration images + trajectories

```bash
conda run --no-capture-output -n diffusionkit python -m src.generate_calibration_data \
  --num-images 100 \
  --num-steps 25 \
  --cfg-weight 1.5 \
  --seed 42 \
  --image-size 256 \
  --calib-dir calibration_data_256 \
  --prompt-csv coco_prompts.csv
```

**Output:** `calibration_data_256/` with manifest.json, 100 images × 25 step `.npz` files
(2500 trajectory samples), images/, latents/.

**Estimated cache size:** ~6 MB/sample at 256×256 → ~15 GB total. (vs 152 GB at 512×512)

---

## Step 3: Capture activations (for future poly schedule)

Run `sample_cali_data.py` (which already defaults to 32×32 latents) with matching settings
to produce a `DiT_cali_data_256.npz` for eventual poly schedule generation:

```bash
conda run --no-capture-output -n diffusionkit python -m src.calibration_sample_generation.sample_cali_data \
  --latent-size 32 32 \
  --cfg-scale 1.5 \
  --seed 42 \
  --num-fid-samples 100 \
  --num-sampling-steps 25 \
  --num-selected-steps 25 \
  --prompt-file coco_prompts.csv \
  --output DiT_cali_data_256.npz
```

Then profile activations:
```bash
conda run --no-capture-output -n diffusionkit python -m src.activation_diagnostics.profile_postgelu \
  --calibration-file DiT_cali_data_256.npz \
  --num-samples 512 \
  --output activation_stats_256.npz
```

**Note:** These activations are for future poly schedule work. Not required for the immediate
no-poly adaround run.

---

## Step 4: Generate adaround cache

```bash
conda run --no-capture-output -n diffusionkit python -m src.cache_adaround_data \
  --calib-dir calibration_data_256 \
  --output-dir calibration_data_256/adaround_cache \
  --num-images 100 \
  --stride 1 \
  --force
```

`--stride 1` uses all 25 saved timesteps (25 steps total vs 100 with stride=4 previously).
Gives 100 × 25 = 2500 calibration samples — 20× more than the current 125.

---

## Step 5: Run adaround from scratch

```bash
conda run --no-capture-output -n diffusionkit python -m src.adaround_optimize \
  --adaround-cache calibration_data_256/adaround_cache \
  --output quantized_weights_256 \
  --iters 1000 \
  --batch-size 4 \
  --grad-accum-steps 4
```

No `--refine` (fresh run). No `--poly-schedule` (baseline). Effective batch = 16.

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/generate_calibration_data.py` | Restore from git + add `--image-size` arg |

---

## Verification

1. After Step 2: confirm `calibration_data_256/manifest.json` has `latent_size: [32, 32]`,
   `cfg_scale: 1.5`, `num_steps: 25`
2. After Step 4: confirm `adaround_cache/metadata.json` shows `arg_shapes` with 256 image
   tokens (`[2, 256, 1, 1536]`) instead of 1024
3. After Step 5: confirm config.json is written incrementally after each block
