# Plan: Animate per-channel distribution evolution

## Goal
Add `--animate` flag to `visualize_activations.py` that generates GIFs showing how
per-channel absmax distributions evolve across the denoising trajectory for:
- Top N most-variable post-GELU layers (`.mlp.fc2`), default 3
- 1 least-variable layer (any type), for contrast

## Method
Reuse existing `plot_channel_dist` layout (same two-panel: sorted bar + histogram).
Render each timestep frame to an in-memory buffer (no temp files on disk), assemble
with Pillow `Image.save(save_all=True)`.

## Key design decisions
- **Fixed axes across all frames** so motion is legible:
  - Left panel y-axis: `max(absmax)` across all timesteps × 1.05
  - Right panel x-axis: same global max
  - Right panel y-axis (count): max histogram count across all frames
- **Frame rate**: 3 fps (333 ms/frame) — 25 timesteps ≈ 8 s per GIF
- **Sigma direction**: frames ordered high-σ → low-σ (high noise → clean image)
- **Layer selection**: compute variability (max/min absmax ratio) separately for
  post-GELU and all layers; pick top N post-GELU + 1 global minimum

## Changes
- `src/visualize_activations.py`: add `animate_channel_dist()` function + `--animate`
  and `--animate-n` CLI flags
- No new test file (output is visual; logic is thin wrapper over existing plot code)

## Output
`{out_dir}/anim_{safe_layer_name}.gif` — one GIF per animated layer
