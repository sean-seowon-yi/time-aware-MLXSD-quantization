# MLX Project

MLX-based diffusion (SD3) pipeline with optimization experiments, sensitivity analysis, and calibration tooling.

## Calibration data (sensitivity analysis)

The notebook `notebooks/optimization_experiments.ipynb` uses calibration data for per-layer, per-bit sensitivity analysis. The helpers in `src/sensitivity_helpers.py` expect the following layout.

### Layout

- **Directory:** `calibration_data/` at the repo root (or path set via the notebook).
- **Manifest (optional):** `calibration_data/manifest.json` with:
  - `n_completed` (int): number of images for which calibration was generated.
  - `num_steps` (int): number of diffusion steps per image (e.g. 50).
  - `sample_files` (optional): list of paths relative to `calibration_data/` pointing to `.npz` sample files. If omitted, the loader looks for `calibration_data/samples/*.npz` and, if none are found, for `calibration_data/*.npz` (root-level).
- **Sample NPZ files:** Each `.npz` must contain:
  - `x`: latent input (e.g. shape `(1, H, W, C)`).
  - `timestep`: scalar or 1-D.
  - `sigma`: scalar or 1-D.
  - `conditioning`: conditioning tensor.
  - `pooled_conditioning`: pooled conditioning tensor.
  - `step_index`: integer step index for timestep bucketing (early/mid/late).
  The loader also accepts alternate keys: `latent` or `arr_0` for `x`; `arr_1`..`arr_5` for the rest (e.g. from `np.savez` with positional args).

Samples are assumed ordered as: image 0 step 0, image 0 step 1, …, image 1 step 0, etc. The notebook loads a subset using `max_images` and `step_stride` (e.g. every 5th step).

### Generating calibration data

From the repo root, run the calibration generator so that `calibration_data/samples/` is populated with `.npz` files and `manifest.json` is updated:

```bash
python -m src.generate_calibration_data --num-images 10
```

Options: `--num-steps` (default 50), `--cfg-weight` (default 7.5), `--calib-dir`, `--prompt-csv` (default `all_prompts.csv`), `--seed`. The script uses the same pipeline config as the notebook (SD3 medium, no T5) and writes one `.npz` per (image, step) with keys `x`, `timestep`, `sigma`, `conditioning`, `pooled_conditioning`, `step_index`.

## Running the sensitivity notebook

1. Install dependencies (e.g. `pip install -e DiffusionKit`, MLX).
2. Run the notebook from the repo root or from `notebooks/`; the sensitivity cell detects the repo root via `src/sensitivity_helpers.py` and adds it to `sys.path` so `src.sensitivity_helpers` imports succeed.
3. Ensure the pipeline and calibration data are available before running the sensitivity analysis cell.
