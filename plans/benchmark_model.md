# Benchmark Model Pipeline

## Goal

Measure whether quantization degrades the model's output distribution.
Complements the existing per-image PSNR/SSIM/LPIPS in `evaluate_quantization.py`
with distributional metrics (FID, IS, KID), latency profiling, and memory profiling.

## Method

1. **Phase 1 – Generate images** with a given config (fp16 / adaround_w4 / adaround_w4a8 / taqdit_w4a8 / mlx_int4) using the same prompts and seeds as the FP16 reference set.
2. **Phase 2 – Compute metrics**: FID / IS / KID via `torch-fidelity`; latency mean/std/p50/p95/min/max; peak Metal + RSS memory.
3. `--resume` skips images whose PNG already exists in `output_dir/images/`.
4. `--skip-generation` / `--skip-metrics` let phases run independently.

## Key Design Decisions

- Seed for image `i` = `base_seed + i` (matches `generate_calibration_data.py`).
- Model loading reuses `load_adaround_model.py` helpers: `load_adaround_weights`, `inject_weights`, `apply_act_quant_hooks`, `run_act_quant_inference`.
- Memory: `mx.metal.get_active_memory()` / `mx.metal.get_peak_memory()` wrapped in `try/except` for MLX version tolerance; fallback to `psutil` RSS.
- `torch-fidelity` graceful degradation: if not installed, `fidelity` key is `null` in JSON and a warning is printed.
- Pipeline is reloaded **per image** (matches `generate_calibration_data.py` pattern) for consistent memory measurements.

## Files to Change

| File | Change |
|------|--------|
| `src/benchmark_model.py` | NEW |
| `tests/test_benchmark_model.py` | NEW |
| `plans/benchmark_model.md` | THIS FILE |

## Output Format

```
benchmark_results/{config}/
├── images/
│   ├── 0000.png
│   └── ...
└── benchmark.json
```

`benchmark.json` schema: `config`, `num_images`, `num_steps`, `cfg_scale`, `seed`,
`latency` (mean/std/p50/p95/min/max/warmup_images/measured_images),
`memory` (peak_metal_mb / peak_rss_mb),
`fidelity` (fid / isc_mean / isc_std / kid_mean / kid_std / reference_dir / counts).

## Verification

```bash
# Unit tests
conda run -n diffusionkit python -m pytest tests/test_benchmark_model.py -v

# Smoke: FP16, 3 images, skip metrics
conda run -n diffusionkit python -m src.benchmark_model \
    --config fp16 --num-images 3 --num-steps 5 \
    --output-dir /tmp/bench_smoke --skip-metrics

# Metrics only (after smoke)
conda run -n diffusionkit python -m src.benchmark_model \
    --skip-generation \
    --generated-dir /tmp/bench_smoke/images \
    --reference-dir calibration_data_100/images \
    --output-dir /tmp/bench_smoke
```
