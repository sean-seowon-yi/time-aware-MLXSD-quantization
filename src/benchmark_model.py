"""
Benchmark the SD3-Medium pipeline for a given quantization config.

Measures distributional image quality (FID / IS / KID via torch-fidelity),
per-image latency statistics, and peak memory usage (Metal + system RSS).

Phases
------
Phase 1 — Generate images
    Reads prompts from all_prompts.csv, generates N images using seed+i for
    image i, saves to output_dir/images/{idx:04d}.png.  ``--resume`` skips
    images whose PNG already exists.

Phase 2 — Compute metrics
    FID / IS / KID: requires ``pip install torch-fidelity``.  Gracefully
    degrades (logs warning, writes null) if the package is not available.
    Pass ``--reference-dir`` to trigger this phase.

Usage
-----
    # FP16 baseline: generate 500 images
    conda run -n diffusionkit python -m src.benchmark_model \\
        --config fp16 --num-images 500 --num-steps 28 \\
        --output-dir benchmark_results/fp16 --resume

    # AdaRound W4: generate + compute metrics in one shot
    conda run -n diffusionkit python -m src.benchmark_model \\
        --config adaround_w4 --adaround-output quantized_weights \\
        --num-images 150 --num-steps 28 \\
        --reference-dir calibration_data_100/images \\
        --output-dir benchmark_results/adaround_w4 --resume

    # Metrics only (reference images already generated)
    conda run -n diffusionkit python -m src.benchmark_model \\
        --skip-generation \\
        --generated-dir benchmark_results/adaround_w4/images \\
        --reference-dir calibration_data_100/images \\
        --output-dir benchmark_results/adaround_w4
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Section 2 — Latency statistics
# ---------------------------------------------------------------------------

def compute_latency_stats(timings: List[float], warmup: int = 0) -> Dict:
    """
    Compute latency statistics from a list of per-image timings.

    Parameters
    ----------
    timings : list of float
        Per-image wall-clock times in seconds (including warmup images).
    warmup : int
        Number of leading images to exclude from statistics.

    Returns
    -------
    dict with keys: mean_s, std_s, p50_s, p95_s, min_s, max_s,
                    warmup_images, measured_images.
    """
    measured = timings[warmup:]
    if not measured:
        return {
            "mean_s": None, "std_s": None, "p50_s": None, "p95_s": None,
            "min_s": None, "max_s": None,
            "warmup_images": warmup, "measured_images": 0,
        }
    arr = np.array(measured, dtype=np.float64)
    return {
        "mean_s": float(np.mean(arr)),
        "std_s": float(np.std(arr)),
        "p50_s": float(np.percentile(arr, 50)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "warmup_images": warmup,
        "measured_images": len(measured),
    }


# ---------------------------------------------------------------------------
# Section 4 — Memory statistics
# ---------------------------------------------------------------------------

def sample_metal_memory() -> Dict:
    """
    Sample current MLX Metal memory usage.

    Returns dict with keys 'active_mb' and 'peak_mb'.  Falls back to zeros
    if the MLX metal API is not available (non-Apple platform or old MLX).
    """
    try:
        import mlx.core as mx
        active_bytes = mx.metal.get_active_memory()
        peak_bytes = mx.metal.get_peak_memory()
        return {
            "active_mb": active_bytes / 1e6,
            "peak_mb": peak_bytes / 1e6,
        }
    except Exception:
        return {"active_mb": 0.0, "peak_mb": 0.0}


def reset_metal_peak_memory() -> None:
    """Reset MLX Metal peak memory counter (no-op if unavailable)."""
    try:
        import mlx.core as mx
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def sample_system_rss_mb() -> float:
    """Return current process RSS in MB via psutil, or 0.0 if unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Section 3 — FID / IS / KID
# ---------------------------------------------------------------------------

def compute_fidelity_metrics(
    generated_dir: str,
    reference_dir: str,
) -> Optional[Dict]:
    """
    Compute FID, IS, and KID between two image directories.

    Requires ``pip install torch-fidelity``.  Returns None gracefully if the
    package is not installed.

    Parameters
    ----------
    generated_dir : str | Path
        Directory containing generated PNG images.
    reference_dir : str | Path
        Directory containing reference (ground-truth) PNG images.

    Returns
    -------
    dict with keys fid, isc_mean, isc_std, kid_mean, kid_std,
    or None if torch-fidelity is unavailable.
    """
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        print("WARNING: torch-fidelity not installed — skipping FID/IS/KID. "
              "Install with: pip install torch-fidelity")
        return None

    n_gen = len(list(Path(generated_dir).glob("*.png")))
    n_ref = len(list(Path(reference_dir).glob("*.png")))
    kid_subset_size = min(n_gen, n_ref, 1000)

    metrics = calculate_metrics(
        input1=str(generated_dir),
        input2=str(reference_dir),
        fid=True,
        isc=True,
        kid=True,
        kid_subset_size=kid_subset_size,
        verbose=False,
        cuda=False,
        save_cpu_ram=True,  # forces num_workers=0, avoids shm_manager on macOS
    )
    return {
        "fid": float(metrics.get("frechet_inception_distance", float("nan"))),
        "isc_mean": float(metrics.get("inception_score_mean", float("nan"))),
        "isc_std": float(metrics.get("inception_score_std", float("nan"))),
        "kid_mean": float(metrics.get("kernel_inception_distance_mean", float("nan"))),
        "kid_std": float(metrics.get("kernel_inception_distance_std", float("nan"))),
    }


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(csv_path: Path, max_count: int) -> List[str]:
    """
    Load up to max_count prompts from a CSV with a single 'prompt' column.
    Falls back to three synthetic prompts if the file does not exist.
    """
    if not csv_path.exists():
        fallback = [
            "a photo of a cat",
            "abstract art with vibrant colors",
            "a landscape with mountains",
        ]
        return fallback[:max_count]

    prompts: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(prompts) >= max_count:
                break
            p = row.get("prompt", "").strip()
            if p:
                prompts.append(p)
    return prompts


# ---------------------------------------------------------------------------
# Section 1 — Image generation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Naive int8 quantization helpers
# ---------------------------------------------------------------------------

def _walk_mmdit_linears(mmdit):
    """
    Generator yielding (parent_obj, attr_name, full_name) for every
    nn.Linear / nn.QuantizedLinear in the DiT transformer blocks.

    Skips adaLN_modulation and any identity projections.
    Works on both pre- and post-quantization models.
    """
    import mlx.nn as nn

    def _walk_block(tb, prefix):
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layer = getattr(tb.attn, attr, None)
            if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                yield (tb.attn, attr, f"{prefix}.attn.{attr}")
        if hasattr(tb, "mlp"):
            for attr in ("fc1", "fc2"):
                layer = getattr(tb.mlp, attr, None)
                if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                    yield (tb.mlp, attr, f"{prefix}.mlp.{attr}")

    if hasattr(mmdit, "multimodal_transformer_blocks"):
        for i, block in enumerate(mmdit.multimodal_transformer_blocks):
            yield from _walk_block(block.image_transformer_block, f"mm{i}.img")
            yield from _walk_block(block.text_transformer_block, f"mm{i}.txt")

    if hasattr(mmdit, "unified_transformer_blocks"):
        for i, block in enumerate(mmdit.unified_transformer_blocks):
            yield from _walk_block(block.transformer_block, f"uni{i}")


def inject_weights_naive_int8(
    pipeline,
    group_size: int = 64,
    bits: int = 8,
) -> int:
    """
    Quantize all eligible nn.Linear weights in the DiT to int8 using
    mlx.quantize and replace them with nn.QuantizedLinear in-place.

    Layers with in_features < max(128, group_size) are skipped (MLX
    minimum-column constraint).

    Returns the count of injected layers.
    """
    import mlx.core as mx
    import mlx.nn as nn

    mmdit = pipeline.mmdit
    min_cols = max(128, group_size)
    pending = []
    count = 0

    for parent, attr, full_name in _walk_mmdit_linears(mmdit):
        layer = getattr(parent, attr)
        w = layer.weight
        in_features = w.shape[1]
        out_features = w.shape[0]

        if in_features < min_cols:
            print(f"WARNING: Skipping {full_name} "
                  f"(in_features={in_features} < {min_cols})")
            continue

        w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        has_bias = getattr(layer, "bias", None) is not None

        ql = nn.QuantizedLinear(
            in_features, out_features,
            bias=has_bias, group_size=group_size, bits=bits,
        )
        ql.weight = w_q
        ql.scales = scales
        ql.biases = biases
        if has_bias:
            ql.bias = layer.bias

        setattr(parent, attr, ql)
        pending.extend([ql.weight, ql.scales, ql.biases])
        count += 1

    if pending:
        mx.eval(*pending)

    return count


class _DynamicInt8ActLayer:
    """
    Proxy that applies dynamic per-tensor symmetric int8 fake-quantization
    to the input activation before forwarding to the wrapped layer.

    scale = max(|x|) / 127
    x_q   = round(x / scale).clip(-127, 127) * scale
    """

    def __init__(self, layer):
        self.layer = layer

    def __call__(self, x):
        import mlx.core as mx
        scale = float(mx.max(mx.abs(x)).item()) / 127.0
        if scale < 1e-8:
            return self.layer(x)
        x = mx.clip(mx.round(x / scale), -127, 127) * scale
        return self.layer(x)

    def __getattr__(self, name):
        return getattr(self.layer, name)


def apply_dynamic_int8_act_hooks(mmdit):
    """
    Wrap every walked linear layer with _DynamicInt8ActLayer.

    Returns (proxies, patches) where patches is a list of
    (parent, attr, original_layer) tuples used for cleanup.
    """
    proxies = []
    patches = []

    for parent, attr, _ in _walk_mmdit_linears(mmdit):
        layer = getattr(parent, attr)
        proxy = _DynamicInt8ActLayer(layer)
        setattr(parent, attr, proxy)
        proxies.append(proxy)
        patches.append((parent, attr, layer))

    return proxies, patches


def remove_dynamic_int8_act_hooks(patches) -> None:
    """Restore original layers from (parent, attr, original) patch tuples."""
    for parent, attr, original in patches:
        setattr(parent, attr, original)


def _load_pipeline(
    config: str,
    adaround_output: Optional[Path],
    adaround_act_config: Optional[Path],
    mlx_int4: bool = False,
    group_size: int = 64,
):
    """
    Load DiffusionPipeline and apply quantization config.

    Returns (pipeline, quant_ctx) where quant_ctx is a dict with keys:
      proxies, act_quant_patches, step_keys_sorted
    needed for V2 activation-quantized inference.
    """
    from diffusionkit.mlx import DiffusionPipeline

    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()

    quant_ctx = {
        "proxies": [],
        "act_quant_patches": [],
        "step_keys_sorted": [],
        "remove_act_fn": None,
    }

    if config == "fp16":
        return pipeline, quant_ctx

    if config == "naive_int8":
        inject_weights_naive_int8(pipeline, group_size=group_size)
        _, patches = apply_dynamic_int8_act_hooks(pipeline.mmdit)
        quant_ctx["act_quant_patches"] = patches
        quant_ctx["remove_act_fn"] = remove_dynamic_int8_act_hooks
        return pipeline, quant_ctx

    # Weight injection (adaround_w4 / adaround_w4a8 / taqdit_w4a8 / mlx_int4)
    if adaround_output is not None:
        from src.load_adaround_model import (
            load_adaround_weights,
            inject_weights,
            inject_weights_mlx_int4,
            apply_act_quant_hooks,
        )
        _, quant_weights = load_adaround_weights(adaround_output)

        if mlx_int4:
            inject_weights_mlx_int4(pipeline, quant_weights, group_size=group_size)
        else:
            inject_weights(pipeline, quant_weights)

    # Activation quantization hooks
    act_config_path = adaround_act_config
    if act_config_path is not None:
        from src.load_adaround_model import apply_act_quant_hooks
        with open(act_config_path) as f:
            quant_cfg = json.load(f)
        per_timestep = quant_cfg.get("per_timestep", {})
        outlier_config = quant_cfg.get("outlier_config", {})
        step_keys_sorted = sorted(int(k) for k in per_timestep.keys())
        proxies, patches = apply_act_quant_hooks(
            pipeline.mmdit, per_timestep, outlier_config
        )
        quant_ctx["proxies"] = proxies
        quant_ctx["act_quant_patches"] = patches
        quant_ctx["step_keys_sorted"] = step_keys_sorted

    return pipeline, quant_ctx


def _generate_single_image(
    pipeline,
    quant_ctx: Dict,
    prompt: str,
    seed: int,
    num_steps: int,
    cfg_scale: float,
):
    """
    Generate one image using the given pipeline + quant_ctx.
    Returns a PIL.Image.
    """
    proxies = quant_ctx.get("proxies", [])
    step_keys_sorted = quant_ctx.get("step_keys_sorted", [])

    if proxies:
        from src.load_adaround_model import run_act_quant_inference
        return run_act_quant_inference(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt="",
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            proxies=proxies,
            step_keys_sorted=step_keys_sorted,
        )
    else:
        images, _ = pipeline.generate_image(
            prompt,
            cfg_weight=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            negative_text="",
        )
        return images


def generate_images(
    config: str,
    prompts: List[str],
    output_dir: Path,
    num_steps: int,
    cfg_scale: float,
    seed_base: int,
    warmup: int,
    resume: bool,
    adaround_output: Optional[Path] = None,
    adaround_act_config: Optional[Path] = None,
    mlx_int4: bool = False,
    group_size: int = 64,
) -> Tuple[List[float], Dict]:
    """
    Generate images for all prompts and return timing + memory stats.

    Pipeline is reloaded per image (mirrors generate_calibration_data.py)
    for consistent memory measurements.

    Returns
    -------
    timings : list of float
        Per-image wall-clock seconds (all images, including warmup).
    memory_stats : dict
        peak_metal_mb, peak_rss_mb.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    timings: List[float] = []
    peak_metal_mb = 0.0
    peak_rss_mb = 0.0

    total = len(prompts)
    completed = 0

    pbar = tqdm(enumerate(prompts), total=len(prompts),
                desc=f"  {config}", unit="img", dynamic_ncols=True)
    for img_idx, prompt in pbar:
        img_path = images_dir / f"{img_idx:04d}.png"
        if resume and img_path.exists():
            print(f"  [resume] skipping {img_idx:04d}.png")
            timings.append(0.0)   # placeholder so indices stay aligned
            continue

        seed = seed_base + img_idx
        reset_metal_peak_memory()

        t0 = time.time()
        pipeline, quant_ctx = _load_pipeline(
            config, adaround_output, adaround_act_config, mlx_int4, group_size
        )
        image = _generate_single_image(
            pipeline, quant_ctx, prompt, seed, num_steps, cfg_scale
        )
        elapsed = time.time() - t0

        # Remove hooks before pipeline goes out of scope
        if quant_ctx["act_quant_patches"]:
            remove_fn = quant_ctx.get("remove_act_fn")
            if remove_fn is not None:
                remove_fn(quant_ctx["act_quant_patches"])
            else:
                from src.load_adaround_model import remove_act_quant_hooks
                remove_act_quant_hooks(quant_ctx["act_quant_patches"])

        image.save(img_path)
        timings.append(elapsed)
        completed += 1

        # Memory
        mem = sample_metal_memory()
        peak_metal_mb = max(peak_metal_mb, mem["peak_mb"])
        peak_rss_mb = max(peak_rss_mb, sample_system_rss_mb())

        # ETA
        measured = [t for t in timings if t > 0]
        measured_after_warmup = measured[warmup:]
        if measured_after_warmup:
            mean_t = np.mean(measured_after_warmup)
            remaining_imgs = total - img_idx - 1
            eta_min = remaining_imgs * mean_t / 60.0
            eta_str = f"ETA {eta_min:.1f} min"
        else:
            eta_str = "ETA --"

        metal_gb = mem["peak_mb"] / 1000.0
        print(f"  [{img_idx + 1}/{total}] {elapsed:.1f}s | {eta_str} | "
              f"peak_metal {metal_gb:.1f} GB")
        pbar.set_postfix({
            "s/img": f"{elapsed:.1f}",
            "ETA": eta_str,
            "metal_GB": f"{peak_metal_mb / 1024:.1f}",
        }, refresh=True)

    return timings, {"peak_metal_mb": peak_metal_mb, "peak_rss_mb": peak_rss_mb}


# ---------------------------------------------------------------------------
# Section 5 — Console output helpers
# ---------------------------------------------------------------------------

def _print_results(config: str, lat: Dict, mem: Dict, fidelity: Optional[Dict]) -> None:
    print(f"\n{'='*50}")
    print(f"=== Benchmark Results: {config} ===")
    print(f"{'='*50}")

    if lat.get("measured_images"):
        print(f"Latency (s):   mean={lat['mean_s']:.1f}  std={lat['std_s']:.1f}  "
              f"p50={lat['p50_s']:.1f}  p95={lat['p95_s']:.1f}")
    else:
        print("Latency:       (no measured images)")

    metal_gb = (mem.get("peak_metal_mb") or 0.0) / 1000.0
    rss_gb = (mem.get("peak_rss_mb") or 0.0) / 1000.0
    print(f"Memory:        peak_metal={metal_gb:.1f} GB  peak_rss={rss_gb:.1f} GB")

    if fidelity is not None:
        print(f"FID:           {fidelity['fid']:.4f}")
        print(f"IS:            {fidelity['isc_mean']:.2f} ± {fidelity['isc_std']:.2f}")
        print(f"KID:           {fidelity['kid_mean']:.5f} ± {fidelity['kid_std']:.5f}")
    else:
        print("FID/IS/KID:    skipped")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SD3-Medium pipeline: generate images + compute FID/IS/KID"
    )
    # Config
    parser.add_argument(
        "--config", type=str, default="fp16",
        choices=["fp16", "naive_int8", "adaround_w4", "adaround_w4a8", "taqdit_w4a8", "mlx_int4"],
        help="Quantization config to benchmark",
    )
    parser.add_argument("--adaround-output", type=Path, default=None,
                        help="AdaRound weights dir (from adaround_optimize.py)")
    parser.add_argument("--adaround-act-config", type=Path, default=None,
                        help="Activation quant config JSON (from analyze_activations.py)")
    parser.add_argument("--taqdit-output", type=Path, default=None,
                        help="TaQ-DiT weights dir (reserved for future use)")
    parser.add_argument("--taqdit-act-config", type=Path, default=None,
                        help="TaQ-DiT act config JSON (reserved for future use)")

    # Generation
    parser.add_argument("--prompt-csv", type=Path, default=None,
                        help="CSV file with 'prompt' column (default: all_prompts.csv)")
    parser.add_argument("--num-images", type=int, default=150,
                        help="Number of images to generate (default: 150)")
    parser.add_argument("--num-steps", type=int, default=28,
                        help="Denoising steps per image (default: 28)")
    parser.add_argument("--cfg-scale", type=float, default=7.0,
                        help="CFG guidance weight (default: 7.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; image i uses seed+i (default: 42)")
    parser.add_argument("--mlx-int4", action="store_true",
                        help="Inject AdaRound weights as native MLX int4")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Group size for MLX int4 (default: 64)")

    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"),
                        help="Root output dir (default: benchmark_results)")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="Reference image dir for FID/IS/KID (omit to skip metrics)")
    parser.add_argument("--generated-dir", type=Path, default=None,
                        help="Override generated image dir for metrics phase")

    # Phase control
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip image generation; compute metrics only")
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip FID/IS/KID; only generate + latency/memory")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup images excluded from latency stats (default: 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images whose PNG already exists in output_dir/images/")

    args = parser.parse_args()

    prompt_csv = args.prompt_csv or (_REPO / "all_prompts.csv")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = args.generated_dir or (output_dir / "images")

    timings: List[float] = []
    memory_stats: Dict = {"peak_metal_mb": 0.0, "peak_rss_mb": 0.0}
    fidelity_result: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Phase 1 — Image generation
    # ------------------------------------------------------------------
    if not args.skip_generation:
        prompts = load_prompts(prompt_csv, args.num_images)
        print(f"=== Generating {len(prompts)} images (config={args.config}) ===")
        print(f"  Output: {output_dir / 'images'}")
        if args.resume:
            print("  Resume mode: existing PNGs will be skipped")

        timings, memory_stats = generate_images(
            config=args.config,
            prompts=prompts,
            output_dir=output_dir,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            seed_base=args.seed,
            warmup=args.warmup,
            resume=args.resume,
            adaround_output=args.adaround_output,
            adaround_act_config=args.adaround_act_config,
            mlx_int4=args.mlx_int4,
            group_size=args.group_size,
        )

    # ------------------------------------------------------------------
    # Phase 2 — Fidelity metrics
    # ------------------------------------------------------------------
    if not args.skip_metrics and args.reference_dir is not None:
        ref_dir = args.reference_dir
        gen_dir = generated_dir

        if not gen_dir.exists():
            print(f"WARNING: generated_dir {gen_dir} does not exist — skipping metrics")
        elif not ref_dir.exists():
            print(f"WARNING: reference_dir {ref_dir} does not exist — skipping metrics")
        else:
            n_gen = len(list(gen_dir.glob("*.png")))
            n_ref = len(list(ref_dir.glob("*.png")))
            print(f"\n=== Computing FID/IS/KID "
                  f"({n_gen} generated vs {n_ref} reference) ===")
            raw = compute_fidelity_metrics(gen_dir, ref_dir)
            if raw is not None:
                fidelity_result = {
                    **raw,
                    "reference_dir": str(ref_dir),
                    "num_reference_images": n_ref,
                    "num_generated_images": n_gen,
                }

    # ------------------------------------------------------------------
    # Latency stats
    # ------------------------------------------------------------------
    # Filter out resume-skipped images (timings == 0.0) before computing stats
    real_timings = [t for t in timings if t > 0.0]
    lat_stats = compute_latency_stats(real_timings, warmup=args.warmup)

    # ------------------------------------------------------------------
    # Write benchmark.json
    # ------------------------------------------------------------------
    benchmark = {
        "config": args.config,
        "num_images": args.num_images,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "latency": lat_stats,
        "memory": memory_stats,
        "fidelity": fidelity_result,
    }

    json_path = output_dir / "benchmark.json"
    with open(json_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n✓ benchmark.json → {json_path}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    _print_results(args.config, lat_stats, memory_stats, fidelity_result)
    print("\n✓ Complete")


if __name__ == "__main__":
    main()
