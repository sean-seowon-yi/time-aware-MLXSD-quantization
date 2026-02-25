"""
Evaluate and compare quantization methods for SD3-Medium.

Generates images from up to three configurations and computes image quality
metrics vs the FP16 baseline:

  Config A — FP16 baseline: unmodified DiffusionKit pipeline
  Config B — AdaRound W4:   weight-only rounding from adaround_optimize.py
  Config C — AdaRound W4A8: weight rounding + fixed per-timestep activation
             quant from analyze_activations.py (use --adaround-act-config)
  Config D — TaQ-DiT W4A8:  joint weight+activation from taqdit_optimize.py
             (use --taqdit-output + --taqdit-act-config)

Metrics (all computed vs FP16 baseline):
  PSNR  — higher is better (pixel-level fidelity, dB)
  SSIM  — higher is better (structural similarity, 0→1)
  LPIPS — lower is better (perceptual distance; requires ``pip install lpips``)

Usage
-----
    conda run -n diffusionkit python -m src.evaluate_quantization \\
        --adaround-output quantized_weights_adaround \\
        --adaround-act-config calibration_data/activations/quant_config.json \\
        --taqdit-output quantized_weights_taqdit \\
        --taqdit-act-config quantized_weights_taqdit/taqdit_act_config.json \\
        --output-dir eval_results \\
        --num-images 10 --seed 42 --save-images

Output layout
-------------
    eval_results/
        eval_results.json        per-image + aggregate metrics
        fp16/                    FP16 baseline images (if --save-images)
        adaround/                AdaRound W4 images
        adaround_w4a8/           AdaRound W4A8 images
        taqdit_w4a8/             TaQ-DiT W4A8 images
        comparison_grid.png      N×4 side-by-side grid (if --save-images)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------------

def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio between two uint8 images (higher = more similar).

    Parameters
    ----------
    img_a, img_b : uint8 arrays of the same shape (H, W, C) or (H, W)
    """
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim_simple(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Simplified global SSIM (luminance + contrast similarity).

    Uses a single global window (equivalent to per-image statistics).
    Range is approximately [-1, 1]; 1.0 = identical images.

    Parameters
    ----------
    img_a, img_b : uint8 arrays of the same shape
    """
    a = img_a.astype(np.float64) / 255.0
    b = img_b.astype(np.float64) / 255.0
    mu_a, mu_b = float(a.mean()), float(b.mean())
    var_a = float(((a - mu_a) ** 2).mean())
    var_b = float(((b - mu_b) ** 2).mean())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    lum = (2.0 * mu_a * mu_b + c1) / (mu_a ** 2 + mu_b ** 2 + c1)
    con_struct = (2.0 * cov + c2) / (var_a + var_b + c2)
    return float(lum * con_struct)


def try_load_lpips():
    """
    Try to import the ``lpips`` library.

    Returns
    -------
    compute_fn : callable(img_a, img_b) -> float  or  None
    available  : bool
    """
    try:
        import lpips as _lpips
        import torch
        lpips_fn = _lpips.LPIPS(net="alex", verbose=False)
        lpips_fn.eval()

        def compute_lpips(img_a: np.ndarray, img_b: np.ndarray) -> float:
            """LPIPS perceptual distance (lower = more similar)."""
            def _to_tensor(arr):
                # uint8 HWC → float32 CHW in [-1, 1]
                return torch.from_numpy(
                    (arr.astype(np.float32) / 127.5 - 1.0)
                ).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                d = lpips_fn(_to_tensor(img_a), _to_tensor(img_b))
            return float(d.item())

        return compute_lpips, True
    except ImportError:
        return None, False


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------

def generate_fp16(
    pipeline,
    prompt: str,
    seed: int,
    num_steps: int,
    cfg_scale: float,
):
    """Generate an image with the unmodified FP16 pipeline."""
    from PIL.Image import Image as _PIL
    images, _ = pipeline.generate_image(
        prompt,
        cfg_weight=cfg_scale,
        num_steps=num_steps,
        seed=seed,
        negative_text="",
    )
    return images[0]


def generate_with_weights(
    pipeline,
    adaround_output: Path,
    act_config_path: Optional[Path],
    prompt: str,
    seed: int,
    num_steps: int,
    cfg_scale: float,
):
    """
    Inject quantized weights (and optionally activation quant), then generate.

    Works for both AdaRound and TaQ-DiT weight outputs since both use the
    same NPZ format produced by ``adaround_optimize.py`` / ``taqdit_optimize.py``.
    """
    from src.load_adaround_model import (
        load_adaround_weights,
        inject_weights,
        apply_act_quant_hooks,
        remove_act_quant_hooks,
        run_act_quant_inference,
    )

    config, quant_weights = load_adaround_weights(adaround_output)
    inject_weights(pipeline, quant_weights)

    if act_config_path is not None:
        with open(act_config_path) as f:
            quant_cfg = json.load(f)
        per_timestep = quant_cfg.get("per_timestep", {})
        outlier_config = quant_cfg.get("outlier_config", {})
        step_keys_sorted = sorted(int(k) for k in per_timestep.keys())
        proxies, patches = apply_act_quant_hooks(
            pipeline.mmdit, per_timestep, outlier_config
        )
        image = run_act_quant_inference(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt="",
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            proxies=proxies,
            step_keys_sorted=step_keys_sorted,
        )
        remove_act_quant_hooks(patches)
    else:
        images, _ = pipeline.generate_image(
            prompt,
            cfg_weight=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            negative_text="",
        )
        image = images[0]

    return image


# ---------------------------------------------------------------------------
# Comparison grid
# ---------------------------------------------------------------------------

def make_comparison_grid(
    images_per_config: Dict[str, List[np.ndarray]],
    config_labels: List[str],
) -> np.ndarray:
    """
    Build an N_images × N_configs tiled image grid (uint8).

    Parameters
    ----------
    images_per_config : dict mapping config_name -> list of uint8 arrays (H, W, 3)
    config_labels     : ordered list of config names to include in the grid

    Returns uint8 array (N_images * H, N_configs * W, 3).
    """
    rows = []
    n_images = len(next(iter(images_per_config.values())))
    for i in range(n_images):
        row = np.concatenate(
            [images_per_config[cfg][i] for cfg in config_labels
             if cfg in images_per_config],
            axis=1,
        )
        rows.append(row)
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    pipeline,
    prompts: List[str],
    seeds: List[int],
    adaround_output: Optional[Path],
    adaround_act_config: Optional[Path],
    taqdit_output: Optional[Path],
    taqdit_act_config: Optional[Path],
    num_steps: int = 28,
    cfg_scale: float = 7.0,
    save_images: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Generate images for all enabled configurations and compute quality metrics.

    Image generation order (to avoid repeated weight reloads):
      1. All FP16 images (no weight injection needed)
      2. All AdaRound W4 images
      3. All AdaRound W4A8 images (if act config provided)
      4. All TaQ-DiT W4A8 images

    Returns a results dict with per-image metrics and aggregate statistics.
    """
    compute_lpips, lpips_available = try_load_lpips()
    if lpips_available:
        print("  LPIPS: available (AlexNet)")
    else:
        print("  LPIPS: not available (install with: pip install lpips torch)")

    # Determine which configs are enabled
    configs: List[Tuple[str, Optional[Path], Optional[Path]]] = [
        ("fp16", None, None),
    ]
    if adaround_output is not None:
        configs.append(("adaround_w4", adaround_output, None))
        if adaround_act_config is not None:
            configs.append(("adaround_w4a8", adaround_output, adaround_act_config))
    if taqdit_output is not None:
        configs.append(("taqdit_w4a8", taqdit_output, taqdit_act_config))

    print(f"  Configs: {[c for c, _, _ in configs]}")

    # Per-config image stores
    config_images: Dict[str, List[np.ndarray]] = {c: [] for c, _, _ in configs}

    # Generate all images for each config in turn
    for cfg_name, weight_dir, act_cfg in configs:
        print(f"\n--- {cfg_name} ---")

        # Reload original weights before each quantized config
        if cfg_name != "fp16":
            pipeline.check_and_load_models()

        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            t0 = time.time()
            if cfg_name == "fp16":
                img = generate_fp16(pipeline, prompt, seed, num_steps, cfg_scale)
            else:
                img = generate_with_weights(
                    pipeline, weight_dir, act_cfg, prompt, seed, num_steps, cfg_scale
                )
            elapsed = time.time() - t0
            arr = np.array(img)
            config_images[cfg_name].append(arr)

            if save_images and output_dir is not None:
                cfg_dir = output_dir / cfg_name
                cfg_dir.mkdir(parents=True, exist_ok=True)
                img.save(cfg_dir / f"img{i:03d}.png")

            print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}s  '{prompt[:40]}'")

    # Compute metrics vs FP16
    fp16_imgs = config_images["fp16"]
    per_image_results = []
    aggregate: Dict[str, Dict] = {}
    all_metrics_by_cfg: Dict[str, Dict[str, List]] = {}

    for cfg_name, _, _ in configs:
        if cfg_name == "fp16":
            continue
        all_metrics_by_cfg[cfg_name] = {"psnr": [], "ssim": []}
        if lpips_available:
            all_metrics_by_cfg[cfg_name]["lpips"] = []

    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        img_entry = {"prompt": prompt, "seed": seed, "metrics": {}}
        fp16_arr = fp16_imgs[i]

        for cfg_name, _, _ in configs:
            if cfg_name == "fp16":
                continue
            arr = config_images[cfg_name][i]
            m: Dict = {
                "psnr": psnr(fp16_arr, arr),
                "ssim": ssim_simple(fp16_arr, arr),
            }
            if lpips_available:
                m["lpips"] = compute_lpips(fp16_arr, arr)

            img_entry["metrics"][cfg_name] = m
            all_metrics_by_cfg[cfg_name]["psnr"].append(m["psnr"])
            all_metrics_by_cfg[cfg_name]["ssim"].append(m["ssim"])
            if lpips_available:
                all_metrics_by_cfg[cfg_name]["lpips"].append(m["lpips"])

        per_image_results.append(img_entry)

    for cfg_name, metric_lists in all_metrics_by_cfg.items():
        agg = {}
        for metric, vals in metric_lists.items():
            if vals:
                agg[f"mean_{metric}"] = float(np.mean(vals))
                agg[f"std_{metric}"] = float(np.std(vals))
        aggregate[cfg_name] = agg

    # Build comparison grid
    if save_images and output_dir is not None and len(config_images["fp16"]) > 0:
        grid_labels = [c for c, _, _ in configs]
        grid = make_comparison_grid(config_images, grid_labels)
        from PIL import Image as PILImage
        PILImage.fromarray(grid).save(output_dir / "comparison_grid.png")
        print(f"\n  Comparison grid saved → {output_dir / 'comparison_grid.png'}")

    return {
        "prompts": prompts,
        "seeds": seeds,
        "configs": [c for c, _, _ in configs],
        "per_image": per_image_results,
        "aggregate": aggregate,
        "lpips_available": lpips_available,
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FP16 vs AdaRound vs TaQ-DiT quantization quality"
    )
    parser.add_argument("--adaround-output", type=Path, default=None,
                        help="AdaRound quantized weights directory (adaround_optimize.py output)")
    parser.add_argument("--adaround-act-config", type=Path, default=None,
                        help="Activation quant config for AdaRound (from analyze_activations.py)")
    parser.add_argument("--taqdit-output", type=Path, default=None,
                        help="TaQ-DiT quantized weights directory (taqdit_optimize.py output)")
    parser.add_argument("--taqdit-act-config", type=Path, default=None,
                        help="TaQ-DiT activation config (taqdit_act_config.json)")
    parser.add_argument("--prompts-file", type=Path, default=None,
                        help="Text file with one prompt per line (default: built-in set)")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"),
                        help="Directory for results JSON and optional images (default: eval_results/)")
    parser.add_argument("--num-images", type=int, default=5,
                        help="Number of evaluation images (default 5)")
    parser.add_argument("--num-steps", type=int, default=28,
                        help="Denoising steps per image (default 28)")
    parser.add_argument("--cfg-scale", type=float, default=7.0,
                        help="CFG guidance scale (default 7.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed; image i uses seed+i (default 42)")
    parser.add_argument("--save-images", action="store_true",
                        help="Save per-config images and comparison grid to output-dir")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load prompts
    # ------------------------------------------------------------------
    _DEFAULT_PROMPTS = [
        "a tabby cat sitting on a wooden table, professional photography",
        "a mountain landscape at sunset with snow-capped peaks",
        "a bowl of colorful fruit on a white marble counter",
        "an astronaut floating in space with Earth in the background",
        "a vintage red bicycle leaning against a brick wall",
        "a golden retriever puppy playing in autumn leaves",
        "a cozy coffee shop interior with warm ambient lighting",
        "a minimalist white room with a single houseplant",
        "a city skyline reflected in a calm river at dusk",
        "a close-up photograph of a blooming cherry blossom branch",
    ]

    if args.prompts_file is not None and args.prompts_file.exists():
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = _DEFAULT_PROMPTS

    prompts = prompts[:args.num_images]
    seeds = [args.seed + i for i in range(len(prompts))]

    print("=== Quantization Evaluation ===")
    print(f"  Images:          {len(prompts)}")
    print(f"  Steps:           {args.num_steps}  CFG={args.cfg_scale}")
    print(f"  AdaRound:        {args.adaround_output or 'n/a'}")
    print(f"  AdaRound act:    {args.adaround_act_config or 'n/a'}")
    print(f"  TaQ-DiT:         {args.taqdit_output or 'n/a'}")
    print(f"  TaQ-DiT act:     {args.taqdit_act_config or 'n/a'}")
    print(f"  Output dir:      {args.output_dir}")

    if args.adaround_output is None and args.taqdit_output is None:
        print("\nWarning: no quantized weight directories specified.")
        print("  Provide at least one of --adaround-output or --taqdit-output.")
        print("  Running FP16-only evaluation as a sanity check.")

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    from diffusionkit.mlx import DiffusionPipeline

    print("\n=== Loading Pipeline ===")
    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    print("✓ Pipeline loaded\n")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print("=== Running Evaluation ===")
    results = run_evaluation(
        pipeline=pipeline,
        prompts=prompts,
        seeds=seeds,
        adaround_output=args.adaround_output,
        adaround_act_config=args.adaround_act_config,
        taqdit_output=args.taqdit_output,
        taqdit_act_config=args.taqdit_act_config,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        save_images=args.save_images,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------
    # Save and print results
    # ------------------------------------------------------------------
    results_path = args.output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results → {results_path}")

    print("\n=== Aggregate Results (vs FP16 baseline) ===")
    header = f"{'Method':<22} {'PSNR (dB)':>12} {'SSIM':>8}"
    if results.get("lpips_available"):
        header += f" {'LPIPS':>8}"
    print(header)
    print("-" * (len(header) + 2))
    for cfg, agg in results["aggregate"].items():
        psnr_v = agg.get("mean_psnr", float("nan"))
        ssim_v = agg.get("mean_ssim", float("nan"))
        row = f"  {cfg:<20} {psnr_v:>12.2f} {ssim_v:>8.4f}"
        if results.get("lpips_available"):
            lpips_v = agg.get("mean_lpips", float("nan"))
            row += f" {lpips_v:>8.4f}"
        print(row)
    print()
    print("  PSNR: higher is better | SSIM: higher is better | LPIPS: lower is better")


if __name__ == "__main__":
    main()
