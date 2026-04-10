"""
Ground-truth comparison benchmark for FP16 vs W4A8.

Given directories of ground-truth images, FP16-generated PNGs, and a Phase-2
quantized checkpoint, optionally generates W4A8 images (same prompts / indices
as the FP16 run), then reports:

- FID: FP16 vs GT, W4A8 vs GT
- CMMD (CLIP embeddings): FP16 vs GT, W4A8 vs GT
- CLIP image-text score (mean cosine): FP16 vs prompts, W4A8 vs prompts
- LPIPS: W4A8 vs FP16 (paired by filename)

Example
-------
    python -m src.benchmark.gt_comparison_pipeline \\
        --ground-truth-dir /path/to/gt \\
        --fp16-images-dir benchmark_results/fp16_p2/images \\
        --quantized-dir quantized/w4a8_tag \\
        --output-dir benchmark_results/w4a8_gt_eval \\
        --prompt-file src/settings/evaluation_set.txt

Requires: torch-fidelity (FID), open_clip_torch (CMMD + CLIP score), lpips (LPIPS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_PROMPT_FILE = _REPO / "src" / "settings" / "evaluation_set.txt"
_PHASE2_CONFIGS = frozenset({"w4a8", "w4a8_static", "w4a8_poly"})


def _infer_img_digits(img_dir: Path) -> int:
    from src.benchmark.image_io import list_png_paths

    paths = list_png_paths(img_dir)
    if not paths:
        return 3
    return max(len(p.stem) for p in paths)


def _w4a8_images_complete(images_dir: Path, img_digits: int, n: int) -> bool:
    for i in range(n):
        if not (images_dir / f"{i:0{img_digits}d}.png").is_file():
            return False
    return True


def _resolve_poly_schedule(path: Optional[Path]) -> Optional[Dict]:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def run_pipeline(
    ground_truth_dir: Path,
    fp16_images_dir: Path,
    quantized_dir: Path,
    output_dir: Path,
    *,
    prompt_file: Path,
    config: str = "w4a8",
    num_images: Optional[int] = None,
    img_digits: Optional[int] = None,
    num_steps: int = 30,
    cfg_scale: float = 4.0,
    seed: int = 42,
    warmup: int = 0,
    resume: bool = True,
    reload_n: Optional[int] = 1,
    force_w4a8_gen: bool = False,
    poly_schedule_path: Optional[Path] = None,
    group_size: int = 64,
    kid_subset_max: int = 1000,
    fidelity_cuda: bool = False,
    clip_arch: str = "ViT-L-14-336",
    clip_pretrained: str = "openai",
    clip_batch_size: int = 16,
    lpips_net: str = "alex",
    lpips_resize: int = 256,
) -> Dict[str, Any]:
    from src.benchmark.generate import generate_images
    from src.benchmark.image_io import list_image_paths, list_png_paths
    from src.benchmark.metric_clip import (
        compute_clip_embeddings,
        compute_clip_image_text_scores,
        compute_cmmd_from_embeddings,
    )
    from src.benchmark.metric_fidelity_torch import compute_fidelity_metrics
    from src.benchmark.metric_paired import compute_lpips_paired
    from src.benchmark.prompts import load_prompts

    if config not in _PHASE2_CONFIGS:
        raise ValueError(
            f"--config must be one of {sorted(_PHASE2_CONFIGS)}, got {config!r}"
        )

    fp16_dir = fp16_images_dir.resolve()
    gt_dir = ground_truth_dir.resolve()
    qdir = quantized_dir.resolve()
    out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_fp16 = len(list_png_paths(fp16_dir))
    if n_fp16 < 2:
        raise ValueError(f"Need at least 2 PNGs in fp16 dir, found {n_fp16}")

    cap = num_images if num_images is not None else n_fp16
    prompts, seeds = load_prompts(prompt_file, max_count=max(cap, n_fp16))
    n_target = min(n_fp16, len(prompts), cap)
    if n_target < 2:
        raise ValueError(
            f"Need at least 2 aligned prompts/images; got n_fp16={n_fp16}, "
            f"prompts={len(prompts)}, cap={cap}"
        )
    prompts = prompts[:n_target]

    digits = (
        int(img_digits)
        if img_digits is not None
        else _infer_img_digits(fp16_dir)
    )
    w4a8_root = out_dir
    w4a8_images = w4a8_root / "images"
    w4a8_images.mkdir(parents=True, exist_ok=True)

    poly_schedule = _resolve_poly_schedule(poly_schedule_path)

    n_gt = len(list_image_paths(gt_dir))
    if n_gt != n_fp16:
        print(
            f"WARNING: ground-truth image count ({n_gt}) != FP16 PNG count ({n_fp16}); "
            "FID/CMMD still use all files in each directory."
        )

    # ---- W4A8 generation (if needed) ------------------------------------
    complete = _w4a8_images_complete(w4a8_images, digits, n_target)
    need_generate = (not complete) or force_w4a8_gen or (not resume)
    resume_effective = resume and not force_w4a8_gen
    if need_generate:
        if complete and (force_w4a8_gen or not resume):
            print("=== Regenerating W4A8 images (--force-w4a8-gen / --no-resume) ===")
        elif not complete:
            print("=== Generating W4A8 images (missing or incomplete) ===")
        seeds_use: Optional[List[int]] = (
            seeds[:n_target] if seeds is not None else None
        )
        generate_images(
            config=config,
            prompts=prompts,
            output_dir=w4a8_root,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            seed_base=seed,
            warmup=warmup,
            resume=resume_effective,
            poly_schedule=poly_schedule,
            group_size=group_size,
            seeds=seeds_use,
            reload_n=reload_n,
            quantized_dir=qdir,
            img_digits=digits,
        )
    elif complete:
        print("=== W4A8 images already complete -- skipping generation ===")

    n_w4 = len(list_png_paths(w4a8_images))
    if n_w4 != n_target:
        print(
            f"WARNING: expected {n_target} W4A8 PNGs after generation, found {n_w4}"
        )

    results: Dict[str, Any] = {
        "ground_truth_dir": str(gt_dir),
        "fp16_images_dir": str(fp16_dir),
        "w4a8_images_dir": str(w4a8_images),
        "quantized_dir": str(qdir),
        "n_target": n_target,
        "img_digits": digits,
        "prompt_file": str(prompt_file),
    }

    # ---- FID -------------------------------------------------------------
    print("\n=== FID: FP16 vs ground truth ===")
    results["fid_fp16_vs_gt"] = compute_fidelity_metrics(
        str(fp16_dir),
        str(gt_dir),
        kid_subset_max=kid_subset_max,
        use_cuda=fidelity_cuda,
    )

    print("\n=== FID: W4A8 vs ground truth ===")
    results["fid_w4a8_vs_gt"] = compute_fidelity_metrics(
        str(w4a8_images),
        str(gt_dir),
        kid_subset_max=kid_subset_max,
        use_cuda=fidelity_cuda,
    )

    # ---- CMMD (CLIP embeddings) ------------------------------------------
    print("\n=== CLIP embeddings for CMMD ===")
    _clip_kw = dict(
        batch_size=clip_batch_size, arch=clip_arch, pretrained=clip_pretrained
    )
    emb_fp16 = compute_clip_embeddings(str(fp16_dir), **_clip_kw)
    emb_w4 = compute_clip_embeddings(str(w4a8_images), **_clip_kw)
    emb_gt = compute_clip_embeddings(str(gt_dir), **_clip_kw)

    def _cmmd_or_none(a, b):
        if (
            a is not None
            and b is not None
            and a["embeddings"].shape[0] >= 2
            and b["embeddings"].shape[0] >= 2
        ):
            return compute_cmmd_from_embeddings(a["embeddings"], b["embeddings"])
        return None

    results["cmmd_fp16_vs_gt"] = _cmmd_or_none(emb_fp16, emb_gt)
    if results["cmmd_fp16_vs_gt"] is None:
        print("WARNING: CMMD FP16 vs GT skipped (embeddings missing or too few).")

    results["cmmd_w4a8_vs_gt"] = _cmmd_or_none(emb_w4, emb_gt)
    if results["cmmd_w4a8_vs_gt"] is None:
        print("WARNING: CMMD W4A8 vs GT skipped (embeddings missing or too few).")

    # ---- CLIP image-text scores ------------------------------------------
    print("\n=== CLIP image-text: FP16 vs prompts ===")
    results["clip_score_fp16"] = compute_clip_image_text_scores(
        fp16_dir, prompts, **_clip_kw
    )

    print("\n=== CLIP image-text: W4A8 vs prompts ===")
    results["clip_score_w4a8"] = compute_clip_image_text_scores(
        w4a8_images, prompts, **_clip_kw
    )

    # ---- LPIPS (W4A8 vs FP16, paired) ------------------------------------
    print("\n=== LPIPS: W4A8 vs FP16 (paired filenames) ===")
    results["lpips_w4a8_vs_fp16"] = compute_lpips_paired(
        str(w4a8_images),
        str(fp16_dir),
        net=lpips_net,
        resize=lpips_resize,
    )

    results["metric_options"] = {
        "config": config,
        "kid_subset_max": kid_subset_max,
        "fidelity_cuda": fidelity_cuda,
        "clip_arch": clip_arch,
        "clip_pretrained": clip_pretrained,
        "clip_batch_size": clip_batch_size,
        "lpips_net": lpips_net,
        "lpips_resize": lpips_resize,
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
    }

    return results


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GT comparison: generate W4A8 if needed; FID, CMMD, CLIP scores, LPIPS."
    )
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--fp16-images-dir", type=Path, required=True)
    parser.add_argument("--quantized-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=f"Prompts file (default: {_DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="w4a8",
        help="W4A8 pipeline config: w4a8, w4a8_static, w4a8_poly (default: w4a8)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Max images / prompts to use (default: all FP16 PNGs)",
    )
    parser.add_argument(
        "--img-digits",
        type=int,
        default=None,
        help="PNG index zero-padding (default: infer from FP16 filenames)",
    )
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate every W4A8 PNG (do not skip existing files)",
    )
    parser.add_argument("--reload-n", type=int, default=1)
    parser.add_argument(
        "--force-w4a8-gen",
        action="store_true",
        help="Always run generation (even if outputs look complete)",
    )
    parser.add_argument(
        "--poly-schedule",
        type=Path,
        default=None,
        help="JSON schedule (required for meaningful w4a8_poly runs)",
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--kid-subset-max", type=int, default=1000)
    parser.add_argument("--fidelity-cuda", action="store_true")
    parser.add_argument("--clip-arch", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--lpips-net", type=str, default="alex")
    parser.add_argument("--lpips-resize", type=int, default=256)

    args = parser.parse_args()
    prompt_path = args.prompt_file or _DEFAULT_PROMPT_FILE

    if args.config not in _PHASE2_CONFIGS:
        parser.error(f"--config must be one of {sorted(_PHASE2_CONFIGS)}")

    try:
        results = run_pipeline(
            args.ground_truth_dir,
            args.fp16_images_dir,
            args.quantized_dir,
            args.output_dir,
            prompt_file=prompt_path,
            config=args.config,
            num_images=args.num_images,
            img_digits=args.img_digits,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            warmup=args.warmup,
            resume=not args.no_resume,
            reload_n=args.reload_n,
            force_w4a8_gen=args.force_w4a8_gen,
            poly_schedule_path=args.poly_schedule,
            group_size=args.group_size,
            kid_subset_max=args.kid_subset_max,
            fidelity_cuda=args.fidelity_cuda,
            clip_arch=args.clip_arch,
            clip_pretrained=args.clip_pretrained,
            clip_batch_size=args.clip_batch_size,
            lpips_net=args.lpips_net,
            lpips_resize=args.lpips_resize,
        )
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    out_json = Path(args.output_dir).resolve() / "gt_comparison_results.json"
    with open(out_json, "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"\nWrote {out_json}")

    # ---- Summary --------------------------------------------------------
    print("\n--- Summary ---")
    f1 = results.get("fid_fp16_vs_gt") or {}
    f2 = results.get("fid_w4a8_vs_gt") or {}
    if isinstance(f1, dict) and "fid" in f1:
        print(f"  FID FP16 vs GT:    {f1['fid']:.4f}")
    if isinstance(f2, dict) and "fid" in f2:
        print(f"  FID W4A8 vs GT:    {f2['fid']:.4f}")
    c1 = results.get("cmmd_fp16_vs_gt")
    c2 = results.get("cmmd_w4a8_vs_gt")
    if c1 is not None:
        print(f"  CMMD FP16 vs GT:   {c1:.6f}")
    if c2 is not None:
        print(f"  CMMD W4A8 vs GT:   {c2:.6f}")
    s1 = results.get("clip_score_fp16") or {}
    s2 = results.get("clip_score_w4a8") or {}
    if isinstance(s1, dict) and "clip_image_text_mean" in s1:
        print(f"  CLIP score FP16:   {s1['clip_image_text_mean']:.4f} (n={s1.get('n_pairs')})")
    if isinstance(s2, dict) and "clip_image_text_mean" in s2:
        print(f"  CLIP score W4A8:   {s2['clip_image_text_mean']:.4f} (n={s2.get('n_pairs')})")
    lp = results.get("lpips_w4a8_vs_fp16") or {}
    if isinstance(lp, dict) and "lpips_mean" in lp:
        print(f"  LPIPS W4A8 vs FP16: {lp['lpips_mean']:.4f}")


if __name__ == "__main__":
    main()
