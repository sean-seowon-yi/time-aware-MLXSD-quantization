#!/usr/bin/env python3
"""Step 3 — Compute LPIPS and CLIPScore for each W4A8 config vs FP16.

For a given W4A8 results directory (or all discovered configs), loads image
pairs (FP16 reference + W4A8 candidate), computes per-image LPIPS distance
and CLIPScore, and saves both per-image and aggregate results.

Models
------
- **LPIPS**: AlexNet backbone (lightweight, well-correlated with human
  perception; Zhang et al., "The Unreasonable Effectiveness of Deep Features
  as a Perceptual Metric", CVPR 2018).
- **CLIPScore**: ``openai/clip-vit-base-patch32`` — cosine similarity between
  CLIP image and text embeddings, scaled to [0, 100].

Usage
-----
# Compute metrics for one config
python -m src.sweep.run_metrics --w4a8-tag w4a8_max_a0.50_gs64

# Compute metrics for all discovered configs
python -m src.sweep.run_metrics --all

# Limit to first 32 images (matches a staged inference run)
python -m src.sweep.run_metrics --all --num-images 32

# Use a different CLIP model
python -m src.sweep.run_metrics --all --clip-model openai/clip-vit-large-patch14

Output
------
metrics/<tag>/per_image.json    — per-image scores
metrics/<tag>/aggregate.json    — mean / median / std summaries
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .sweep_config import (
    METRICS_ROOT,
    RESULTS_ROOT,
    config_tag,
    fp16_dir,
    load_prompt_pairs,
    resolve_configs,
)


# ── Image loading helpers ─────────────────────────────────────────────────

def _pil_to_lpips_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to LPIPS input: [1, 3, H, W] float32 in [-1, 1]."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


# ── Metric wrappers ──────────────────────────────────────────────────────

class LPIPSScorer:
    """Wraps the ``lpips`` package with the AlexNet backbone."""

    def __init__(self, net: str = "alex"):
        import lpips as _lpips
        self.net_name = net
        self.fn = _lpips.LPIPS(net=net, verbose=False)
        self.fn.eval()

    @torch.no_grad()
    def score(self, img_a: Image.Image, img_b: Image.Image) -> float:
        t_a = _pil_to_lpips_tensor(img_a)
        t_b = _pil_to_lpips_tensor(img_b)
        return self.fn(t_a, t_b).item()


class CLIPScorer:
    """Compute CLIPScore = 100 * cos(image_embed, text_embed)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        inputs = self.processor(
            text=[text], images=[image.convert("RGB")],
            return_tensors="pt", padding=True, truncation=True,
        )
        outputs = self.model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return (img_emb @ txt_emb.T).item() * 100.0


# ── Core logic ────────────────────────────────────────────────────────────

def compute_metrics_for_tag(
    tag: str,
    *,
    fp16_root: Path,
    results_root: Path,
    metrics_root: Path,
    prompt_pairs: list[tuple[int, str]],
    num_images: int | None,
    eval_indices: list[int] | None,
    lpips_scorer: LPIPSScorer,
    clip_scorer: CLIPScorer,
) -> dict:
    """Compute per-image and aggregate metrics for a single config tag.

    When *eval_indices* is provided, only those specific prompt indices are
    evaluated (supports randomized staged selection).  Otherwise falls back
    to the first *num_images* pairs.

    Previously computed per-image scores are reused when the scorer
    configuration (LPIPS backbone, CLIP model) has not changed, so
    staged evaluation only pays for newly added images.

    Returns the aggregate dict (also saved to disk).
    """
    w4a8_root = results_root / tag
    out_dir = metrics_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if eval_indices is not None:
        target_indices = set(eval_indices)
    elif num_images is not None:
        target_indices = set(range(num_images))
    else:
        target_indices = set(range(len(prompt_pairs)))

    # ── Incremental: reuse previously computed per-image scores ───────
    existing_by_idx: dict[int, dict] = {}
    per_image_path = out_dir / "per_image.json"
    scorer_meta_path = out_dir / "scorer_config.json"
    current_scorer_cfg = {
        "lpips_net": lpips_scorer.net_name,
        "clip_model": clip_scorer.model_name,
    }

    if per_image_path.exists():
        scorer_match = False
        if scorer_meta_path.exists():
            try:
                scorer_match = json.loads(scorer_meta_path.read_text()) == current_scorer_cfg
            except (json.JSONDecodeError, KeyError):
                pass
        if scorer_match:
            try:
                for entry in json.loads(per_image_path.read_text()):
                    existing_by_idx[entry["index"]] = entry
            except (json.JSONDecodeError, KeyError):
                existing_by_idx.clear()
        else:
            print("  Scorer config changed — recomputing all metrics")

    # Find matching image indices within the target set
    available_indices: list[int] = []
    for idx in sorted(target_indices):
        if idx >= len(prompt_pairs):
            continue
        fp16_path = fp16_root / f"{idx:03d}.png"
        w4a8_path = w4a8_root / f"{idx:03d}.png"
        if fp16_path.exists() and w4a8_path.exists():
            available_indices.append(idx)

    if not available_indices:
        print(f"  WARNING: no matching image pairs found for {tag}")
        return {}

    cached_indices = [i for i in available_indices if i in existing_by_idx]
    new_indices = [i for i in available_indices if i not in existing_by_idx]

    print(f"  Processing {len(available_indices)} image pairs "
          f"({len(cached_indices)} cached, {len(new_indices)} new) ...")

    per_image: list[dict] = [existing_by_idx[i] for i in cached_indices]

    if new_indices:
        t0 = time.time()
        for count, idx in enumerate(new_indices):
            seed, prompt = prompt_pairs[idx]
            fp16_img = Image.open(fp16_root / f"{idx:03d}.png")
            w4a8_img = Image.open(w4a8_root / f"{idx:03d}.png")

            lpips_val = lpips_scorer.score(fp16_img, w4a8_img)
            clip_fp16 = clip_scorer.score(fp16_img, prompt)
            clip_w4a8 = clip_scorer.score(w4a8_img, prompt)

            per_image.append({
                "index": idx,
                "seed": seed,
                "prompt": prompt,
                "lpips": round(lpips_val, 5),
                "clip_fp16": round(clip_fp16, 3),
                "clip_w4a8": round(clip_w4a8, 3),
                "clip_delta": round(clip_w4a8 - clip_fp16, 3),
            })

            if (count + 1) % 20 == 0 or count + 1 == len(new_indices):
                elapsed = time.time() - t0
                print(f"    [{count+1}/{len(new_indices)}] {elapsed:.0f}s")

    # Preserve cached entries outside the current target set so that
    # later (wider) stages do not lose their cache on re-runs.
    current_idx_set = {r["index"] for r in per_image}
    for idx, entry in existing_by_idx.items():
        if idx not in target_indices and idx not in current_idx_set:
            per_image.append(entry)

    per_image.sort(key=lambda r: r["index"])

    # Save complete per-image data (current set + preserved from other stages)
    (out_dir / "per_image.json").write_text(
        json.dumps(per_image, indent=2, ensure_ascii=False),
    )
    scorer_meta_path.write_text(json.dumps(current_scorer_cfg, indent=2))

    # Aggregate only over current target set
    in_range = [r for r in per_image if r["index"] in target_indices]
    lpips_vals = np.array([r["lpips"] for r in in_range])
    clip_fp16_vals = np.array([r["clip_fp16"] for r in in_range])
    clip_w4a8_vals = np.array([r["clip_w4a8"] for r in in_range])
    clip_deltas = np.array([r["clip_delta"] for r in in_range])

    aggregate = {
        "tag": tag,
        "num_images": len(in_range),
        "lpips": {
            "mean": round(float(lpips_vals.mean()), 5),
            "median": round(float(np.median(lpips_vals)), 5),
            "std": round(float(lpips_vals.std()), 5),
            "max": round(float(lpips_vals.max()), 5),
            "p90": round(float(np.percentile(lpips_vals, 90)), 5),
        },
        "clip_fp16": {
            "mean": round(float(clip_fp16_vals.mean()), 3),
            "std": round(float(clip_fp16_vals.std()), 3),
        },
        "clip_w4a8": {
            "mean": round(float(clip_w4a8_vals.mean()), 3),
            "std": round(float(clip_w4a8_vals.std()), 3),
        },
        "clip_delta": {
            "mean": round(float(clip_deltas.mean()), 3),
            "median": round(float(np.median(clip_deltas)), 3),
            "std": round(float(clip_deltas.std()), 3),
        },
    }
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
    )

    print(f"  LPIPS  mean={aggregate['lpips']['mean']:.4f}  "
          f"p90={aggregate['lpips']['p90']:.4f}")
    print(f"  CLIP   fp16={aggregate['clip_fp16']['mean']:.2f}  "
          f"w4a8={aggregate['clip_w4a8']['mean']:.2f}  "
          f"delta={aggregate['clip_delta']['mean']:+.2f}")
    print(f"  Saved to {out_dir}")

    return aggregate


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: compute LPIPS & CLIPScore for W4A8 configs vs FP16",
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--w4a8-tag", type=str,
        help="Single config tag, e.g. w4a8_max_a0.50_gs64",
    )
    target.add_argument(
        "--all", action="store_true",
        help="Run metrics for all configs that have result images",
    )
    target.add_argument(
        "--configs", type=int, nargs="*",
        help="Indices into SWEEP_MATRIX",
    )

    parser.add_argument(
        "--num-images", type=int, default=None,
        help="Limit to first N images (default: all available)",
    )
    parser.add_argument(
        "--eval-indices-file", type=str, default=None,
        help="JSON file listing prompt indices to evaluate (overrides --num-images)",
    )
    parser.add_argument(
        "--fp16-dir", type=str, default=None,
        help=f"FP16 results directory (default: {RESULTS_ROOT / 'fp16'})",
    )
    parser.add_argument(
        "--results-root", type=str, default=str(RESULTS_ROOT),
        help=f"Root for result images (default: {RESULTS_ROOT})",
    )
    parser.add_argument(
        "--metrics-root", type=str, default=str(METRICS_ROOT),
        help=f"Root for metric outputs (default: {METRICS_ROOT})",
    )
    parser.add_argument(
        "--prompts-file", type=str, default=None,
        help="Override evaluation prompt set path",
    )
    parser.add_argument(
        "--clip-model", type=str, default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model for CLIPScore (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone network (default: alex)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    metrics_root = Path(args.metrics_root)
    fp16_root = Path(args.fp16_dir) if args.fp16_dir else fp16_dir()

    if not fp16_root.exists():
        parser.error(f"FP16 directory not found: {fp16_root}")

    # Determine which tags to evaluate
    if args.w4a8_tag:
        tags = [args.w4a8_tag]
    elif args.configs is not None:
        tags = [config_tag(c) for c in resolve_configs(args.configs)]
    else:
        # --all: discover tags that have result directories with images
        if not results_root.exists():
            print(f"Results directory not found: {results_root}")
            return
        tags = sorted([
            d.name for d in results_root.iterdir()
            if d.is_dir() and d.name != "fp16" and any(d.glob("*.png"))
        ])

    if not tags:
        print("No W4A8 result directories found.")
        return

    prompt_pairs = load_prompt_pairs(
        Path(args.prompts_file) if args.prompts_file else None,
    )

    eval_indices: list[int] | None = None
    if args.eval_indices_file is not None:
        eval_indices = json.loads(Path(args.eval_indices_file).read_text())

    print(f"\n{'='*60}")
    print(f"  METRICS — {len(tags)} configs")
    print(f"  FP16 reference: {fp16_root}  ({len(list(fp16_root.glob('*.png')))} images)")
    if eval_indices is not None:
        print(f"  Eval indices:   {len(eval_indices)} (randomized)")
    print(f"  CLIP model: {args.clip_model}")
    print(f"  LPIPS net:  {args.lpips_net}")
    print(f"{'='*60}\n")

    # Load scoring models once
    print("Loading LPIPS model ...")
    lpips_scorer = LPIPSScorer(net=args.lpips_net)
    print("Loading CLIP model ...")
    clip_scorer = CLIPScorer(model_name=args.clip_model)
    print()

    all_aggregates: list[dict] = []

    for i, tag in enumerate(tags):
        print(f"[{i+1}/{len(tags)}] {tag}")
        w4a8_dir = results_root / tag
        if not w4a8_dir.exists() or not any(w4a8_dir.glob("*.png")):
            print(f"  SKIP — no images at {w4a8_dir}")
            continue

        agg = compute_metrics_for_tag(
            tag,
            fp16_root=fp16_root,
            results_root=results_root,
            metrics_root=metrics_root,
            prompt_pairs=prompt_pairs,
            num_images=args.num_images,
            eval_indices=eval_indices,
            lpips_scorer=lpips_scorer,
            clip_scorer=clip_scorer,
        )
        if agg:
            all_aggregates.append(agg)
        print()

    # Save combined summary
    if all_aggregates:
        summary_path = metrics_root / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(all_aggregates, indent=2, ensure_ascii=False),
        )
        print(f"Combined summary saved to {summary_path}")


if __name__ == "__main__":
    main()
