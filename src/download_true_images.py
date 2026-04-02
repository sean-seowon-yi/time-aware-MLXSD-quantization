"""
Download the ground-truth MS-COCO val2017 images that correspond to each
prompt in evaluation_set.txt.

Matches prompts by exact caption text against the cached captions_val2017.json,
then downloads each image from the COCO image server.

Usage
-----
    python -m src.download_true_images

Output: benchmark_results/true/<index>.jpg  (0-indexed, matching evaluation_set.txt order)
"""

import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image

CAPTIONS_JSON = Path(".coco_cache/captions_val2017.json")
EVALUATION_SET = Path("src/calibration_sample_generation/evaluation_set.txt")
OUTPUT_DIR = Path("benchmark_results/true")
IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{image_id:012d}.jpg"
IMAGE_SIZE = (512, 512)


def load_caption_to_image_id(json_path: Path) -> dict[str, int]:
    with open(json_path) as f:
        data = json.load(f)
    mapping = {}
    for ann in data["annotations"]:
        mapping[ann["caption"].strip()] = ann["image_id"]
    return mapping


def load_prompts(txt_path: Path) -> list[tuple[int, str]]:
    """Return list of (index, clean_prompt) from evaluation_set.txt."""
    prompts = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                idx_str, caption = line.split("\t", 1)
                prompts.append((int(idx_str), caption.strip()))
            else:
                prompts.append((len(prompts), line))
    return prompts


def download_image(image_id: int, dest: Path) -> None:
    url = IMAGE_URL_TEMPLATE.format(image_id=image_id)
    with urllib.request.urlopen(url) as resp:
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    img.save(dest)


def main() -> None:
    if not CAPTIONS_JSON.exists():
        print(f"Error: {CAPTIONS_JSON} not found. Run download_coco_prompts.py first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading caption -> image_id map...")
    caption_to_id = load_caption_to_image_id(CAPTIONS_JSON)

    print(f"Loading prompts from {EVALUATION_SET}...")
    prompts = load_prompts(EVALUATION_SET)
    print(f"  {len(prompts)} prompts loaded")

    missing = [p for _, p in prompts if p not in caption_to_id]
    if missing:
        print(f"Error: {len(missing)} prompts not found in captions JSON:")
        for p in missing:
            print(f"  {p!r}")
        sys.exit(1)

    print(f"Downloading {len(prompts)} images to {OUTPUT_DIR}/...")
    for i, (idx, prompt) in enumerate(prompts):
        dest = OUTPUT_DIR / f"{i:04d}.jpg"
        if dest.exists():
            print(f"  [{i+1:3d}/{len(prompts)}] {dest.name} already exists, skipping")
            continue
        image_id = caption_to_id[prompt]
        print(f"  [{i+1:3d}/{len(prompts)}] image_id={image_id} -> {dest.name}")
        try:
            download_image(image_id, dest)
        except Exception as e:
            print(f"    WARNING: failed to download image_id={image_id}: {e}")

    downloaded = sum(1 for i in range(len(prompts)) if (OUTPUT_DIR / f"{i:04d}.jpg").exists())
    print(f"\nDone. {downloaded}/{len(prompts)} images in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
