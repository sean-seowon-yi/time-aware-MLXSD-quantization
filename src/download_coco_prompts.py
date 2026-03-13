"""
Download MS-COCO 2014 validation captions and export as prompts CSV.

Usage
-----
    python -m src.download_coco_prompts --output coco_prompts.csv --count 10000

Downloads captions_val2014.json (~1 MB) from the official COCO API, picks one
caption per image (first annotation), shuffles with a fixed seed, and writes
a CSV with a single 'prompt' column.

The resulting CSV is used with benchmark_model.py --prompt-csv.
"""

import argparse
import csv
import json
import random
import urllib.request
from pathlib import Path

COCO_CAPTIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)
CAPTIONS_FILENAME = "annotations/captions_val2014.json"


def download_coco_captions(cache_dir: Path) -> Path:
    """Download and unzip COCO 2014 val captions. Returns path to JSON file."""
    import zipfile

    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / "captions_val2014.json"
    if json_path.exists():
        print(f"  Cached: {json_path}")
        return json_path

    zip_path = cache_dir / "annotations_trainval2014.zip"
    if not zip_path.exists():
        print(f"  Downloading COCO annotations (~250 MB)...")
        urllib.request.urlretrieve(COCO_CAPTIONS_URL, zip_path)
        print(f"  Downloaded: {zip_path}")

    print(f"  Extracting {CAPTIONS_FILENAME}...")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(CAPTIONS_FILENAME) as src, open(json_path, "wb") as dst:
            dst.write(src.read())
    print(f"  Extracted: {json_path}")
    return json_path


def load_one_caption_per_image(json_path: Path) -> list[str]:
    """
    Parse captions_val2014.json and return one caption per image.
    Uses the first annotation encountered for each image_id.
    """
    with open(json_path) as f:
        data = json.load(f)

    seen = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in seen:
            seen[img_id] = ann["caption"].strip()

    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Export MS-COCO val captions as prompts CSV")
    parser.add_argument("--output", type=Path, default=Path("coco_prompts.csv"),
                        help="Output CSV path (default: coco_prompts.csv)")
    parser.add_argument("--count", type=int, default=10000,
                        help="Number of prompts to export (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--cache-dir", type=Path, default=Path(".coco_cache"),
                        help="Cache dir for downloaded files (default: .coco_cache)")
    args = parser.parse_args()

    print("=== MS-COCO 2014 Val Captions ===")
    json_path = download_coco_captions(args.cache_dir)

    captions = load_one_caption_per_image(json_path)
    print(f"  Loaded {len(captions)} unique image captions")

    rng = random.Random(args.seed)
    rng.shuffle(captions)
    selected = captions[: args.count]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt"])
        writer.writeheader()
        for caption in selected:
            writer.writerow({"prompt": caption})

    print(f"  Wrote {len(selected)} prompts → {args.output}")
    print(f"\nUsage:")
    print(f"  python -m src.benchmark_model --prompt-csv {args.output} --num-images {args.count} ...")


if __name__ == "__main__":
    main()
