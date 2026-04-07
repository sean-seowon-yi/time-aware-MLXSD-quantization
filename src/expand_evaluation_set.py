"""
Expand evaluation_set.txt from 256 to 1024 prompts using MS-COCO val2017 captions.

Samples 768 new prompts that do not overlap with evaluation_set.txt or
sample_prompts.txt, distributed across 8 topic buckets for diversity.

Usage
-----
    python -m src.expand_evaluation_set

After running, execute:
    python -m src.download_true_images
to download the corresponding reference images (existing images are skipped).
"""

import json
import random
from pathlib import Path

CAPTIONS_JSON = Path(".coco_cache/captions_val2017.json")
EVALUATION_SET = Path("src/calibration_sample_generation/evaluation_set.txt")
SAMPLE_PROMPTS = Path("src/calibration_sample_generation/sample_prompts.txt")

TARGET_NEW = 768
START_SEED = 298  # continues from current last seed (297)
SHUFFLE_SEED = 2210

BUCKETS = {
    "animals":  ["cat", "dog", "elephant", "giraffe", "horse", "cow", "sheep",
                 "zebra", "bear", "bird", "kitten", "puppy", "wildlife"],
    "food":     ["pizza", "sandwich", "cake", "donut", "banana", "apple", "food",
                 "eat", "meal", "bread", "fruit", "vegetable", "salad", "burger",
                 "sushi", "dessert", "coffee", "drink"],
    "sports":   ["tennis", "baseball", "skateboard", "ski", "snowboard", "frisbee",
                 "surfboard", "soccer", "football", "basketball", "volleyball",
                 "golf", "cycling", "swim", "skate", "sport"],
    "vehicles": ["car", "bus", "train", "airplane", "motorcycle", "boat", "truck",
                 "bike", "bicycle", "helicopter", "vehicle", "taxi", "jet",
                 "tractor", "van"],
    "indoor":   ["kitchen", "bedroom", "bathroom", "living room", "office",
                 "desk", "couch", "sofa", "room", "table", "chair", "shelf",
                 "window", "floor", "ceiling", "cabinet"],
    "outdoor":  ["beach", "park", "street", "mountain", "field", "forest",
                 "city", "road", "river", "lake", "ocean", "sky", "garden",
                 "sidewalk", "building", "bridge"],
    "people":   ["man", "woman", "person", "child", "crowd", "girl", "boy",
                 "people", "group", "family", "baby", "adult", "player",
                 "worker", "student"],
}


def load_exclusion_set(*paths: Path) -> set[str]:
    excluded = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                caption = line.split("\t", 1)[-1].strip().lower()
                excluded.add(caption)
    return excluded


def load_candidates(json_path: Path) -> list[str]:
    """One caption per image_id, then deduplicated by caption text."""
    with open(json_path) as f:
        data = json.load(f)
    seen_id: dict[int, str] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in seen_id:
            seen_id[img_id] = ann["caption"].strip()
    # Deduplicate by caption text (some images share identical captions)
    seen_text: dict[str, str] = {}
    for cap in seen_id.values():
        key = cap.lower()
        if key not in seen_text:
            seen_text[key] = cap
    return list(seen_text.values())


def assign_bucket(caption: str) -> str:
    lower = caption.lower()
    for bucket, keywords in BUCKETS.items():
        if any(kw in lower for kw in keywords):
            return bucket
    return "other"


def sample_diverse(candidates: list[str], n: int, rng: random.Random) -> list[str]:
    """Sample n captions distributed proportionally across topic buckets."""
    bucketed: dict[str, list[str]] = {b: [] for b in list(BUCKETS.keys()) + ["other"]}
    for cap in candidates:
        bucketed[assign_bucket(cap)].append(cap)

    # Shuffle each bucket
    for caps in bucketed.values():
        rng.shuffle(caps)

    bucket_names = [b for b, caps in bucketed.items() if caps]
    total_available = sum(len(bucketed[b]) for b in bucket_names)

    # Proportional allocation
    allocation: dict[str, int] = {}
    remaining = n
    for i, b in enumerate(bucket_names):
        if i == len(bucket_names) - 1:
            allocation[b] = remaining
        else:
            share = round(n * len(bucketed[b]) / total_available)
            share = min(share, len(bucketed[b]), remaining)
            allocation[b] = share
            remaining -= share

    # Fill shortfalls round-robin
    selected: list[str] = []
    for b, count in allocation.items():
        selected.extend(bucketed[b][:count])

    # If still short (due to rounding), fill from largest buckets
    if len(selected) < n:
        pool = [c for b in bucket_names for c in bucketed[b][allocation.get(b, 0):]]
        rng.shuffle(pool)
        selected.extend(pool[: n - len(selected)])

    return selected[:n]


def main() -> None:
    print("Loading exclusion set...")
    excluded = load_exclusion_set(EVALUATION_SET, SAMPLE_PROMPTS)
    print(f"  {len(excluded)} prompts to exclude")

    print(f"Loading candidates from {CAPTIONS_JSON}...")
    all_captions = load_candidates(CAPTIONS_JSON)
    print(f"  {len(all_captions)} unique image captions loaded")

    candidates = [c for c in all_captions if c.lower() not in excluded]
    print(f"  {len(candidates)} candidates after exclusion filter")

    rng = random.Random(SHUFFLE_SEED)
    new_prompts = sample_diverse(candidates, TARGET_NEW, rng)
    print(f"  {len(new_prompts)} prompts selected")

    # Print bucket distribution
    bucket_counts: dict[str, int] = {}
    for cap in new_prompts:
        b = assign_bucket(cap)
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    print("\nBucket distribution:")
    for b, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        print(f"  {b:<12} {count:>4}")

    # Append to evaluation_set.txt (ensure file ends with newline first)
    with open(EVALUATION_SET, "rb+") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            f.write(b"\n")
    with open(EVALUATION_SET, "a") as f:
        for i, caption in enumerate(new_prompts):
            seed = START_SEED + i
            f.write(f"{seed}\t{caption}\n")

    # Verify no duplicates
    with open(EVALUATION_SET) as f:
        lines = [l.strip() for l in f if l.strip()]
    captions_in_file = [l.split("\t", 1)[-1].strip().lower() for l in lines]
    if len(captions_in_file) != len(set(captions_in_file)):
        print("\nWARNING: duplicate captions detected in evaluation_set.txt")
    else:
        print(f"\nevaluation_set.txt now has {len(lines)} prompts (no duplicates)")

    print(f"Seeds assigned: {START_SEED} – {START_SEED + TARGET_NEW - 1}")
    print("\nNext step: python -m src.download_true_images")


if __name__ == "__main__":
    main()
