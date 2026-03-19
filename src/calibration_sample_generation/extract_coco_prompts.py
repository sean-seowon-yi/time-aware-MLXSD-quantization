"""
Extract 64 MS-COCO val2017 captions with category-stratified sampling.

Strategy:
  - Load cached captions_val2017.json
  - Apply word-count filter (5–30 words) + deduplication
  - Assign each caption to one of 12 COCO supercategories via keyword matching
  - Sample ~5–6 captions per supercategory, targeting 64 total
  - Save to a plain-text file (one prompt per line) and print to stdout

Output: eda_output/coco_64_prompts.txt
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
COCO_JSON = REPO_ROOT / ".coco_cache" / "captions_val2017.json"
OUTPUT_FILE = REPO_ROOT / "eda_output" / "coco_64_prompts.txt"
SEED = 42
WORD_MIN, WORD_MAX = 5, 30
TARGET_N = 64

# ---------------------------------------------------------------------------
# COCO supercategory keyword map
# Each entry: (supercategory_name, [keywords to match in caption text])
# Priority order: first match wins.
# ---------------------------------------------------------------------------
SUPERCATEGORY_KEYWORDS = [
    ("animal",      ["dog", "cat", "bird", "horse", "cow", "elephant", "bear",
                     "zebra", "giraffe", "sheep", "animal", "duck", "deer",
                     "rabbit", "puppy", "kitten", "monkey", "lion", "tiger"]),
    ("food",        ["food", "eating", "banana", "apple", "sandwich", "orange",
                     "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                     "fruit", "vegetable", "meal", "dinner", "lunch", "bread",
                     "cheese", "salad", "burger", "taco", "sushi", "rice",
                     "pasta", "soup", "dessert", "cookie"]),
    ("vehicle",     ["car", "truck", "bus", "motorcycle", "bicycle", "bike",
                     "boat", "train", "airplane", "plane", "vehicle", "taxi",
                     "van", "suv", "jeep", "helicopter", "ship", "ferry",
                     "scooter", "tractor"]),
    ("sports",      ["sport", "playing", "game", "frisbee", "ski", "skiing",
                     "snowboard", "ball", "kite", "bat", "glove", "skateboard",
                     "surfboard", "tennis", "baseball", "basketball", "football",
                     "soccer", "golf", "hockey", "swimming", "running", "race",
                     "jump", "athlete", "player", "court", "field", "pitch"]),
    ("outdoor",     ["street", "road", "park", "city", "building", "sidewalk",
                     "traffic", "sign", "bench", "bridge", "sky", "tree",
                     "garden", "beach", "ocean", "mountain", "river", "lake",
                     "forest", "field", "outdoor", "outside"]),
    ("person",      ["person", "man", "woman", "people", "boy", "girl",
                     "child", "baby", "kid", "lady", "gentleman", "adult",
                     "crowd", "group", "family", "couple", "team"]),
    ("furniture",   ["chair", "couch", "sofa", "bed", "table", "desk", "shelf",
                     "cabinet", "furniture", "seat", "cushion", "pillow",
                     "mattress", "drawer"]),
    ("kitchen",     ["kitchen", "bottle", "cup", "bowl", "fork", "knife",
                     "spoon", "plate", "glass", "mug", "pan", "pot",
                     "cutting board", "utensil", "cooking", "chef"]),
    ("electronic",  ["tv", "television", "laptop", "computer", "mouse",
                     "keyboard", "phone", "cell phone", "monitor", "screen",
                     "tablet", "camera", "remote", "electronic", "device"]),
    ("appliance",   ["microwave", "oven", "toaster", "refrigerator", "fridge",
                     "washer", "dryer", "dishwasher", "appliance", "stove",
                     "sink", "vacuum"]),
    ("accessory",   ["backpack", "umbrella", "handbag", "bag", "tie",
                     "suitcase", "luggage", "purse", "wallet", "hat",
                     "helmet", "glasses", "sunglasses", "scarf", "gloves",
                     "jacket", "coat", "shirt", "dress", "shoes", "boots"]),
    ("indoor",      ["book", "clock", "vase", "scissors", "teddy bear",
                     "toy", "candle", "painting", "picture", "frame",
                     "window", "door", "floor", "wall", "ceiling", "room",
                     "living room", "bedroom", "bathroom", "hallway", "stairs"]),
]


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def assign_supercategory(caption: str) -> str:
    lower = caption.lower()
    for name, keywords in SUPERCATEGORY_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return name
    return "other"


def load_and_filter(json_path: Path) -> list[str]:
    with open(json_path) as f:
        data = json.load(f)
    captions = [ann["caption"].strip() for ann in data["annotations"]]

    # Word-count filter
    captions = [c for c in captions if WORD_MIN <= word_count(c) <= WORD_MAX]

    # Deduplication by lowercase MD5
    seen: set[str] = set()
    unique: list[str] = []
    for c in captions:
        key = hashlib.md5(c.lower().strip().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def stratified_sample(captions: list[str], target: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)

    # Group by supercategory
    buckets: dict[str, list[str]] = defaultdict(list)
    for c in captions:
        buckets[assign_supercategory(c)].append(c)

    cats = sorted(buckets.keys())
    print(f"\nSupercategory distribution (before sampling):")
    for cat in cats:
        print(f"  {cat:<12}: {len(buckets[cat])} captions")

    # Remove "other" if named categories cover enough
    named = [c for c in cats if c != "other"]
    n_cats = len(named)
    base, extra = divmod(target, n_cats)

    selected: list[str] = []
    for i, cat in enumerate(named):
        n = base + (1 if i < extra else 0)
        pool = buckets[cat]
        n = min(n, len(pool))
        idxs = rng.choice(len(pool), size=n, replace=False)
        selected.extend(pool[j] for j in idxs)

    # If short (some buckets were too small), top-up from "other" or remainder
    if len(selected) < target:
        remainder = [c for c in captions if c not in set(selected)]
        rng.shuffle(remainder)
        selected.extend(remainder[: target - len(selected)])

    # Shuffle final list for uniform ordering
    selected_arr = np.array(selected)
    rng.shuffle(selected_arr)
    return list(selected_arr[:target])


def main():
    if not COCO_JSON.exists():
        raise FileNotFoundError(
            f"COCO captions JSON not found at {COCO_JSON}.\n"
            "Run sample_cali_data.py first to download and cache it."
        )

    print(f"Loading captions from {COCO_JSON}")
    captions = load_and_filter(COCO_JSON)
    print(f"After filter+dedup: {len(captions)} captions")

    prompts = stratified_sample(captions, TARGET_N, SEED)
    assert len(prompts) == TARGET_N, f"Expected {TARGET_N}, got {len(prompts)}"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for p in prompts:
            f.write(p + "\n")

    print(f"\nSaved {TARGET_N} prompts to {OUTPUT_FILE}")
    print("\n--- 64 sampled prompts ---")
    for i, p in enumerate(prompts, 1):
        cat = assign_supercategory(p)
        print(f"[{i:02d}] [{cat:<12}] {p}")


if __name__ == "__main__":
    main()
