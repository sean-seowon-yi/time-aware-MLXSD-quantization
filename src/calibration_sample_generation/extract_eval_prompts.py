"""
Extract 256 hard MS-COCO val2017 captions for FID evaluation.

"Hard" criteria — captions that are challenging for generative models:
  - Longer captions (8–30 words) to test compositional understanding
  - Prefer captions with multiple objects, spatial relationships, counting,
    unusual scenes, or fine-grained attributes (colors, sizes, actions)
  - Category-stratified to ensure topic diversity
  - No overlap with sample_prompts.txt (used for calibration)

Output: evaluation_set.txt (same directory as this script)
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
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
COCO_JSON = REPO_ROOT / ".coco_cache" / "captions_val2017.json"
EXISTING_PROMPTS = SCRIPT_DIR / "sample_prompts.txt"
OUTPUT_FILE = SCRIPT_DIR / "evaluation_set.txt"
SEED = 123
WORD_MIN, WORD_MAX = 8, 30
TARGET_N = 256

# ---------------------------------------------------------------------------
# Difficulty heuristics
# ---------------------------------------------------------------------------
# Spatial / relational keywords that make prompts harder for generative models
SPATIAL_WORDS = {
    "next to", "in front of", "behind", "on top of", "underneath", "beneath",
    "between", "above", "below", "beside", "across from", "surrounding",
    "leaning against", "hanging from", "attached to", "inside", "outside",
    "through", "along", "facing", "reflected in", "casting", "shadow",
}

# Counting / quantity words
COUNT_WORDS = {
    "two", "three", "four", "five", "six", "seven", "eight", "several",
    "many", "multiple", "numerous", "group of", "pair of", "bunch of",
    "row of", "stack of", "collection of",
}

# Attribute words (colors, sizes, textures)
ATTRIBUTE_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "brown", "gray", "grey", "golden", "silver", "large", "small",
    "tiny", "huge", "tall", "short", "long", "round", "square", "striped",
    "spotted", "wooden", "metal", "glass", "plastic", "stone", "brick",
    "old", "new", "modern", "vintage", "rusty", "shiny", "wet", "dry",
    "colorful", "bright", "dark", "blurry", "ornate",
}

# Action words that create complex scenes
ACTION_WORDS = {
    "riding", "jumping", "flying", "catching", "throwing", "holding",
    "carrying", "pulling", "pushing", "climbing", "swimming", "running",
    "walking", "standing", "sitting", "lying", "kneeling", "bending",
    "reaching", "pointing", "waving", "dancing", "cooking", "eating",
    "drinking", "reading", "writing", "painting", "cutting", "washing",
}

# COCO supercategory keyword map (same as extract_coco_prompts.py)
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


def difficulty_score(caption: str) -> float:
    """Score how 'hard' a caption is for a generative model."""
    lower = caption.lower()
    score = 0.0

    # Longer captions are harder
    wc = word_count(caption)
    score += min(wc / 10.0, 2.0)  # up to 2 pts for length

    # Spatial relationships
    for kw in SPATIAL_WORDS:
        if kw in lower:
            score += 1.5

    # Counting / multiple objects
    for kw in COUNT_WORDS:
        if kw in lower:
            score += 1.2

    # Attributes (colors, sizes, textures)
    attr_count = sum(1 for kw in ATTRIBUTE_WORDS if f" {kw} " in f" {lower} ")
    score += min(attr_count * 0.5, 2.0)  # up to 2 pts

    # Actions
    action_count = sum(1 for kw in ACTION_WORDS if kw in lower)
    score += min(action_count * 0.4, 1.5)

    # Multiple distinct nouns (proxy: comma-separated clauses or "and")
    score += lower.count(" and ") * 0.5
    score += lower.count(",") * 0.3

    return score


def assign_supercategory(caption: str) -> str:
    lower = caption.lower()
    for name, keywords in SUPERCATEGORY_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return name
    return "other"


def load_existing_prompts(path: Path) -> set[str]:
    """Load existing prompts to avoid overlap."""
    if not path.exists():
        return set()
    with open(path) as f:
        return {line.strip().lower() for line in f if line.strip()}


def load_and_filter(json_path: Path, exclude: set[str]) -> list[str]:
    with open(json_path) as f:
        data = json.load(f)
    captions = [ann["caption"].strip() for ann in data["annotations"]]

    # Word-count filter (longer minimum for harder prompts)
    captions = [c for c in captions if WORD_MIN <= word_count(c) <= WORD_MAX]

    # Deduplication by lowercase MD5
    seen: set[str] = set()
    unique: list[str] = []
    for c in captions:
        key = hashlib.md5(c.lower().strip().encode()).hexdigest()
        if key not in seen and c.lower().strip() not in exclude:
            seen.add(key)
            unique.append(c)

    return unique


def stratified_hard_sample(
    captions: list[str], target: int, seed: int
) -> list[str]:
    rng = np.random.default_rng(seed)

    # Score all captions
    scored = [(c, difficulty_score(c)) for c in captions]

    # Group by supercategory
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for c, s in scored:
        buckets[assign_supercategory(c)].append((c, s))

    # Sort each bucket by difficulty (hardest first)
    for cat in buckets:
        buckets[cat].sort(key=lambda x: -x[1])

    cats = sorted(buckets.keys())
    print(f"\nSupercategory distribution (before sampling):")
    for cat in cats:
        print(f"  {cat:<12}: {len(buckets[cat]):>5} captions")

    # Allocate per category
    named = [c for c in cats if c != "other"]
    n_cats = len(named)
    base, extra = divmod(target, n_cats)

    selected: list[str] = []
    for i, cat in enumerate(named):
        n = base + (1 if i < extra else 0)
        pool = buckets[cat]
        # Take from top-50% hardest, then randomly sample within that
        hard_pool = pool[: max(len(pool) // 2, n * 2)]
        n = min(n, len(hard_pool))
        idxs = rng.choice(len(hard_pool), size=n, replace=False)
        selected.extend(hard_pool[j][0] for j in idxs)

    # Top-up if needed
    if len(selected) < target:
        selected_set = set(selected)
        remainder = [(c, s) for c, s in scored if c not in selected_set]
        remainder.sort(key=lambda x: -x[1])
        selected.extend(c for c, _ in remainder[: target - len(selected)])

    # Shuffle final list
    selected_arr = np.array(selected)
    rng.shuffle(selected_arr)
    return list(selected_arr[:target])


def main():
    if not COCO_JSON.exists():
        raise FileNotFoundError(
            f"COCO captions JSON not found at {COCO_JSON}.\n"
            "Run sample_cali_data.py first to download and cache it."
        )

    # Load existing prompts to exclude
    exclude = load_existing_prompts(EXISTING_PROMPTS)
    print(f"Excluding {len(exclude)} existing prompts from sample_prompts.txt")

    print(f"Loading captions from {COCO_JSON}")
    captions = load_and_filter(COCO_JSON, exclude)
    print(f"After filter+dedup+exclusion: {len(captions)} captions")

    prompts = stratified_hard_sample(captions, TARGET_N, SEED)
    assert len(prompts) == TARGET_N, f"Expected {TARGET_N}, got {len(prompts)}"

    with open(OUTPUT_FILE, "w") as f:
        for p in prompts:
            f.write(p + "\n")

    print(f"\nSaved {TARGET_N} prompts to {OUTPUT_FILE}")

    # Summary stats
    print(f"\n--- Difficulty stats ---")
    scores = [difficulty_score(p) for p in prompts]
    print(f"  Min score:  {min(scores):.2f}")
    print(f"  Max score:  {max(scores):.2f}")
    print(f"  Mean score: {np.mean(scores):.2f}")
    print(f"  Median:     {np.median(scores):.2f}")

    # Category distribution
    cat_counts: dict[str, int] = defaultdict(int)
    for p in prompts:
        cat_counts[assign_supercategory(p)] += 1
    print(f"\n--- Category distribution ---")
    for cat in sorted(cat_counts):
        print(f"  {cat:<12}: {cat_counts[cat]}")

    # Print first 10 as sample
    print(f"\n--- First 10 prompts ---")
    for i, p in enumerate(prompts[:10], 1):
        cat = assign_supercategory(p)
        print(f"[{i:03d}] [{cat:<12}] {p}")


if __name__ == "__main__":
    main()
