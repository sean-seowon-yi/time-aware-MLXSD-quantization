"""Phase 1 diagnostic configuration: inference settings, prompts, and visual constants."""

import csv
from pathlib import Path

SETTINGS_DIR = Path(__file__).resolve().parent.parent / "settings"


def _load_prompts_csv(path: Path) -> list[str]:
    """Load prompts from a CSV file with a 'prompt' column."""
    with open(path) as f:
        reader = csv.DictReader(f)
        return [row["prompt"] for row in reader]


DIAG_CONFIG = {
    "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
    "w16": True,
    "shift": 1.0,
    "use_t5": True,
    "low_memory_mode": False,
    "num_steps": 30,
    "cfg_weight": 4.0,
    "latent_size": (64, 64),
    "seed_range": [42],
    "top_k": 32,
}

OUTPUT_DIR = Path("diagnostics")
PLOTS_DIR = OUTPUT_DIR / "plots"
ACTIVATION_STATS_DIR = OUTPUT_DIR / "activation_stats"

CALIBRATION_PROMPTS_CSV = SETTINGS_DIR / "coco_100_calibration_prompts.csv"
DIAGNOSTIC_PROMPTS = _load_prompts_csv(CALIBRATION_PROMPTS_CSV)

REPRESENTATIVE_LAYERS = [
    "blocks.0.image.attn.q_proj",
    "blocks.12.text.attn.q_proj",
    "blocks.12.image.attn.o_proj",
    "blocks.12.image.mlp.fc1",
    "blocks.12.image.mlp.fc2",
    "context_embedder",
    "final_layer.linear",
]

FAMILY_COLORS = {
    "q_proj": "#2980b9",
    "k_proj": "#1abc9c",
    "v_proj": "#8e44ad",
    "o_proj": "#7f8c8d",
    "fc1": "#e67e22",
    "fc2": "#e74c3c",
    "context_embedder": "#27ae60",
    "final_linear": "#2c3e50",
}

SIDE_COLORS = {
    "image": "#3498db",
    "text": "#e67e22",
    "shared": "#27ae60",
}
