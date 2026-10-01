"""Phase 1 diagnostic configuration: inference settings, prompts, and visual constants."""

from pathlib import Path

SETTINGS_DIR = Path(__file__).resolve().parent.parent / "settings"


def _load_seed_prompt_pairs(path: Path) -> list[tuple[int, str]]:
    """Load ``(seed, prompt)`` pairs from a tab-separated text file.

    Each non-blank line has the format ``<seed>\\t<prompt>``.
    """
    pairs: list[tuple[int, str]] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        seed_str, prompt = line.split("\t", 1)
        pairs.append((int(seed_str), prompt.strip()))
    return pairs


DIAG_CONFIG = {
    "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
    "w16": True,
    "shift": 3.0,
    "use_t5": True,
    "low_memory_mode": False,
    "num_steps": 30,
    "cfg_weight": 4.0,
    "latent_size": (64, 64),
    "top_k": 32,
}

OUTPUT_DIR = Path("diagnostics")
PLOTS_DIR = OUTPUT_DIR / "plots"
ACTIVATION_STATS_DIR = OUTPUT_DIR / "activation_stats"

CALIBRATION_PROMPTS_FILE = SETTINGS_DIR / "coco_100_calibration_prompts.txt"
CALIBRATION_PAIRS = _load_seed_prompt_pairs(CALIBRATION_PROMPTS_FILE)
DIAGNOSTIC_PROMPTS = [prompt for _, prompt in CALIBRATION_PAIRS]
DIAGNOSTIC_SEEDS = [seed for seed, _ in CALIBRATION_PAIRS]

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
