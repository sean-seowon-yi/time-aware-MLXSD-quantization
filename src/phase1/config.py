"""Phase 1 diagnostic configuration: inference settings, prompts, and visual constants."""

from pathlib import Path

DIAG_CONFIG = {
    "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
    "w16": True,
    "shift": 1.0,
    "use_t5": True,
    "low_memory_mode": False,
    "num_steps": 28,
    "cfg_weight": 0.0,
    "latent_size": (64, 64),
    "seed_range": list(range(42, 50)),
    "top_k": 32,
}

OUTPUT_DIR = Path("diagnostics")
PLOTS_DIR = OUTPUT_DIR / "plots"
ACTIVATION_STATS_DIR = OUTPUT_DIR / "activation_stats"

DIAGNOSTIC_PROMPTS = [
    "a red cube on a white table",
    "a Victorian library with dust motes in golden afternoon light, leather-bound books, and a sleeping cat on a velvet armchair",
    "a neon sign reading OPEN 24 HOURS against a dark alley wall",
    "three blue spheres and two yellow cones arranged on a checkerboard floor",
    "an oil painting in the style of Vermeer depicting a woman reading a letter",
    "a bustling Tokyo intersection at night with crowds, taxis, and neon signs",
    "portrait of an elderly woman smiling, soft studio lighting",
    "entropy and order in visual tension, abstract geometric composition",
    "a single photorealistic water droplet on a leaf, macro photography",
    "a medieval castle on a cliff overlooking a stormy sea at sunset",
    "a child's drawing of a house with a sun and clouds, crayon on paper",
    "an astronaut riding a horse on the surface of Mars, cinematic",
    "a bowl of ramen with steam rising, overhead view, food photography",
    "blueprint technical drawing of a spacecraft with annotations",
    "a field of sunflowers under dramatic cumulonimbus clouds",
    "two cats sitting symmetrically on a windowsill, silhouette against sunset",
    "a dense jungle with a hidden ancient temple, volumetric light rays",
    "minimalist flat vector illustration of a coffee cup",
    "a crowded bookshelf seen through a magnifying glass, tilt-shift effect",
    "the word HELLO written in fire against a black background",
]

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
