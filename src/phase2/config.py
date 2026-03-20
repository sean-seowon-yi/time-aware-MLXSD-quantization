"""Phase 2 quantization configuration and constants."""

from pathlib import Path

PHASE2_CONFIG = {
    "alpha": 0.5,
    "b_min": 1e-2,
    "b_max": 1e2,
    "w_eps": 1e-12,
    "group_size": 64,
    "bits": 4,
    "a_bits": 8,
    "qkv_method": "max",
    "final_layer_bits": 4,
    "exclude_layers": ["context_embedder"],
}

DIAGNOSTICS_DIR = Path("diagnostics")
MODEL_VERSION = "argmaxinc/mlx-stable-diffusion-3-medium"
HIDDEN_SIZE = 1536

DIFFUSIONKIT_SRC = str(Path("DiffusionKit/python/src"))

PIPELINE_KWARGS = {
    "w16": True,
    "shift": 1.0,
    "use_t5": True,
    "low_memory_mode": False,
}

QUANTIZED_WEIGHTS_FILENAME = "mmdit_quantized.safetensors"
QUANTIZE_CONFIG_FILENAME = "quantize_config.json"
