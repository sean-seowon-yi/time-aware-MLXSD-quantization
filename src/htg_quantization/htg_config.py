"""
Phase 3 HTG configuration: paper-aligned constants.

References
----------
arXiv:2503.06930, Sec. 5.1 "Quantization Configurations"
"""

# Number of temporal groups. None → auto-set to T // 10 per the paper.
# Table 3 shows G = T // 10 achieves near-identical FID to G = T at <1% storage overhead.
NUM_GROUPS: int | None = None

# EMA coefficient α for the channel-wise scaling accumulation (Equation 7).
# Paper ablation (Table 4): α = 0.99 yields best FID.
EMA_ALPHA: float = 0.99

# Which input activations to profile and apply HTG to.
# "fc1"   → input to mlp.fc1 (preceded by AdaLN post_norm2)
# "qkv"   → input to attn.q/k/v_proj (preceded by AdaLN post_norm1)
# "oproj" → input to attn.o_proj (preceded by SDPA output, no AdaLN)
TARGET_LAYER_TYPES: list[str] = ["fc1", "qkv", "oproj"]

# Quantization precision for model weights (W8 or W4).
WEIGHT_BITS: int = 8

# Activation quantization bit-width (used for per-tensor static quantizer ranges).
ACTIVATION_BITS: int = 8

# Group size for MLX block-wise weight quantization.
# MLX default is 64; must evenly divide hidden_size (1536 for SD3 Medium).
QUANTIZATION_GROUP_SIZE: int = 64

# DiffusionKit model identifier (matches Phase 1 and Phase 2).
MODEL_VERSION: str = "argmaxinc/mlx-stable-diffusion-3-medium"

# Default filenames (relative to working directory unless overridden via CLI).
DEFAULT_CALIBRATION_FILE: str = "DiT_cali_data.npz"
DEFAULT_INPUT_STATS_FILE: str = "htg_input_activation_stats.npz"
DEFAULT_HTG_PARAMS_FILE: str = "htg_params.npz"
DEFAULT_OUTPUT_DIR: str = "htg_output"
