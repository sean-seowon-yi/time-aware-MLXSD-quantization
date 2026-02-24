"""
Phase 1 calibration config: paper-aligned constants for calibration data generation.
"""

from pathlib import Path

# Paper: "Set the sampling steps to 100"
NUM_SAMPLING_STEPS = 100

# Paper: "Generate 256 samples"
NUM_CALIBRATION_SAMPLES = 256

# Paper: "Uniformly select 25 steps from the total steps"
NUM_SELECTED_TIMESTEPS = 25

# DiffusionKit SD3 Medium for MLX (default)
MODEL_VERSION = "argmaxinc/mlx-stable-diffusion-3-medium"

# Latent size (height//8, width//8) for 256x256
DEFAULT_LATENT_SIZE = (32, 32)

# CFG scale for calibration (paper: 1.50)
DEFAULT_CFG_WEIGHT = 1.5

# Default prompt file (shipped alongside this module)
DEFAULT_PROMPT_FILE = str(Path(__file__).resolve().parent / "sample_prompts.txt")
