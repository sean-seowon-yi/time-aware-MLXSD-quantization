"""
Phase 1 calibration config: paper-aligned constants for calibration data generation.

Aligned with arXiv:2503.06930 (HTG), Section 5.1:
  "We randomly generate 32 class-conditioned samples and save both the
   intermediate and output feature maps at each timestep."

SD3 adaptation: class-conditioned → text-conditioned (20 diverse prompts).
"""

from pathlib import Path

# Paper: 100-step diffusion process
NUM_SAMPLING_STEPS = 100

# Paper: "randomly generate 32 ... samples"
NUM_CALIBRATION_SAMPLES = 32

# DiffusionKit SD3 Medium for MLX (default)
MODEL_VERSION = "argmaxinc/mlx-stable-diffusion-3-medium"

# Latent size (height//8, width//8) for 256x256
DEFAULT_LATENT_SIZE = (32, 32)

# CFG scale for calibration (paper: 1.50)
DEFAULT_CFG_WEIGHT = 1.5

# Default prompt file (shipped alongside this module)
DEFAULT_PROMPT_FILE = str(Path(__file__).resolve().parent / "sample_prompts.txt")
