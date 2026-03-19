"""
Calibration config for Phase 1 EDA (PTQ4DiT + Q-Diffusion fusion study).

EDA departs from the original TaQ-DiT paper setup in two ways:
  - 100 prompts (sample_prompts.txt) instead of 256 fixed prompts
  - 30 Euler steps collected in full instead of 25-of-100 (dense timestep coverage
    for temporal drift analysis; no subsampling needed with flow-matching schedule)
"""

from pathlib import Path

# SD3 flow-matching Euler schedule: run 30 steps end-to-end
NUM_SAMPLING_STEPS = 30

# 100 prompts (one trajectory each) → 100 × 30 = 3,000 calibration points
NUM_CALIBRATION_SAMPLES = 100

# Collect ALL 30 timesteps (no subsampling)
NUM_SELECTED_TIMESTEPS = 30

# DiffusionKit SD3 Medium for MLX (default)
MODEL_VERSION = "argmaxinc/mlx-stable-diffusion-3-medium"

# Latent size (height//8, width//8) for 512x512
DEFAULT_LATENT_SIZE = (64, 64)

# CFG scale for calibration
DEFAULT_CFG_WEIGHT = 4

# MS-COCO sampling parameters
COCO_SEED = 42
COCO_WORD_COUNT_MIN = 5
COCO_WORD_COUNT_MAX = 30

# Output directory for EDA artifacts
EDA_OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "eda_output")

# Default prompt file: 100 prompts used for calibration
DEFAULT_PROMPT_FILE = str(Path(__file__).resolve().parent / "sample_prompts.txt")
