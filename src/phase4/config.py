"""Phase 4 AdaRound configuration."""

from pathlib import Path

# ---------------------------------------------------------------------------
# AdaRound hyperparameters
# ---------------------------------------------------------------------------

PHASE4_CONFIG = {
    # Optimisation
    "n_iters": 1000,          # gradient steps per layer
    "lr": 1e-3,               # Adam learning rate
    "batch_size": 8,          # calibration samples per iter
    # Data collection
    "n_prompts": 16,          # prompt-seed pairs to run
    "n_steps": 30,            # denoising steps per prompt
    "cfg_weight": 4.0,        # classifier-free guidance scale
    "max_tokens_per_sample": 64,   # token subsampling per forward pass
    # Quantisation (must match Phase 2)
    "bits": 4,
    "group_size": 64,
}

PHASE4_DATA_DIR = "phase4_calibration"
PHASE4_META_FILENAME = "phase4_meta.json"
