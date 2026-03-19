"""Q-Diffusion configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class QDiffusionConfig:
    # Bit widths
    weight_bits: int = 4                # 4 for W4A8, 8 for W8A8
    activation_bits: int = 8            # Always 8

    # Weight quantization
    weight_symmetric: bool = True
    weight_per_channel: bool = True

    # Activation quantization
    act_symmetric: bool = True          # Default; fc2 inputs override to asymmetric (post-GELU)

    # AdaRound
    adaround_iters: int = 3000          # Per block
    adaround_lr: float = 1e-3
    adaround_beta_start: float = 2.0    # Initial beta (soft rounding)
    adaround_beta_end: float = 20.0     # Final beta (hard rounding, pushes V -> 0/1)
    adaround_warmup: float = 0.2        # Fraction of iters for beta annealing
    adaround_reg_weight: float = 0.01

    # Block reconstruction
    batch_size: int = 16                # Mini-batch for AdaRound optimization (FinalLayer)
    adaround_batch_groups: int = 2      # Timestep groups sampled per loss_fn call
    n_samples: int = 256                # Subset of cali data per block

    # Activation calibration
    act_calibration_method: str = "mse_search"  # "mse_search" or "percentile"
    act_percentile: float = 99.99       # Used when method == "percentile"
    act_search_candidates: List[float] = field(
        default_factory=lambda: [99.0, 99.5, 99.9, 99.95, 99.99, 100.0]
    )

    # Fisher weighting
    use_fisher: bool = False

    # Paths
    calibration_file: str = "eda_output/coco_cali_data.npz"
    output_dir: str = "q_diffusion_output"

    # Options
    quantize_sdpa: bool = False
    skip_final_layer: bool = False

    # Logging
    log_every: int = 100                # Print loss summary every N iterations

    # Resume
    resume: bool = False                # Reuse existing .fp_target_cache / .naive_input_cache if valid
