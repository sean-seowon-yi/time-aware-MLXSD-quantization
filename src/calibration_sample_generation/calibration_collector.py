"""
Phase 1 calibration collector: run Euler sampling (batch_size=1) and collect
(x, t) at each denoising step.

batch_size is fixed to 1 because CFGDenoiser doubles the batch internally
when cfg_weight > 0, and conditioning from encode_text is already shaped for
that (batch=2 for [positive, negative]).  Running batch > 1 would create a
shape mismatch between the doubled latent and the conditioning tensors.
"""

import mlx.core as mx
from tqdm import trange

from diffusionkit.mlx import (
    CFGDenoiser,
    to_d,
)


def sample_euler_with_calibration(
    pipeline,
    x_init,
    sigmas,
    conditioning,
    pooled_conditioning,
    cfg_weight: float = 1.5,
):
    """
    Euler (deterministic ODE) sampler with per-step (x, t) collection.

    x_init must have batch=1.  conditioning / pooled_conditioning come
    directly from pipeline.encode_text (shape [2, ...] when CFG > 0).

    Returns:
        x_cali_list: list of length num_steps, each shape (1, H, W, C)
        t_cali_list: list of length num_steps, each is a timestep scalar
    """
    assert x_init.shape[0] == 1, (
        f"batch must be 1 for correct CFG handling, got {x_init.shape[0]}"
    )

    model = CFGDenoiser(pipeline)
    timesteps = pipeline.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
    model.cache_modulation_params(pooled_conditioning, timesteps)

    x = x_init
    x_cali_list = []
    t_cali_list = []

    n_steps = len(sigmas) - 1
    for i in trange(n_steps, desc="euler calibration", leave=False):
        x_cali_list.append(x)
        t_cali_list.append(timesteps[i])

        denoised = model(
            x,
            timesteps[i],
            sigmas[i],
            conditioning=conditioning,
            cfg_weight=cfg_weight,
            pooled_conditioning=pooled_conditioning,
        )
        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt
        mx.eval(x)

    model.clear_cache()
    return x_cali_list, t_cali_list
