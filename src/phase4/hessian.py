"""Hessian collection via transparent proxy layers.

Install _HessianCollector proxies on ALL block linears (across all
24 blocks) plus ``final_layer.linear``.  Run calibration prompts
through the Euler denoising loop to accumulate H = X^T X per layer.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import mlx.core as mx

from .utils import (
    _get_block_linears,
    _set_nested,
    _reset_modulation_cache,
    full_path_to_poly_key,
    get_poly_alpha,
)

logger = logging.getLogger(__name__)

FINAL_LAYER_KEY = "final_layer.linear"
FINAL_BLOCK_IDX = "final"


# ---------------------------------------------------------------------------
# Hessian collector proxy
# ---------------------------------------------------------------------------

class _HessianCollector:
    """Proxy that replaces an nn.Linear during Hessian collection."""

    def __init__(
        self,
        wrapped,
        poly_entry: Optional[dict] = None,
        raw_hessian: bool = False,
    ):
        self._wrapped = wrapped
        self._H: Optional[mx.array] = None
        self._n_samples: int = 0
        self._poly_entry = poly_entry
        self._sigma: Optional[float] = None
        self._raw_hessian: bool = raw_hessian

    def __call__(self, x):
        out = self._wrapped(x)

        # The Hessian must be computed from the input that the weight
        # matrix (qlinear) actually receives.  The wrapped W4A8*Linear
        # applies b_inv scaling before the qlinear — we must replicate
        # that here so the Hessian reflects post-CSB activations.
        x_h = x
        if hasattr(self._wrapped, "b_inv"):
            x_h = (x_h * self._wrapped.b_inv).astype(x.dtype)

        if self._raw_hessian:
            x_fq = x_h
        elif self._poly_entry is not None and self._sigma is not None:
            alpha = get_poly_alpha(self._poly_entry, self._sigma)
            scale = alpha / 127.0
            x_fq = mx.clip(mx.round(x_h / scale), -128, 127) * scale
        else:
            x_fq = x_h

        x_2d = x_fq.reshape(-1, x_fq.shape[-1]).astype(mx.float32)
        xtx = x_2d.T @ x_2d
        if self._H is None:
            self._H = mx.zeros_like(xtx)
        self._H = self._H + xtx
        self._n_samples += x_2d.shape[0]

        return out

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def set_sigma(self, sigma: float):
        self._sigma = sigma

    def get_hessian(self) -> np.ndarray:
        """Return 2 * H as a NumPy float32 array."""
        if self._H is None:
            raise RuntimeError("No data collected")
        mx.eval(self._H)
        return 2.0 * np.array(self._H, dtype=np.float32)


# ---------------------------------------------------------------------------
# Install / remove helpers
# ---------------------------------------------------------------------------

def install_collectors(
    mmdit,
    block_idx: int,
    poly_schedule: dict,
    raw_hessian: bool = False,
) -> Dict[str, _HessianCollector]:
    """Install collectors on one block's linears. Returns {poly_key: collector}."""
    block = mmdit.multimodal_transformer_blocks[block_idx]
    layers_dict = poly_schedule.get("layers", {})
    collectors = {}
    for full_path, layer in _get_block_linears(block, is_mm=True):
        poly_key = full_path_to_poly_key(block_idx, full_path)
        poly_entry = layers_dict.get(poly_key)
        collector = _HessianCollector(layer, poly_entry, raw_hessian=raw_hessian)
        _set_nested(block, full_path, collector)
        collectors[poly_key] = collector
    return collectors


def install_collectors_all_blocks(
    mmdit,
    poly_schedule: dict,
    raw_hessian: bool = False,
    include_final: bool = True,
) -> Dict[int | str, Dict[str, _HessianCollector]]:
    """Install Hessian collectors on ALL blocks (+ ``final_layer.linear``).

    Returns {block_idx: {poly_key: collector}, "final": {key: collector}}.
    The ``"final"`` entry is only included when *include_final* is True.
    """
    all_collectors: Dict[int | str, Dict[str, _HessianCollector]] = {}
    for block_idx in range(len(mmdit.multimodal_transformer_blocks)):
        all_collectors[block_idx] = install_collectors(
            mmdit, block_idx, poly_schedule, raw_hessian=raw_hessian,
        )

    if include_final:
        layers_dict = poly_schedule.get("layers", {})
        poly_entry = layers_dict.get(FINAL_LAYER_KEY)
        fl = mmdit.final_layer.linear
        collector = _HessianCollector(fl, poly_entry, raw_hessian=raw_hessian)
        mmdit.final_layer.linear = collector
        all_collectors[FINAL_BLOCK_IDX] = {FINAL_LAYER_KEY: collector}

    return all_collectors


def remove_collectors(
    mmdit,
    block_idx: int,
    collectors: Dict[str, _HessianCollector],
):
    """Restore original linears for one block."""
    block = mmdit.multimodal_transformer_blocks[block_idx]
    for full_path, _ in _get_block_linears(block, is_mm=True):
        poly_key = full_path_to_poly_key(block_idx, full_path)
        if poly_key in collectors:
            _set_nested(block, full_path, collectors[poly_key]._wrapped)


def remove_collectors_all_blocks(
    mmdit,
    all_collectors: Dict[int | str, Dict[str, _HessianCollector]],
):
    """Restore original linears for all blocks + final_layer."""
    for block_idx, collectors in all_collectors.items():
        if block_idx == FINAL_BLOCK_IDX:
            if FINAL_LAYER_KEY in collectors:
                mmdit.final_layer.linear = collectors[FINAL_LAYER_KEY]._wrapped
        else:
            remove_collectors(mmdit, block_idx, collectors)


def set_sigma_all_blocks(
    all_collectors: Dict[int | str, Dict[str, _HessianCollector]],
    sigma: float,
):
    """Set sigma on all collectors across all blocks (+ final_layer)."""
    for collectors in all_collectors.values():
        for c in collectors.values():
            c.set_sigma(sigma)


def eval_hessians_all_blocks(
    all_collectors: Dict[int | str, Dict[str, _HessianCollector]],
):
    """Call mx.eval() on all H matrices to bound the computation graph."""
    pending = []
    for collectors in all_collectors.values():
        pending.extend(c._H for c in collectors.values() if c._H is not None)
    if pending:
        mx.eval(*pending)


# ---------------------------------------------------------------------------
# Main collection entry point
# ---------------------------------------------------------------------------

def collect_hessians_global(
    pipeline,
    denoiser,
    prompt_entries: List[Tuple[int, str]],
    poly_schedule: dict,
    num_steps: int,
    cfg_weight: float = 4.0,
    latent_size: int = 64,
    raw_hessian: bool = False,
    include_final: bool = True,
) -> Dict[int | str, Dict[str, _HessianCollector]]:
    """Install collectors on ALL blocks (+ final_layer), run calibration prompts.

    Returns {block_idx: {poly_key: collector}, "final": {...}} with
    Hessians populated.  Collectors are removed from the model before
    return; Hessian data is accessible via ``collector.get_hessian()``.
    """
    from tqdm import tqdm

    mmdit = pipeline.mmdit
    all_collectors = install_collectors_all_blocks(
        mmdit, poly_schedule, raw_hessian=raw_hessian,
        include_final=include_final,
    )

    n_layers = sum(len(c) for c in all_collectors.values())
    n_blocks = len(mmdit.multimodal_transformer_blocks)
    mode = "raw" if raw_hessian else "poly"
    final_msg = " + final_layer" if include_final else ""
    logger.info(
        "Collecting Hessians on %d layers across %d blocks%s (%s mode)",
        n_layers, n_blocks, final_msg, mode,
    )

    n_prompts = len(prompt_entries)
    for pi, (seed, prompt) in enumerate(
        tqdm(prompt_entries, desc="Hessian prompts"), start=1,
    ):
        # tqdm only advances between prompts; first prompt can take many minutes.
        logger.info(
            "Hessian prompt %d/%d (seed=%d) — encoding + denoising ...",
            pi, n_prompts, seed,
        )
        conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
        mx.eval(conditioning, pooled)

        sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
        timesteps = pipeline.sampler.timestep(sigmas).astype(
            pipeline.activation_dtype
        )
        denoiser.cache_modulation_params(pooled, timesteps)

        mx.random.seed(seed)
        latent_shape = (1, latent_size, latent_size, 16)
        noise = mx.random.normal(latent_shape).astype(pipeline.activation_dtype)
        x = pipeline.sampler.noise_scaling(
            sigmas[0], noise, mx.zeros(latent_shape),
            pipeline.max_denoise(sigmas),
        )
        mx.eval(x)

        n_steps = len(sigmas) - 1
        for i in range(n_steps):
            set_sigma_all_blocks(all_collectors, float(sigmas[i]))
            denoised = denoiser(
                x, timesteps[i], sigmas[i],
                conditioning=conditioning, cfg_weight=cfg_weight,
            )
            if (i + 1) % 3 == 0 or i == n_steps - 1:
                eval_hessians_all_blocks(all_collectors)

            d = (x - denoised) / sigmas[i]
            x = x + d * (sigmas[i + 1] - sigmas[i])
            mx.eval(x)

        _reset_modulation_cache(pipeline)

    eval_hessians_all_blocks(all_collectors)
    remove_collectors_all_blocks(mmdit, all_collectors)
    return all_collectors
