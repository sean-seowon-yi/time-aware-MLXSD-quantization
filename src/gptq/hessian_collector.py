"""Hessian collection via transparent proxy layers.

Installs _HessianCollector proxies on target block linears during the
denoising forward pass. Each collector accumulates H = X^T X in MLX
and optionally caches (input, sigma, output) tuples for alpha search.

Two-phase design:
  Phase A — install on ALL blocks, cache_io=False. One pass through all
            prompts collects Hessians globally.
  Phase B — install on ONE block, cache_io=True. A short pass (few prompts)
            collects I/O samples for alpha search.
"""

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


class _HessianCollector:
    """Proxy that replaces an nn.Linear during Hessian collection."""

    def __init__(self, wrapped, poly_entry: Optional[dict] = None,
                 cache_io: bool = False, max_cache: int = 50,
                 cache_max_rows: int = 256,
                 static_alpha: Optional[float] = None):
        self._wrapped = wrapped
        self._H: Optional[mx.array] = None
        self._n_samples: int = 0
        self._poly_entry = poly_entry
        self._sigma: Optional[float] = None
        self._cache_io = cache_io
        self._input_cache: List[Tuple[np.ndarray, float]] = []
        self._output_cache: List[np.ndarray] = []
        self._max_cache: int = max_cache
        self._cache_max_rows: int = cache_max_rows  # subsample spatial dim
        self._call_count: int = 0
        self._static_alpha: Optional[float] = static_alpha  # if set, ignore sigma

    def __call__(self, x):
        out = self._wrapped(x)  # FP16 forward (unchanged)

        # Fake-quantize input for Hessian collection
        if self._static_alpha is not None:
            # Static mode: fixed alpha regardless of timestep
            scale = self._static_alpha / 127.0
            x_fq = mx.clip(mx.round(x / scale), -127, 127) * scale
        elif self._poly_entry is not None and self._sigma is not None:
            # Poly mode: timestep-varying alpha
            alpha = get_poly_alpha(self._poly_entry, self._sigma)
            scale = alpha / 127.0
            x_fq = mx.clip(mx.round(x / scale), -127, 127) * scale
        else:
            x_fq = x

        # Accumulate H += x_2d^T @ x_2d in float32 to avoid overflow.
        # Inputs are float16 (activation_dtype), but x^T x over ~4096
        # tokens easily exceeds float16 max (65504).
        x_2d = x_fq.reshape(-1, x_fq.shape[-1]).astype(mx.float32)
        xtx = x_2d.T @ x_2d
        if self._H is None:
            self._H = mx.zeros_like(xtx)
        self._H = self._H + xtx
        self._n_samples += x_2d.shape[0]

        # Cache for alpha search (only when enabled)
        if self._cache_io and len(self._input_cache) < self._max_cache:
            self._call_count += 1
            # Flatten to 2D and subsample rows to cap memory.
            # Full tensor can be (batch, 4096, 1536) ≈ 25 MB — too large to
            # cache hundreds of times.  Subsample to cache_max_rows rows so
            # each cached entry is at most cache_max_rows * d_in * 4 bytes.
            x_np = np.array(x).reshape(-1, x.shape[-1])   # (N, d_in)
            o_np = np.array(out).reshape(-1, out.shape[-1])  # (N, d_out)
            if x_np.shape[0] > self._cache_max_rows:
                idx = np.random.choice(
                    x_np.shape[0], self._cache_max_rows, replace=False
                )
                x_np = x_np[idx]
                o_np = o_np[idx]
            self._input_cache.append((x_np, self._sigma))
            self._output_cache.append(o_np)

        return out

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def set_sigma(self, sigma: float):
        self._sigma = sigma

    def get_hessian(self) -> np.ndarray:
        """Return 2 * H as a NumPy array."""
        if self._H is None:
            raise RuntimeError("No data collected")
        mx.eval(self._H)
        return 2.0 * np.array(self._H, dtype=np.float32)

    def get_cached_io(self) -> Tuple[List[Tuple[np.ndarray, float]], List[np.ndarray]]:
        return self._input_cache, self._output_cache


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def install_collectors(
    mmdit, block_idx: int, poly_schedule: dict,
    cache_io: bool = False, max_cache: int = 50,
    static_alphas: Optional[Dict[str, float]] = None,
) -> Dict[str, _HessianCollector]:
    """Install collectors on one block's linears. Returns {poly_key: collector}.

    Args:
        static_alphas: If provided, collectors use fixed alpha per layer
                       (timestep-agnostic) instead of poly evaluation.
    """
    block = mmdit.multimodal_transformer_blocks[block_idx]
    layers_dict = poly_schedule.get("layers", {})
    collectors = {}
    for full_path, layer in _get_block_linears(block, is_mm=True):
        poly_key = full_path_to_poly_key(block_idx, full_path)
        if static_alphas is not None:
            # Static mode: fixed alpha, no poly
            collector = _HessianCollector(
                layer, poly_entry=None,
                cache_io=cache_io, max_cache=max_cache,
                static_alpha=static_alphas.get(poly_key),
            )
        else:
            # Poly mode: timestep-varying alpha
            poly_entry = layers_dict.get(poly_key)
            collector = _HessianCollector(
                layer, poly_entry,
                cache_io=cache_io, max_cache=max_cache,
            )
        _set_nested(block, full_path, collector)
        collectors[poly_key] = collector
    return collectors


def install_collectors_all_blocks(
    mmdit, poly_schedule: dict,
    static_alphas: Optional[Dict[str, float]] = None,
) -> Dict[int, Dict[str, _HessianCollector]]:
    """Install Hessian-only collectors (no I/O cache) on ALL blocks.

    Returns {block_idx: {poly_key: collector}}.
    """
    all_collectors = {}
    for block_idx, _ in enumerate(mmdit.multimodal_transformer_blocks):
        all_collectors[block_idx] = install_collectors(
            mmdit, block_idx, poly_schedule, cache_io=False,
            static_alphas=static_alphas,
        )
    return all_collectors


def remove_collectors(mmdit, block_idx: int, collectors: Dict[str, "_HessianCollector"]):
    """Restore original linears."""
    block = mmdit.multimodal_transformer_blocks[block_idx]
    for full_path, _ in _get_block_linears(block, is_mm=True):
        poly_key = full_path_to_poly_key(block_idx, full_path)
        if poly_key in collectors:
            _set_nested(block, full_path, collectors[poly_key]._wrapped)


def remove_collectors_all_blocks(
    mmdit, all_collectors: Dict[int, Dict[str, "_HessianCollector"]],
):
    """Restore original linears for all blocks."""
    for block_idx, collectors in all_collectors.items():
        remove_collectors(mmdit, block_idx, collectors)


def set_sigma_all(collectors: Dict[str, _HessianCollector], sigma: float):
    """Set sigma on all collectors before a denoising step."""
    for c in collectors.values():
        c.set_sigma(sigma)


def set_sigma_all_blocks(
    all_collectors: Dict[int, Dict[str, _HessianCollector]], sigma: float,
):
    """Set sigma on all collectors across all blocks."""
    for collectors in all_collectors.values():
        set_sigma_all(collectors, sigma)


def eval_hessians(collectors: Dict[str, _HessianCollector]):
    """Call mx.eval() on all H matrices to bound computation graph."""
    pending = [c._H for c in collectors.values() if c._H is not None]
    if pending:
        mx.eval(*pending)


def eval_hessians_all_blocks(
    all_collectors: Dict[int, Dict[str, _HessianCollector]],
):
    """Eval H matrices across all blocks."""
    pending = []
    for collectors in all_collectors.values():
        pending.extend(c._H for c in collectors.values() if c._H is not None)
    if pending:
        mx.eval(*pending)


# ---------------------------------------------------------------------------
# Phase A: Global Hessian collection (all blocks, no I/O cache)
# ---------------------------------------------------------------------------

def collect_hessians_global(
    pipeline,
    denoiser,
    prompts: List[str],
    poly_schedule: dict,
    num_steps: int,
    cfg_weight: float = 4.0,
    seed: int = 42,
    latent_size: int = 64,
    static_alphas: Optional[Dict[str, float]] = None,
) -> Dict[int, Dict[str, _HessianCollector]]:
    """Phase A: install collectors on ALL blocks, run all prompts once.

    Returns {block_idx: {poly_key: collector}} with Hessians populated.
    No I/O is cached — use collect_io_for_block for alpha search data.

    Args:
        static_alphas: If provided, use fixed alpha per layer instead of poly.
    """
    from tqdm import tqdm

    mmdit = pipeline.mmdit
    all_collectors = install_collectors_all_blocks(
        mmdit, poly_schedule, static_alphas=static_alphas,
    )
    n_blocks = len(mmdit.multimodal_transformer_blocks)
    mode = "static" if static_alphas else "poly"

    print(f"Phase A: collecting Hessians on all {n_blocks} blocks ({mode} mode)")
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Hessian prompts")):
        conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
        mx.eval(conditioning, pooled)

        sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
        timesteps = pipeline.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
        denoiser.cache_modulation_params(pooled, timesteps)

        mx.random.seed(seed)
        latent_shape = (1, latent_size, latent_size, 16)
        noise = mx.random.normal(latent_shape).astype(pipeline.activation_dtype)
        x = pipeline.sampler.noise_scaling(
            sigmas[0], noise, mx.zeros(latent_shape), pipeline.max_denoise(sigmas)
        )
        mx.eval(x)

        for i in range(len(sigmas) - 1):
            set_sigma_all_blocks(all_collectors, float(sigmas[i]))
            denoised = denoiser(
                x, timesteps[i], sigmas[i],
                conditioning=conditioning, cfg_weight=cfg_weight,
            )
            # Eval Hessians EVERY step to prevent the lazy computation
            # graph from exploding (285 collectors × 30 steps = 8550
            # pending XtX ops would OOM before any eval fires).
            eval_hessians_all_blocks(all_collectors)

            d = (x - denoised) / sigmas[i]
            x = x + d * (sigmas[i + 1] - sigmas[i])
            mx.eval(x)

        _reset_modulation_cache(pipeline)

    eval_hessians_all_blocks(all_collectors)
    # Remove all collectors — Hessians are already materialized in NumPy-accessible form
    remove_collectors_all_blocks(mmdit, all_collectors)
    return all_collectors


# ---------------------------------------------------------------------------
# Phase B: Per-block I/O cache collection (one block, short pass)
# ---------------------------------------------------------------------------

def collect_io_for_block(
    pipeline,
    denoiser,
    block_idx: int,
    prompts: List[str],
    poly_schedule: dict,
    num_steps: int,
    max_cache: int = 50,
    cfg_weight: float = 4.0,
    seed: int = 42,
    latent_size: int = 64,
    static_alphas: Optional[Dict[str, float]] = None,
) -> Dict[str, _HessianCollector]:
    """Phase B: install cache-only collectors on one block, run a short prompt set.

    The collectors do NOT accumulate Hessians (they will, but we ignore them).
    Returns {poly_key: collector} with cached I/O for alpha search.

    Args:
        static_alphas: If provided, use fixed alpha per layer instead of poly.
    """
    from tqdm import tqdm

    mmdit = pipeline.mmdit
    collectors = install_collectors(
        mmdit, block_idx, poly_schedule, cache_io=True, max_cache=max_cache,
        static_alphas=static_alphas,
    )

    for prompt_idx, prompt in enumerate(tqdm(
        prompts, desc=f"Block {block_idx} I/O cache", leave=False
    )):
        conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
        mx.eval(conditioning, pooled)

        sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
        timesteps = pipeline.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
        denoiser.cache_modulation_params(pooled, timesteps)

        mx.random.seed(seed)
        latent_shape = (1, latent_size, latent_size, 16)
        noise = mx.random.normal(latent_shape).astype(pipeline.activation_dtype)
        x = pipeline.sampler.noise_scaling(
            sigmas[0], noise, mx.zeros(latent_shape), pipeline.max_denoise(sigmas)
        )
        mx.eval(x)

        for i in range(len(sigmas) - 1):
            set_sigma_all(collectors, float(sigmas[i]))
            denoised = denoiser(
                x, timesteps[i], sigmas[i],
                conditioning=conditioning, cfg_weight=cfg_weight,
            )
            eval_hessians(collectors)

            d = (x - denoised) / sigmas[i]
            x = x + d * (sigmas[i + 1] - sigmas[i])
            mx.eval(x)

        _reset_modulation_cache(pipeline)

    remove_collectors(mmdit, block_idx, collectors)
    return collectors
