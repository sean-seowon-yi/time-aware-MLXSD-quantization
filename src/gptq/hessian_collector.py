"""Hessian collection and alpha search via transparent proxy layers.

Two-phase design:
  Phase A — Install _HessianCollector proxies on ALL 285 linears across
            all 24 blocks. Run all prompts once to accumulate H = X^T X
            per layer. No I/O caching.

  Phase B — Install _AlphaAccumulator proxies on ALL linears. Run a small
            subset of prompts. Each accumulator evaluates 19 alpha_scale
            candidates (0.01–100.0) per forward call via vectorized NumPy
            matmuls and accumulates per-candidate MSE sums.
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
    dequantize,
)


class _HessianCollector:
    """Proxy that replaces an nn.Linear during Hessian collection."""

    def __init__(self, wrapped, poly_entry: Optional[dict] = None,
                 static_alpha: Optional[float] = None,
                 raw_hessian: bool = False):
        self._wrapped = wrapped
        self._H: Optional[mx.array] = None
        self._n_samples: int = 0
        self._poly_entry = poly_entry
        self._sigma: Optional[float] = None
        self._static_alpha: Optional[float] = static_alpha
        self._raw_hessian: bool = raw_hessian

    def __call__(self, x):
        out = self._wrapped(x)  # FP16 forward (unchanged)

        if self._raw_hessian:
            # Use full-precision activations for Hessian — avoids conditioning
            # the weight quantization on activation clipping parameters that
            # aren't finalized until Phase B alpha search.
            x_fq = x
        elif self._static_alpha is not None:
            scale = self._static_alpha / 127.0
            x_fq = mx.clip(mx.round(x / scale), -127, 127) * scale
        elif self._poly_entry is not None and self._sigma is not None:
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def install_collectors(
    mmdit, block_idx: int, poly_schedule: dict,
    static_alphas: Optional[Dict[str, float]] = None,
    raw_hessian: bool = False,
) -> Dict[str, _HessianCollector]:
    """Install collectors on one block's linears. Returns {poly_key: collector}.

    Args:
        static_alphas: If provided, collectors use fixed alpha per layer
                       (timestep-agnostic) instead of poly evaluation.
        raw_hessian: If True, accumulate H from full-precision activations
                     instead of fake-quantized activations.
    """
    block = mmdit.multimodal_transformer_blocks[block_idx]
    layers_dict = poly_schedule.get("layers", {})
    collectors = {}
    for full_path, layer in _get_block_linears(block, is_mm=True):
        poly_key = full_path_to_poly_key(block_idx, full_path)
        if static_alphas is not None:
            collector = _HessianCollector(
                layer, poly_entry=None,
                static_alpha=static_alphas.get(poly_key),
                raw_hessian=raw_hessian,
            )
        else:
            poly_entry = layers_dict.get(poly_key)
            collector = _HessianCollector(layer, poly_entry, raw_hessian=raw_hessian)
        _set_nested(block, full_path, collector)
        collectors[poly_key] = collector
    return collectors


def install_collectors_all_blocks(
    mmdit, poly_schedule: dict,
    static_alphas: Optional[Dict[str, float]] = None,
    raw_hessian: bool = False,
) -> Dict[int, Dict[str, _HessianCollector]]:
    """Install Hessian-only collectors on ALL blocks.

    Returns {block_idx: {poly_key: collector}}.
    """
    all_collectors = {}
    for block_idx, _ in enumerate(mmdit.multimodal_transformer_blocks):
        all_collectors[block_idx] = install_collectors(
            mmdit, block_idx, poly_schedule,
            static_alphas=static_alphas,
            raw_hessian=raw_hessian,
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
    prompt_entries: List[Tuple[int, str]],
    poly_schedule: dict,
    num_steps: int,
    cfg_weight: float = 4.0,
    latent_size: int = 64,
    static_alphas: Optional[Dict[str, float]] = None,
    raw_hessian: bool = False,
) -> Dict[int, Dict[str, _HessianCollector]]:
    """Phase A: install collectors on ALL blocks, run all prompts once.

    Returns {block_idx: {poly_key: collector}} with Hessians populated.

    Args:
        prompt_entries: List of (seed, prompt) tuples.
        static_alphas: If provided, use fixed alpha per layer instead of poly.
        raw_hessian: If True, accumulate H from full-precision activations
                     instead of fake-quantized activations.
    """
    from tqdm import tqdm

    mmdit = pipeline.mmdit
    all_collectors = install_collectors_all_blocks(
        mmdit, poly_schedule, static_alphas=static_alphas, raw_hessian=raw_hessian,
    )
    n_blocks = len(mmdit.multimodal_transformer_blocks)
    if raw_hessian:
        mode = "raw"
    elif static_alphas:
        mode = "static"
    else:
        mode = "poly"

    print(f"Phase A: collecting Hessians on all {n_blocks} blocks ({mode} mode)")
    for seed, prompt in tqdm(prompt_entries, desc="Hessian prompts"):
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

        n_steps = len(sigmas) - 1
        for i in range(n_steps):
            set_sigma_all_blocks(all_collectors, float(sigmas[i]))
            denoised = denoiser(
                x, timesteps[i], sigmas[i],
                conditioning=conditioning, cfg_weight=cfg_weight,
            )
            # Eval Hessians every 3 steps to bound the lazy computation
            # graph (285 collectors × 3 steps = 855 pending XtX ops is
            # safe; 285 × 30 = 8550 would OOM).
            if (i + 1) % 3 == 0 or i == n_steps - 1:
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
# Phase B v2: Global alpha MSE accumulation (all blocks, one pass)
# ---------------------------------------------------------------------------

# Alpha candidates: non-uniform grid covering [0.01, 100.0]
_ALPHA_CANDIDATES = [0.01, 0.1, 0.3, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6, 8, 10, 25, 50, 100]


class _AlphaAccumulator:
    """Proxy that accumulates MSE for all alpha_scale candidates on-the-fly.

    Instead of caching raw (input, output) pairs, this evaluates all 19
    alpha_scale candidates in a single vectorized matmul per forward call
    and accumulates per-candidate MSE sums. Memory per layer: 19 floats
    (negligible).

    After collection, call get_best_alpha() to retrieve the optimal scale.
    """

    # Pre-compute alpha scales array once (shared across instances)
    _ALPHA_SCALES = np.array(_ALPHA_CANDIDATES, dtype=np.float32)  # (C,)

    def __init__(self, wrapped, W_q_dequant: np.ndarray,
                 bias: Optional[np.ndarray],
                 poly_entry: Optional[dict] = None,
                 static_alpha: Optional[float] = None,
                 subsample_rows: int = 128):
        self._wrapped = wrapped
        self._W_q_dequant = W_q_dequant        # (d_out, d_in) float32
        self._bias = bias                        # (d_out,) or None
        self._poly_entry = poly_entry
        self._static_alpha = static_alpha
        self._sigma: Optional[float] = None
        self._subsample_rows = subsample_rows

        # Per-candidate accumulators
        n_cands = len(_ALPHA_CANDIDATES)
        self._total_se = np.zeros(n_cands, dtype=np.float64)
        self._total_elements = 0

    def __call__(self, x):
        out = self._wrapped(x)  # FP16 forward (unchanged)

        # Convert to numpy, flatten, subsample
        x_np = np.array(x).reshape(-1, x.shape[-1])    # (N, d_in)
        o_np = np.array(out).reshape(-1, out.shape[-1])  # (N, d_out)
        if x_np.shape[0] > self._subsample_rows:
            idx = np.linspace(
                0, x_np.shape[0] - 1, self._subsample_rows, dtype=int
            )
            x_np = x_np[idx]
            o_np = o_np[idx]

        self._total_elements += o_np.size

        # Compute base alpha from sigma
        if self._static_alpha is not None:
            base_alpha = self._static_alpha
        elif self._poly_entry is not None and self._sigma is not None:
            base_alpha = get_poly_alpha(self._poly_entry, self._sigma)
        else:
            return out

        # Vectorized evaluation of all candidates in one batched matmul.
        # scales: (C,) — one per candidate
        scales = (self._ALPHA_SCALES * base_alpha) / 127.0  # (C,)
        qmax = 127

        # Broadcast fake-quantize: x_np (N, d_in) / scales (C, 1, 1)
        # → x_all (C, N, d_in)
        scales_3d = scales[:, None, None]                    # (C, 1, 1)
        x_all = np.clip(
            np.round(x_np[None, :, :] / scales_3d),         # (C, N, d_in)
            -qmax, qmax,
        ) * scales_3d                                        # (C, N, d_in)

        # Batched matmul: (C, N, d_in) @ (d_in, d_out) → (C, N, d_out)
        y_all = x_all @ self._W_q_dequant.T                 # (C, N, d_out)
        if self._bias is not None:
            y_all = y_all + self._bias                       # broadcast (d_out,)

        # Per-candidate SE: sum over (N, d_out) dimensions
        se = np.sum((y_all - o_np[None, :, :]) ** 2, axis=(1, 2))  # (C,)
        self._total_se += se

        return out

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def set_sigma(self, sigma: float):
        self._sigma = sigma

    def get_best_alpha(self) -> Tuple[float, float]:
        """Return (best_alpha_scale, best_mse)."""
        if self._total_elements == 0:
            return 1.0, float("inf")
        mses = self._total_se / self._total_elements
        best_idx = int(np.argmin(mses))
        return _ALPHA_CANDIDATES[best_idx], float(mses[best_idx])


def install_alpha_accumulators(
    mmdit, poly_schedule: dict,
    weight_results: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    static_alphas: Optional[Dict[str, float]] = None,
    subsample_rows: int = 128,
) -> Dict[str, _AlphaAccumulator]:
    """Install alpha accumulators on ALL blocks' linears.

    Args:
        weight_results: {poly_key: (W_q_int, scales, weight_mse)} from GPTQ.
        static_alphas: If provided, use fixed alpha per layer.

    Returns {poly_key: accumulator}.
    """
    layers_dict = poly_schedule.get("layers", {})
    accumulators = {}

    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            if poly_key not in weight_results:
                continue

            W_q_int, scales, _ = weight_results[poly_key]
            W_q_dequant = dequantize(W_q_int, scales)

            bias = None
            if hasattr(layer, "bias") and layer.bias is not None:
                bias = np.array(layer.bias, dtype=np.float32)

            poly_entry = layers_dict.get(poly_key) if static_alphas is None else None
            static_alpha = static_alphas.get(poly_key) if static_alphas else None

            acc = _AlphaAccumulator(
                layer, W_q_dequant, bias,
                poly_entry=poly_entry,
                static_alpha=static_alpha,
                subsample_rows=subsample_rows,
            )
            _set_nested(block, full_path, acc)
            accumulators[poly_key] = acc

    return accumulators


def remove_alpha_accumulators(mmdit, accumulators: Dict[str, _AlphaAccumulator]):
    """Restore original linears."""
    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, _ in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            if poly_key in accumulators:
                _set_nested(block, full_path, accumulators[poly_key]._wrapped)


def collect_alpha_mse_global(
    pipeline,
    denoiser,
    prompt_entries: List[Tuple[int, str]],
    poly_schedule: dict,
    weight_results: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    num_steps: int,
    cfg_weight: float = 4.0,
    latent_size: int = 64,
    static_alphas: Optional[Dict[str, float]] = None,
    subsample_rows: int = 128,
) -> Dict[str, Tuple[float, float]]:
    """Phase B v2: install alpha accumulators on ALL blocks, run prompts once.

    Evaluates all 19 alpha_scale candidates on-the-fly per layer per step.
    No I/O caching needed — MSE is accumulated as running sums.

    Args:
        prompt_entries: List of (seed, prompt) tuples.

    Returns {poly_key: (best_alpha_scale, best_activation_mse)}.
    """
    from tqdm import tqdm

    mmdit = pipeline.mmdit
    accumulators = install_alpha_accumulators(
        mmdit, poly_schedule, weight_results,
        static_alphas=static_alphas,
        subsample_rows=subsample_rows,
    )
    n_layers = len(accumulators)

    print(f"Phase B: alpha search on all {n_layers} layers simultaneously "
          f"({len(prompt_entries)} prompts)")

    for seed, prompt in tqdm(prompt_entries, desc="Alpha search prompts"):
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
            sigma_val = float(sigmas[i])
            for acc in accumulators.values():
                acc.set_sigma(sigma_val)

            denoised = denoiser(
                x, timesteps[i], sigmas[i],
                conditioning=conditioning, cfg_weight=cfg_weight,
            )
            # No Hessian eval needed — accumulators use NumPy only
            d = (x - denoised) / sigmas[i]
            x = x + d * (sigmas[i + 1] - sigmas[i])
            mx.eval(x)

        _reset_modulation_cache(pipeline)

    # Extract results
    results = {}
    for poly_key, acc in accumulators.items():
        results[poly_key] = acc.get_best_alpha()

    remove_alpha_accumulators(mmdit, accumulators)
    return results
