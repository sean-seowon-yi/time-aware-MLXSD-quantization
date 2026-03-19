"""
Phase 1 EDA: One-time weight statistics collection.

Walks all linear layers in the MMDiT backbone and computes:
  - Per-output-channel weight range (max - min across input channels)
  - Global 512-bin histogram over [-0.5, 0.5]
  - Per-layer excess kurtosis (scalar; checks Gaussian assumption for AdaRound)

Layer enumeration (consistent with htg_corrections.npz key format):
  mm_{idx:02d}_{stream}_{proj}
  where stream ∈ {img, txt}, proj ∈ {q_proj, k_proj, v_proj, o_proj, fc1, fc2, adaLN}

  Notes:
  - mm_23_txt has skip_post_sdpa=True → no fc1/fc2 (adaLN has only 2 params)
  - adaLN refers to the Linear inside adaLN_modulation (index 1 of the Sequential)

Saved npz key format: {layer_id}::{stat_name}
  {layer_id}::weight_range        (out_features,)  per-output-channel range (max - min)
  {layer_id}::weight_min          (out_features,)  per-output-channel min over input channels
  {layer_id}::weight_max          (out_features,)  per-output-channel max over input channels
  {layer_id}::weight_mean         (out_features,)  per-output-channel mean over input channels
  {layer_id}::weight_p25          (out_features,)  per-output-channel 25th percentile
  {layer_id}::weight_p75          (out_features,)  per-output-channel 75th percentile
  {layer_id}::weight_histogram    (512,)            global histogram counts
  {layer_id}::weight_histogram_edges (513,)         bin edges
  {layer_id}::kurtosis            scalar            excess kurtosis of full weight matrix
  {layer_id}::shape               (2,)              (out_features, in_features)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np


WEIGHT_HIST_BINS = 512
WEIGHT_HIST_RANGE = (-0.5, 0.5)


def _iter_linear_layers(mmdit) -> Iterator[Tuple[str, object]]:
    """Yield (layer_id, linear_module) for all relevant linear layers in MMDiT."""
    if hasattr(mmdit, "multimodal_transformer_blocks"):
        for idx, block in enumerate(mmdit.multimodal_transformer_blocks):
            for stream, tb in [
                ("img", block.image_transformer_block),
                ("txt", block.text_transformer_block),
            ]:
                base = f"mm_{idx:02d}_{stream}"
                attn = tb.attn

                yield f"{base}_q_proj", attn.q_proj
                yield f"{base}_k_proj", attn.k_proj
                yield f"{base}_v_proj", attn.v_proj
                # skip_post_sdpa replaces o_proj with nn.Identity() — no weight
                if not getattr(tb, "skip_post_sdpa", False):
                    yield f"{base}_o_proj", attn.o_proj

                # adaLN_modulation is nn.Sequential(SiLU, Linear)
                adaLN_linear = tb.adaLN_modulation.layers[1]
                yield f"{base}_adaLN", adaLN_linear

                # fc1 / fc2 only for blocks that have an FFN
                if not getattr(tb, "skip_post_sdpa", False):
                    yield f"{base}_fc1", tb.mlp.fc1
                    yield f"{base}_fc2", tb.mlp.fc2


def _compute_weight_stats(layer_id: str, linear_module) -> Dict[str, np.ndarray]:
    """Compute weight statistics for a single nn.Linear."""
    import mlx.core as mx

    w = linear_module.weight  # MLX array (out_features, in_features)
    w_np = np.asarray(w.astype(mx.float32))  # cast to float32 first

    out_features, in_features = w_np.shape

    # Per-output-channel stats (distribution over input channels)
    ch_min  = w_np.min(axis=1)   # (out_features,)
    ch_max  = w_np.max(axis=1)   # (out_features,)
    ch_mean = w_np.mean(axis=1)  # (out_features,)
    ch_p25  = np.percentile(w_np, 25, axis=1).astype(np.float32)  # (out_features,)
    ch_p75  = np.percentile(w_np, 75, axis=1).astype(np.float32)  # (out_features,)
    channel_range = ch_max - ch_min  # (out_features,)

    # Global histogram
    flat = w_np.flatten()
    hist_counts, hist_edges = np.histogram(flat, bins=WEIGHT_HIST_BINS, range=WEIGHT_HIST_RANGE)

    # Excess kurtosis: kurt = E[(x - mu)^4] / sigma^4 - 3
    mu = flat.mean()
    sigma = flat.std()
    if sigma > 1e-8:
        kurtosis = float(np.mean(((flat - mu) / sigma) ** 4) - 3.0)
    else:
        kurtosis = 0.0

    return {
        f"{layer_id}::weight_range": channel_range.astype(np.float32),
        f"{layer_id}::weight_min":   ch_min.astype(np.float32),
        f"{layer_id}::weight_max":   ch_max.astype(np.float32),
        f"{layer_id}::weight_mean":  ch_mean.astype(np.float32),
        f"{layer_id}::weight_p25":   ch_p25,
        f"{layer_id}::weight_p75":   ch_p75,
        f"{layer_id}::weight_histogram": hist_counts.astype(np.int64),
        f"{layer_id}::weight_histogram_edges": hist_edges.astype(np.float32),
        f"{layer_id}::kurtosis": np.array(kurtosis, dtype=np.float32),
        f"{layer_id}::shape": np.array([out_features, in_features], dtype=np.int32),
    }


def collect_weight_stats(mmdit) -> Dict[str, np.ndarray]:
    """
    Collect weight stats for all linear layers in mmdit.

    Returns a flat dict suitable for np.savez_compressed.
    Must be called BEFORE any cache_modulation_params call (adaLN weights may
    be offloaded after that; reload with load_weights(only_modulation_dict=True) if needed).
    """
    flat: Dict[str, np.ndarray] = {}
    layer_ids = []

    for layer_id, linear in _iter_linear_layers(mmdit):
        stats = _compute_weight_stats(layer_id, linear)
        flat.update(stats)
        layer_ids.append(layer_id)
        print(f"  [weight] {layer_id}  shape={flat[layer_id + '::shape']}  "
              f"kurt={float(flat[layer_id + '::kurtosis']):.3f}")

    flat["layer_ids"] = np.array(layer_ids, dtype=object)
    flat["histogram_bin_edges"] = np.linspace(
        WEIGHT_HIST_RANGE[0], WEIGHT_HIST_RANGE[1], WEIGHT_HIST_BINS + 1, dtype=np.float32
    )
    return flat


def load_weight_stats(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load weight stats from npz.
    Returns: {layer_id: {stat_name: array}}
    """
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key in data.files:
        if key in ("layer_ids", "histogram_bin_edges"):
            continue
        if "::" not in key:
            continue
        layer_id, stat_name = key.split("::", 1)
        out.setdefault(layer_id, {})[stat_name] = data[key]
    return out


__all__ = [
    "collect_weight_stats",
    "load_weight_stats",
    "WEIGHT_HIST_BINS",
    "WEIGHT_HIST_RANGE",
]
