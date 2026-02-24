"""
Activation tracing for SD3 / MMDiT FFN blocks (Phase 2 diagnostics).

This module instruments the DiffusionKit MMDiT implementation to expose
post-activation FFN outputs (post-GELU) without modifying DiffusionKit's
source on disk. It works by:

- Tagging each TransformerBlock inside the MMDiT backbone with a stable
  layer identifier (image/text/unified + depth index).
- Monkey-patching TransformerBlock.pre_sdpa / post_sdpa to:
  - Thread the scalar timestep used for modulation through the block.
  - Maintain a lightweight "current context" (layer_id, timestep).
- Monkey-patching FFN.__call__ to:
  - Reproduce the original computation: fc1 -> GELU -> fc2.
  - Record statistics for the *post-GELU* activations from fc1.

Only calls executed while a tracing context is active are recorded, so
other uses of FFN elsewhere in DiffusionKit are left untouched apart
from a small wrapper overhead.

What we record (per layer_id, per timestep):
- count: total number of activation elements seen per channel
- sum:   sum over all positions for each hidden unit
- sq_sum: sum of squares (for variance)
- min / max: elementwise extrema per hidden unit
- histogram: binned counts across a fixed range for distribution analysis

From these, you can reconstruct:
- mean, variance, std
- channel-wise ranges
- full activation histograms (TaQ-DiT Figs 2-3 style)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import mlx.core as mx


try:
    from diffusionkit.mlx.mmdit import MMDiT, TransformerBlock, FFN  # type: ignore
except Exception:  # pragma: no cover
    MMDiT = object  # type: ignore
    TransformerBlock = object  # type: ignore
    FFN = object  # type: ignore


HISTOGRAM_NUM_BINS = 512
HISTOGRAM_RANGE = (-8.0, 8.0)


@dataclass
class PerLayerTimeStats:
    """
    Aggregate statistics for a single (layer_id, timestep) pair.

    We deliberately keep everything in NumPy here to avoid holding on to
    MLX device memory across many calibration points.
    """

    count: int = 0
    sum: Optional[np.ndarray] = None
    sq_sum: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    histogram: Optional[np.ndarray] = None

    def update(
        self,
        n_positions: int,
        sum_vec: np.ndarray,
        sq_sum_vec: np.ndarray,
        min_vec: np.ndarray,
        max_vec: np.ndarray,
        hist_counts: np.ndarray,
    ) -> None:
        if self.sum is None:
            self.sum = sum_vec.astype(np.float64)
            self.sq_sum = sq_sum_vec.astype(np.float64)
            self.min = min_vec.astype(np.float64)
            self.max = max_vec.astype(np.float64)
            self.histogram = hist_counts.astype(np.int64)
            self.count = int(n_positions)
            return

        self.count += int(n_positions)
        self.sum += sum_vec
        self.sq_sum += sq_sum_vec
        self.min = np.minimum(self.min, min_vec)
        self.max = np.maximum(self.max, max_vec)
        self.histogram += hist_counts

    def finalize(self) -> Dict[str, np.ndarray]:
        """Return mean / std / min / max / histogram as NumPy arrays."""
        assert self.sum is not None and self.sq_sum is not None
        assert self.min is not None and self.max is not None
        assert self.histogram is not None
        denom = max(self.count, 1)
        mean = self.sum / denom
        var = np.maximum(self.sq_sum / denom - mean**2, 0.0)
        std = np.sqrt(var)
        return {
            "count": np.array(self.count, dtype=np.int64),
            "mean": mean,
            "std": std,
            "min": self.min,
            "max": self.max,
            "histogram": self.histogram,
        }


@dataclass
class ActivationTracer:
    """
    Global tracing state.

    A single ActivationTracer instance is intended to be active at once.
    """

    # id(block) -> layer_id string (keyed by Python object id since nn.Module is unhashable)
    layer_ids: Dict[int, str] = field(default_factory=dict)

    # stats[layer_id][timestep_key] -> PerLayerTimeStats
    # timestep_key is a string to avoid float dict-key precision issues
    stats: Dict[str, Dict[str, PerLayerTimeStats]] = field(
        default_factory=dict
    )

    _context_stack: List[Tuple[str, str]] = field(default_factory=list)

    def push_context(self, layer_id: str, timestep_key: str) -> None:
        self._context_stack.append((layer_id, timestep_key))

    def pop_context(self) -> None:
        if self._context_stack:
            self._context_stack.pop()

    @property
    def current_context(self) -> Optional[Tuple[str, str]]:
        return self._context_stack[-1] if self._context_stack else None

    def record_post_activation(self, act: mx.array) -> None:
        """
        Record statistics for a single FFN post-activation tensor.

        act: MLX array of shape (B, S, 1, D_hidden) or similar.
        """
        ctx = self.current_context
        if ctx is None:
            return

        layer_id, timestep_key = ctx

        axes = tuple(range(act.ndim - 1))
        n_positions = int(np.prod([act.shape[a] for a in axes]))

        act_f32 = act.astype(mx.float32)

        sum_vec = mx.sum(act_f32, axis=axes)
        sq_sum_vec = mx.sum(act_f32 * act_f32, axis=axes)
        min_vec = mx.min(act_f32, axis=axes)
        max_vec = mx.max(act_f32, axis=axes)

        # Flatten to 1-D for global histogram (across all channels and positions)
        flat_np = np.asarray(act_f32.reshape(-1))
        hist_counts, _ = np.histogram(
            flat_np,
            bins=HISTOGRAM_NUM_BINS,
            range=HISTOGRAM_RANGE,
        )

        sum_np = np.asarray(sum_vec)
        sq_sum_np = np.asarray(sq_sum_vec)
        min_np = np.asarray(min_vec)
        max_np = np.asarray(max_vec)

        layer_stats = self.stats.setdefault(layer_id, {})
        per_t_stats = layer_stats.get(timestep_key)
        if per_t_stats is None:
            per_t_stats = PerLayerTimeStats()
            layer_stats[timestep_key] = per_t_stats

        per_t_stats.update(
            n_positions=n_positions,
            sum_vec=sum_np,
            sq_sum_vec=sq_sum_np,
            min_vec=min_np,
            max_vec=max_np,
            hist_counts=hist_counts,
        )

    def summarize(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Convert internal structures into a nested dict:

        {
          "layer_id": {
            "timestep_key": {
              "count": ..., "mean": ..., "std": ...,
              "min": ..., "max": ..., "histogram": ...,
            },
            ...
          },
        }
        """
        out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for layer_id, tdict in self.stats.items():
            layer_out: Dict[str, Dict[str, np.ndarray]] = {}
            for t_key, stats in sorted(tdict.items(), key=lambda kv: kv[0]):
                layer_out[t_key] = stats.finalize()
            out[layer_id] = layer_out
        return out


# --- Monkey-patching machinery ------------------------------------------------

_ORIG_PRE_SDPA = None
_ORIG_POST_SDPA = None
_ORIG_FFN_CALL = None
_ACTIVE_TRACER: Optional[ActivationTracer] = None


def _ensure_not_already_patched() -> None:
    global _ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL
    if any(x is not None for x in (_ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL)):
        raise RuntimeError("activation_tracer is already patched. Call remove_tracing() first.")


def _timestep_to_key(timestep: mx.array) -> str:
    """
    Convert an MLX timestep scalar to a stable dict key string.

    Uses the same .item() path that DiffusionKit's cache_modulation_params
    and pre_sdpa use internally, then formats to fixed precision.
    """
    if hasattr(timestep, "size") and timestep.size > 1:
        val = timestep[0].item()
    else:
        val = timestep.item()
    return f"{val:.6f}"


def install_tracing(mmdit: MMDiT) -> ActivationTracer:
    """
    Instrument a loaded MMDiT instance for activation tracing.

    This:
    - Assigns stable layer_ids to each TransformerBlock that has an FFN.
    - Patches TransformerBlock.pre_sdpa / post_sdpa and FFN.__call__.
    - Returns the ActivationTracer instance holding all collected stats.

    The caller is responsible for eventually calling remove_tracing().
    """
    global _ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL, _ACTIVE_TRACER
    _ensure_not_already_patched()

    tracer = ActivationTracer()

    if hasattr(mmdit, "multimodal_transformer_blocks"):
        for idx, block in enumerate(mmdit.multimodal_transformer_blocks):
            img_block = block.image_transformer_block
            txt_block = block.text_transformer_block

            img_id = f"mm_{idx:02d}_img"
            img_block._trace_id = img_id  # type: ignore[attr-defined]
            tracer.layer_ids[id(img_block)] = img_id

            # Only tag text blocks that actually have an FFN
            if not getattr(txt_block, "skip_post_sdpa", False):
                txt_id = f"mm_{idx:02d}_txt"
                txt_block._trace_id = txt_id  # type: ignore[attr-defined]
                tracer.layer_ids[id(txt_block)] = txt_id

    if hasattr(mmdit, "unified_transformer_blocks"):
        for idx, block in enumerate(mmdit.unified_transformer_blocks):
            uni_id = f"uni_{idx:02d}"
            block.transformer_block._trace_id = uni_id  # type: ignore[attr-defined]
            tracer.layer_ids[id(block.transformer_block)] = uni_id

    # --- Patched methods ---

    def pre_sdpa_patched(
        self_block: TransformerBlock,
        tensor: mx.array,
        timestep: mx.array,
    ):
        t_key = _timestep_to_key(timestep)
        intermediates = _ORIG_PRE_SDPA(self_block, tensor, timestep)  # type: ignore[misc]
        if isinstance(intermediates, dict):
            intermediates["_timestep_key"] = t_key
        return intermediates

    def post_sdpa_patched(
        self_block: TransformerBlock,
        residual: mx.array,
        sdpa_output: mx.array,
        modulated_pre_attention: mx.array,
        post_attn_scale: Optional[mx.array] = None,
        post_norm2_shift: Optional[mx.array] = None,
        post_norm2_residual_scale: Optional[mx.array] = None,
        post_mlp_scale: Optional[mx.array] = None,
        **kwargs,
    ):
        tracer_local = _ACTIVE_TRACER
        layer_id = getattr(self_block, "_trace_id", None)
        timestep_key = kwargs.pop("_timestep_key", None)

        if tracer_local is not None and layer_id is not None and timestep_key is not None:
            tracer_local.push_context(layer_id, timestep_key)
            try:
                out = _ORIG_POST_SDPA(  # type: ignore[misc]
                    self_block,
                    residual,
                    sdpa_output,
                    modulated_pre_attention,
                    post_attn_scale,
                    post_norm2_shift,
                    post_norm2_residual_scale,
                    post_mlp_scale,
                    **kwargs,
                )
            finally:
                tracer_local.pop_context()
            return out

        return _ORIG_POST_SDPA(  # type: ignore[misc]
            self_block,
            residual,
            sdpa_output,
            modulated_pre_attention,
            post_attn_scale,
            post_norm2_shift,
            post_norm2_residual_scale,
            post_mlp_scale,
            **kwargs,
        )

    def ffn_call_patched(self_ffn: FFN, x: mx.array) -> mx.array:
        hidden = self_ffn.act_fn(self_ffn.fc1(x))

        tracer_local = _ACTIVE_TRACER
        if tracer_local is not None and tracer_local.current_context is not None:
            tracer_local.record_post_activation(hidden)

        return self_ffn.fc2(hidden)

    _ORIG_PRE_SDPA = TransformerBlock.pre_sdpa
    _ORIG_POST_SDPA = TransformerBlock.post_sdpa
    _ORIG_FFN_CALL = FFN.__call__

    TransformerBlock.pre_sdpa = pre_sdpa_patched  # type: ignore[assignment]
    TransformerBlock.post_sdpa = post_sdpa_patched  # type: ignore[assignment]
    FFN.__call__ = ffn_call_patched  # type: ignore[assignment]

    _ACTIVE_TRACER = tracer
    return tracer


def remove_tracing() -> None:
    """Restore original methods and clear global tracing state."""
    global _ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL, _ACTIVE_TRACER
    from diffusionkit.mlx.mmdit import TransformerBlock, FFN  # type: ignore

    if _ORIG_PRE_SDPA is not None:
        TransformerBlock.pre_sdpa = _ORIG_PRE_SDPA  # type: ignore[assignment]
    if _ORIG_POST_SDPA is not None:
        TransformerBlock.post_sdpa = _ORIG_POST_SDPA  # type: ignore[assignment]
    if _ORIG_FFN_CALL is not None:
        FFN.__call__ = _ORIG_FFN_CALL  # type: ignore[assignment]

    _ORIG_PRE_SDPA = None
    _ORIG_POST_SDPA = None
    _ORIG_FFN_CALL = None
    _ACTIVE_TRACER = None


__all__ = [
    "ActivationTracer",
    "HISTOGRAM_NUM_BINS",
    "HISTOGRAM_RANGE",
    "install_tracing",
    "remove_tracing",
]
