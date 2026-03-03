"""
Input activation tracing for SD3 / MMDiT HTG quantization (Phase 3).

Records the **input** activations to the three target linear layers identified
by the HTG paper (arXiv:2503.06930):

    fc1     → input to mlp.fc1 (AdaLN output before FFN)
    qkv     → input to attn.q/k/v_proj (AdaLN output before attention)
    oproj   → input to attn.o_proj (SDPA output)

HTG only needs per-channel (min, max) — not mean/std/histogram — because the
shifting vector is z_t[i] = (max[i] + min[i]) / 2.

Monkey-patching strategy (same pattern as Phase 2 activation_tracer.py):

    pre_sdpa_patched
        → calls original, then records modulated_pre_attention as qkv input

    post_sdpa_patched
        → records sdpa_output as oproj input
        → pushes a context onto fc1_context_stack so FFN.__call__ knows
          which (layer_id, timestep_key) to tag the fc1 input with

    ffn_call_patched
        → records x (the FFN input) as fc1 input using the stacked context

Layer IDs follow the Phase 2 convention with a type suffix:
    mm_{idx:02d}_img_qkv / mm_{idx:02d}_img_fc1 / mm_{idx:02d}_img_oproj
    mm_{idx:02d}_txt_qkv / mm_{idx:02d}_txt_fc1 / mm_{idx:02d}_txt_oproj
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


# ---------------------------------------------------------------------------
# Per-channel statistics (lighter than Phase 2's PerLayerTimeStats)
# ---------------------------------------------------------------------------

@dataclass
class PerChannelMinMax:
    """
    Running per-channel min/max for a single (layer_id, timestep) pair.

    HTG only needs (max + min) / 2 per channel to compute the shifting
    vector z_t, so we skip mean, variance, and histogram.
    """

    min: Optional[np.ndarray] = None   # shape (D,), float64
    max: Optional[np.ndarray] = None   # shape (D,), float64
    count: int = 0

    def update(self, min_vec: np.ndarray, max_vec: np.ndarray, n: int) -> None:
        if self.min is None:
            self.min = min_vec.astype(np.float64)
            self.max = max_vec.astype(np.float64)
        else:
            self.min = np.minimum(self.min, min_vec)
            self.max = np.maximum(self.max, max_vec)
        self.count += n

    def finalize(self) -> Dict[str, np.ndarray]:
        assert self.min is not None and self.max is not None
        return {
            "min": self.min,
            "max": self.max,
            "count": np.array(self.count, dtype=np.int64),
        }


# ---------------------------------------------------------------------------
# Global tracer state
# ---------------------------------------------------------------------------

@dataclass
class InputActivationTracer:
    """
    Collects per-channel min/max for fc1, qkv, and oproj inputs.

    stats[layer_id_with_type][timestep_key] → PerChannelMinMax
    where layer_id_with_type looks like "mm_05_img_fc1".
    """

    # id(TransformerBlock) → base layer id (e.g. "mm_05_img")
    layer_ids: Dict[int, str] = field(default_factory=dict)

    # stats[full_layer_id][t_key] → PerChannelMinMax
    stats: Dict[str, Dict[str, PerChannelMinMax]] = field(default_factory=dict)

    # Stack of (base_layer_id, timestep_key) pushed by post_sdpa, consumed by FFN
    _fc1_context_stack: List[Tuple[str, str]] = field(default_factory=list)

    def push_fc1_context(self, base_layer_id: str, timestep_key: str) -> None:
        self._fc1_context_stack.append((base_layer_id, timestep_key))

    def pop_fc1_context(self) -> None:
        if self._fc1_context_stack:
            self._fc1_context_stack.pop()

    @property
    def current_fc1_context(self) -> Optional[Tuple[str, str]]:
        return self._fc1_context_stack[-1] if self._fc1_context_stack else None

    def _record(self, full_layer_id: str, timestep_key: str, act: mx.array) -> None:
        """Record per-channel min/max from an MLX activation tensor."""
        # act shape: (B, S, 1, D) or (B, S, D) — reduce all dims except last
        act_f32 = act.astype(mx.float32)
        axes = tuple(range(act_f32.ndim - 1))
        n = int(np.prod([act_f32.shape[a] for a in axes]))

        min_vec = np.asarray(mx.min(act_f32, axis=axes))
        max_vec = np.asarray(mx.max(act_f32, axis=axes))

        layer_stats = self.stats.setdefault(full_layer_id, {})
        entry = layer_stats.get(timestep_key)
        if entry is None:
            entry = PerChannelMinMax()
            layer_stats[timestep_key] = entry
        entry.update(min_vec, max_vec, n)

    def record_qkv_input(
        self, base_layer_id: str, timestep_key: str, act: mx.array
    ) -> None:
        self._record(f"{base_layer_id}_qkv", timestep_key, act)

    def record_oproj_input(
        self, base_layer_id: str, timestep_key: str, act: mx.array
    ) -> None:
        self._record(f"{base_layer_id}_oproj", timestep_key, act)

    def record_fc1_input(self, act: mx.array) -> None:
        ctx = self.current_fc1_context
        if ctx is None:
            return
        base_layer_id, timestep_key = ctx
        self._record(f"{base_layer_id}_fc1", timestep_key, act)

    def summarize(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for full_id, tdict in self.stats.items():
            layer_out: Dict[str, Dict[str, np.ndarray]] = {}
            for t_key, stats in sorted(tdict.items()):
                layer_out[t_key] = stats.finalize()
            out[full_id] = layer_out
        return out


# ---------------------------------------------------------------------------
# Monkey-patching globals
# ---------------------------------------------------------------------------

_ORIG_PRE_SDPA = None
_ORIG_POST_SDPA = None
_ORIG_FFN_CALL = None
_ACTIVE_TRACER: Optional[InputActivationTracer] = None


def _ensure_not_already_patched() -> None:
    global _ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL
    if any(x is not None for x in (_ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL)):
        raise RuntimeError(
            "input_activation_tracer is already patched. "
            "Call remove_input_tracing() first."
        )


def _timestep_to_key(timestep: mx.array) -> str:
    """Same key format as Phase 2 activation_tracer._timestep_to_key."""
    if hasattr(timestep, "size") and timestep.size > 1:
        val = timestep[0].item()
    else:
        val = timestep.item()
    return f"{val:.6f}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_input_tracing(mmdit: MMDiT) -> InputActivationTracer:
    """
    Instrument a loaded MMDiT instance to capture input activations.

    Patches TransformerBlock.pre_sdpa, post_sdpa, and FFN.__call__ at the
    class level (one patch covers all instances). Assigns stable layer IDs
    to every TransformerBlock that has an FFN.

    Returns the InputActivationTracer; caller must call remove_input_tracing().
    """
    global _ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL, _ACTIVE_TRACER
    _ensure_not_already_patched()

    tracer = InputActivationTracer()

    # Assign stable IDs — same naming convention as Phase 2
    if hasattr(mmdit, "multimodal_transformer_blocks"):
        for idx, block in enumerate(mmdit.multimodal_transformer_blocks):
            img_block = block.image_transformer_block
            txt_block = block.text_transformer_block

            img_id = f"mm_{idx:02d}_img"
            img_block._htg_trace_id = img_id  # type: ignore[attr-defined]
            tracer.layer_ids[id(img_block)] = img_id

            if not getattr(txt_block, "skip_post_sdpa", False):
                txt_id = f"mm_{idx:02d}_txt"
                txt_block._htg_trace_id = txt_id  # type: ignore[attr-defined]
                tracer.layer_ids[id(txt_block)] = txt_id

    if hasattr(mmdit, "unified_transformer_blocks"):
        for idx, block in enumerate(mmdit.unified_transformer_blocks):
            uni_id = f"uni_{idx:02d}"
            block.transformer_block._htg_trace_id = uni_id  # type: ignore[attr-defined]
            tracer.layer_ids[id(block.transformer_block)] = uni_id

    # --- Patched methods ---------------------------------------------------

    def pre_sdpa_patched(
        self_block: TransformerBlock,
        tensor: mx.array,
        timestep: mx.array,
    ):
        t_key = _timestep_to_key(timestep)
        intermediates = _ORIG_PRE_SDPA(self_block, tensor, timestep)  # type: ignore

        tracer_local = _ACTIVE_TRACER
        base_id = getattr(self_block, "_htg_trace_id", None)

        if tracer_local is not None and base_id is not None:
            # Record the AdaLN output used as input to q/k/v_proj
            mod_pre_attn = intermediates.get("modulated_pre_attention")
            if mod_pre_attn is not None:
                tracer_local.record_qkv_input(base_id, t_key, mod_pre_attn)

        # Propagate the timestep key for post_sdpa (same technique as Phase 2)
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
        base_id = getattr(self_block, "_htg_trace_id", None)
        t_key = kwargs.pop("_timestep_key", None)

        has_mlp = (
            tracer_local is not None
            and base_id is not None
            and t_key is not None
            and not getattr(self_block, "skip_post_sdpa", False)
            and hasattr(self_block, "mlp")
        )

        if tracer_local is not None and base_id is not None and t_key is not None:
            # Record SDPA output = input to o_proj
            tracer_local.record_oproj_input(base_id, t_key, sdpa_output)

        if has_mlp:
            tracer_local.push_fc1_context(base_id, t_key)  # type: ignore[union-attr]

        try:
            out = _ORIG_POST_SDPA(  # type: ignore
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
            if has_mlp:
                tracer_local.pop_fc1_context()  # type: ignore[union-attr]

        return out

    def ffn_call_patched(self_ffn: FFN, x: mx.array) -> mx.array:
        tracer_local = _ACTIVE_TRACER
        if tracer_local is not None and tracer_local.current_fc1_context is not None:
            # x IS the fc1 input (output of the preceding AdaLN affine_transform)
            tracer_local.record_fc1_input(x)
        return self_ffn.fc2(self_ffn.act_fn(self_ffn.fc1(x)))

    # Store originals and install patches
    _ORIG_PRE_SDPA = TransformerBlock.pre_sdpa
    _ORIG_POST_SDPA = TransformerBlock.post_sdpa
    _ORIG_FFN_CALL = FFN.__call__

    TransformerBlock.pre_sdpa = pre_sdpa_patched   # type: ignore[assignment]
    TransformerBlock.post_sdpa = post_sdpa_patched  # type: ignore[assignment]
    FFN.__call__ = ffn_call_patched                 # type: ignore[assignment]

    _ACTIVE_TRACER = tracer
    return tracer


def remove_input_tracing() -> None:
    """Restore original methods and clear global tracing state."""
    global _ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL, _ACTIVE_TRACER
    try:
        from diffusionkit.mlx.mmdit import TransformerBlock, FFN  # type: ignore
    except Exception:
        pass
    else:
        if _ORIG_PRE_SDPA is not None:
            TransformerBlock.pre_sdpa = _ORIG_PRE_SDPA   # type: ignore[assignment]
        if _ORIG_POST_SDPA is not None:
            TransformerBlock.post_sdpa = _ORIG_POST_SDPA  # type: ignore[assignment]
        if _ORIG_FFN_CALL is not None:
            FFN.__call__ = _ORIG_FFN_CALL                 # type: ignore[assignment]

    _ORIG_PRE_SDPA = None
    _ORIG_POST_SDPA = None
    _ORIG_FFN_CALL = None
    _ACTIVE_TRACER = None


__all__ = [
    "InputActivationTracer",
    "PerChannelMinMax",
    "install_input_tracing",
    "remove_input_tracing",
]
