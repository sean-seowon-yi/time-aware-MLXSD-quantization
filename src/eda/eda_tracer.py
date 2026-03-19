"""
EDA activation tracer for Phase 1 analysis.

Captures 6 activation families per (layer_id, timestep):

  pre_attn      — modulated input before q/k/v projection
                  shape (B, S, D); from pre_sdpa → intermediates["modulated_pre_attention"]
  q_proj        — query projection output, per stream
                  shape (B, S, 1, D) as returned by pre_sdpa (reshape to (B,S,D) for stats)
  k_proj        — key projection output, per stream
  v_proj        — value projection output, per stream
  post_gelu     — FFN activation after fc1 + GELU, before fc2
                  shape (B, S, 1, D_ffn)
  sdpa_out      — raw SDPA output before o_proj, per stream
                  shape (B, S, 1, D); from post_sdpa argument
  post_sdpa_res — post-o_proj + residual stream, before norm2 + FFN
                  shape (B, S, 1, D); computed in patched post_sdpa (adds minor o_proj
                  overhead — acceptable for EDA use)

Layer IDs follow the format:  mm_{idx:02d}_img  /  mm_{idx:02d}_txt
All 24 blocks × both streams are tagged, including mm_23_txt (skip_post_sdpa),
which contributes pre_attn/q/k/v but NOT post_gelu, sdpa_out, or post_sdpa_res.

Note: unlike the existing activation_tracer.py, this tracer is EDA-only and
must not be installed simultaneously with that tracer.
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

# 20 channel IDs for per-channel histograms — fixed seed for reproducibility.
# All other per-channel stats (mean/std/min/max/p25/p75) are still recorded for
# all D=1536 channels; only the histogram is restricted to these 20 channels.
CHANNEL_HISTOGRAM_IDS: Tuple[int, ...] = tuple(
    sorted(np.random.default_rng(42).choice(1536, size=20, replace=False).tolist())
)

FAMILIES = (
    "pre_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "post_gelu",
    "sdpa_out",
    "post_sdpa_res",
)


# ---------------------------------------------------------------------------
# Per-(family, layer, timestep) statistics accumulator
# ---------------------------------------------------------------------------

@dataclass
class PerFamilyLayerTimeStats:
    count: int = 0
    sum: Optional[np.ndarray] = None
    sq_sum: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    histogram: Optional[np.ndarray] = None
    channel_histograms: Dict[int, np.ndarray] = field(default_factory=dict)
    call_means: list = field(default_factory=list)  # per-call spatial mean, shape (D,) each

    def update(
        self,
        n_positions: int,
        sum_vec: np.ndarray,
        sq_sum_vec: np.ndarray,
        min_vec: np.ndarray,
        max_vec: np.ndarray,
        hist_counts: np.ndarray,
        ch_hist_dict: Dict[int, np.ndarray],
    ) -> None:
        if self.sum is None:
            self.sum = sum_vec.astype(np.float64)
            self.sq_sum = sq_sum_vec.astype(np.float64)
            self.min = min_vec.astype(np.float64)
            self.max = max_vec.astype(np.float64)
            self.histogram = hist_counts.astype(np.int64)
            self.count = int(n_positions)
        else:
            self.count += int(n_positions)
            self.sum += sum_vec
            self.sq_sum += sq_sum_vec
            self.min = np.minimum(self.min, min_vec)
            self.max = np.maximum(self.max, max_vec)
            self.histogram += hist_counts
        # Accumulate per-channel histograms for tracked channels
        for c, ch_hist in ch_hist_dict.items():
            if c in self.channel_histograms:
                self.channel_histograms[c] += ch_hist
            else:
                self.channel_histograms[c] = ch_hist.astype(np.int64)
        # Accumulate per-call spatial mean for cross-prompt p25/p75
        self.call_means.append((sum_vec / max(n_positions, 1)).astype(np.float32))

    def finalize(self) -> Dict[str, np.ndarray]:
        assert self.sum is not None
        denom = max(self.count, 1)
        mean = self.sum / denom
        var = np.maximum(self.sq_sum / denom - mean ** 2, 0.0)
        # Cross-prompt p25/p75: percentiles of per-call spatial means across N_calls
        if len(self.call_means) >= 2:
            stacked = np.stack(self.call_means, axis=0)  # (N_calls, D)
            p25 = np.percentile(stacked, 25, axis=0).astype(np.float32)
            p75 = np.percentile(stacked, 75, axis=0).astype(np.float32)
        else:
            p25 = mean.astype(np.float32)
            p75 = mean.astype(np.float32)
        result: Dict[str, np.ndarray] = {
            "count": np.array(self.count, dtype=np.int64),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "histogram": self.histogram.astype(np.int64),
            "p25": p25,
            "p75": p75,
        }
        # Per-channel histograms — keyed as ch_hist_{c}
        for c, ch_hist in self.channel_histograms.items():
            result[f"ch_hist_{c}"] = ch_hist.astype(np.int64)
        return result


# ---------------------------------------------------------------------------
# EDATracer
# ---------------------------------------------------------------------------

@dataclass
class EDATracer:
    """
    Accumulates per-(family, layer_id, timestep) statistics.

    stats[family][layer_id][timestep_key] -> PerFamilyLayerTimeStats
    """
    stats: Dict[str, Dict[str, Dict[str, PerFamilyLayerTimeStats]]] = field(
        default_factory=lambda: {f: {} for f in FAMILIES}
    )
    _context: Optional[Tuple[str, str]] = field(default=None, repr=False)

    def set_context(self, layer_id: str, t_key: str) -> None:
        self._context = (layer_id, t_key)

    def clear_context(self) -> None:
        self._context = None

    def has_context(self) -> bool:
        return self._context is not None

    def get_context(self) -> Optional[Tuple[str, str]]:
        return self._context

    def record(self, family: str, layer_id: str, t_key: str, act: mx.array) -> None:
        """
        Accumulate statistics for one tensor.

        Reduces over all axes except the last (channel) dimension.
        act can be (B, S, D) or (B, S, 1, D) — channel is always the last axis.
        """
        axes = tuple(range(act.ndim - 1))
        n_positions = int(np.prod([act.shape[a] for a in axes]))

        act_f32 = act.astype(mx.float32)

        sum_vec = np.asarray(mx.sum(act_f32, axis=axes))
        sq_sum_vec = np.asarray(mx.sum(act_f32 * act_f32, axis=axes))
        min_vec = np.asarray(mx.min(act_f32, axis=axes))
        max_vec = np.asarray(mx.max(act_f32, axis=axes))

        # Materialize once; reuse for both global and per-channel histograms
        D = act_f32.shape[-1]
        flat = np.asarray(act_f32.reshape(-1))
        hist_counts, _ = np.histogram(flat, bins=HISTOGRAM_NUM_BINS, range=HISTOGRAM_RANGE)

        # Per-channel histograms for tracked channels (reshape flat → (N, D))
        act_2d = flat.reshape(-1, D)
        ch_hist_dict: Dict[int, np.ndarray] = {}
        for c in CHANNEL_HISTOGRAM_IDS:
            if c < D:
                ch_hist, _ = np.histogram(
                    act_2d[:, c], bins=HISTOGRAM_NUM_BINS, range=HISTOGRAM_RANGE
                )
                ch_hist_dict[c] = ch_hist

        family_stats = self.stats[family]
        layer_stats = family_stats.setdefault(layer_id, {})
        entry = layer_stats.get(t_key)
        if entry is None:
            entry = PerFamilyLayerTimeStats()
            layer_stats[t_key] = entry

        entry.update(n_positions, sum_vec, sq_sum_vec, min_vec, max_vec, hist_counts, ch_hist_dict)

    def summarize(self) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
        """Return nested dict: family → layer_id → t_key → {mean, std, min, max, histogram}."""
        out: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}
        for family, layer_map in self.stats.items():
            family_out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
            for layer_id, t_map in layer_map.items():
                layer_out: Dict[str, Dict[str, np.ndarray]] = {}
                for t_key in sorted(t_map):
                    layer_out[t_key] = t_map[t_key].finalize()
                family_out[layer_id] = layer_out
            out[family] = family_out
        return out


# ---------------------------------------------------------------------------
# Monkey-patching machinery
# ---------------------------------------------------------------------------

_ORIG_PRE_SDPA = None
_ORIG_POST_SDPA = None
_ORIG_FFN_CALL = None
_ACTIVE_TRACER: Optional[EDATracer] = None


def _timestep_to_key(timestep: mx.array) -> str:
    if hasattr(timestep, "size") and timestep.size > 1:
        val = timestep[0].item()
    else:
        val = timestep.item()
    return f"{val:.6f}"


def install_eda_tracing(
    mmdit: MMDiT,
    profiled_layer_ids: Optional[set] = None,
) -> EDATracer:
    """
    Instrument a loaded MMDiT instance for EDA profiling.

    Tags TransformerBlocks with _trace_id. If profiled_layer_ids is provided,
    only blocks whose layer_id is in the set receive a _trace_id (subsampling).
    Patches TransformerBlock.pre_sdpa, post_sdpa, and FFN.__call__ at class level.

    The caller must call remove_eda_tracing() after profiling is complete.
    """
    global _ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL, _ACTIVE_TRACER
    if any(x is not None for x in (_ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL)):
        raise RuntimeError("EDA tracing is already installed. Call remove_eda_tracing() first.")

    tracer = EDATracer()

    # Tag TransformerBlocks with stable IDs; filter by profiled_layer_ids if given
    if hasattr(mmdit, "multimodal_transformer_blocks"):
        for idx, block in enumerate(mmdit.multimodal_transformer_blocks):
            img_id = f"mm_{idx:02d}_img"
            txt_id = f"mm_{idx:02d}_txt"
            if profiled_layer_ids is None or img_id in profiled_layer_ids:
                block.image_transformer_block._trace_id = img_id
            if profiled_layer_ids is None or txt_id in profiled_layer_ids:
                block.text_transformer_block._trace_id = txt_id

    if hasattr(mmdit, "unified_transformer_blocks"):
        for idx, block in enumerate(mmdit.unified_transformer_blocks):
            block.transformer_block._trace_id = f"uni_{idx:02d}"

    # ------------------------------------------------------------------
    # Patched pre_sdpa: capture pre_attn, q_proj, k_proj, v_proj
    # ------------------------------------------------------------------
    def pre_sdpa_patched(
        self_block: TransformerBlock,
        tensor: mx.array,
        timestep: mx.array,
    ):
        t_key = _timestep_to_key(timestep)
        intermediates = _ORIG_PRE_SDPA(self_block, tensor, timestep)

        tr = _ACTIVE_TRACER
        layer_id = getattr(self_block, "_trace_id", None)

        if tr is not None and layer_id is not None:
            tr.record("pre_attn", layer_id, t_key,
                      intermediates["modulated_pre_attention"])
            tr.record("q_proj", layer_id, t_key, intermediates["q"])
            tr.record("k_proj", layer_id, t_key, intermediates["k"])
            tr.record("v_proj", layer_id, t_key, intermediates["v"])

        # Thread timestep key for post_sdpa
        if isinstance(intermediates, dict):
            intermediates["_timestep_key"] = t_key
        return intermediates

    # ------------------------------------------------------------------
    # Patched post_sdpa: capture sdpa_out, post_sdpa_res; set FFN context
    # ------------------------------------------------------------------
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
        tr = _ACTIVE_TRACER
        layer_id = getattr(self_block, "_trace_id", None)
        t_key = kwargs.pop("_timestep_key", None)

        if tr is not None and layer_id is not None and t_key is not None:
            # Raw SDPA output before o_proj
            tr.record("sdpa_out", layer_id, t_key, sdpa_output)

            # Post-o_proj + residual (pre-FFN residual stream).
            # Minor overhead: o_proj runs twice (here + inside _ORIG_POST_SDPA).
            # Acceptable for EDA — not used in inference.
            if (
                not getattr(self_block, "skip_post_sdpa", False)
                and not getattr(self_block, "parallel_mlp", False)
                and post_attn_scale is not None
            ):
                attn_out = self_block.attn.o_proj(sdpa_output)
                post_res = residual + attn_out * post_attn_scale
                tr.record("post_sdpa_res", layer_id, t_key, post_res)

            # Set context so the FFN patch can attribute post_gelu to this block/timestep
            tr.set_context(layer_id, t_key)
            try:
                result = _ORIG_POST_SDPA(
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
                tr.clear_context()
            return result

        return _ORIG_POST_SDPA(
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

    # ------------------------------------------------------------------
    # Patched FFN.__call__: capture post_gelu
    # ------------------------------------------------------------------
    def ffn_call_patched(self_ffn: FFN, x: mx.array) -> mx.array:
        hidden = self_ffn.act_fn(self_ffn.fc1(x))

        tr = _ACTIVE_TRACER
        if tr is not None and tr.has_context():
            layer_id, t_key = tr.get_context()
            tr.record("post_gelu", layer_id, t_key, hidden)

        return self_ffn.fc2(hidden)

    _ORIG_PRE_SDPA = TransformerBlock.pre_sdpa
    _ORIG_POST_SDPA = TransformerBlock.post_sdpa
    _ORIG_FFN_CALL = FFN.__call__

    TransformerBlock.pre_sdpa = pre_sdpa_patched   # type: ignore[assignment]
    TransformerBlock.post_sdpa = post_sdpa_patched  # type: ignore[assignment]
    FFN.__call__ = ffn_call_patched                 # type: ignore[assignment]

    _ACTIVE_TRACER = tracer
    return tracer


def remove_eda_tracing() -> None:
    """Restore original methods and clear global tracing state."""
    global _ORIG_PRE_SDPA, _ORIG_POST_SDPA, _ORIG_FFN_CALL, _ACTIVE_TRACER
    try:
        from diffusionkit.mlx.mmdit import TransformerBlock, FFN  # type: ignore
        if _ORIG_PRE_SDPA is not None:
            TransformerBlock.pre_sdpa = _ORIG_PRE_SDPA  # type: ignore[assignment]
        if _ORIG_POST_SDPA is not None:
            TransformerBlock.post_sdpa = _ORIG_POST_SDPA  # type: ignore[assignment]
        if _ORIG_FFN_CALL is not None:
            FFN.__call__ = _ORIG_FFN_CALL  # type: ignore[assignment]
    finally:
        _ORIG_PRE_SDPA = None
        _ORIG_POST_SDPA = None
        _ORIG_FFN_CALL = None
        _ACTIVE_TRACER = None


def save_tracer_stats(tracer: EDATracer, path: str, unique_ts: List[float]) -> None:
    """Flatten EDATracer summary to npz.

    Key format: {family}::{layer_id}::t={t_key}::{stat_name}
    """
    summary = tracer.summarize()
    flat: Dict[str, np.ndarray] = {}

    for family, layer_map in summary.items():
        for layer_id, t_map in layer_map.items():
            for t_key, stats in t_map.items():
                prefix = f"{family}::{layer_id}::t={t_key}"
                for stat_name, arr in stats.items():
                    flat[f"{prefix}::{stat_name}"] = arr

    flat["families"] = np.array(list(summary.keys()), dtype=object)
    flat["layer_ids"] = np.array(
        sorted({lid for fam in summary.values() for lid in fam}), dtype=object
    )
    flat["timesteps_unique"] = np.array(unique_ts, dtype=np.float32)
    flat["histogram_bin_edges"] = np.linspace(
        HISTOGRAM_RANGE[0], HISTOGRAM_RANGE[1], HISTOGRAM_NUM_BINS + 1, dtype=np.float32
    )

    np.savez_compressed(path, **flat)


def load_tracer_stats(
    path: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Load activation stats from npz into nested dict.
    Returns: {family: {layer_id: {t_key: {stat_name: array}}}}
    """
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}

    for key in data.files:
        if key in ("families", "layer_ids", "timesteps_unique", "histogram_bin_edges"):
            continue
        parts = key.split("::")
        if len(parts) != 4:
            continue
        family, layer_id, t_part, stat_name = parts
        t_key = t_part[2:]  # strip "t="
        out.setdefault(family, {}).setdefault(layer_id, {}).setdefault(t_key, {})[
            stat_name
        ] = data[key]

    return out


__all__ = [
    "EDATracer",
    "FAMILIES",
    "HISTOGRAM_NUM_BINS",
    "HISTOGRAM_RANGE",
    "CHANNEL_HISTOGRAM_IDS",
    "install_eda_tracing",
    "remove_eda_tracing",
    "save_tracer_stats",
    "load_tracer_stats",
]
