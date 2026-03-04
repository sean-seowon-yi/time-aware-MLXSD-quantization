"""
HTGTransform — InferenceTransform for Hierarchical Timestep Grouping (HTG).

Applies the two inference-time corrections from arXiv:2503.06930:

1. Weight rescaling (static, once at load):
       Ŵ = W * s[None, :]          (per-channel column scaling)
   Re-derived from s in htg_corrections.npz so the session-specific
   id()-based keys in htg_mmdit_weights.npz are not needed.

2. AdaLN modulation correction (dynamic, per timestep group):
   After cache_modulation_params builds the per-block/timestep cache,
   the 6-chunk packed params are corrected in-place:
       QKV (chunks 0,1):  β̂₁ = (β₁ - z_g_qkv[g]) / s_qkv
                          γ̂₁ = (1 + γ₁) / s_qkv - 1   [DiffusionKit uses (1+γ) convention]
       fc1 (chunks 3,4):  β̂₂ = (β₂ - z_g_fc1[g]) / s_fc1
                          γ̂₂ = (1 + γ₂) / s_fc1 - 1
       oproj (no AdaLN):  sdpa_output = (sdpa_output - z_g_oproj[g]) / s_oproj  (in post_sdpa)

Ablation flags
--------------
All four correction components can be toggled independently:
    apply_weight_rescaling  — Ŵ = W * s
    apply_qkv_correction    — β̂₁, γ̂₁ for attention input
    apply_fc1_correction    — β̂₂, γ̂₂ for FFN input
    apply_oproj_correction  — (sdpa_output - z_g) / s before o_proj

Example ablations:
    # Weight rescaling only
    HTGTransform(path, apply_qkv_correction=False,
                 apply_fc1_correction=False, apply_oproj_correction=False)

    # adaLN corrections only (no weight rescaling)
    HTGTransform(path, apply_weight_rescaling=False)

    # Attention stream only
    HTGTransform(path, apply_fc1_correction=False, apply_oproj_correction=False)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import InferenceTransform

_ROOT = Path(__file__).resolve().parents[2]


def _ensure_diffusionkit() -> None:
    try:
        import diffusionkit.mlx  # type: ignore  # noqa: F401
    except ImportError:
        dk = _ROOT / "DiffusionKit" / "python" / "src"
        if dk.is_dir() and str(dk) not in sys.path:
            sys.path.insert(0, str(dk))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t_key(val: float) -> str:
    """Consistent timestep → string key (matches Phase 1–3 format)."""
    return f"{val:.6f}"


def _build_ts_to_group(
    timesteps_sorted: np.ndarray,
    group_assignments: np.ndarray,
) -> Dict[str, int]:
    """Map t_key → group index for a single layer."""
    return {
        _t_key(float(t_val)): int(group_assignments[i])
        for i, t_val in enumerate(timesteps_sorted)
    }


# ---------------------------------------------------------------------------
# HTGTransform
# ---------------------------------------------------------------------------

class HTGTransform(InferenceTransform):
    """
    HTG inference corrections loaded from htg_corrections.npz.

    Parameters
    ----------
    corrections_path : str
        Path to htg_corrections.npz produced by apply_htg.py Stage 3.
    apply_weight_rescaling : bool
        Apply Ŵ = W * s[None, :] to fc1, qkv, oproj linear layers.
    apply_qkv_correction : bool
        Apply z_g/s correction to adaLN chunks [β₁, γ₁] (QKV input).
    apply_fc1_correction : bool
        Apply z_g/s correction to adaLN chunks [β₂, γ₂] (fc1 input).
    apply_oproj_correction : bool
        Apply (sdpa_output - z_g_oproj) / s_oproj before o_proj.
    quantize : bool
        After weight rescaling, apply MLX block-wise integer quantization.
    weight_bits : int
        Quantization bit-width (4 or 8). Only used when quantize=True.
    group_size : int
        MLX quantization group size. Only used when quantize=True.
    """

    def __init__(
        self,
        corrections_path: str,
        apply_weight_rescaling: bool = True,
        apply_qkv_correction: bool = True,
        apply_fc1_correction: bool = True,
        apply_oproj_correction: bool = True,
        quantize: bool = False,
        weight_bits: int = 8,
        group_size: int = 64,
    ) -> None:
        self.corrections_path = corrections_path
        self.apply_weight_rescaling = apply_weight_rescaling
        self.apply_qkv_correction = apply_qkv_correction
        self.apply_fc1_correction = apply_fc1_correction
        self.apply_oproj_correction = apply_oproj_correction
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.group_size = group_size

        # Populated by _load_corrections()
        self._corrections: Dict[str, Dict] = {}       # layer_id → {z_g, s, ...}
        self._ts_to_group: Dict[str, Dict[str, int]] = {}  # layer_id → {t_key → g}
        self._timesteps_sorted: Optional[np.ndarray] = None
        self._blocks_info: List[Tuple] = []            # (tb, base_id) pairs

        self._load_corrections()

    # ------------------------------------------------------------------
    # Load corrections
    # ------------------------------------------------------------------

    def _load_corrections(self) -> None:
        """Parse htg_corrections.npz into per-layer dicts."""
        import mlx.core as mx

        raw = np.load(self.corrections_path, allow_pickle=True)
        ts_sorted = raw["timesteps_sorted"].astype(np.float32)
        self._timesteps_sorted = ts_sorted

        # Collect layer IDs
        layer_ids = set()
        for key in raw.files:
            if "::" in key and key not in ("timesteps_sorted", "num_groups", "ema_alpha"):
                layer_ids.add(key.split("::")[0])

        for lid in layer_ids:
            z_g_np = raw[f"{lid}::z_g"].astype(np.float16)   # (G, D)
            s_np   = raw[f"{lid}::s"].astype(np.float16)      # (D,)
            ga_np  = raw[f"{lid}::group_assignments"]          # (T,)

            self._corrections[lid] = {
                "z_g": mx.array(z_g_np),   # (G, D) float16
                "s":   mx.array(s_np),      # (D,)   float16
                "group_assignments": ga_np,
            }
            self._ts_to_group[lid] = _build_ts_to_group(ts_sorted, ga_np)

    # ------------------------------------------------------------------
    # Hook 1: weight modifications
    # ------------------------------------------------------------------

    def apply_weight_modifications(self, mmdit) -> None:
        """Re-apply Ŵ = W * s[None, :] to all HTG target linear layers."""
        if not self.apply_weight_rescaling:
            return

        import mlx.core as mx
        import mlx.nn as nn

        # Import _resolve_target_layer from Phase 3 — avoids rewriting layer navigation
        _ensure_diffusionkit()
        from src.htg_quantization.htg_reparameterize import _resolve_target_layer  # type: ignore

        for layer_id, data in self._corrections.items():
            s = np.array(data["s"], dtype=np.float32)  # (D,)

            tb, layer_type, linear_layers = _resolve_target_layer(mmdit, layer_id)
            if linear_layers is None:
                print(f"[HTGTransform] Warning: layer not found: {layer_id}")
                continue

            for linear in linear_layers:
                w_np = np.array(linear.weight, dtype=np.float32)
                w_hat = (w_np * s[None, :]).astype(np.float16)
                linear.weight = mx.array(w_hat)
                mx.eval(linear.weight)

            if self.quantize and tb is not None:
                if layer_type == "fc1":
                    tb.mlp.fc1 = nn.QuantizedLinear.from_linear(
                        tb.mlp.fc1, bits=self.weight_bits, group_size=self.group_size
                    )
                elif layer_type == "qkv":
                    tb.attn.q_proj = nn.QuantizedLinear.from_linear(
                        tb.attn.q_proj, bits=self.weight_bits, group_size=self.group_size
                    )
                    tb.attn.k_proj = nn.QuantizedLinear.from_linear(
                        tb.attn.k_proj, bits=self.weight_bits, group_size=self.group_size
                    )
                    tb.attn.v_proj = nn.QuantizedLinear.from_linear(
                        tb.attn.v_proj, bits=self.weight_bits, group_size=self.group_size
                    )
                elif layer_type == "oproj":
                    tb.attn.o_proj = nn.QuantizedLinear.from_linear(
                        tb.attn.o_proj, bits=self.weight_bits, group_size=self.group_size
                    )

        print(f"[HTGTransform] Weight rescaling applied to {len(self._corrections)} layers"
              + (" + quantized" if self.quantize else "") + ".")

    # ------------------------------------------------------------------
    # Hook 2: cache_modulation_params wrapper
    # ------------------------------------------------------------------

    def wrap_cache_modulation_params(self, mmdit, fn: Callable) -> Callable:
        """
        Returns a wrapper that:
          1. Calls fn (the original or previously-wrapped cache fn)
          2. Post-processes tb._modulation_params for qkv/fc1 corrections
          3. Stores tb._htg_oproj_z_g for oproj correction in post_sdpa
        """
        # Snapshot state needed inside the closure
        corrections      = self._corrections
        ts_to_group      = self._ts_to_group
        apply_qkv        = self.apply_qkv_correction
        apply_fc1        = self.apply_fc1_correction
        apply_oproj      = self.apply_oproj_correction

        # Pre-compute block list so we don't walk mmdit on every denoising call
        blocks_info = _collect_blocks(mmdit)
        self._blocks_info = blocks_info

        def htg_cache_modulation_params(pooled_text_embeddings, timesteps):
            fn(pooled_text_embeddings, timesteps)
            _post_process_modulation_cache(
                blocks_info, corrections, ts_to_group,
                apply_qkv, apply_fc1, apply_oproj,
            )

        return htg_cache_modulation_params

    # ------------------------------------------------------------------
    # Hook 3: pre_sdpa wrapper — thread _htg_t_key into intermediates
    # ------------------------------------------------------------------

    def wrap_pre_sdpa(self, fn: Callable) -> Callable:
        """Adds _htg_t_key to the intermediates dict for post_sdpa to consume."""
        def pre_sdpa_with_tkey(self_block, tensor, timestep):
            intermediates = fn(self_block, tensor, timestep)
            if isinstance(intermediates, dict):
                t_val = timestep[0].item() if timestep.size > 1 else timestep.item()
                intermediates["_htg_t_key"] = _t_key(t_val)
            return intermediates
        return pre_sdpa_with_tkey

    # ------------------------------------------------------------------
    # Hook 4: post_sdpa wrapper — apply oproj shift correction
    # ------------------------------------------------------------------

    def wrap_post_sdpa(self, fn: Callable) -> Callable:
        """Subtracts z_g_oproj from sdpa_output before o_proj (if enabled)."""
        apply = self.apply_oproj_correction

        def post_sdpa_with_oproj(
            self_block,
            residual,
            sdpa_output,
            modulated_pre_attention,
            post_attn_scale=None,
            post_norm2_shift=None,
            post_norm2_residual_scale=None,
            post_mlp_scale=None,
            **kwargs,
        ):
            t_key = kwargs.pop("_htg_t_key", None)

            if apply and t_key is not None:
                oproj_corrections = getattr(self_block, "_htg_oproj_z_g", None)
                if oproj_corrections is not None:
                    corr = oproj_corrections.get(t_key)
                    if corr is not None:
                        z_g, s = corr
                        sdpa_output = (sdpa_output - z_g) / s

            return fn(
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

        return post_sdpa_with_oproj

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove(self, mmdit) -> None:
        """Clear _htg_oproj_z_g attrs from all TransformerBlocks."""
        for tb, _ in self._blocks_info:
            try:
                object.__delattr__(tb, "_htg_oproj_z_g")
            except AttributeError:
                pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_blocks(mmdit) -> List[Tuple]:
    """Walk mmdit and return a flat list of (TransformerBlock, base_id) pairs."""
    blocks = []
    for idx, mmblock in enumerate(getattr(mmdit, "multimodal_transformer_blocks", [])):
        blocks.append((mmblock.image_transformer_block, f"mm_{idx:02d}_img"))
        blocks.append((mmblock.text_transformer_block,  f"mm_{idx:02d}_txt"))
    for idx, ublock in enumerate(getattr(mmdit, "unified_transformer_blocks", [])):
        blocks.append((ublock.transformer_block, f"uni_{idx:02d}"))
    return blocks


def _post_process_modulation_cache(
    blocks_info: List[Tuple],
    corrections: Dict[str, Dict],
    ts_to_group: Dict[str, Dict[str, int]],
    apply_qkv: bool,
    apply_fc1: bool,
    apply_oproj: bool,
) -> None:
    """
    For every (TransformerBlock, base_id) pair, post-process the block's
    _modulation_params dict to apply HTG adaLN corrections in-place,
    and optionally populate _htg_oproj_z_g for oproj correction.
    """
    import mlx.core as mx

    for tb, base_id in blocks_info:
        mod_params = getattr(tb, "_modulation_params", None)
        if mod_params is None:
            continue

        num_mod = getattr(tb, "num_modulation_params", None)
        if num_mod is None:
            continue

        qkv_id   = f"{base_id}_qkv"
        fc1_id   = f"{base_id}_fc1"
        oproj_id = f"{base_id}_oproj"

        qkv_data   = corrections.get(qkv_id)   if apply_qkv   else None
        fc1_data   = corrections.get(fc1_id)   if apply_fc1   else None
        oproj_data = corrections.get(oproj_id) if apply_oproj else None

        if qkv_data is not None or fc1_data is not None:
            _correct_adaLN_cache(tb, mod_params, num_mod, qkv_data, fc1_data,
                                  ts_to_group, qkv_id, fc1_id)

        if oproj_data is not None:
            _store_oproj_corrections(tb, mod_params, oproj_data,
                                     ts_to_group, oproj_id)


def _correct_adaLN_cache(
    tb,
    mod_params: dict,
    num_mod: int,
    qkv_data: Optional[Dict],
    fc1_data: Optional[Dict],
    ts_to_group: Dict[str, Dict[str, int]],
    qkv_id: str,
    fc1_id: str,
) -> None:
    """Apply in-place z_g/s corrections to tb._modulation_params."""
    import mlx.core as mx

    parallel_mlp = getattr(tb, "parallel_mlp", False)

    for t_key, packed in list(mod_params.items()):
        # packed: (..., num_mod * hidden_size) — split along last axis
        chunks = list(mx.split(packed, num_mod, axis=-1))
        modified = False

        # QKV correction → chunks 0 (β₁) and 1 (γ₁)
        if qkv_data is not None and num_mod >= 2:
            g = ts_to_group.get(qkv_id, {}).get(t_key, 0)
            z_g = qkv_data["z_g"][g]   # (D,) float16
            s   = qkv_data["s"]         # (D,) float16
            chunks[0] = (chunks[0] - z_g) / s
            chunks[1] = (1.0 + chunks[1]) / s - 1.0  # (1+γ) convention: γ̂ = (1+γ)/s - 1
            modified = True

        # fc1 correction → chunks 3 (β₂) and 4 (γ₂)
        # Only applies when num_mod == 6 and not using parallel_mlp
        if fc1_data is not None and num_mod == 6 and not parallel_mlp:
            g = ts_to_group.get(fc1_id, {}).get(t_key, 0)
            z_g = fc1_data["z_g"][g]
            s   = fc1_data["s"]
            chunks[3] = (chunks[3] - z_g) / s
            chunks[4] = (1.0 + chunks[4]) / s - 1.0  # (1+γ) convention: γ̂ = (1+γ)/s - 1
            modified = True

        if modified:
            mod_params[t_key] = mx.concatenate(chunks, axis=-1)
            mx.eval(mod_params[t_key])


def _store_oproj_corrections(
    tb,
    mod_params: dict,
    oproj_data: Dict,
    ts_to_group: Dict[str, Dict[str, int]],
    oproj_id: str,
) -> None:
    """
    Store per-timestep oproj (z_g, s) corrections on tb._htg_oproj_z_g.
    These are read by the post_sdpa wrapper to apply (sdpa_output - z_g) / s.
    """
    import mlx.core as mx

    s = oproj_data["s"]   # (D,) float16 mx.array — same for every timestep

    oproj_corrections: Dict[str, object] = {}
    for t_key in mod_params:
        g = ts_to_group.get(oproj_id, {}).get(t_key, 0)
        z_g = oproj_data["z_g"][g]   # (D,) float16 mx.array
        oproj_corrections[t_key] = (z_g, s)

    object.__setattr__(tb, "_htg_oproj_z_g", oproj_corrections)
