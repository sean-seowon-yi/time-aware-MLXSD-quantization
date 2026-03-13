"""
HTGTransform — InferenceTransform for Hierarchical Timestep Grouping (HTG).

Applies the inference-time corrections from arXiv:2503.06930 (Algorithm 2):

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

3. Weight quantization (optional):
       Ŵ → nn.QuantizedLinear  (MLX block-wise INT8/INT4, dequantize-then-multiply)

4. Activation fake-quantization (optional, paper Algorithm 2 Step 8-9):
       X̂ = fake_quantize(X̂, act_min[g], act_max[g], bits)
   Simulates INT8 activation quantization using static per-group ranges from
   htg_activation_ranges.npz. MLX has no true W+A INT8 GEMM, so this is the
   standard PTQ simulation approach (correct quantization error, float16 compute).

Quantization pipeline — independent toggles
-------------------------------------------
    quantize_weights     — Ŵ → MLX QuantizedLinear (weight-only INT8/4)
    quantize_activations — fake-quantize layer inputs (requires activation_ranges_path)
    Both enabled         — full W+A simulation (paper-aligned W8A8/W4A8)

Ablation flags
--------------
All correction components can be toggled independently:
    apply_weight_rescaling  — Ŵ = W * s
    apply_qkv_correction    — β̂₁, γ̂₁ for attention input
    apply_fc1_correction    — β̂₂, γ̂₂ for FFN input
    apply_oproj_correction  — (sdpa_output - z_g) / s before o_proj
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import InferenceTransform

_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Activation fake-quantization helpers (Algorithm 2, Steps 8-9)
# ---------------------------------------------------------------------------

@dataclass
class _HTGActQuantCtx:
    """
    Shared mutable context that tracks the current denoising timestep key.
    All FakeQuantizedLinear instances hold a reference to one instance of
    this class so they can look up the correct per-group quantizer range
    without explicit argument passing.
    """
    current_t_key: str = ""
    debug: bool = False
    # Accumulator: layer_id → list of {clip_frac, mse, sqnr}
    act_stats: Dict[str, List[Dict]] = field(default_factory=dict)


def fake_quantize(
    x: "mx.array",
    x_min: float,
    x_max: float,
    bits: int = 8,
) -> "mx.array":
    """
    Asymmetric uniform fake-quantization (Eq. 1 of arXiv:2503.06930).

    Simulates INT{bits} quantization by clipping, rounding, and dequantizing
    back to float16.  Computation remains in float16; only the quantization
    error is faithfully modelled.

        scale = (x_max - x_min) / (2^bits - 1)
        x_int = clip(round((x - x_min) / scale), 0, 2^bits - 1)
        x_hat = x_int * scale + x_min
    """
    import mlx.core as mx
    levels = float((1 << bits) - 1)
    scale = (x_max - x_min) / levels
    x_int = mx.clip(mx.round((x - x_min) / scale), 0.0, levels)
    return (x_int * scale + x_min).astype(x.dtype)


class FakeQuantizedLinear:
    """
    Drop-in wrapper around nn.Linear or nn.QuantizedLinear that fake-quantizes
    the input activation before the matrix multiply.

    Uses a shared _HTGActQuantCtx to look up the current timestep group at
    forward-pass time, then reads the corresponding (act_min, act_max) range
    from group_ranges.

    Parameters
    ----------
    linear : nn.Linear or nn.QuantizedLinear
        The wrapped layer whose __call__ is invoked after quantization.
    group_ranges : {g: (act_min, act_max)}
        Per-group static activation quantizer ranges from htg_activation_ranges.npz.
    ts_to_group : {t_key: g_idx}
        Mapping from timestep string key to group index for this layer.
    ctx : _HTGActQuantCtx
        Shared context; current_t_key is updated by wrap_pre_sdpa each step.
    bits : int
        Activation quantization bit-width (default 8).
    """

    def __init__(
        self,
        linear,
        group_ranges: Dict[int, Tuple[float, float]],
        ts_to_group: Dict[str, int],
        ctx: _HTGActQuantCtx,
        bits: int = 8,
        layer_id: str = "",
    ) -> None:
        self.linear = linear
        self.group_ranges = group_ranges
        self.ts_to_group = ts_to_group
        self.ctx = ctx
        self.bits = bits
        self.layer_id = layer_id

    def __call__(self, x: "mx.array") -> "mx.array":
        g = self.ts_to_group.get(self.ctx.current_t_key, 0)
        lo, hi = self.group_ranges.get(g, (-1.0, 1.0))
        x_q = fake_quantize(x, lo, hi, self.bits)

        if self.ctx.debug and self.layer_id:
            x_np   = np.array(x, dtype=np.float32)
            x_q_np = np.array(x_q, dtype=np.float32)
            clip_frac = float(((x_np < lo) | (x_np > hi)).mean())
            err = x_np - x_q_np
            mse = float(np.mean(err ** 2))
            sig_var = float(np.var(x_np))
            sqnr = 10.0 * np.log10(sig_var / (mse + 1e-12))
            self.ctx.act_stats.setdefault(self.layer_id, []).append(
                {"clip_frac": clip_frac, "mse": mse, "sqnr": sqnr}
            )

        return self.linear(x_q)

    # Forward attribute access to the wrapped layer (weight, scales, etc.)
    def __getattr__(self, name: str):
        return getattr(self.linear, name)


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
        quantize: bool = False,              # backward-compat alias for quantize_weights
        quantize_weights: bool = False,
        quantize_activations: bool = False,
        weight_bits: int = 8,
        activation_bits: int = 8,
        group_size: int = 64,
        activation_ranges_path: str | None = None,
        debug: bool = False,
    ) -> None:
        self.corrections_path = corrections_path
        self.apply_weight_rescaling = apply_weight_rescaling
        self.apply_qkv_correction = apply_qkv_correction
        self.apply_fc1_correction = apply_fc1_correction
        self.apply_oproj_correction = apply_oproj_correction
        self.quantize_weights = quantize_weights or quantize   # resolve alias
        self.quantize_activations = quantize_activations
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.group_size = group_size
        self.activation_ranges_path = activation_ranges_path
        self.debug = debug

        # Populated by _load_corrections()
        self._corrections: Dict[str, Dict] = {}       # layer_id → {z_g, s, ...}
        self._ts_to_group: Dict[str, Dict[str, int]] = {}  # layer_id → {t_key → g}
        self._timesteps_sorted: Optional[np.ndarray] = None
        self._blocks_info: List[Tuple] = []            # (tb, base_id) pairs

        # Activation fake-quantization state
        self._act_ranges: Dict[str, Dict[int, Tuple[float, float]]] = {}
        self._act_ctx = _HTGActQuantCtx(debug=debug)

        # Debug accumulators
        self._debug_weight_errors: Dict[str, Dict] = {}   # layer_key → {mse, sqnr, ...}
        self._debug_ts_total: int = 0
        self._debug_ts_misses: int = 0

        self._load_corrections()
        if self.quantize_activations and activation_ranges_path:
            self._load_act_ranges()

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

    def _load_act_ranges(self) -> None:
        """Parse htg_activation_ranges.npz into per-layer per-group range dicts."""
        raw = np.load(self.activation_ranges_path, allow_pickle=True)
        tmp: Dict[str, Dict[int, list]] = {}
        for key in raw.files:
            parts = key.split("::")   # "{layer_id}", "g{g}", "act_min"|"act_max"
            if len(parts) != 3:
                continue
            lid, g_tag, stat = parts
            if not g_tag.startswith("g"):
                continue
            g = int(g_tag[1:])
            tmp.setdefault(lid, {}).setdefault(g, [None, None])
            if stat == "act_min":
                tmp[lid][g][0] = float(raw[key])
            elif stat == "act_max":
                tmp[lid][g][1] = float(raw[key])
        self._act_ranges = {
            lid: {g: (v[0], v[1]) for g, v in gdict.items()}
            for lid, gdict in tmp.items()
        }
        print(f"[HTGTransform] Loaded activation ranges for {len(self._act_ranges)} layers.")

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _record_weight_quant_error(
        self, layer_id: str, lin_idx: int, w_before: np.ndarray, qlinear
    ) -> None:
        """Compute and store weight quantization MSE/SQNR for a single linear."""
        if not self.debug:
            return
        import mlx.core as mx

        w_dequant = np.array(
            mx.dequantize(
                qlinear.weight,
                qlinear.scales,
                qlinear.biases,
                qlinear.group_size,
                qlinear.bits,
            ),
            dtype=np.float32,
        )
        err = w_before - w_dequant
        mse = float(np.mean(err ** 2))
        sig_var = float(np.var(w_before))
        sqnr = 10.0 * np.log10(sig_var / (mse + 1e-12))
        key = f"{layer_id}::{lin_idx}"
        self._debug_weight_errors[key] = {
            "mse": mse,
            "sqnr": sqnr,
            "w_max": float(np.max(np.abs(w_before))),
            "w_range": float(w_before.max() - w_before.min()),
        }

    def print_debug_summary(self) -> None:
        """Print three diagnostic tables after a full inference run."""
        print("\n" + "=" * 76)
        print("HTGTransform Debug Summary")
        print("=" * 76)

        # Table 1: Weight quantization error
        print("\n--- Table 1: Weight quantization error (sorted by MSE descending) ---")
        if not self._debug_weight_errors:
            print("  (no weight quantization performed or debug disabled)")
        else:
            print(f"{'Layer key':<46} {'MSE':>10} {'SQNR(dB)':>10} {'w_max':>8} {'w_range':>8}")
            print("-" * 84)
            for key, v in sorted(
                self._debug_weight_errors.items(), key=lambda x: -x[1]["mse"]
            ):
                flag = " ← LOW SQNR" if v["sqnr"] < 25.0 else ""
                print(
                    f"{key:<46} {v['mse']:>10.6f} {v['sqnr']:>10.2f}"
                    f" {v['w_max']:>8.4f} {v['w_range']:>8.4f}{flag}"
                )

        # Table 2: Activation quantization
        print("\n--- Table 2: Activation quantization (sorted by clip_frac descending) ---")
        act_summary = {}
        for lid, records in self._act_ctx.act_stats.items():
            if not records:
                continue
            clip_fracs = [r["clip_frac"] for r in records]
            mses       = [r["mse"]       for r in records]
            sqnrs      = [r["sqnr"]      for r in records]
            act_summary[lid] = {
                "mean_clip_frac": float(np.mean(clip_fracs)),
                "mean_mse":       float(np.mean(mses)),
                "mean_sqnr":      float(np.mean(sqnrs)),
            }
        if not act_summary:
            print("  (no activation quantization performed or debug disabled)")
        else:
            print(f"{'Layer':<40} {'clip_frac':>10} {'mean_MSE':>10} {'mean_SQNR':>10}")
            print("-" * 72)
            for lid, v in sorted(
                act_summary.items(), key=lambda x: -x[1]["mean_clip_frac"]
            ):
                flag = " ← HIGH CLIP" if v["mean_clip_frac"] > 0.05 else ""
                print(
                    f"{lid:<40} {v['mean_clip_frac']:>10.4f}"
                    f" {v['mean_mse']:>10.6f} {v['mean_sqnr']:>10.2f}{flag}"
                )

        # Table 3: Timestep key misses
        print("\n--- Table 3: Timestep key misses ---")
        total  = self._debug_ts_total
        misses = self._debug_ts_misses
        rate   = 100.0 * misses / total if total > 0 else 0.0
        print(f"  total_calls={total}  missed_calls={misses}  miss_rate={rate:.2f}%")
        if misses > 0:
            print("  WARNING: timestep key misses cause activation range fallback to group 0!")
        else:
            print("  OK: no timestep key misses.")

        print("=" * 76)

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

            for lin_idx, linear in enumerate(linear_layers):
                w_np = np.array(linear.weight, dtype=np.float32)
                w_hat = (w_np * s[None, :]).astype(np.float16)
                linear.weight = mx.array(w_hat)
                mx.eval(linear.weight)

            # Pass A: weight quantization
            if self.quantize_weights and tb is not None:
                if layer_type == "fc1":
                    w_before = np.array(tb.mlp.fc1.weight, dtype=np.float32)
                    tb.mlp.fc1 = nn.QuantizedLinear.from_linear(
                        tb.mlp.fc1, bits=self.weight_bits, group_size=self.group_size
                    )
                    self._record_weight_quant_error(layer_id, 0, w_before, tb.mlp.fc1)
                elif layer_type == "qkv":
                    for q_idx, attr in enumerate(("q_proj", "k_proj", "v_proj")):
                        lin = getattr(tb.attn, attr)
                        w_before = np.array(lin.weight, dtype=np.float32)
                        q_lin = nn.QuantizedLinear.from_linear(
                            lin, bits=self.weight_bits, group_size=self.group_size
                        )
                        setattr(tb.attn, attr, q_lin)
                        self._record_weight_quant_error(layer_id, q_idx, w_before, q_lin)
                elif layer_type == "oproj":
                    w_before = np.array(tb.attn.o_proj.weight, dtype=np.float32)
                    tb.attn.o_proj = nn.QuantizedLinear.from_linear(
                        tb.attn.o_proj, bits=self.weight_bits, group_size=self.group_size
                    )
                    self._record_weight_quant_error(layer_id, 0, w_before, tb.attn.o_proj)
                elif layer_type == "fc2":
                    w_before = np.array(tb.mlp.fc2.weight, dtype=np.float32)
                    tb.mlp.fc2 = nn.QuantizedLinear.from_linear(
                        tb.mlp.fc2, bits=self.weight_bits, group_size=self.group_size
                    )
                    self._record_weight_quant_error(layer_id, 0, w_before, tb.mlp.fc2)

            # Pass B: activation fake-quantization (wraps linear after Pass A)
            if self.quantize_activations and tb is not None:
                ranges = self._act_ranges.get(layer_id)
                ts2g = self._ts_to_group.get(layer_id, {})
                if ranges:
                    if layer_type == "fc1":
                        tb.mlp.fc1 = FakeQuantizedLinear(
                            tb.mlp.fc1, ranges, ts2g, self._act_ctx, self.activation_bits,
                            layer_id=layer_id,
                        )
                    elif layer_type == "qkv":
                        for attr in ("q_proj", "k_proj", "v_proj"):
                            lin = getattr(tb.attn, attr)
                            setattr(tb.attn, attr, FakeQuantizedLinear(
                                lin, ranges, ts2g, self._act_ctx, self.activation_bits,
                                layer_id=layer_id,
                            ))
                    elif layer_type == "oproj":
                        tb.attn.o_proj = FakeQuantizedLinear(
                            tb.attn.o_proj, ranges, ts2g, self._act_ctx, self.activation_bits,
                            layer_id=layer_id,
                        )
                    elif layer_type == "fc2":
                        tb.mlp.fc2 = FakeQuantizedLinear(
                            tb.mlp.fc2, ranges, ts2g, self._act_ctx, self.activation_bits,
                            layer_id=layer_id,
                        )

        status = []
        if self.apply_weight_rescaling:
            status.append("rescaled")
        if self.quantize_weights:
            status.append(f"W{self.weight_bits}")
        if self.quantize_activations:
            status.append(f"A{self.activation_bits}")
        print(f"[HTGTransform] {len(self._corrections)} layers: {', '.join(status) or 'corrections only'}.")

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
        """Adds _htg_t_key to the intermediates dict and updates the activation context."""
        act_ctx = self._act_ctx
        ts_to_group = self._ts_to_group
        debug = self.debug
        debug_state = self  # reference for counter updates

        def pre_sdpa_with_tkey(self_block, tensor, timestep):
            intermediates = fn(self_block, tensor, timestep)
            if isinstance(intermediates, dict):
                t_val = timestep[0].item() if timestep.size > 1 else timestep.item()
                tk = _t_key(t_val)
                intermediates["_htg_t_key"] = tk
                act_ctx.current_t_key = tk   # update shared context for FakeQuantizedLinear

                if debug:
                    # Check timestep key match across all layers that have ts_to_group data
                    for lid, tsg in ts_to_group.items():
                        debug_state._debug_ts_total += 1
                        if tk not in tsg:
                            debug_state._debug_ts_misses += 1
                        break  # one representative layer is sufficient per block call
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
