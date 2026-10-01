"""Static W4A8 quantization: pre-computed activation scales from calibration.

Static A8 uses fixed scales derived from Phase 1 calibration statistics,
avoiding per-forward max-reduction at the cost of potential clipping of
unseen activations.

Scale-computation modes
-----------------------
- ``ssc_weighted``: SSC-weighted representative activation range (time-aware).
- ``global_max``:   conservative max across all calibration timesteps.

Granularities
-------------
- ``per_tensor``:  one scalar scale per layer.
- ``per_channel``: one scale per input channel per layer.

CSB divides every layer's activation by its balancing vector ``b``,
whether via adaLN absorption (q/k/v_proj, fc1) or online ``b_inv``
multiplication (o_proj, fc2).  Scale computation therefore adjusts all
Phase 1 activation trajectories by ``1/b`` to match what the quantizer
sees at runtime.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ..phase1.analyze import compute_spearman_trajectory, compute_ssc_weights
from .config import MODEL_VERSION, PHASE2_CONFIG, QUANTIZED_WEIGHTS_FILENAME
from .fp_snapshot import attach_fp_pre_rtn_meta
from .quantize import _navigate_to_parent, patch_pipeline_for_quantized_inference

logger = logging.getLogger(__name__)

STATIC_SCALES_FILENAME = "static_scales.npz"
QUANTIZE_CONFIG_FILENAME = "quantize_config.json"


# ---------------------------------------------------------------------------
# Static fake-quantization primitives
# ---------------------------------------------------------------------------

def fake_quantize_a8_static(x: mx.array, scale: mx.array) -> mx.array:
    """Static per-tensor symmetric 8-bit fake quantization with fixed scale."""
    x_q = mx.clip(mx.round(x / scale), -128, 127)
    return x_q * scale


def fake_quantize_a8_static_per_channel(x: mx.array, scales: mx.array) -> mx.array:
    """Static per-channel symmetric 8-bit fake quantization.

    ``scales`` has shape ``[d_in]`` and broadcasts over leading dimensions.
    """
    x_q = mx.clip(mx.round(x / scales), -128, 127)
    return x_q * scales


# ---------------------------------------------------------------------------
# Static W4A8 module
# ---------------------------------------------------------------------------

class W4A8StaticLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with W4 weights and static A8.

    Wraps ``nn.QuantizedLinear`` with CSB ``b_inv`` (when needed) and static
    symmetric 8-bit fake-quantization using stored scale(s).
    """

    def __init__(
        self,
        qlinear: nn.QuantizedLinear,
        b_inv: mx.array | None = None,
        scale: mx.array | None = None,
        per_channel: bool = False,
    ):
        super().__init__()
        self.qlinear = qlinear
        self._per_channel = per_channel
        if b_inv is not None:
            self.b_inv = b_inv
        if scale is not None:
            self.static_scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        if hasattr(self, "b_inv"):
            x = (x * self.b_inv).astype(orig_dtype)
        if hasattr(self, "static_scale"):
            if self._per_channel:
                x = fake_quantize_a8_static_per_channel(x, self.static_scale)
            else:
                x = fake_quantize_a8_static(x, self.static_scale)
        return self.qlinear(x)


# ---------------------------------------------------------------------------
# Scale computation from Phase 1 calibration data
# ---------------------------------------------------------------------------

def compute_static_scales(
    registry: list[dict],
    diagnostics_dir: Path,
    calibration: dict,
    config: dict | None = None,
    mode: str = "ssc_weighted",
    granularity: str = "per_tensor",
) -> dict[str, np.ndarray | float]:
    """Compute per-layer static A8 scales from Phase 1 calibration data.

    CSB divides every activation by ``b`` (either via adaLN absorption or
    online ``b_inv``).  The quantizer therefore always sees ``x / b``.
    Phase 1 statistics record raw ``x``, so we must adjust by ``1 / b``
    for *all* calibrated layers — both absorbed and online.

    Parameters
    ----------
    registry : list[dict]
        Layer registry entries (need ``name``).
    diagnostics_dir : Path
        Directory containing Phase 1 outputs.
    calibration : dict
        Output of ``calibrate_all_layers``, containing:
        - ``balancing_vectors``: dict[name → b] for all calibrated layers
        - ``b_inv_layers``: list[str] of online layers (not used directly
          here — we adjust all layers uniformly by ``1/b``).
    config : dict, optional
        Override ``PHASE2_CONFIG`` entries.
    mode : ``"ssc_weighted"`` | ``"global_max"``
        How to aggregate across timesteps.
    granularity : ``"per_tensor"`` | ``"per_channel"``
        Scale granularity.

    Returns
    -------
    dict mapping layer name to scale (float for per_tensor, ndarray for
    per_channel).
    """
    cfg = {**PHASE2_CONFIG, **(config or {})}
    exclude = set(cfg["exclude_layers"])
    ssc_tau = cfg.get("ssc_tau", 1.0)
    b_vectors = calibration.get("balancing_vectors", {})

    wt_data = None  # lazy-loaded only when mode == "ssc_weighted"

    scales: dict[str, np.ndarray | float] = {}

    for entry in registry:
        name = entry["name"]
        if name in exclude:
            continue

        act_path = diagnostics_dir / "activation_stats" / f"{name}.npz"
        if not act_path.exists():
            logger.warning("No activation stats for %s — skipping", name)
            continue

        raw_act_traj = np.load(act_path)["act_channel_max"]  # [T, d_in]

        # CSB makes the quantizer see x/b for every calibrated layer,
        # regardless of whether b was absorbed into adaLN or applied online.
        if name in b_vectors:
            b = b_vectors[name]
            adjusted_traj = raw_act_traj / b[np.newaxis, :]
        else:
            adjusted_traj = raw_act_traj

        if mode == "ssc_weighted":
            if wt_data is None:
                wt_data = np.load(diagnostics_dir / "weight_stats.npz")
            wt_key = f"{name}/w_channel_max"
            if wt_key not in wt_data:
                logger.warning("No weight stats for %s — skipping", name)
                continue
            wt_salience = wt_data[wt_key]  # [d_in]
            # Spearman ρ uses raw Phase-1 activations (same as calibrate.py SSC).
            # The weighted sum applies those timestep weights to *post-CSB* magnitudes
            # the quantizer actually sees (adjusted_traj).
            rho_traj = compute_spearman_trajectory(raw_act_traj, wt_salience)
            ssc_w = compute_ssc_weights(rho_traj, tau=ssc_tau)
            representative = ssc_w @ adjusted_traj  # [d_in]
        elif mode == "global_max":
            representative = adjusted_traj.max(axis=0)  # [d_in]
        else:
            raise ValueError(f"Unknown static-scale mode: {mode!r}")

        if granularity == "per_tensor":
            s = float(representative.max()) / 127.0
            scales[name] = max(s, 1e-8)
        elif granularity == "per_channel":
            s = representative / 127.0
            scales[name] = np.maximum(s, 1e-8).astype(np.float32)
        else:
            raise ValueError(f"Unknown granularity: {granularity!r}")

    logger.info(
        "Computed static scales for %d layers (mode=%s, granularity=%s)",
        len(scales), mode, granularity,
    )
    return scales


# ---------------------------------------------------------------------------
# Model-wide static quantization
# ---------------------------------------------------------------------------

def quantize_model_static(
    mmdit,
    registry: list[dict],
    b_inv_map: dict[str, np.ndarray],
    static_scales: dict[str, np.ndarray | float],
    config: dict | None = None,
) -> dict[str, dict]:
    """Replace target ``nn.Linear`` layers with ``W4A8StaticLinear`` modules.

    Returns per-layer metadata dict for save/load.
    """
    cfg = {**PHASE2_CONFIG, **(config or {})}
    exclude = set(cfg["exclude_layers"])
    group_size = cfg["group_size"]
    bits = cfg["bits"]
    final_bits = cfg["final_layer_bits"]

    layer_meta: dict[str, dict] = {}
    count = 0

    for entry in registry:
        name = entry["name"]
        if name in exclude:
            continue

        linear = entry["module"]
        layer_bits = final_bits if entry["family"] == "final_linear" else bits
        if layer_bits >= 16:
            continue

        qlinear = nn.QuantizedLinear.from_linear(
            linear, group_size=group_size, bits=layer_bits,
        )

        b_inv = None
        if name in b_inv_map:
            b_inv = mx.array(b_inv_map[name], dtype=mx.float32)

        raw_scale = static_scales.get(name)
        if raw_scale is None:
            logger.warning("No static scale for %s — layer will have no A8", name)
            scale = None
            per_channel = False
        elif isinstance(raw_scale, (int, float)):
            scale = mx.array(raw_scale, dtype=mx.float32)
            per_channel = False
        else:
            scale = mx.array(raw_scale, dtype=mx.float32)
            per_channel = True

        w4a8 = W4A8StaticLinear(qlinear, b_inv, scale, per_channel)

        parent, attr_name = _navigate_to_parent(mmdit, name)
        setattr(parent, attr_name, w4a8)

        layer_meta[name] = {
            "d_in": int(entry["d_in"]),
            "d_out": int(linear.weight.shape[0]),
            "has_bias": getattr(linear, "bias", None) is not None,
            "bits": layer_bits,
            "has_b_inv": name in b_inv_map,
            "per_channel": per_channel,
            "has_static_scale": scale is not None,
        }
        count += 1

    logger.info(
        "Static-quantized %d layers to W%dA8 (group_size=%d)",
        count, bits, group_size,
    )
    return layer_meta


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_quantized_model_static(
    mmdit,
    output_dir: Path,
    config: dict,
    layer_meta: dict[str, dict],
    b_inv_layers: list[str],
    static_scales: dict[str, np.ndarray | float],
    granularity: str,
    mode: str,
) -> None:
    """Save the static-quantized model, config, and scales."""
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        k: v for k, v in tree_flatten(mmdit.parameters())
        if not k.startswith("to_offload.")
    }
    weight_path = output_dir / QUANTIZED_WEIGHTS_FILENAME
    mx.save_safetensors(str(weight_path), weights)
    logger.info("Saved %d parameter tensors to %s", len(weights), weight_path)

    scale_arrays = {}
    for name, s in static_scales.items():
        if isinstance(s, (int, float)):
            scale_arrays[name] = np.array([s], dtype=np.float32)
        else:
            scale_arrays[name] = np.asarray(s, dtype=np.float32)
    np.savez(output_dir / STATIC_SCALES_FILENAME, **scale_arrays)
    logger.info("Saved static scales for %d layers", len(scale_arrays))

    meta = {
        "model_version": config.get("model_version", MODEL_VERSION),
        "group_size": config["group_size"],
        "bits": config["bits"],
        "a_bits": config.get("a_bits", 8),
        "final_layer_bits": config["final_layer_bits"],
        "alpha": config["alpha"],
        "qkv_method": config["qkv_method"],
        "ssc_tau": config.get("ssc_tau", 1.0),
        "exclude_layers": config["exclude_layers"],
        "b_inv_layers": b_inv_layers,
        "act_quant": "static",
        "static_mode": mode,
        "static_granularity": granularity,
        "quantized_layers": layer_meta,
    }
    attach_fp_pre_rtn_meta(meta, output_dir)
    meta_path = output_dir / QUANTIZE_CONFIG_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved static quantization config to %s", meta_path)


def load_quantized_model_static(pipeline, output_dir: Path) -> dict:
    """Load a static-quantized model into an existing pipeline.

    Returns the loaded metadata dict.
    """
    meta_path = output_dir / QUANTIZE_CONFIG_FILENAME
    meta = json.loads(meta_path.read_text())

    if meta.get("act_quant") != "static":
        raise ValueError(
            f"Expected act_quant='static' in {meta_path}, "
            f"got {meta.get('act_quant')!r}"
        )

    group_size = meta["group_size"]

    scales_path = output_dir / STATIC_SCALES_FILENAME
    raw_scales = np.load(scales_path)

    for name, info in meta["quantized_layers"].items():
        d_in = info["d_in"]
        d_out = info["d_out"]
        has_bias = info["has_bias"]
        layer_bits = info["bits"]
        has_b_inv = info["has_b_inv"]
        per_channel = info.get("per_channel", False)

        qlinear = nn.QuantizedLinear(
            d_in, d_out,
            bias=has_bias,
            group_size=group_size,
            bits=layer_bits,
        )

        b_inv = mx.zeros(d_in, dtype=mx.float32) if has_b_inv else None

        scale = None
        if name in raw_scales:
            s = raw_scales[name]
            if per_channel:
                scale = mx.array(s, dtype=mx.float32)
            else:
                scale = mx.array(float(s.item() if s.ndim == 0 else s[0]),
                                 dtype=mx.float32)

        w4a8 = W4A8StaticLinear(qlinear, b_inv, scale, per_channel)

        parent, attr_name = _navigate_to_parent(pipeline.mmdit, name)
        setattr(parent, attr_name, w4a8)

    weight_path = output_dir / QUANTIZED_WEIGHTS_FILENAME
    weights = mx.load(str(weight_path))
    filtered = [
        (k, v) for k, v in weights.items()
        if not k.startswith("to_offload.")
    ]
    pipeline.mmdit.load_weights(filtered)
    logger.info(
        "Loaded static-quantized model from %s (%d layers)",
        output_dir, len(meta["quantized_layers"]),
    )

    patch_pipeline_for_quantized_inference(pipeline)
    return meta
