"""W4A8 quantization: fake-quantize activations, quantized linear module,
model-wide quantization, and save/load helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .config import MODEL_VERSION, PHASE2_CONFIG, QUANTIZE_CONFIG_FILENAME, QUANTIZED_WEIGHTS_FILENAME

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation fake-quantization
# ---------------------------------------------------------------------------

def fake_quantize_a8(x: mx.array) -> mx.array:
    """Dynamic per-tensor symmetric 8-bit fake quantization.

    scale = max(|X|) / 127
    X_q   = clamp(round(X / scale), -128, 127)
    X_hat = X_q * scale

    The output is a float tensor that simulates 8-bit precision loss.
    """
    scale = mx.max(mx.abs(x)) / 127.0
    scale = mx.maximum(scale, mx.array(1e-8))
    x_q = mx.clip(mx.round(x / scale), -128, 127)
    return x_q * scale


def fake_quantize_a8_per_token(x: mx.array) -> mx.array:
    """Dynamic per-token symmetric 8-bit fake quantization.

    One scale factor per sequence position (row), computed over the channel
    dimension.  More expressive than per-tensor for layers where CSB
    incompletely equalises cross-channel dynamic range (high-rho layers).

    Input shape is typically (..., d_in).  The reduction is over the last axis.
    """
    scale = mx.max(mx.abs(x), axis=-1, keepdims=True) / 127.0
    scale = mx.maximum(scale, mx.array(1e-8))
    x_q = mx.clip(mx.round(x / scale), -128, 127)
    return x_q * scale


# ---------------------------------------------------------------------------
# W4A8 quantized linear module
# ---------------------------------------------------------------------------

class W4A8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with W4 weights and A8 activations.

    Wraps an ``nn.QuantizedLinear`` (4-bit weights, per-group affine) and
    prepends dynamic 8-bit fake-quantization of activations.  For layers that
    require online CSB balancing (o_proj, fc2), a ``b_inv`` vector is stored
    and applied before quantization.

    When ``per_token`` is True, activation quantization uses one scale per
    sequence position rather than one scale for the entire tensor.  This is
    used for high-rho layers where CSB incompletely equalises dynamic range.
    """

    def __init__(
        self,
        qlinear: nn.QuantizedLinear,
        b_inv: mx.array | None = None,
        per_token: bool = False,
    ):
        super().__init__()
        self.qlinear = qlinear
        self._per_token = per_token
        if b_inv is not None:
            self.b_inv = b_inv

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        if hasattr(self, "b_inv"):
            x = (x * self.b_inv).astype(orig_dtype)
        if self._per_token:
            x = fake_quantize_a8_per_token(x)
        else:
            x = fake_quantize_a8(x)
        return self.qlinear(x)


# ---------------------------------------------------------------------------
# Model-tree navigation
# ---------------------------------------------------------------------------

def _navigate_to_parent(mmdit, name: str):
    """Navigate the MMDiT module tree by registry name.

    Returns (parent_module, attribute_name) so that
    ``setattr(parent, attr, new_module)`` replaces the target.
    """
    if name == "context_embedder":
        return mmdit, "context_embedder"
    if name == "final_layer.linear":
        return mmdit.final_layer, "linear"

    parts = name.split(".")
    bidx = int(parts[1])
    side = parts[2]

    block = mmdit.multimodal_transformer_blocks[bidx]
    tb = (
        block.image_transformer_block
        if side == "image"
        else block.text_transformer_block
    )

    parent = tb
    for part in parts[3:-1]:
        parent = getattr(parent, part)

    return parent, parts[-1]


# ---------------------------------------------------------------------------
# Model-wide quantization
# ---------------------------------------------------------------------------

def quantize_model(
    mmdit,
    registry: list[dict],
    b_inv_map: dict[str, np.ndarray],
    config: dict | None = None,
    mean_rhos: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Replace all target ``nn.Linear`` layers with ``W4A8Linear`` modules.

    Layers whose mean Spearman rho exceeds ``per_token_rho_threshold`` use
    per-token activation quantization instead of per-tensor.

    Returns per-layer metadata dict used for save/load.
    """
    cfg = {**PHASE2_CONFIG, **(config or {})}
    exclude = set(cfg["exclude_layers"])
    group_size = cfg["group_size"]
    bits = cfg["bits"]
    final_bits = cfg["final_layer_bits"]
    rho_threshold = cfg.get("per_token_rho_threshold", 0.5)
    rhos = mean_rhos or {}

    layer_meta: dict[str, dict] = {}
    n_per_token = 0
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

        per_token = rhos.get(name, 0.0) > rho_threshold
        w4a8 = W4A8Linear(qlinear, b_inv, per_token=per_token)

        parent, attr_name = _navigate_to_parent(mmdit, name)
        setattr(parent, attr_name, w4a8)

        layer_meta[name] = {
            "d_in": int(entry["d_in"]),
            "d_out": int(linear.weight.shape[0]),
            "has_bias": getattr(linear, "bias", None) is not None,
            "bits": layer_bits,
            "has_b_inv": name in b_inv_map,
            "per_token": per_token,
        }
        if per_token:
            n_per_token += 1
        count += 1

    logger.info(
        "Quantized %d layers to W%dA8 (group_size=%d, %d per-token A8)",
        count, bits, group_size, n_per_token,
    )
    return layer_meta


# ---------------------------------------------------------------------------
# Pipeline patching (fix adaLN weight restoration for quantized models)
# ---------------------------------------------------------------------------

def patch_pipeline_for_quantized_inference(pipeline) -> None:
    """Patch DiffusionPipeline so ``clear_cache`` restores *modified* adaLN
    weights instead of the original ones from HuggingFace.

    DiffusionKit's ``cache_modulation_params`` offloads adaLN weights after
    pre-computing modulation parameters.  The subsequent ``clear_cache`` call
    restores weights via ``load_mmdit(only_modulation_dict=True)`` which loads
    the *original* (un-absorbed) weights.  This patch captures the current
    (post-CSB) adaLN weights and ensures they are restored instead.
    """
    adaln_weights = [
        (k, v) for k, v in tree_flatten(pipeline.mmdit.parameters())
        if "adaLN" in k
    ]
    pipeline._modified_adaln_weights = adaln_weights

    original_load_mmdit = pipeline.load_mmdit

    def _patched_load_mmdit(only_modulation_dict=False):
        if only_modulation_dict:
            return pipeline._modified_adaln_weights
        return original_load_mmdit(only_modulation_dict=False)

    pipeline.load_mmdit = _patched_load_mmdit
    logger.info("Patched pipeline for quantized adaLN restoration (%d tensors)", len(adaln_weights))


# ---------------------------------------------------------------------------
# Save / load quantized model
# ---------------------------------------------------------------------------

def save_quantized_model(
    mmdit,
    output_dir: Path,
    config: dict,
    layer_meta: dict[str, dict],
    b_inv_layers: list[str],
) -> None:
    """Save the quantized MMDiT weights and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        k: v for k, v in tree_flatten(mmdit.parameters())
        if not k.startswith("to_offload.")
    }
    weight_path = output_dir / QUANTIZED_WEIGHTS_FILENAME
    mx.save_safetensors(str(weight_path), weights)
    logger.info("Saved %d parameter tensors to %s", len(weights), weight_path)

    meta = {
        "model_version": config.get("model_version", MODEL_VERSION),
        "group_size": config["group_size"],
        "bits": config["bits"],
        "a_bits": config.get("a_bits", 8),
        "final_layer_bits": config["final_layer_bits"],
        "alpha": config["alpha"],
        "qkv_method": config["qkv_method"],
        "ssc_tau": config.get("ssc_tau", 1.0),
        "per_token_rho_threshold": config.get("per_token_rho_threshold", 0.5),
        "exclude_layers": config["exclude_layers"],
        "b_inv_layers": b_inv_layers,
        "quantized_layers": layer_meta,
    }
    meta_path = output_dir / QUANTIZE_CONFIG_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved quantization config to %s", meta_path)


def load_quantized_model(pipeline, output_dir: Path) -> dict:
    """Load a quantized MMDiT into an existing pipeline.

    Steps:
      1. Read metadata to learn which layers are quantized.
      2. Replace each target ``nn.Linear`` with a ``W4A8Linear`` stub.
      3. Load saved weights (overwriting stubs and adaLN).
      4. Patch the pipeline for correct adaLN restoration.

    Returns the loaded metadata dict.
    """
    meta_path = output_dir / QUANTIZE_CONFIG_FILENAME
    meta = json.loads(meta_path.read_text())

    group_size = meta["group_size"]
    b_inv_set = set(meta["b_inv_layers"])

    for name, info in meta["quantized_layers"].items():
        d_in = info["d_in"]
        d_out = info["d_out"]
        has_bias = info["has_bias"]
        layer_bits = info["bits"]
        has_b_inv = info["has_b_inv"]
        per_token = info.get("per_token", False)

        qlinear = nn.QuantizedLinear(
            d_in, d_out,
            bias=has_bias,
            group_size=group_size,
            bits=layer_bits,
        )

        b_inv = mx.zeros(d_in, dtype=mx.float32) if has_b_inv else None
        w4a8 = W4A8Linear(qlinear, b_inv, per_token=per_token)

        parent, attr_name = _navigate_to_parent(pipeline.mmdit, name)
        setattr(parent, attr_name, w4a8)

    weight_path = output_dir / QUANTIZED_WEIGHTS_FILENAME
    weights = mx.load(str(weight_path))
    filtered = [(k, v) for k, v in weights.items() if not k.startswith("to_offload.")]
    pipeline.mmdit.load_weights(filtered)
    logger.info("Loaded quantized model from %s (%d layers)", output_dir, len(meta["quantized_layers"]))

    patch_pipeline_for_quantized_inference(pipeline)

    return meta
