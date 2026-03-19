"""Save and load quantized models."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import QDiffusionConfig
from .quant_linear import QuantizedLinear


def save_quantized_model(mmdit, output_dir: str, config: QDiffusionConfig):
    """Save quantized model weights, scales, and config.

    Saves:
    - quantized_weights.npz: weight + scale + v_param (hard-rounded) per layer
    - activation_params.npz: alpha + scale per layer for activation quantizers
    - config.json: QDiffusionConfig
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    weight_data = {}
    act_data = {}
    layer_count = 0

    def _collect_from_block(block, prefix: str):
        nonlocal layer_count
        for name, child in block.children().items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, QuantizedLinear):
                weight_data[f"{full_name}.weight"] = np.array(child.weight)
                weight_data[f"{full_name}.weight_scale"] = np.array(child.weight_scale)
                weight_data[f"{full_name}.v_param"] = np.array(child.v_param)
                weight_data[f"{full_name}.weight_bits"] = np.array(child.weight_bits)
                if child.bias is not None:
                    weight_data[f"{full_name}.bias"] = np.array(child.bias)

                if child.act_quantizer is not None and child.act_quantizer.enabled:
                    act_data[f"{full_name}.alpha"] = np.array(child.act_quantizer.alpha)
                    act_data[f"{full_name}.scale"] = np.array(child.act_quantizer.scale)
                    act_data[f"{full_name}.symmetric"] = np.array(child.act_quantizer.symmetric)
                    if child.act_quantizer.zero_point is not None:
                        act_data[f"{full_name}.zero_point"] = np.array(child.act_quantizer.zero_point)

                layer_count += 1
            elif hasattr(child, "children"):
                _collect_from_block(child, full_name)

    # Iterate over multimodal transformer blocks
    if hasattr(mmdit, "multimodal_transformer_blocks"):
        for idx, block in enumerate(mmdit.multimodal_transformer_blocks):
            _collect_from_block(block, f"mm_block_{idx:02d}")

    # FinalLayer
    if hasattr(mmdit, "final_layer"):
        _collect_from_block(mmdit.final_layer, "final_layer")

    np.savez_compressed(str(out / "quantized_weights.npz"), **weight_data)
    np.savez_compressed(str(out / "activation_params.npz"), **act_data)

    # Save config
    config_dict = {k: v for k, v in config.__dict__.items()}
    # Convert non-serializable types
    for k, v in config_dict.items():
        if isinstance(v, (list, tuple)):
            config_dict[k] = list(v)

    with open(out / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved {layer_count} quantized layers to {output_dir}")
    print(f"  quantized_weights.npz: {len(weight_data)} arrays")
    print(f"  activation_params.npz: {len(act_data)} arrays")
