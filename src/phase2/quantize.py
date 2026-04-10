"""Shared quantization utilities: model-tree navigation and pipeline patching."""

from __future__ import annotations

import logging

from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)


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
