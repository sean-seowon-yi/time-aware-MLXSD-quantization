"""Build a registry of every nn.Linear layer in the MMDiT denoiser backbone."""

import mlx.nn as nn


def build_layer_registry(mmdit):
    """Walk the MMDiT model tree and return a list of dicts describing each
    target nn.Linear layer with its canonical name, module reference, block
    index, family, modality side, and input channel count.

    Expected total for SD3 2b: 287 linear layers.
    """
    registry = []

    for bidx, block in enumerate(mmdit.multimodal_transformer_blocks):
        skip_text_post = block.text_transformer_block.skip_post_sdpa

        for side, tb in [
            ("image", block.image_transformer_block),
            ("text", block.text_transformer_block),
        ]:
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                layer = getattr(tb.attn, proj_name)
                registry.append({
                    "name": f"blocks.{bidx}.{side}.attn.{proj_name}",
                    "module": layer,
                    "block": bidx,
                    "family": proj_name,
                    "side": side,
                    "d_in": layer.weight.shape[1],
                })

            if not (side == "text" and skip_text_post):
                o_proj = getattr(tb.attn, "o_proj")
                if not isinstance(o_proj, nn.Identity):
                    registry.append({
                        "name": f"blocks.{bidx}.{side}.attn.o_proj",
                        "module": o_proj,
                        "block": bidx,
                        "family": "o_proj",
                        "side": side,
                        "d_in": o_proj.weight.shape[1],
                    })

                for ff_name in ("fc1", "fc2"):
                    layer = getattr(tb.mlp, ff_name)
                    registry.append({
                        "name": f"blocks.{bidx}.{side}.mlp.{ff_name}",
                        "module": layer,
                        "block": bidx,
                        "family": ff_name,
                        "side": side,
                        "d_in": layer.weight.shape[1],
                    })

    registry.append({
        "name": "context_embedder",
        "module": mmdit.context_embedder,
        "block": -1,
        "family": "context_embedder",
        "side": "shared",
        "d_in": mmdit.context_embedder.weight.shape[1],
    })

    registry.append({
        "name": "final_layer.linear",
        "module": mmdit.final_layer.linear,
        "block": -1,
        "family": "final_linear",
        "side": "image",
        "d_in": mmdit.final_layer.linear.weight.shape[1],
    })

    return registry
