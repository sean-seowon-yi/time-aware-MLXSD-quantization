"""
src/inference — modular post-training quantization inference pipeline.

Architecture: composable InferenceTransform objects plugged into
QuantizedInferencePipeline. Each transform handles one PTQ strategy
(HTG, AdaRound, Bayesian Bits, …) and exposes four optional hooks:

    apply_weight_modifications   — called once at load time
    wrap_cache_modulation_params — wraps SD3's modulation caching
    wrap_pre_sdpa                — wraps TransformerBlock.pre_sdpa
    wrap_post_sdpa               — wraps TransformerBlock.post_sdpa

Ablation studies: pass different subsets / flag combinations to
QuantizedInferencePipeline(pipeline, transforms=[...]).

Adding a new strategy: create <strategy>_transform.py, subclass
InferenceTransform, implement the relevant hooks.
"""

from .base import InferenceTransform, QuantizedInferencePipeline
from .htg_transform import HTGTransform

__all__ = [
    "InferenceTransform",
    "QuantizedInferencePipeline",
    "HTGTransform",
]
