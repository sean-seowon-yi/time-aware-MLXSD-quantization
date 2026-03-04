"""
Modular inference pipeline base classes.

InferenceTransform
------------------
Abstract base class for a single PTQ strategy applied at inference.
Subclass it and override whichever of the four hooks your strategy needs.
Unoverridden hooks are pass-throughs, so transforms compose safely.

Injection points (in call order per denoising step):
  1. apply_weight_modifications   — once at pipeline build time
  2. wrap_cache_modulation_params — wraps MMDiT.cache_modulation_params
  3. wrap_pre_sdpa                — wraps TransformerBlock.pre_sdpa (class-level)
  4. wrap_post_sdpa               — wraps TransformerBlock.post_sdpa (class-level)

QuantizedInferencePipeline
--------------------------
Composes an ordered list of InferenceTransforms into a single pipeline.
Transforms are applied in list order for weight modifications and chained
(earlier = outer wrapper) for method patches.

Example — full HTG + future AdaRound:

    pipeline = DiffusionPipeline(...)
    qpipe = QuantizedInferencePipeline(pipeline, transforms=[
        HTGTransform("htg_output/htg_corrections.npz"),
        AdaRoundTransform("adaround_output/weights.npz"),   # not yet implemented
    ])
    img = qpipe.generate("a red fox", num_steps=50)
    qpipe.remove()

Ablation — HTG weight rescaling only (no adaLN corrections):

    qpipe = QuantizedInferencePipeline(pipeline, transforms=[
        HTGTransform(
            "htg_output/htg_corrections.npz",
            apply_qkv_correction=False,
            apply_fc1_correction=False,
            apply_oproj_correction=False,
        ),
    ])
"""

from __future__ import annotations

from abc import ABC
from typing import Callable, List

from PIL import Image


class InferenceTransform(ABC):
    """
    Base class for a single inference-time PTQ correction strategy.

    All hooks have safe defaults (pass-through), so you only override
    the ones your strategy needs.
    """

    def apply_weight_modifications(self, mmdit) -> None:
        """
        Modify linear layer weights in-place (rescaling, rounding, etc.).
        Called once during QuantizedInferencePipeline._install().
        """

    def wrap_cache_modulation_params(self, mmdit, fn: Callable) -> Callable:
        """
        Return a new callable wrapping `fn` (MMDiT.cache_modulation_params).
        The wrapper is called with (pooled_text_embeddings, timesteps).
        Chain-of-responsibility: each transform wraps the previous result.
        Default: pass-through.
        """
        return fn

    def wrap_pre_sdpa(self, fn: Callable) -> Callable:
        """
        Return a new function to replace TransformerBlock.pre_sdpa.
        `fn` is the current (possibly already-wrapped) pre_sdpa.
        The returned function must accept (self_block, tensor, timestep)
        and return the intermediates dict.
        Default: pass-through.
        """
        return fn

    def wrap_post_sdpa(self, fn: Callable) -> Callable:
        """
        Return a new function to replace TransformerBlock.post_sdpa.
        `fn` is the current (possibly already-wrapped) post_sdpa.
        The returned function must accept the same signature as the
        original TransformerBlock.post_sdpa (including **kwargs).
        Default: pass-through.
        """
        return fn

    def remove(self, mmdit) -> None:
        """
        Undo any side-effects on mmdit (clear cached attrs, etc.).
        Called by QuantizedInferencePipeline.remove() in reverse order.
        """


class QuantizedInferencePipeline:
    """
    Composes a list of InferenceTransforms into a runnable inference pipeline.

    Parameters
    ----------
    pipeline : DiffusionPipeline
        A loaded DiffusionKit DiffusionPipeline instance.
    transforms : list[InferenceTransform]
        PTQ strategies to apply, in order. Pass [] for an unmodified baseline.
    """

    def __init__(self, pipeline, transforms: List[InferenceTransform]):
        self.pipeline = pipeline
        self.transforms = transforms
        self._install()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _install(self) -> None:
        from diffusionkit.mlx.mmdit import TransformerBlock  # type: ignore

        mmdit = self.pipeline.mmdit
        self._mmdit = mmdit  # keep reference; low_memory_mode drops pipeline.mmdit after generate

        # 1. Weight modifications — applied in transform list order
        for t in self.transforms:
            t.apply_weight_modifications(mmdit)

        # 2. cache_modulation_params — chain wrapping (t[0] is outermost)
        self._orig_cache_fn = mmdit.cache_modulation_params
        fn = self._orig_cache_fn
        for t in self.transforms:
            fn = t.wrap_cache_modulation_params(mmdit, fn)
        mmdit.cache_modulation_params = fn

        # 3. Class-level pre/post_sdpa patches — chain wrapping
        self._orig_pre_sdpa = TransformerBlock.pre_sdpa
        self._orig_post_sdpa = TransformerBlock.post_sdpa

        pre = self._orig_pre_sdpa
        post = self._orig_post_sdpa
        for t in self.transforms:
            pre = t.wrap_pre_sdpa(pre)
            post = t.wrap_post_sdpa(post)

        TransformerBlock.pre_sdpa = pre    # type: ignore[assignment]
        TransformerBlock.post_sdpa = post  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        num_steps: int = 50,
        cfg_scale: float = 7.0,
        latent_size: tuple = (64, 64),
        seed: int = 42,
    ) -> Image.Image:
        """Run HTG-corrected (or baseline) image generation."""
        img, _ = self.pipeline.generate_image(
            prompt,
            num_steps=num_steps,
            cfg_weight=cfg_scale,
            latent_size=latent_size,
            seed=seed,
        )
        return img

    def remove(self) -> None:
        """
        Restore DiffusionKit to its original (unpatched) state.
        Call this when done to avoid leaving class-level patches active.
        """
        from diffusionkit.mlx.mmdit import TransformerBlock  # type: ignore

        # Restore class-level patches in reverse order
        for t in reversed(self.transforms):
            t.remove(self._mmdit)

        TransformerBlock.pre_sdpa = self._orig_pre_sdpa    # type: ignore[assignment]
        TransformerBlock.post_sdpa = self._orig_post_sdpa  # type: ignore[assignment]
        self._mmdit.cache_modulation_params = self._orig_cache_fn
