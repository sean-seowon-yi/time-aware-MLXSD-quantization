"""W4A8 with polynomial-clipped A8 activations.

Provides
--------
- :class:`W4A8PolyLinear` — drop-in module with σ-dependent A8 scaling.
- σ register — module-level sigma state for the denoising loop.
- :func:`install_sigma_hook` — patches ``mmdit.__call__`` to extract the
  per-step timestep and set the register automatically.
- :func:`load_quantized_model_poly` — loads a Phase 2 W4A8 checkpoint,
  reads ``poly_schedule.json``, upgrades to poly A8, and installs the hook.

σ propagation design
--------------------
Both DiffusionKit's ``sample_euler`` and Phase 1/2's custom loops call
``mmdit.cache_modulation_params(pooled, timesteps)`` **once** before the
loop with ALL timesteps.  The per-step sigma is then conveyed through
``mmdit(timestep=timesteps[i])`` at each denoising step.

The hook therefore wraps ``mmdit.__call__`` (not ``cache_modulation_params``)
to extract the per-step ``timestep`` kwarg and convert it to sigma via
``sigma = timestep / 1000`` (the sampler's ``timestep(sigma) = sigma * 1000``
convention, used identically by both DiffusionKit and Phase 1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .poly_clipping import POLY_SCHEDULE_FILENAME
from .poly_eval import poly_eval

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# σ register
# ---------------------------------------------------------------------------

_SIGMA_REGISTER: mx.array | None = None


def set_current_sigma(sigma: mx.array) -> None:
    """Set the module-level σ register (call before each denoiser step)."""
    global _SIGMA_REGISTER
    _SIGMA_REGISTER = sigma


def _get_current_sigma() -> mx.array:
    if _SIGMA_REGISTER is None:
        raise RuntimeError(
            "σ register not set — call set_current_sigma() before each "
            "denoiser step, or use install_sigma_hook()."
        )
    return _SIGMA_REGISTER


def reset_sigma_register() -> None:
    """Clear the σ register (useful after inference to avoid stale state)."""
    global _SIGMA_REGISTER
    _SIGMA_REGISTER = None


# ---------------------------------------------------------------------------
# W4A8PolyLinear
# ---------------------------------------------------------------------------

class W4A8PolyLinear(nn.Module):
    """W4 weights + polynomial-clipped A8 activations.

    At each forward pass the clipping bound ``α = poly(σ)`` is evaluated
    from the per-layer polynomial coefficients and the current noise level
    (read from the module-level σ register).  This replaces both the
    dynamic max-reduction and fixed static scale with a smooth,
    σ-conditioned clipping bound.
    """

    def __init__(
        self,
        qlinear: nn.QuantizedLinear,
        poly_coeffs: mx.array,
        b_inv: mx.array | None = None,
        shift_coeffs: mx.array | None = None,
        *,
        alpha_multiplier: float = 1.0,
    ):
        super().__init__()
        self.qlinear = qlinear
        self.poly_coeffs = poly_coeffs
        self._alpha_multiplier = mx.array(float(alpha_multiplier), dtype=mx.float32)
        if b_inv is not None:
            self.b_inv = b_inv
        if shift_coeffs is not None:
            self.shift_coeffs = shift_coeffs

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype

        if hasattr(self, "b_inv"):
            x = (x * self.b_inv).astype(orig_dtype)

        sigma = _get_current_sigma()

        alpha = poly_eval(self.poly_coeffs, sigma) * self._alpha_multiplier
        # Floor on α for numerical safety (same as get_poly_alpha in phase4_1.utils).
        alpha = mx.maximum(alpha, mx.array(0.01))
        scale = alpha / 127.0

        if hasattr(self, "shift_coeffs"):
            mu = poly_eval(self.shift_coeffs, sigma)
            x_centered = x - mu
            x_q = mx.clip(mx.round(x_centered / scale), -128, 127)
            x = x_q * scale + mu
        else:
            x_q = mx.clip(mx.round(x / scale), -128, 127)
            x = x_q * scale

        return self.qlinear(x.astype(orig_dtype))


# ---------------------------------------------------------------------------
# Model-tree navigation
# ---------------------------------------------------------------------------

def _navigate_to_parent(mmdit, name: str):
    """Navigate the MMDiT module tree to find a layer's parent and attr name.

    Mirrors ``phase2.quantize._navigate_to_parent`` for independence.
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
# σ hook installation
# ---------------------------------------------------------------------------

def install_sigma_hook(mmdit) -> None:
    """Wrap ``mmdit.__call__`` to set the σ register at each denoising step.

    Both DiffusionKit's ``sample_euler`` and Phase 1/2's custom loops call
    ``mmdit(timestep=timesteps[i])`` per step, where
    ``timestep = sampler.timestep(sigma) = sigma * 1000``.

    The hook extracts ``timestep`` from the call kwargs, converts to sigma
    via ``sigma = timestep / 1000``, and writes to the module-level σ
    register so that every :class:`W4A8PolyLinear` can read it.

    Implementation uses a dynamic subclass (same pattern as Phase 1 hooks)
    to avoid issues with Python's method resolution order on ``__call__``.
    """
    if mmdit.__class__.__name__.endswith("_SigmaHooked"):
        logger.info("σ hook already installed — skipping")
        return

    original_cls = mmdit.__class__
    original_call = original_cls.__call__

    def _hooked_call(self, *args, **kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) >= 3:
            timestep = args[2]
        if timestep is not None:
            ts_scalar = timestep.reshape(-1)[0]
            sigma = ts_scalar / 1000.0
            set_current_sigma(sigma)
        return original_call(self, *args, **kwargs)

    mmdit.__class__ = type(
        original_cls.__name__ + "_SigmaHooked",
        (original_cls,),
        {"__call__": _hooked_call},
    )
    logger.info(
        "Installed σ hook on %s.__call__ (dynamic subclass)",
        original_cls.__name__,
    )


def uninstall_sigma_hook(mmdit) -> None:
    """Restore the original mmdit class (undo :func:`install_sigma_hook`)."""
    cls_name = mmdit.__class__.__name__
    if cls_name.endswith("_SigmaHooked"):
        mmdit.__class__ = mmdit.__class__.__bases__[0]
        logger.info("Removed σ hook from mmdit")


# ---------------------------------------------------------------------------
# Load Phase 2 model + upgrade to polynomial A8
# ---------------------------------------------------------------------------

def load_quantized_model_poly(
    pipeline,
    quantized_dir: Path,
    schedule_override: dict | None = None,
) -> dict:
    """Load a Phase 2 W4A8 checkpoint and upgrade to polynomial A8.

    Steps
    -----
    1. Load Phase 2 quantized model (``W4A8StaticLinear``).
    2. Read ``poly_schedule.json`` (or use ``schedule_override`` if given).
    3. Replace each wrapped module that has a schedule entry with
       :class:`W4A8PolyLinear`.
    4. Install the σ hook on the pipeline's MMDiT.

    Returns the loaded metadata dict (extended with poly info).
    """
    from ..phase2.quantize_static import W4A8StaticLinear, load_quantized_model_static

    meta = load_quantized_model_static(pipeline, quantized_dir)
    _accepted_types = (W4A8StaticLinear,)

    if schedule_override is not None:
        schedule = schedule_override
    else:
        schedule_path = quantized_dir / POLY_SCHEDULE_FILENAME
        if not schedule_path.exists():
            raise FileNotFoundError(
                f"Missing {schedule_path} — run generate_schedule.py first"
            )
        with open(schedule_path) as f:
            schedule = json.load(f)

    n_replaced = 0
    for name, entry in schedule["layers"].items():
        try:
            parent, attr_name = _navigate_to_parent(pipeline.mmdit, name)
        except (AttributeError, IndexError):
            logger.warning("Cannot navigate to %s — skipping", name)
            continue

        module = getattr(parent, attr_name)
        if not isinstance(module, _accepted_types):
            logger.warning(
                "%s is %s, not a W4A8 wrapper — skipping poly upgrade",
                name, type(module).__name__,
            )
            continue

        coeffs = mx.array(entry["coeffs"], dtype=mx.float32)
        b_inv = module.b_inv if hasattr(module, "b_inv") else None
        shift_coeffs = None
        if "shift_coeffs" in entry:
            shift_coeffs = mx.array(entry["shift_coeffs"], dtype=mx.float32)

        alpha_multiplier = float(entry.get("alpha_multiplier", 1.0))

        poly_module = W4A8PolyLinear(
            module.qlinear,
            coeffs,
            b_inv,
            shift_coeffs,
            alpha_multiplier=alpha_multiplier,
        )
        setattr(parent, attr_name, poly_module)
        n_replaced += 1

    install_sigma_hook(pipeline.mmdit)

    logger.info(
        "Upgraded %d / %d schedule layers to W4A8PolyLinear (version: %s)",
        n_replaced, len(schedule["layers"]), schedule.get("version"),
    )

    meta["act_quant"] = "poly"
    meta["poly_schedule_version"] = schedule.get("version")
    return meta
