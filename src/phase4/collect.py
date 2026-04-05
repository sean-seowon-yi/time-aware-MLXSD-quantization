"""Phase 4 data collection: capture linear-layer input tensors.

Installs lightweight wrappers on every target nn.Linear, runs N calibration
prompts through the CSB-balanced FP16 denoiser, and saves per-layer input
tensors as NPZ files under ``output_dir/``.

Storage layout
--------------
``output_dir/<layer_name>.npz``  →  array ``inputs`` of shape
``[n_samples, max_tokens, d_in]`` where ``n_samples = n_prompts × n_steps``
and ``max_tokens`` is the token-subsampling cap from config.

b_inv is NOT applied during collection.  Layers that need it (o_proj, fc2)
will have b_inv applied during the optimisation step.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input-capture wrapper
# ---------------------------------------------------------------------------

class _InputCapture:
    """Wraps an nn.Linear to record input tensors (without b_inv scaling)."""

    def __init__(self, linear: Any, max_tokens: int = 64):
        self._linear = linear
        self._max_tokens = max_tokens
        self._inputs: list[np.ndarray] = []

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            t = min(x.shape[1], self._max_tokens)
            self._inputs.append(np.array(x[:, :t, :], dtype=np.float16))
        else:
            t = min(x.shape[0], self._max_tokens)
            self._inputs.append(np.array(x[:t, :], dtype=np.float16))
        return self._linear(x)

    # Proxy attributes so the rest of the model sees a normal linear
    @property
    def weight(self):
        return self._linear.weight

    @property
    def bias(self):
        return getattr(self._linear, "bias", None)

    def parameters(self):
        return self._linear.parameters()

    def trainable_parameters(self):
        return self._linear.trainable_parameters()


# ---------------------------------------------------------------------------
# Navigate to parent (mirrors phase2/quantize.py)
# ---------------------------------------------------------------------------

def _navigate_to_parent(mmdit, name: str):
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
# Hook install / remove
# ---------------------------------------------------------------------------

def install_capture_hooks(
    mmdit,
    registry: list[dict],
    max_tokens: int,
) -> dict[str, _InputCapture]:
    """Wrap every target nn.Linear with an _InputCapture."""
    hooks: dict[str, _InputCapture] = {}
    for entry in registry:
        name = entry["name"]
        module = entry.get("module")
        if module is None:
            continue
        parent, attr = _navigate_to_parent(mmdit, name)
        capture = _InputCapture(getattr(parent, attr), max_tokens=max_tokens)
        setattr(parent, attr, capture)
        hooks[name] = capture
    logger.info("Installed capture hooks on %d layers", len(hooks))
    return hooks


def remove_capture_hooks(
    mmdit,
    registry: list[dict],
    hooks: dict[str, _InputCapture],
) -> None:
    """Restore original nn.Linear modules."""
    for entry in registry:
        name = entry["name"]
        if name not in hooks:
            continue
        parent, attr = _navigate_to_parent(mmdit, name)
        setattr(parent, attr, hooks[name]._linear)
    logger.info("Removed %d capture hooks", len(hooks))


# ---------------------------------------------------------------------------
# Main collection routine
# ---------------------------------------------------------------------------

def collect_layer_inputs(
    pipeline,
    registry: list[dict],
    pairs: list[tuple[int, str]],
    output_dir: Path,
    config: dict,
) -> None:
    """Run calibration prompts through the denoiser and save per-layer inputs.

    Args:
        pipeline: Loaded DiffusionPipeline with CSB-balanced FP16 weights.
        registry: Phase 1/2 layer registry (with ``module`` references).
        pairs: List of (seed, prompt) calibration pairs.
        output_dir: Directory to write ``<layer_name>.npz`` files.
        config: PHASE4_CONFIG dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    max_tokens = config["max_tokens_per_sample"]
    n_steps = config["n_steps"]
    cfg_weight = config["cfg_weight"]

    hooks = install_capture_hooks(pipeline.mmdit, registry, max_tokens)

    t0 = time.time()
    for idx, (seed, prompt) in enumerate(pairs):
        logger.info("[%d/%d] seed=%d  prompt=%r", idx + 1, len(pairs), seed, prompt[:60])
        mx.random.seed(seed)
        pipeline.generate_image(
            prompt,
            num_steps=n_steps,
            cfg_weight=cfg_weight,
            seed=seed,
        )

    elapsed = time.time() - t0
    logger.info(
        "Collection finished in %.1f s (%d prompts × %d steps = %d hook calls each)",
        elapsed, len(pairs), n_steps, len(pairs) * n_steps,
    )

    remove_capture_hooks(pipeline.mmdit, registry, hooks)

    # Save per-layer
    n_saved = 0
    for name, capture in hooks.items():
        if not capture._inputs:
            continue
        arr = np.concatenate(capture._inputs, axis=0)   # [n_calls, tokens, d_in]
        out_path = output_dir / f"{name}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), inputs=arr)
        n_saved += 1

    logger.info("Saved input tensors for %d layers to %s", n_saved, output_dir)
