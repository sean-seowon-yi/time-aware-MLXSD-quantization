"""Phase 4 data collection: capture block-level I/O for AdaRound reconstruction.

Installs a lightweight hook on each MMDiT transformer block, runs N calibration
prompts through the CSB-balanced FP16 denoiser, and saves per-block NPZ files.

Storage layout
--------------
``output_dir/block_{idx:02d}/{sample:04d}.npz``

Each NPZ contains:
  img_in   : (1, seq_img, hidden)  float16
  txt_in   : (1, seq_txt, hidden)  float16
  vec      : (1, hidden)           float16  — adaLN conditioning
  pe       : (...)                 float16  — rotary position embeddings
  img_out  : (1, seq_img, hidden)  float16
  txt_out  : (1, seq_txt, hidden)  float16
  sigma    : ()                    float32  — noise level for this step
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
# Block I/O capture hook
# ---------------------------------------------------------------------------

class _BlockCapture:
    """Wraps a transformer block to record (inputs, outputs) per denoising step."""

    def __init__(self, block: Any):
        self._block = block
        self.samples: list[dict] = []
        self._current_sigma: float = 0.0

    def __call__(self, img: mx.array, txt: mx.array, vec: mx.array, pe: mx.array) -> tuple:
        out = self._block(img, txt, vec, pe)
        img_out, txt_out = out
        self.samples.append({
            "img_in":  np.array(img,     dtype=np.float16),
            "txt_in":  np.array(txt,     dtype=np.float16),
            "vec":     np.array(vec,     dtype=np.float16),
            "pe":      np.array(pe,      dtype=np.float16),
            "img_out": np.array(img_out, dtype=np.float16),
            "txt_out": np.array(txt_out, dtype=np.float16),
            "sigma":   np.float32(self._current_sigma),
        })
        return out

    # Proxy attributes so pipeline internals can introspect the block
    def parameters(self):
        return self._block.parameters()

    def trainable_parameters(self):
        return self._block.trainable_parameters()


# ---------------------------------------------------------------------------
# Sigma tracking hook on the MMDiT forward
# ---------------------------------------------------------------------------

class _SigmaTracker:
    """Wraps mmdit.__call__ to read the per-step sigma and push it to captures."""

    def __init__(self, mmdit: Any, captures: list[_BlockCapture]):
        self._mmdit = mmdit
        self._captures = captures
        self._original_call = mmdit.__class__.__call__

    def install(self) -> None:
        """Patch mmdit.__call__ to extract sigma before each denoising step."""
        captures = self._captures
        original_call = self._mmdit.__call__

        def _patched(*args, **kwargs):
            # DiffusionKit passes timestep as a positional or keyword arg.
            # Sigma = timestep / 1000 (sampler convention: timestep = sigma * 1000)
            ts = kwargs.get("timestep", None)
            if ts is None and len(args) > 2:
                ts = args[2]
            if ts is not None:
                sigma = float(mx.mean(mx.array(ts)).item()) / 1000.0
                for cap in captures:
                    cap._current_sigma = sigma
            return original_call(*args, **kwargs)

        self._mmdit.__call__ = _patched
        self._patched_call = _patched

    def remove(self) -> None:
        self._mmdit.__call__ = self._mmdit.__class__.__call__.__get__(self._mmdit)


# ---------------------------------------------------------------------------
# Hook install / remove
# ---------------------------------------------------------------------------

def install_block_hooks(mmdit) -> dict[int, _BlockCapture]:
    """Wrap every multimodal_transformer_block with a _BlockCapture."""
    captures: dict[int, _BlockCapture] = {}
    for i, block in enumerate(mmdit.multimodal_transformer_blocks):
        cap = _BlockCapture(block)
        mmdit.multimodal_transformer_blocks[i] = cap
        captures[i] = cap
    logger.info("Installed block hooks on %d blocks", len(captures))
    return captures


def remove_block_hooks(mmdit, captures: dict[int, _BlockCapture]) -> None:
    """Restore original transformer blocks."""
    for i, cap in captures.items():
        mmdit.multimodal_transformer_blocks[i] = cap._block
    logger.info("Removed %d block hooks", len(captures))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_block_samples(captures: dict[int, _BlockCapture], output_dir: Path) -> None:
    """Write per-block NPZ files to disk."""
    for idx, cap in captures.items():
        block_dir = output_dir / f"block_{idx:02d}"
        block_dir.mkdir(parents=True, exist_ok=True)
        for s, sample in enumerate(cap.samples):
            np.savez_compressed(str(block_dir / f"{s:04d}.npz"), **sample)
    total = sum(len(c.samples) for c in captures.values())
    logger.info("Saved %d block samples to %s", total, output_dir)


# ---------------------------------------------------------------------------
# Main collection routine
# ---------------------------------------------------------------------------

def collect_block_io(
    pipeline,
    pairs: list[tuple[int, str]],
    output_dir: Path,
    config: dict,
) -> None:
    """Run calibration prompts through the denoiser and save block I/O.

    Args:
        pipeline  : DiffusionPipeline with CSB-balanced FP16 weights.
        pairs     : (seed, prompt) calibration pairs.
        output_dir: Root directory to write per-block NPZ files.
        config    : PHASE4_CONFIG dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_steps   = config["n_steps"]
    cfg_weight = config["cfg_weight"]

    captures = install_block_hooks(pipeline.mmdit)

    t0 = time.time()
    for idx, (seed, prompt) in enumerate(pairs):
        logger.info("[%d/%d] seed=%d  %r", idx + 1, len(pairs), seed, prompt[:60])
        pipeline.generate_image(
            prompt,
            num_steps=n_steps,
            cfg_weight=cfg_weight,
            seed=seed,
        )

    elapsed = time.time() - t0
    logger.info(
        "Collection finished in %.1f s  (%d prompts × %d steps = %d samples/block)",
        elapsed, len(pairs), n_steps, len(pairs) * n_steps,
    )

    remove_block_hooks(pipeline.mmdit, captures)
    save_block_samples(captures, output_dir)
