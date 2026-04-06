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

def _to_f16_safe(arr: mx.array) -> np.ndarray:
    """Convert to float16, clipping values that would overflow."""
    a = np.array(arr, dtype=np.float32)
    return np.clip(a, -65504, 65504).astype(np.float16)


class _BlockCapture:
    """Wraps a transformer block, streaming I/O samples directly to disk.

    Avoids accumulating float32 tensors in RAM (which OOMs for many blocks).
    """

    def __init__(self, block: Any, output_dir: Path):
        self._block = block
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_idx: int = 0
        self._current_sigma: float = 0.0

    def __call__(self, img: mx.array, txt: mx.array, vec: mx.array,
                 pe: mx.array = None, *, positional_encodings: mx.array = None) -> tuple:
        if pe is None:
            pe = positional_encodings

        # Grab pre-computed modulation params (set by cache_modulation_params)
        itb = self._block.image_transformer_block
        ttb = self._block.text_transformer_block
        ts_key = float(vec[0]) if vec.size > 1 else float(vec)
        img_mod = _to_f16_safe(itb._modulation_params[ts_key])
        txt_mod = _to_f16_safe(ttb._modulation_params[ts_key])

        out = self._block(img, txt, vec, positional_encodings=pe)
        img_out, txt_out = out

        arrays = {
            "img_in":  np.array(img,      dtype=np.float16),
            "txt_in":  _to_f16_safe(txt),
            "vec":     np.array(vec,      dtype=np.float32),
            "pe":      np.array(pe,       dtype=np.float16),
            "img_out": np.array(img_out,  dtype=np.float16),
            "img_mod": img_mod,
            "txt_mod": txt_mod,
            "sigma":   np.float32(self._current_sigma),
        }
        if txt_out is not None:
            arrays["txt_out"] = _to_f16_safe(txt_out)

        np.savez(str(self._output_dir / f"{self._sample_idx:04d}.npz"), **arrays)
        self._sample_idx += 1
        return out

    # Proxy attributes so pipeline internals can introspect the block
    def __getattr__(self, name: str):
        return getattr(self._block, name)

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

def install_block_hooks(mmdit, output_dir: Path,
                        block_subset: set[int] | None = None) -> dict[int, _BlockCapture]:
    """Wrap transformer blocks with _BlockCapture.

    If *block_subset* is given, only those blocks are captured (saves disk).
    """
    captures: dict[int, _BlockCapture] = {}
    for i, block in enumerate(mmdit.multimodal_transformer_blocks):
        if block_subset is not None and i not in block_subset:
            continue
        cap = _BlockCapture(block, output_dir / f"block_{i:02d}")
        mmdit.multimodal_transformer_blocks[i] = cap
        captures[i] = cap
    logger.info("Installed block hooks on %d / %d blocks",
                len(captures), len(mmdit.multimodal_transformer_blocks))
    return captures


def remove_block_hooks(mmdit, captures: dict[int, _BlockCapture]) -> None:
    """Restore original transformer blocks."""
    for i, cap in captures.items():
        mmdit.multimodal_transformer_blocks[i] = cap._block
    logger.info("Removed %d block hooks", len(captures))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _count_saved_samples(captures: dict[int, _BlockCapture]) -> int:
    """Count total samples written to disk across all captures."""
    return sum(cap._sample_idx for cap in captures.values())


# ---------------------------------------------------------------------------
# Main collection routine
# ---------------------------------------------------------------------------

def _snapshot_adaln_weights(mmdit) -> list[tuple[str, mx.array]]:
    """Snapshot all adaLN modulation weights so they can be restored after
    DiffusionKit's ``cache_modulation_params`` offloads them."""
    from mlx.utils import tree_flatten
    return [
        (k, v) for k, v in tree_flatten(mmdit.parameters())
        if "adaLN" in k
    ]


def _restore_adaln_weights(mmdit, captures: dict[int, "_BlockCapture"],
                           snapshot: list[tuple[str, mx.array]]) -> None:
    """Restore adaLN weights from a snapshot.

    Temporarily swaps the real blocks back into the list so that
    ``load_weights`` can traverse the module tree normally.
    """
    for i, cap in captures.items():
        mmdit.multimodal_transformer_blocks[i] = cap._block
    mmdit.load_weights(snapshot, strict=False)
    for i, cap in captures.items():
        mmdit.multimodal_transformer_blocks[i] = cap


def collect_block_io(
    pipeline,
    pairs: list[tuple[int, str]],
    output_dir: Path,
    config: dict,
    block_subset: set[int] | None = None,
) -> None:
    """Run calibration prompts through the denoiser and save block I/O.

    Args:
        pipeline     : DiffusionPipeline with CSB-balanced FP16 weights.
        pairs        : (seed, prompt) calibration pairs.
        output_dir   : Root directory to write per-block NPZ files.
        config       : PHASE4_CONFIG dict.
        block_subset : If given, only capture these block indices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_steps   = config["n_steps"]
    cfg_weight = config["cfg_weight"]

    # Snapshot adaLN weights *before* installing hooks, since
    # cache_modulation_params offloads them and load_weights can't restore
    # through _BlockCapture wrappers.
    adaln_snapshot = _snapshot_adaln_weights(pipeline.mmdit)
    logger.info("Snapshotted %d adaLN tensors for inter-prompt restoration",
                len(adaln_snapshot))

    captures = install_block_hooks(pipeline.mmdit, output_dir, block_subset)

    from tqdm import tqdm

    t0 = time.time()
    pbar = tqdm(pairs, desc="Collecting", unit="prompt")
    for idx, (seed, prompt) in enumerate(pbar):
        pbar.set_postfix_str(f"seed={seed}  {prompt[:40]}")
        pipeline.generate_image(
            prompt,
            num_steps=n_steps,
            cfg_weight=cfg_weight,
            seed=seed,
        )
        # cache_modulation_params offloads adaLN weights during denoising.
        # clear_cache (at end of sample_euler) tries to restore them via
        # load_weights, but that fails to traverse _BlockCapture wrappers.
        # Restore from our snapshot instead.
        _restore_adaln_weights(pipeline.mmdit, captures, adaln_snapshot)

    elapsed = time.time() - t0
    total = _count_saved_samples(captures)
    logger.info(
        "Collection finished in %.1f s  (%d prompts × %d steps = %d samples/block, "
        "%d total written to %s)",
        elapsed, len(pairs), n_steps, len(pairs) * n_steps, total, output_dir,
    )

    remove_block_hooks(pipeline.mmdit, captures)
