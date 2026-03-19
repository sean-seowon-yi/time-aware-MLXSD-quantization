"""Load calibration data and collect block I/O for AdaRound optimization."""

from __future__ import annotations

import gc
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mlx.core as mx


_ROOT = Path(__file__).resolve().parents[2]


def _ensure_diffusionkit() -> None:
    try:
        import diffusionkit.mlx  # noqa: F401
        return
    except ImportError:
        pass
    dk_src = _ROOT / "DiffusionKit" / "python" / "src"
    if dk_src.is_dir() and str(dk_src) not in sys.path:
        sys.path.insert(0, str(dk_src))
    import diffusionkit.mlx  # noqa: F401


def load_calibration_data(path: str) -> dict:
    """Load calibration npz file.

    Returns dict with keys: xs, ts, prompt_indices, cs, cs_pooled, prompts, cfg_scale
    """
    data = np.load(path, allow_pickle=True)
    required = ["xs", "ts", "prompt_indices", "cs", "cs_pooled"]
    for k in required:
        if k not in data:
            raise KeyError(f"Calibration file missing key '{k}'")
    return data


def group_by_prompt(prompt_indices: np.ndarray) -> Dict[int, List[int]]:
    """Group calibration indices by prompt index."""
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(prompt_indices)):
        groups[int(prompt_indices[idx])].append(idx)
    return dict(sorted(groups.items()))


def subsample_indices(
    ts: np.ndarray,
    n_samples: int,
    total: int,
) -> np.ndarray:
    """Subsample calibration indices with uniform timestep coverage.

    Stratifies by timestep, taking equal samples from each timestep bin.
    """
    if n_samples >= total:
        return np.arange(total)

    # Group indices by timestep
    ts_groups: Dict[float, List[int]] = defaultdict(list)
    for idx in range(total):
        ts_groups[float(ts[idx])].append(idx)

    # Take equal samples per timestep
    unique_ts = sorted(ts_groups.keys())
    per_ts = max(1, n_samples // len(unique_ts))
    selected = []
    for t_val in unique_ts:
        indices = ts_groups[t_val]
        rng = np.random.RandomState(42)
        chosen = rng.choice(indices, size=min(per_ts, len(indices)), replace=False)
        selected.extend(chosen.tolist())

    # If we haven't reached n_samples, fill from remaining
    if len(selected) < n_samples:
        remaining = set(range(total)) - set(selected)
        rng = np.random.RandomState(43)
        extra = rng.choice(list(remaining), size=n_samples - len(selected), replace=False)
        selected.extend(extra.tolist())

    return np.array(sorted(selected[:n_samples]))


# ---------------------------------------------------------------------------
# Disk-backed block input cache
# ---------------------------------------------------------------------------

class BlockInputCache:
    """Disk-backed storage for block input samples.

    Stores (img, txt, timestep) tuples per block per sample to disk.
    Only one block's inputs loaded into memory at a time.
    """

    def __init__(self, cache_dir: str | Path, n_blocks: int):
        self.cache_dir = Path(cache_dir)
        self.n_blocks = n_blocks
        self._counts: Dict[int, int] = {}
        for i in range(n_blocks + 1):
            d = self.cache_dir / f"block_{i:02d}"
            d.mkdir(parents=True, exist_ok=True)
            self._counts[i] = 0

    def append(self, block_idx: int, inputs: tuple):
        """Save one sample's block inputs to disk."""
        d = self.cache_dir / f"block_{block_idx:02d}"
        s = self._counts[block_idx]
        np.save(d / f"s{s:04d}_img.npy", np.array(inputs[0], copy=False))
        np.save(d / f"s{s:04d}_txt.npy", np.array(inputs[1], copy=False))
        np.save(d / f"s{s:04d}_t.npy", np.array(inputs[2], copy=False))
        self._counts[block_idx] = s + 1

    def append_final(self, block_idx: int, inputs: tuple):
        """Save FinalLayer inputs (positional args tuple)."""
        d = self.cache_dir / f"block_{block_idx:02d}"
        s = self._counts[block_idx]
        for i, a in enumerate(inputs):
            np.save(d / f"s{s:04d}_arg{i}.npy", np.array(a, copy=False))
        self._counts[block_idx] = s + 1

    def load_block(self, block_idx: int, n_mm_blocks: int) -> list:
        """Load all inputs for one block into memory as mx.arrays."""
        n = self._counts[block_idx]
        d = self.cache_dir / f"block_{block_idx:02d}"
        results = []
        is_final = (block_idx >= n_mm_blocks)
        for s in range(n):
            if not is_final:
                img = mx.array(np.load(d / f"s{s:04d}_img.npy"))
                txt = mx.array(np.load(d / f"s{s:04d}_txt.npy"))
                t   = mx.array(np.load(d / f"s{s:04d}_t.npy"))
                results.append((img, txt, t))
            else:
                args = []
                i = 0
                while (d / f"s{s:04d}_arg{i}.npy").exists():
                    args.append(mx.array(np.load(d / f"s{s:04d}_arg{i}.npy")))
                    i += 1
                results.append(tuple(args))
        return results

    def sample_count(self, block_idx: int = 0) -> int:
        return self._counts.get(block_idx, 0)

    @classmethod
    def from_existing_dir(cls, cache_dir: str | Path, n_blocks: int) -> "BlockInputCache":
        """Reconstruct a BlockInputCache from an already-populated directory.

        Does not re-run any forward passes. Counts existing sample files per
        block to restore _counts so load_block() works correctly.
        """
        obj = object.__new__(cls)
        obj.cache_dir = Path(cache_dir)
        obj.n_blocks = n_blocks
        obj._counts = {}
        for i in range(n_blocks + 1):
            d = obj.cache_dir / f"block_{i:02d}"
            # Normal blocks use _img.npy; FinalLayer uses _arg0.npy
            img_count = len(list(d.glob("s????_img.npy"))) if d.exists() else 0
            arg_count = len(list(d.glob("s????_arg0.npy"))) if d.exists() else 0
            obj._counts[i] = img_count if img_count else arg_count
        return obj

    def is_complete(self, expected_samples: int) -> bool:
        """Return True if every block has exactly expected_samples samples."""
        for i in range(self.n_blocks + 1):
            if self._counts.get(i, 0) != expected_samples:
                return False
        return True

    def cleanup(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"Cleaned up block input cache at {self.cache_dir}")

    def disk_usage_mb(self) -> float:
        total = sum(f.stat().st_size for f in self.cache_dir.rglob("*.npy"))
        return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Disk-backed FP target cache
# ---------------------------------------------------------------------------

class FPTargetCache:
    """Disk-backed storage for FP block output targets.

    Streams block outputs to disk during collection to avoid holding
    all 25 blocks × N samples in memory simultaneously.
    Each block's targets are stored in a subdirectory as individual .npy files.
    Only one block's targets are loaded into memory at a time.
    """

    def __init__(self, cache_dir: str | Path, n_blocks: int):
        self.cache_dir = Path(cache_dir)
        self.n_blocks = n_blocks
        self._counts: Dict[int, int] = {}

        for i in range(n_blocks + 1):
            d = self.cache_dir / f"block_{i:02d}"
            d.mkdir(parents=True, exist_ok=True)
            self._counts[i] = 0

    def append(self, block_idx: int, output):
        """Save a single sample's block output to disk immediately."""
        d = self.cache_dir / f"block_{block_idx:02d}"
        s = self._counts[block_idx]

        if isinstance(output, tuple):
            # MMDiT block: (img_out, txt_out)
            np.save(d / f"s{s:04d}_img.npy", np.array(output[0], copy=False))
            if output[1] is not None:
                np.save(d / f"s{s:04d}_txt.npy", np.array(output[1], copy=False))
        else:
            # FinalLayer: single tensor
            np.save(d / f"s{s:04d}.npy", np.array(output, copy=False))

        self._counts[block_idx] = s + 1

    def load_block(self, block_idx: int) -> list:
        """Load all samples for one block into memory as mx.arrays."""
        n = self._counts[block_idx]
        d = self.cache_dir / f"block_{block_idx:02d}"
        results = []

        for s in range(n):
            img_p = d / f"s{s:04d}_img.npy"
            txt_p = d / f"s{s:04d}_txt.npy"
            plain_p = d / f"s{s:04d}.npy"

            if img_p.exists():
                img = mx.array(np.load(img_p))
                txt = mx.array(np.load(txt_p)) if txt_p.exists() else None
                results.append((img, txt))
            elif plain_p.exists():
                results.append(mx.array(np.load(plain_p)))

        return results

    def sample_count(self, block_idx: int = 0) -> int:
        return self._counts.get(block_idx, 0)

    @classmethod
    def from_existing_dir(cls, cache_dir: str | Path, n_blocks: int) -> "FPTargetCache":
        """Reconstruct an FPTargetCache from an already-populated directory.

        Does not re-run any forward passes. Counts existing sample files per
        block to restore _counts so load_block() works correctly.
        """
        obj = object.__new__(cls)
        obj.cache_dir = Path(cache_dir)
        obj.n_blocks = n_blocks
        obj._counts = {}
        for i in range(n_blocks + 1):
            d = obj.cache_dir / f"block_{i:02d}"
            # MMDiT blocks use _img.npy; FinalLayer uses plain s????.npy
            img_count = len(list(d.glob("s????_img.npy"))) if d.exists() else 0
            plain_count = len(list(d.glob("s????.npy"))) if d.exists() else 0
            obj._counts[i] = img_count if img_count else plain_count
        return obj

    def is_complete(self, expected_samples: int) -> bool:
        """Return True if every block has exactly expected_samples samples."""
        for i in range(self.n_blocks + 1):
            if self._counts.get(i, 0) != expected_samples:
                return False
        return True

    def cleanup(self):
        """Remove all cache files."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"Cleaned up FP target cache at {self.cache_dir}")

    def disk_usage_mb(self) -> float:
        """Estimate total disk usage in MB."""
        total = 0
        for f in self.cache_dir.rglob("*.npy"):
            total += f.stat().st_size
        return total / (1024 * 1024)


class BlockIOCollector:
    """Collect block inputs/outputs for AdaRound optimization.

    Two main operations:
    1. collect_all_fp_targets: Single forward pass through FP model (before quantization),
       streaming every block's output to disk as reconstruction targets.
    2. collect_block_inputs: Forward through the (progressively quantized) model up to
       block_idx, collecting that block's inputs.
    """

    def __init__(self, pipeline, cali_data: dict, n_samples: int = 256):
        self.pipeline = pipeline
        self.cali_data = cali_data
        self.n_samples = n_samples

        xs = cali_data["xs"]
        ts = cali_data["ts"]
        prompt_indices = cali_data["prompt_indices"]

        # Subsample indices
        self.sample_indices = subsample_indices(ts, n_samples, len(xs))
        self.groups = group_by_prompt(prompt_indices)

        print(f"BlockIOCollector: {len(self.sample_indices)} samples from {len(xs)} total, "
              f"{len(self.groups)} prompt groups")

    def _run_forward_collecting(
        self,
        hook_fn,
        sample_indices: np.ndarray | None = None,
    ):
        """Run forward passes over calibration data, calling hook_fn on each block.

        hook_fn receives the model's block outputs during forward. This handles
        the prompt-grouped forward loop with adaLN cache/offload.
        """
        if sample_indices is None:
            sample_indices = self.sample_indices

        xs = self.cali_data["xs"]
        ts = self.cali_data["ts"]
        prompt_indices = self.cali_data["prompt_indices"]
        cs = self.cali_data["cs"]
        cs_pooled = self.cali_data["cs_pooled"]

        sample_set = set(sample_indices.tolist())
        mmdit = self.pipeline.mmdit

        # Group calibration indices by prompt, filter to our subsample
        for prompt_idx, cal_indices in self.groups.items():
            relevant = [i for i in cal_indices if i in sample_set]
            if not relevant:
                continue

            # Cache modulation params for this prompt group
            group_ts = sorted(set(float(ts[i]) for i in cal_indices))
            pooled_mx = mx.array(cs_pooled[prompt_idx]).astype(self.pipeline.activation_dtype)
            cond_mx = mx.array(cs[prompt_idx]).astype(self.pipeline.activation_dtype)
            ts_mx = mx.array(group_ts, dtype=self.pipeline.activation_dtype)

            mmdit.cache_modulation_params(
                pooled_text_embeddings=pooled_mx,
                timesteps=ts_mx,
            )

            for cal_idx in relevant:
                t_val = float(ts[cal_idx])
                x_single = mx.array(xs[cal_idx][None, ...]).astype(self.pipeline.activation_dtype)
                x_doubled = mx.concatenate([x_single] * 2, axis=0)
                t_mx = mx.array([t_val], dtype=self.pipeline.activation_dtype)
                t_broadcast = mx.broadcast_to(t_mx, [x_doubled.shape[0]])

                hook_fn(
                    x_doubled, cond_mx, t_broadcast, cal_idx
                )

            # Reload adaLN weights
            mmdit.load_weights(
                self.pipeline.load_mmdit(only_modulation_dict=True), strict=False
            )
            gc.collect()

    def collect_all_fp_targets(self, cache_dir: str = ".fp_target_cache") -> FPTargetCache:
        """Run single forward pass through FP model, stream all block outputs to disk.

        Must be called BEFORE any quantization is applied to the model.

        Each sample's block outputs are written to disk immediately after the
        forward pass, keeping memory bounded regardless of n_samples.

        Returns:
            FPTargetCache: disk-backed cache; use cache.load_block(idx) to load
            one block's targets into memory at a time.
        """
        mmdit = self.pipeline.mmdit
        n_blocks = len(mmdit.multimodal_transformer_blocks)

        cache = FPTargetCache(cache_dir, n_blocks)

        # Monkey-patch all blocks to capture outputs
        from diffusionkit.mlx.mmdit import MultiModalTransformerBlock
        orig_mm = MultiModalTransformerBlock.__call__

        for idx, block in enumerate(mmdit.multimodal_transformer_blocks):
            block._fp_collect_idx = idx

        captured = {}  # block_idx -> output for current forward

        def patched_mm_call(self_block, *args, **kwargs):
            result = orig_mm(self_block, *args, **kwargs)
            bidx = getattr(self_block, "_fp_collect_idx", None)
            if bidx is not None:
                captured[bidx] = (result[0], result[1])
            return result

        MultiModalTransformerBlock.__call__ = patched_mm_call

        final_idx = n_blocks
        orig_final = None
        if hasattr(mmdit, "final_layer"):
            from diffusionkit.mlx.mmdit import FinalLayer
            orig_final = FinalLayer.__call__
            def patched_final_call(self_block, *args, **kwargs):
                result = orig_final(self_block, *args, **kwargs)
                captured[final_idx] = result
                return result
            FinalLayer.__call__ = patched_final_call

        sample_count = 0
        def hook_fn(x_doubled, cond_mx, t_broadcast, cal_idx):
            nonlocal sample_count, captured
            captured = {}
            out = mmdit(
                latent_image_embeddings=x_doubled,
                token_level_text_embeddings=mx.expand_dims(cond_mx, 2),
                timestep=t_broadcast,
            )
            mx.eval(out)

            # Stream each block's output to disk immediately
            for bidx, output in captured.items():
                cache.append(bidx, output)

            # Clear references so MLX can free GPU memory
            captured = {}
            del out

            sample_count += 1
            if sample_count % 25 == 0 or sample_count == len(self.sample_indices):
                pct = 100 * sample_count / len(self.sample_indices)
                print(f"  [{sample_count:4d}/{len(self.sample_indices)}  {pct:5.1f}%]  streaming to disk...")

            # Periodically clear MLX cache
            if sample_count % 100 == 0:
                gc.collect()
                try:
                    mx.metal.clear_cache()
                except Exception:
                    pass

        print("Collecting FP block outputs (streaming to disk)...")
        self._run_forward_collecting(hook_fn)

        # Restore original methods
        MultiModalTransformerBlock.__call__ = orig_mm
        if orig_final is not None:
            from diffusionkit.mlx.mmdit import FinalLayer
            FinalLayer.__call__ = orig_final

        # Cleanup tags
        for block in mmdit.multimodal_transformer_blocks:
            if hasattr(block, "_fp_collect_idx"):
                delattr(block, "_fp_collect_idx")

        disk_mb = cache.disk_usage_mb()
        print(f"  Cached FP targets for {n_blocks + 1} blocks, "
              f"{sample_count} samples each ({disk_mb:.0f} MB on disk)")
        return cache

    def collect_all_block_inputs(self, cache_dir: str) -> BlockInputCache:
        """Run single forward pass, capture ALL blocks' inputs simultaneously.

        Must be called on the model in the desired state (naive-quantized or
        post-AdaRound), as inputs reflect predecessor blocks' current state.

        Returns:
            BlockInputCache: disk-backed cache; use cache.load_block(idx, n_mm_blocks)
        """
        mmdit = self.pipeline.mmdit
        n_blocks = len(mmdit.multimodal_transformer_blocks)
        cache = BlockInputCache(cache_dir, n_blocks)

        from diffusionkit.mlx.mmdit import MultiModalTransformerBlock
        orig_mm = MultiModalTransformerBlock.__call__

        def patched_mm_call(self_block, latent_image_embeddings,
                            token_level_text_embeddings, timestep, **kwargs):
            bidx = getattr(self_block, "_input_collect_idx", None)
            if bidx is not None:
                cache.append(bidx, (latent_image_embeddings,
                                    token_level_text_embeddings,
                                    timestep))
            return orig_mm(self_block, latent_image_embeddings,
                           token_level_text_embeddings, timestep, **kwargs)

        for idx, block in enumerate(mmdit.multimodal_transformer_blocks):
            block._input_collect_idx = idx

        MultiModalTransformerBlock.__call__ = patched_mm_call

        orig_final = None
        if hasattr(mmdit, "final_layer"):
            from diffusionkit.mlx.mmdit import FinalLayer
            orig_final = FinalLayer.__call__
            final_idx = n_blocks

            def patched_final_call(self_block, *args, **kwargs):
                cache.append_final(final_idx, args)
                return orig_final(self_block, *args, **kwargs)

            FinalLayer.__call__ = patched_final_call

        sample_count = [0]

        def hook_fn(x_doubled, cond_mx, t_broadcast, cal_idx):
            out = mmdit(
                latent_image_embeddings=x_doubled,
                token_level_text_embeddings=mx.expand_dims(cond_mx, 2),
                timestep=t_broadcast,
            )
            mx.eval(out)
            del out
            sample_count[0] += 1
            if sample_count[0] % 25 == 0 or sample_count[0] == len(self.sample_indices):
                pct = 100 * sample_count[0] / len(self.sample_indices)
                print(f"  [{sample_count[0]:4d}/{len(self.sample_indices)}  {pct:5.1f}%]  streaming inputs to disk...")
            if sample_count[0] % 100 == 0:
                gc.collect()
                try:
                    mx.metal.clear_cache()
                except Exception:
                    pass

        print("Collecting block inputs for all blocks (single sweep)...")
        self._run_forward_collecting(hook_fn)

        # Restore
        MultiModalTransformerBlock.__call__ = orig_mm
        if orig_final is not None:
            from diffusionkit.mlx.mmdit import FinalLayer
            FinalLayer.__call__ = orig_final

        for block in mmdit.multimodal_transformer_blocks:
            if hasattr(block, "_input_collect_idx"):
                delattr(block, "_input_collect_idx")

        disk_mb = cache.disk_usage_mb()
        print(f"  Cached inputs for {n_blocks + 1} blocks, "
              f"{sample_count[0]} samples each ({disk_mb:.0f} MB on disk)")
        return cache

    def collect_block_inputs(
        self,
        block_idx: int,
    ) -> List[Tuple[mx.array, mx.array, mx.array]]:
        """Forward through quantized model up to block_idx, collect that block's inputs.

        Returns:
            List of (img_input, txt_input, timestep) tuples for each sample
        """
        mmdit = self.pipeline.mmdit
        n_blocks = len(mmdit.multimodal_transformer_blocks)
        block_inputs: List[Tuple[mx.array, mx.array, mx.array]] = []

        if block_idx < n_blocks:
            # Hook the target MultiModalTransformerBlock to capture inputs
            from diffusionkit.mlx.mmdit import MultiModalTransformerBlock
            orig_call = MultiModalTransformerBlock.__call__
            target_block = mmdit.multimodal_transformer_blocks[block_idx]

            def patched_call(self_block, latent_image_embeddings,
                           token_level_text_embeddings, timestep, **kwargs):
                if self_block is target_block:
                    block_inputs.append((
                        mx.array(latent_image_embeddings),
                        mx.array(token_level_text_embeddings),
                        mx.array(timestep),
                    ))
                return orig_call(self_block, latent_image_embeddings,
                               token_level_text_embeddings, timestep, **kwargs)

            MultiModalTransformerBlock.__call__ = patched_call
        else:
            # FinalLayer: capture its input
            from diffusionkit.mlx.mmdit import FinalLayer
            orig_call = FinalLayer.__call__
            target_block = mmdit.final_layer

            def patched_call(self_block, *args, **kwargs):
                if self_block is target_block:
                    block_inputs.append(tuple(mx.array(a) for a in args))
                return orig_call(self_block, *args, **kwargs)

            FinalLayer.__call__ = patched_call

        sample_count = [0]
        def hook_fn(x_doubled, cond_mx, t_broadcast, cal_idx):
            out = mmdit(
                latent_image_embeddings=x_doubled,
                token_level_text_embeddings=mx.expand_dims(cond_mx, 2),
                timestep=t_broadcast,
            )
            mx.eval(out)
            sample_count[0] += 1

        label = f"block {block_idx}" if block_idx < n_blocks else "FinalLayer"
        print(f"  Collecting inputs for {label} ({len(self.sample_indices)} samples)...")
        self._run_forward_collecting(hook_fn)

        # Restore
        if block_idx < n_blocks:
            from diffusionkit.mlx.mmdit import MultiModalTransformerBlock
            MultiModalTransformerBlock.__call__ = orig_call
        else:
            from diffusionkit.mlx.mmdit import FinalLayer
            FinalLayer.__call__ = orig_call

        print(f"  ✓ {len(block_inputs)} input samples ready for {label}")
        return block_inputs
