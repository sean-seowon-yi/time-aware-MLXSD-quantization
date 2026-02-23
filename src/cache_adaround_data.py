"""
Cache block-level inputs/outputs for AdaRound weight quantization.

Runs calibration samples through the full-precision model with hooks on every
transformer block, saving each block's FP16 inputs and outputs to disk.  These
cached tensors are then consumed by the AdaRound optimization pass to minimize
per-block reconstruction error without touching model weights.

Collection strategy
-------------------
ALL blocks are hooked simultaneously so only N_images × N_timesteps total
forward passes are required (not multiplied by the number of blocks).  After
each forward pass the hooks are flushed to numpy and the result is written to
one compressed npz per (image, timestep) pair.  Peak memory is therefore:

    n_blocks × batch × seq × hidden × sizeof(float16)

which for SD3-medium (62 blocks, batch=2, ~1200 tokens, dim=1536) is ≈ 430 MB
— comfortably within the unified-memory budget of an M-series Mac.

Output layout
-------------
  <output_dir>/
    metadata.json           block names, shapes, sample list
    samples/
      {img:04d}_{step:03d}.npz    all block I/O for one (image, timestep) pair

NPZ key convention inside each sample file
  For a block named "mm3":
    mm3__arg0   first positional arg  (image hidden states for MM blocks,
                                       unified hidden states for Uni blocks)
    mm3__arg1   second positional arg (text hidden states for MM blocks,
                                       timestep modulation tensor for Uni blocks)
    mm3__arg2   third positional arg  (timestep modulation tensor for MM blocks)
    mm3__kw_positional_encodings   RoPE encodings if present
    mm3__out0   first output  (image output for MM, unified output for Uni)
    mm3__out1   second output (text output for MM; absent for Uni)

Usage
-----
    conda run -n diffusionkit python -m src.cache_adaround_data \\
        --calib-dir /Users/davidholt/ai_projects/mlxproject/calibration_data \\
        --num-images 5 \\
        --stride 5 \\
        --force

With stride=5 on a 50-step schedule this collects 10 timesteps per image.
For 5 images that is 50 sample files.  Each file is ~430 MB uncompressed;
with float16 + lz4 compression expect ~200 MB/file → ~10 GB total.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
from diffusionkit.mlx import DiffusionPipeline, CFGDenoiser


# ---------------------------------------------------------------------------
# Block-level hooking
# ---------------------------------------------------------------------------

class BlockHook:
    """
    Proxy object that replaces a transformer block in the model's block list.

    When the model calls ``block(img, txt, timestep, ...)`` the hook forwards
    the call to the original block, then stores all positional args, keyword
    args, and the return value so they can be flushed to numpy after the
    forward pass completes.

    Usage::

        hook = BlockHook(original_block, "mm5", is_mm=True, list_idx=5)
        pipeline.mmdit.multimodal_transformer_blocks[5] = hook
        # ... run forward pass ...
        # hook._last_args / _last_kwargs / _last_output are populated (MLX lazy)
    """

    def __init__(self, wrapped, block_name: str, is_mm: bool, list_idx: int):
        self._wrapped = wrapped
        self.block_name = block_name
        self.is_mm = is_mm
        self._list_idx = list_idx   # index in the model's block list for restore
        self._last_args: Optional[Tuple] = None
        self._last_kwargs: Optional[Dict] = None
        self._last_output: Optional[Any] = None

    def __call__(self, *args, **kwargs):
        result = self._wrapped(*args, **kwargs)
        self._last_args = args
        self._last_kwargs = kwargs
        self._last_output = result
        return result

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)

    def clear(self):
        self._last_args = None
        self._last_kwargs = None
        self._last_output = None


def install_block_hooks(pipeline) -> List[BlockHook]:
    """
    Replace every transformer block with a BlockHook proxy.

    Returns hooks in model-traversal order:
      [mm0, mm1, ..., mm{N-1}, uni0, uni1, ..., uni{M-1}]
    """
    mmdit = pipeline.mmdit
    hooks: List[BlockHook] = []

    if hasattr(mmdit, "multimodal_transformer_blocks"):
        for i, block in enumerate(mmdit.multimodal_transformer_blocks):
            hook = BlockHook(block, f"mm{i}", is_mm=True, list_idx=i)
            mmdit.multimodal_transformer_blocks[i] = hook
            hooks.append(hook)

    if hasattr(mmdit, "unified_transformer_blocks"):
        for i, block in enumerate(mmdit.unified_transformer_blocks):
            hook = BlockHook(block, f"uni{i}", is_mm=False, list_idx=i)
            mmdit.unified_transformer_blocks[i] = hook
            hooks.append(hook)

    return hooks


def remove_block_hooks(pipeline, hooks: List[BlockHook]) -> None:
    """Restore original blocks, removing all proxies."""
    mmdit = pipeline.mmdit
    for hook in hooks:
        if hook.is_mm:
            mmdit.multimodal_transformer_blocks[hook._list_idx] = hook._wrapped
        else:
            mmdit.unified_transformer_blocks[hook._list_idx] = hook._wrapped


# ---------------------------------------------------------------------------
# Flushing hook buffers to numpy
# ---------------------------------------------------------------------------

def _to_numpy(val: Any) -> Any:
    """Convert an mx.array to numpy; pass through everything else."""
    if isinstance(val, mx.array):
        return np.array(val)
    return val


def flush_hooks(hooks: List[BlockHook]) -> Dict[str, Optional[Dict]]:
    """
    1. Collect all pending mx.arrays from every hook into one batch eval.
    2. Convert each to numpy.
    3. Clear hook buffers and return a dict keyed by block name.

    Returns
    -------
    dict mapping block_name -> {
        'args':   list of (np.ndarray | other),
        'kwargs': dict of (str -> np.ndarray | other),
        'output': np.ndarray | list[np.ndarray] | other,
    }
    or None if that hook fired no call this pass.
    """
    # --- single mx.eval for all pending tensors across all hooks ---
    pending: List[mx.array] = []
    for hook in hooks:
        if hook._last_args is None:
            continue
        for a in hook._last_args:
            if isinstance(a, mx.array):
                pending.append(a)
        for v in hook._last_kwargs.values():
            if isinstance(v, mx.array):
                pending.append(v)
        out = hook._last_output
        if isinstance(out, (list, tuple)):
            for o in out:
                if isinstance(o, mx.array):
                    pending.append(o)
        elif isinstance(out, mx.array):
            pending.append(out)

    if pending:
        mx.eval(*pending)

    # --- convert to numpy ---
    result: Dict[str, Optional[Dict]] = {}
    for hook in hooks:
        if hook._last_args is None:
            result[hook.block_name] = None
            continue

        args_np = [_to_numpy(a) for a in hook._last_args]
        kwargs_np = {k: _to_numpy(v) for k, v in hook._last_kwargs.items()}

        out = hook._last_output
        if isinstance(out, (list, tuple)):
            out_np = [_to_numpy(o) for o in out]
        else:
            out_np = _to_numpy(out)

        result[hook.block_name] = {
            "args": args_np,
            "kwargs": kwargs_np,
            "output": out_np,
        }
        hook.clear()

    return result


# ---------------------------------------------------------------------------
# NPZ packing / unpacking
# ---------------------------------------------------------------------------

def pack_sample(block_data: Dict[str, Optional[Dict]]) -> Dict[str, np.ndarray]:
    """
    Flatten block_data into a dict suitable for np.savez_compressed.

    Key format:
        {block_name}__arg{i}            positional args
        {block_name}__kw_{kwarg_name}   keyword args
        {block_name}__out{i}            outputs (output may be a single array
                                         or a list; always indexed 0, 1, ...)
    Block names have dots replaced by underscores to form valid npz keys.
    """
    flat: Dict[str, np.ndarray] = {}
    for block_name, data in block_data.items():
        if data is None:
            continue
        safe = block_name.replace(".", "_")

        for i, a in enumerate(data["args"]):
            if isinstance(a, np.ndarray):
                flat[f"{safe}__arg{i}"] = a

        for k, v in data["kwargs"].items():
            if isinstance(v, np.ndarray):
                flat[f"{safe}__kw_{k}"] = v

        out = data["output"]
        if isinstance(out, (list, tuple)):
            for i, o in enumerate(out):
                if isinstance(o, np.ndarray):
                    flat[f"{safe}__out{i}"] = o
        elif isinstance(out, np.ndarray):
            flat[f"{safe}__out0"] = out

    return flat


def load_block_data(
    block_name: str,
    sample_files: List[Path],
) -> Dict[str, np.ndarray]:
    """
    Load and stack all per-sample data for a single block.

    Returns a dict with stacked arrays:
        'arg0', 'arg1', ...     shape (N_samples, *original_shape)
        'kw_positional_encodings', ...
        'out0', 'out1', ...

    Only keys present in ALL sample files are returned (handles missing data
    gracefully by skipping that sample).
    """
    safe = block_name.replace(".", "_")
    prefix = safe + "__"

    per_sample: List[Dict[str, np.ndarray]] = []
    for path in sample_files:
        npz = np.load(path)
        sample = {
            k[len(prefix):]: npz[k]
            for k in npz.files
            if k.startswith(prefix)
        }
        if sample:
            per_sample.append(sample)

    if not per_sample:
        return {}

    # Find keys common to all samples
    common_keys = set(per_sample[0].keys())
    for s in per_sample[1:]:
        common_keys &= set(s.keys())

    return {k: np.stack([s[k] for s in per_sample]) for k in sorted(common_keys)}


# ---------------------------------------------------------------------------
# Shape recording utilities
# ---------------------------------------------------------------------------

def _record_shapes(block_data: Dict[str, Optional[Dict]]) -> Dict[str, Dict]:
    """Extract shape info from a single flush result (for metadata)."""
    info: Dict[str, Dict] = {}
    for block_name, data in block_data.items():
        if data is None:
            continue
        info[block_name] = {
            "arg_shapes": [
                list(a.shape) if isinstance(a, np.ndarray) else None
                for a in data["args"]
            ],
            "kwarg_shapes": {
                k: list(v.shape) if isinstance(v, np.ndarray) else None
                for k, v in data["kwargs"].items()
            },
            "output_shapes": (
                [list(o.shape) if isinstance(o, np.ndarray) else None
                 for o in data["output"]]
                if isinstance(data["output"], (list, tuple))
                else [list(data["output"].shape)
                      if isinstance(data["output"], np.ndarray) else None]
            ),
        }
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cache block-level FP16 I/O for AdaRound optimization"
    )
    parser.add_argument("--calib-dir", type=Path, required=True,
                        help="Directory containing manifest.json and samples/")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write the cache (default: <calib-dir>/adaround_cache)")
    parser.add_argument("--num-images", type=int, default=5,
                        help="Number of images to use (evenly spaced from manifest)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Use every Nth denoising step (default 5 → 10 steps for 50-step)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing cache")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.calib_dir / "adaround_cache")
    samples_out = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_out.mkdir(exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.force:
        print(f"Cache already exists at {output_dir}. Use --force to regenerate.")
        return

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    manifest_path = args.calib_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    num_steps = manifest["num_steps"]
    cfg_weight = manifest["cfg_scale"]
    key_timesteps = list(range(0, num_steps, args.stride))

    total_images = len(manifest["images"])
    img_stride = max(1, total_images // args.num_images)
    selected_image_ids = [
        i * img_stride for i in range(min(args.num_images, total_images))
    ]
    total_samples = len(selected_image_ids) * len(key_timesteps)

    print("=== AdaRound Cache Collection ===")
    print(f"  Images:    {len(selected_image_ids)} (ids: {selected_image_ids})")
    print(f"  Timesteps: {len(key_timesteps)} (stride={args.stride}): {key_timesteps}")
    print(f"  Samples:   {total_samples}")
    print(f"  Output:    {output_dir}\n")

    # ------------------------------------------------------------------
    # Initialize pipeline
    # ------------------------------------------------------------------
    print("=== Initializing Pipeline ===")
    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    print("✓ Pipeline loaded")

    # ------------------------------------------------------------------
    # Install block hooks
    # ------------------------------------------------------------------
    print("\n=== Installing Block Hooks ===")
    hooks = install_block_hooks(pipeline)
    n_mm = sum(1 for h in hooks if h.is_mm)
    n_uni = sum(1 for h in hooks if not h.is_mm)
    block_names = [h.block_name for h in hooks]
    print(f"  {len(hooks)} hooks installed ({n_mm} multimodal, {n_uni} unified)")

    # ------------------------------------------------------------------
    # Collection loop
    # ------------------------------------------------------------------
    print("\n=== Collecting Samples ===")
    calib_samples_dir = args.calib_dir / "samples"

    collected: List[Dict] = []
    shape_info: Dict[str, Dict] = {}
    errors = 0
    start_time = time.time()

    img_pbar = tqdm(total=len(selected_image_ids), desc="Images",
                    unit="img", position=0)
    step_pbar = tqdm(total=total_samples, desc="Samples",
                     unit="sample", position=1, leave=True)

    for img_idx in selected_image_ids:
        img_meta = manifest["images"][img_idx]
        prompt = img_meta["prompt"]
        tqdm.write(f"\n[img {img_idx}] {prompt[:70]}...")

        # Encode text
        try:
            conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
            mx.eval(conditioning, pooled)
        except Exception as e:
            tqdm.write(f"  ERROR encode_text: {e}")
            errors += len(key_timesteps)
            step_pbar.update(len(key_timesteps))
            img_pbar.update(1)
            continue

        # Build full timestep list for modulation cache
        all_timesteps = []
        for si in range(num_steps):
            sf = calib_samples_dir / f"{img_idx:04d}_{si:03d}.npz"
            if sf.exists():
                all_timesteps.append(float(np.load(sf)["timestep"]))

        if not all_timesteps:
            tqdm.write(f"  ERROR: no sample files found for img {img_idx}")
            errors += len(key_timesteps)
            step_pbar.update(len(key_timesteps))
            img_pbar.update(1)
            continue

        # Cache modulation params
        try:
            denoiser = CFGDenoiser(pipeline)
            ts_mx = mx.array(all_timesteps).astype(pipeline.activation_dtype)
            denoiser.cache_modulation_params(pooled, ts_mx)
        except Exception as e:
            tqdm.write(f"  ERROR modulation cache: {e}")
            errors += len(key_timesteps)
            step_pbar.update(len(key_timesteps))
            img_pbar.update(1)
            continue

        tqdm.write(f"  Collecting {len(key_timesteps)} timesteps...")

        for step_idx in key_timesteps:
            sf = calib_samples_dir / f"{img_idx:04d}_{step_idx:03d}.npz"
            if not sf.exists():
                tqdm.write(f"  SKIP missing step {step_idx}")
                errors += 1
                step_pbar.update(1)
                continue

            out_path = samples_out / f"{img_idx:04d}_{step_idx:03d}.npz"

            try:
                data = np.load(sf)
                x = mx.array(data["x"])
                timestep = mx.array(data["timestep"])
                sigma = mx.array(data["sigma"])

                # Forward pass — all hooks capture I/O in this single call
                _ = denoiser(
                    x, timestep, sigma,
                    conditioning=conditioning,
                    cfg_weight=cfg_weight,
                )

                # Evaluate + convert all hooked tensors to numpy in one batch
                block_data = flush_hooks(hooks)

                # Record shapes from first successful sample
                if not shape_info:
                    shape_info = _record_shapes(block_data)

                # Write per-sample npz
                flat = pack_sample(block_data)
                np.savez_compressed(out_path, **flat)

                collected.append({
                    "img_idx": img_idx,
                    "step_idx": step_idx,
                    "file": out_path.name,
                })

            except Exception as e:
                tqdm.write(f"  ERROR img={img_idx} step={step_idx}: {e}")
                errors += 1

            step_pbar.update(1)

        try:
            denoiser.clear_cache()
        except Exception:
            pass

        img_pbar.update(1)

    step_pbar.close()
    img_pbar.close()

    # ------------------------------------------------------------------
    # Restore original blocks
    # ------------------------------------------------------------------
    remove_block_hooks(pipeline, hooks)

    elapsed = time.time() - start_time
    print(f"\n=== Collection Complete ===")
    print(f"  {len(collected)}/{total_samples} samples collected")
    print(f"  {errors} errors")
    print(f"  {elapsed / 60:.1f} min")

    # ------------------------------------------------------------------
    # Write metadata
    # ------------------------------------------------------------------
    total_bytes = sum(p.stat().st_size for p in samples_out.glob("*.npz"))
    metadata = {
        "format": "adaround_cache_v1",
        "block_names": block_names,
        "n_mm_blocks": n_mm,
        "n_uni_blocks": n_uni,
        "block_shapes": shape_info,
        "n_samples": len(collected),
        "samples": collected,
        "calib_dir": str(args.calib_dir),
        "selected_image_ids": selected_image_ids,
        "key_timesteps": key_timesteps,
        "stride": args.stride,
        "cfg_weight": cfg_weight,
        "num_steps": num_steps,
        "collection_time_minutes": elapsed / 60,
        "errors": errors,
        "cache_size_gb": total_bytes / 1e9,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Metadata: {metadata_path}")
    print(f"  Cache: {total_bytes / 1e9:.2f} GB ({len(collected)} npz files)")

    # Print a few example shapes
    if shape_info:
        print("\n  Block shapes (first sample):")
        for bname in list(shape_info.keys())[:4]:
            s = shape_info[bname]
            print(f"    {bname}: in={s['arg_shapes']}, out={s['output_shapes']}")
        if len(shape_info) > 4:
            print(f"    ... and {len(shape_info) - 4} more blocks")


if __name__ == "__main__":
    main()
