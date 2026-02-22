"""
Collect per-layer, per-channel activation statistics for TaQ-DiT quantization.



TaQ-DiT key concepts implemented here:
  1. Per-channel min/max (ch_axis = last dim = output channels of the input tensor)
  2. Moving-average shift for post-GELU layers (fc2 inputs):
       shift = 0.95 * shift + 0.05 * (min + max) / 2
  3. AvgMinMax observer: running average of per-image min/max at each timestep
  4. Per-timestep stats (every other step) instead of coarse buckets

Output structure:
  layer_statistics.json:
    {
      "timesteps": {
        "0":  {"mm0.img.attn.q_proj": {avg_min, avg_max, shift?, ...}, ...},
        "2":  {...},
        ...
        "50": {...}
      },
      "sigma_map": {"0": 1.0, "2": 0.96, ...},   # step_idx -> sigma value
      "metadata": {...}
    }

Which layers get shift treatment:
  - mlp.fc2 inputs  →  post-GELU, skewed distribution, needs shift
  - all others      →  roughly symmetric, shift = 0

Usage:
    conda run -n diffusionkit python -m src.collect_layer_activations \\
        --calib-dir /Users/davidholt/ai_projects/mlxproject/calibration_data \\
        --num-images 5 \\
        --output-dir /Users/davidholt/ai_projects/mlxproject/calibration_data/activations \\
        --force

two separate steps:

Step 1: generate_calibration_data.py

Runs the full diffusion pipeline for N images
Saves the intermediate latent state x at every denoising step as .npz files
Saves the final generated images as .png
Writes a manifest.json
This is the slow step — loads the model once per image, runs 50 denoising steps
Step 2: collect_layer_activations.py

Reads the saved .npz files from Step 1 (doesn't regenerate images)
Replays selected timesteps through the model with hooks installed
Records per-channel min/max statistics at every hooked linear layer
Writes layer_statistics.json
Much faster than Step 1 since it only runs 25 of the 50 steps per image
So the workflow is:

generate_calibration_data.py  →  calibration_data/samples/*.npz
                                  calibration_data/images/*.png
                                  calibration_data/manifest.json

collect_layer_activations.py  →  calibration_data/activations/layer_statistics.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
import mlx.nn as nn
from diffusionkit.mlx import DiffusionPipeline, CFGDenoiser


# ---------------------------------------------------------------------------
# Timestep selection
# ---------------------------------------------------------------------------

SHIFT_MOMENTUM = 0.95   # from TaQ-DiT paper (same as ZeroQuant)

# Coarse bucket definitions kept for backward compat with analyze_activations.py
TIMESTEP_BUCKETS = {
    "early": (0.7, 1.0),
    "mid":   (0.3, 0.7),
    "late":  (0.0, 0.3),
}


def get_timestep_bucket(sigma: float) -> str:
    if sigma > 0.7:
        return "early"
    elif sigma > 0.3:
        return "mid"
    else:
        return "late"


def select_key_timesteps(num_steps: int, stride: int = 2) -> List[int]:
    """Every `stride` steps. Default stride=2 → 25-26 steps for 51-step schedule."""
    return list(range(0, num_steps, stride))


def select_representative_images(manifest: Dict, num_images: int) -> List[int]:
    total = len(manifest["images"])
    if num_images >= total:
        return list(range(total))
    stride = total // num_images
    return [i * stride for i in range(num_images)]


# ---------------------------------------------------------------------------
# Per-channel statistics accumulator
# ---------------------------------------------------------------------------

HIST_BINS = 256        # number of histogram bins per layer (across all channels/tokens)
HIST_MAX_ROWS = 256    # max rows to sample from x_2d per batch for histogram
                       # Each row is one token; 256 rows × C channels sampled in MLX
                       # before numpy conversion — keeps histogram fast even for large layers


class ChannelStats:
    """
    Accumulates per-channel min/max with AvgMinMax (running average over batches)
    and a moving-average shift for post-GELU layers.
    Also accumulates a per-layer histogram of ALL activation values (across all
    channels and tokens) for percentile clipping and distribution visualization.

    Channel axis: last dimension of the input tensor (input features to the linear).
    This matches TaQ-DiT's ch_axis convention for activation quantization.

    Histogram strategy:
      - Two-pass: first pass tracks global min/max to set bin edges,
        subsequent passes accumulate into fixed bins.
      - Bins are re-fitted after the first batch — so the first batch's values
        are retroactively histogrammed with the final edges on the second pass.
        For simplicity we just accumulate starting from batch 2 onward with
        fixed edges set from batch 1's observed range (padded by 10%).
      - Shape: hist_counts (HIST_BINS,), hist_edges (HIST_BINS+1,)
    """

    def __init__(self, is_post_gelu: bool = False):
        self.is_post_gelu = is_post_gelu
        self.n_batches = 0
        # Accumulated AvgMinMax (averaged per-batch channel min/max)
        self.avg_min: Optional[np.ndarray] = None   # shape (C,)
        self.avg_max: Optional[np.ndarray] = None   # shape (C,)
        # Moving-average shift (only for post-GELU layers)
        self.shift: Optional[np.ndarray] = None     # shape (C,)
        # Per-layer histogram of all activation values
        self.hist_counts: Optional[np.ndarray] = None  # shape (HIST_BINS,)
        self.hist_edges: Optional[np.ndarray] = None   # shape (HIST_BINS+1,)
        # Running global min/max for histogram range (updated every batch)
        self._global_min: float = float("inf")
        self._global_max: float = float("-inf")
        # Pending raw values from batch 0 to retroactively histogram
        self._first_batch_vals: Optional[np.ndarray] = None

    def update(self, x: mx.array):
        """
        Update stats with one batch tensor x.
        x shape: (..., C) — last dim is the channel dimension.
        """
        # Flatten all but last dim: (N, C)
        x_f = x.astype(mx.float32)
        shape = x_f.shape
        C = shape[-1]
        x_2d = x_f.reshape(-1, C)   # (N, C)

        # Compute per-channel min/max for this batch
        batch_min = np.array(mx.min(x_2d, axis=0))   # (C,)
        batch_max = np.array(mx.max(x_2d, axis=0))   # (C,)

        if self.n_batches == 0:
            self.avg_min = batch_min
            self.avg_max = batch_max
        else:
            # AvgMinMax: running average (equal weight per batch)
            self.avg_min = self.avg_min + (batch_min - self.avg_min) / (self.n_batches + 1)
            self.avg_max = self.avg_max + (batch_max - self.avg_max) / (self.n_batches + 1)

        if self.is_post_gelu:
            batch_shift = (batch_min + batch_max) / 2.0
            if self.shift is None:
                self.shift = batch_shift
            else:
                # Momentum update: shift = 0.95 * shift + 0.05 * batch_shift
                self.shift = SHIFT_MOMENTUM * self.shift + (1 - SHIFT_MOMENTUM) * batch_shift

        # --- Histogram accumulation ---
        # Sample rows from x_2d IN MLX before converting to numpy.
        # This avoids materializing the full (N, C) tensor in numpy.
        # For large layers (e.g. fc2: 2048 tokens × 6144 ch = 12M floats),
        # converting everything is the bottleneck. Instead pick HIST_MAX_ROWS
        # rows randomly and convert only those.
        N_rows = x_2d.shape[0]
        if N_rows > HIST_MAX_ROWS:
            row_idx = np.random.choice(N_rows, size=HIST_MAX_ROWS, replace=False)
            row_idx_mx = mx.array(row_idx)
            vals = np.array(x_2d[row_idx_mx]).ravel()
        else:
            vals = np.array(x_2d).ravel()

        # Track global range using full batch min/max (already computed as numpy)
        batch_vmin = float(batch_min.min())
        batch_vmax = float(batch_max.max())

        if self.n_batches == 0:
            # Save first batch sample; edges set after batch 2 widens the range
            self._first_batch_vals = vals
            self._global_min = batch_vmin
            self._global_max = batch_vmax
        else:
            # Update global range
            self._global_min = min(self._global_min, batch_vmin)
            self._global_max = max(self._global_max, batch_vmax)

            if self.hist_edges is None:
                # Set edges from first two batches' range, padded 5% each side
                span = self._global_max - self._global_min
                pad = max(span * 0.05, 1e-6)
                lo = self._global_min - pad
                hi = self._global_max + pad
                self.hist_edges = np.linspace(lo, hi, HIST_BINS + 1)
                self.hist_counts = np.zeros(HIST_BINS, dtype=np.int64)
                # Retroactively histogram first batch sample
                if self._first_batch_vals is not None:
                    c0, _ = np.histogram(self._first_batch_vals, bins=self.hist_edges)
                    self.hist_counts += c0
                    self._first_batch_vals = None

            # Clamp to edge range (so outliers beyond pad go into edge bins)
            vals_clamped = np.clip(vals, self.hist_edges[0], self.hist_edges[-1])
            c, _ = np.histogram(vals_clamped, bins=self.hist_edges)
            self.hist_counts += c

        self.n_batches += 1

    def percentile(self, p: float) -> float:
        """Estimate percentile from histogram. p in [0, 100]."""
        if self.hist_counts is None or self.hist_counts.sum() == 0:
            return 0.0
        cdf = np.cumsum(self.hist_counts).astype(float)
        cdf /= cdf[-1]
        target = p / 100.0
        idx = np.searchsorted(cdf, target)
        idx = min(idx, HIST_BINS - 1)
        # Linear interpolation within bin
        lo = self.hist_edges[idx]
        hi = self.hist_edges[idx + 1]
        if idx == 0:
            frac = 0.0
        else:
            prev = cdf[idx - 1]
            cur = cdf[idx]
            frac = (target - prev) / (cur - prev + 1e-12)
        return float(lo + frac * (hi - lo))

    def to_dict(self) -> Dict:
        if self.avg_min is None:
            return {}
        result = {
            "avg_min": self.avg_min.tolist(),
            "avg_max": self.avg_max.tolist(),
            "n_batches": self.n_batches,
            # Derived tensor-level stats (for W4A8 per-tensor scale fallback)
            "tensor_min": float(self.avg_min.min()),
            "tensor_max": float(self.avg_max.max()),
            "tensor_absmax": float(max(abs(self.avg_min.min()), abs(self.avg_max.max()))),
        }
        if self.shift is not None:
            result["shift"] = self.shift.tolist()
            result["shift_absmax"] = float(np.abs(self.shift).max())
        # Histogram percentiles (for clipping analysis)
        if self.hist_counts is not None:
            result["hist_p999"] = self.percentile(99.9)
            result["hist_p99"]  = self.percentile(99.0)
            result["hist_p01"]  = self.percentile(1.0)
            result["hist_p001"] = self.percentile(0.1)
        return result


# ---------------------------------------------------------------------------
# Sub-layer collector with per-channel stats
# ---------------------------------------------------------------------------

# Layers whose *inputs* are post-GELU (need shift treatment)
_POST_GELU_SUFFIX = ".mlp.fc2"


class SubLayerCollector:
    """
    Replaces linear sub-layer attributes with hooked wrappers.
    Collects per-channel AvgMinMax + moving-average shift.
    Stats are stored per timestep index (every other step).
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._patches: List[tuple] = []    # (parent_obj, attr_name, original_layer)
        self._active = False
        self._step_idx: Optional[int] = None
        self._pending: list = []           # (step_idx, layer_name, x_mx_array) tuples

        # stats[step_idx][layer_name] = ChannelStats
        # step_idx is an int; populated lazily as steps are collected
        self.stats: Dict[int, Dict[str, ChannelStats]] = {}
        # sigma_map[step_idx] = sigma float (filled during collection)
        self.sigma_map: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Patching
    # ------------------------------------------------------------------

    def _patch_linear(self, parent, attr: str, full_name: str):
        layer = getattr(parent, attr, None)
        if layer is None:
            return

        collector = self
        original_layer = layer
        is_post_gelu = full_name.endswith(_POST_GELU_SUFFIX)

        class _HookedLayer:
            def __call__(self_, *args, **kwargs):
                result = original_layer(*args, **kwargs)
                if collector._active and collector._step_idx is not None and args:
                    # Queue the input tensor for deferred per-channel stats.
                    # Actual numpy conversion happens in _flush_pending after
                    # the full forward pass via a single mx.eval batch.
                    collector._pending.append((collector._step_idx, full_name,
                                               is_post_gelu, args[0]))
                return result

            def __getattr__(self_, name):
                return getattr(original_layer, name)

        setattr(parent, attr, _HookedLayer())
        self._patches.append((parent, attr, original_layer))

    def install_patches(self):
        model = self.pipeline.mmdit

        if hasattr(model, "multimodal_transformer_blocks"):
            for i, block in enumerate(model.multimodal_transformer_blocks):
                prefix = f"mm{i}"
                itb = block.image_transformer_block
                ttb = block.text_transformer_block

                for stream, tb in [("img", itb), ("txt", ttb)]:
                    p = f"{prefix}.{stream}"
                    self._patch_linear(tb.attn, "q_proj", f"{p}.attn.q_proj")
                    self._patch_linear(tb.attn, "k_proj", f"{p}.attn.k_proj")
                    self._patch_linear(tb.attn, "v_proj", f"{p}.attn.v_proj")
                    self._patch_linear(tb.attn, "o_proj", f"{p}.attn.o_proj")
                    if hasattr(tb, "mlp"):
                        self._patch_linear(tb.mlp, "fc1", f"{p}.mlp.fc1")
                        self._patch_linear(tb.mlp, "fc2", f"{p}.mlp.fc2")

        if hasattr(model, "unified_transformer_blocks"):
            for i, block in enumerate(model.unified_transformer_blocks):
                prefix = f"uni{i}"
                tb = block.transformer_block
                self._patch_linear(tb.attn, "q_proj", f"{prefix}.attn.q_proj")
                self._patch_linear(tb.attn, "k_proj", f"{prefix}.attn.k_proj")
                self._patch_linear(tb.attn, "v_proj", f"{prefix}.attn.v_proj")
                self._patch_linear(tb.attn, "o_proj", f"{prefix}.attn.o_proj")
                if hasattr(tb, "mlp"):
                    self._patch_linear(tb.mlp, "fc1", f"{prefix}.mlp.fc1")
                    self._patch_linear(tb.mlp, "fc2", f"{prefix}.mlp.fc2")

        print(f"  Installed {len(self._patches)} layer patches")

    def remove_patches(self):
        for parent, attr, original in self._patches:
            setattr(parent, attr, original)
        self._patches.clear()

    # ------------------------------------------------------------------
    # Stats collection
    # ------------------------------------------------------------------

    def _flush_pending(self):
        """
        Process all queued (step_idx, layer_name, is_post_gelu, tensor) entries.
        Called once per forward pass after the graph has been computed.
        We do ONE mx.eval of all queued tensors, then update ChannelStats.
        """
        if not self._pending:
            return

        tensors = [t for _, _, _, t in self._pending]
        mx.eval(*tensors)

        for step_idx, layer_name, is_post_gelu, x_mx in self._pending:
            if step_idx not in self.stats:
                self.stats[step_idx] = {}
            if layer_name not in self.stats[step_idx]:
                self.stats[step_idx][layer_name] = ChannelStats(is_post_gelu=is_post_gelu)
            self.stats[step_idx][layer_name].update(x_mx)

        self._pending.clear()

    def collect(self, x, timestep, sigma, conditioning, cfg_weight,
                pooled, denoiser: CFGDenoiser, step_idx: int) -> None:
        sigma_val = float(np.array(sigma))
        self.sigma_map[step_idx] = sigma_val
        self._step_idx = step_idx
        self._active = True

        try:
            _ = denoiser(
                x, timestep, sigma,
                conditioning=conditioning,
                cfg_weight=cfg_weight,
            )
            self._flush_pending()
        finally:
            self._active = False
            self._step_idx = None

    def finalize(self) -> Dict:
        result = {}
        for step_idx in sorted(self.stats.keys()):
            result[str(step_idx)] = {}
            for layer_name, cs in self.stats[step_idx].items():
                d = cs.to_dict()
                if d:
                    # Attach raw histogram arrays for npz saving (not JSON-serializable)
                    if cs.hist_counts is not None:
                        d["_hist_counts"] = cs.hist_counts
                        d["_hist_edges"]  = cs.hist_edges
                    result[str(step_idx)][layer_name] = d
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect per-layer TaQ-DiT activation statistics"
    )
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=2,
                        help="Collect every Nth timestep (default 2 → ~25 steps)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.calib_dir / "activations")
    output_dir.mkdir(exist_ok=True, parents=True)
    stats_file = output_dir / "layer_statistics.json"

    if stats_file.exists() and not args.force:
        print(f"Statistics already exist at {stats_file}. Use --force to regenerate.")
        return

    # Load manifest
    print("=== Loading Calibration Metadata ===")
    manifest_path = args.calib_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    num_steps = manifest["num_steps"]
    cfg_weight = manifest["cfg_scale"]
    print(f"Images: {len(manifest['images'])},  Steps: {num_steps},  CFG: {cfg_weight}")

    selected_image_ids = select_representative_images(manifest, args.num_images)
    key_timesteps = select_key_timesteps(num_steps, stride=args.stride)
    total_fwd = len(selected_image_ids) * len(key_timesteps)

    print(f"Processing {len(selected_image_ids)} images × {len(key_timesteps)} timesteps = {total_fwd} forward passes")

    # Init pipeline
    print("\n=== Initializing Pipeline ===")
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

    # Install patches
    print("\n=== Installing Layer Patches ===")
    collector = SubLayerCollector(pipeline)
    collector.install_patches()

    # Collect
    print("\n=== Collecting Activations ===")
    samples_dir = args.calib_dir / "samples"
    processed = 0
    errors = 0
    start_time = time.time()

    img_pbar = tqdm(total=len(selected_image_ids), desc="Images",
                    unit="img", position=0)
    step_pbar = tqdm(total=total_fwd, desc="Steps ",
                     unit="step", position=1, leave=True)

    for img_idx in selected_image_ids:
        img_meta = manifest["images"][img_idx]
        prompt = img_meta["prompt"]
        img_start = time.time()

        tqdm.write(f"\n[img {img_idx}] {prompt[:70]}...")

        try:
            conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
            mx.eval(conditioning, pooled)
        except Exception as e:
            tqdm.write(f"  ERROR encode: {e}")
            step_pbar.update(len(key_timesteps))
            errors += len(key_timesteps)
            img_pbar.update(1)
            continue

        # Load key step data
        step_data_list = []
        for step_idx in key_timesteps:
            sf = samples_dir / f"{img_idx:04d}_{step_idx:03d}.npz"
            if sf.exists():
                step_data_list.append((step_idx, np.load(sf)))

        # Build full timestep list for modulation cache
        all_timesteps = []
        for si in range(num_steps):
            sf = samples_dir / f"{img_idx:04d}_{si:03d}.npz"
            if sf.exists():
                all_timesteps.append(float(np.load(sf)["timestep"]))

        if not all_timesteps or not step_data_list:
            tqdm.write(f"  ERROR: no sample files found for img {img_idx}")
            step_pbar.update(len(key_timesteps))
            errors += len(key_timesteps)
            img_pbar.update(1)
            continue

        tqdm.write(f"  Caching modulation ({len(all_timesteps)} timesteps)...")

        # Cache modulation for all timesteps (batch=2 pooled to match CFGDenoiser)
        try:
            denoiser = CFGDenoiser(pipeline)
            ts_mx = mx.array(all_timesteps).astype(pipeline.activation_dtype)
            denoiser.cache_modulation_params(pooled, ts_mx)
        except Exception as e:
            tqdm.write(f"  ERROR modulation cache: {e}")
            step_pbar.update(len(key_timesteps))
            errors += len(key_timesteps)
            img_pbar.update(1)
            continue

        tqdm.write(f"  Collecting {len(step_data_list)} steps...")

        for step_idx, data in step_data_list:
            try:
                x = mx.array(data["x"])
                timestep = mx.array(data["timestep"])
                sigma = mx.array(data["sigma"])

                collector.collect(x, timestep, sigma,
                                  conditioning, cfg_weight, pooled, denoiser,
                                  step_idx=step_idx)
                processed += 1
            except Exception as e:
                tqdm.write(f"  ERROR step {step_idx}: {e}")
                errors += 1

            step_pbar.update(1)

        # Account for any missing steps
        missing = len(key_timesteps) - len(step_data_list)
        if missing > 0:
            step_pbar.update(missing)
            errors += missing

        try:
            denoiser.clear_cache()
        except Exception:
            pass

        img_time = time.time() - img_start
        elapsed = time.time() - start_time
        imgs_done = img_pbar.n + 1
        rate = imgs_done / elapsed if elapsed > 0 else 0
        remaining = (len(selected_image_ids) - imgs_done) / rate if rate > 0 else 0
        tqdm.write(f"  ✓ {len(step_data_list)} steps in {img_time:.1f}s  |  "
                   f"ETA {remaining/60:.1f} min")
        img_pbar.update(1)

    img_pbar.close()
    step_pbar.close()

    elapsed = time.time() - start_time
    collector.remove_patches()

    print(f"\n=== Collection Complete ===")
    print(f"Processed: {processed}/{total_fwd}   Errors: {errors}   Time: {elapsed/60:.1f} min")

    # Finalize
    print("\n=== Computing Final Statistics ===")
    final_stats = collector.finalize()

    # Summary
    print("\n=== Layer Statistics Summary ===")
    step_keys = sorted(final_stats.keys(), key=int)
    n_layers = len(final_stats[step_keys[0]]) if step_keys else 0
    print(f"Collected {len(step_keys)} timesteps × {n_layers} layers")
    if step_keys:
        print(f"Sigma range: {collector.sigma_map.get(int(step_keys[-1]), '?'):.3f} → "
              f"{collector.sigma_map.get(int(step_keys[0]), '?'):.3f}")

    # Show a sample step
    if step_keys:
        mid_step = step_keys[len(step_keys) // 2]
        layers = final_stats[mid_step]
        post_gelu = [n for n in layers if n.endswith(".mlp.fc2")]
        print(f"\nSample step {mid_step} (sigma={collector.sigma_map.get(int(mid_step), '?'):.3f}):"
              f"  {len(layers)} layers  {len(post_gelu)} post-GELU")
        for layer_name in sorted(layers)[:4]:
            s = layers[layer_name]
            shift_str = f"  shift_absmax={s['shift_absmax']:.4f}" if "shift" in s else ""
            print(f"  {layer_name}: absmax={s['tensor_absmax']:.4f}{shift_str}")
        if len(layers) > 4:
            print(f"  ... and {len(layers) - 4} more")

    # Build sigma_map with string keys for JSON
    sigma_map_str = {str(k): v for k, v in collector.sigma_map.items()}

    # Save using npz for the large per-channel arrays (avoids 1GB JSON OOM)
    # Write a compact index JSON + per-timestep npz files
    print(f"\n=== Saving ===")
    ts_dir = output_dir / "timestep_stats"
    ts_dir.mkdir(exist_ok=True)

    for step_key, layers in final_stats.items():
        npz_data = {}
        layer_index = {}
        for layer_name, s in layers.items():
            safe = layer_name.replace(".", "_")
            npz_data[f"{safe}__avg_min"] = np.array(s["avg_min"])
            npz_data[f"{safe}__avg_max"] = np.array(s["avg_max"])
            if "shift" in s:
                npz_data[f"{safe}__shift"] = np.array(s["shift"])
            if "_hist_counts" in s:
                npz_data[f"{safe}__hist_counts"] = s["_hist_counts"]
                npz_data[f"{safe}__hist_edges"]  = s["_hist_edges"]
            layer_index[layer_name] = {
                "n_batches": s["n_batches"],
                "tensor_absmax": s["tensor_absmax"],
                "tensor_min": s["tensor_min"],
                "tensor_max": s["tensor_max"],
                "has_shift": "shift" in s,
                "has_hist": "_hist_counts" in s,
            }
            if "shift_absmax" in s:
                layer_index[layer_name]["shift_absmax"] = s["shift_absmax"]
            # Store percentile clipping stats in index for fast access
            for pk in ("hist_p999", "hist_p99", "hist_p01", "hist_p001"):
                if pk in s:
                    layer_index[layer_name][pk] = s[pk]
        np.savez_compressed(ts_dir / f"step_{step_key}.npz", **npz_data)
        with open(ts_dir / f"step_{step_key}_index.json", "w") as f:
            json.dump(layer_index, f)

    # Write compact top-level manifest
    manifest_out = {
        "format": "per_timestep_npz_v2",   # v2 adds per-layer histograms
        "hist_bins": HIST_BINS,
        "timestep_dir": str(ts_dir),
        "sigma_map": sigma_map_str,
        "bucket_definitions": {k: list(v) for k, v in TIMESTEP_BUCKETS.items()},
        "shift_momentum": SHIFT_MOMENTUM,
        "post_gelu_suffix": _POST_GELU_SUFFIX,
        "step_keys": sorted(final_stats.keys(), key=int),
        "metadata": {
            "num_images": len(selected_image_ids),
            "num_timesteps": len(key_timesteps),
            "total_processed": processed,
            "key_timesteps": key_timesteps,
            "selected_image_ids": selected_image_ids,
            "collection_time_minutes": elapsed / 60,
            "errors": errors,
        },
    }
    with open(stats_file, "w") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"✓ Saved manifest to {stats_file}")
    print(f"✓ Per-timestep npz files in {ts_dir}")

    total_pairs = sum(len(v) for v in final_stats.values())
    print(f"\n{'='*60}")
    print(f"Done. {len(step_keys)} timesteps × {n_layers} layers = {total_pairs} entries.")
    print(f"Post-GELU layers have per-channel shift values for TaQ-DiT.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
