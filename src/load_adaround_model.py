"""
Load AdaRound quantized weights into the DiffusionKit pipeline for inference.

Strategy (V1 — FP16 dequantize)
---------------------------------
Dequantize the int8 weights saved by adaround_optimize.py back to float16:

    W_fp16 = (weight_int.astype(float32) * scale).astype(float16)

then inject them into the model's nn.Linear layers and run inference normally.
This preserves AdaRound's exact rounding decisions while using the standard FP16
forward pass — no custom layer types needed.

No memory savings vs baseline FP16, but directly validates quantization quality.

Future work (see Notes in README):
  V2 — nn.quantize(mmdit, bits=4) after injection for ~4× memory reduction
       (re-rounds from float16, slight quality loss vs exact AdaRound)
  V3 — custom QuantizedLinear storing int8 + scale for exact AdaRound + 2× savings

Usage
-----
    conda run -n diffusionkit python -m src.load_adaround_model \\
        --adaround-output /path/to/quantized_weights \\
        --prompt "a tabby cat sitting on a sofa" \\
        --output-image quant_test.png \\
        --compare

With --compare a second baseline image is generated with the unmodified FP16 model
for side-by-side quality inspection.

Output layout of adaround_optimize.py (required input)
-------------------------------------------------------
    <adaround-output>/
        config.json
        weights/
            mm0.npz       keys: {safe_path}__weight_int  (int8)
            mm1.npz              {safe_path}__scale       (float32, (out,1))
            ...                  {safe_path}__a_scale     (float32, (1,))
            uni37.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
import mlx.nn as nn

from src.adaround_optimize import _get_nested, _set_nested


# ---------------------------------------------------------------------------
# Load quantized weights from adaround_optimize.py output
# ---------------------------------------------------------------------------

def load_adaround_weights(
    output_dir: Path,
) -> Tuple[Dict, Dict[str, Dict[str, Dict]]]:
    """
    Parse the adaround_optimize.py output directory.

    Returns
    -------
    config : dict
        Contents of config.json (quantisation hyperparameters + per-block metrics).
    quant_weights : dict
        Nested structure:
            block_name (str)
              └─ linear_path (str)
                   └─ 'weight_int' : np.ndarray (int8, shape (out, in))
                      'scale'      : np.ndarray (float32, shape (out, 1))
                      'a_scale'    : float
                      'bits_w'     : int
                      'bits_a'     : int
    """
    output_dir = Path(output_dir)
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {output_dir}")

    with open(config_path) as f:
        config = json.load(f)

    weights_dir = output_dir / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"weights/ directory not found in {output_dir}")

    bits_w = config.get("bits_w", 4)
    bits_a = config.get("bits_a", 8)

    # Build a per-block path lookup from config.json's block_metrics.
    # adaround_optimize.py stores metrics["quant_paths"] = linear_paths for each block.
    # Using the stored paths avoids ambiguity when reversing the safe encoding
    # (e.g. "q_proj" has an underscore that must NOT become a dot).
    path_lookup: Dict[str, List[str]] = {}
    for bm in config.get("block_metrics", []):
        bname = bm.get("block_name")
        paths = bm.get("quant_paths", [])
        if bname and paths:
            path_lookup[bname] = paths

    quant_weights: Dict[str, Dict[str, Dict]] = {}

    for npz_path in sorted(weights_dir.glob("*.npz")):
        block_name = npz_path.stem   # e.g. "mm3" or "uni12"
        npz = np.load(npz_path)
        block_linears: Dict[str, Dict] = {}

        known_paths = path_lookup.get(block_name)
        if known_paths:
            # Preferred path: look up each linear path by its safe-encoded keys
            for lpath in known_paths:
                safe = lpath.replace(".", "_")
                wi_key = f"{safe}__weight_int"
                sc_key = f"{safe}__scale"
                as_key = f"{safe}__a_scale"
                if wi_key not in npz.files or sc_key not in npz.files:
                    continue
                block_linears[lpath] = {
                    "weight_int": npz[wi_key],
                    "scale": npz[sc_key],
                    "a_scale": float(npz[as_key][0]) if as_key in npz.files else 1.0,
                    "bits_w": bits_w,
                    "bits_a": bits_a,
                }
        else:
            # Fallback for NPZ files without config path info.
            # Key format: {safe_path}__{field}
            # Reversal is best-effort; safe only if the original paths contain
            # no underscores (not true for DiffusionKit paths, but kept for
            # backwards compatibility with partial exports).
            tmp: Dict[str, Dict] = {}
            for key in npz.files:
                if "__" not in key:
                    continue
                safe_path, field = key.rsplit("__", 1)
                linear_path = safe_path.replace("_", ".")
                if linear_path not in tmp:
                    tmp[linear_path] = {"bits_w": bits_w, "bits_a": bits_a}
                if field == "weight_int":
                    tmp[linear_path]["weight_int"] = npz[key]
                elif field == "scale":
                    tmp[linear_path]["scale"] = npz[key]
                elif field == "a_scale":
                    tmp[linear_path]["a_scale"] = float(npz[key][0])
            block_linears = {
                p: d for p, d in tmp.items()
                if "weight_int" in d and "scale" in d
            }

        if block_linears:
            quant_weights[block_name] = block_linears

    return config, quant_weights


# ---------------------------------------------------------------------------
# Dequantize and inject into model
# ---------------------------------------------------------------------------

def dequantize(weight_int: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Dequantize int8 AdaRound weight to float16.

    weight_int : int8, shape (out, in)   values in [-8, 7] for 4-bit
    scale      : float32, shape (out, 1) per-output-channel scale

    Returns float16 array of the same shape.
    """
    return (weight_int.astype(np.float32) * scale).astype(np.float16)


def inject_weights(
    pipeline,
    quant_weights: Dict[str, Dict[str, Dict]],
) -> int:
    """
    Dequantize and inject AdaRound weights into the pipeline's MMDiT blocks.

    For each block in quant_weights, for each linear_path:
      1. Dequantize: W_fp16 = weight_int.astype(float32) * scale
      2. Set layer.weight = mx.array(W_fp16)

    A single mx.eval() is called after all injections.

    Returns the number of linear layers successfully updated.
    """
    mmdit = pipeline.mmdit
    injected = 0
    pending: List[mx.array] = []

    for block_name, linears in quant_weights.items():
        is_mm = block_name.startswith("mm")
        idx = int(block_name[2:] if is_mm else block_name[3:])

        try:
            block = (
                mmdit.multimodal_transformer_blocks[idx]
                if is_mm
                else mmdit.unified_transformer_blocks[idx]
            )
        except (AttributeError, IndexError) as e:
            print(f"  WARNING: could not locate block {block_name}: {e}")
            continue

        for linear_path, data in linears.items():
            try:
                layer = _get_nested(block, linear_path)
            except (AttributeError, IndexError) as e:
                print(f"  WARNING: {block_name}.{linear_path}: {e}")
                continue

            W_fp16 = dequantize(data["weight_int"], data["scale"])
            new_weight = mx.array(W_fp16)
            layer.weight = new_weight
            pending.append(new_weight)
            injected += 1

    if pending:
        mx.eval(*pending)

    return injected


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def weight_diff_stats(
    pipeline,
    quant_weights: Dict[str, Dict[str, Dict]],
) -> Dict:
    """
    Compute mean absolute difference between injected and original FP16 weights.

    Useful for sanity-checking that injection changed the weights.
    (Only meaningful if called on a *fresh* pipeline before injection.)
    """
    # We can only measure the diff after injection by comparing to the
    # re-dequantized version vs the (now overwritten) original — so this
    # function is most useful when called with a fresh, unmodified pipeline.
    mmdit = pipeline.mmdit
    diffs: List[float] = []

    for block_name, linears in quant_weights.items():
        is_mm = block_name.startswith("mm")
        idx = int(block_name[2:] if is_mm else block_name[3:])
        try:
            block = (
                mmdit.multimodal_transformer_blocks[idx]
                if is_mm
                else mmdit.unified_transformer_blocks[idx]
            )
        except (AttributeError, IndexError):
            continue

        for linear_path, data in linears.items():
            try:
                layer = _get_nested(block, linear_path)
            except (AttributeError, IndexError):
                continue

            orig_np = np.array(layer.weight)
            quant_np = dequantize(data["weight_int"], data["scale"])
            diff = float(np.mean(np.abs(orig_np - quant_np.astype(orig_np.dtype))))
            diffs.append(diff)

    if not diffs:
        return {}
    return {
        "n_layers": len(diffs),
        "mean_abs_diff": float(np.mean(diffs)),
        "max_abs_diff": float(np.max(diffs)),
        "min_abs_diff": float(np.min(diffs)),
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load AdaRound quantised weights and run a test generation"
    )
    parser.add_argument("--adaround-output", type=Path, required=True,
                        help="Output dir from adaround_optimize.py (contains config.json)")
    parser.add_argument("--prompt", type=str,
                        default="a photo of a tabby cat sitting on a wooden table",
                        help="Text prompt for test image generation")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt (default: empty)")
    parser.add_argument("--output-image", type=Path, default=Path("quant_test.png"),
                        help="Where to save the generated image (default: quant_test.png)")
    parser.add_argument("--num-steps", type=int, default=28,
                        help="Denoising steps (default 28)")
    parser.add_argument("--cfg-scale", type=float, default=7.0,
                        help="CFG guidance scale (default 7.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--compare", action="store_true",
                        help="Also generate a baseline FP16 image for comparison")
    parser.add_argument("--blocks", type=str, default=None,
                        help="Comma-separated block names to inject (default: all)")
    parser.add_argument("--diff-stats", action="store_true",
                        help="Print weight diff stats before injection")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load quantised weight map
    # ------------------------------------------------------------------
    print("=== Loading AdaRound Weights ===")
    config, quant_weights = load_adaround_weights(args.adaround_output)

    if args.blocks:
        selected = set(args.blocks.split(","))
        quant_weights = {k: v for k, v in quant_weights.items() if k in selected}

    n_blocks = len(quant_weights)
    n_layers = sum(len(v) for v in quant_weights.values())
    print(f"  Blocks:       {n_blocks}")
    print(f"  Linear layers: {n_layers}")
    print(f"  W{config.get('bits_w', 4)}A{config.get('bits_a', 8)} "
          f"({config.get('iters', '?')} iters)")

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    from diffusionkit.mlx import DiffusionPipeline

    print("\n=== Loading Pipeline ===")
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
    # Optional: weight diff stats (before injection)
    # ------------------------------------------------------------------
    if args.diff_stats:
        print("\n=== Weight Diff Stats (FP16 baseline vs AdaRound) ===")
        stats = weight_diff_stats(pipeline, quant_weights)
        if stats:
            print(f"  Layers compared:  {stats['n_layers']}")
            print(f"  Mean |Δw|:        {stats['mean_abs_diff']:.6f}")
            print(f"  Max  |Δw|:        {stats['max_abs_diff']:.6f}")
        else:
            print("  (no stats computed)")

    # ------------------------------------------------------------------
    # Baseline comparison image (generated BEFORE injection)
    # ------------------------------------------------------------------
    if args.compare:
        baseline_path = args.output_image.with_stem(
            args.output_image.stem + "_baseline"
        )
        print(f"\n=== Generating Baseline FP16 Image → {baseline_path} ===")
        t0 = time.time()
        images, _ = pipeline.generate_image(
            args.prompt,
            cfg_weight=args.cfg_scale,
            num_steps=args.num_steps,
            seed=args.seed,
            negative_text=args.negative_prompt,
        )
        images[0].save(baseline_path)
        print(f"  Done in {time.time() - t0:.1f}s  →  {baseline_path}")

    # ------------------------------------------------------------------
    # Inject quantised weights
    # ------------------------------------------------------------------
    print("\n=== Injecting AdaRound Weights ===")
    t0 = time.time()
    n_injected = inject_weights(pipeline, quant_weights)
    print(f"  {n_injected} layers injected in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Generate test image
    # ------------------------------------------------------------------
    print(f"\n=== Generating Quantised Image → {args.output_image} ===")
    t0 = time.time()
    images, _ = pipeline.generate_image(
        args.prompt,
        cfg_weight=args.cfg_scale,
        num_steps=args.num_steps,
        seed=args.seed,
        negative_text=args.negative_prompt,
    )
    images[0].save(args.output_image)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  →  {args.output_image}")

    if args.compare:
        print(f"\n  Baseline: {baseline_path}")
        print(f"  Quantised: {args.output_image}")

    print("\n✓ Complete")


if __name__ == "__main__":
    main()
