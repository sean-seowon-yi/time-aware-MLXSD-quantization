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

Strategy (V2 — fake-quantized activations)
------------------------------------------
When --quant-config is provided, wraps each nn.Linear with _ActQuantLayer which
applies per-(layer, timestep) fake activation quantization:

  1. Shift  (post-GELU fc2 inputs only): x = x - shift
  2. Outlier scaling (two-scale TaQ-DiT): x = x / multiplier_vector
  3. Fake-quantize: round → clip → dequant (stays float)
  4. Restore: x = x * multiplier_vector, x = x + shift
  5. Forward through the original nn.Linear

Uses a custom Euler inference loop so step_key can be threaded into proxies
before each denoising step.

Usage
-----
    conda run -n diffusionkit python -m src.load_adaround_model \\
        --adaround-output /path/to/quantized_weights \\
        --prompt "a tabby cat sitting on a sofa" \\
        --output-image quant_test.png \\
        --compare

    # V2: fake activation quantization
    conda run -n diffusionkit python -m src.load_adaround_model \\
        --adaround-output /path/to/quantized_weights \\
        --quant-config calibration_data/activations/quant_config.json \\
        --prompt "a tabby cat sitting on a sofa" \\
        --output-image quant_w4a8_actquant.png --compare

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
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
import mlx.nn as nn

from src.adaround_optimize import _get_nested, _set_nested


# ---------------------------------------------------------------------------
# Euler sampling helpers (mirrors generate_calibration_data.py)
# ---------------------------------------------------------------------------

def _append_dims(x: mx.array, target_dims: int) -> mx.array:
    """Append dimensions to the end of x until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x.reshape(x.shape + (1,) * dims_to_append)


def _to_d(x: mx.array, sigma: mx.array, denoised: mx.array) -> mx.array:
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / _append_dims(sigma, x.ndim)


# ---------------------------------------------------------------------------
# Fake quantization helper
# ---------------------------------------------------------------------------

def fake_quant_int(x: mx.array, scale: float, bits: int) -> mx.array:
    """Round-clip-dequantize (fake quantization, stays float)."""
    qmax = 2 ** (bits - 1) - 1
    x_int = mx.round(x / scale)
    x_int = mx.clip(x_int, -qmax, qmax)
    return x_int * scale


# ---------------------------------------------------------------------------
# Activation quantization proxy
# ---------------------------------------------------------------------------

class _ActQuantLayer:
    """
    Wraps an nn.Linear and applies per-(layer, timestep) fake activation
    quantization to its input before the linear transform.

    Two-scale TaQ-DiT approach for post-GELU layers with outlier channels:
      1. Shift (fc2 inputs only)
      2. Divide outlier channels by multiplier_vector
      3. Fake-quantize with scale_normal
      4. Restore outlier channels
      5. Un-shift
    """

    def __init__(
        self,
        layer,
        layer_name: str,
        per_timestep: Dict,
        outlier_cfg: Dict,
    ):
        self.layer = layer
        self.layer_name = layer_name
        self.per_timestep = per_timestep   # step_key (str) → {bits, scale, shift?}
        self.outlier_cfg = outlier_cfg     # {multiplier_vector, ...} or {}
        self.current_step_key: Optional[int] = None

    def __call__(self, x: mx.array) -> mx.array:
        cfg = self.per_timestep.get(str(self.current_step_key))
        if cfg is None:
            return self.layer(x)

        bits = cfg["bits"]
        scale = cfg["scale"]
        if scale < 1e-8:
            return self.layer(x)

        # 1. Shift (post-GELU layers only)
        shift = None
        if "shift" in cfg:
            shift = mx.array(cfg["shift"], dtype=x.dtype)
            x = x - shift

        # 2. Normalize outlier channels
        multiplier = None
        if self.outlier_cfg:
            multiplier = mx.array(self.outlier_cfg["multiplier_vector"], dtype=x.dtype)
            x = x / multiplier

        # 3. Fake-quantize
        x = fake_quant_int(x, scale, bits)

        # 4. Restore outlier channels
        if multiplier is not None:
            x = x * multiplier

        # 5. Un-shift
        if shift is not None:
            x = x + shift

        return self.layer(x)

    def __getattr__(self, name: str):
        # Forward attribute access (e.g. .weight) to the wrapped layer.
        # __getattr__ is only called when normal lookup fails, so self.layer
        # itself is found normally and does not recurse.
        return getattr(self.layer, name)


# ---------------------------------------------------------------------------
# Apply / remove activation quantization hooks
# ---------------------------------------------------------------------------

def apply_act_quant_hooks(
    mmdit,
    per_timestep: Dict,
    outlier_config: Dict,
) -> Tuple[List, List]:
    """
    Walk the MMDiT block hierarchy and wrap each nn.Linear whose name appears
    in per_timestep with an _ActQuantLayer proxy.

    Layer naming convention (matches collect_layer_activations.py):
      mm{i}.img.{attn|mlp}.{q,k,v,o}_proj / fc1 / fc2
      mm{i}.txt.{attn|mlp}.{q,k,v,o}_proj / fc1 / fc2
      uni{i}.{attn|mlp}.{q,k,v,o}_proj / fc1 / fc2

    Returns
    -------
    proxies : list of _ActQuantLayer
        Set proxy.current_step_key before each denoising step.
    patches : list of (parent_obj, attr_name, original_layer)
        Pass to remove_act_quant_hooks() to restore originals.
    """
    # Build set of all layer names that appear in at least one timestep
    quant_layer_names: set = set()
    for step_layers in per_timestep.values():
        quant_layer_names.update(step_layers.keys())

    proxies: List[_ActQuantLayer] = []
    patches: List[Tuple] = []

    def _patch_sub(sub, attr: str, full_name: str):
        layer = getattr(sub, attr, None)
        if layer is None or full_name not in quant_layer_names:
            return
        per_ts = {sk: per_timestep[sk][full_name]
                  for sk in per_timestep if full_name in per_timestep[sk]}
        outlier_cfg = outlier_config.get(full_name, {})
        proxy = _ActQuantLayer(layer, full_name, per_ts, outlier_cfg)
        setattr(sub, attr, proxy)
        patches.append((sub, attr, layer))
        proxies.append(proxy)

    def _patch_transformer_block(tb, prefix: str):
        for lin in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _patch_sub(tb.attn, lin, f"{prefix}.attn.{lin}")
        if hasattr(tb, "mlp"):
            _patch_sub(tb.mlp, "fc1", f"{prefix}.mlp.fc1")
            _patch_sub(tb.mlp, "fc2", f"{prefix}.mlp.fc2")

    if hasattr(mmdit, "multimodal_transformer_blocks"):
        for i, block in enumerate(mmdit.multimodal_transformer_blocks):
            _patch_transformer_block(block.image_transformer_block, f"mm{i}.img")
            _patch_transformer_block(block.text_transformer_block, f"mm{i}.txt")

    if hasattr(mmdit, "unified_transformer_blocks"):
        for i, block in enumerate(mmdit.unified_transformer_blocks):
            _patch_transformer_block(block.transformer_block, f"uni{i}")

    return proxies, patches


def remove_act_quant_hooks(patches: List) -> None:
    """Restore original nn.Linear layers replaced by _ActQuantLayer proxies."""
    for parent, attr, original in patches:
        setattr(parent, attr, original)


# ---------------------------------------------------------------------------
# Custom inference loop for V2 (fake activation quantization)
# ---------------------------------------------------------------------------

def run_act_quant_inference(
    pipeline,
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    num_steps: int,
    seed: int,
    proxies: List[_ActQuantLayer],
    step_keys_sorted: List[int],
):
    """
    Custom Euler inference loop that threads step_key into activation proxies
    before each denoising step.  Matches the DiffusionKit pipeline exactly
    (same pattern as generate_calibration_data.py).

    Returns a PIL.Image.
    """
    from PIL import Image as _PILImage
    from diffusionkit.mlx import CFGDenoiser

    mx.random.seed(seed)

    conditioning, pooled = pipeline.encode_text(prompt, cfg_scale, negative_prompt)
    mx.eval(conditioning, pooled)
    conditioning = conditioning.astype(pipeline.activation_dtype)
    pooled = pooled.astype(pipeline.activation_dtype)

    x_T = pipeline.get_empty_latent(64, 64)
    noise = pipeline.get_noise(seed, x_T)
    sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)

    model = CFGDenoiser(pipeline)
    timesteps = model.model.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
    model.cache_modulation_params(pooled, timesteps)

    noise_scaled = pipeline.sampler.noise_scaling(
        sigmas[0], noise, x_T, pipeline.max_denoise(sigmas)
    )
    x = noise_scaled

    extra_args = {"conditioning": conditioning, "cfg_weight": cfg_scale}

    for i in range(len(sigmas) - 1):
        # Find nearest collected step_key for this step index
        nearest_key = min(step_keys_sorted, key=lambda k: abs(k - i))
        for proxy in proxies:
            proxy.current_step_key = nearest_key

        denoised = model(x, timesteps[i], sigmas[i], **extra_args)
        d = _to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt
        mx.eval(x)

    model.clear_cache()

    latent = pipeline.latent_format.process_out(x)
    mx.eval(latent)
    latent = latent.astype(pipeline.activation_dtype)
    decoded = pipeline.decode_latents_to_image(latent)
    mx.eval(decoded)

    x_img = mx.concatenate([decoded], axis=0)
    x_img = (x_img * 255).astype(mx.uint8)
    return _PILImage.fromarray(np.array(x_img[0]))


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
    parser.add_argument("--quant-config", type=Path, default=None,
                        help="quant_config.json from analyze_activations.py; "
                             "enables V2 fake activation quantization")
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
    # V2: Load quant config and apply activation quantization hooks
    # ------------------------------------------------------------------
    proxies: List[_ActQuantLayer] = []
    act_quant_patches: List = []
    step_keys_sorted: List[int] = []

    if args.quant_config is not None:
        print(f"\n=== Loading Activation Quant Config: {args.quant_config} ===")
        with open(args.quant_config) as f:
            quant_cfg = json.load(f)

        per_timestep = quant_cfg.get("per_timestep", {})
        outlier_config = quant_cfg.get("outlier_config", {})
        step_keys_sorted = sorted(int(k) for k in per_timestep.keys())

        proxies, act_quant_patches = apply_act_quant_hooks(
            pipeline.mmdit, per_timestep, outlier_config
        )

        # Count bits distribution
        n_a8 = n_a4 = n_other = 0
        for step_layers in per_timestep.values():
            for layer_cfg in step_layers.values():
                b = layer_cfg.get("bits", 8)
                if b == 8:
                    n_a8 += 1
                elif b == 4:
                    n_a4 += 1
                else:
                    n_other += 1
        total_dec = n_a8 + n_a4 + n_other
        print(f"  Activation quantization enabled: {len(proxies)} layers")
        if total_dec > 0:
            print(f"    A8: {100*n_a8//total_dec}%  A4: {100*n_a4//total_dec}%  "
                  f"other: {n_other}")
        print(f"    Outlier handling: {len(outlier_config)} layers")

    # ------------------------------------------------------------------
    # Generate test image
    # ------------------------------------------------------------------
    print(f"\n=== Generating Quantised Image → {args.output_image} ===")
    t0 = time.time()
    if proxies:
        # V2 path: custom Euler loop threads step_key into proxies
        image = run_act_quant_inference(
            pipeline=pipeline,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            seed=args.seed,
            proxies=proxies,
            step_keys_sorted=step_keys_sorted,
        )
        image.save(args.output_image)
    else:
        # V1 path: standard pipeline.generate_image
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

    # Remove activation quant hooks before cleanup
    if act_quant_patches:
        remove_act_quant_hooks(act_quant_patches)

    if args.compare:
        print(f"\n  Baseline: {baseline_path}")
        print(f"  Quantised: {args.output_image}")

    print("\n✓ Complete")


if __name__ == "__main__":
    main()
