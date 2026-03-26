"""
Collect per-layer, per-timestep activation absmax statistics for all 285
quantisable linears in SD3 MMDiT.

Outputs the ``activations/`` directory format consumed by
``generate_poly_schedule.py``:

    <output_dir>/
        layer_statistics.json          # metadata + sigma_map
        timestep_stats/
            step_<idx>_index.json      # per-layer scalar stats

Prompt file format: tab-separated ``seed<TAB>prompt`` per line.

Usage:
    python -m src.collect_activation_stats \
        --prompts src/calibration_sample_generation/sample_prompts.txt \
        --output-dir calibration_data_100/activations \
        --num-steps 28
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import mlx.core as mx

from .gptq.utils import load_prompt_file

# ---------------------------------------------------------------------------
# Linear layer enumeration (self-contained, no external deps)
# ---------------------------------------------------------------------------

_LINEAR_PATHS = [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "mlp.fc1",
    "mlp.fc2",
]


def _get_nested(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if "[" in part:
            attr, idx_s = part.split("[", 1)
            obj = getattr(obj, attr)[int(idx_s.rstrip("]"))]
        else:
            obj = getattr(obj, part)
    return obj


def _set_nested(obj: Any, path: str, val: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        if "[" in part:
            attr, idx_s = part.split("[", 1)
            obj = getattr(obj, attr)[int(idx_s.rstrip("]"))]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if "[" in last:
        attr, idx_s = last.split("[", 1)
        getattr(obj, attr)[int(idx_s.rstrip("]"))] = val
    else:
        setattr(obj, last, val)


def _get_block_linears(block, is_mm: bool) -> List[Tuple[str, Any]]:
    """Return (dotted_path, layer) for all quantisable linears in a block."""
    prefixes = (
        ["image_transformer_block", "text_transformer_block"]
        if is_mm else ["transformer_block"]
    )
    results = []
    for prefix in prefixes:
        for local in _LINEAR_PATHS:
            full = f"{prefix}.{local}"
            try:
                layer = _get_nested(block, full)
            except AttributeError:
                continue
            if not hasattr(layer, "weight"):
                continue
            results.append((full, layer))
    return results


# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

class _OutputHook:
    """Transparent proxy that records the last output of a linear layer."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._last_output = None

    def __call__(self, *args, **kwargs):
        result = self._wrapped(*args, **kwargs)
        self._last_output = result
        return result

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def clear(self):
        self._last_output = None


def install_hooks(mmdit) -> Dict[str, Tuple["_OutputHook", int, str]]:
    """Install output hooks on all quantisable linears.

    Returns dict: layer_key -> (hook, block_idx, full_path).
    Keys like 'mm0.img.attn.q_proj' (dot-separated for index JSON compat).
    """
    hooks = {}
    for i, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            short = (
                full_path
                .replace("image_transformer_block", "img")
                .replace("text_transformer_block", "txt")
            )
            key = f"mm{i}.{short}"
            hook = _OutputHook(layer)
            _set_nested(block, full_path, hook)
            hooks[key] = (hook, i, full_path)
    return hooks


def remove_hooks(mmdit, hooks):
    """Restore original layers."""
    for _key, (hook, block_idx, full_path) in hooks.items():
        block = mmdit.multimodal_transformer_blocks[block_idx]
        _set_nested(block, full_path, hook._wrapped)


def collect_step_stats(hooks) -> Dict[str, Dict[str, float]]:
    """Compute per-layer tensor absmax + min + max after one denoising step.

    Returns dict: layer_key -> {tensor_absmax, tensor_min, tensor_max}.
    """
    pending = [
        h._last_output for h, _, _ in hooks.values()
        if h._last_output is not None and isinstance(h._last_output, mx.array)
    ]
    if pending:
        mx.eval(*pending)

    stats = {}
    for key, (hook, _, _) in hooks.items():
        if hook._last_output is not None:
            arr = np.array(hook._last_output)
            abs_arr = np.abs(arr).ravel()
            n = abs_arr.size
            k99 = int(n * 0.99)
            k999 = int(n * 0.999)
            partitioned = np.partition(abs_arr, [k99, k999])
            stats[key] = {
                "tensor_absmax": float(partitioned[-1]) if n > 0 else 0.0,
                "tensor_min": float(arr.min()),
                "tensor_max": float(arr.max()),
                "hist_p99": float(partitioned[k99]),
                "hist_p999": float(partitioned[k999]),
            }
        hook.clear()
    return stats


# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------

def run_prompt(pipeline, denoiser, prompt, hooks, num_steps, cfg_weight=4.0,
               latent_size=64, seed=42):
    """Run full denoising, collect per-step activation stats.

    Returns (list_of_step_stats, sigmas_array).
    """
    conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
    mx.eval(conditioning, pooled)

    sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
    timesteps = pipeline.sampler.timestep(sigmas).astype(pipeline.activation_dtype)
    denoiser.cache_modulation_params(pooled, timesteps)

    mx.random.seed(seed)
    latent_shape = (1, latent_size, latent_size, 16)
    noise = mx.random.normal(latent_shape).astype(pipeline.activation_dtype)
    x = pipeline.sampler.noise_scaling(
        sigmas[0], noise, mx.zeros(latent_shape), pipeline.max_denoise(sigmas)
    )
    mx.eval(x)

    all_step_stats = []
    recorded_sigmas = []
    for i in range(len(sigmas) - 1):
        denoised = denoiser(
            x, timesteps[i], sigmas[i],
            conditioning=conditioning, cfg_weight=cfg_weight,
        )
        all_step_stats.append(collect_step_stats(hooks))
        recorded_sigmas.append(float(sigmas[i].item()))

        d = (x - denoised) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
        mx.eval(x)

    return all_step_stats, np.array(recorded_sigmas)


def _reset_modulation_cache(pipeline):
    """Reload adaLN weights that were offloaded by cache_modulation_params."""
    try:
        pipeline.mmdit.load_weights(
            pipeline.load_mmdit(only_modulation_dict=True), strict=False
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True,
                        help="Tab-separated file: seed<TAB>prompt per line")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("calibration_data_100/activations"))
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--cfg-weight", type=float, default=4.0)
    parser.add_argument("--latent-size", type=int, default=64,
                        help="Spatial size of latent (64 = 512px)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts (for testing)")
    args = parser.parse_args()

    # Load prompts (seed, prompt) pairs
    prompt_entries = load_prompt_file(args.prompts)
    if args.max_prompts is not None:
        prompt_entries = prompt_entries[:args.max_prompts]
    print(f"Loaded {len(prompt_entries)} prompts from {args.prompts}")

    # Load pipeline
    print("Loading pipeline...")
    from diffusionkit.mlx import DiffusionPipeline, CFGDenoiser

    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    denoiser = CFGDenoiser(pipeline)
    print("Pipeline loaded")

    # Install hooks
    hooks = install_hooks(pipeline.mmdit)
    print(f"Installed hooks on {len(hooks)} linear layers")

    # Accumulate: step_idx -> layer_key -> list of per-prompt stats dicts
    acc = defaultdict(lambda: defaultdict(list))
    all_sigmas = None

    for seed, prompt in tqdm(prompt_entries, desc="Prompts"):
        try:
            step_stats, sigmas = run_prompt(
                pipeline, denoiser, prompt, hooks,
                args.num_steps, args.cfg_weight, args.latent_size,
                seed=seed,
            )
            if all_sigmas is None:
                all_sigmas = sigmas
            for step_idx, stats in enumerate(step_stats):
                for layer_key, layer_stats in stats.items():
                    acc[step_idx][layer_key].append(layer_stats)
        except Exception as e:
            print(f"WARNING: skipping prompt: {e}")
        _reset_modulation_cache(pipeline)

    remove_hooks(pipeline.mmdit, hooks)

    if all_sigmas is None:
        raise RuntimeError("All prompts failed — no data collected.")

    # Write output
    print(f"\nWriting output to {args.output_dir}/ ...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = args.output_dir / "timestep_stats"
    ts_dir.mkdir(exist_ok=True)

    num_steps_actual = len(all_sigmas)
    step_keys = [str(i) for i in range(num_steps_actual)]
    sigma_map = {str(i): float(all_sigmas[i]) for i in range(num_steps_actual)}

    for step_idx in range(num_steps_actual):
        index = {}
        for layer_key, stat_list in acc[step_idx].items():
            # Average across prompts
            absmax_vals = [s["tensor_absmax"] for s in stat_list]
            min_vals = [s["tensor_min"] for s in stat_list]
            max_vals = [s["tensor_max"] for s in stat_list]
            p99_vals = [s["hist_p99"] for s in stat_list]
            p999_vals = [s["hist_p999"] for s in stat_list]
            index[layer_key] = {
                "n_batches": len(stat_list),
                "tensor_absmax": float(np.max(absmax_vals)),
                "tensor_min": float(np.min(min_vals)),
                "tensor_max": float(np.max(max_vals)),
                "hist_p99": float(np.max(p99_vals)),
                "hist_p999": float(np.max(p999_vals)),
            }
        idx_path = ts_dir / f"step_{step_idx}_index.json"
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)

    # Write metadata
    meta = {
        "step_keys": step_keys,
        "sigma_map": sigma_map,
        "num_prompts": len(prompt_entries),
        "num_steps": num_steps_actual,
        "num_layers": len(hooks),
    }
    with open(args.output_dir / "layer_statistics.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done: {num_steps_actual} steps x {len(hooks)} layers "
          f"x {len(prompt_entries)} prompts")
    print(f"  sigma range: [{all_sigmas.min():.4f}, {all_sigmas.max():.4f}]")


if __name__ == "__main__":
    main()
