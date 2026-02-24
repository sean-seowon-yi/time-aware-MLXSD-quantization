"""
Phase 1: Generate calibration data for time-aware quantization of SD3 (MMDiT).

Aligned with TaQ-DiT (arXiv:2411.14172):
  - 100 sampling steps per trajectory
  - 256 trajectories total
  - Uniformly select 25 timesteps from the 100
  - Shuffle the calibration pool

Improvements over the paper's single-class-label setup for SD3:
  - Supports 1-N diverse text prompts (loaded from a file)
  - Prompts are cycled round-robin across trajectories so the calibration
    set covers varied conditioning, producing more representative activation
    distributions for quantization.
  - Conditioning is stored *once per unique prompt* (not duplicated per
    calibration point) to keep the .npz small (~200 MB instead of ~8 GB).

Saved .npz keys:
  xs              (n_cal, H, W, C)                  noisy latent inputs
  ts              (n_cal,)                           timesteps
  prompt_indices  (n_cal,)                           maps each point -> prompt
  cs              (num_prompts, cfg_batch, seq, dim) token-level text embeddings
  cs_pooled       (num_prompts, cfg_batch, dim_p)    pooled text embeddings
  prompts         (num_prompts,)                     the prompt strings
  cfg_scale       scalar                             CFG weight used
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    from diffusionkit.mlx import DiffusionPipeline
except ImportError:
    _diffusionkit_src = Path(__file__).resolve().parents[2] / "DiffusionKit" / "python" / "src"
    if _diffusionkit_src.is_dir():
        sys.path.insert(0, str(_diffusionkit_src))
        from diffusionkit.mlx import DiffusionPipeline
    else:
        raise ImportError(
            "diffusionkit not found. Install it or run from repo root with "
            "PYTHONPATH=DiffusionKit/python/src"
        )

from calibration_config import (
    NUM_CALIBRATION_SAMPLES,
    NUM_SAMPLING_STEPS,
    NUM_SELECTED_TIMESTEPS,
    MODEL_VERSION,
    DEFAULT_LATENT_SIZE,
    DEFAULT_CFG_WEIGHT,
    DEFAULT_PROMPT_FILE,
)
from calibration_collector import sample_euler_with_calibration


def load_prompts(path: str) -> list[str]:
    """Load prompts from a text file (one per line, blank lines ignored)."""
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def encode_all_prompts(pipeline, prompts, cfg_weight):
    """Encode each prompt once, return lists of (conditioning, pooled) arrays."""
    all_cond = []
    all_pooled = []
    for i, prompt in enumerate(prompts):
        print(f"  Encoding prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
        cond, pooled = pipeline.encode_text(prompt, cfg_weight, "")
        mx.eval(cond)
        mx.eval(pooled)
        cond = cond.astype(pipeline.activation_dtype)
        pooled = pooled.astype(pipeline.activation_dtype)
        all_cond.append(np.asarray(cond))
        all_pooled.append(np.asarray(pooled))
    return all_cond, all_pooled


def get_initial_latent(pipeline, latent_size, seed: int):
    """Build initial noisy latent for a single sample."""
    h, w = latent_size
    x_T = pipeline.get_empty_latent(h, w)
    noise = pipeline.get_noise(seed, x_T)
    sigmas = pipeline.get_sigmas(pipeline.sampler, NUM_SAMPLING_STEPS)
    noise_scaled = pipeline.sampler.noise_scaling(
        sigmas[0], noise, x_T, pipeline.max_denoise(sigmas)
    )
    return noise_scaled, sigmas


def save_calibration_data(
    xs, ts, prompt_indices, cs_list, cs_pooled_list, prompts, cfg_scale, save_path,
):
    """
    Save calibration data to .npz.

    Conditioning is stored once per unique prompt (not per calibration point).
    Phase 2 reconstructs per-sample conditioning via prompt_indices.
    """
    cs = np.stack(cs_list, axis=0)
    cs_pooled = np.stack(cs_pooled_list, axis=0)
    prompts_arr = np.array(prompts, dtype=object)

    np.savez_compressed(
        save_path,
        xs=xs,
        ts=ts,
        prompt_indices=prompt_indices,
        cs=cs,
        cs_pooled=cs_pooled,
        prompts=prompts_arr,
        cfg_scale=np.array(cfg_scale, dtype=np.float32),
    )

    xs_mb = xs.nbytes / 1024**2
    cs_mb = cs.nbytes / 1024**2
    total_mb = xs_mb + ts.nbytes / 1024**2 + cs_mb + cs_pooled.nbytes / 1024**2
    print(f"\nSaved calibration data to {save_path}")
    print(f"  xs              {xs.shape}  ({xs_mb:.1f} MB)")
    print(f"  ts              {ts.shape}")
    print(f"  prompt_indices  {prompt_indices.shape}")
    print(f"  cs              {cs.shape}  ({cs_mb:.1f} MB)")
    print(f"  cs_pooled       {cs_pooled.shape}")
    print(f"  prompts         {len(prompts)} unique")
    print(f"  total           ~{total_mb:.1f} MB (before compression)")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate calibration data for TaQ-DiT on SD3/MMDiT"
    )
    parser.add_argument(
        "--model-version", type=str, default=MODEL_VERSION,
        help="DiffusionKit model key",
    )
    parser.add_argument(
        "--num-fid-samples", type=int, default=NUM_CALIBRATION_SAMPLES,
        help="Total number of trajectories to generate",
    )
    parser.add_argument(
        "--num-sampling-steps", type=int, default=NUM_SAMPLING_STEPS,
        help="Denoising steps per trajectory",
    )
    parser.add_argument(
        "--num-selected-steps", type=int, default=NUM_SELECTED_TIMESTEPS,
        help="Uniformly selected timesteps for calibration set",
    )
    parser.add_argument(
        "--latent-size", type=int, nargs=2, default=list(DEFAULT_LATENT_SIZE),
        metavar=("H", "W"), help="Latent size (e.g. 64 64 for 512x512)",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=DEFAULT_CFG_WEIGHT,
        help="Classifier-free guidance scale (paper: 1.5)",
    )
    parser.add_argument(
        "--prompt-file", type=str, default=DEFAULT_PROMPT_FILE,
        help="Text file with prompts (one per line). Prompts are cycled "
             "round-robin across trajectories.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Global random seed",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="DiT_cali_data.npz",
        help="Output path for calibration .npz",
    )
    parser.add_argument(
        "--low-memory-mode", action="store_true", default=True,
    )
    parser.add_argument(
        "--no-low-memory-mode", action="store_false", dest="low_memory_mode",
    )
    parser.add_argument(
        "--local-ckpt", type=str, default=None,
        help="Path to local MMDiT checkpoint",
    )
    args = parser.parse_args()

    latent_size = tuple(args.latent_size)
    assert latent_size[0] % 2 == 0 and latent_size[1] % 2 == 0, "Latent H,W must be even"

    # --- Load prompts ---
    prompts = load_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    # --- Load pipeline ---
    print("Loading SD3 Medium pipeline...")
    pipeline = DiffusionPipeline(
        w16=True,
        shift=3.0,
        use_t5=True,
        model_version=args.model_version,
        low_memory_mode=args.low_memory_mode,
        a16=True,
        local_ckpt=args.local_ckpt,
    )

    # --- Encode all prompts once ---
    print("Encoding prompts...")
    all_cond, all_pooled = encode_all_prompts(pipeline, prompts, args.cfg_scale)

    # --- Generate trajectories (batch_size=1 for correct CFG) ---
    n_samples = args.num_fid_samples
    n_prompts = len(prompts)
    print(f"\nGenerating {n_samples} trajectories x {args.num_sampling_steps} steps "
          f"(batch_size=1, {n_prompts} prompt(s) cycled round-robin)")

    # Per-trajectory collectors: xs[step][sample], ts[step][sample]
    all_xs = []   # will become (steps, n_samples, 1, H, W, C) then squeeze
    all_ts = []   # will become (steps, n_samples)
    traj_prompt_idx = np.zeros(n_samples, dtype=np.int32)

    t_start = time.time()
    for sample_idx in range(n_samples):
        prompt_idx = sample_idx % n_prompts
        traj_prompt_idx[sample_idx] = prompt_idx

        cond_mx = mx.array(all_cond[prompt_idx])
        pooled_mx = mx.array(all_pooled[prompt_idx])

        seed = args.seed + sample_idx
        x_init, sigmas = get_initial_latent(pipeline, latent_size, seed)
        mx.eval(x_init)

        x_list, t_list = sample_euler_with_calibration(
            pipeline,
            x_init,
            sigmas,
            cond_mx,
            pooled_mx,
            cfg_weight=args.cfg_scale,
        )

        # Stack steps for this trajectory: (steps, 1, H, W, C)
        xs_traj = np.asarray(mx.stack(x_list, axis=0))
        ts_traj = np.asarray(mx.stack(t_list, axis=0))
        all_xs.append(xs_traj)
        all_ts.append(ts_traj)

        elapsed = time.time() - t_start
        eta = elapsed / (sample_idx + 1) * (n_samples - sample_idx - 1)
        print(f"  [{sample_idx+1}/{n_samples}] prompt={prompt_idx} "
              f"seed={seed}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        gc.collect()

    # Stack trajectories:
    #   xs: each (steps, 1, H, W, C) -> concat axis=1 -> (steps, n_samples, H, W, C)
    #   ts: each (steps,)            -> stack  axis=1 -> (steps, n_samples)
    xs = np.concatenate(all_xs, axis=1)
    ts = np.stack(all_ts, axis=1)

    # --- Paper: uniformly select 25 steps from 100 ---
    total_steps = xs.shape[0]
    selected_indices = np.linspace(0, total_steps - 1, args.num_selected_steps, dtype=int)
    xs = xs[selected_indices]   # (25, n_samples, H, W, C)
    ts = ts[selected_indices]   # (25, n_samples)

    # Flatten (steps, n_samples, ...) -> (n_cal, ...)
    N, S = xs.shape[0], xs.shape[1]
    n_cal = N * S
    xs = xs.reshape(n_cal, *xs.shape[2:])   # (6400, H, W, C)
    ts = ts.reshape(n_cal)                   # (6400,)

    # Expand per-trajectory prompt indices to per-calibration-point
    # Each trajectory contributes N_selected_steps calibration points
    prompt_indices = np.tile(traj_prompt_idx, N)

    # Shuffle consistently
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_cal)
    xs = xs[perm]
    ts = ts[perm]
    prompt_indices = prompt_indices[perm]

    save_calibration_data(
        xs, ts, prompt_indices, all_cond, all_pooled,
        prompts, args.cfg_scale, args.output,
    )


if __name__ == "__main__":
    main()
