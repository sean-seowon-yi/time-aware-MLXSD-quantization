"""
CLI entry point for modular quantized inference on SD3 MMDiT.

Builds a QuantizedInferencePipeline from the specified transforms and
generates one image per invocation. Designed for ablation studies —
toggle individual HTG components via flags, and future strategies
(AdaRound, Bayesian Bits) will add their own --adaround-* / --bb-*
argument groups.

Usage examples
--------------
# FP16 baseline (no quantization)
python -m src.inference.run_inference \\
    --prompt "a red fox running through snow" \\
    --output baseline.png

# Full HTG
python -m src.inference.run_inference \\
    --htg-corrections htg_output/htg_corrections.npz \\
    --prompt "a red fox running through snow" \\
    --output htg_full.png

# Ablation: weight rescaling only (disable all adaLN corrections)
python -m src.inference.run_inference \\
    --htg-corrections htg_output/htg_corrections.npz \\
    --no-htg-qkv --no-htg-fc1 --no-htg-oproj \\
    --prompt "a red fox running through snow" \\
    --output htg_weight_only.png

# Ablation: adaLN corrections only (no weight rescaling)
python -m src.inference.run_inference \\
    --htg-corrections htg_output/htg_corrections.npz \\
    --no-htg-weight-rescaling \\
    --prompt "a red fox running through snow" \\
    --output htg_adaln_only.png

# HTG + weight quantization only (W8)
python -m src.inference.run_inference \\
    --htg-corrections htg_output/htg_corrections.npz \\
    --htg-quantize-weights --htg-bits 8 \\
    --prompt "a red fox running through snow" \\
    --output htg_w8.png

# HTG + activation fake-quantization only (A8)
python -m src.inference.run_inference \\
    --htg-corrections htg_output/htg_corrections.npz \\
    --htg-quantize-activations \\
    --htg-activation-ranges htg_output/htg_activation_ranges.npz \\
    --prompt "a red fox running through snow" \\
    --output htg_a8.png

# HTG + full W8A8 (paper-aligned)
python -m src.inference.run_inference \\
    --htg-corrections htg_output/htg_corrections.npz \\
    --htg-quantize-weights --htg-bits 8 \\
    --htg-quantize-activations \\
    --htg-activation-ranges htg_output/htg_activation_ranges.npz \\
    --prompt "a red fox running through snow" \\
    --output htg_w8a8.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_pipeline(model_version: str, local_ckpt: str | None, low_memory: bool):
    """Load the base SD3 DiffusionPipeline."""
    try:
        from diffusionkit.mlx import DiffusionPipeline  # type: ignore
    except ImportError:
        dk = _ROOT / "DiffusionKit" / "python" / "src"
        sys.path.insert(0, str(dk))
        from diffusionkit.mlx import DiffusionPipeline  # type: ignore

    return DiffusionPipeline(
        w16=True,
        shift=3.0,
        use_t5=True,
        model_version=model_version,
        low_memory_mode=low_memory,
        a16=True,
        local_ckpt=local_ckpt,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Modular quantized inference for SD3 MMDiT. "
            "Add --htg-corrections to enable HTG corrections; "
            "omit for an unmodified FP16 baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Generation ────────────────────────────────────────────────────
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative text prompt")
    parser.add_argument("--output", "-o", type=str, default="output.png",
                        help="Output image path")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--cfg-scale", type=float, default=7.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--latent-size", type=int, nargs=2, default=[64, 64],
                        metavar=("H", "W"),
                        help="Latent spatial size (e.g. 64 64 → 512×512 image)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # ── Model ─────────────────────────────────────────────────────────
    parser.add_argument("--model-version", type=str,
                        default="argmaxinc/mlx-stable-diffusion-3-medium",
                        help="DiffusionKit model identifier")
    parser.add_argument("--local-ckpt", type=str, default=None,
                        help="Optional local checkpoint path")
    parser.add_argument("--low-memory-mode", action="store_true", default=True,
                        help="Enable DiffusionKit low_memory_mode")
    parser.add_argument("--no-low-memory-mode", action="store_false",
                        dest="low_memory_mode")

    # ── HTG transform ─────────────────────────────────────────────────
    htg = parser.add_argument_group("HTG transform")
    htg.add_argument("--htg-corrections", type=str, default=None,
                     help="Path to htg_corrections.npz. Omit for baseline.")
    htg.add_argument("--no-htg-weight-rescaling", action="store_false",
                     dest="htg_weight_rescaling", default=True,
                     help="Disable Ŵ = W * s weight rescaling")
    htg.add_argument("--no-htg-qkv", action="store_false",
                     dest="htg_qkv", default=True,
                     help="Disable QKV adaLN correction (β̂₁, γ̂₁)")
    htg.add_argument("--no-htg-fc1", action="store_false",
                     dest="htg_fc1", default=True,
                     help="Disable fc1 adaLN correction (β̂₂, γ̂₂)")
    htg.add_argument("--no-htg-oproj", action="store_false",
                     dest="htg_oproj", default=True,
                     help="Disable oproj sdpa_output shift correction")
    htg.add_argument("--htg-quantize", action="store_true", default=False,
                     help="(Backward compat) Alias for --htg-quantize-weights")
    htg.add_argument("--htg-quantize-weights", action="store_true", default=False,
                     help="Apply MLX block-wise integer weight quantization (W8/W4)")
    htg.add_argument("--htg-bits", type=int, default=8, choices=[4, 8],
                     help="Weight quantization bit-width")
    htg.add_argument("--htg-group-size", type=int, default=64,
                     help="MLX weight quantization group size")
    htg.add_argument("--htg-quantize-activations", action="store_true", default=False,
                     help="Fake-quantize activation inputs (simulates A8, paper Algorithm 2)")
    htg.add_argument("--htg-activation-ranges", type=str, default=None,
                     help="Path to htg_activation_ranges.npz (required for --htg-quantize-activations)")
    htg.add_argument("--htg-activation-bits", type=int, default=8, choices=[4, 8],
                     help="Activation fake-quantization bit-width")
    htg.add_argument("--debug", action="store_true", default=False,
                     help="Enable HTG quantization diagnostics (weight/activation error, timestep misses)")

    # ── Placeholder groups for future strategies ──────────────────────
    # Add --adaround-* and --bb-* argument groups here when implemented.

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Load base pipeline ─────────────────────────────────────────────

    print(f"Loading model: {args.model_version}")
    pipeline = _load_pipeline(args.model_version, args.local_ckpt, args.low_memory_mode)

    # ── Build transform list ───────────────────────────────────────────
    from src.inference.base import QuantizedInferencePipeline
    from src.inference.htg_transform import HTGTransform

    transforms = []

    if args.htg_corrections is not None:
        q_weights = args.htg_quantize_weights or args.htg_quantize
        q_acts = args.htg_quantize_activations
        if q_acts and not args.htg_activation_ranges:
            parser.error("--htg-quantize-activations requires --htg-activation-ranges")
        print(f"HTG corrections: {args.htg_corrections}")
        htg_transform = HTGTransform(
            corrections_path=args.htg_corrections,
            apply_weight_rescaling=args.htg_weight_rescaling,
            apply_qkv_correction=args.htg_qkv,
            apply_fc1_correction=args.htg_fc1,
            apply_oproj_correction=args.htg_oproj,
            quantize_weights=q_weights,
            quantize_activations=q_acts,
            weight_bits=args.htg_bits,
            activation_bits=args.htg_activation_bits,
            group_size=args.htg_group_size,
            activation_ranges_path=args.htg_activation_ranges,
            debug=args.debug,
        )
        transforms.append(htg_transform)
        mode = "/".join(filter(None, [
            f"W{args.htg_bits}" if q_weights else None,
            f"A{args.htg_activation_bits}" if q_acts else None,
        ])) or "corrections only"
        print(f"  weight_rescaling={args.htg_weight_rescaling}, "
              f"qkv={args.htg_qkv}, fc1={args.htg_fc1}, oproj={args.htg_oproj}, "
              f"quantization={mode}")
    else:
        print("No transforms configured — running FP16 baseline.")

    # Future: add AdaRoundTransform, BayesianBitsTransform here based on args

    qpipe = QuantizedInferencePipeline(pipeline, transforms)

    # ── Generate ───────────────────────────────────────────────────────
    print(f"\nGenerating: '{args.prompt}'")
    print(f"  steps={args.num_steps}, cfg={args.cfg_scale}, "
          f"latent={args.latent_size}, seed={args.seed}")

    img = qpipe.generate(
        prompt=args.prompt,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        latent_size=tuple(args.latent_size),
        seed=args.seed,
    )

    # ── Save ───────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    print(f"\nSaved → {out_path.resolve()}")

    qpipe.remove()

    if args.debug and args.htg_corrections is not None:
        htg_transform.print_debug_summary()


if __name__ == "__main__":
    main()
