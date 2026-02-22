"""
Analyze collected activation statistics to determine per-layer quantization strategy.

Reads layer_statistics.json produced by collect_layer_activations.py and outputs:
  1. Per-layer outlier score (how "safe" is 4-bit for this layer)
  2. Shift effectiveness for post-GELU layers
  3. Recommended bit assignments: W4A4 vs W4A8 per layer per bucket
  4. SmoothQuant candidates: layers with few extreme channels
  5. A quantization config JSON ready for use by quantize_model.py

Outlier score (robust):
  - We have per-channel avg_min/avg_max arrays from calibration.
  - For non-post-GELU layers: outlier proxy = p99/p50 of per-channel absmax
    Avoids sensitivity to a single spike; p99/p50 > threshold -> use A8
  - For post-GELU layers: median is near-zero (dead GELU neurons), so
    outlier_ratio is meaningless. Instead we only check shifted_absmax.

SmoothQuant detection:
  - If n_channels_above_99pct / total_channels < SMOOTHQUANT_FRACTION_THRESHOLD
    AND p100/p99 > SMOOTHQUANT_SPIKE_THRESHOLD (isolated spike), flag as
    smoothquant_candidate. Per-channel weight scaling can absorb these.

W4A4 feasibility rules (conservative, tunable via CLI):
  - Post-GELU layers: check shifted_absmax > SHIFTED_ABSMAX_A8_THRESHOLD -> A8
  - Non-post-GELU: if tensor absmax > 12.0 OR p99/p50 > 4.0 -> A8
  - SmoothQuant candidates can be downgraded from A8 to A4 with scaling

Step schedule:
  - Steps 0-34:  W4A4 where feasible (aggressive, ~35 steps)
  - Steps 35-50: W4A8 always (refinement phase, highest sensitivity)

Usage:
    conda run -n diffusionkit python -m src.analyze_activations \\
        --stats /path/to/calibration_data/activations/layer_statistics.json \\
        --output /path/to/calibration_data/activations/quant_config.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Thresholds (tunable via CLI)
# ---------------------------------------------------------------------------

OUTLIER_RATIO_THRESHOLD = 4.0      # p99/p50 per-channel absmax (robust, non-post-GELU)
ABSMAX_A8_THRESHOLD = 12.0         # tensor-level absmax
SHIFTED_ABSMAX_A8_THRESHOLD = 8.0  # post-GELU after shift
LATE_STEP_START = 35               # steps >= this always use A8

# SmoothQuant detection thresholds
SMOOTHQUANT_FRACTION_THRESHOLD = 0.05   # < 5% channels are outliers
SMOOTHQUANT_SPIKE_THRESHOLD = 5.0       # p100/p99 spike ratio to flag isolated spikes


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def robust_outlier_ratio(per_channel_absmax: np.ndarray) -> float:
    """
    p99/p50 of per-channel absmax. More stable than max/median:
    - Ignores single extreme spikes (those go to smoothquant_candidate)
    - Unaffected by near-zero medians (post-GELU case handled separately)
    """
    p50 = float(np.percentile(per_channel_absmax, 50))
    p99 = float(np.percentile(per_channel_absmax, 99))
    if p50 < 1e-6:
        return 1.0
    return p99 / p50


def outlier_ratio(per_channel_absmax: np.ndarray) -> float:
    """max / median of per-channel absmax. Kept for reporting; use robust_outlier_ratio for decisions."""
    med = np.median(per_channel_absmax)
    if med < 1e-6:
        return float("inf") if np.max(per_channel_absmax) > 1e-6 else 1.0
    return float(np.max(per_channel_absmax) / med)


def smoothquant_candidate(per_channel_absmax: np.ndarray) -> Tuple[bool, float, float]:
    """
    Detect layers where a small fraction of channels have isolated extreme values.
    These can be fixed by per-channel weight scaling (SmoothQuant) rather than A8.

    Returns:
        (is_candidate, outlier_fraction, spike_ratio)
        - outlier_fraction: fraction of channels above p99
        - spike_ratio: p100/p99 (how isolated the top channels are)
    """
    p99 = float(np.percentile(per_channel_absmax, 99))
    p100 = float(np.max(per_channel_absmax))
    n_total = len(per_channel_absmax)
    n_above_p99 = int(np.sum(per_channel_absmax > p99))
    outlier_frac = n_above_p99 / n_total

    if p99 < 1e-6:
        return False, outlier_frac, 1.0

    spike_ratio = p100 / p99
    is_candidate = (outlier_frac < SMOOTHQUANT_FRACTION_THRESHOLD and
                    spike_ratio > SMOOTHQUANT_SPIKE_THRESHOLD)
    return is_candidate, outlier_frac, spike_ratio


def smoothquant_scales(per_channel_absmax: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
    """
    Compute per-channel SmoothQuant scaling factors.
    s_c = act_absmax_c^alpha / mean(act_absmax)^alpha
    Weights get divided by s_c, activations get divided by s_c before quant.
    alpha=0.5 balances the migration equally between weights and activations.
    """
    s = per_channel_absmax ** alpha
    s = s / (np.mean(per_channel_absmax) ** alpha + 1e-8)
    return s


def shifted_absmax(per_channel_min: np.ndarray,
                   per_channel_max: np.ndarray,
                   shift: np.ndarray) -> float:
    """After centering by shift, worst-case absmax across channels."""
    return float(np.maximum(np.abs(per_channel_min - shift),
                            np.abs(per_channel_max - shift)).max())


def recommend_act_bits(stats: Dict,
                       threshold_absmax: float,
                       threshold_outlier: float) -> Tuple[int, str]:
    """Return (bits, reason) for activation quantization."""
    avg_min = np.array(stats["avg_min"])
    avg_max = np.array(stats["avg_max"])
    per_ch_absmax = np.maximum(np.abs(avg_min), np.abs(avg_max))
    tensor_absmax = float(per_ch_absmax.max())

    if "shift" in stats:
        # Post-GELU: median is near-zero (dead neurons), outlier_ratio meaningless.
        # Only check shifted_absmax — per-channel + shift handles the bimodal distribution.
        shift = np.array(stats["shift"])
        s_absmax = shifted_absmax(avg_min, avg_max, shift)
        if s_absmax > SHIFTED_ABSMAX_A8_THRESHOLD:
            return 8, f"post-GELU shifted_absmax={s_absmax:.2f}>{SHIFTED_ABSMAX_A8_THRESHOLD}"
        return 4, f"post-GELU ok (shifted_absmax={s_absmax:.2f})"

    # Non-post-GELU: use robust p99/p50 ratio instead of max/median
    rob_ratio = robust_outlier_ratio(per_ch_absmax)
    sq_candidate, sq_frac, sq_spike = smoothquant_candidate(per_ch_absmax)

    if tensor_absmax > threshold_absmax:
        if sq_candidate:
            return 4, (f"smoothquant_candidate: absmax={tensor_absmax:.2f} but "
                       f"only {sq_frac*100:.1f}% channels outlier, spike={sq_spike:.1f}x")
        return 8, f"absmax={tensor_absmax:.2f}>{threshold_absmax}"
    if rob_ratio > threshold_outlier:
        if sq_candidate:
            return 4, (f"smoothquant_candidate: p99/p50={rob_ratio:.2f} but "
                       f"only {sq_frac*100:.1f}% channels outlier, spike={sq_spike:.1f}x")
        return 8, f"p99/p50={rob_ratio:.2f}>{threshold_outlier}"
    return 4, f"ok (absmax={tensor_absmax:.2f}, p99/p50={rob_ratio:.2f})"


def analyze_bucket(layers: Dict,
                   threshold_absmax: float,
                   threshold_outlier: float) -> Dict:
    results = {}
    for layer_name, stats in layers.items():
        if not stats.get("avg_min"):
            continue
        avg_min = np.array(stats["avg_min"])
        avg_max = np.array(stats["avg_max"])
        per_ch_absmax = np.maximum(np.abs(avg_min), np.abs(avg_max))
        is_post_gelu = "shift" in stats

        act_bits, reason = recommend_act_bits(stats, threshold_absmax, threshold_outlier)

        sq_candidate, sq_frac, sq_spike = (False, 0.0, 1.0)
        if not is_post_gelu:
            sq_candidate, sq_frac, sq_spike = smoothquant_candidate(per_ch_absmax)

        entry = {
            "act_bits": act_bits,
            "weight_bits": 4,
            "reason": reason,
            "tensor_absmax": float(per_ch_absmax.max()),
            "outlier_ratio_max_med": round(outlier_ratio(per_ch_absmax), 3),
            "outlier_ratio_p99_p50": round(robust_outlier_ratio(per_ch_absmax), 3),
            "n_channels": int(len(avg_min)),
            "n_batches": stats.get("n_batches", 0),
            "is_post_gelu": is_post_gelu,
            "smoothquant_candidate": sq_candidate,
        }
        if sq_candidate:
            sq_scales = smoothquant_scales(per_ch_absmax)
            entry["smoothquant_outlier_frac"] = round(sq_frac, 4)
            entry["smoothquant_spike_ratio"] = round(sq_spike, 2)
            entry["smoothquant_scales"] = sq_scales.tolist()
        if is_post_gelu:
            shift = np.array(stats["shift"])
            entry["shift_absmax"] = float(stats["shift_absmax"])
            entry["shifted_absmax"] = round(shifted_absmax(avg_min, avg_max, shift), 4)

        results[layer_name] = entry
    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(analysis: Dict):
    for bucket, layers in analysis.items():
        if not layers:
            continue
        a4 = [n for n, v in layers.items() if v["act_bits"] == 4]
        a8 = [n for n, v in layers.items() if v["act_bits"] == 8]
        pg = [n for n, v in layers.items() if v["is_post_gelu"]]
        sq = [n for n, v in layers.items() if v.get("smoothquant_candidate")]

        print(f"\n{'='*60}")
        print(f"Bucket [{bucket}]: {len(layers)} layers  "
              f"A4={len(a4)} ({100*len(a4)//max(1,len(layers))}%)  "
              f"A8={len(a8)}  post-GELU={len(pg)}  SmoothQuant={len(sq)}")

        if a8:
            true_a8 = [n for n in a8 if not layers[n].get("smoothquant_candidate")]
            if true_a8:
                print(f"  Layers requiring A8 (no fix):")
                for name in sorted(true_a8)[:8]:
                    print(f"    {name}: {layers[name]['reason']}")
                if len(true_a8) > 8:
                    print(f"    ... and {len(true_a8)-8} more")

        if sq:
            print(f"  SmoothQuant candidates (A4 with per-channel scaling):")
            for name in sorted(sq)[:8]:
                v = layers[name]
                print(f"    {name}: {v['smoothquant_outlier_frac']*100:.1f}% outlier channels  "
                      f"spike={v['smoothquant_spike_ratio']:.1f}x  "
                      f"absmax={v['tensor_absmax']:.2f}")
            if len(sq) > 8:
                print(f"    ... and {len(sq)-8} more")

        by_outlier = sorted(layers.items(),
                            key=lambda x: x[1]["outlier_ratio_p99_p50"], reverse=True)
        print(f"  Top outlier layers (p99/p50):")
        for name, v in by_outlier[:5]:
            sq_tag = " [SQ]" if v.get("smoothquant_candidate") else ""
            print(f"    {name}: p99/p50={v['outlier_ratio_p99_p50']:.2f}  "
                  f"max/med={v['outlier_ratio_max_med']:.2f}  "
                  f"absmax={v['tensor_absmax']:.2f}  -> A{v['act_bits']}{sq_tag}")

        if pg:
            raw_abs = [layers[n]["tensor_absmax"] for n in pg]
            shifted = [layers[n]["shifted_absmax"] for n in pg]
            reduction = (1 - np.mean(shifted) / np.mean(raw_abs)) * 100
            print(f"  Post-GELU shift: raw absmax mean={np.mean(raw_abs):.2f}  "
                  f"shifted mean={np.mean(shifted):.2f}  "
                  f"reduction={reduction:.1f}%")


# ---------------------------------------------------------------------------
# Quantization config builder
# ---------------------------------------------------------------------------

def build_quant_config(analysis: Dict, raw_stats: Dict, metadata: Dict) -> Dict:
    """
    Build per-layer quantization config for two phases:
      Phase 1 (steps 0-34):  W4A4 where analysis says feasible, else W4A8
      Phase 2 (steps 35-50): W4A8 always (most sensitive refinement steps)

    Also stores per-channel shift arrays for post-GELU layers so
    quantize_model.py can apply them at inference time.
    """
    # Use "mid" bucket for phase-1 config (most representative — most steps)
    # Fall back to "early" if "mid" is empty
    ref_bucket = "mid" if analysis.get("mid") else "early"
    ref_layers = analysis.get(ref_bucket, {})
    ref_stats = raw_stats.get(ref_bucket, {})

    phase1 = {}
    for layer_name, info in ref_layers.items():
        entry = {
            "weight_bits": 4,
            "act_bits": info["act_bits"],
        }
        # Include shift array for post-GELU layers
        if info["is_post_gelu"] and layer_name in ref_stats:
            entry["shift"] = ref_stats[layer_name].get("shift")
            entry["shift_absmax"] = info.get("shift_absmax")
        # Include SmoothQuant scales for candidates
        if info.get("smoothquant_candidate"):
            entry["smoothquant_scales"] = info.get("smoothquant_scales")
        phase1[layer_name] = entry

    # Phase 2: all A8, include shift if available (still useful for centering)
    all_names = set(ref_layers.keys())
    for b in analysis:
        all_names |= set(analysis[b].keys())

    phase2 = {}
    for layer_name in all_names:
        entry = {"weight_bits": 4, "act_bits": 8}
        # Carry shift from late bucket if available
        late_stats = raw_stats.get("late", {})
        if layer_name in late_stats and "shift" in late_stats[layer_name]:
            entry["shift"] = late_stats[layer_name]["shift"]
        phase2[layer_name] = entry

    n_a4 = sum(1 for v in phase1.values() if v["act_bits"] == 4)
    n_a8 = sum(1 for v in phase1.values() if v["act_bits"] == 8)
    n_sq = sum(1 for v in ref_layers.values() if v.get("smoothquant_candidate"))

    return {
        "strategy": {
            f"steps_0_to_{LATE_STEP_START - 1}": phase1,
            f"steps_{LATE_STEP_START}_to_end": phase2,
        },
        "late_step_start": LATE_STEP_START,
        "reference_bucket_for_phase1": ref_bucket,
        "global_defaults": {"weight_bits": 4, "act_bits": 8},
        "summary": {
            "phase1_total": len(phase1),
            "phase1_a4": n_a4,
            "phase1_a8": n_a8,
            "phase1_a4_pct": round(100 * n_a4 / max(1, len(phase1)), 1),
            "phase1_smoothquant_candidates": n_sq,
            "phase2_total": len(phase2),
            "phase2_all_a8": True,
        },
        "thresholds": {
            "outlier_ratio_p99_p50": OUTLIER_RATIO_THRESHOLD,
            "absmax_a8": ABSMAX_A8_THRESHOLD,
            "shifted_absmax_a8": SHIFTED_ABSMAX_A8_THRESHOLD,
            "smoothquant_fraction": SMOOTHQUANT_FRACTION_THRESHOLD,
            "smoothquant_spike": SMOOTHQUANT_SPIKE_THRESHOLD,
        },
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_stats_v2(stats_path: Path):
    """
    Load per_timestep_npz_v2 format from collect_layer_activations.py.
    Returns: (timesteps_dict, layer_names, metadata)
    where timesteps_dict[step_key][layer_name] = {tensor_absmax, hist_p999, ...}
    """
    with open(stats_path) as f:
        manifest = json.load(f)

    if manifest.get("format") != "per_timestep_npz_v2":
        raise ValueError(f"Expected per_timestep_npz_v2, got {manifest.get('format')}")

    ts_dir = Path(manifest["timestep_dir"])
    step_keys = manifest["step_keys"]
    metadata = manifest.get("metadata", {})

    # Load all timesteps and layers
    timesteps = {}
    layer_names = set()

    for step_key in step_keys:
        npz_path = ts_dir / f"step_{step_key}.npz"
        index_path = ts_dir / f"step_{step_key}_index.json"

        with open(index_path) as f:
            index = json.load(f)

        timesteps[step_key] = index
        layer_names.update(index.keys())

    return timesteps, sorted(layer_names), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, required=True,
                        help="layer_statistics.json from collect_layer_activations.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output quant_config.json path")
    parser.add_argument("--outlier-threshold", type=float, default=OUTLIER_RATIO_THRESHOLD)
    parser.add_argument("--absmax-threshold", type=float, default=ABSMAX_A8_THRESHOLD)
    parser.add_argument("--use-hist-p999", action="store_true", default=False,
                        help="Use hist_p999 (percentile clipping) instead of tensor_absmax for scales")
    args = parser.parse_args()

    output_path = args.output or (args.stats.parent / "quant_config.json")

    print(f"Loading {args.stats}")
    timesteps, layer_names, metadata = load_stats_v2(args.stats)

    print(f"Calibration: {metadata.get('num_images',0)} images  "
          f"× {metadata.get('num_timesteps',0)} timesteps  "
          f"= {metadata.get('total_processed',0)} forward passes")
    print(f"Timesteps collected: {len(timesteps)}  Layers: {len(layer_names)}")
    print(f"Using {'hist_p999 (percentile)' if args.use_hist_p999 else 'tensor_absmax'} for scale computation")

    # Compute per-layer statistics averaged across all timesteps
    print("\n=== Computing Per-Layer Statistics ===")
    layer_stats = {}
    for layer_name in layer_names:
        # Gather stats across all timesteps for this layer
        absmax_vals = []
        hist_p999_vals = []
        for step_key, layers in timesteps.items():
            if layer_name in layers:
                s = layers[layer_name]
                absmax_vals.append(s.get("tensor_absmax", 0))
                if "hist_p999" in s:
                    hist_p999_vals.append(s["hist_p999"])

        layer_stats[layer_name] = {
            "mean_absmax": float(np.mean(absmax_vals)) if absmax_vals else 0,
            "max_absmax": float(np.max(absmax_vals)) if absmax_vals else 0,
            "mean_hist_p999": float(np.mean(hist_p999_vals)) if hist_p999_vals else 0,
            "max_hist_p999": float(np.max(hist_p999_vals)) if hist_p999_vals else 0,
            "is_post_gelu": layer_name.endswith(".mlp.fc2"),
        }

    # Decide A4 vs A8 for each layer
    print("\n=== Per-Layer Quantization Decision ===")
    a4_layers = []
    a8_layers = []
    smoothquant_candidates = []

    for layer_name in sorted(layer_names):
        stats = layer_stats[layer_name]
        # Use hist_p999 for clipping if available and requested
        scale_val = stats["max_hist_p999"] if (args.use_hist_p999 and stats["max_hist_p999"] > 0) else stats["max_absmax"]

        if stats["is_post_gelu"]:
            # Post-GELU: use 8.0 threshold on shifted values
            if scale_val > SHIFTED_ABSMAX_A8_THRESHOLD:
                a8_layers.append((layer_name, scale_val, "post-GELU > threshold"))
            else:
                a4_layers.append((layer_name, scale_val, "post-GELU low"))
        else:
            # Non-post-GELU: check threshold
            if scale_val > args.absmax_threshold:
                a8_layers.append((layer_name, scale_val, f"absmax > {args.absmax_threshold}"))
            else:
                a4_layers.append((layer_name, scale_val, "safe for A4"))

    print(f"A4 candidates: {len(a4_layers)}")
    print(f"A8 required: {len(a8_layers)}")
    for name, scale, reason in a8_layers[:5]:
        print(f"  {name}: {scale:.2f}  ({reason})")
    if len(a8_layers) > 5:
        print(f"  ... and {len(a8_layers) - 5} more")

    # Build and save config
    print(f"\n=== Quantization Config ===")
    phase1_a4 = len(a4_layers)
    phase1_total = len(a4_layers) + len(a8_layers)
    phase1_pct = int(100 * phase1_a4 / phase1_total) if phase1_total > 0 else 0

    print(f"\nPhase 1 (steps 0-{LATE_STEP_START-1}, W4A? mixed):")
    print(f"  {phase1_a4}/{phase1_total} layers -> A4 ({phase1_pct}%)")
    print(f"  {len(a8_layers)}/{phase1_total} layers -> A8")
    print(f"\nPhase 2 (steps {LATE_STEP_START}-50):")
    print(f"  All {phase1_total} layers -> A8 (refinement, max quality)")

    # Build quant config
    quant_config = {
        "format": "per_timestep_quant_config",
        "phase_split": LATE_STEP_START,
        "phase1": {"a4": [n for n, _, _ in a4_layers], "a8": [n for n, _, _ in a8_layers]},
        "phase2": {"a8": sorted(layer_names)},
        "summary": {
            "phase1_a4": phase1_a4,
            "phase1_a8": len(a8_layers),
            "phase1_total": phase1_total,
            "phase1_a4_pct": phase1_pct,
            "phase2_total": len(layer_names),
        },
        "metadata": metadata,
    }

    with open(output_path, "w") as f:
        json.dump(quant_config, f, indent=2)
    print(f"\n✓ Quant config -> {output_path}")

    # Save detailed layer analysis
    analysis = {}
    for layer_name in sorted(layer_names):
        analysis[layer_name] = layer_stats[layer_name]
    analysis_path = args.stats.parent / "layer_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Full analysis -> {analysis_path}")


if __name__ == "__main__":
    main()
