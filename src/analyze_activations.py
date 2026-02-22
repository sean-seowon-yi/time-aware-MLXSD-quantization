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

# Three-tier activation quantization: A4 / A6 / A8
# Based on empirical analysis: A4 step size ≈ scale/8, A6 ≈ scale/32, A8 ≈ scale/128

# For non-post-GELU layers
A4_THRESHOLD = 6.0    # scale < 6.0 → A4 (aggressive, 16 levels)
A6_THRESHOLD = 10.0   # 6.0 ≤ scale < 10.0 → A6 (moderate, 64 levels)
# scale ≥ 10.0 → A8 (conservative, 256 levels)

# For post-GELU layers (after shift centering)
SHIFTED_A4_THRESHOLD = 5.0   # shifted scale < 5.0 → A4
SHIFTED_A6_THRESHOLD = 8.0   # 5.0 ≤ shifted scale < 8.0 → A6
# shifted scale ≥ 8.0 → A8

OUTLIER_RATIO_THRESHOLD = 4.0      # p99/p50 per-channel absmax (robust, non-post-GELU)
LATE_STEP_START = 35               # steps >= this always use A8 (deprecated, using per-timestep now)

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

def analyze_shift_effectiveness(hist_counts: np.ndarray, hist_edges: np.ndarray,
                                shift_val: float = None) -> Dict:
    """
    Analyze whether shift centering helps for this distribution.
    Returns metrics about distribution shape, skewness, and shift effectiveness.
    """
    if hist_counts.sum() == 0:
        return {}

    # Compute distribution statistics from histogram
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    probs = hist_counts / hist_counts.sum()

    # Moments
    mean = np.sum(bin_centers * probs)
    variance = np.sum(((bin_centers - mean) ** 2) * probs)
    std = np.sqrt(variance + 1e-8)

    # Skewness (measure of asymmetry)
    skewness = np.sum(((bin_centers - mean) ** 3) * probs) / (std ** 3 + 1e-8)

    # Fraction of mass below zero
    frac_negative = probs[bin_centers < 0].sum()

    # If we have a shift, check how well it centers
    shift_benefit = 0.0
    if shift_val is not None:
        # Distance of mean from zero before shift
        dist_before = abs(mean)
        # Distance of mean from zero after shift
        dist_after = abs(mean - shift_val)
        shift_benefit = (dist_before - dist_after) / (dist_before + 1e-8)

    return {
        "mean": float(mean),
        "std": float(std),
        "skewness": float(skewness),
        "frac_negative": float(frac_negative),
        "shift_benefit": float(shift_benefit) if shift_val is not None else None,
    }


def load_stats_v2(stats_path: Path):
    """
    Load per_timestep_npz_v2 format from collect_layer_activations.py.
    Returns: (timesteps_dict, per_step_full, layer_names, metadata, sigma_map)
    where:
      - timesteps_dict[step_key][layer_name] = {tensor_absmax, hist_p999, ...}  (index only)
      - per_step_full[step_key][layer_name] = {avg_min, avg_max, shift, hist_counts, hist_edges, ...}
      - sigma_map[step_key] = sigma value
    """
    with open(stats_path) as f:
        manifest = json.load(f)

    if manifest.get("format") != "per_timestep_npz_v2":
        raise ValueError(f"Expected per_timestep_npz_v2, got {manifest.get('format')}")

    ts_dir = Path(manifest["timestep_dir"])
    step_keys = manifest["step_keys"]
    metadata = manifest.get("metadata", {})
    sigma_map = {int(k): float(v) for k, v in manifest.get("sigma_map", {}).items()}

    # Load all timesteps with full data (per-channel arrays + histograms)
    timesteps = {}
    per_step_full = {}
    layer_names = set()

    for step_key in step_keys:
        npz_path = ts_dir / f"step_{step_key}.npz"
        index_path = ts_dir / f"step_{step_key}_index.json"

        with open(index_path) as f:
            index = json.load(f)

        # Load per-channel arrays and histograms from npz
        npz = np.load(npz_path)

        timesteps[step_key] = index
        per_step_full[step_key] = {}
        layer_names.update(index.keys())

        for layer_name in index.keys():
            safe = layer_name.replace(".", "_")
            per_step_full[step_key][layer_name] = {
                "avg_min": npz[f"{safe}__avg_min"].copy() if f"{safe}__avg_min" in npz else None,
                "avg_max": npz[f"{safe}__avg_max"].copy() if f"{safe}__avg_max" in npz else None,
                "shift": npz[f"{safe}__shift"].copy() if f"{safe}__shift" in npz else None,
                "hist_counts": npz[f"{safe}__hist_counts"].copy() if f"{safe}__hist_counts" in npz else None,
                "hist_edges": npz[f"{safe}__hist_edges"].copy() if f"{safe}__hist_edges" in npz else None,
                **index[layer_name],  # Include index fields (tensor_absmax, hist_p999, etc.)
            }

    return timesteps, per_step_full, sorted(layer_names), metadata, sigma_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, required=True,
                        help="layer_statistics.json from collect_layer_activations.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output quant_config.json path")
    parser.add_argument("--a4-threshold", type=float, default=A4_THRESHOLD,
                        help="Scale threshold for A4 vs A6 (default: 6.0)")
    parser.add_argument("--a6-threshold", type=float, default=A6_THRESHOLD,
                        help="Scale threshold for A6 vs A8 (default: 10.0)")
    parser.add_argument("--shifted-a4-threshold", type=float, default=SHIFTED_A4_THRESHOLD,
                        help="Post-GELU shifted scale threshold for A4 vs A6 (default: 5.0)")
    parser.add_argument("--shifted-a6-threshold", type=float, default=SHIFTED_A6_THRESHOLD,
                        help="Post-GELU shifted scale threshold for A6 vs A8 (default: 8.0)")
    parser.add_argument("--use-hist-p999", action="store_true", default=False,
                        help="Use hist_p999 (percentile clipping) instead of tensor_absmax for scales")
    parser.add_argument("--outlier-threshold", type=float, default=OUTLIER_RATIO_THRESHOLD,
                        help="Deprecated: p99/p50 outlier ratio threshold")
    args = parser.parse_args()

    output_path = args.output or (args.stats.parent / "quant_config.json")

    print(f"Loading {args.stats}")
    timesteps, per_step_full, layer_names, metadata, sigma_map = load_stats_v2(args.stats)

    print(f"Calibration: {metadata.get('num_images',0)} images  "
          f"× {metadata.get('num_timesteps',0)} timesteps  "
          f"= {metadata.get('total_processed',0)} forward passes")
    print(f"Timesteps collected: {len(timesteps)}  Layers: {len(layer_names)}")
    print(f"Using {'hist_p999 (percentile)' if args.use_hist_p999 else 'tensor_absmax'} for scale computation")

    step_keys_sorted = sorted(timesteps.keys(), key=int)

    # Analyze shift effectiveness for post-GELU layers
    print("\n=== Shift Effectiveness Analysis ===")
    shift_analysis = {}
    for layer_name in layer_names:
        if not layer_name.endswith(".mlp.fc2"):
            continue
        # Average shift effectiveness across all timesteps
        shift_benefits = []
        skewness_vals = []
        for step_key in step_keys_sorted:
            if layer_name not in per_step_full[step_key]:
                continue
            data = per_step_full[step_key][layer_name]
            if data.get("hist_counts") is not None and data.get("shift") is not None:
                analysis = analyze_shift_effectiveness(
                    data["hist_counts"], data["hist_edges"],
                    shift_val=float(np.mean(data["shift"]))
                )
                if analysis.get("shift_benefit") is not None:
                    shift_benefits.append(analysis["shift_benefit"])
                    skewness_vals.append(analysis["skewness"])

        if shift_benefits:
            shift_analysis[layer_name] = {
                "mean_shift_benefit": float(np.mean(shift_benefits)),
                "mean_skewness": float(np.mean(skewness_vals)),
                "effective": bool(np.mean(shift_benefits) > 0.1),  # >10% reduction in mean offset
            }

    # Count effective shifts
    effective_shifts = [n for n, a in shift_analysis.items() if a["effective"]]
    print(f"Post-GELU layers with effective shift: {len(effective_shifts)}/{len(shift_analysis)}")
    if effective_shifts[:5]:
        for name in effective_shifts[:5]:
            a = shift_analysis[name]
            print(f"  {name}: benefit={a['mean_shift_benefit']*100:.1f}%, skewness={a['mean_skewness']:.2f}")

    # Compute per-timestep quantization decisions
    print("\n=== Per-Timestep Quantization Decisions ===")
    per_timestep_config = {}
    layer_temporal_stats = {}  # Track temporal variability per layer

    for layer_name in layer_names:
        absmax_vals = []
        p999_vals = []
        a4_count = 0
        a6_count = 0
        a8_count = 0

        for step_key in step_keys_sorted:
            if layer_name not in per_step_full[step_key]:
                continue

            data = per_step_full[step_key][layer_name]
            scale_val = data.get("hist_p999", data.get("tensor_absmax", 0)) if args.use_hist_p999 else data.get("tensor_absmax", 0)
            absmax_vals.append(data.get("tensor_absmax", 0))
            if data.get("hist_p999"):
                p999_vals.append(data["hist_p999"])

            # Decide A4 / A6 / A8 for this specific timestep
            is_post_gelu = layer_name.endswith(".mlp.fc2")

            # Check SmoothQuant candidacy (use first timestep's per-channel stats)
            sq_candidate = False
            sq_scales = None
            if step_key == step_keys_sorted[0]:  # Only compute once
                if data.get("avg_min") is not None and data.get("avg_max") is not None:
                    per_ch_absmax = np.maximum(np.abs(data["avg_min"]), np.abs(data["avg_max"]))
                    sq_candidate, sq_frac, sq_spike = smoothquant_candidate(per_ch_absmax)
                    if sq_candidate:
                        sq_scales = smoothquant_scales(per_ch_absmax).tolist()

            # Three-tier quantization decision
            if is_post_gelu:
                if scale_val < args.shifted_a4_threshold:
                    bits = 4
                    reason = "post-GELU < A4 threshold"
                elif scale_val < args.shifted_a6_threshold:
                    bits = 6
                    reason = "post-GELU A4-A6 range"
                else:
                    bits = 8
                    reason = "post-GELU > A6 threshold"
            else:
                if scale_val < args.a4_threshold:
                    bits = 4
                    reason = "< A4 threshold"
                elif scale_val < args.a6_threshold:
                    bits = 6
                    reason = "A4-A6 range"
                else:
                    bits = 8
                    reason = "> A6 threshold"

            # SmoothQuant can downgrade by one tier
            if bits == 8 and sq_candidate:
                bits = 6
                reason = f"{reason} (SmoothQuant downgrade)"
            elif bits == 6 and sq_candidate and scale_val < args.a4_threshold * 1.2:
                # Only downgrade A6→A4 if close to A4 threshold
                bits = 4
                reason = f"{reason} (SmoothQuant downgrade)"

            # Track counts
            if bits == 4:
                a4_count += 1
            elif bits == 6:
                a6_count += 1
            else:
                a8_count += 1

            if step_key not in per_timestep_config:
                per_timestep_config[step_key] = {}
            cfg_entry = {
                "bits": int(bits),
                "scale": float(scale_val),
                "reason": reason,
            }
            # Only include smoothquant info in first timestep to save space
            if step_key == step_keys_sorted[0] and sq_candidate:
                cfg_entry["smoothquant"] = bool(sq_candidate)
                if sq_scales:
                    cfg_entry["smoothquant_scales"] = sq_scales
            per_timestep_config[step_key][layer_name] = cfg_entry

        # Track temporal variability
        if absmax_vals:
            total_steps = len(absmax_vals)
            layer_temporal_stats[layer_name] = {
                "mean_absmax": float(np.mean(absmax_vals)),
                "min_absmax": float(np.min(absmax_vals)),
                "max_absmax": float(np.max(absmax_vals)),
                "variability_ratio": float(np.max(absmax_vals) / (np.min(absmax_vals) + 1e-6)),
                "a4_steps": a4_count,
                "a6_steps": a6_count,
                "a8_steps": a8_count,
                "a4_fraction": a4_count / total_steps,
                "a6_fraction": a6_count / total_steps,
                "a8_fraction": a8_count / total_steps,
            }

    # Summarize temporal variability
    high_variability = [(n, s["variability_ratio"]) for n, s in layer_temporal_stats.items()
                        if s["variability_ratio"] > 2.0]
    high_variability.sort(key=lambda x: -x[1])

    print(f"\nHigh temporal variability layers (>2x swing):")
    for name, ratio in high_variability[:10]:
        stats = layer_temporal_stats[name]
        total = stats['a4_steps'] + stats['a6_steps'] + stats['a8_steps']
        print(f"  {name}: {ratio:.1f}x swing  "
              f"(min={stats['min_absmax']:.1f}, max={stats['max_absmax']:.1f}, "
              f"A4/A6/A8={stats['a4_steps']}/{stats['a6_steps']}/{stats['a8_steps']} of {total})")

    # Summarize layers that switch bits across timesteps
    switchers = [(n, s) for n, s in layer_temporal_stats.items()
                 if s["a4_steps"] > 0 and s["a8_steps"] > 0]  # Has both A4 and A8
    switchers.sort(key=lambda x: -x[1]["a6_steps"])  # Sort by most A6 usage

    print(f"\nLayers that switch bits across timesteps: {len(switchers)}")
    for name, stats in switchers[:8]:
        total = stats['a4_steps'] + stats['a6_steps'] + stats['a8_steps']
        print(f"  {name}: A4/A6/A8 = {stats['a4_steps']}/{stats['a6_steps']}/{stats['a8_steps']} "
              f"({stats['a4_fraction']*100:.0f}%/{stats['a6_fraction']*100:.0f}%/{stats['a8_fraction']*100:.0f}%)")

    # Identify always-same-bits layers
    always_a8 = [n for n, s in layer_temporal_stats.items() if s["a4_steps"] == 0 and s["a6_steps"] == 0]
    always_a4 = [n for n, s in layer_temporal_stats.items() if s["a6_steps"] == 0 and s["a8_steps"] == 0]
    always_a6 = [n for n, s in layer_temporal_stats.items() if s["a4_steps"] == 0 and s["a8_steps"] == 0]
    print(f"\nAlways A8: {len(always_a8)} layers")
    print(f"Always A6: {len(always_a6)} layers")
    print(f"Always A4: {len(always_a4)} layers")

    # Build and save per-timestep config
    print(f"\n=== Generating Quantization Config ===")

    # Compute summary stats across all timesteps
    total_decisions = len(step_keys_sorted) * len(layer_names)
    total_a4 = sum(1 for sk in step_keys_sorted for ln in layer_names
                   if per_timestep_config.get(sk, {}).get(ln, {}).get("bits") == 4)
    total_a6 = sum(1 for sk in step_keys_sorted for ln in layer_names
                   if per_timestep_config.get(sk, {}).get(ln, {}).get("bits") == 6)
    total_a8 = sum(1 for sk in step_keys_sorted for ln in layer_names
                   if per_timestep_config.get(sk, {}).get(ln, {}).get("bits") == 8)

    # Count SmoothQuant usage (only counted once per layer)
    smoothquant_layers = set()
    for step_key in step_keys_sorted:
        for layer_name, cfg in per_timestep_config.get(step_key, {}).items():
            if cfg.get("smoothquant"):
                smoothquant_layers.add(layer_name)

    print(f"Per-timestep decisions: {total_decisions} total")
    print(f"  A4: {total_a4}/{total_decisions} ({100*total_a4/total_decisions:.1f}%)")
    print(f"  A6: {total_a6}/{total_decisions} ({100*total_a6/total_decisions:.1f}%)")
    print(f"  A8: {total_a8}/{total_decisions} ({100*total_a8/total_decisions:.1f}%)")
    print(f"  SmoothQuant layers: {len(smoothquant_layers)}")
    print(f"  High-variability layers (switch bits): {len(switchers)}")

    quant_config = {
        "format": "per_timestep_quant_config_v3",  # v3 adds three-tier A4/A6/A8
        "per_timestep": per_timestep_config,
        "sigma_map": {str(k): v for k, v in sigma_map.items()},
        "shift_analysis": shift_analysis,
        "summary": {
            "total_timesteps": len(step_keys_sorted),
            "total_layers": len(layer_names),
            "total_decisions": total_decisions,
            "total_a4": total_a4,
            "total_a6": total_a6,
            "total_a8": total_a8,
            "a4_percentage": round(100 * total_a4 / total_decisions, 1),
            "a6_percentage": round(100 * total_a6 / total_decisions, 1),
            "a8_percentage": round(100 * total_a8 / total_decisions, 1),
            "smoothquant_layers": len(smoothquant_layers),
            "high_variability_layers": len(high_variability),
            "switching_layers": len(switchers),
            "always_a8": len(always_a8),
            "always_a6": len(always_a6),
            "always_a4": len(always_a4),
        },
        "thresholds": {
            "a4_threshold": args.a4_threshold,
            "a6_threshold": args.a6_threshold,
            "shifted_a4_threshold": args.shifted_a4_threshold,
            "shifted_a6_threshold": args.shifted_a6_threshold,
            "outlier_ratio": args.outlier_threshold,
            "smoothquant_fraction": SMOOTHQUANT_FRACTION_THRESHOLD,
            "smoothquant_spike": SMOOTHQUANT_SPIKE_THRESHOLD,
        },
        "metadata": metadata,
    }

    with open(output_path, "w") as f:
        json.dump(quant_config, f, indent=2)
    print(f"\n✓ Quant config -> {output_path}")

    # Save detailed layer temporal analysis
    analysis_path = args.stats.parent / "layer_temporal_analysis.json"
    temporal_analysis = {
        "layer_stats": layer_temporal_stats,
        "shift_analysis": shift_analysis,
        "high_variability": [{"layer": n, "ratio": r} for n, r in high_variability],
        "switchers": [{"layer": n, "a4_fraction": s["a4_fraction"]} for n, s in switchers],
        "always_a8": always_a8,
        "always_a4": always_a4,
    }
    with open(analysis_path, "w") as f:
        json.dump(temporal_analysis, f, indent=2)
    print(f"✓ Temporal analysis -> {analysis_path}")


if __name__ == "__main__":
    main()
