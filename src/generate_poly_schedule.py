"""
Generate polynomial clipping schedule for timestep-aware A8 fake quantization.

Fits tiered polynomials to per-layer p100 absmax activation trajectories across σ steps,
outputting a compact JSON schedule that replaces static per-timestep scale lookups
with continuous polynomial evaluation.

Degree selection tiers:
  All layers          → degree 2 minimum (CV gate removed for cross-group consistency)
  Quadratic R² > 0.85 → degree 2
  Cubic R² gain > 0.15 over quad → degree 3
  Quartic R² gain > 0.10 over cubic → degree 4
  Quartic R² gain > 0.10 over cubic → degree 4 (max)

Range-based degree capping (for derivative-weighted AdaRound stability):
  absmax range < 2 → degree 0 (constant = max absmax)
  absmax range < 5 → degree 2 max (quadratic)

SmoothQuant mode (--smoothquant-scales):
  Instead of using the pre-aggregated p100 absmax trajectories, loads per-channel
  avg_max arrays from timestep_stats NPZ files, applies the SmoothQuant inverse
  scale (x' = x / s[c]), then takes max_c(avg_max_smoothed[c, t]) as the
  per-timestep absmax for polynomial fitting.  This gives an activation schedule
  calibrated to the post-smoothing distribution seen at inference.

Usage:
    conda run -n diffusionkit python -m src.generate_poly_schedule \
        --activations-dir calibration_data_100/activations \
        --output polynomial_clipping_schedule.json

    # SmoothQuant mode:
    conda run -n diffusionkit python -m src.generate_poly_schedule \
        --activations-dir calibration_data_512/activations \
        --smoothquant-scales smoothquant_scales.json \
        --output polynomial_clipping_schedule_smoothquant.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.explore_curve_fits import load_percentile_trajectories, poly_r2


# ---------------------------------------------------------------------------
# Degree selection
# ---------------------------------------------------------------------------

CV_STATIC_THRESHOLD = 0.07
QUAD_R2_THRESHOLD = 0.85
CUBIC_R2_GAIN_THRESHOLD = 0.15
QUARTIC_R2_GAIN_THRESHOLD = 0.10

# Range-based degree capping: layers whose absmax varies less than this across
# all sigma steps are effectively flat. High-degree polynomials on flat data
# produce wild derivatives that corrupt derivative-weighted AdaRound loss.
# Layers below RANGE_STATIC are forced to degree 0 (constant = max absmax).
# Layers below RANGE_MAX_QUAD are capped at degree 2 (smooth, stable derivative).
RANGE_STATIC_THRESHOLD = 2.0     # absmax range < 2 → constant (no σ dependence)
RANGE_MAX_QUAD_THRESHOLD = 5.0   # absmax range < 5 → cap at degree 2

# Shift thresholds: layers with small or constant shift skip asymmetric quant
SHIFT_MIN_MAGNITUDE = 0.5    # abs(shift) < this → no shift entry (symmetric is fine)
SHIFT_CV_STATIC = 0.05       # CV of shift < this → constant shift (degree 0)


def select_degree(sigmas: np.ndarray, vals: np.ndarray):
    """
    Select polynomial degree using tiered thresholds.

    Returns (degree, coeffs, r2, cv).
    """
    cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0.0
    val_range = float(np.max(vals) - np.min(vals))

    # Range gate: flat layers get simple polynomials to avoid wild derivatives
    if val_range < RANGE_STATIC_THRESHOLD:
        # Nearly constant — use max value as static clipping bound
        const = float(np.max(vals))
        return 0, [const], 1.0, cv

    if val_range < RANGE_MAX_QUAD_THRESHOLD:
        # Low range — cap at quadratic for stable derivatives
        r2_q, coeffs_q = poly_r2(sigmas, vals, 2)
        return 2, [float(c) for c in coeffs_q], float(r2_q), cv

    # Tier 1: quadratic (always — CV gate removed to ensure consistent degree across groups)
    r2_q, coeffs_q = poly_r2(sigmas, vals, 2)

    # Tier 2: cubic (only if quad isn't good enough or cubic gains significantly)
    r2_c, coeffs_c = poly_r2(sigmas, vals, 3)
    cubic_gain = r2_c - r2_q

    # Tier 3: quartic
    r2_4, coeffs_4 = poly_r2(sigmas, vals, 4)
    quartic_gain = r2_4 - r2_c

    if quartic_gain > QUARTIC_R2_GAIN_THRESHOLD:
        return 4, [float(c) for c in coeffs_4], float(r2_4), cv
    if cubic_gain > CUBIC_R2_GAIN_THRESHOLD:
        return 3, [float(c) for c in coeffs_c], float(r2_c), cv
    if r2_q > QUAD_R2_THRESHOLD:
        return 2, [float(c) for c in coeffs_q], float(r2_q), cv

    # Fallback: pick the best fit across all degrees
    candidates = [
        (2, coeffs_q, r2_q),
        (3, coeffs_c, r2_c),
        (4, coeffs_4, r2_4),
    ]
    best_deg, best_coeffs, best_r2 = max(candidates, key=lambda x: x[2])
    return best_deg, [float(c) for c in best_coeffs], float(best_r2), cv


def select_shift_degree(sigmas: np.ndarray, centers: np.ndarray):
    """
    Select polynomial degree for the shift (center) trajectory.

    Returns (degree, coeffs, r2) or None if shift is negligible.
    """
    mean_abs = float(np.mean(np.abs(centers)))
    if mean_abs < SHIFT_MIN_MAGNITUDE:
        return None  # Shift too small — symmetric quant is fine

    cv = float(np.std(centers) / (mean_abs + 1e-8))
    if cv < SHIFT_CV_STATIC:
        # Shift is nearly constant across σ
        return 0, [float(np.mean(centers))], 1.0

    # Fit quadratic (usually sufficient for shift trajectories)
    r2_q, coeffs_q = poly_r2(sigmas, centers, 2)
    if r2_q > 0.85:
        return 2, [float(c) for c in coeffs_q], float(r2_q)

    # Try cubic
    r2_c, coeffs_c = poly_r2(sigmas, centers, 3)
    if r2_c - r2_q > 0.10:
        return 3, [float(c) for c in coeffs_c], float(r2_c)

    return 2, [float(c) for c in coeffs_q], float(r2_q)


# ---------------------------------------------------------------------------
# Shift trajectory loading from timestep_stats
# ---------------------------------------------------------------------------

def load_shift_trajectories(activations_dir: Path):
    """
    Load per-layer center trajectories from timestep_stats index JSONs.

    Center = (tensor_max + tensor_min) / 2 for each layer at each timestep.

    Returns dict: layer_name → (sigmas_array, centers_array)
    """
    ts_dir = activations_dir / "timestep_stats"
    if not ts_dir.exists():
        return {}

    # Collect per-step data
    step_data = {}  # step_idx → {layer_name: {min, max}}
    for idx_path in sorted(ts_dir.glob("step_*_index.json")):
        step_idx = int(idx_path.stem.split("_")[1])
        with open(idx_path) as f:
            index = json.load(f)
        step_data[step_idx] = index

    if not step_data:
        return {}

    # Also need sigmas — try to get from the index files or from the parent
    # The sigma is typically stored in the activations dir's layer_statistics.json
    sigma_map = {}
    stats_path = activations_dir / "layer_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        sigma_map = stats.get("sigma_map", {})

    # Build per-layer trajectories
    all_layers = set()
    for step_info in step_data.values():
        all_layers.update(step_info.keys())

    trajectories = {}
    for layer in sorted(all_layers):
        sigmas = []
        centers = []
        for step_idx in sorted(step_data.keys()):
            layer_info = step_data[step_idx].get(layer)
            if layer_info is None:
                continue
            t_min = layer_info.get("tensor_min")
            t_max = layer_info.get("tensor_max")
            if t_min is None or t_max is None:
                continue
            center = (t_max + t_min) / 2.0
            # Get sigma for this step
            sigma = sigma_map.get(str(step_idx), 0.0)
            sigmas.append(sigma)
            centers.append(center)
        if len(sigmas) >= 3:  # Need at least 3 points for fitting
            trajectories[layer] = (np.array(sigmas), np.array(centers))

    return trajectories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_smoothquant_trajectories(activations_dir: Path, sq_scales: dict):
    """
    Derive smoothed absmax trajectories from per-channel calibration stats.

    For each layer and each timestep:
        smoothed_absmax(t) = max_c( avg_max[c, t] / s[c] )

    where s[c] is the SmoothQuant scale (so activations at inference are x/s[c]).

    Returns:
        trajs  : dict { layer_name (underscore) → (sigmas_array, absmax_array) }
        sigmas : np.ndarray of all sigma values (sorted descending, matching trajs)
    """
    ts_dir = activations_dir / "timestep_stats"
    npz_files = sorted(ts_dir.glob("step_*.npz"),
                       key=lambda p: int(p.stem.split("_")[1]))
    if not npz_files:
        raise FileNotFoundError(f"No step_*.npz files in {ts_dir}")

    # Load sigma map
    sigma_map = {}
    stats_path = activations_dir / "layer_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        sigma_map = {int(k): float(v) for k, v in stats.get("sigma_map", {}).items()}

    layer_scales = sq_scales.get("layers", {})  # calib_name → list[float]

    # Collect per-layer timeseries: layer → [(sigma, smoothed_absmax), ...]
    timeseries = {}  # layer_name → list of (sigma, absmax)

    for npz_path in npz_files:
        step_idx = int(npz_path.stem.split("_")[1])
        sigma = sigma_map.get(step_idx)
        if sigma is None:
            continue

        data = np.load(npz_path)
        # Group keys by layer
        layer_keys = {}
        for key in data.files:
            if key.endswith("__avg_max"):
                lname = key[:-9]
                layer_keys[lname] = key

        for lname, avg_max_key in layer_keys.items():
            avg_max = data[avg_max_key].astype(np.float32)  # (C,)

            if lname in layer_scales:
                s = np.array(layer_scales[lname], dtype=np.float32)
                s = np.maximum(s, 1e-5)
                if s.shape == avg_max.shape:
                    smoothed = avg_max / s
                    absmax = float(np.max(smoothed))
                else:
                    absmax = float(np.max(avg_max))
            else:
                # No SQ scale for this layer — use raw absmax
                absmax = float(np.max(avg_max))

            if lname not in timeseries:
                timeseries[lname] = []
            timeseries[lname].append((sigma, absmax))

    # Sort by sigma descending (matching load_percentile_trajectories convention)
    trajs = {}
    all_sigmas = set()
    for lname, pts in timeseries.items():
        pts.sort(key=lambda x: -x[0])  # descending sigma
        sigmas_arr = np.array([p[0] for p in pts])
        vals_arr = np.array([p[1] for p in pts])
        trajs[lname] = (sigmas_arr, vals_arr)
        all_sigmas.update(sigmas_arr.tolist())

    sigmas_global = np.array(sorted(all_sigmas, reverse=True))
    return trajs, sigmas_global


def generate_schedule(activations_dir: Path, percentile: str = "p100_absmax",
                      include_shifts: bool = False, max_degree: int = 4,
                      smoothquant_scales: dict = None):
    """Generate polynomial clipping schedule from activation trajectories.

    If include_shifts=True, also fits shift (center) trajectories from
    timestep_stats for asymmetric activation quantization (Module C).

    If max_degree=0, all layers are forced to degree 0 (static = max absmax),
    producing a timestep-independent activation schedule.

    If smoothquant_scales is provided, derives smoothed absmax trajectories from
    per-channel timestep stats instead of using pre-aggregated percentile trajectories.
    """
    if smoothquant_scales is not None:
        print("  Using SmoothQuant-smoothed trajectories from per-channel stats...")
        trajs, sigmas = load_smoothquant_trajectories(activations_dir, smoothquant_scales)
        if not trajs:
            raise ValueError("No smoothed trajectories derived — check activations_dir and scales")
    else:
        pct_trajectories, sigmas = load_percentile_trajectories(activations_dir)
        trajs = pct_trajectories.get(percentile, {})

    if not trajs:
        if smoothquant_scales is not None:
            raise ValueError("No smoothed trajectories derived — check activations_dir and scales")
        raise ValueError(f"No trajectories found for percentile '{percentile}'")

    # Load shift trajectories if requested
    shift_trajs = {}
    if include_shifts:
        shift_trajs = load_shift_trajectories(activations_dir)

    layers = {}
    n_shifts = 0
    for layer_raw, (layer_sigmas, vals) in sorted(trajs.items()):
        if max_degree == 0:
            cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0.0
            degree, coeffs, r2, cv = 0, [float(np.max(vals))], 1.0, cv
        else:
            degree, coeffs, r2, cv = select_degree(layer_sigmas, vals)
            if degree > max_degree:
                # Re-fit capped at max_degree
                r2_cap, coeffs_cap = poly_r2(layer_sigmas, vals, max_degree)
                degree, coeffs, r2 = max_degree, [float(c) for c in coeffs_cap], float(r2_cap)
        entry = {
            "degree": degree,
            "coeffs": coeffs,
            "r2": round(r2, 4),
            "cv": round(cv, 4),
        }

        # Add shift coefficients if available
        if layer_raw in shift_trajs:
            shift_sigmas, shift_centers = shift_trajs[layer_raw]
            result = select_shift_degree(shift_sigmas, shift_centers)
            if result is not None:
                s_deg, s_coeffs, s_r2 = result
                entry["shift_degree"] = s_deg
                entry["shift_coeffs"] = s_coeffs
                entry["shift_r2"] = round(s_r2, 4)
                n_shifts += 1

        layers[layer_raw] = entry

    if smoothquant_scales is not None:
        version = "poly_v2_smoothquant" if include_shifts else "poly_v1_smoothquant"
    else:
        version = "poly_v2" if include_shifts else "poly_v1"

    schedule = {
        "version": version,
        "percentile": percentile if smoothquant_scales is None else "smoothquant_absmax",
        "max_degree": max_degree,
        "sigma_range": [float(sigmas.min()), float(sigmas.max())],
        "layers": layers,
    }
    if include_shifts:
        schedule["n_shift_layers"] = n_shifts
    if smoothquant_scales is not None:
        schedule["smoothquant_alpha"] = smoothquant_scales.get("alpha")
    return schedule


def print_summary(schedule: dict):
    """Print summary table of degree distribution."""
    layers = schedule["layers"]
    degree_counts = {}
    total_coeffs = 0

    for info in layers.values():
        d = info["degree"]
        degree_counts[d] = degree_counts.get(d, 0) + 1
        total_coeffs += len(info["coeffs"])

    print("\n=== Polynomial Clipping Schedule Summary ===")
    print(f"  Percentile: {schedule['percentile']}")
    print(f"  Total layers: {len(layers)}")
    print(f"\n  Degree distribution:")
    for d in sorted(degree_counts):
        label = "static" if d == 0 else f"degree {d}"
        print(f"    {label:>10}: {degree_counts[d]:>4} layers")
    print(f"\n  Total coefficients: {total_coeffs}")

    # Estimate JSON size
    json_str = json.dumps({"layers": layers})
    print(f"  Estimated JSON size: {len(json_str) / 1024:.1f} KB")

    # R² stats by degree
    print(f"\n  R² stats by degree:")
    for d in sorted(degree_counts):
        r2s = [info["r2"] for info in layers.values() if info["degree"] == d]
        if r2s:
            label = "static" if d == 0 else f"deg {d}"
            print(f"    {label:>8}: median={np.median(r2s):.3f}  "
                  f"min={np.min(r2s):.3f}  mean={np.mean(r2s):.3f}")

    # Shift stats (Module C)
    n_shift = schedule.get("n_shift_layers", 0)
    if n_shift > 0:
        shift_layers = {k: v for k, v in layers.items() if "shift_coeffs" in v}
        print(f"\n  Shift trajectories: {n_shift} layers with asymmetric shift")
        shift_degrees = {}
        for info in shift_layers.values():
            sd = info["shift_degree"]
            shift_degrees[sd] = shift_degrees.get(sd, 0) + 1
        for d in sorted(shift_degrees):
            label = "static" if d == 0 else f"deg {d}"
            print(f"    {label:>8}: {shift_degrees[d]:>4} layers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-dir", type=Path,
                        default=Path("calibration_data_100/activations"))
    parser.add_argument("--output", type=Path,
                        default=Path("polynomial_clipping_schedule.json"))
    parser.add_argument("--percentile", default="p100_absmax",
                        choices=["p99", "p999", "mean_absmax", "p100_absmax"])
    parser.add_argument("--include-shifts", action="store_true",
                        help="Fit shift (center) trajectories for asymmetric activation quant")
    parser.add_argument("--max-degree", type=int, default=4,
                        help="Maximum polynomial degree (0 = static/constant for all layers)")
    parser.add_argument("--smoothquant-scales", type=Path, default=None,
                        help="SmoothQuant scales JSON (from compute_smoothquant_scales.py). "
                             "When set, derives smoothed absmax trajectories from per-channel "
                             "timestep stats instead of pre-aggregated percentile trajectories.")
    args = parser.parse_args()

    sq_scales = None
    if args.smoothquant_scales is not None:
        print(f"Loading SmoothQuant scales from {args.smoothquant_scales}...")
        with open(args.smoothquant_scales) as f:
            sq_scales = json.load(f)
        print(f"  alpha={sq_scales.get('alpha')}  layers={sq_scales.get('n_layers')}")

    print(f"Loading activation trajectories from {args.activations_dir}...")
    schedule = generate_schedule(args.activations_dir, args.percentile,
                                 include_shifts=args.include_shifts,
                                 max_degree=args.max_degree,
                                 smoothquant_scales=sq_scales)

    with open(args.output, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"Wrote {args.output}")

    print_summary(schedule)


if __name__ == "__main__":
    main()
