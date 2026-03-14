"""
Generate polynomial clipping schedule for timestep-aware A8 fake quantization.

Fits tiered polynomials to per-layer p999 activation trajectories across σ steps,
outputting a compact JSON schedule that replaces static per-timestep scale lookups
with continuous polynomial evaluation.

Degree selection tiers:
  CV < 0.10           → static (single float, degree 0)
  Quadratic R² > 0.85 → degree 2
  Cubic R² gain > 0.15 over quad → degree 3
  Quartic R² gain > 0.10 over cubic → degree 4

Usage:
    conda run -n diffusionkit python -m src.generate_poly_schedule \
        --activations-dir calibration_data_100/activations \
        --output polynomial_clipping_schedule.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.explore_curve_fits import load_percentile_trajectories, poly_r2


# ---------------------------------------------------------------------------
# Degree selection
# ---------------------------------------------------------------------------

CV_STATIC_THRESHOLD = 0.10
QUAD_R2_THRESHOLD = 0.85
CUBIC_R2_GAIN_THRESHOLD = 0.15
QUARTIC_R2_GAIN_THRESHOLD = 0.10

# Shift thresholds: layers with small or constant shift skip asymmetric quant
SHIFT_MIN_MAGNITUDE = 0.5    # abs(shift) < this → no shift entry (symmetric is fine)
SHIFT_CV_STATIC = 0.05       # CV of shift < this → constant shift (degree 0)


def select_degree(sigmas: np.ndarray, vals: np.ndarray):
    """
    Select polynomial degree using tiered thresholds.

    Returns (degree, coeffs, r2, cv).
    """
    cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0.0

    # Tier 0: static — activation barely changes across σ
    if cv < CV_STATIC_THRESHOLD:
        return 0, [float(np.mean(vals))], 1.0, cv

    # Tier 1: quadratic
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

    # Fallback: use quadratic even if R² is low (better than static)
    return 2, [float(c) for c in coeffs_q], float(r2_q), cv


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

def generate_schedule(activations_dir: Path, percentile: str = "p999",
                      include_shifts: bool = False):
    """Generate polynomial clipping schedule from activation trajectories.

    If include_shifts=True, also fits shift (center) trajectories from
    timestep_stats for asymmetric activation quantization (Module C).
    """
    pct_trajectories, sigmas = load_percentile_trajectories(activations_dir)
    trajs = pct_trajectories.get(percentile, {})

    if not trajs:
        raise ValueError(f"No trajectories found for percentile '{percentile}'")

    # Load shift trajectories if requested
    shift_trajs = {}
    if include_shifts:
        shift_trajs = load_shift_trajectories(activations_dir)

    layers = {}
    n_shifts = 0
    for layer_raw, (layer_sigmas, vals) in sorted(trajs.items()):
        degree, coeffs, r2, cv = select_degree(layer_sigmas, vals)
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

    schedule = {
        "version": "poly_v2" if include_shifts else "poly_v1",
        "percentile": percentile,
        "sigma_range": [float(sigmas.min()), float(sigmas.max())],
        "layers": layers,
    }
    if include_shifts:
        schedule["n_shift_layers"] = n_shifts
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
    parser.add_argument("--percentile", default="p999",
                        choices=["p99", "p999", "mean_absmax", "p100_absmax"])
    parser.add_argument("--include-shifts", action="store_true",
                        help="Fit shift (center) trajectories for asymmetric activation quant")
    args = parser.parse_args()

    print(f"Loading activation trajectories from {args.activations_dir}...")
    schedule = generate_schedule(args.activations_dir, args.percentile,
                                 include_shifts=args.include_shifts)

    with open(args.output, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"Wrote {args.output}")

    print_summary(schedule)


if __name__ == "__main__":
    main()
