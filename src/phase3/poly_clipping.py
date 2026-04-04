"""Polynomial clipping schedule generation for timestep-aware A8 quantization.

Fits tiered polynomials to per-layer post-CSB absmax activation trajectories
across σ steps, producing a compact JSON schedule.  At inference the schedule
replaces dynamic max-reduction (or a single static scale) with a cheap
polynomial evaluation: ``α(σ) = poly(σ)``.

Two granularities are supported:

- **Per-tensor** (default): one polynomial per layer → scalar α(σ).
- **Per-channel** (for high-ρ layers): one polynomial per input channel
  → vector α(σ) of shape ``[d_in]``.  Selected when the layer's mean
  Spearman ρ exceeds ``per_channel_rho_threshold``.

Degree selection tiers
----------------------
- absmax range < 2 across σ → degree 0 (constant = max absmax)
- absmax range < 5           → degree 2 max (stable derivatives)
- Quadratic R² > 0.85       → degree 2
- Cubic R² gain > 0.15      → degree 3
- Quartic R² gain > 0.10    → degree 4 (max)

Data sources
------------
- Phase 1 diagnostics: ``diagnostics/activation_stats/{layer}.npz``
  containing ``act_channel_max`` of shape ``[T, d_in]`` and
  ``sigma_values`` of shape ``[T]``.
- Phase 2 calibration: ``calibration.npz`` with per-layer balancing
  vectors ``b`` from CSB, and ``calibration_meta.json`` with
  ``mean_rhos`` from SSC.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Polynomial fitting primitive
# ---------------------------------------------------------------------------

POLY_SCHEDULE_FILENAME = "poly_schedule.json"


def poly_r2(
    sigmas: np.ndarray, vals: np.ndarray, degree: int
) -> tuple[float, np.ndarray]:
    """Fit a polynomial and return (R², coefficients in ascending-power order).

    Parameters
    ----------
    sigmas : array, shape [T]
    vals   : array, shape [T]
    degree : int

    Returns
    -------
    r2     : float — coefficient of determination
    coeffs : ndarray, shape [degree+1] — ascending-power ``[c0, c1, …, c_d]``
    """
    if len(sigmas) <= degree:
        return 0.0, np.zeros(degree + 1)
    # np.polyfit returns highest-degree-first; reverse for ascending order
    p = np.polyfit(sigmas, vals, degree)
    coeffs = p[::-1]
    predicted = np.polyval(p, sigmas)
    ss_res = float(np.sum((vals - predicted) ** 2))
    ss_tot = float(np.sum((vals - np.mean(vals)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return r2, coeffs


# ---------------------------------------------------------------------------
# Degree selection thresholds
# ---------------------------------------------------------------------------

QUAD_R2_THRESHOLD = 0.85
CUBIC_R2_GAIN_THRESHOLD = 0.15
QUARTIC_R2_GAIN_THRESHOLD = 0.10

RANGE_STATIC_THRESHOLD = 2.0
RANGE_MAX_QUAD_THRESHOLD = 5.0

SHIFT_MIN_MAGNITUDE = 0.5
SHIFT_CV_STATIC = 0.05


def select_degree(
    sigmas: np.ndarray, vals: np.ndarray
) -> tuple[int, list[float], float, float]:
    """Select polynomial degree using tiered R² thresholds.

    Returns ``(degree, coeffs, r2, cv)``.
    """
    cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0.0
    val_range = float(np.max(vals) - np.min(vals))

    if val_range < RANGE_STATIC_THRESHOLD:
        const = float(np.max(vals))
        return 0, [const], 1.0, cv

    if val_range < RANGE_MAX_QUAD_THRESHOLD:
        r2_q, coeffs_q = poly_r2(sigmas, vals, 2)
        return 2, [float(c) for c in coeffs_q], float(r2_q), cv

    r2_q, coeffs_q = poly_r2(sigmas, vals, 2)
    r2_c, coeffs_c = poly_r2(sigmas, vals, 3)
    cubic_gain = r2_c - r2_q
    r2_4, coeffs_4 = poly_r2(sigmas, vals, 4)
    quartic_gain = r2_4 - r2_c

    if quartic_gain > QUARTIC_R2_GAIN_THRESHOLD:
        return 4, [float(c) for c in coeffs_4], float(r2_4), cv
    if cubic_gain > CUBIC_R2_GAIN_THRESHOLD:
        return 3, [float(c) for c in coeffs_c], float(r2_c), cv
    if r2_q > QUAD_R2_THRESHOLD:
        return 2, [float(c) for c in coeffs_q], float(r2_q), cv

    candidates = [
        (2, coeffs_q, r2_q),
        (3, coeffs_c, r2_c),
        (4, coeffs_4, r2_4),
    ]
    best_deg, best_coeffs, best_r2 = max(candidates, key=lambda x: x[2])
    return best_deg, [float(c) for c in best_coeffs], float(best_r2), cv


def select_shift_degree(
    sigmas: np.ndarray, centers: np.ndarray
) -> tuple[int, list[float], float] | None:
    """Select polynomial degree for the shift (center) trajectory.

    Returns ``(degree, coeffs, r2)`` or ``None`` if shift is negligible.
    """
    mean_abs = float(np.mean(np.abs(centers)))
    if mean_abs < SHIFT_MIN_MAGNITUDE:
        return None

    cv = float(np.std(centers) / (mean_abs + 1e-8))
    if cv < SHIFT_CV_STATIC:
        return 0, [float(np.mean(centers))], 1.0

    r2_q, coeffs_q = poly_r2(sigmas, centers, 2)
    if r2_q > 0.85:
        return 2, [float(c) for c in coeffs_q], float(r2_q)

    r2_c, coeffs_c = poly_r2(sigmas, centers, 3)
    if r2_c - r2_q > 0.10:
        return 3, [float(c) for c in coeffs_c], float(r2_c)

    return 2, [float(c) for c in coeffs_q], float(r2_q)


# ---------------------------------------------------------------------------
# Per-channel polynomial fitting (vectorised)
# ---------------------------------------------------------------------------

def _fit_per_channel_polynomials(
    sigmas: np.ndarray, post_csb: np.ndarray, degree: int
) -> tuple[list[list[float]], float]:
    """Fit one polynomial per input channel via least-squares.

    Parameters
    ----------
    sigmas : [T]
    post_csb : [T, d_in] — post-CSB per-channel absmax trajectories.
    degree : int — uniform degree for all channels.

    Returns
    -------
    coeffs : list[list[float]], shape ``[d_in][degree+1]`` — ascending power.
    mean_r2 : float — mean R² across channels.
    """
    T, d_in = post_csb.shape

    if degree == 0:
        # Conservative: per-channel max (not mean from lstsq)
        max_vals = post_csb.max(axis=0)  # [d_in]
        return [[round(float(v), 6)] for v in max_vals], 1.0

    if T <= degree:
        coeffs = np.zeros((d_in, degree + 1))
        coeffs[:, 0] = post_csb.max(axis=0)
        return [[round(float(c), 6) for c in row] for row in coeffs], 1.0

    # Vandermonde [T, degree+1] in ascending power: [1, σ, σ², …]
    V = np.vander(sigmas, degree + 1, increasing=True)

    # Solve V @ C = post_csb  →  C shape [degree+1, d_in]
    C, _, _, _ = np.linalg.lstsq(V, post_csb, rcond=None)

    predicted = V @ C  # [T, d_in]
    ss_res = np.sum((post_csb - predicted) ** 2, axis=0)  # [d_in]
    mean_vals = post_csb.mean(axis=0, keepdims=True)
    ss_tot = np.sum((post_csb - mean_vals) ** 2, axis=0)  # [d_in]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_per_ch = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 1.0)
    mean_r2 = float(np.mean(r2_per_ch))

    # C is [degree+1, d_in] → transpose to [d_in, degree+1]
    coeffs = C.T
    return [[round(float(c), 6) for c in row] for row in coeffs], mean_r2


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_sigma_values(diagnostics_dir: Path) -> np.ndarray:
    """Load the sigma schedule from any per-layer npz in ``activation_stats/``.

    Each npz produced by Phase 1 stores its own ``sigma_values`` array.
    They are identical across layers (same denoising run), so we just
    read the first available file.
    """
    act_dir = diagnostics_dir / "activation_stats"
    if not act_dir.exists():
        raise FileNotFoundError(f"Missing {act_dir}")
    for npz_path in sorted(act_dir.glob("*.npz")):
        data = np.load(npz_path)
        if "sigma_values" in data:
            return data["sigma_values"].astype(np.float64)
    raise FileNotFoundError(
        f"No npz file with 'sigma_values' found in {act_dir}"
    )


def _load_calibration_data(
    calibration_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Load balancing vectors and mean Spearman ρ from Phase 2 calibration.

    Returns ``(b_vectors, mean_rhos)``.
    """
    cal_path = calibration_dir / "calibration.npz"
    meta_path = calibration_dir / "calibration_meta.json"
    if not cal_path.exists():
        raise FileNotFoundError(f"Missing {cal_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)
    layer_names: list[str] = meta["layer_names"]

    raw = np.load(cal_path)
    b_vectors: dict[str, np.ndarray] = {}
    for name in layer_names:
        if name in raw:
            b_vectors[name] = raw[name]
        else:
            logger.warning("No balancing vector for %s in calibration.npz", name)

    mean_rhos: dict[str, float] = {
        k: float(v) for k, v in meta.get("mean_rhos", {}).items()
    }
    return b_vectors, mean_rhos


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------

def generate_schedule_from_diagnostics(
    diagnostics_dir: Path,
    calibration_dir: Path,
    *,
    max_degree: int = 4,
    include_shifts: bool = False,
    exclude_layers: list[str] | None = None,
    per_channel_rho_threshold: float | None = None,
) -> dict:
    """Build a polynomial clipping schedule from Phase 1/2 artifacts.

    For each layer:
      1. Load ``act_channel_max[T, d_in]`` from diagnostics.
      2. Load balancing vector ``b`` from Phase 2 calibration.
      3. Compute post-CSB absmax trajectory.
      4. If layer ρ > *per_channel_rho_threshold*: fit **per-channel** polys.
         Otherwise: fit **per-tensor** polynomial via :func:`select_degree`.

    Parameters
    ----------
    diagnostics_dir : Path
        Phase 1 output (``diagnostics/``).
    calibration_dir : Path
        Phase 2 quantized output containing ``calibration.npz`` and
        ``calibration_meta.json``.
    max_degree : int
        Cap polynomial degree (0 = all-static).
    include_shifts : bool
        (Future) Fit shift polynomials — currently ignored.
    exclude_layers : list[str] or None
        Layer names to skip (e.g. ``["context_embedder"]``).
    per_channel_rho_threshold : float or None
        Layers whose mean Spearman ρ exceeds this threshold use
        per-channel polynomial fitting.  ``None`` disables per-channel.

    Returns
    -------
    dict — the full schedule (ready for JSON serialisation).
    """
    if include_shifts:
        logger.warning(
            "include_shifts requested but Phase 1 data is unsigned "
            "(max|X| only) — signed statistics needed for meaningful "
            "shift polynomials. Ignoring --include-shifts."
        )
        include_shifts = False

    sigmas = _load_sigma_values(diagnostics_dir)
    b_vectors, mean_rhos = _load_calibration_data(calibration_dir)
    act_dir = diagnostics_dir / "activation_stats"
    if not act_dir.exists():
        raise FileNotFoundError(f"Missing {act_dir}")

    excluded = set(exclude_layers or [])

    layers: dict[str, dict] = {}
    n_per_channel = 0

    for name, b in sorted(b_vectors.items()):
        if name in excluded:
            continue
        npz_path = act_dir / f"{name}.npz"
        if not npz_path.exists():
            logger.warning("No activation stats for %s — skipping", name)
            continue

        act_traj = np.load(npz_path)["act_channel_max"]  # [T, d_in]
        if act_traj.shape[0] != len(sigmas):
            logger.warning(
                "%s: T mismatch (act=%d, sigmas=%d) — skipping",
                name, act_traj.shape[0], len(sigmas),
            )
            continue

        # Post-CSB per-channel trajectories
        b_safe = np.maximum(b, 1e-12)
        post_csb = act_traj / b_safe[np.newaxis, :]   # [T, d_in]
        absmax_traj = post_csb.max(axis=1)             # [T] per-tensor

        # Decide granularity: per-channel for high-ρ layers
        use_per_channel = (
            per_channel_rho_threshold is not None
            and mean_rhos.get(name, 0.0) > per_channel_rho_threshold
        )

        if use_per_channel:
            entry = _build_per_channel_entry(
                sigmas, post_csb, absmax_traj, max_degree,
            )
            n_per_channel += 1
        else:
            entry = _build_per_tensor_entry(
                sigmas, absmax_traj, max_degree,
            )

        layers[name] = entry

    schedule: dict = {
        "version": "poly_v3_csb",
        "max_degree": max_degree,
        "sigma_range": [float(sigmas.min()), float(sigmas.max())],
        "n_layers": len(layers),
        "n_per_channel": n_per_channel,
        "layers": layers,
    }
    if per_channel_rho_threshold is not None:
        schedule["per_channel_rho_threshold"] = per_channel_rho_threshold

    logger.info(
        "Generated polynomial schedule: %d layers "
        "(%d per-tensor, %d per-channel), max_degree=%d",
        len(layers), len(layers) - n_per_channel, n_per_channel, max_degree,
    )
    return schedule


def _build_per_tensor_entry(
    sigmas: np.ndarray, absmax_traj: np.ndarray, max_degree: int,
) -> dict:
    """Build a per-tensor schedule entry."""
    if max_degree == 0:
        cv = float(np.std(absmax_traj) / np.mean(absmax_traj)) if np.mean(absmax_traj) > 0 else 0.0
        return {
            "degree": 0,
            "coeffs": [float(np.max(absmax_traj))],
            "r2": 1.0,
            "cv": round(cv, 4),
        }

    degree, coeffs, r2, cv = select_degree(sigmas, absmax_traj)
    if degree > max_degree:
        r2_cap, coeffs_cap = poly_r2(sigmas, absmax_traj, max_degree)
        degree = max_degree
        coeffs = [float(c) for c in coeffs_cap]
        r2 = float(r2_cap)

    return {
        "degree": degree,
        "coeffs": coeffs,
        "r2": round(r2, 4),
        "cv": round(cv, 4),
    }


def _build_per_channel_entry(
    sigmas: np.ndarray,
    post_csb: np.ndarray,
    absmax_traj: np.ndarray,
    max_degree: int,
) -> dict:
    """Build a per-channel schedule entry.

    Uses the per-tensor trajectory to select a uniform degree, then
    fits all ``d_in`` channels at that degree via vectorised lstsq.
    """
    d_in = post_csb.shape[1]
    cv = float(np.std(absmax_traj) / np.mean(absmax_traj)) if np.mean(absmax_traj) > 0 else 0.0

    if max_degree == 0:
        degree = 0
    else:
        degree, _, _, _ = select_degree(sigmas, absmax_traj)
        if degree > max_degree:
            degree = max_degree

    coeffs_2d, mean_r2 = _fit_per_channel_polynomials(
        sigmas, post_csb, degree,
    )

    return {
        "degree": degree,
        "coeffs": coeffs_2d,
        "r2": round(mean_r2, 4),
        "cv": round(cv, 4),
        "granularity": "per_channel",
        "n_channels": d_in,
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(schedule: dict) -> None:
    """Print a summary table of degree distribution and R² stats."""
    layers = schedule["layers"]

    pt_layers = {k: v for k, v in layers.items() if v.get("granularity") != "per_channel"}
    pc_layers = {k: v for k, v in layers.items() if v.get("granularity") == "per_channel"}

    print("\n=== Polynomial Clipping Schedule Summary ===")
    print(f"  Version: {schedule.get('version', '?')}")
    print(f"  Total layers: {len(layers)}  "
          f"(per-tensor: {len(pt_layers)}, per-channel: {len(pc_layers)})")

    # Per-tensor degree distribution
    if pt_layers:
        pt_deg: dict[int, int] = {}
        for info in pt_layers.values():
            d = info["degree"]
            pt_deg[d] = pt_deg.get(d, 0) + 1

        print(f"\n  Per-tensor degree distribution:")
        for d in sorted(pt_deg):
            label = "static" if d == 0 else f"degree {d}"
            print(f"    {label:>10}: {pt_deg[d]:>4} layers")

        print(f"\n  Per-tensor R² stats:")
        for d in sorted(pt_deg):
            r2s = [v["r2"] for v in pt_layers.values() if v["degree"] == d]
            label = "static" if d == 0 else f"deg {d}"
            print(
                f"    {label:>8}: median={np.median(r2s):.3f}  "
                f"min={np.min(r2s):.3f}  mean={np.mean(r2s):.3f}"
            )

    # Per-channel summary
    if pc_layers:
        pc_deg: dict[int, int] = {}
        pc_total_coeffs = 0
        for info in pc_layers.values():
            d = info["degree"]
            pc_deg[d] = pc_deg.get(d, 0) + 1
            pc_total_coeffs += info["n_channels"] * (d + 1)

        rho_thresh = schedule.get("per_channel_rho_threshold", "?")
        print(f"\n  Per-channel layers: {len(pc_layers)}  (ρ threshold: {rho_thresh})")
        for d in sorted(pc_deg):
            label = "static" if d == 0 else f"degree {d}"
            print(f"    {label:>10}: {pc_deg[d]:>4} layers")
        print(f"    coefficients: {pc_total_coeffs:>7}")

        r2s = [v["r2"] for v in pc_layers.values()]
        print(
            f"    mean R²:  median={np.median(r2s):.3f}  "
            f"min={np.min(r2s):.3f}  mean={np.mean(r2s):.3f}"
        )

    # Total schedule size
    json_str = json.dumps({"layers": layers})
    print(f"\n  Estimated JSON size: {len(json_str) / 1024:.1f} KB")
