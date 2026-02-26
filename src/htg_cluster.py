"""
HTG (Hierarchical Timestep Grouping) clustering for SD3-Medium DiT.

Partitions the diffusion trajectory into G groups with similar per-layer
activation statistics using adjacency-constrained agglomerative clustering
on per-channel shift vectors (arXiv 2503.06930 Algorithm 1).

Algorithm overview (per layer ℓ):
  1. For each selected timestep t, compute shift vector z_t_ℓ[c] = (avg_max[c] + avg_min[c]) / 2
  2. Agglomerative clustering with adjacency constraint: only adjacent pairs merge;
     distance = L2 between group centroids (average-linkage).
  3. Merge until G groups remain → per-layer partition boundaries.
  4. For each group g, compute z̄_g_ℓ = mean(z_t_ℓ for t in group g).

Global consensus partition: derive shared timestep groups from per-layer boundaries
(median boundary index across all layers) for use by Stages 1 and 4+5.

Usage:
    conda run -n diffusionkit python -m src.htg_cluster \\
        --stats calibration_data_100/activations/layer_statistics.json \\
        --output htg_groups.json \\
        [--n-groups 5]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analyze_activations import load_stats_v2


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------

def adjacent_agglomerative(z: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Adjacency-constrained agglomerative clustering on shift vectors.

    Parameters
    ----------
    z : np.ndarray, shape (T, D)
        Shift vectors ordered by timestep index (D = n_channels for one layer).
    n_groups : int
        Desired number of groups (must be >= 1 and <= T).

    Returns
    -------
    assignments : np.ndarray, shape (T,)
        Integer group labels 0..n_groups-1, contiguous and monotone
        (all timesteps in group g precede group g+1).
    """
    T = len(z)
    if T == 0:
        return np.zeros(0, dtype=int)
    n_groups = max(1, min(n_groups, T))

    # Start: each timestep is its own cluster (list of index lists)
    clusters: List[List[int]] = [[i] for i in range(T)]

    while len(clusters) > n_groups:
        # Find adjacent pair with minimum centroid L2 distance
        best_i, best_dist = 0, float("inf")
        for i in range(len(clusters) - 1):
            c1 = z[clusters[i]].mean(axis=0)
            c2 = z[clusters[i + 1]].mean(axis=0)
            d = float(np.linalg.norm(c1 - c2))
            if d < best_dist:
                best_dist = d
                best_i = i
        # Merge best_i and best_i+1
        clusters[best_i] = clusters[best_i] + clusters[best_i + 1]
        del clusters[best_i + 1]

    assignments = np.zeros(T, dtype=int)
    for g, indices in enumerate(clusters):
        for idx in indices:
            assignments[idx] = g
    return assignments


def _per_layer_boundaries(assignments: np.ndarray) -> List[int]:
    """
    Extract boundary positions (first index of each new group after group 0).

    For assignments [0,0,1,1,2] → boundaries [2, 4] (positions where group changes).
    """
    boundaries = []
    for i in range(1, len(assignments)):
        if assignments[i] != assignments[i - 1]:
            boundaries.append(i)
    return boundaries


def derive_consensus_partition(
    per_layer_assignments: Dict[str, np.ndarray],
    n_groups: int,
    T: int,
) -> np.ndarray:
    """
    Derive a global consensus timestep partition from per-layer group assignments.

    For each boundary slot (G-1 boundaries total), take the median boundary
    position across all layers and convert to global group assignments.

    Parameters
    ----------
    per_layer_assignments : dict layer_name -> assignments (T,)
    n_groups : int
    T : int — number of timesteps

    Returns
    -------
    global_assignments : np.ndarray shape (T,) with values 0..n_groups-1
    """
    if n_groups == 1 or T <= 1:
        return np.zeros(T, dtype=int)

    # Collect per-layer boundary lists (each has n_groups-1 entries)
    all_boundaries = []
    for layer, assgn in per_layer_assignments.items():
        bounds = _per_layer_boundaries(assgn)
        # Some degenerate layers may produce fewer boundaries; pad to n_groups-1
        while len(bounds) < n_groups - 1:
            # Fill missing boundaries by splitting the last span
            if bounds:
                bounds.append(min(T - 1, bounds[-1] + 1))
            else:
                bounds.append(T // n_groups)
        # Truncate to exactly n_groups-1
        bounds = bounds[: n_groups - 1]
        all_boundaries.append(bounds)

    if not all_boundaries:
        # Fallback: uniform partition
        return np.array([min(g, n_groups - 1) for g in
                         np.arange(T) * n_groups // T], dtype=int)

    # Median boundary per slot
    arr = np.array(all_boundaries, dtype=float)  # (n_layers, n_groups-1)
    consensus = np.round(np.median(arr, axis=0)).astype(int)
    # Ensure monotone and within [1, T-1]
    for i in range(len(consensus)):
        lo = consensus[i - 1] + 1 if i > 0 else 1
        consensus[i] = max(lo, min(T - 1, consensus[i]))

    # Build assignments from consensus boundary positions
    global_assgn = np.zeros(T, dtype=int)
    for g, boundary in enumerate(consensus):
        global_assgn[boundary:] = g + 1
    return global_assgn


def compute_per_layer_z_bar(
    per_step_full: Dict,
    step_keys: List[str],
    layer_names: List[str],
    per_layer_assignments: Dict[str, np.ndarray],
    global_assignments: np.ndarray,
    n_groups: int,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute averaged shift vectors per layer per group (z̄_g_ℓ).

    For layers with per-layer boundaries, use the per-layer assignment.
    Returns dict: layer_name -> group_id (str) -> shift vector (list of floats).
    """
    z_bar: Dict[str, Dict[str, List[float]]] = {}

    for layer_name in layer_names:
        layer_assgn = per_layer_assignments.get(layer_name, global_assignments)
        z_bar[layer_name] = {}

        for g in range(n_groups):
            group_step_mask = (layer_assgn == g)
            group_step_keys = [sk for sk, flag in zip(step_keys, group_step_mask) if flag]

            if not group_step_keys:
                continue

            # Collect shift vectors for this group
            shifts = []
            for sk in group_step_keys:
                data = per_step_full.get(sk, {}).get(layer_name, {})
                avg_min = data.get("avg_min")
                avg_max = data.get("avg_max")
                if avg_min is not None and avg_max is not None:
                    z_t = (avg_max + avg_min) / 2.0
                    shifts.append(z_t)

            if shifts:
                z_bar[layer_name][str(g)] = np.mean(shifts, axis=0).tolist()

    return z_bar


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_htg_groups(
    stats_path: Path,
    n_groups: int = 5,
) -> Dict:
    """
    Run HTG clustering and return the htg_groups dict.

    Parameters
    ----------
    stats_path : Path
        Path to layer_statistics.json from collect_layer_activations.py.
    n_groups : int
        Desired number of timestep groups.

    Returns
    -------
    htg_groups : dict with keys global_groups, per_layer_z_bar, sigma_map, n_groups.
    """
    print(f"Loading {stats_path}")
    timesteps, per_step_full, layer_names, metadata, sigma_map = load_stats_v2(stats_path)

    step_keys = sorted(timesteps.keys(), key=int)
    T = len(step_keys)
    n_groups = max(1, min(n_groups, T))

    print(f"Timesteps: {T}   Layers: {len(layer_names)}   Target groups: {n_groups}")

    # ------------------------------------------------------------------
    # Per-layer clustering
    # ------------------------------------------------------------------
    per_layer_assignments: Dict[str, np.ndarray] = {}

    for layer_name in layer_names:
        # Build (T, C_in) shift matrix for this layer
        z_rows = []
        for sk in step_keys:
            data = per_step_full.get(sk, {}).get(layer_name, {})
            avg_min = data.get("avg_min")
            avg_max = data.get("avg_max")
            if avg_min is not None and avg_max is not None:
                z_t = (avg_max + avg_min) / 2.0
                z_rows.append(z_t)
            else:
                # Missing data: use zero vector
                # Try to infer dimension from another timestep
                dim = 1
                for sk2 in step_keys:
                    d2 = per_step_full.get(sk2, {}).get(layer_name, {})
                    mn = d2.get("avg_min")
                    if mn is not None:
                        dim = len(mn)
                        break
                z_rows.append(np.zeros(dim, dtype=np.float32))

        if not z_rows:
            continue

        z_layer = np.stack(z_rows, axis=0)  # (T, C_in)
        assignments = adjacent_agglomerative(z_layer, n_groups)
        per_layer_assignments[layer_name] = assignments

    # ------------------------------------------------------------------
    # Global consensus partition
    # ------------------------------------------------------------------
    global_assignments = derive_consensus_partition(
        per_layer_assignments, n_groups, T
    )

    # ------------------------------------------------------------------
    # Build global_groups dict with timestep indices and sigma ranges
    # ------------------------------------------------------------------
    global_groups: Dict[str, Dict] = {}
    for g in range(n_groups):
        group_indices = [i for i, a in enumerate(global_assignments) if a == g]
        if not group_indices:
            continue
        group_step_keys = [step_keys[i] for i in group_indices]
        group_sigmas = [sigma_map.get(int(sk), float("nan")) for sk in group_step_keys]
        valid_sigmas = [s for s in group_sigmas if not (s != s)]  # filter NaN
        global_groups[str(g)] = {
            "timestep_indices": group_indices,
            "step_keys": group_step_keys,
            "sigma_range": [
                float(min(valid_sigmas)) if valid_sigmas else 0.0,
                float(max(valid_sigmas)) if valid_sigmas else 1.0,
            ],
        }

    print(f"\nGlobal consensus partition ({n_groups} groups):")
    for g_id, info in global_groups.items():
        print(f"  Group {g_id}: {len(info['timestep_indices'])} timesteps  "
              f"sigma=[{info['sigma_range'][0]:.3f}, {info['sigma_range'][1]:.3f}]")

    # ------------------------------------------------------------------
    # Compute per-layer z̄_g (averaged shift vectors)
    # ------------------------------------------------------------------
    print("\nComputing per-layer averaged shift vectors...")
    per_layer_z_bar = compute_per_layer_z_bar(
        per_step_full, step_keys, layer_names,
        per_layer_assignments, global_assignments, n_groups,
    )

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    htg_groups = {
        "format": "htg_groups_v1",
        "n_groups": n_groups,
        "n_timesteps": T,
        "global_groups": global_groups,
        "per_layer_z_bar": per_layer_z_bar,
        "sigma_map": {str(k): v for k, v in sigma_map.items()},
        "step_keys": step_keys,
        "metadata": metadata,
    }

    n_layers_with_zbar = sum(1 for v in per_layer_z_bar.values() if v)
    print(f"\n{n_layers_with_zbar}/{len(layer_names)} layers have z_bar computed")

    return htg_groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTG: Hierarchical Timestep Grouping for SD3-Medium DiT"
    )
    parser.add_argument("--stats", type=Path, required=True,
                        help="layer_statistics.json from collect_layer_activations.py")
    parser.add_argument("--output", type=Path, default=Path("htg_groups.json"),
                        help="Output htg_groups.json path (default: htg_groups.json)")
    parser.add_argument("--n-groups", type=int, default=5,
                        help="Number of timestep groups (default: 5)")
    args = parser.parse_args()

    htg_groups = build_htg_groups(args.stats, n_groups=args.n_groups)

    with open(args.output, "w") as f:
        json.dump(htg_groups, f, indent=2)
    print(f"\n✓ HTG groups -> {args.output}")


if __name__ == "__main__":
    main()
