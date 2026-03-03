"""
Phase 3b: Compute HTG quantization parameters from input activation statistics.

Implements the two core algorithms from arXiv:2503.06930:

    Algorithm 1  Constrained hierarchical clustering
        Divides T denoising timesteps into G groups where every group's
        timesteps are contiguous, using shifting vectors z_t as features.

    HTG Algorithm (Alg. 2, steps 1-5):
        1. Compute channel-wise shifting vectors z_t  (Eq. 2)
        2. Partition timesteps into G groups           (Alg. 1)
        3. Compute per-group averaged shifts z_g       (Eq. 4)
        4. Compute EMA channel-wise scaling s           (Eq. 7)
        5. Rescale weight column max for s formula

All computation is pure NumPy — no model forward passes required.
Model weights are loaded read-only once to get max column magnitudes for s.

Output .npz schema (per layer, keyed by full layer ID):
    {layer_id}::z_g               float32  (G, D)   per-group shift vectors
    {layer_id}::z_t               float32  (T, D)   per-timestep shift vectors
    {layer_id}::s                 float32  (D,)     channel-wise scaling vector
    {layer_id}::group_assignments int32    (T,)     group index for each timestep
    {layer_id}::group_boundaries  int32    (G-1,)   partition boundary indices

Global:
    timesteps_sorted              float32  (T,)     timesteps in denoising order
    num_groups                    int32    scalar
    ema_alpha                     float32  scalar

CLI:
    python -m src.htg_quantization.compute_htg_params \\
        --input-stats htg_input_activation_stats.npz \\
        --model-version argmaxinc/mlx-stable-diffusion-3-medium \\
        --output htg_params.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .htg_config import (
    NUM_GROUPS,
    EMA_ALPHA,
    MODEL_VERSION,
    DEFAULT_INPUT_STATS_FILE,
    DEFAULT_HTG_PARAMS_FILE,
)

_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Load input activation stats
# ---------------------------------------------------------------------------

def load_input_stats(path: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    """
    Load the .npz produced by profile_input_activations.py.

    Returns
    -------
    stats : {full_layer_id: {t_key: {"min": (D,), "max": (D,)}}}
    timesteps_sorted : (T,) float32 array, sorted in denoising order (high→low)
    """
    data = np.load(path, allow_pickle=True)

    # Collect all unique layer IDs
    layer_ids_arr = data.get("layer_ids", np.array([], dtype=object))
    layer_ids: List[str] = list(layer_ids_arr)

    # Parse flat key format "{layer_id}::t={t_key}::{stat_name}"
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for key in data.files:
        if key in ("timesteps_unique", "layer_ids"):
            continue
        parts = key.split("::")
        if len(parts) != 3:
            continue
        full_layer_id, t_part, stat_name = parts
        t_key = t_part[2:]  # strip "t="

        layer_stats = stats.setdefault(full_layer_id, {})
        t_stats = layer_stats.setdefault(t_key, {})
        t_stats[stat_name] = data[key]

    # Timesteps in denoising order: SD3 Euler goes high→low, so sort descending
    ts_unique = data["timesteps_unique"].astype(np.float64)
    timesteps_sorted = np.sort(ts_unique)[::-1].copy()  # high → low

    return stats, timesteps_sorted


# ---------------------------------------------------------------------------
# Step A: Shifting vectors z_t
# ---------------------------------------------------------------------------

def compute_z_t(
    stats: Dict[str, Dict[str, np.ndarray]],
    timesteps_sorted: np.ndarray,
) -> np.ndarray:
    """
    Compute the channel-wise shifting vector for a single layer.

    z_t[i] = (max_t[i] + min_t[i]) / 2   (Eq. 2 of the paper)

    Parameters
    ----------
    stats : {t_key: {"min": (D,), "max": (D,)}} for one layer
    timesteps_sorted : (T,) timesteps in denoising order

    Returns
    -------
    z_t_matrix : (T, D) float64 — one row per timestep, in denoising order.
                 Rows for missing timesteps are zeros (should not occur if
                 calibration covers all selected timesteps).
    """
    # Build t_key → index mapping
    def _t_key(val: float) -> str:
        return f"{val:.6f}"

    T = len(timesteps_sorted)
    # Infer D from first available entry
    sample_entry = next(iter(stats.values()))
    D = sample_entry["min"].shape[0]

    z_t = np.zeros((T, D), dtype=np.float64)
    for i, t_val in enumerate(timesteps_sorted):
        key = _t_key(float(t_val))
        entry = stats.get(key)
        if entry is not None:
            z_t[i] = (entry["max"].astype(np.float64) + entry["min"].astype(np.float64)) / 2.0

    return z_t


# ---------------------------------------------------------------------------
# Step B: Constrained hierarchical clustering (Algorithm 1)
# ---------------------------------------------------------------------------

def constrained_hierarchical_clustering(
    z_t_matrix: np.ndarray,
    num_groups: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Partition T timesteps into num_groups contiguous groups.

    Implements Algorithm 1 from arXiv:2503.06930:
      1. Start: each timestep is its own cluster.
      2. Compute centroid distance between every pair of ADJACENT clusters.
      3. Merge the adjacent pair with minimum distance.
      4. Repeat until num_groups clusters remain.

    Distance metric: squared L2 between cluster centroids (centroid-linkage).

    Parameters
    ----------
    z_t_matrix : (T, D) float64
    num_groups  : G target clusters

    Returns
    -------
    group_assignments : (T,) int32 — group index for each row of z_t_matrix
    clusters          : list of G lists, each containing timestep indices
    """
    T = z_t_matrix.shape[0]
    if num_groups >= T:
        # Edge case: one group per timestep
        return np.arange(T, dtype=np.int32), [[t] for t in range(T)]

    # Initial clusters: one per timestep
    clusters: List[List[int]] = [[t] for t in range(T)]
    # Cache centroids to avoid recomputing
    centroids: List[np.ndarray] = [z_t_matrix[t].copy() for t in range(T)]

    while len(clusters) > num_groups:
        n_clusters = len(clusters)

        # Find adjacent pair with minimum centroid distance
        min_dist = float("inf")
        min_idx = 0
        for i in range(n_clusters - 1):
            diff = centroids[i] - centroids[i + 1]
            dist = float(np.dot(diff, diff))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        # Merge cluster min_idx and min_idx+1
        merged_members = clusters[min_idx] + clusters[min_idx + 1]
        merged_centroid = np.mean(z_t_matrix[merged_members], axis=0)

        clusters[min_idx] = merged_members
        centroids[min_idx] = merged_centroid
        del clusters[min_idx + 1]
        del centroids[min_idx + 1]

    # Assign group labels
    group_assignments = np.zeros(T, dtype=np.int32)
    for g, members in enumerate(clusters):
        for t in members:
            group_assignments[t] = g

    return group_assignments, clusters


# ---------------------------------------------------------------------------
# Step C: Per-group averaged shifting z_g
# ---------------------------------------------------------------------------

def compute_z_g(
    z_t_matrix: np.ndarray,
    clusters: List[List[int]],
) -> np.ndarray:
    """
    Compute per-group averaged shifting vectors.

    z_g = mean(z_t for t in group_g)    (Eq. 4)

    Returns
    -------
    z_g : (G, D) float64
    """
    G = len(clusters)
    D = z_t_matrix.shape[1]
    z_g = np.zeros((G, D), dtype=np.float64)
    for g, members in enumerate(clusters):
        z_g[g] = np.mean(z_t_matrix[members], axis=0)
    return z_g


# ---------------------------------------------------------------------------
# Step D: EMA channel-wise scaling s
# ---------------------------------------------------------------------------

def compute_ema_scaling(
    stats: Dict[str, Dict[str, np.ndarray]],
    z_t_matrix: np.ndarray,
    timesteps_sorted: np.ndarray,
    max_weight_col: np.ndarray,
    alpha: float = EMA_ALPHA,
) -> np.ndarray:
    """
    Compute the temporally-aggregated channel-wise scaling vector s.

    Implements Eq. 7 from the paper:
        m_T[i] = max(X̃_T[i])
        m_t[i] = α * m_{t+1}[i] + (1-α) * max(X̃_t[i]),  t = T-1, ..., 1
        s[i] = sqrt(m_1[i] / max(W^T[i]))

    Here X̃_t = X_t - z_t (shifted activation), so:
        max(X̃_t[i]) = max(X_t[i]) - z_t[i]

    Timesteps are processed in denoising order (timesteps_sorted: high→low).
    T = first step (high noise), 1 = last step (low noise).

    Parameters
    ----------
    stats           : {t_key: {"min": (D,), "max": (D,)}} for one layer
    z_t_matrix      : (T, D) shifting vectors, aligned to timesteps_sorted
    timesteps_sorted: (T,) timesteps in denoising order (high→low)
    max_weight_col  : (D,) max absolute value per input-channel column of W
    alpha           : EMA coefficient (default 0.99)

    Returns
    -------
    s : (D,) float64 channel-wise scaling vector
    """
    def _t_key(val: float) -> str:
        return f"{val:.6f}"

    T = len(timesteps_sorted)
    D = z_t_matrix.shape[1]

    # Compute max(X̃_t) = max_t - z_t for each timestep
    max_x_tilde = np.zeros((T, D), dtype=np.float64)
    for i, t_val in enumerate(timesteps_sorted):
        key = _t_key(float(t_val))
        entry = stats.get(key)
        if entry is not None:
            max_raw = entry["max"].astype(np.float64)
        else:
            max_raw = np.zeros(D, dtype=np.float64)
        # Shifted max: X̃_t = X_t - z_t  →  max(X̃_t) = max(X_t) - z_t
        max_x_tilde[i] = max_raw - z_t_matrix[i]

    # EMA from T (index 0) to 1 (index T-1)
    # m_T = max(X̃_T)  (initialize at first denoising step)
    m = max_x_tilde[0].copy()
    for i in range(1, T):
        m = alpha * m + (1.0 - alpha) * max_x_tilde[i]
    # m is now m_1 (accumulated to the last denoising step)

    # Clamp to avoid division by zero / negative sqrt input
    m = np.maximum(m, 1e-8)
    max_w = np.maximum(np.abs(max_weight_col), 1e-8)

    s = np.sqrt(m / max_w)
    return s


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------

def _ensure_diffusionkit_on_path() -> None:
    try:
        import diffusionkit.mlx  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass
    dk_src = _ROOT / "DiffusionKit" / "python" / "src"
    if dk_src.is_dir() and str(dk_src) not in sys.path:
        sys.path.insert(0, str(dk_src))


def load_target_layer_weight_max(
    model_version: str,
    layer_id: str,
    local_ckpt: str | None = None,
) -> Optional[np.ndarray]:
    """
    Load the max absolute column value of the target linear layer's weight.

    For MLX nn.Linear, weight has shape (Cout, Cin).
    max_weight_col[i] = max(|weight[:, i]|) = max over output channels for
    input channel i.  This corresponds to max([W^T]_i) in the paper.

    Returns None if the layer cannot be found.
    """
    _ensure_diffusionkit_on_path()
    try:
        from diffusionkit.mlx import DiffusionPipeline  # type: ignore
        import mlx.core as mx
    except ImportError:
        return None

    pipeline = DiffusionPipeline(
        w16=True,
        shift=3.0,
        use_t5=True,
        model_version=model_version,
        low_memory_mode=True,
        a16=True,
        local_ckpt=local_ckpt,
    )

    # Parse layer_id: e.g. "mm_05_img_fc1", "mm_00_txt_qkv", "uni_02_oproj"
    weight = _resolve_layer_weight(pipeline.mmdit, layer_id)
    if weight is None:
        return None

    w_np = np.abs(np.array(weight))  # shape (Cout, Cin)
    return w_np.max(axis=0)          # (Cin,) = max per input channel


def _resolve_layer_weight(mmdit, layer_id: str):
    """
    Navigate MMDiT to find the weight array for a given layer_id string.

    layer_id format: "mm_{idx}_{stream}_{type}" or "uni_{idx}_{type}"
    type ∈ {fc1, qkv, oproj}

    For qkv, we take q_proj.weight (all three share the same input activation,
    so they have the same column max in practice — we pick q_proj as representative).
    """
    import mlx.core as mx

    parts = layer_id.split("_")
    # Determine block type (mm vs uni) and parse idx
    if layer_id.startswith("mm_"):
        # "mm_05_img_fc1" → block_idx=5, stream="img", layer_type="fc1"
        block_idx = int(parts[1])
        stream = parts[2]  # "img" or "txt"
        layer_type = parts[3]

        blocks = getattr(mmdit, "multimodal_transformer_blocks", [])
        if block_idx >= len(blocks):
            return None
        block = blocks[block_idx]
        if stream == "img":
            tb = block.image_transformer_block
        else:
            tb = block.text_transformer_block

    elif layer_id.startswith("uni_"):
        # "uni_02_fc1" → block_idx=2, layer_type="fc1"
        block_idx = int(parts[1])
        layer_type = parts[2]
        blocks = getattr(mmdit, "unified_transformer_blocks", [])
        if block_idx >= len(blocks):
            return None
        tb = blocks[block_idx].transformer_block
    else:
        return None

    if layer_type == "fc1":
        return getattr(tb.mlp, "fc1", None) and tb.mlp.fc1.weight
    elif layer_type == "qkv":
        # q_proj is representative; all three see the same input
        return tb.attn.q_proj.weight
    elif layer_type == "oproj":
        return tb.attn.o_proj.weight
    return None


# ---------------------------------------------------------------------------
# Main computation pipeline
# ---------------------------------------------------------------------------

def compute_htg_params(
    input_stats_path: str,
    model_version: str,
    num_groups: Optional[int],
    ema_alpha: float,
    local_ckpt: Optional[str],
) -> Dict[str, np.ndarray]:
    """
    Compute HTG parameters for all profiled layers.

    Returns a flat dict ready for np.savez_compressed.
    """
    print(f"Loading input activation stats from {input_stats_path} ...")
    all_stats, timesteps_sorted = load_input_stats(input_stats_path)

    T = len(timesteps_sorted)
    G = num_groups if num_groups is not None else max(1, T // 10)
    print(f"  Timesteps (T={T}), target groups (G={G}), EMA α={ema_alpha}")
    print(f"  Layers to process: {len(all_stats)}")

    flat: Dict[str, np.ndarray] = {}

    # Load model once to get all weight column maxima
    print(f"\nLoading model weights ({model_version}) for scaling computation ...")
    _ensure_diffusionkit_on_path()
    try:
        from diffusionkit.mlx import DiffusionPipeline  # type: ignore
        pipeline = DiffusionPipeline(
            w16=True, shift=3.0, use_t5=True,
            model_version=model_version,
            low_memory_mode=True, a16=True,
            local_ckpt=local_ckpt,
        )
        pipeline_loaded = True
    except Exception as e:
        print(f"  Warning: could not load model ({e}). Scaling s will be skipped.")
        pipeline_loaded = False

    for layer_idx, (full_layer_id, layer_stats) in enumerate(sorted(all_stats.items())):
        print(f"  [{layer_idx + 1}/{len(all_stats)}] {full_layer_id} ...", end=" ")

        # Step A: z_t
        z_t = compute_z_t(layer_stats, timesteps_sorted)  # (T, D)

        # Step B: clustering
        group_assignments, clusters = constrained_hierarchical_clustering(z_t, G)

        # Step C: z_g
        z_g = compute_z_g(z_t, clusters)  # (G, D)

        # Step D: s (needs weight column max)
        s = np.ones(z_t.shape[1], dtype=np.float64)
        if pipeline_loaded:
            w_col_max = _get_weight_col_max(pipeline.mmdit, full_layer_id)
            if w_col_max is not None:
                s = compute_ema_scaling(
                    layer_stats, z_t, timesteps_sorted, w_col_max, ema_alpha
                )
            else:
                print(f"(weight not found, s=1) ", end="")

        # Group boundaries: timestep indices where group changes
        boundaries = _boundaries_from_assignments(group_assignments)

        # Store
        flat[f"{full_layer_id}::z_t"] = z_t.astype(np.float32)
        flat[f"{full_layer_id}::z_g"] = z_g.astype(np.float32)
        flat[f"{full_layer_id}::s"] = s.astype(np.float32)
        flat[f"{full_layer_id}::group_assignments"] = group_assignments
        flat[f"{full_layer_id}::group_boundaries"] = boundaries

        print(f"G_actual={len(clusters)}, D={z_t.shape[1]}")

    # Global metadata
    flat["timesteps_sorted"] = timesteps_sorted.astype(np.float32)
    flat["num_groups"] = np.array(G, dtype=np.int32)
    flat["ema_alpha"] = np.array(ema_alpha, dtype=np.float32)

    return flat


def _get_weight_col_max(mmdit, full_layer_id: str) -> Optional[np.ndarray]:
    """Extract max absolute column value for the weight of a target layer."""
    weight = _resolve_layer_weight(mmdit, full_layer_id)
    if weight is None:
        return None
    w_np = np.abs(np.array(weight))  # (Cout, Cin)
    return w_np.max(axis=0)          # (Cin,)


def _boundaries_from_assignments(group_assignments: np.ndarray) -> np.ndarray:
    """
    Convert a group_assignments array into boundary indices.

    E.g. [0,0,0,1,1,2,2,2] → [3, 5]  (indices where group changes)
    """
    boundaries = []
    for i in range(1, len(group_assignments)):
        if group_assignments[i] != group_assignments[i - 1]:
            boundaries.append(i)
    return np.array(boundaries, dtype=np.int32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3b: Compute HTG parameters (z_t, z_g, s, group assignments) "
            "from input activation statistics."
        )
    )
    parser.add_argument(
        "--input-stats", type=str, default=DEFAULT_INPUT_STATS_FILE,
        help="Path to input activation stats .npz from profile_input_activations.py",
    )
    parser.add_argument(
        "--model-version", type=str, default=MODEL_VERSION,
        help="DiffusionKit model key for loading weights (default: %(default)s)",
    )
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="Target number of timestep groups G (default: T // 10 per paper)",
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=EMA_ALPHA,
        help="EMA coefficient α for scaling accumulation (default: %(default)s)",
    )
    parser.add_argument(
        "--local-ckpt", type=str, default=None,
        help="Optional local checkpoint path for model weight loading",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=DEFAULT_HTG_PARAMS_FILE,
        help="Output path for HTG parameters .npz (default: %(default)s)",
    )

    args = parser.parse_args()

    flat = compute_htg_params(
        input_stats_path=args.input_stats,
        model_version=args.model_version,
        num_groups=args.num_groups,
        ema_alpha=args.ema_alpha,
        local_ckpt=args.local_ckpt,
    )

    np.savez_compressed(args.output, **flat)
    print(f"\nSaved HTG parameters to {args.output}")
    G = int(flat["num_groups"])
    T = len(flat["timesteps_sorted"])
    n_layers = len([k for k in flat if k.endswith("::z_g")])
    print(f"  Layers: {n_layers}, Timesteps T={T}, Groups G={G}")


if __name__ == "__main__":
    main()
