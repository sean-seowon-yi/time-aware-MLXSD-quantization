"""
Analyze W4 bucket utilization in quantized weight NPZs.

Measures how well 4-bit quantization uses its 16 levels [-8..7] per row.
Key metrics per layer:
  - unique_levels  : avg number of distinct values used per row (max 16)
  - entropy        : avg Shannon entropy in bits (max 4.0 = uniform)
  - saturation_pct : % of weights at ±7 (clipped to boundary)
  - fill_pct       : unique_levels / 16 * 100

Also simulates what per-group scales (group_size=64, 128) would give using
RTN on the existing FP16 weights, so you can compare utilization improvement
without re-running AdaRound.

Usage:
    python -m src.analyze_weight_utilization \
        --weights quantized_weights_w4a8_adaround_poly_p100 \
        [--fp16-weights path/to/fp16_model]  # optional, for group-quant simulation
        [--group-sizes 64 128]               # group sizes to simulate
        [--top-n 20]                         # show N worst layers
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


BITS = 4
QMIN = -(2 ** (BITS - 1))      # -8
QMAX = 2 ** (BITS - 1) - 1     # +7
N_LEVELS = 2 ** BITS            # 16
ALL_LEVELS = np.arange(QMIN, QMAX + 1)  # [-8..7]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def row_entropy(row: np.ndarray) -> float:
    """Shannon entropy of quantized weight distribution (bits, max=4.0)."""
    counts = np.bincount(row - QMIN, minlength=N_LEVELS).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def weight_int_metrics(weight_int: np.ndarray) -> Dict[str, float]:
    """
    Compute utilization metrics for a (out, in) int8 weight matrix.
    Values should be in range [QMIN, QMAX].
    """
    out, in_ = weight_int.shape
    entropies = []
    unique_counts = []
    saturation_counts = []

    for row in weight_int:
        entropies.append(row_entropy(row))
        unique_counts.append(len(np.unique(row)))
        saturation_counts.append(np.sum((row == QMIN) | (row == QMAX)))

    total_weights = out * in_
    return {
        "entropy_mean": float(np.mean(entropies)),
        "entropy_min": float(np.min(entropies)),
        "unique_mean": float(np.mean(unique_counts)),
        "unique_min": float(np.min(unique_counts)),
        "fill_pct": float(np.mean(unique_counts)) / N_LEVELS * 100,
        "saturation_pct": float(np.sum(saturation_counts)) / total_weights * 100,
        "shape": list(weight_int.shape),
    }


def simulate_group_rtn(W_fp: np.ndarray, group_size: int) -> np.ndarray:
    """
    RTN quantize W_fp with per-group scales (consecutive input-channel groups).
    Returns int8 weight_int in [QMIN, QMAX].
    """
    out, in_ = W_fp.shape
    weight_int = np.zeros_like(W_fp, dtype=np.int8)

    if group_size <= 0 or group_size >= in_:
        # Per-row
        scale = np.max(np.abs(W_fp), axis=1, keepdims=True) / QMAX
        scale = np.where(scale == 0, 1.0, scale)
        weight_int = np.clip(np.round(W_fp / scale), QMIN, QMAX).astype(np.int8)
    else:
        n_groups = (in_ + group_size - 1) // group_size
        for g in range(n_groups):
            c0 = g * group_size
            c1 = min(c0 + group_size, in_)
            chunk = W_fp[:, c0:c1]
            scale = np.max(np.abs(chunk), axis=1, keepdims=True) / QMAX
            scale = np.where(scale == 0, 1.0, scale)
            weight_int[:, c0:c1] = np.clip(
                np.round(chunk / scale), QMIN, QMAX
            ).astype(np.int8)

    return weight_int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_block_utilization(
    weights_dir: Path,
    block_file: Path,
) -> Dict[str, Dict]:
    """Load all weight_int arrays from one block NPZ and compute metrics."""
    npz = np.load(block_file)
    results = {}
    for key in npz.files:
        if not key.endswith("__weight_int"):
            continue
        layer_name = key[: -len("__weight_int")]
        w = npz[key].astype(np.int32)  # avoid uint8 aliasing issues
        if w.ndim != 2:
            continue
        results[layer_name] = weight_int_metrics(w)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(all_metrics: Dict[str, Dict], top_n: int = 20):
    """Print worst-utilization layers sorted by entropy."""
    rows = sorted(all_metrics.items(), key=lambda x: x[1]["entropy_mean"])

    print(f"\n{'Layer':<60} {'Entropy':>8} {'Fill%':>7} {'Sat%':>7} {'Shape'}")
    print("-" * 100)
    for name, m in rows[:top_n]:
        shape_str = f"{m['shape'][0]}×{m['shape'][1]}"
        print(
            f"{name:<60} {m['entropy_mean']:>8.3f} {m['fill_pct']:>7.1f}"
            f" {m['saturation_pct']:>7.2f} {shape_str}"
        )

    entropies = [m["entropy_mean"] for m in all_metrics.values()]
    fills = [m["fill_pct"] for m in all_metrics.values()]
    sats = [m["saturation_pct"] for m in all_metrics.values()]
    print(f"\n{'OVERALL':<60} {'Entropy':>8} {'Fill%':>7} {'Sat%':>7}")
    print(
        f"  mean: {np.mean(entropies):>8.3f} {np.mean(fills):>7.1f} {np.mean(sats):>7.2f}"
    )
    print(
        f"  min:  {np.min(entropies):>8.3f} {np.min(fills):>7.1f} {np.min(sats):>7.2f}"
    )


def print_group_comparison(
    baseline: Dict[str, Dict],
    simulated: Dict[int, Dict[str, Dict]],
):
    """Compare baseline per-row vs simulated group-quant utilization."""
    print("\n" + "=" * 70)
    print("GROUP QUANTIZATION UTILIZATION COMPARISON (RTN simulation)")
    print("=" * 70)

    group_sizes = sorted(simulated.keys())
    header = f"{'Group size':>12}" + "".join(f"{'gs='+str(g):>12}" for g in group_sizes)
    print(f"\n{'Metric':<20} {'per-row':>12}" + "".join(f"{'gs='+str(g):>12}" for g in group_sizes))
    print("-" * (20 + 12 + 12 * len(group_sizes)))

    for metric_key, label in [
        ("entropy_mean", "Entropy (bits)"),
        ("fill_pct", "Fill % (16 lvls)"),
        ("saturation_pct", "Saturation %"),
    ]:
        base_val = np.mean([m[metric_key] for m in baseline.values()])
        row = f"{label:<20} {base_val:>12.3f}"
        for gs in group_sizes:
            gs_val = np.mean([m[metric_key] for m in simulated[gs].values()])
            delta = gs_val - base_val
            sign = "+" if delta >= 0 else ""
            row += f"  {gs_val:>6.3f}({sign}{delta:.3f})"
        print(row)

    # Per-layer improvement highlights
    print("\nTop 10 layers with most entropy gain (group_size=64 vs per-row):")
    if 64 in simulated:
        gains = {
            name: simulated[64][name]["entropy_mean"] - baseline[name]["entropy_mean"]
            for name in baseline
            if name in simulated[64]
        }
        for name, gain in sorted(gains.items(), key=lambda x: -x[1])[:10]:
            b = baseline[name]["entropy_mean"]
            s = simulated[64][name]["entropy_mean"]
            print(f"  {name:<55} {b:.3f} → {s:.3f}  (+{gain:.3f} bits)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Quantized weights directory (contains weights/*.npz)")
    parser.add_argument("--fp16-weights", type=Path, default=None,
                        help="FP16 model weights dir for group-quant simulation")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[64, 128],
                        help="Group sizes to simulate (default: 64 128)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of worst layers to show")
    args = parser.parse_args()

    weights_dir = args.weights
    npz_files = sorted((weights_dir / "weights").glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {weights_dir}/weights/")

    print(f"Loading {len(npz_files)} block files from {weights_dir}...")
    all_metrics: Dict[str, Dict] = {}
    for f in npz_files:
        block = f.stem  # e.g. "mm0"
        block_metrics = load_block_utilization(weights_dir, f)
        for layer_name, m in block_metrics.items():
            all_metrics[f"{block}.{layer_name}"] = m

    print(f"  {len(all_metrics)} layers analyzed")

    print(f"\n{'='*70}")
    print(f"BASELINE (per-row W4, group_size=0)")
    print(f"{'='*70}")
    print_summary(all_metrics, top_n=args.top_n)

    # Group-quant simulation using existing weight_int arrays
    # We reconstruct approximate FP16 weights from weight_int * scale, then
    # re-quantize with group scales via RTN. This isn't identical to running
    # AdaRound with group_size, but gives a fair comparison of scale granularity benefit.
    if args.group_sizes:
        print(f"\nSimulating group quantization from reconstructed FP16 weights...")
        simulated: Dict[int, Dict[str, Dict]] = {gs: {} for gs in args.group_sizes}

        for f in npz_files:
            block = f.stem
            npz = np.load(f)
            for key in npz.files:
                if not key.endswith("__weight_int"):
                    continue
                layer_name = key[: -len("__weight_int")]
                full_name = f"{block}.{layer_name}"

                w_int = npz[key].astype(np.float32)
                scale_key = key.replace("__weight_int", "__scale")
                if scale_key not in npz.files:
                    continue
                scale = npz[scale_key].astype(np.float32)

                # Reconstruct approximate FP16
                if scale.ndim == 2 and scale.shape[1] == 1:
                    W_fp = w_int * scale  # per-row broadcast
                elif scale.shape == w_int.shape:
                    W_fp = w_int * scale  # already expanded
                else:
                    # Compact group scale — expand manually
                    W_fp = w_int * np.repeat(scale, w_int.shape[1] // scale.shape[1], axis=1)

                for gs in args.group_sizes:
                    w_sim = simulate_group_rtn(W_fp, gs)
                    simulated[gs][full_name] = weight_int_metrics(w_sim.astype(np.int32))

        print_group_comparison(all_metrics, simulated)


if __name__ == "__main__":
    main()
