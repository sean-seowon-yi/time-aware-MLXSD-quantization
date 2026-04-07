"""
Plot FID and CMMD metrics across N non-overlapping folds of a benchmark run.

Splits generated and reference images into N equal folds and computes metrics
independently per fold (e.g. with --folds 4: 0-256, 256-512, 512-768, 768-1024).
RBF kernel bandwidth is fixed from the full embedding set so CMMD values are
comparable across folds.

Usage
-----
    python -m src.plot_fold_metrics \\
        --benchmark-dir benchmark_results/gptq_full_g32 \\
        --reference-dir benchmark_results/fp16/images \\
        --folds 4 \\
        --output gptq_folds.png
"""

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
_REQUIRED = {
    "numpy":      "pip install numpy",
    "matplotlib": "pip install matplotlib",
}
_OPTIONAL = {
    "torch_fidelity": ("pip install torch-fidelity", "FID computation will be skipped"),
}

_missing_required = []
for pkg, install in _REQUIRED.items():
    if importlib.util.find_spec(pkg) is None:
        _missing_required.append(f"  {pkg}  →  {install}")

if _missing_required:
    print("ERROR: required packages are missing. Install them and re-run:\n")
    print("\n".join(_missing_required))
    sys.exit(1)

_fidelity_available = importlib.util.find_spec("torch_fidelity") is not None
if not _fidelity_available:
    hint, effect = _OPTIONAL["torch_fidelity"]
    print(f"WARNING: torch-fidelity is not installed ({effect}).")
    print(f"  Install with: {hint}\n")

import numpy as np
import matplotlib.pyplot as plt

from src.benchmark_model import (
    compute_fidelity_metrics,
    _compute_clip_embeddings,
    _pairwise_sq_dists,
)


def compute_cmmd_fixed_bandwidth(
    gen_emb: np.ndarray,
    ref_emb: np.ndarray,
    gamma: float,
) -> float:
    """Compute MMD² with a pre-fixed RBF gamma (1 / 2σ²)."""
    d_rr = _pairwise_sq_dists(ref_emb, ref_emb)
    d_ff = _pairwise_sq_dists(gen_emb, gen_emb)
    d_fr = _pairwise_sq_dists(gen_emb, ref_emb)
    k_rr = np.exp(-gamma * d_rr)
    k_ff = np.exp(-gamma * d_ff)
    k_fr = np.exp(-gamma * d_fr)
    return float(k_rr.mean() + k_ff.mean() - 2.0 * k_fr.mean())


def estimate_gamma(gen_emb: np.ndarray, ref_emb: np.ndarray) -> float:
    """Estimate RBF gamma from the full embedding sets (median heuristic)."""
    all_emb = np.concatenate([gen_emb, ref_emb], axis=0)
    dists = _pairwise_sq_dists(all_emb, all_emb)
    tri = dists[np.triu_indices_from(dists, k=1)]
    med = np.median(tri[tri > 0])
    sigma2 = med if (np.isfinite(med) and med > 0) else 1.0
    return 1.0 / (2.0 * sigma2)


def _collect_images(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.png"))
    files += sorted(directory.glob("*.jpg"))
    return sorted(files)


def compute_fid_for_files(
    gen_files: list[Path],
    ref_files: list[Path],
) -> float | None:
    with tempfile.TemporaryDirectory() as tmp_gen, \
         tempfile.TemporaryDirectory() as tmp_ref:
        for img in gen_files:
            (Path(tmp_gen) / img.name).symlink_to(img.resolve())
        for img in ref_files:
            (Path(tmp_ref) / img.name).symlink_to(img.resolve())
        result = compute_fidelity_metrics(str(tmp_gen), str(tmp_ref))
    return result["fid"] if result else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-dir", type=Path, required=True,
                        help="Completed benchmark directory (has images/ and embeddings.npz)")
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Reference images directory (for both FID and CMMD)")
    parser.add_argument("--folds", type=int, default=4,
                        help="Number of non-overlapping folds (default: 4)")
    parser.add_argument("--output", type=Path, default=Path("fold_metrics.png"),
                        help="Output plot path (default: fold_metrics.png)")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title (default: benchmark dir basename)")
    args = parser.parse_args()

    gen_images_dir = args.benchmark_dir / "images"
    embeddings_path = args.benchmark_dir / "embeddings.npz"

    print(f"Loading generated embeddings from {embeddings_path}...")
    emb_data = np.load(embeddings_path, allow_pickle=True)
    gen_emb = emb_data["generated_embeddings"].astype(np.float64)
    print(f"  generated: {gen_emb.shape}")

    print(f"Computing reference embeddings from {args.reference_dir}...")
    ref_result = _compute_clip_embeddings(str(args.reference_dir))
    if ref_result is None:
        print("ERROR: could not compute reference embeddings (CLIP unavailable).")
        sys.exit(1)
    ref_emb = ref_result["embeddings"].astype(np.float64)
    print(f"  reference: {ref_emb.shape}")

    gamma = estimate_gamma(gen_emb, ref_emb)
    print(f"  RBF gamma (fixed): {gamma:.6f}")

    gen_files = _collect_images(gen_images_dir)
    ref_files = _collect_images(args.reference_dir)

    max_n = min(len(gen_files), len(ref_files), len(gen_emb), len(ref_emb))
    fold_size = max_n // args.folds
    if fold_size == 0:
        raise ValueError(f"Not enough images for {args.folds} folds (max available: {max_n})")

    fid_values: list[float | None] = []
    cmmd_values: list[float] = []
    x_labels: list[str] = []

    for i in range(args.folds):
        lo, hi = i * fold_size, (i + 1) * fold_size
        label = f"{lo}–{hi}"
        print(f"Fold {i+1}/{args.folds}: images {label}...")
        cmmd = compute_cmmd_fixed_bandwidth(gen_emb[lo:hi], ref_emb[lo:hi], gamma)
        fid = compute_fid_for_files(gen_files[lo:hi], ref_files[lo:hi])
        fid_str = f"{fid:.4f}" if fid is not None else "N/A"
        print(f"  FID={fid_str}  CMMD={cmmd:.6f}")
        fid_values.append(fid)
        cmmd_values.append(cmmd)
        x_labels.append(label)

    # Summary stats
    valid_fids = [v for v in fid_values if v is not None]
    print(f"\nFID  — mean: {np.mean(valid_fids):.4f}  std: {np.std(valid_fids):.4f}" if valid_fids else "\nFID: N/A")
    print(f"CMMD — mean: {np.mean(cmmd_values):.6f}  std: {np.std(cmmd_values):.6f}")

    # --- Plot ---
    x = list(range(1, args.folds + 1))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    color_fid = "#1f77b4"
    color_cmmd = "#ff7f0e"

    lines = []
    if any(v is not None for v in fid_values):
        fid_plot = [v if v is not None else float("nan") for v in fid_values]
        line1, = ax1.plot(x, fid_plot, color=color_fid, marker="o", linewidth=2, label="FID")
        lines.append(line1)
    else:
        ax1.set_visible(False)

    line2, = ax2.plot(x, cmmd_values, color=color_cmmd, marker="s", linewidth=2, label="CMMD")
    lines.append(line2)

    ax1.set_xlabel(f"Fold  (size={fold_size} images each)", fontsize=12)
    ax1.set_ylabel("FID", color=color_fid, fontsize=12)
    ax2.set_ylabel("CMMD", color=color_cmmd, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_fid)
    ax2.tick_params(axis="y", labelcolor=color_cmmd)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=15, ha="right")
    ax1.grid(True, alpha=0.3)

    title = args.title or args.benchmark_dir.name
    ax1.set_title(title, fontsize=13)
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=11)

    # Summary stats as text box
    stats_lines = []
    if valid_fids:
        stats_lines.append(f"FID   mean={np.mean(valid_fids):.2f}  std={np.std(valid_fids):.2f}")
    stats_lines.append(f"CMMD  mean={np.mean(cmmd_values):.4f}  std={np.std(cmmd_values):.4f}")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.04, "\n".join(stats_lines), ha="center", va="bottom",
             fontsize=10, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.8))
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot → {args.output}")


if __name__ == "__main__":
    main()
