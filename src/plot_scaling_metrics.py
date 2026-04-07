"""
Plot FID and CMMD metrics at increasing sample counts for a completed benchmark run.

Reuses precomputed CLIP embeddings (embeddings.npz) for generated images; computes
reference embeddings fresh from --reference-dir so any reference set can be used.
RBF kernel bandwidth is fixed from the full (n=max) embedding set so CMMD values
are comparable across sample counts.

Usage
-----
    python -m src.plot_scaling_metrics \
        --benchmark-dir benchmark_results/fp16 \
        --reference-dir benchmark_results/true \
        --sample-counts 256 512 768 1024 \
        --output fp16_scaling.png
"""

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check — fail early with actionable messages
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
import matplotlib.ticker as ticker

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
    """Estimate RBF gamma from the full embedding set (median heuristic)."""
    all_emb = np.concatenate([gen_emb, ref_emb], axis=0)
    dists = _pairwise_sq_dists(all_emb, all_emb)
    tri = dists[np.triu_indices_from(dists, k=1)]
    med = np.median(tri[tri > 0])
    sigma2 = med if (np.isfinite(med) and med > 0) else 1.0
    return 1.0 / (2.0 * sigma2)


def _collect_images(directory: Path, n: int) -> list[Path]:
    files = sorted(directory.glob("*.png"))[:n]
    if len(files) < n:
        files += sorted(directory.glob("*.jpg"))[:n - len(files)]
    return files


def compute_fid_at_n(
    n: int,
    gen_images_dir: Path,
    reference_dir: Path,
) -> float | None:
    """Run torch-fidelity on the first n generated vs first n reference images."""
    gen_files = _collect_images(gen_images_dir, n)
    ref_files = _collect_images(reference_dir, n)
    with tempfile.TemporaryDirectory() as tmp_gen, \
         tempfile.TemporaryDirectory() as tmp_ref:
        for img in gen_files:
            (Path(tmp_gen) / img.name).symlink_to(img.resolve())
        for img in ref_files:
            (Path(tmp_ref) / img.name).symlink_to(img.resolve())
        result = compute_fidelity_metrics(str(tmp_gen), str(tmp_ref))
    return result["fid"] if result else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True,
                        help="Completed benchmark directory (has images/ and embeddings.npz)")
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Reference images directory (for both FID and CMMD)")
    parser.add_argument("--sample-counts", type=int, nargs="+",
                        default=[256, 512, 768, 1024],
                        help="Sample counts to evaluate (default: 256 512 768 1024)")
    parser.add_argument("--output", type=Path, default=Path("scaling_metrics.png"),
                        help="Output plot path (default: scaling_metrics.png)")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title (default: benchmark dir basename)")
    args = parser.parse_args()

    gen_images_dir = args.benchmark_dir / "images"
    embeddings_path = args.benchmark_dir / "embeddings.npz"

    # --- Load generated embeddings from stored npz ---
    print(f"Loading generated embeddings from {embeddings_path}...")
    emb_data = np.load(embeddings_path, allow_pickle=True)
    gen_emb = emb_data["generated_embeddings"].astype(np.float64)
    print(f"  generated: {gen_emb.shape}")

    # --- Compute reference embeddings fresh from --reference-dir (Bug 1 fix) ---
    print(f"Computing reference embeddings from {args.reference_dir}...")
    ref_result = _compute_clip_embeddings(str(args.reference_dir))
    if ref_result is None:
        print("ERROR: could not compute reference embeddings (CLIP unavailable).")
        sys.exit(1)
    ref_emb = ref_result["embeddings"].astype(np.float64)
    print(f"  reference: {ref_emb.shape}")

    # --- Fix bandwidth from the full set once (Bug 2 fix) ---
    gamma = estimate_gamma(gen_emb, ref_emb)
    print(f"  RBF gamma (fixed): {gamma:.6f}")

    sample_counts = sorted(args.sample_counts)
    max_available = len(sorted(gen_images_dir.glob("*.png")))
    sample_counts = [n for n in sample_counts if n <= max_available]
    if not sample_counts:
        raise ValueError(f"No valid sample counts — only {max_available} images available")

    fid_values: list[float | None] = []
    cmmd_values: list[float] = []

    for n in sample_counts:
        print(f"Computing metrics at n={n}...")
        cmmd = compute_cmmd_fixed_bandwidth(gen_emb[:n], ref_emb[:n], gamma)
        fid = compute_fid_at_n(n, gen_images_dir, args.reference_dir)
        fid_str = f"{fid:.4f}" if fid is not None else "N/A"
        print(f"  FID={fid_str}  CMMD={cmmd:.6f}")
        fid_values.append(fid)
        cmmd_values.append(cmmd)

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    color_fid = "#1f77b4"   # blue
    color_cmmd = "#ff7f0e"  # orange

    lines = []
    if any(v is not None for v in fid_values):
        fid_plot = [v if v is not None else float("nan") for v in fid_values]
        line1, = ax1.plot(sample_counts, fid_plot, color=color_fid,
                          marker="o", linewidth=2, label="FID")
        lines.append(line1)
    else:
        ax1.set_visible(False)

    line2, = ax2.plot(sample_counts, cmmd_values, color=color_cmmd,
                      marker="s", linewidth=2, label="CMMD")
    lines.append(line2)

    ax1.set_xlabel("Number of samples", fontsize=12)
    ax1.set_ylabel("FID", color=color_fid, fontsize=12)
    ax2.set_ylabel("CMMD", color=color_cmmd, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_fid)
    ax2.tick_params(axis="y", labelcolor=color_cmmd)
    ax1.set_xticks(sample_counts)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.grid(True, alpha=0.3)

    title = args.title or args.benchmark_dir.name
    ax1.set_title(title, fontsize=13)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=11)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved plot → {args.output}")


if __name__ == "__main__":
    main()
