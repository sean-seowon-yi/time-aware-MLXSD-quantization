"""FID, IS, KID, Precision, Recall via torch-fidelity."""

from __future__ import annotations

from typing import Dict, Optional

from .image_io import list_image_paths


def compute_fidelity_metrics(
    generated_dir: str,
    reference_dir: str,
    *,
    kid_subset_max: int = 1000,
    use_cuda: bool = False,
    isc: bool = True,
    kid: bool = True,
    prc: bool = True,
) -> Optional[Dict]:
    """
    Compute FID, IS, KID, Precision, and Recall between two image directories.

    Requires ``pip install torch-fidelity``. Returns None if unavailable.

    Parameters
    ----------
    generated_dir, reference_dir : str | Path
        Directories containing images (PNG/JPEG).
    isc, kid, prc : bool
        When False, skip that metric in torch-fidelity (faster; missing keys
        become NaN in the returned dict). FID is always requested.

    Returns
    -------
    dict with keys fid, isc_mean, isc_std, kid_mean, kid_std, precision, recall,
    or None if torch-fidelity is not installed.
    """
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        print(
            "WARNING: torch-fidelity not installed — skipping FID/IS/KID. "
            "Install with: pip install torch-fidelity"
        )
        return None

    n_gen = len(list_image_paths(generated_dir))
    n_ref = len(list_image_paths(reference_dir))
    if n_gen == 0 or n_ref == 0:
        print(
            "WARNING: torch-fidelity skipped — at least one directory has no images."
        )
        return None
    cap = max(1, int(kid_subset_max))
    kid_subset_size = min(n_gen, n_ref, cap)

    cm_kw: Dict = dict(
        input1=str(generated_dir),
        input2=str(reference_dir),
        fid=True,
        isc=isc,
        kid=kid,
        prc=prc,
        verbose=False,
        cuda=use_cuda,
        save_cpu_ram=not use_cuda,
    )
    if kid:
        cm_kw["kid_subset_size"] = kid_subset_size
    metrics = calculate_metrics(**cm_kw)
    return {
        "fid": float(metrics.get("frechet_inception_distance", float("nan"))),
        "isc_mean": float(metrics.get("inception_score_mean", float("nan"))),
        "isc_std": float(metrics.get("inception_score_std", float("nan"))),
        "kid_mean": float(metrics.get("kernel_inception_distance_mean", float("nan"))),
        "kid_std": float(metrics.get("kernel_inception_distance_std", float("nan"))),
        "precision": float(metrics.get("precision", float("nan"))),
        "recall": float(metrics.get("recall", float("nan"))),
    }
