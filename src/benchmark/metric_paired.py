"""Paired-image LPIPS metric (matched by filename)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .image_io import image_paths_by_name


def compute_lpips_paired(
    generated_dir: str,
    baseline_dir: str,
    *,
    net: str = "alex",
    resize: int = 256,
) -> Optional[Dict]:
    """Mean LPIPS between matched pairs (same filename in both dirs).

    Requires ``pip install lpips``. Images resized to ``resize`` x ``resize``, [-1, 1].
    """
    try:
        import lpips as lpips_lib
        import torch
    except ImportError:
        print(
            "WARNING: lpips not installed -- skipping LPIPS. "
            "Install with: pip install lpips"
        )
        return None

    from PIL import Image as PILImage
    import torchvision.transforms as T

    loss_fn = lpips_lib.LPIPS(net=net, verbose=False)
    loss_fn.eval()

    sz = int(resize)
    preprocess = T.Compose(
        [
            T.Resize((sz, sz)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    gen_paths = image_paths_by_name(generated_dir)
    base_paths = image_paths_by_name(baseline_dir)
    common = sorted(set(gen_paths) & set(base_paths))

    if not common:
        print(
            "WARNING: LPIPS -- no matching filenames between generated and baseline dirs."
        )
        return None

    scores: list[float] = []
    with torch.no_grad():
        for name in common:
            gen_t = preprocess(
                PILImage.open(gen_paths[name]).convert("RGB")
            ).unsqueeze(0)
            base_t = preprocess(
                PILImage.open(base_paths[name]).convert("RGB")
            ).unsqueeze(0)
            scores.append(float(loss_fn(gen_t, base_t).item()))

    return {
        "lpips_mean": float(np.mean(scores)),
        "lpips_std": float(np.std(scores)),
        "n_pairs": len(scores),
    }
