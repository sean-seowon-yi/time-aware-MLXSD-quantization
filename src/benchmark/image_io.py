"""Image path listing for benchmark metrics (PNG + JPEG)."""

from __future__ import annotations

from pathlib import Path
from typing import List


def list_png_paths(img_dir: str | Path) -> List[Path]:
    """Sorted PNG paths only (for CLIP cache key alignment with saved outputs)."""
    return sorted(Path(img_dir).glob("*.png"))


def list_image_paths(img_dir: str | Path) -> List[Path]:
    """Sorted PNG and JPEG paths — use for FID-style dirs and paired metrics."""
    d = Path(img_dir)
    return sorted(
        list(d.glob("*.png"))
        + list(d.glob("*.jpg"))
        + list(d.glob("*.jpeg"))
    )


def image_paths_by_name(img_dir: str | Path) -> dict[str, Path]:
    """Map basename → path for pairing across directories."""
    return {p.name: p for p in list_image_paths(img_dir)}
