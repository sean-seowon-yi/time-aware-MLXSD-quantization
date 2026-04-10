"""W4A8 image generation for benchmark evaluation.

Loads a Phase 2/3 quantized pipeline and generates images from prompts,
saving PNGs with resume support.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


def _load_pipeline(
    config: str,
    *,
    quantized_dir: Optional[Path] = None,
    poly_schedule: Optional[Dict] = None,
):
    """Load DiffusionPipeline and apply W4A8 quantization config.

    Supported configs: ``fp16_p2``, ``w4a8``, ``w4a8_static``, ``w4a8_poly``.
    """
    from src.phase2.config import DIFFUSIONKIT_SRC, MODEL_VERSION, PIPELINE_KWARGS

    if DIFFUSIONKIT_SRC not in sys.path:
        sys.path.insert(0, DIFFUSIONKIT_SRC)
    from diffusionkit.mlx import DiffusionPipeline

    pipeline = DiffusionPipeline(**PIPELINE_KWARGS, model_version=MODEL_VERSION)

    if config in ("w4a8", "w4a8_static", "w4a8_poly"):
        if quantized_dir is None:
            raise ValueError(
                f"--quantized-dir is required for config={config!r}"
            )
        if config == "w4a8_poly":
            from src.phase3.quantize_poly import load_quantized_model_poly

            load_quantized_model_poly(
                pipeline, quantized_dir, schedule_override=poly_schedule,
            )
        else:
            from src.phase2.quantize_static import load_quantized_model_static

            load_quantized_model_static(pipeline, quantized_dir)

    return pipeline


def generate_images(
    config: str,
    prompts: List[str],
    output_dir: Path,
    num_steps: int,
    cfg_scale: float,
    seed_base: int,
    warmup: int = 0,
    resume: bool = True,
    poly_schedule: Optional[Dict] = None,
    group_size: int = 64,
    seeds: Optional[List[int]] = None,
    reload_n: Optional[int] = None,
    quantized_dir: Optional[Path] = None,
    img_digits: int = 3,
) -> None:
    """Generate images for all prompts and save to ``output_dir/images/``.

    Parameters
    ----------
    resume : bool
        Skip PNGs that already exist on disk.
    reload_n : int or None
        Reload pipeline for the first *reload_n* images, then persist.
        ``None`` reloads every image.
    img_digits : int
        Zero-pad width for filenames (e.g. 3 -> ``000.png``).
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    total = len(prompts)
    _pipeline_cache = None

    pbar = tqdm(
        enumerate(prompts),
        total=total,
        desc=f"  {config}",
        unit="img",
        dynamic_ncols=True,
    )
    for img_idx, prompt in pbar:
        img_path = images_dir / f"{img_idx:0{img_digits}d}.png"
        if resume and img_path.exists():
            continue

        seed = seeds[img_idx] if seeds is not None else seed_base + img_idx

        persist = reload_n is not None and img_idx >= reload_n
        if persist and _pipeline_cache is not None:
            pipeline = _pipeline_cache
        else:
            pipeline = _load_pipeline(
                config,
                quantized_dir=quantized_dir,
                poly_schedule=poly_schedule,
            )
            if persist:
                _pipeline_cache = pipeline

        t0 = time.time()
        image, _ = pipeline.generate_image(
            prompt,
            cfg_weight=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            negative_text="",
            verbose=False,
        )
        elapsed = time.time() - t0

        image.save(img_path)
        pbar.set_postfix({"s/img": f"{elapsed:.1f}"}, refresh=True)
