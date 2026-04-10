"""CLIP embeddings, image-text scores, and CMMD."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .image_io import list_image_paths

_CLIP_BUNDLES: Dict[Tuple[str, str], Dict] = {}


def _infer_embed_dim(model, preprocess) -> int:
    import torch
    from PIL import Image as PILImage

    dummy = PILImage.new("RGB", (256, 256))
    t = preprocess(dummy).unsqueeze(0)
    with torch.no_grad():
        e = model.encode_image(t)
    return int(e.shape[-1])


def load_clip_bundle(
    arch: str = "ViT-L-14-336",
    pretrained: str = "openai",
) -> Optional[Tuple[object, object, str, int]]:
    """Load open_clip model, preprocess, model id, and embedding dim.

    Returns None if open_clip/torch unavailable.
    """
    key = (arch, pretrained)
    if key in _CLIP_BUNDLES:
        b = _CLIP_BUNDLES[key]
        return b["model"], b["preprocess"], b["model_id"], b["embed_dim"]
    try:
        import open_clip
    except ImportError:
        print(
            "WARNING: open_clip_torch or torch not installed -- "
            "skipping CLIP metrics. Install with: pip install open_clip_torch"
        )
        return None
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained
    )
    model.eval()
    model.to("cpu")
    embed_dim = _infer_embed_dim(model, preprocess)
    model_id = f"{arch}:{pretrained}"
    _CLIP_BUNDLES[key] = {
        "model": model,
        "preprocess": preprocess,
        "model_id": model_id,
        "embed_dim": embed_dim,
    }
    return model, preprocess, model_id, embed_dim


def compute_clip_embeddings(
    img_dir: str,
    *,
    batch_size: int = 16,
    arch: str = "ViT-L-14-336",
    pretrained: str = "openai",
) -> Optional[Dict]:
    """CLIP embeddings for all images (PNG/JPEG) in a directory (sorted).

    Returns dict with ``embeddings``, ``filenames``, ``model_id``, ``embed_dim``
    or None if CLIP is unavailable.
    """
    bundle = load_clip_bundle(arch=arch, pretrained=pretrained)
    if bundle is None:
        return None
    model, preprocess, model_id, embed_dim = bundle
    import torch
    from PIL import Image as PILImage

    paths = list_image_paths(img_dir)
    if not paths:
        return {
            "embeddings": np.zeros((0, embed_dim), dtype=np.float32),
            "filenames": [],
            "model_id": model_id,
            "embed_dim": embed_dim,
        }

    bs = max(1, int(batch_size))
    embeddings: List[np.ndarray] = []
    filenames: List[str] = []
    batch = []
    batch_names: List[str] = []

    def _flush() -> None:
        if not batch:
            return
        x = torch.stack(batch, dim=0)
        with torch.no_grad():
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy().astype(np.float32))
        filenames.extend(batch_names)
        batch.clear()
        batch_names.clear()

    for p in paths:
        img = PILImage.open(p).convert("RGB")
        tensor = preprocess(img)
        batch.append(tensor)
        batch_names.append(p.name)
        if len(batch) >= bs:
            _flush()
    _flush()

    emb = (
        np.concatenate(embeddings, axis=0)
        if embeddings
        else np.zeros((0, embed_dim), dtype=np.float32)
    )
    return {
        "embeddings": emb,
        "filenames": filenames,
        "model_id": model_id,
        "embed_dim": embed_dim,
    }


def compute_clip_image_text_scores(
    img_dir: str | Path,
    prompts: List[str],
    *,
    arch: str = "ViT-L-14-336",
    pretrained: str = "openai",
    batch_size: int = 16,
) -> Optional[Dict]:
    """Mean CLIP cosine similarity between each image and the prompt at the same index.

    Images are ordered by numeric filename stem (``000.png`` -> index 0).
    """
    try:
        import open_clip
        import torch
    except ImportError:
        print("WARNING: open_clip/torch not installed -- skipping CLIP image-text score.")
        return None
    from PIL import Image as PILImage

    bundle = load_clip_bundle(arch=arch, pretrained=pretrained)
    if bundle is None:
        return None
    model, preprocess, model_id, _ = bundle
    tokenizer = open_clip.get_tokenizer(arch)

    all_paths = list_image_paths(img_dir)
    numeric_paths = []
    for p in all_paths:
        try:
            int(p.stem)
            numeric_paths.append(p)
        except ValueError:
            continue
    paths = sorted(numeric_paths, key=lambda p: int(p.stem))
    bs = max(1, int(batch_size))
    batch_paths: List[Path] = []
    batch_prompts: List[str] = []
    sims: List[float] = []

    def _flush() -> None:
        if not batch_paths:
            return
        imgs = [
            preprocess(PILImage.open(p).convert("RGB")) for p in batch_paths
        ]
        x = torch.stack(imgs, dim=0)
        tokens = tokenizer(batch_prompts)
        with torch.no_grad():
            ie = model.encode_image(x)
            te = model.encode_text(tokens)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            row = (ie * te).sum(dim=-1)
        sims.extend(row.cpu().numpy().astype(np.float64).tolist())
        batch_paths.clear()
        batch_prompts.clear()

    for p in paths:
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        if idx < 0 or idx >= len(prompts):
            continue
        batch_paths.append(Path(p))
        batch_prompts.append(prompts[idx])
        if len(batch_paths) >= bs:
            _flush()
    _flush()

    if not sims:
        print("WARNING: no valid image-prompt pairs for CLIP score.")
        return None
    arr = np.array(sims, dtype=np.float64)
    return {
        "clip_image_text_mean": float(np.mean(arr)),
        "clip_image_text_std": float(np.std(arr)),
        "n_pairs": len(sims),
        "clip_model_id": model_id,
    }


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between row vectors."""
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    return a2 + b2 - 2.0 * (a @ b.T)


def compute_cmmd_from_embeddings(
    gen_emb: np.ndarray,
    ref_emb: np.ndarray,
) -> Optional[float]:
    """CMMD via RBF-kernel MMD over CLIP embeddings (median heuristic for gamma)."""
    if gen_emb.shape[0] < 2 or ref_emb.shape[0] < 2:
        print("WARNING: CMMD requires >= 2 images per directory -- returning None.")
        return None
    all_emb = np.concatenate([gen_emb, ref_emb], axis=0)
    dists = _pairwise_sq_dists(all_emb, all_emb)
    tri = dists[np.triu_indices_from(dists, k=1)]
    med = np.median(tri[tri > 0])
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    gamma = 1.0 / (2.0 * med)

    d_rr = _pairwise_sq_dists(ref_emb, ref_emb)
    d_ff = _pairwise_sq_dists(gen_emb, gen_emb)
    d_fr = _pairwise_sq_dists(gen_emb, ref_emb)

    k_rr = np.exp(-gamma * d_rr)
    k_ff = np.exp(-gamma * d_ff)
    k_fr = np.exp(-gamma * d_fr)

    mmd2 = float(k_rr.mean() + k_ff.mean() - 2.0 * k_fr.mean())
    return mmd2
