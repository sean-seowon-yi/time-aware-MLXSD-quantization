"""
Benchmark the SD3-Medium pipeline for a given quantization config.

Measures distributional image quality (FID / IS / KID via torch-fidelity),
per-image latency statistics, and peak memory usage (Metal + system RSS).

Phases
------
Phase 1 — Generate images
    Reads prompts from all_prompts.csv, generates N images using seed+i for
    image i, saves to output_dir/images/{idx:04d}.png.  ``--resume`` skips
    images whose PNG already exists.

Phase 2 — Compute metrics
    FID / IS / KID: requires ``pip install torch-fidelity``.  Gracefully
    degrades (logs warning, writes null) if the package is not available.
    PRDC / CMMD (CLIP): requires ``pip install open_clip_torch``.  Gracefully
    skipped if unavailable or if ``--skip-clip-metrics`` is set.
    Pass ``--reference-dir`` to trigger this phase.

Usage
-----
    # FP16 baseline: generate 500 images
    conda run -n diffusionkit python -m src.benchmark_model \\
        --config fp16 --num-images 500 --num-steps 28 \\
        --output-dir benchmark_results/fp16 --resume

    # AdaRound W4: generate + compute metrics in one shot
    conda run -n diffusionkit python -m src.benchmark_model \\
        --config adaround_w4 --adaround-output quantized_weights \\
        --num-images 150 --num-steps 28 \\
        --reference-dir calibration_data_100/images \\
        --output-dir benchmark_results/adaround_w4 --resume

    # Metrics only (reference images already generated)
    conda run -n diffusionkit python -m src.benchmark_model \\
        --skip-generation \\
        --generated-dir benchmark_results/adaround_w4/images \\
        --reference-dir calibration_data_100/images \\
        --output-dir benchmark_results/adaround_w4
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Section 2 — Latency statistics
# ---------------------------------------------------------------------------

def compute_latency_stats(timings: List[float], warmup: int = 0) -> Dict:
    """
    Compute latency statistics from a list of per-image timings.

    Parameters
    ----------
    timings : list of float
        Per-image wall-clock times in seconds (including warmup images).
    warmup : int
        Number of leading images to exclude from statistics.

    Returns
    -------
    dict with keys: mean_s, std_s, p50_s, p95_s, min_s, max_s,
                    warmup_images, measured_images.
    """
    measured = timings[warmup:]
    if not measured:
        return {
            "mean_s": None, "std_s": None, "p50_s": None, "p95_s": None,
            "min_s": None, "max_s": None,
            "warmup_images": warmup, "measured_images": 0,
        }
    arr = np.array(measured, dtype=np.float64)
    return {
        "mean_s": float(np.mean(arr)),
        "std_s": float(np.std(arr)),
        "p50_s": float(np.percentile(arr, 50)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "warmup_images": warmup,
        "measured_images": len(measured),
    }


# ---------------------------------------------------------------------------
# Section 4 — Memory statistics
# ---------------------------------------------------------------------------

def sample_metal_memory() -> Dict:
    """
    Sample current MLX Metal memory usage.

    Returns dict with keys 'active_mb' and 'peak_mb'.  Falls back to zeros
    if the MLX metal API is not available (non-Apple platform or old MLX).
    """
    try:
        import mlx.core as mx
        active_bytes = mx.metal.get_active_memory()
        peak_bytes = mx.metal.get_peak_memory()
        return {
            "active_mb": active_bytes / 1e6,
            "peak_mb": peak_bytes / 1e6,
        }
    except Exception:
        return {"active_mb": 0.0, "peak_mb": 0.0}


def reset_metal_peak_memory() -> None:
    """Reset MLX Metal peak memory counter (no-op if unavailable)."""
    try:
        import mlx.core as mx
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def sample_system_rss_mb() -> float:
    """Return current process RSS in MB via psutil, or 0.0 if unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Section 3 — FID / IS / KID / CLIP metrics
# ---------------------------------------------------------------------------

def compute_fidelity_metrics(
    generated_dir: str,
    reference_dir: str,
) -> Optional[Dict]:
    """
    Compute FID, IS, KID, Precision, and Recall between two image directories.

    Requires ``pip install torch-fidelity``.  Returns None gracefully if the
    package is not installed.

    Parameters
    ----------
    generated_dir : str | Path
        Directory containing generated PNG images.
    reference_dir : str | Path
        Directory containing reference (ground-truth) PNG images.

    Returns
    -------
    dict with keys fid, isc_mean, isc_std, kid_mean, kid_std, precision, recall,
    or None if torch-fidelity is unavailable.
    """
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        print("WARNING: torch-fidelity not installed — skipping FID/IS/KID. "
              "Install with: pip install torch-fidelity")
        return None

    n_gen = len(list(Path(generated_dir).glob("*.png")))
    n_ref = len(list(Path(reference_dir).glob("*.png")))
    kid_subset_size = min(n_gen, n_ref, 1000)

    metrics = calculate_metrics(
        input1=str(generated_dir),
        input2=str(reference_dir),
        fid=True,
        isc=True,
        kid=True,
        prc=True,
        kid_subset_size=kid_subset_size,
        verbose=False,
        cuda=False,
        save_cpu_ram=True,  # forces num_workers=0, avoids shm_manager on macOS
    )
    return {
        "fid": float(metrics.get("frechet_inception_distance", float("nan"))),
        "isc_mean": float(metrics.get("inception_score_mean", float("nan"))),
        "isc_std": float(metrics.get("inception_score_std", float("nan"))),
        "kid_mean": float(metrics.get("kernel_inception_distance_mean", float("nan"))),
        "kid_std": float(metrics.get("kernel_inception_distance_std", float("nan"))),
        "precision": float(metrics.get("precision", float("nan"))),
        "recall": float(metrics.get("recall", float("nan"))),
    }


_CLIP_CACHE = {}


def _load_clip_model():
    """
    Load CLIP model + preprocess once (open_clip).

    Returns (model, preprocess, model_id) or (None, None, None) if unavailable.
    """
    if _CLIP_CACHE.get("model") is not None:
        return _CLIP_CACHE["model"], _CLIP_CACHE["preprocess"], _CLIP_CACHE["model_id"]
    try:
        import open_clip
        import torch
    except ImportError:
        print("WARNING: open_clip_torch or torch not installed — "
              "skipping CLIP metrics. Install with: pip install open_clip_torch")
        return None, None, None
    model_id = "openai/clip-vit-large-patch14-336"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-336", pretrained="openai"
    )
    model.eval()
    model.to("cpu")
    _CLIP_CACHE["model"] = model
    _CLIP_CACHE["preprocess"] = preprocess
    _CLIP_CACHE["model_id"] = model_id
    return model, preprocess, model_id


def _list_pngs(img_dir: str) -> List[Path]:
    return sorted(Path(img_dir).glob("*.png"))


def _compute_clip_embeddings(img_dir: str) -> Optional[Dict]:
    """
    Compute CLIP embeddings for all PNGs in a directory.

    Returns dict with keys: embeddings (np.ndarray), filenames (list[str]), model_id (str).
    """
    model, preprocess, model_id = _load_clip_model()
    if model is None:
        return None
    import torch
    import numpy as np
    from PIL import Image as PILImage

    paths = _list_pngs(img_dir)
    if not paths:
        return {"embeddings": np.zeros((0, 768), dtype=np.float32),
                "filenames": [], "model_id": model_id}

    embeddings = []
    filenames: List[str] = []
    batch = []
    batch_names = []
    batch_size = 16

    def _flush():
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
        if len(batch) >= batch_size:
            _flush()
    _flush()

    emb = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 768), dtype=np.float32)
    return {"embeddings": emb, "filenames": filenames, "model_id": model_id}


def _load_clip_cache(cache_path: Path) -> Optional[Dict]:
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k].tolist() if data[k].dtype == object else data[k] for k in data.files}
    except Exception:
        return None


def _save_clip_cache(cache_path: Path, cache: Dict) -> None:
    np.savez(
        cache_path,
        generated_embeddings=cache["generated_embeddings"],
        generated_filenames=np.array(cache["generated_filenames"], dtype=object),
        reference_embeddings=cache["reference_embeddings"],
        reference_filenames=np.array(cache["reference_filenames"], dtype=object),
        clip_model_id=cache["clip_model_id"],
    )


def _get_clip_embeddings_with_cache(
    generated_dir: str,
    reference_dir: str,
    cache_path: Path,
) -> Optional[Dict]:
    cache = _load_clip_cache(cache_path)
    gen_files = [p.name for p in _list_pngs(generated_dir)]
    ref_files = [p.name for p in _list_pngs(reference_dir)]
    _, _, model_id = _load_clip_model()
    if model_id is None:
        return None

    if cache is not None:
        if (cache.get("clip_model_id") == model_id and
                cache.get("generated_filenames") == gen_files and
                cache.get("reference_filenames") == ref_files):
            return {
                "generated_embeddings": cache["generated_embeddings"],
                "reference_embeddings": cache["reference_embeddings"],
                "clip_model_id": cache["clip_model_id"],
            }

    gen = _compute_clip_embeddings(generated_dir)
    ref = _compute_clip_embeddings(reference_dir)
    if gen is None or ref is None:
        return None

    cache_out = {
        "generated_embeddings": gen["embeddings"],
        "generated_filenames": gen["filenames"],
        "reference_embeddings": ref["embeddings"],
        "reference_filenames": ref["filenames"],
        "clip_model_id": gen["model_id"],
    }
    _save_clip_cache(cache_path, cache_out)
    return {
        "generated_embeddings": cache_out["generated_embeddings"],
        "reference_embeddings": cache_out["reference_embeddings"],
        "clip_model_id": cache_out["clip_model_id"],
    }


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    return a2 + b2 - 2.0 * (a @ b.T)


def compute_prdc_metrics(
    gen_emb: np.ndarray,
    ref_emb: np.ndarray,
    k: int = 5,
) -> Optional[Dict]:
    """
    Compute PRDC (precision/recall/density/coverage) in embedding space.
    """
    n_gen, n_ref = gen_emb.shape[0], ref_emb.shape[0]
    if n_gen < 2 or n_ref < 2:
        print("WARNING: PRDC requires ≥2 images per directory — returning None.")
        return None
    k_gen = min(k, n_gen - 1)
    k_ref = min(k, n_ref - 1)
    if k_gen < 1 or k_ref < 1:
        print("WARNING: PRDC requires at least 2 images per directory — returning None.")
        return None

    d_rr = _pairwise_sq_dists(ref_emb, ref_emb)
    d_ff = _pairwise_sq_dists(gen_emb, gen_emb)
    d_fr = _pairwise_sq_dists(gen_emb, ref_emb)

    np.fill_diagonal(d_rr, np.inf)
    np.fill_diagonal(d_ff, np.inf)

    r_ref = np.partition(d_rr, k_ref, axis=1)[:, k_ref]
    r_gen = np.partition(d_ff, k_gen, axis=1)[:, k_gen]

    # Precision
    min_d_fr = d_fr.min(axis=1)
    nn_ref = d_fr.argmin(axis=1)
    precision = float(np.mean(min_d_fr <= r_ref[nn_ref]))

    # Recall
    min_d_rf = d_fr.min(axis=0)
    nn_gen = d_fr.argmin(axis=0)
    recall = float(np.mean(min_d_rf <= r_gen[nn_gen]))

    # Density
    density = float(np.mean((d_fr <= r_ref[None, :]).sum(axis=1) / float(k_ref)))

    # Coverage
    coverage = float(np.mean(min_d_rf <= r_ref))

    return {
        "prdc_precision": precision,
        "prdc_recall": recall,
        "prdc_density": density,
        "prdc_coverage": coverage,
        "prdc_k": k,
    }


def compute_clip_cosine_similarity(
    gen_emb: np.ndarray,
    ref_emb: np.ndarray,
) -> Optional[float]:
    """
    Compute mean cosine similarity between generated and reference CLIP embeddings.

    Embeddings are L2-normalized 768-d CLS vectors from ViT-L/14, so
    cosine similarity reduces to a dot product.  Returns the mean over all
    cross-set pairs (n_gen × n_ref), giving a distribution-level similarity
    score in [-1, 1] (higher = more similar).
    """
    if gen_emb.shape[0] < 1 or ref_emb.shape[0] < 1:
        return None
    sim_matrix = gen_emb @ ref_emb.T  # (n_gen, n_ref)
    return float(sim_matrix.mean())


def compute_psnr_paired(
    generated_dir: str,
    baseline_dir: str,
) -> Optional[Dict]:
    """
    Compute mean PSNR between matched image pairs (same filename in both dirs).

    Matches images by sorted filename (0000.png vs 0000.png, etc.).  Warns
    and skips unmatched files when counts differ.

    Returns dict with keys psnr_mean, psnr_std, n_pairs, or None if no pairs found.
    """
    from PIL import Image as PILImage

    gen_paths = {p.name: p for p in sorted(Path(generated_dir).glob("*.png"))}
    base_paths = {p.name: p for p in sorted(Path(baseline_dir).glob("*.png"))}
    common = sorted(set(gen_paths) & set(base_paths))

    if not common:
        print("WARNING: PSNR — no matching filenames between generated and baseline dirs.")
        return None

    n_gen, n_base = len(gen_paths), len(base_paths)
    if n_gen != n_base:
        print(f"WARNING: PSNR — {n_gen} generated vs {n_base} baseline images; "
              f"computing on {len(common)} matched pairs.")

    psnr_values = []
    for name in common:
        gen_img = np.array(PILImage.open(gen_paths[name]).convert("RGB"), dtype=np.float64)
        base_img = np.array(PILImage.open(base_paths[name]).convert("RGB"), dtype=np.float64)
        mse = np.mean((gen_img - base_img) ** 2)
        psnr = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")
        psnr_values.append(psnr)

    finite = [v for v in psnr_values if np.isfinite(v)]
    return {
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(finite)) if finite else 0.0,
        "n_pairs": len(psnr_values),
    }


def compute_lpips_paired(
    generated_dir: str,
    baseline_dir: str,
) -> Optional[Dict]:
    """
    Compute mean LPIPS between matched image pairs using AlexNet.

    Requires ``pip install lpips``.  Returns None gracefully if unavailable.
    Images are resized to 256×256 and normalised to [-1, 1].

    Returns dict with keys lpips_mean, lpips_std, n_pairs.
    """
    try:
        import lpips as lpips_lib
        import torch
    except ImportError:
        print("WARNING: lpips not installed — skipping LPIPS. "
              "Install with: pip install lpips")
        return None

    from PIL import Image as PILImage
    import torchvision.transforms as T

    loss_fn = lpips_lib.LPIPS(net="alex", verbose=False)
    loss_fn.eval()

    preprocess = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),                        # [0, 1]
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])

    gen_paths = {p.name: p for p in sorted(Path(generated_dir).glob("*.png"))}
    base_paths = {p.name: p for p in sorted(Path(baseline_dir).glob("*.png"))}
    common = sorted(set(gen_paths) & set(base_paths))

    if not common:
        print("WARNING: LPIPS — no matching filenames between generated and baseline dirs.")
        return None

    scores = []
    with torch.no_grad():
        for name in common:
            gen_t = preprocess(PILImage.open(gen_paths[name]).convert("RGB")).unsqueeze(0)
            base_t = preprocess(PILImage.open(base_paths[name]).convert("RGB")).unsqueeze(0)
            scores.append(float(loss_fn(gen_t, base_t).item()))

    return {
        "lpips_mean": float(np.mean(scores)),
        "lpips_std": float(np.std(scores)),
        "n_pairs": len(scores),
    }


def compute_cmmd_from_embeddings(
    gen_emb: np.ndarray,
    ref_emb: np.ndarray,
) -> Optional[float]:
    """
    Compute CMMD using RBF-kernel MMD over CLIP embeddings.
    """
    if gen_emb.shape[0] < 2 or ref_emb.shape[0] < 2:
        print("WARNING: CMMD requires ≥2 images per directory — returning None.")
        return None
    all_emb = np.concatenate([gen_emb, ref_emb], axis=0)
    dists = _pairwise_sq_dists(all_emb, all_emb)
    tri = dists[np.triu_indices_from(dists, k=1)]
    med = np.median(tri[tri > 0])
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    sigma2 = med
    gamma = 1.0 / (2.0 * sigma2)

    d_rr = _pairwise_sq_dists(ref_emb, ref_emb)
    d_ff = _pairwise_sq_dists(gen_emb, gen_emb)
    d_fr = _pairwise_sq_dists(gen_emb, ref_emb)

    k_rr = np.exp(-gamma * d_rr)
    k_ff = np.exp(-gamma * d_ff)
    k_fr = np.exp(-gamma * d_fr)

    mmd2 = float(k_rr.mean() + k_ff.mean() - 2.0 * k_fr.mean())
    return mmd2


def compute_sfid(
    generated_dir: str,
    reference_dir: str,
) -> Optional[float]:
    """
    Compute sFID using InceptionV3 spatial features from the Mixed_6e block.

    sFID uses global-average-pooled 768-d vectors from Mixed_6e (instead of
    the 2048-d final-pooling features used by FID) and is more sensitive to
    spatial structure.

    Requires torchvision and scipy.  Returns None gracefully if unavailable.

    Parameters
    ----------
    generated_dir : str | Path
        Directory containing generated PNG images.
    reference_dir : str | Path
        Directory containing reference PNG images.

    Returns
    -------
    float or None
    """
    try:
        import torch
        import torchvision.models as tv_models
        import torchvision.transforms as transforms
        from scipy.linalg import sqrtm as scipy_sqrtm
    except ImportError:
        print("WARNING: torchvision or scipy not installed — skipping sFID.")
        return None

    import numpy as np
    from PIL import Image as PILImage

    # Load InceptionV3 — try new-style weights API, fall back to legacy
    try:
        model = tv_models.inception_v3(
            weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1
        )
    except (AttributeError, TypeError):
        model = tv_models.inception_v3(pretrained=True)  # type: ignore[call-arg]
    model.eval()

    # Hook Mixed_6e to capture spatial features (N x 768 x 17 x 17)
    _captured: Dict = {}

    def _hook(module, input, output):  # noqa: ARG001
        # Global-average-pool spatial dims → (N, 768)
        _captured["feat"] = output.mean(dim=[2, 3]).detach().cpu().numpy()

    hook_handle = model.Mixed_6e.register_forward_hook(_hook)

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def _extract(img_dir: str) -> np.ndarray:
        paths = sorted(Path(img_dir).glob("*.png"))
        feats = []
        for p in paths:
            img = PILImage.open(p).convert("RGB")
            tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                model(tensor)
            feats.append(_captured["feat"].copy())
        if not feats:
            return np.zeros((0, 768), dtype=np.float64)
        return np.concatenate(feats, axis=0).astype(np.float64)

    try:
        feats_gen = _extract(str(generated_dir))
        feats_ref = _extract(str(reference_dir))
    finally:
        hook_handle.remove()

    if feats_gen.shape[0] < 2 or feats_ref.shape[0] < 2:
        print("WARNING: sFID requires ≥2 images per directory — returning None.")
        return None

    mu1 = feats_gen.mean(axis=0)
    mu2 = feats_ref.mean(axis=0)
    sigma1 = np.cov(feats_gen, rowvar=False)
    sigma2 = np.cov(feats_ref, rowvar=False)

    diff = mu1 - mu2
    # disp=False was deprecated in scipy 1.17 and removed in 1.18; use the
    # forward-compatible form and discard the (now-gone) error-estimate return.
    sqrtm_result = scipy_sqrtm(sigma1 @ sigma2)
    covmean = sqrtm_result[0] if isinstance(sqrtm_result, tuple) else sqrtm_result
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    sfid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return sfid


def compute_model_size(pipeline) -> Dict:
    """
    Compute DiT model size in GB and total parameter count (millions).

    Walks ``pipeline.mmdit.parameters()`` and sums actual tensor bytes.

    Parameters
    ----------
    pipeline : DiffusionPipeline
        A loaded pipeline; only ``pipeline.mmdit`` is inspected.

    Returns
    -------
    dict with keys ``size_gb`` (float) and ``total_params_M`` (float).
    """
    import mlx.core as mx

    total_bytes = 0
    total_params = 0

    def _walk(obj) -> None:
        nonlocal total_bytes, total_params
        if isinstance(obj, mx.array):
            total_bytes += obj.nbytes
            total_params += obj.size
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)

    _walk(pipeline.mmdit.parameters())
    return {
        "size_gb": total_bytes / 1e9,
        "total_params_M": total_params / 1e6,
    }


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(
    prompt_path: Path, max_count: int
) -> Tuple[List[str], Optional[List[int]]]:
    """
    Load up to max_count prompts from a CSV ('prompt' column) or a plain-text
    file (.txt, one prompt per line).  Falls back to three synthetic prompts if
    the file does not exist.

    Tab-separated .txt files with format ``seed<TAB>prompt`` are detected
    automatically; in that case a list of per-image seeds is also returned.

    Returns
    -------
    prompts : list of str
    seeds : list of int or None
        Per-image seeds when the file has the ``seed<TAB>prompt`` format,
        otherwise None (caller should use seed_base + img_idx).
    """
    if not prompt_path.exists():
        fallback = [
            "a photo of a cat",
            "abstract art with vibrant colors",
            "a landscape with mountains",
        ]
        return fallback[:max_count], None

    prompts: List[str] = []
    seeds: Optional[List[int]] = None

    if prompt_path.suffix.lower() == ".txt":
        with open(prompt_path, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]

        # Detect tab-separated seed<TAB>prompt format from first non-empty line
        if lines and "\t" in lines[0]:
            seeds = []
            for line in lines:
                if len(prompts) >= max_count:
                    break
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    try:
                        seeds.append(int(parts[0].strip()))
                        prompts.append(parts[1].strip())
                    except ValueError:
                        pass  # skip malformed lines
        else:
            for line in lines:
                if len(prompts) >= max_count:
                    break
                if line:
                    prompts.append(line)
        return prompts, seeds

    with open(prompt_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(prompts) >= max_count:
                break
            p = row.get("prompt", "").strip()
            if p:
                prompts.append(p)
    return prompts, None


# ---------------------------------------------------------------------------
# Section 1 — Image generation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Naive int8 quantization helpers
# ---------------------------------------------------------------------------

def _walk_mmdit_linears(mmdit):
    """
    Generator yielding (parent_obj, attr_name, full_name) for every
    nn.Linear / nn.QuantizedLinear in the DiT transformer blocks.

    Skips adaLN_modulation and any identity projections.
    Works on both pre- and post-quantization models.
    """
    import mlx.nn as nn

    def _walk_block(tb, prefix):
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layer = getattr(tb.attn, attr, None)
            if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                yield (tb.attn, attr, f"{prefix}.attn.{attr}")
        if hasattr(tb, "mlp"):
            for attr in ("fc1", "fc2"):
                layer = getattr(tb.mlp, attr, None)
                if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                    yield (tb.mlp, attr, f"{prefix}.mlp.{attr}")

    if hasattr(mmdit, "multimodal_transformer_blocks"):
        for i, block in enumerate(mmdit.multimodal_transformer_blocks):
            yield from _walk_block(block.image_transformer_block, f"mm{i}.img")
            yield from _walk_block(block.text_transformer_block, f"mm{i}.txt")

    if hasattr(mmdit, "unified_transformer_blocks"):
        for i, block in enumerate(mmdit.unified_transformer_blocks):
            yield from _walk_block(block.transformer_block, f"uni{i}")


def inject_weights_naive_int8(
    pipeline,
    group_size: int = 64,
    bits: int = 8,
) -> int:
    """
    Quantize all eligible nn.Linear weights in the DiT to int8 using
    mlx.quantize and replace them with nn.QuantizedLinear in-place.

    Layers with in_features < max(128, group_size) are skipped (MLX
    minimum-column constraint).

    Returns the count of injected layers.
    """
    import mlx.core as mx
    import mlx.nn as nn

    mmdit = pipeline.mmdit
    min_cols = max(128, group_size)
    pending = []
    count = 0

    for parent, attr, full_name in _walk_mmdit_linears(mmdit):
        layer = getattr(parent, attr)
        w = layer.weight
        in_features = w.shape[1]
        out_features = w.shape[0]

        if in_features < min_cols:
            print(f"WARNING: Skipping {full_name} "
                  f"(in_features={in_features} < {min_cols})")
            continue

        w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        has_bias = getattr(layer, "bias", None) is not None

        ql = nn.QuantizedLinear(
            in_features, out_features,
            bias=has_bias, group_size=group_size, bits=bits,
        )
        ql.weight = w_q
        ql.scales = scales
        ql.biases = biases
        if has_bias:
            ql.bias = layer.bias

        setattr(parent, attr, ql)
        pending.extend([ql.weight, ql.scales, ql.biases])
        count += 1

    if pending:
        mx.eval(*pending)

    return count


class _DynamicInt8ActLayer:
    """
    Proxy that applies dynamic per-tensor symmetric int8 fake-quantization
    to the input activation before forwarding to the wrapped layer.

    scale = max(|x|) / 127
    x_q   = round(x / scale).clip(-127, 127) * scale
    """

    def __init__(self, layer):
        self.layer = layer

    def __call__(self, x):
        import mlx.core as mx
        scale = float(mx.max(mx.abs(x)).item()) / 127.0
        if scale < 1e-8:
            return self.layer(x)
        x = mx.clip(mx.round(x / scale), -127, 127) * scale
        return self.layer(x)

    def __getattr__(self, name):
        return getattr(self.layer, name)


def apply_dynamic_int8_act_hooks(mmdit):
    """
    Wrap every walked linear layer with _DynamicInt8ActLayer.

    Returns (proxies, patches) where patches is a list of
    (parent, attr, original_layer) tuples used for cleanup.
    """
    proxies = []
    patches = []

    for parent, attr, _ in _walk_mmdit_linears(mmdit):
        layer = getattr(parent, attr)
        proxy = _DynamicInt8ActLayer(layer)
        setattr(parent, attr, proxy)
        proxies.append(proxy)
        patches.append((parent, attr, layer))

    return proxies, patches


def remove_dynamic_int8_act_hooks(patches) -> None:
    """Restore original layers from (parent, attr, original) patch tuples."""
    for parent, attr, original in patches:
        setattr(parent, attr, original)


def _load_pipeline(
    config: str,
    adaround_output: Optional[Path],
    adaround_act_config: Optional[Path],
    mlx_int4: bool = False,
    group_size: int = 64,
    poly_schedule: Optional[Dict] = None,
    lut_schedule: Optional[Dict] = None,
    poly_margin: float = 1.0,
):
    """
    Load DiffusionPipeline and apply quantization config.

    Returns (pipeline, quant_ctx) where quant_ctx is a dict with keys:
      proxies, act_quant_patches, step_keys_sorted
    needed for V2 activation-quantized inference.
    """
    from diffusionkit.mlx import DiffusionPipeline

    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()

    quant_ctx = {
        "proxies": [],
        "act_quant_patches": [],
        "step_keys_sorted": [],
        "remove_act_fn": None,
    }

    if config == "fp16":
        return pipeline, quant_ctx

    if config == "naive_int8":
        inject_weights_naive_int8(pipeline, group_size=group_size)
        _, patches = apply_dynamic_int8_act_hooks(pipeline.mmdit)
        quant_ctx["act_quant_patches"] = patches
        quant_ctx["remove_act_fn"] = remove_dynamic_int8_act_hooks
        return pipeline, quant_ctx

    # Weight injection (adaround_w4 / adaround_w4a8 / taqdit_w4a8 / mlx_int4)
    if adaround_output is not None:
        from src.load_adaround_model import (
            load_adaround_weights,
            inject_weights,
            inject_weights_mlx_int4,
            apply_act_quant_hooks,
        )
        _, quant_weights = load_adaround_weights(adaround_output)

        if mlx_int4:
            inject_weights_mlx_int4(pipeline, quant_weights, group_size=group_size)
        else:
            inject_weights(pipeline, quant_weights)

    # Activation quantization hooks
    # Supports three modes (in priority order):
    #   1. --adaround-act-config  : static per-timestep config JSON
    #   2. --poly-schedule        : sigma-polynomial clipping schedule
    #   3. --lut-schedule         : per-timestep lookup table schedule
    # Modes 2 and 3 can be combined (poly takes priority per-layer).
    act_config_path = adaround_act_config
    per_timestep: Dict = {}
    outlier_config: Dict = {}
    if act_config_path is not None:
        with open(act_config_path) as f:
            quant_cfg = json.load(f)
        per_timestep = quant_cfg.get("per_timestep", {})
        outlier_config = quant_cfg.get("outlier_config", {})

    if act_config_path is not None or poly_schedule is not None or lut_schedule is not None:
        from src.load_adaround_model import apply_act_quant_hooks
        step_keys_sorted = sorted(int(k) for k in per_timestep.keys())
        proxies, patches = apply_act_quant_hooks(
            pipeline.mmdit, per_timestep, outlier_config,
            poly_schedule=poly_schedule,
            lut_schedule=lut_schedule,
        )
        if poly_margin != 1.0:
            for proxy in proxies:
                proxy.poly_margin = poly_margin
        quant_ctx["proxies"] = proxies
        quant_ctx["act_quant_patches"] = patches
        quant_ctx["step_keys_sorted"] = step_keys_sorted

    return pipeline, quant_ctx


def _generate_single_image(
    pipeline,
    quant_ctx: Dict,
    prompt: str,
    seed: int,
    num_steps: int,
    cfg_scale: float,
):
    """
    Generate one image using the given pipeline + quant_ctx.
    Returns a PIL.Image.
    """
    proxies = quant_ctx.get("proxies", [])
    step_keys_sorted = quant_ctx.get("step_keys_sorted", [])

    if proxies:
        from src.load_adaround_model import run_act_quant_inference
        return run_act_quant_inference(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt="",
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            proxies=proxies,
            step_keys_sorted=step_keys_sorted,
        )
    else:
        images, _ = pipeline.generate_image(
            prompt,
            cfg_weight=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            negative_text="",
        )
        return images


def _load_image_stats(output_dir: Path) -> Dict[int, Dict]:
    """Load per-image stats from image_stats.jsonl, keyed by img_idx."""
    stats_path = output_dir / "image_stats.jsonl"
    saved: Dict[int, Dict] = {}
    if stats_path.exists():
        with open(stats_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    saved[entry["img_idx"]] = entry
    return saved


def _append_image_stat(output_dir: Path, img_idx: int, elapsed_s: float,
                        metal_peak_mb: float, rss_peak_mb: float) -> None:
    """Append one per-image stat record to image_stats.jsonl."""
    entry = {
        "img_idx": img_idx,
        "elapsed_s": elapsed_s,
        "metal_peak_mb": metal_peak_mb,
        "rss_peak_mb": rss_peak_mb,
    }
    with open(output_dir / "image_stats.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def generate_images(
    config: str,
    prompts: List[str],
    output_dir: Path,
    num_steps: int,
    cfg_scale: float,
    seed_base: int,
    warmup: int,
    resume: bool,
    adaround_output: Optional[Path] = None,
    adaround_act_config: Optional[Path] = None,
    mlx_int4: bool = False,
    group_size: int = 64,
    poly_schedule: Optional[Dict] = None,
    lut_schedule: Optional[Dict] = None,
    poly_margin: float = 1.0,
    seeds: Optional[List[int]] = None,
) -> Tuple[List[float], Dict]:
    """
    Generate images for all prompts and return timing + memory stats.

    Pipeline is reloaded per image (mirrors generate_calibration_data.py)
    for consistent memory measurements.

    Returns
    -------
    timings : list of float
        Per-image wall-clock seconds (all images, including warmup).
    memory_stats : dict
        peak_metal_mb, peak_rss_mb.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load any previously saved per-image stats (populated when resuming)
    saved_stats = _load_image_stats(output_dir)

    timings: List[float] = []
    peak_metal_mb = 0.0
    peak_rss_mb = 0.0

    # Recover aggregate peaks from previously completed images
    for stat in saved_stats.values():
        peak_metal_mb = max(peak_metal_mb, stat.get("metal_peak_mb", 0.0))
        peak_rss_mb = max(peak_rss_mb, stat.get("rss_peak_mb", 0.0))

    total = len(prompts)
    completed = 0

    pbar = tqdm(enumerate(prompts), total=len(prompts),
                desc=f"  {config}", unit="img", dynamic_ncols=True)
    for img_idx, prompt in pbar:
        img_path = images_dir / f"{img_idx:04d}.png"
        if resume and img_path.exists():
            saved = saved_stats.get(img_idx)
            if saved:
                timings.append(saved["elapsed_s"])
                print(f"  [resume] skipping {img_idx:04d}.png "
                      f"(saved: {saved['elapsed_s']:.1f}s)")
            else:
                timings.append(0.0)  # pre-new-format run; no timing available
                print(f"  [resume] skipping {img_idx:04d}.png")
            continue

        seed = seeds[img_idx] if seeds is not None else seed_base + img_idx
        reset_metal_peak_memory()

        t0 = time.time()
        pipeline, quant_ctx = _load_pipeline(
            config, adaround_output, adaround_act_config, mlx_int4, group_size,
            poly_schedule=poly_schedule, lut_schedule=lut_schedule,
            poly_margin=poly_margin,
        )
        image = _generate_single_image(
            pipeline, quant_ctx, prompt, seed, num_steps, cfg_scale
        )
        elapsed = time.time() - t0

        # Remove hooks before pipeline goes out of scope
        if quant_ctx["act_quant_patches"]:
            remove_fn = quant_ctx.get("remove_act_fn")
            if remove_fn is not None:
                remove_fn(quant_ctx["act_quant_patches"])
            else:
                from src.load_adaround_model import remove_act_quant_hooks
                remove_act_quant_hooks(quant_ctx["act_quant_patches"])

        image.save(img_path)
        timings.append(elapsed)
        completed += 1

        # Memory
        mem = sample_metal_memory()
        img_metal_mb = mem["peak_mb"]
        img_rss_mb = sample_system_rss_mb()
        peak_metal_mb = max(peak_metal_mb, img_metal_mb)
        peak_rss_mb = max(peak_rss_mb, img_rss_mb)

        _append_image_stat(output_dir, img_idx, elapsed, img_metal_mb, img_rss_mb)

        # ETA
        measured = [t for t in timings if t > 0]
        measured_after_warmup = measured[warmup:]
        if measured_after_warmup:
            mean_t = np.mean(measured_after_warmup)
            remaining_imgs = total - img_idx - 1
            eta_min = remaining_imgs * mean_t / 60.0
            eta_str = f"ETA {eta_min:.1f} min"
        else:
            eta_str = "ETA --"

        metal_gb = mem["peak_mb"] / 1000.0
        print(f"  [{img_idx + 1}/{total}] {elapsed:.1f}s | {eta_str} | "
              f"peak_metal {metal_gb:.1f} GB")
        pbar.set_postfix({
            "s/img": f"{elapsed:.1f}",
            "ETA": eta_str,
            "metal_GB": f"{peak_metal_mb / 1024:.1f}",
        }, refresh=True)

    return timings, {"peak_metal_mb": peak_metal_mb, "peak_rss_mb": peak_rss_mb}


# ---------------------------------------------------------------------------
# Section 5 — Console output helpers
# ---------------------------------------------------------------------------

def _print_results(config: str, lat: Dict, mem: Dict, fidelity: Optional[Dict],
                   paired: Optional[Dict] = None) -> None:
    print(f"\n{'='*50}")
    print(f"=== Benchmark Results: {config} ===")
    print(f"{'='*50}")

    if lat.get("measured_images"):
        print(f"Latency (s):   mean={lat['mean_s']:.1f}  std={lat['std_s']:.1f}  "
              f"p50={lat['p50_s']:.1f}  p95={lat['p95_s']:.1f}")
    else:
        print("Latency:       (no measured images)")

    metal_gb = (mem.get("peak_metal_mb") or 0.0) / 1000.0
    rss_gb = (mem.get("peak_rss_mb") or 0.0) / 1000.0
    print(f"Memory:        peak_metal={metal_gb:.1f} GB  peak_rss={rss_gb:.1f} GB")

    if fidelity is not None:
        print(f"FID:           {fidelity['fid']:.4f}")
        sfid = fidelity.get("sfid")
        if sfid is not None:
            print(f"sFID:          {sfid:.4f}")
        print(f"IS:            {fidelity['isc_mean']:.2f} ± {fidelity['isc_std']:.2f}")
        print(f"KID:           {fidelity['kid_mean']:.5f} ± {fidelity['kid_std']:.5f}")
        prec = fidelity.get("precision")
        rec = fidelity.get("recall")
        if prec is not None:
            print(f"Precision:     {prec:.4f}")
        if rec is not None:
            print(f"Recall:        {rec:.4f}")
        cmmd = fidelity.get("cmmd")
        if cmmd is not None:
            print(f"CMMD:          {cmmd:.4f}")
        cos_sim = fidelity.get("clip_cosine_sim")
        if cos_sim is not None:
            print(f"CLIP cos sim:  {cos_sim:.4f}")
        prdc_p = fidelity.get("prdc_precision")
        prdc_r = fidelity.get("prdc_recall")
        prdc_d = fidelity.get("prdc_density")
        prdc_c = fidelity.get("prdc_coverage")
        if prdc_p is not None:
            print(f"PRDC Precision:{prdc_p:>10.4f}")
        if prdc_r is not None:
            print(f"PRDC Recall:   {prdc_r:>10.4f}")
        if prdc_d is not None:
            print(f"PRDC Density:  {prdc_d:>10.4f}")
        if prdc_c is not None:
            print(f"PRDC Coverage: {prdc_c:>10.4f}")
    else:
        print("FID/IS/KID:    skipped")

    if paired is not None:
        psnr = paired.get("psnr_mean")
        lpips_val = paired.get("lpips_mean")
        if psnr is not None:
            print(f"PSNR:          {psnr:.2f} dB  (std={paired.get('psnr_std', 0):.2f}, "
                  f"n={paired.get('n_pairs', 0)})")
        if lpips_val is not None:
            print(f"LPIPS:         {lpips_val:.4f}  (std={paired.get('lpips_std', 0):.4f}, "
                  f"n={paired.get('n_pairs', 0)})")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SD3-Medium pipeline: generate images + compute FID/IS/KID"
    )
    # Config
    parser.add_argument(
        "--config", type=str, default="fp16",
        help="Quantization config label (e.g. fp16, w4a8_rtn, adaround_w4a8_poly). "
             "Use 'fp16' or 'naive_int8' for built-in configs; any other string is a "
             "descriptive label and the actual quantization is determined by "
             "--adaround-output, --poly-schedule, etc.",
    )
    parser.add_argument("--adaround-output", type=Path, default=None,
                        help="AdaRound weights dir (from adaround_optimize.py)")
    parser.add_argument("--adaround-act-config", type=Path, default=None,
                        help="Activation quant config JSON (from analyze_activations.py)")
    parser.add_argument("--poly-schedule", type=Path, default=None,
                        help="Polynomial clipping schedule JSON (from generate_poly_schedule.py)")
    parser.add_argument("--lut-schedule", type=Path, default=None,
                        help="LUT clipping schedule JSON (from generate_lut_schedule.py)")
    parser.add_argument("--poly-margin", type=float, default=1.0,
                        help="Multiplier applied to poly/LUT clipping bounds (default: 1.0)")
    parser.add_argument("--taqdit-output", type=Path, default=None,
                        help="TaQ-DiT weights dir (reserved for future use)")
    parser.add_argument("--taqdit-act-config", type=Path, default=None,
                        help="TaQ-DiT act config JSON (reserved for future use)")

    # Generation
    parser.add_argument("--prompt-csv", type=Path, default=None,
                        help="CSV file with 'prompt' column (default: all_prompts.csv)")
    parser.add_argument("--num-images", type=int, default=150,
                        help="Number of images to generate (default: 150)")
    parser.add_argument("--num-steps", type=int, default=28,
                        help="Denoising steps per image (default: 28)")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="CFG guidance weight (default: 1.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; image i uses seed+i (default: 42)")
    parser.add_argument("--mlx-int4", action="store_true",
                        help="Inject AdaRound weights as native MLX int4")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Group size for MLX int4 (default: 64)")

    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"),
                        help="Root output dir (default: benchmark_results)")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="Reference image dir for FID/IS/KID (omit to skip metrics)")
    parser.add_argument("--generated-dir", type=Path, default=None,
                        help="Override generated image dir for metrics phase")
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="FP16 baseline image dir for paired metrics (PSNR + LPIPS)")
    parser.add_argument("--skip-clip-metrics", action="store_true",
                        help="Skip CLIP-based metrics (PRDC + CMMD)")
    parser.add_argument("--skip-paired-metrics", action="store_true",
                        help="Skip paired metrics (PSNR + LPIPS) even if --baseline-dir is set")

    # Phase control
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip image generation; compute metrics only")
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip FID/IS/KID; only generate + latency/memory")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup images excluded from latency stats (default: 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images whose PNG already exists in output_dir/images/")

    args = parser.parse_args()

    prompt_csv = args.prompt_csv or (_REPO / "all_prompts.csv")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load poly/LUT schedules once (shared across all images)
    poly_schedule: Optional[Dict] = None
    lut_schedule: Optional[Dict] = None
    if args.poly_schedule is not None:
        with open(args.poly_schedule) as f:
            poly_schedule = json.load(f)
        print(f"  Poly schedule: {args.poly_schedule} "
              f"(percentile={poly_schedule.get('percentile', 'unknown')})")
    if args.lut_schedule is not None:
        with open(args.lut_schedule) as f:
            lut_schedule = json.load(f)
        print(f"  LUT schedule:  {args.lut_schedule} "
              f"(percentile={lut_schedule.get('percentile', 'unknown')})")
    if args.poly_margin != 1.0:
        print(f"  Poly margin:   {args.poly_margin}×")

    generated_dir = args.generated_dir or (output_dir / "images")

    timings: List[float] = []
    memory_stats: Dict = {"peak_metal_mb": 0.0, "peak_rss_mb": 0.0}
    fidelity_result: Optional[Dict] = None
    paired_result: Optional[Dict] = None
    model_stats: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Phase 1 — Image generation
    # ------------------------------------------------------------------
    if not args.skip_generation:
        prompts, prompt_seeds = load_prompts(prompt_csv, args.num_images)
        print(f"=== Generating {len(prompts)} images (config={args.config}) ===")
        print(f"  Output: {output_dir / 'images'}")
        if prompt_seeds is not None:
            print("  Seeds: per-image seeds loaded from prompt file")
        if args.resume:
            print("  Resume mode: existing PNGs will be skipped")

        timings, memory_stats = generate_images(
            config=args.config,
            prompts=prompts,
            output_dir=output_dir,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            seed_base=args.seed,
            warmup=args.warmup,
            resume=args.resume,
            adaround_output=args.adaround_output,
            adaround_act_config=args.adaround_act_config,
            mlx_int4=args.mlx_int4,
            group_size=args.group_size,
            poly_schedule=poly_schedule,
            lut_schedule=lut_schedule,
            poly_margin=args.poly_margin,
            seeds=prompt_seeds,
        )

        # Compute model size once after generation (pipeline discarded per-image;
        # load once more just for parameter counting, then discard).
        print("\n=== Computing model size ===")
        try:
            pipeline_sz, _ = _load_pipeline(
                args.config, args.adaround_output, args.adaround_act_config,
                args.mlx_int4, args.group_size,
                poly_schedule=poly_schedule, lut_schedule=lut_schedule,
                poly_margin=args.poly_margin,
            )
            model_stats = compute_model_size(pipeline_sz)
            del pipeline_sz
            print(f"  size_gb={model_stats['size_gb']:.3f}  "
                  f"params={model_stats['total_params_M']:.0f}M")
        except Exception as e:
            print(f"WARNING: model size computation failed: {e}")

    # ------------------------------------------------------------------
    # Phase 2 — Fidelity metrics
    # ------------------------------------------------------------------
    if not args.skip_metrics and args.reference_dir is not None:
        ref_dir = args.reference_dir
        gen_dir = generated_dir

        if not gen_dir.exists():
            print(f"WARNING: generated_dir {gen_dir} does not exist — skipping metrics")
        elif not ref_dir.exists():
            print(f"WARNING: reference_dir {ref_dir} does not exist — skipping metrics")
        else:
            n_gen = len(list(gen_dir.glob("*.png")))
            n_ref = len(list(ref_dir.glob("*.png")))
            print(f"\n=== Computing FID/IS/KID/Precision "
                  f"({n_gen} generated vs {n_ref} reference) ===")
            raw = compute_fidelity_metrics(gen_dir, ref_dir)
            if raw is not None:
                fidelity_result = {
                    **raw,
                    "reference_dir": str(ref_dir),
                    "num_reference_images": n_ref,
                    "num_generated_images": n_gen,
                }

            print(f"\n=== Computing sFID ===")
            sfid_val = compute_sfid(str(gen_dir), str(ref_dir))
            if fidelity_result is not None:
                fidelity_result["sfid"] = sfid_val
            elif sfid_val is not None:
                fidelity_result = {"sfid": sfid_val}

            if not args.skip_clip_metrics:
                print(f"\n=== Computing CLIP metrics (PRDC + CMMD) ===")
                cache_path = output_dir / "embeddings.npz"
                clip_cache = _get_clip_embeddings_with_cache(
                    str(gen_dir), str(ref_dir), cache_path
                )
                if clip_cache is not None:
                    gen_emb = clip_cache["generated_embeddings"]
                    ref_emb = clip_cache["reference_embeddings"]
                    prdc = compute_prdc_metrics(gen_emb, ref_emb, k=5)
                    cmmd_val = compute_cmmd_from_embeddings(gen_emb, ref_emb)
                    cos_sim = compute_clip_cosine_similarity(gen_emb, ref_emb)
                    if fidelity_result is None:
                        fidelity_result = {}
                    if prdc is not None:
                        fidelity_result.update(prdc)
                    if cmmd_val is not None:
                        fidelity_result["cmmd"] = cmmd_val
                    if cos_sim is not None:
                        fidelity_result["clip_cosine_sim"] = cos_sim
                    fidelity_result["clip_model"] = clip_cache.get("clip_model_id")
                else:
                    print("WARNING: CLIP metrics unavailable — skipping PRDC/CMMD.")

    # ------------------------------------------------------------------
    # Phase 3 — Paired metrics (PSNR + LPIPS vs FP16 baseline)
    # ------------------------------------------------------------------
    if not args.skip_paired_metrics and args.baseline_dir is not None:
        base_dir = args.baseline_dir
        if not base_dir.exists():
            print(f"WARNING: baseline_dir {base_dir} does not exist — skipping paired metrics")
        else:
            n_gen = len(list(generated_dir.glob("*.png")))
            n_base = len(list(base_dir.glob("*.png")))
            print(f"\n=== Computing paired metrics "
                  f"({n_gen} generated vs {n_base} baseline) ===")
            psnr_result = compute_psnr_paired(str(generated_dir), str(base_dir))
            lpips_result = compute_lpips_paired(str(generated_dir), str(base_dir))
            paired_result = {}
            if psnr_result is not None:
                paired_result.update(psnr_result)
            if lpips_result is not None:
                paired_result["lpips_mean"] = lpips_result["lpips_mean"]
                paired_result["lpips_std"] = lpips_result["lpips_std"]
            paired_result["baseline_dir"] = str(base_dir)

    # ------------------------------------------------------------------
    # Latency stats
    # ------------------------------------------------------------------
    # Filter out resume-skipped images (timings == 0.0) before computing stats
    real_timings = [t for t in timings if t > 0.0]
    lat_stats = compute_latency_stats(real_timings, warmup=args.warmup)

    # ------------------------------------------------------------------
    # Write benchmark.json
    # ------------------------------------------------------------------
    benchmark = {
        "config": args.config,
        "num_images": args.num_images,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "adaround_output": str(args.adaround_output) if args.adaround_output else None,
        "poly_schedule": str(args.poly_schedule) if args.poly_schedule else None,
        "poly_schedule_percentile": poly_schedule.get("percentile") if poly_schedule else None,
        "lut_schedule": str(args.lut_schedule) if args.lut_schedule else None,
        "lut_schedule_percentile": lut_schedule.get("percentile") if lut_schedule else None,
        "poly_margin": args.poly_margin if args.poly_margin != 1.0 else None,
        "latency": lat_stats,
        "memory": memory_stats,
        "model": model_stats,
        "fidelity": fidelity_result,
        "paired": paired_result,
    }

    json_path = output_dir / "benchmark.json"
    with open(json_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n✓ benchmark.json → {json_path}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    _print_results(args.config, lat_stats, memory_stats, fidelity_result, paired=paired_result)
    print("\n✓ Complete")


if __name__ == "__main__":
    main()
