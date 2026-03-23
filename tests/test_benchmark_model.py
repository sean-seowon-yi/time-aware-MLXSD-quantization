"""Tests for paired and distributional metrics in benchmark_model."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.benchmark_model import (
    compute_clip_cosine_similarity,
    compute_psnr_paired,
    compute_lpips_paired,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_png(path: Path, array: np.ndarray) -> None:
    """Save uint8 HxWx3 numpy array as PNG."""
    Image.fromarray(array.astype(np.uint8)).save(path)


def _make_image_dir(tmp_path: Path, name: str, images: dict) -> Path:
    """Create a subdirectory with named PNG files from uint8 arrays."""
    d = tmp_path / name
    d.mkdir()
    for fname, arr in images.items():
        _save_png(d / fname, arr)
    return d


# ---------------------------------------------------------------------------
# compute_psnr_paired
# ---------------------------------------------------------------------------

class TestComputePsnrPaired:
    def test_identical_images_returns_inf(self, tmp_path):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        gen = _make_image_dir(tmp_path, "gen", {"0000.png": img})
        base = _make_image_dir(tmp_path, "base", {"0000.png": img})

        result = compute_psnr_paired(str(gen), str(base))

        assert result is not None
        assert result["n_pairs"] == 1
        assert result["psnr_mean"] == float("inf")

    def test_known_psnr_value(self, tmp_path):
        # MSE = 1.0 over all pixels → PSNR = 20*log10(255/1) ≈ 48.13 dB
        rng = np.random.default_rng(0)
        base_arr = rng.integers(1, 200, (64, 64, 3), dtype=np.uint8)
        # Add exactly ±1 noise so mean squared error ≈ 1
        noise = np.ones((64, 64, 3), dtype=np.int32)
        gen_arr = np.clip(base_arr.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        gen = _make_image_dir(tmp_path, "gen", {"0000.png": gen_arr})
        base = _make_image_dir(tmp_path, "base", {"0000.png": base_arr})

        result = compute_psnr_paired(str(gen), str(base))

        assert result is not None
        # MSE = 1.0 exactly → PSNR = 48.13 dB
        assert result["psnr_mean"] == pytest.approx(48.13, abs=0.1)
        assert result["n_pairs"] == 1

    def test_multiple_pairs(self, tmp_path):
        img_a = np.full((32, 32, 3), 100, dtype=np.uint8)
        img_b = np.full((32, 32, 3), 200, dtype=np.uint8)
        gen = _make_image_dir(tmp_path, "gen", {
            "0000.png": img_a,
            "0001.png": img_a,
        })
        base = _make_image_dir(tmp_path, "base", {
            "0000.png": img_a,
            "0001.png": img_b,
        })

        result = compute_psnr_paired(str(gen), str(base))

        assert result is not None
        assert result["n_pairs"] == 2
        # First pair: identical (inf), second pair: large difference
        assert result["psnr_mean"] == float("inf")

    def test_no_matching_files_returns_none(self, tmp_path):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        gen = _make_image_dir(tmp_path, "gen", {"a.png": img})
        base = _make_image_dir(tmp_path, "base", {"b.png": img})

        result = compute_psnr_paired(str(gen), str(base))

        assert result is None

    def test_partial_overlap_uses_common(self, tmp_path):
        img = np.full((32, 32, 3), 50, dtype=np.uint8)
        gen = _make_image_dir(tmp_path, "gen", {
            "0000.png": img,
            "0001.png": img,
        })
        base = _make_image_dir(tmp_path, "base", {
            "0000.png": img,         # only one match
        })

        result = compute_psnr_paired(str(gen), str(base))

        assert result is not None
        assert result["n_pairs"] == 1


# ---------------------------------------------------------------------------
# compute_lpips_paired
# ---------------------------------------------------------------------------

class TestComputeLpipsPaired:
    def test_returns_none_when_lpips_missing(self, tmp_path):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        gen = _make_image_dir(tmp_path, "gen", {"0000.png": img})
        base = _make_image_dir(tmp_path, "base", {"0000.png": img})

        with patch.dict("sys.modules", {"lpips": None, "torch": None}):
            result = compute_lpips_paired(str(gen), str(base))

        assert result is None

    def test_returns_none_on_no_matching_files(self, tmp_path):
        """No common filenames → None (mocking lpips to avoid install requirement)."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        gen = _make_image_dir(tmp_path, "gen", {"a.png": img})
        base = _make_image_dir(tmp_path, "base", {"b.png": img})

        try:
            import lpips  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("lpips not installed")

        result = compute_lpips_paired(str(gen), str(base))
        assert result is None

    def test_identical_images_low_lpips(self, tmp_path):
        try:
            import lpips  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("lpips not installed")

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        gen = _make_image_dir(tmp_path, "gen", {"0000.png": img})
        base = _make_image_dir(tmp_path, "base", {"0000.png": img})

        result = compute_lpips_paired(str(gen), str(base))

        assert result is not None
        assert result["lpips_mean"] == pytest.approx(0.0, abs=1e-4)
        assert result["n_pairs"] == 1


# ---------------------------------------------------------------------------
# compute_clip_cosine_similarity
# ---------------------------------------------------------------------------

class TestComputeClipCosineSimilarity:
    def test_single_identical_vector_returns_one(self):
        # A single L2-normalised vector vs itself → sim_matrix = [[1]] → mean = 1
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        result = compute_clip_cosine_similarity(emb, emb)

        assert result is not None
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_cross_set_mean_matches_numpy(self):
        rng = np.random.default_rng(42)
        gen = rng.standard_normal((8, 768)).astype(np.float32)
        ref = rng.standard_normal((6, 768)).astype(np.float32)
        gen /= np.linalg.norm(gen, axis=1, keepdims=True)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)

        expected = float((gen @ ref.T).mean())
        result = compute_clip_cosine_similarity(gen, ref)

        assert result == pytest.approx(expected, abs=1e-6)

    def test_orthogonal_embeddings_near_zero(self):
        # Construct two orthogonal sets: e1 all in first half of dims, e2 in second
        d = 768
        e1 = np.zeros((1, d), dtype=np.float32)
        e1[0, : d // 2] = 1.0 / np.sqrt(d // 2)
        e2 = np.zeros((1, d), dtype=np.float32)
        e2[0, d // 2 :] = 1.0 / np.sqrt(d - d // 2)

        result = compute_clip_cosine_similarity(e1, e2)

        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_returns_none_for_empty_input(self):
        emb = np.zeros((0, 768), dtype=np.float32)
        ref = np.ones((10, 768), dtype=np.float32)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)

        result = compute_clip_cosine_similarity(emb, ref)

        assert result is None

    def test_scalar_range(self):
        rng = np.random.default_rng(7)
        gen = rng.standard_normal((20, 768)).astype(np.float32)
        ref = rng.standard_normal((15, 768)).astype(np.float32)
        gen /= np.linalg.norm(gen, axis=1, keepdims=True)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)

        result = compute_clip_cosine_similarity(gen, ref)

        assert result is not None
        assert -1.0 <= result <= 1.0

    def test_mean_over_all_pairs(self):
        # 2×2 case: verify mean(sim_matrix) directly
        e1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        e2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        # sim_matrix = [[1, 0], [0, 1]] → mean = 0.5
        result = compute_clip_cosine_similarity(e1, e2)
        assert result == pytest.approx(0.5, abs=1e-6)
