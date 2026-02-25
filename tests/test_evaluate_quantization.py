"""
Tests for src/evaluate_quantization.py

All tests use synthetic arrays — no DiffusionKit pipeline or real images.
"""

import numpy as np
import pytest

from src.evaluate_quantization import (
    psnr,
    ssim_simple,
    make_comparison_grid,
    try_load_lpips,
)


# ---------------------------------------------------------------------------
# TestPsnr
# ---------------------------------------------------------------------------

class TestPsnr:

    def test_identical_images_returns_inf(self):
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        result = psnr(img, img.copy())
        assert result == float("inf")

    def test_positive_and_finite_for_different_images(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        result = psnr(a, b)
        assert np.isfinite(result)
        assert result > 0.0

    def test_all_zero_vs_all_max_is_finite(self):
        a = np.zeros((16, 16, 3), dtype=np.uint8)
        b = np.full((16, 16, 3), 255, dtype=np.uint8)
        result = psnr(a, b)
        # PSNR = 10*log10(255^2 / 255^2) = 0.0 dB for worst-case error
        assert np.isfinite(result)
        assert result >= 0.0

    def test_max_error_gives_low_psnr(self):
        """Max possible error → lowest (but still positive finite) PSNR."""
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.full((4, 4, 3), 255, dtype=np.uint8)
        low_psnr = psnr(a, b)
        assert low_psnr < 10.0   # should be ~ 0 dB for max error

    def test_higher_for_smaller_difference(self):
        base = np.full((8, 8, 3), 128, dtype=np.uint8)
        small_diff = base.copy()
        small_diff[0, 0, 0] = 129    # tiny perturbation
        large_diff = base.copy()
        large_diff[:, :, :] = 0      # large perturbation
        assert psnr(base, small_diff) > psnr(base, large_diff)

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        assert psnr(a, b) == pytest.approx(psnr(b, a), rel=1e-10)

    def test_works_with_2d_grayscale(self):
        a = np.full((8, 8), 128, dtype=np.uint8)
        b = np.full((8, 8), 100, dtype=np.uint8)
        result = psnr(a, b)
        assert np.isfinite(result)
        assert result > 0.0

    def test_known_value(self):
        """PSNR for a constant difference should match analytical formula."""
        # All pixels differ by 1 → MSE = 1 → PSNR = 10*log10(255^2) ≈ 48.13 dB
        a = np.full((16, 16, 3), 100, dtype=np.uint8)
        b = np.full((16, 16, 3), 101, dtype=np.uint8)
        expected = 10.0 * np.log10(255.0 ** 2 / 1.0)
        assert psnr(a, b) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# TestSsimSimple
# ---------------------------------------------------------------------------

class TestSsimSimple:

    def test_identical_images_returns_one(self):
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        result = ssim_simple(img, img.copy())
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_returns_value_in_valid_range(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        result = ssim_simple(a, b)
        assert -1.5 <= result <= 1.0 + 1e-5

    def test_higher_for_more_similar_images(self):
        base = np.full((16, 16, 3), 128, dtype=np.uint8)
        close = base.copy()
        close[0, 0, 0] = 130   # very similar
        far = np.zeros((16, 16, 3), dtype=np.uint8)  # very different

        assert ssim_simple(base, close) > ssim_simple(base, far)

    def test_symmetric(self):
        rng = np.random.default_rng(3)
        a = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        assert ssim_simple(a, b) == pytest.approx(ssim_simple(b, a), abs=1e-10)

    def test_constant_image_vs_different_constant(self):
        """Two uniform images of different values: luminance differs, low SSIM."""
        a = np.full((8, 8, 3), 50, dtype=np.uint8)
        b = np.full((8, 8, 3), 200, dtype=np.uint8)
        result = ssim_simple(a, b)
        # Luminance component is below 1 since means differ significantly
        assert result < 1.0

    def test_works_with_2d_grayscale(self):
        a = np.full((8, 8), 128, dtype=np.uint8)
        b = np.full((8, 8), 100, dtype=np.uint8)
        result = ssim_simple(a, b)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# TestMakeComparisonGrid
# ---------------------------------------------------------------------------

class TestMakeComparisonGrid:

    def _make_image(self, h: int, w: int, val: int) -> np.ndarray:
        return np.full((h, w, 3), val, dtype=np.uint8)

    def test_output_shape_n_images_by_n_configs(self):
        h, w = 4, 4
        images = {
            "fp16":     [self._make_image(h, w, 100) for _ in range(3)],
            "adaround": [self._make_image(h, w, 150) for _ in range(3)],
            "taqdit":   [self._make_image(h, w, 200) for _ in range(3)],
        }
        labels = ["fp16", "adaround", "taqdit"]
        grid = make_comparison_grid(images, labels)
        # Rows: n_images * h, Cols: n_configs * w
        assert grid.shape == (3 * h, 3 * w, 3)

    def test_single_image_single_config(self):
        img = self._make_image(8, 8, 42)
        grid = make_comparison_grid({"fp16": [img]}, ["fp16"])
        assert grid.shape == (8, 8, 3)
        np.testing.assert_array_equal(grid, img)

    def test_config_values_appear_in_correct_columns(self):
        h, w = 2, 2
        images = {
            "a": [self._make_image(h, w, 10)],
            "b": [self._make_image(h, w, 20)],
        }
        grid = make_comparison_grid(images, ["a", "b"])
        # Left column should be 10, right column 20
        np.testing.assert_array_equal(grid[:, :w, :], np.full((h, w, 3), 10))
        np.testing.assert_array_equal(grid[:, w:, :], np.full((h, w, 3), 20))

    def test_skips_configs_not_in_label_list(self):
        h, w = 2, 2
        images = {
            "fp16":     [self._make_image(h, w, 1)],
            "adaround": [self._make_image(h, w, 2)],
            "taqdit":   [self._make_image(h, w, 3)],
        }
        # Only include two of the three configs
        grid = make_comparison_grid(images, ["fp16", "taqdit"])
        assert grid.shape == (h, 2 * w, 3)

    def test_multiple_images_stacked_vertically(self):
        h, w = 3, 3
        images = {
            "fp16": [
                self._make_image(h, w, i * 10)
                for i in range(4)
            ]
        }
        grid = make_comparison_grid(images, ["fp16"])
        assert grid.shape == (4 * h, w, 3)


# ---------------------------------------------------------------------------
# TestTryLoadLpips
# ---------------------------------------------------------------------------

class TestTryLoadLpips:

    def test_returns_tuple_of_two(self):
        fn, available = try_load_lpips()
        assert isinstance(available, bool)
        if available:
            assert callable(fn)
        else:
            assert fn is None

    def test_lpips_fn_returns_float_when_available(self):
        fn, available = try_load_lpips()
        if not available:
            pytest.skip("lpips not installed")
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        result = fn(a, b)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_lpips_zero_for_identical_when_available(self):
        fn, available = try_load_lpips()
        if not available:
            pytest.skip("lpips not installed")
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = fn(img, img.copy())
        assert result == pytest.approx(0.0, abs=1e-4)
