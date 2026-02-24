"""
Tests for src/analyze_activations.py

All tests use synthetic per-channel arrays — no model loading, no filesystem I/O.
"""

import numpy as np
import pytest

from src.analyze_activations import identify_outlier_channels


# ---------------------------------------------------------------------------
# TestIdentifyOutlierChannels
# ---------------------------------------------------------------------------

class TestIdentifyOutlierChannels:

    def test_clear_outliers_detected(self):
        """Three channels with 10× range should be flagged as outliers."""
        # 100 normal channels with range ≈ 1.0
        avg_min = np.full(103, -0.5, dtype=np.float32)
        avg_max = np.full(103, 0.5, dtype=np.float32)
        # 3 outlier channels with range ≈ 10.0
        avg_min[-3:] = -5.0
        avg_max[-3:] = 5.0

        result = identify_outlier_channels(avg_min, avg_max)

        assert result, "Expected outliers to be detected"
        assert set(result["outlier_indices"]) == {100, 101, 102}

    def test_returns_all_required_keys(self):
        avg_min = np.array([-0.5] * 98 + [-5.0, -5.0], dtype=np.float32)
        avg_max = np.array([0.5] * 98 + [5.0, 5.0], dtype=np.float32)

        result = identify_outlier_channels(avg_min, avg_max)

        assert "outlier_indices" in result
        assert "multiplier_vector" in result
        assert "scale_normal" in result
        assert "scale_outlier" in result

    def test_multiplier_vector_length_matches_channels(self):
        n_ch = 64
        avg_min = np.full(n_ch, -0.5, dtype=np.float32)
        avg_max = np.full(n_ch, 0.5, dtype=np.float32)
        avg_min[-4:] = -8.0
        avg_max[-4:] = 8.0

        result = identify_outlier_channels(avg_min, avg_max)

        assert len(result["multiplier_vector"]) == n_ch

    def test_scale_normal_less_than_scale_outlier(self):
        avg_min = np.full(100, -1.0, dtype=np.float32)
        avg_max = np.full(100, 1.0, dtype=np.float32)
        avg_min[-5:] = -10.0
        avg_max[-5:] = 10.0

        result = identify_outlier_channels(avg_min, avg_max)

        assert result["scale_outlier"] > result["scale_normal"]

    def test_custom_threshold_multiplier(self):
        """Higher threshold_multiplier requires more extreme outliers."""
        avg_min = np.full(100, -0.5, dtype=np.float32)
        avg_max = np.full(100, 0.5, dtype=np.float32)
        # Channels with 3× median range — detected at threshold 2.5 but not at 4.0
        avg_min[-3:] = -1.5
        avg_max[-3:] = 1.5

        result_low = identify_outlier_channels(avg_min, avg_max, threshold_multiplier=2.5)
        result_high = identify_outlier_channels(avg_min, avg_max, threshold_multiplier=4.0)

        assert result_low, "Should detect outliers at threshold 2.5"
        assert result_high == {}, "Should NOT detect outliers at threshold 4.0"


# ---------------------------------------------------------------------------
# TestNoOutliersFound
# ---------------------------------------------------------------------------

class TestNoOutliersFound:

    def test_uniform_range_returns_empty(self):
        avg_min = np.full(64, -1.0, dtype=np.float32)
        avg_max = np.full(64, 1.0, dtype=np.float32)

        result = identify_outlier_channels(avg_min, avg_max)

        assert result == {}

    def test_near_zero_median_range_returns_empty(self):
        """All channels near-zero → median_range < 1e-6 → safe fallback."""
        avg_min = np.zeros(32, dtype=np.float32)
        avg_max = np.full(32, 1e-8, dtype=np.float32)

        result = identify_outlier_channels(avg_min, avg_max)

        assert result == {}

    def test_single_channel_returns_empty(self):
        """Single channel: no comparison possible → no outliers."""
        avg_min = np.array([-5.0], dtype=np.float32)
        avg_max = np.array([5.0], dtype=np.float32)

        result = identify_outlier_channels(avg_min, avg_max)

        assert result == {}

    def test_two_channels_equal_range_returns_empty(self):
        avg_min = np.array([-1.0, -1.0], dtype=np.float32)
        avg_max = np.array([1.0, 1.0], dtype=np.float32)

        result = identify_outlier_channels(avg_min, avg_max)

        assert result == {}


# ---------------------------------------------------------------------------
# TestMultiplierVector
# ---------------------------------------------------------------------------

class TestMultiplierVector:

    def test_normal_channels_have_multiplier_one(self):
        avg_min = np.full(100, -0.5, dtype=np.float32)
        avg_max = np.full(100, 0.5, dtype=np.float32)
        avg_min[-2:] = -5.0
        avg_max[-2:] = 5.0

        result = identify_outlier_channels(avg_min, avg_max)
        mv = np.array(result["multiplier_vector"])

        # All non-outlier channels should be 1.0
        np.testing.assert_array_equal(mv[:98], np.ones(98))

    def test_outlier_channels_have_multiplier_at_least_one(self):
        avg_min = np.full(100, -0.5, dtype=np.float32)
        avg_max = np.full(100, 0.5, dtype=np.float32)
        avg_min[-2:] = -5.0
        avg_max[-2:] = 5.0

        result = identify_outlier_channels(avg_min, avg_max)
        mv = np.array(result["multiplier_vector"])

        assert all(mv[-2:] >= 1.0), "Outlier multipliers must be >= 1.0"

    def test_multiplier_is_rounded_ratio(self):
        """
        multiplier = max(1.0, round(scale_outlier / scale_normal))
        For 100 channels at absmax=0.5 (qmax=127 → scale_normal=0.5/127≈0.00394)
        and 2 channels at absmax=5.0 (scale_outlier=5/127≈0.0394):
        ratio = 10.0 → multiplier = 10.
        """
        avg_min = np.full(100, -0.5, dtype=np.float32)
        avg_max = np.full(100, 0.5, dtype=np.float32)
        avg_min[-2:] = -5.0
        avg_max[-2:] = 5.0

        result = identify_outlier_channels(avg_min, avg_max)
        mv = np.array(result["multiplier_vector"])
        expected_multiplier = round(result["scale_outlier"] / result["scale_normal"])
        expected_multiplier = max(1.0, expected_multiplier)

        np.testing.assert_allclose(mv[-2:], expected_multiplier, atol=1e-5)

    def test_multiplier_vector_is_list(self):
        avg_min = np.full(50, -0.5, dtype=np.float32)
        avg_max = np.full(50, 0.5, dtype=np.float32)
        avg_min[-3:] = -10.0
        avg_max[-3:] = 10.0

        result = identify_outlier_channels(avg_min, avg_max)

        assert isinstance(result["multiplier_vector"], list)

    def test_outlier_indices_are_list_of_ints(self):
        avg_min = np.full(50, -0.5, dtype=np.float32)
        avg_max = np.full(50, 0.5, dtype=np.float32)
        avg_min[-3:] = -10.0
        avg_max[-3:] = 10.0

        result = identify_outlier_channels(avg_min, avg_max)

        assert isinstance(result["outlier_indices"], list)
        assert all(isinstance(i, int) for i in result["outlier_indices"])


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_negative_ranges_handled(self):
        """avg_min can be positive and avg_max larger — still computes range correctly."""
        avg_min = np.full(50, 1.0, dtype=np.float32)   # positive floor
        avg_max = np.full(50, 2.0, dtype=np.float32)   # range = 1.0
        avg_min[-2:] = 0.0
        avg_max[-2:] = 15.0  # range = 15.0 → outlier

        result = identify_outlier_channels(avg_min, avg_max)

        # Should detect the two wide-range channels
        assert result
        assert 48 in result["outlier_indices"] or 49 in result["outlier_indices"]

    def test_asymmetric_distribution(self):
        """Post-GELU layers have asymmetric distributions; outlier detection still works."""
        rng = np.random.default_rng(42)
        avg_min = -rng.uniform(0.1, 0.5, 100).astype(np.float32)
        avg_max = rng.uniform(0.5, 2.0, 100).astype(np.float32)
        # Add two extreme outlier channels
        avg_min[-2:] = -20.0
        avg_max[-2:] = 5.0

        result = identify_outlier_channels(avg_min, avg_max)

        assert result, "Should detect outlier channels with asymmetric distribution"
        assert 98 in result["outlier_indices"] or 99 in result["outlier_indices"]

    def test_bits_parameter_affects_scale(self):
        """Higher bits → finer quantization scale → smaller scale values."""
        avg_min = np.full(100, -1.0, dtype=np.float32)
        avg_max = np.full(100, 1.0, dtype=np.float32)
        avg_min[-3:] = -10.0
        avg_max[-3:] = 10.0

        result_a8 = identify_outlier_channels(avg_min, avg_max, bits=8)
        result_a4 = identify_outlier_channels(avg_min, avg_max, bits=4)

        # A4 has qmax=7 so scale is larger than A8 (qmax=127)
        assert result_a4["scale_normal"] > result_a8["scale_normal"]
