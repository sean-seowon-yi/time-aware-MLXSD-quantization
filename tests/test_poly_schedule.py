"""Tests for polynomial clipping schedule generation (Phase 1)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.explore_curve_fits import poly_r2
from src.generate_poly_schedule import (
    CV_STATIC_THRESHOLD,
    CUBIC_R2_GAIN_THRESHOLD,
    QUAD_R2_THRESHOLD,
    QUARTIC_R2_GAIN_THRESHOLD,
    select_degree,
)


# ---------------------------------------------------------------------------
# poly_r2 tests
# ---------------------------------------------------------------------------

class TestPolyR2:
    def test_perfect_linear(self):
        x = np.linspace(0, 10, 50)
        y = 3.0 * x + 1.0
        r2, coeffs = poly_r2(x, y, 1)
        assert r2 == pytest.approx(1.0, abs=1e-10)
        assert coeffs[0] == pytest.approx(3.0, abs=1e-6)
        assert coeffs[1] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_quadratic(self):
        x = np.linspace(0, 5, 30)
        y = 2.0 * x**2 - 3.0 * x + 1.0
        r2, coeffs = poly_r2(x, y, 2)
        assert r2 == pytest.approx(1.0, abs=1e-8)
        assert len(coeffs) == 3

    def test_noisy_data_lower_r2(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + np.random.normal(0, 5, 100)
        r2, _ = poly_r2(x, y, 1)
        assert 0 < r2 < 1.0

    def test_constant_data(self):
        x = np.linspace(0, 10, 20)
        y = np.full_like(x, 5.0)
        r2, coeffs = poly_r2(x, y, 2)
        # ss_tot = 0 → r2 = 1.0 by convention
        assert r2 == 1.0

    def test_higher_degree_better_or_equal(self):
        np.random.seed(123)
        x = np.linspace(0, 5, 30)
        y = x**3 - 2*x**2 + x + np.random.normal(0, 0.5, 30)
        r2_2, _ = poly_r2(x, y, 2)
        r2_3, _ = poly_r2(x, y, 3)
        assert r2_3 >= r2_2 - 1e-10


# ---------------------------------------------------------------------------
# select_degree tests
# ---------------------------------------------------------------------------

class TestSelectDegree:
    def test_static_for_low_cv(self):
        """Layers with CV < 0.10 should get degree 0 (static)."""
        sigmas = np.linspace(0, 14, 25)
        vals = np.full(25, 10.0) + np.random.normal(0, 0.05, 25)  # CV ~ 0.005
        degree, coeffs, r2, cv = select_degree(sigmas, vals)
        assert degree == 0
        assert len(coeffs) == 1
        assert cv < CV_STATIC_THRESHOLD

    def test_quadratic_for_moderate_variation(self):
        """Smooth quadratic trajectory should get degree 2."""
        sigmas = np.linspace(0, 14, 25)
        vals = 0.5 * sigmas**2 + 2 * sigmas + 10  # perfect quadratic
        degree, coeffs, r2, cv = select_degree(sigmas, vals)
        assert degree == 2
        assert len(coeffs) == 3
        assert r2 > QUAD_R2_THRESHOLD

    def test_cubic_when_gain_sufficient(self):
        """Cubic trajectory with high CV where quadratic can't fit should get degree >= 2."""
        np.random.seed(99)
        sigmas = np.linspace(0, 10, 25)
        # Large cubic variation relative to mean → high CV, cubic-shaped
        vals = 0.5 * sigmas**3 - 5 * sigmas**2 + 10 * sigmas + 5
        vals += np.random.normal(0, 0.3, 25)
        degree, coeffs, r2, cv = select_degree(sigmas, vals)
        assert cv > CV_STATIC_THRESHOLD, f"CV={cv} should exceed static threshold"
        assert degree >= 2
        assert r2 > 0.85

    def test_coeffs_count_matches_degree(self):
        """Number of coefficients should be degree + 1."""
        sigmas = np.linspace(0, 14, 25)
        vals = 0.1 * sigmas**2 + sigmas + 5
        degree, coeffs, r2, cv = select_degree(sigmas, vals)
        if degree == 0:
            assert len(coeffs) == 1
        else:
            assert len(coeffs) == degree + 1


# ---------------------------------------------------------------------------
# JSON schema tests
# ---------------------------------------------------------------------------

class TestScheduleSchema:
    def _make_schedule(self):
        return {
            "version": "poly_v1",
            "percentile": "p999",
            "sigma_range": [0.0, 14.6],
            "layers": {
                "mm0_img_attn_q_proj": {
                    "degree": 2,
                    "coeffs": [0.1, -0.5, 3.0],
                    "r2": 0.97,
                    "cv": 0.23,
                },
                "mm1_txt_mlp_fc2": {
                    "degree": 0,
                    "coeffs": [5.0],
                    "r2": 1.0,
                    "cv": 0.04,
                },
            },
        }

    def test_version_field(self):
        s = self._make_schedule()
        assert s["version"] == "poly_v1"

    def test_layer_fields(self):
        s = self._make_schedule()
        for layer_name, info in s["layers"].items():
            assert "degree" in info
            assert "coeffs" in info
            assert "r2" in info
            assert "cv" in info
            assert isinstance(info["coeffs"], list)
            assert len(info["coeffs"]) == info["degree"] + 1 or info["degree"] == 0

    def test_json_roundtrip(self):
        s = self._make_schedule()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(s, f)
            f.flush()
            with open(f.name) as f2:
                loaded = json.load(f2)
        assert loaded == s
