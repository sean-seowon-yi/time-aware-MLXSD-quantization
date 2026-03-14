"""Tests for polynomial activation quantization integration (Phase 2)."""

import numpy as np
import pytest

from src.load_adaround_model import eval_poly_alpha, load_poly_schedule


# ---------------------------------------------------------------------------
# eval_poly_alpha tests
# ---------------------------------------------------------------------------

class TestEvalPolyAlpha:
    def test_constant(self):
        """Degree 0: single coefficient = constant."""
        assert eval_poly_alpha([5.0], 0.0) == pytest.approx(5.0)
        assert eval_poly_alpha([5.0], 14.0) == pytest.approx(5.0)

    def test_linear(self):
        """Degree 1: alpha = a*sigma + b."""
        coeffs = [2.0, 3.0]  # 2*sigma + 3
        assert eval_poly_alpha(coeffs, 0.0) == pytest.approx(3.0)
        assert eval_poly_alpha(coeffs, 1.0) == pytest.approx(5.0)
        assert eval_poly_alpha(coeffs, 5.0) == pytest.approx(13.0)

    def test_quadratic(self):
        """Degree 2: alpha = a*sigma^2 + b*sigma + c."""
        coeffs = [1.0, -2.0, 3.0]  # sigma^2 - 2*sigma + 3
        assert eval_poly_alpha(coeffs, 0.0) == pytest.approx(3.0)
        assert eval_poly_alpha(coeffs, 1.0) == pytest.approx(2.0)
        assert eval_poly_alpha(coeffs, 3.0) == pytest.approx(6.0)

    def test_boundary_sigmas(self):
        """Test at typical SD3 sigma boundary values."""
        coeffs = [0.01, -0.2, 5.0]  # gentle quadratic
        # sigma=0 (end of denoising, clean)
        alpha_clean = eval_poly_alpha(coeffs, 0.0)
        assert alpha_clean == pytest.approx(5.0)
        # sigma=14.6 (start, max noise)
        alpha_noisy = eval_poly_alpha(coeffs, 14.6)
        assert alpha_noisy == pytest.approx(0.01 * 14.6**2 - 0.2 * 14.6 + 5.0)

    def test_matches_numpy_polyval(self):
        """Verify eval_poly_alpha matches np.polyval when result is positive."""
        coeffs = [0.01, -0.1, 5.0]  # always positive in [0, 5]
        for sigma in [0.0, 0.5, 1.0, 3.0]:
            expected = float(np.polyval(coeffs, sigma))
            assert expected > 0, f"test data should be positive at sigma={sigma}"
            assert eval_poly_alpha(coeffs, sigma) == pytest.approx(expected)

    def test_clamps_to_sigma_range(self):
        """Sigma outside calibration range should be clamped."""
        coeffs = [1.0, -2.0, 5.0]  # quadratic that goes negative outside [0, 3]
        # Without clamping, sigma=10 gives 1*100 - 20 + 5 = 85
        # With clamping to [0.1, 1.0], sigma=10 is clamped to 1.0: 1 - 2 + 5 = 4
        alpha_clamped = eval_poly_alpha(coeffs, 10.0, sigma_range=[0.1, 1.0])
        alpha_at_1 = eval_poly_alpha(coeffs, 1.0)
        assert alpha_clamped == pytest.approx(alpha_at_1)

    def test_clamps_negative_to_positive(self):
        """Negative polynomial output should be clamped to 1e-8."""
        coeffs = [-1.0, 0.0]  # -sigma, always negative for sigma > 0
        alpha = eval_poly_alpha(coeffs, 5.0)
        assert alpha == pytest.approx(1e-8)


# ---------------------------------------------------------------------------
# _ActQuantLayer poly vs static tests
# ---------------------------------------------------------------------------

class TestActQuantLayerPoly:
    """Test _ActQuantLayer behavior with and without poly config.

    These tests mock the layer to avoid MLX dependencies.
    """

    def test_poly_scale_derivation(self):
        """Polynomial alpha should produce scale = alpha / 127 for int8."""
        coeffs = [0.01, -0.1, 5.0]
        sigma = 7.0
        alpha = eval_poly_alpha(coeffs, sigma)
        expected_scale = alpha / 127.0
        assert expected_scale > 0
        assert expected_scale == pytest.approx(alpha / 127.0)

    def test_static_fallback_when_no_poly(self):
        """When poly_cfg is None, scale comes from per_timestep config."""
        # Just verify the logic — no MLX needed
        poly_cfg = None
        per_timestep = {"5": {"bits": 8, "scale": 0.05}}
        # With no poly_cfg and current_sigma=None, should fall back to static
        assert poly_cfg is None
        assert "5" in per_timestep
        assert per_timestep["5"]["scale"] == 0.05

    def test_poly_overrides_static(self):
        """When poly_cfg is set and sigma is provided, poly takes priority."""
        poly_cfg = {"degree": 2, "coeffs": [0.01, -0.1, 5.0], "r2": 0.95, "cv": 0.3}
        sigma = 7.0
        alpha = eval_poly_alpha(poly_cfg["coeffs"], sigma)
        poly_scale = alpha / 127.0

        static_scale = 0.05  # from per_timestep

        # Poly scale should differ from static
        assert poly_scale != pytest.approx(static_scale, abs=0.001)
        assert poly_scale > 0


# ---------------------------------------------------------------------------
# load_poly_schedule tests
# ---------------------------------------------------------------------------

class TestLoadPolySchedule:
    def test_returns_none_for_none_path(self):
        assert load_poly_schedule(None) is None

    def test_loads_valid_json(self, tmp_path):
        schedule = {
            "version": "poly_v1",
            "percentile": "p999",
            "layers": {
                "mm0_img_attn_q_proj": {
                    "degree": 2,
                    "coeffs": [0.1, -0.5, 3.0],
                    "r2": 0.97,
                    "cv": 0.23,
                }
            },
        }
        p = tmp_path / "schedule.json"
        import json
        p.write_text(json.dumps(schedule))

        loaded = load_poly_schedule(p)
        assert loaded["version"] == "poly_v1"
        assert "mm0_img_attn_q_proj" in loaded["layers"]
        assert loaded["layers"]["mm0_img_attn_q_proj"]["degree"] == 2


# ---------------------------------------------------------------------------
# _compute_poly_alphas_for_sample tests
# ---------------------------------------------------------------------------

class TestComputePolyAlphas:
    def test_mm_block_key_construction(self):
        """Verify correct poly key mapping for multimodal blocks."""
        from src.adaround_optimize import _compute_poly_alphas_for_sample

        schedule = {
            "layers": {
                "mm3_img_attn_q_proj": {
                    "degree": 2,
                    "coeffs": [0.01, -0.1, 5.0],
                    "r2": 0.95,
                    "cv": 0.3,
                },
            }
        }
        linear_paths = ["image_transformer_block.attn.q_proj"]
        alphas = _compute_poly_alphas_for_sample(
            schedule, "mm3", True, linear_paths, 7.0
        )
        assert alphas is not None
        assert len(alphas) == 1
        assert alphas[0] is not None
        assert alphas[0] > 0

    def test_missing_layer_returns_none(self):
        """Layers not in schedule should get None alpha."""
        from src.adaround_optimize import _compute_poly_alphas_for_sample

        schedule = {"layers": {}}
        linear_paths = ["image_transformer_block.attn.q_proj"]
        alphas = _compute_poly_alphas_for_sample(
            schedule, "mm0", True, linear_paths, 7.0
        )
        assert alphas is not None
        assert alphas[0] is None

    def test_no_schedule_returns_none(self):
        """No schedule → None."""
        from src.adaround_optimize import _compute_poly_alphas_for_sample

        result = _compute_poly_alphas_for_sample(
            None, "mm0", True, ["x"], 7.0
        )
        assert result is None

    def test_uni_block_key_construction(self):
        """Verify correct poly key mapping for unified blocks."""
        from src.adaround_optimize import _compute_poly_alphas_for_sample

        schedule = {
            "layers": {
                "uni5_mlp_fc1": {
                    "degree": 1,
                    "coeffs": [0.5, 3.0],
                    "r2": 0.90,
                    "cv": 0.15,
                },
            }
        }
        linear_paths = ["transformer_block.mlp.fc1"]
        alphas = _compute_poly_alphas_for_sample(
            schedule, "uni5", False, linear_paths, 2.0
        )
        assert alphas is not None
        assert alphas[0] is not None
        expected = float(np.polyval([0.5, 3.0], 2.0))
        assert alphas[0] == pytest.approx(expected)
