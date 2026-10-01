"""Tests for src.phase2.calibrate — balancing vectors, QKV merge, persistence."""

import numpy as np
import pytest

from src.phase2.calibrate import (
    build_lightweight_registry,
    calibrate_all_layers,
    compute_balancing_vector,
    compute_qkv_balancing,
    load_calibration,
    load_phase1_data,
    save_calibration,
)

H = 256
FFN_H = 512
T = 5


# ====================================================================
# compute_balancing_vector
# ====================================================================

class TestComputeBalancingVector:

    def _make_data(self, seed=0):
        rng = np.random.default_rng(seed)
        act = rng.exponential(1.0, size=(T, H)).astype(np.float32)
        wt = rng.exponential(0.5, size=(H,)).astype(np.float32)
        return act, wt

    def test_shape_and_finite(self):
        act, wt = self._make_data()
        b = compute_balancing_vector(act, wt)
        assert b.shape == (H,)
        assert np.all(np.isfinite(b))

    def test_clamping(self):
        act, wt = self._make_data()
        b = compute_balancing_vector(act, wt, b_min=0.1, b_max=10.0)
        assert np.all(b >= 0.1)
        assert np.all(b <= 10.0)

    def test_dead_channels_get_one(self):
        act, wt = self._make_data()
        wt[:8] = 0.0
        b = compute_balancing_vector(act, wt)
        np.testing.assert_equal(b[:8], 1.0)

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
    def test_alpha_sensitivity(self, alpha):
        act, wt = self._make_data()
        b = compute_balancing_vector(act, wt, alpha=alpha)
        assert b.shape == (H,)
        assert np.all(np.isfinite(b))

    def test_alpha_monotonicity(self):
        """Higher alpha → wider spread of b values."""
        act, wt = self._make_data()
        spread = {}
        for alpha in [0.3, 0.5, 0.7]:
            b = compute_balancing_vector(act, wt, alpha=alpha, b_min=1e-10, b_max=1e10)
            spread[alpha] = b.max() / (b.min() + 1e-12)
        assert spread[0.3] < spread[0.5] < spread[0.7]

    def test_extreme_ratio_clamped(self):
        act = np.ones((T, H), dtype=np.float32) * 1e8
        wt = np.ones(H, dtype=np.float32) * 1e-8
        b = compute_balancing_vector(act, wt, b_max=100.0)
        assert np.all(b <= 100.0)


# ====================================================================
# compute_qkv_balancing
# ====================================================================

class TestComputeQkvBalancing:

    @pytest.mark.parametrize("method", ["max", "geomean", "l2"])
    def test_produces_valid_vector(self, mock_diagnostics, method):
        b = compute_qkv_balancing(
            0, "image", mock_diagnostics, method=method,
        )
        assert b.shape == (H,)
        assert np.all(b >= 1e-5)
        assert np.all(b <= 1e5)
        assert np.all(np.isfinite(b))

    def test_methods_differ(self, mock_diagnostics):
        b_max = compute_qkv_balancing(0, "image", mock_diagnostics, method="max")
        b_geo = compute_qkv_balancing(0, "image", mock_diagnostics, method="geomean")
        b_l2 = compute_qkv_balancing(0, "image", mock_diagnostics, method="l2")
        assert not np.allclose(b_max, b_geo)
        assert not np.allclose(b_max, b_l2)
        assert not np.allclose(b_geo, b_l2)

    def test_invalid_method_raises(self, mock_diagnostics):
        with pytest.raises(ValueError, match="Unknown QKV merge method"):
            compute_qkv_balancing(0, "image", mock_diagnostics, method="invalid")

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
    def test_alpha_sweep(self, mock_diagnostics, alpha):
        b = compute_qkv_balancing(0, "image", mock_diagnostics, alpha=alpha)
        assert np.all(np.isfinite(b))


# ====================================================================
# calibrate_all_layers
# ====================================================================

class TestCalibrateAllLayers:

    def test_all_target_layers_calibrated(self, registry, mock_diagnostics, test_config):
        cal = calibrate_all_layers(registry, mock_diagnostics, test_config)
        bv = cal["balancing_vectors"]

        excluded = set(test_config["exclude_layers"])
        expected = {e["name"] for e in registry if e["name"] not in excluded}
        assert set(bv.keys()) == expected

    def test_b_inv_only_o_proj_and_fc2(self, registry, mock_diagnostics, test_config):
        cal = calibrate_all_layers(registry, mock_diagnostics, test_config)
        for name in cal["b_inv_layers"]:
            assert "o_proj" in name or "fc2" in name

    def test_qkv_share_same_vector(self, registry, mock_diagnostics, test_config):
        cal = calibrate_all_layers(registry, mock_diagnostics, test_config)
        bv = cal["balancing_vectors"]
        q = bv["blocks.0.image.attn.q_proj"]
        k = bv["blocks.0.image.attn.k_proj"]
        v = bv["blocks.0.image.attn.v_proj"]
        np.testing.assert_array_equal(q, k)
        np.testing.assert_array_equal(q, v)

    @pytest.mark.parametrize("method", ["max", "geomean", "l2"])
    def test_qkv_method_override(self, registry, mock_diagnostics, test_config, method):
        cfg = {**test_config, "qkv_method": method}
        cal = calibrate_all_layers(registry, mock_diagnostics, cfg)
        assert len(cal["balancing_vectors"]) > 0

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
    def test_different_alpha(self, registry, mock_diagnostics, test_config, alpha):
        cfg = {**test_config, "alpha": alpha}
        cal = calibrate_all_layers(registry, mock_diagnostics, cfg)
        for b in cal["balancing_vectors"].values():
            assert np.all(np.isfinite(b))


# ====================================================================
# save / load calibration
# ====================================================================

class TestCalibrationPersistence:

    def test_roundtrip_values_preserved(
        self, registry, mock_diagnostics, test_config, tmp_path,
    ):
        cal = calibrate_all_layers(registry, mock_diagnostics, test_config)

        out = tmp_path / "cal_out"
        save_calibration(cal, out)
        loaded = load_calibration(out)

        assert set(loaded["balancing_vectors"].keys()) == set(cal["balancing_vectors"].keys())
        assert loaded["b_inv_layers"] == cal["b_inv_layers"]
        for name in cal["balancing_vectors"]:
            np.testing.assert_array_almost_equal(
                loaded["balancing_vectors"][name],
                cal["balancing_vectors"][name],
            )


# ====================================================================
# build_lightweight_registry
# ====================================================================

class TestBuildLightweightRegistry:

    def test_families_present(self, mock_diagnostics):
        reg = build_lightweight_registry(mock_diagnostics)
        families = {e["family"] for e in reg}
        for f in ("q_proj", "k_proj", "v_proj", "o_proj",
                   "fc1", "fc2", "context_embedder", "final_linear"):
            assert f in families

    def test_matches_layer_count(self, mock_diagnostics, layer_names):
        reg = build_lightweight_registry(mock_diagnostics)
        assert len(reg) == len(layer_names)

    def test_block_23_text_skip(self, mock_diagnostics):
        """Block 1 text (simulating block-23) should have no o_proj/fc1/fc2."""
        reg = build_lightweight_registry(mock_diagnostics)
        blk1_text = [
            e for e in reg
            if e["block"] == 1 and e["side"] == "text"
        ]
        blk1_text_families = {e["family"] for e in blk1_text}
        assert "o_proj" not in blk1_text_families
        assert "fc1" not in blk1_text_families
        assert "fc2" not in blk1_text_families


# ====================================================================
# load_phase1_data
# ====================================================================

class TestLoadPhase1Data:

    def test_shapes(self, mock_diagnostics):
        data = load_phase1_data("blocks.0.image.attn.q_proj", mock_diagnostics)
        assert data["act_trajectory"].shape == (T, H)
        assert data["wt_salience"].shape == (H,)

    def test_fc2_d_in(self, mock_diagnostics):
        data = load_phase1_data("blocks.0.image.mlp.fc2", mock_diagnostics)
        assert data["act_trajectory"].shape == (T, FFN_H)
        assert data["wt_salience"].shape == (FFN_H,)
