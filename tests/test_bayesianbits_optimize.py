"""
Tests for src/bayesianbits_optimize.py

8 test classes covering BB primitives, hierarchical quant, and finalization.
All tests use small synthetic tensors — no model loading, no filesystem I/O.
"""

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from src.bayesianbits_optimize import (
    BB_ZETA,
    BB_GAMMA,
    BB_BETA,
    SCALE_FACTOR_4,
    SCALE_FACTOR_8,
    hc_prob_pos,
    sample_gate,
    deterministic_gate,
    round_ste,
    compute_bb_scales,
    hierarchical_quant,
    BBParams,
    _BBProxy,
    _lp_loss,
    finalize_bb_bits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weight(out=8, inp=4, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (out, inp)).astype(np.float32)


def _make_linear(out=8, inp=4, bias=False, seed=0) -> nn.Linear:
    layer = nn.Linear(inp, out, bias=bias)
    layer.weight = mx.array(_make_weight(out, inp, seed))
    return layer


# ---------------------------------------------------------------------------
# Class 1: TestHardConcretePrimitives
# ---------------------------------------------------------------------------

class TestHardConcretePrimitives:

    def test_hc_prob_pos_at_large_positive_logit(self):
        """Large positive log_alpha → P(gate>0) ≈ 1."""
        log_alpha = mx.array([10.0])
        p = float(hc_prob_pos(log_alpha)[0])
        assert p > 0.95, f"Expected >0.95, got {p}"

    def test_hc_prob_pos_at_large_negative_logit(self):
        """Large negative log_alpha → P(gate>0) ≈ 0."""
        log_alpha = mx.array([-10.0])
        p = float(hc_prob_pos(log_alpha)[0])
        assert p < 0.05, f"Expected <0.05, got {p}"

    def test_hc_prob_pos_output_in_zero_one(self):
        log_alpha = mx.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        p = hc_prob_pos(log_alpha)
        mx.eval(p)
        p_np = np.array(p)
        assert np.all(p_np >= 0.0) and np.all(p_np <= 1.0)

    def test_sample_gate_output_in_zero_one(self):
        """Hard-concrete samples must be clipped to [0, 1]."""
        log_alpha = mx.zeros((100,))
        g = sample_gate(log_alpha)
        mx.eval(g)
        g_np = np.array(g)
        assert np.all(g_np >= 0.0) and np.all(g_np <= 1.0), \
            f"gate out of [0,1]: min={g_np.min():.4f}, max={g_np.max():.4f}"

    def test_deterministic_gate_binary(self):
        """Deterministic gate must return only 0.0 or 1.0."""
        log_alpha = mx.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        g = deterministic_gate(log_alpha)
        mx.eval(g)
        g_np = np.array(g)
        assert set(g_np.tolist()).issubset({0.0, 1.0})

    def test_deterministic_gate_pos_logit_gives_one(self):
        """Strongly positive logit → gate = 1."""
        log_alpha = mx.array([10.0, 10.0])
        g = deterministic_gate(log_alpha)
        mx.eval(g)
        assert np.all(np.array(g) == 1.0)

    def test_deterministic_gate_neg_logit_gives_zero(self):
        """Strongly negative logit → gate = 0."""
        log_alpha = mx.array([-10.0, -10.0])
        g = deterministic_gate(log_alpha)
        mx.eval(g)
        assert np.all(np.array(g) == 0.0)


# ---------------------------------------------------------------------------
# Class 2: TestRoundSTE
# ---------------------------------------------------------------------------

class TestRoundSTE:

    def test_forward_equals_round(self):
        x = mx.array([0.1, 0.6, 1.4, 1.9, -0.4, -0.7])
        y = round_ste(x)
        mx.eval(y)
        expected = np.round(np.array([0.1, 0.6, 1.4, 1.9, -0.4, -0.7]))
        np.testing.assert_allclose(np.array(y), expected, atol=1e-6)

    def test_gradient_is_identity(self):
        """STE: gradient of round_ste(x) w.r.t. x should be 1.0 (identity)."""
        x = mx.array([0.3, 0.7, 1.2])
        loss_fn = lambda x: round_ste(x).sum()
        grads = mx.grad(loss_fn)(x)
        mx.eval(grads)
        np.testing.assert_allclose(np.array(grads), np.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Class 3: TestComputeBBScales
# ---------------------------------------------------------------------------

class TestComputeBBScales:

    def test_scale_shapes(self):
        W = _make_weight(8, 4)
        s_2, s_4, s_8 = compute_bb_scales(W)
        assert s_2.shape == (8, 1)
        assert s_4.shape == (8, 1)
        assert s_8.shape == (8, 1)

    def test_scale_ordering(self):
        """s_2 >= s_4 >= s_8 > 0 for all output channels."""
        W = _make_weight(16, 8)
        s_2, s_4, s_8 = compute_bb_scales(W)
        assert np.all(s_2 >= s_4), "s_2 must be >= s_4"
        assert np.all(s_4 >= s_8), "s_4 must be >= s_8"
        assert np.all(s_8 > 0), "s_8 must be > 0"

    def test_scale_ratios(self):
        """Verify s_4 = s_2/3 and s_8 = s_4/9."""
        W = _make_weight(4, 4)
        s_2, s_4, s_8 = compute_bb_scales(W)
        np.testing.assert_allclose(s_4, s_2 / SCALE_FACTOR_4, rtol=1e-5)
        np.testing.assert_allclose(s_8, s_4 / SCALE_FACTOR_8, rtol=1e-5)

    def test_all_zero_weight_returns_floor_scale(self):
        """All-zero weights → absmax=0 → s_2 clamped to 1e-8."""
        W = np.zeros((4, 4), dtype=np.float32)
        s_2, s_4, s_8 = compute_bb_scales(W)
        assert np.all(s_2 >= 1e-8)

    def test_s_2_equals_absmax(self):
        W = _make_weight(8, 4)
        s_2, _, _ = compute_bb_scales(W)
        expected_s2 = np.abs(W).max(axis=1, keepdims=True)
        np.testing.assert_allclose(s_2, expected_s2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Class 4: TestHierarchicalQuant
# ---------------------------------------------------------------------------

class TestHierarchicalQuant:

    def _scales(self, W_np):
        s_2, s_4, s_8 = compute_bb_scales(W_np)
        return mx.array(s_2), mx.array(s_4), mx.array(s_8)

    def test_gate0_gives_w2_output(self):
        """gate_4=0 → W_q = x_q2 (2-bit quantisation only)."""
        W_np = _make_weight(4, 4, seed=1)
        s_2, s_4, s_8 = self._scales(W_np)
        W = mx.array(W_np)

        gate_4 = mx.zeros_like(W)
        gate_8 = mx.zeros_like(W)

        W_q = hierarchical_quant(W, s_2, s_4, s_8, gate_4, gate_8)
        mx.eval(W_q)

        # x_q2 = s_2 * round(W / s_2)
        W_q2_expected = np.array(s_2) * np.round(W_np / np.array(s_2))
        np.testing.assert_allclose(np.array(W_q), W_q2_expected, rtol=1e-4)

    def test_gate1_gate0_gives_w4_output(self):
        """gate_4=1, gate_8=0 → W_q adds the 4-bit residual."""
        W_np = _make_weight(4, 4, seed=2)
        s_2, s_4, s_8 = self._scales(W_np)
        W = mx.array(W_np)

        gate_4 = mx.ones_like(W)
        gate_8 = mx.zeros_like(W)

        W_q = hierarchical_quant(W, s_2, s_4, s_8, gate_4, gate_8)
        mx.eval(W_q)

        s_2_np, s_4_np = np.array(s_2), np.array(s_4)
        x_q2 = s_2_np * np.round(W_np / s_2_np)
        x_q4 = s_4_np * np.round((W_np - x_q2) / s_4_np)
        expected = x_q2 + x_q4
        np.testing.assert_allclose(np.array(W_q), expected, rtol=1e-4)

    def test_gate1_gate1_gives_w8_output(self):
        """gate_4=1, gate_8=1 → W_q adds both residuals."""
        W_np = _make_weight(4, 4, seed=3)
        s_2, s_4, s_8 = self._scales(W_np)
        W = mx.array(W_np)

        gate_4 = mx.ones_like(W)
        gate_8 = mx.ones_like(W)

        W_q = hierarchical_quant(W, s_2, s_4, s_8, gate_4, gate_8)
        mx.eval(W_q)

        s_2_np, s_4_np, s_8_np = np.array(s_2), np.array(s_4), np.array(s_8)
        x_q2 = s_2_np * np.round(W_np / s_2_np)
        x_q4 = s_4_np * np.round((W_np - x_q2) / s_4_np)
        x_q8 = s_8_np * np.round((W_np - x_q2 - x_q4) / s_8_np)
        expected = x_q2 + x_q4 + x_q8
        np.testing.assert_allclose(np.array(W_q), expected, rtol=1e-4)

    def test_output_shape_matches_input(self):
        W_np = _make_weight(6, 3)
        s_2, s_4, s_8 = self._scales(W_np)
        W = mx.array(W_np)
        gate_4 = mx.ones_like(W)
        gate_8 = mx.zeros_like(W)
        W_q = hierarchical_quant(W, s_2, s_4, s_8, gate_4, gate_8)
        mx.eval(W_q)
        assert W_q.shape == W.shape


# ---------------------------------------------------------------------------
# Class 5: TestBBParams
# ---------------------------------------------------------------------------

class TestBBParams:

    def test_log_alphas_shape_matches_weight(self):
        W_fps_np = [_make_weight(8, 4), _make_weight(6, 3)]
        params = BBParams(W_fps_np, bits_a=8)
        mx.eval(params.parameters())
        assert params.log_alphas_4[0].shape == (8, 4)
        assert params.log_alphas_4[1].shape == (6, 3)
        assert params.log_alphas_8[0].shape == (8, 4)

    def test_a_scales_shape(self):
        W_fps_np = [_make_weight(8, 4), _make_weight(4, 4)]
        params = BBParams(W_fps_np)
        mx.eval(params.parameters())
        assert params.a_scales[0].shape == (1,)
        assert params.a_scales[1].shape == (1,)

    def test_n_parameter_groups_matches_n_linears(self):
        n = 5
        W_fps_np = [_make_weight(4, 4, seed=i) for i in range(n)]
        params = BBParams(W_fps_np)
        assert len(params.log_alphas_4) == n
        assert len(params.log_alphas_8) == n
        assert len(params.a_scales) == n

    def test_initial_a_scale_is_one(self):
        W_fps_np = [_make_weight(8, 4)]
        params = BBParams(W_fps_np)
        mx.eval(params.a_scales[0])
        np.testing.assert_allclose(np.array(params.a_scales[0]), [1.0], atol=1e-6)

    def test_bits_a_stored_correctly(self):
        params = BBParams([_make_weight(4, 4)], bits_a=4)
        assert params.bits_a == 4
        assert params.qmin_a == -8
        assert params.qmax_a == 7


# ---------------------------------------------------------------------------
# Class 6: TestBBProxy
# ---------------------------------------------------------------------------

class TestBBProxy:

    def _make_proxy(self, out=8, inp=4, seed=0) -> _BBProxy:
        layer = _make_linear(out, inp, seed=seed)
        W = mx.array(np.array(layer.weight))
        a_scale = mx.array([1.0])
        return _BBProxy(layer, W, a_scale, -128, 127)

    def test_output_shape(self):
        proxy = self._make_proxy(8, 4)
        x = mx.array(np.random.randn(2, 4).astype(np.float32))
        y = proxy(x)
        mx.eval(y)
        assert y.shape == (2, 8)

    def test_output_is_matrix_multiply(self):
        """With a_scale=1, quantization is ~identity for small weights; at least check shape."""
        proxy = self._make_proxy(4, 4)
        x = mx.array(np.eye(4, dtype=np.float32))
        y = proxy(x)
        mx.eval(y)
        assert y.shape == (4, 4)

    def test_attribute_delegation(self):
        """Attribute lookup for weight should fall through to original layer."""
        layer = _make_linear(8, 4)
        proxy = _BBProxy(layer, mx.array(np.array(layer.weight)), mx.array([1.0]), -128, 127)
        # Weight attribute should delegate to original layer
        np.testing.assert_allclose(np.array(proxy.weight), np.array(layer.weight), atol=1e-6)


# ---------------------------------------------------------------------------
# Class 7: TestFinalizeBBBits
# ---------------------------------------------------------------------------

class TestFinalizeBBBits:

    def test_all_gate0_gives_w2(self):
        """All-zero log_alphas → gates evaluate to 0 → W2."""
        W_fps_np = [_make_weight(4, 4)]
        params = BBParams(W_fps_np)
        # Set log_alphas to very negative → gate=0
        params.log_alphas_4[0] = mx.full((4, 4), -10.0)
        params.log_alphas_8[0] = mx.full((4, 4), -10.0)
        mx.eval(params.parameters())

        bits = finalize_bb_bits(params, ["attn.q_proj"])
        assert bits["attn.q_proj"] == 2

    def test_gate4_one_gate8_zero_gives_w4(self):
        W_fps_np = [_make_weight(4, 4)]
        params = BBParams(W_fps_np)
        params.log_alphas_4[0] = mx.full((4, 4), 10.0)   # gate_4 = 1
        params.log_alphas_8[0] = mx.full((4, 4), -10.0)  # gate_8 = 0
        mx.eval(params.parameters())

        bits = finalize_bb_bits(params, ["attn.q_proj"])
        assert bits["attn.q_proj"] == 4

    def test_both_gates_one_gives_w8(self):
        W_fps_np = [_make_weight(4, 4)]
        params = BBParams(W_fps_np)
        params.log_alphas_4[0] = mx.full((4, 4), 10.0)
        params.log_alphas_8[0] = mx.full((4, 4), 10.0)
        mx.eval(params.parameters())

        bits = finalize_bb_bits(params, ["attn.q_proj"])
        assert bits["attn.q_proj"] == 8

    def test_output_bits_in_valid_set(self):
        W_fps_np = [_make_weight(4, 4, seed=i) for i in range(3)]
        params = BBParams(W_fps_np)
        mx.eval(params.parameters())

        paths = ["path_a", "path_b", "path_c"]
        bits = finalize_bb_bits(params, paths)

        assert set(bits.values()).issubset({2, 4, 8})

    def test_output_keys_match_linear_paths(self):
        paths = ["layer1", "layer2", "layer3"]
        W_fps_np = [_make_weight(4, 4, seed=i) for i in range(3)]
        params = BBParams(W_fps_np)
        mx.eval(params.parameters())

        bits = finalize_bb_bits(params, paths)
        assert set(bits.keys()) == set(paths)


# ---------------------------------------------------------------------------
# Class 8: TestLpLoss
# ---------------------------------------------------------------------------

class TestLpLoss:

    def test_zero_loss_for_identical_tensors(self):
        x = mx.array(np.random.randn(4, 6, 8).astype(np.float32))
        loss = _lp_loss(x, x)
        mx.eval(loss)
        assert float(loss) < 1e-6

    def test_loss_positive_for_different_tensors(self):
        pred = mx.array(np.ones((3, 4), dtype=np.float32))
        tgt = mx.array(np.zeros((3, 4), dtype=np.float32))
        loss = _lp_loss(pred, tgt)
        mx.eval(loss)
        assert float(loss) > 0.0

    def test_loss_scales_with_magnitude(self):
        pred_large = mx.array(np.full((2, 4), 10.0, dtype=np.float32))
        pred_small = mx.array(np.full((2, 4), 1.0, dtype=np.float32))
        tgt = mx.array(np.zeros((2, 4), dtype=np.float32))
        loss_large = float(_lp_loss(pred_large, tgt))
        loss_small = float(_lp_loss(pred_small, tgt))
        assert loss_large > loss_small

    def test_3d_tensor_handled(self):
        pred = mx.array(np.random.randn(2, 10, 16).astype(np.float32))
        tgt = mx.array(np.zeros((2, 10, 16), dtype=np.float32))
        loss = _lp_loss(pred, tgt)
        mx.eval(loss)
        assert float(loss) > 0.0
