"""
Tests for src/adaround_optimize.py

Covers all pure-Python / pure-MLX functions without loading DiffusionKit or
the full pipeline.  Uses small synthetic weights/tensors throughout.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import tempfile

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from src.adaround_optimize import (
    GAMMA,
    ZETA,
    ROUND_WEIGHT,
    LP_NORM,
    LinearTempDecay,
    rectified_sigmoid,
    compute_per_channel_scale,
    init_alpha,
    fake_quant_per_tensor,
    _get_nested,
    _set_nested,
    get_block_linears,
    _QuantProxy,
    AdaRoundParams,
    _lp_loss,
    _round_loss,
    block_loss_fn,
    finalize_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weight(out=8, inp=4, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (out, inp)).astype(np.float32)


def _make_linear(out=8, inp=4, bias=True, seed=0) -> nn.Linear:
    layer = nn.Linear(inp, out, bias=bias)
    layer.weight = mx.array(_make_weight(out, inp, seed))
    if bias:
        layer.bias = mx.array(np.zeros(out, dtype=np.float32))
    return layer


# ---------------------------------------------------------------------------
# TestLinearTempDecay
# ---------------------------------------------------------------------------

class TestLinearTempDecay:

    def test_before_warmup_returns_start_b(self):
        td = LinearTempDecay(t_max=100, warm_up=0.2, start_b=20.0, end_b=2.0)
        assert td(0) == 20.0
        assert td(19) == 20.0   # warm_up ends at t=20

    def test_at_start_decay_returns_start_b(self):
        td = LinearTempDecay(t_max=100, warm_up=0.2, start_b=20.0, end_b=2.0)
        assert td(20) == pytest.approx(20.0)

    def test_at_t_max_returns_end_b(self):
        td = LinearTempDecay(t_max=100, warm_up=0.2, start_b=20.0, end_b=2.0)
        assert td(100) == 2.0

    def test_beyond_t_max_returns_end_b(self):
        td = LinearTempDecay(t_max=100, warm_up=0.2, start_b=20.0, end_b=2.0)
        assert td(200) == 2.0

    def test_midpoint_between_start_decay_and_t_max(self):
        td = LinearTempDecay(t_max=100, warm_up=0.2, start_b=20.0, end_b=2.0)
        # midpoint of [20, 100] is 60; rel = 0.5 → b = 2 + 18 * 0.5 = 11
        b = td(60)
        assert b == pytest.approx(11.0)

    def test_monotonically_decreasing(self):
        td = LinearTempDecay(t_max=200, warm_up=0.0, start_b=20.0, end_b=2.0)
        vals = [td(t) for t in range(0, 210, 10)]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-9

    def test_zero_warmup(self):
        td = LinearTempDecay(t_max=100, warm_up=0.0, start_b=10.0, end_b=2.0)
        assert td(0) == pytest.approx(10.0)
        assert td(100) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TestRectifiedSigmoid
# ---------------------------------------------------------------------------

class TestRectifiedSigmoid:

    def test_output_range_in_zero_one(self):
        alpha = mx.array(np.linspace(-10, 10, 50, dtype=np.float32))
        r = rectified_sigmoid(alpha)
        mx.eval(r)
        r_np = np.array(r)
        assert r_np.min() >= 0.0
        assert r_np.max() <= 1.0

    def test_large_positive_alpha_approaches_one(self):
        alpha = mx.array([100.0])
        r = rectified_sigmoid(alpha)
        mx.eval(r)
        assert float(np.array(r)[0]) == pytest.approx(1.0, abs=1e-4)

    def test_large_negative_alpha_approaches_zero(self):
        alpha = mx.array([-100.0])
        r = rectified_sigmoid(alpha)
        mx.eval(r)
        assert float(np.array(r)[0]) == pytest.approx(0.0, abs=1e-4)

    def test_alpha_zero_gives_gamma_plus_half_range(self):
        # sigmoid(0) = 0.5  → (zeta-gamma)*0.5 + gamma = 0.5*(1.2) - 0.1 = 0.5
        alpha = mx.array([0.0])
        r = rectified_sigmoid(alpha)
        mx.eval(r)
        assert float(np.array(r)[0]) == pytest.approx(0.5, abs=1e-5)

    def test_gradient_flows_through(self):
        alpha = mx.array([0.0])
        def fn(a):
            return rectified_sigmoid(a).sum()
        grad = mx.grad(fn)(alpha)
        mx.eval(grad)
        assert float(np.array(grad)[0]) != 0.0


# ---------------------------------------------------------------------------
# TestComputePerChannelScale
# ---------------------------------------------------------------------------

class TestComputePerChannelScale:

    def test_shape(self):
        W = _make_weight(8, 4)
        s = compute_per_channel_scale(W, bits=4)
        assert s.shape == (8, 1)

    def test_values_are_absmax_over_qmax(self):
        W = np.array([[4.0, -7.0], [2.0, 1.0]], dtype=np.float32)
        s = compute_per_channel_scale(W, bits=4)
        # qmax = 7 for 4-bit; row0 absmax = 7 → scale = 1.0; row1 absmax = 2 → scale = 2/7
        assert float(s[0, 0]) == pytest.approx(1.0)
        assert float(s[1, 0]) == pytest.approx(2.0 / 7.0)

    def test_never_below_eps(self):
        W = np.zeros((4, 4), dtype=np.float32)
        s = compute_per_channel_scale(W, bits=4)
        assert (s >= 1e-8).all()

    def test_8bit_uses_qmax_127(self):
        W = np.array([[127.0, 0.0]], dtype=np.float32)
        s = compute_per_channel_scale(W, bits=8)
        assert float(s[0, 0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestInitAlpha
# ---------------------------------------------------------------------------

class TestInitAlpha:

    def test_shape_matches_weight(self):
        W = _make_weight(6, 4)
        s = compute_per_channel_scale(W, bits=4)
        alpha = init_alpha(W, s)
        assert alpha.shape == W.shape

    def test_rectified_sigmoid_recovers_frac_part(self):
        W = _make_weight(8, 4, seed=42)
        s = compute_per_channel_scale(W, bits=4)
        alpha_np = init_alpha(W, s)

        rest_true = W / s - np.floor(W / s)

        alpha_mx = mx.array(alpha_np)
        r_mx = rectified_sigmoid(alpha_mx)
        mx.eval(r_mx)
        r_np = np.array(r_mx)

        np.testing.assert_allclose(r_np, rest_true, atol=1e-5)

    def test_no_nans_or_infs(self):
        W = _make_weight(16, 8)
        s = compute_per_channel_scale(W, bits=4)
        alpha = init_alpha(W, s)
        assert np.isfinite(alpha).all()


# ---------------------------------------------------------------------------
# TestFakeQuantPerTensor
# ---------------------------------------------------------------------------

class TestFakeQuantPerTensor:

    def test_output_within_quant_range_times_scale(self):
        x = mx.array(np.linspace(-10, 10, 50, dtype=np.float32))
        s = mx.array([0.1])
        out = fake_quant_per_tensor(x, s, -128, 127)
        mx.eval(out)
        out_np = np.array(out)
        assert out_np.min() >= -128 * 0.1 - 1e-5
        assert out_np.max() <= 127 * 0.1 + 1e-5

    def test_scale_clamps_minimum(self):
        # scale=0 should be treated as 1e-8
        x = mx.array([1.0])
        s = mx.array([0.0])
        out = fake_quant_per_tensor(x, s, -128, 127)
        mx.eval(out)
        assert np.isfinite(float(np.array(out)[0]))

    def test_negative_scale_treated_same_as_positive(self):
        x = mx.array([1.5])
        s_pos = mx.array([0.5])
        s_neg = mx.array([-0.5])
        out_pos = fake_quant_per_tensor(x, s_pos, -128, 127)
        out_neg = fake_quant_per_tensor(x, s_neg, -128, 127)
        mx.eval(out_pos, out_neg)
        assert float(np.array(out_pos)[0]) == pytest.approx(float(np.array(out_neg)[0]))

    def test_exact_quantisation_step(self):
        # x = 1.0, scale = 0.25 → quantised int = 4 → dequant = 1.0
        x = mx.array([1.0])
        s = mx.array([0.25])
        out = fake_quant_per_tensor(x, s, -128, 127)
        mx.eval(out)
        assert float(np.array(out)[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestGetSetNested
# ---------------------------------------------------------------------------

class TestGetSetNested:

    def _make_obj(self):
        class Inner:
            pass
        class Outer:
            pass
        inner = Inner()
        inner.val = 42
        outer = Outer()
        outer.inner = inner
        return outer

    def test_get_single_level(self):
        obj = self._make_obj()
        assert _get_nested(obj, "inner") is obj.inner

    def test_get_two_levels(self):
        obj = self._make_obj()
        assert _get_nested(obj, "inner.val") == 42

    def test_set_two_levels(self):
        obj = self._make_obj()
        _set_nested(obj, "inner.val", 99)
        assert obj.inner.val == 99

    def test_get_list_index(self):
        class Container:
            items = [10, 20, 30]
        c = Container()
        assert _get_nested(c, "items[1]") == 20

    def test_set_list_index(self):
        class Container:
            items = [10, 20, 30]
        c = Container()
        _set_nested(c, "items[2]", 99)
        assert c.items[2] == 99


# ---------------------------------------------------------------------------
# TestGetBlockLinears (using mock blocks)
# ---------------------------------------------------------------------------

def _make_mock_transformer_block():
    """Returns a minimal mock of a DiffusionKit TransformerBlock."""
    class MockAttn:
        q_proj = _make_linear(8, 8)
        k_proj = _make_linear(8, 8, bias=False)
        v_proj = _make_linear(8, 8)
        o_proj = _make_linear(8, 8)

    class MockMLP:
        fc1 = _make_linear(8, 16)
        fc2 = _make_linear(16, 8)

    class MockTransBlock:
        attn = MockAttn()
        mlp = MockMLP()

    return MockTransBlock()


def _make_mock_mm_block():
    class MockMM:
        image_transformer_block = _make_mock_transformer_block()
        text_transformer_block = _make_mock_transformer_block()
    return MockMM()


def _make_mock_uni_block():
    class MockUni:
        transformer_block = _make_mock_transformer_block()
    return MockUni()


class TestGetBlockLinears:

    def test_uni_returns_six_linears(self):
        block = _make_mock_uni_block()
        linears = get_block_linears(block, is_mm=False)
        assert len(linears) == 6

    def test_mm_returns_twelve_linears(self):
        block = _make_mock_mm_block()
        linears = get_block_linears(block, is_mm=True)
        assert len(linears) == 12

    def test_paths_contain_expected_names(self):
        block = _make_mock_uni_block()
        paths = [p for p, _, _ in get_block_linears(block, is_mm=False)]
        assert "transformer_block.attn.q_proj" in paths
        assert "transformer_block.mlp.fc2" in paths

    def test_fc2_flagged_as_post_gelu(self):
        block = _make_mock_uni_block()
        for path, _, is_post_gelu in get_block_linears(block, is_mm=False):
            if "fc2" in path:
                assert is_post_gelu is True
            else:
                assert is_post_gelu is False

    def test_mm_paths_have_image_and_text_prefixes(self):
        block = _make_mock_mm_block()
        paths = [p for p, _, _ in get_block_linears(block, is_mm=True)]
        img_paths = [p for p in paths if p.startswith("image_")]
        txt_paths = [p for p in paths if p.startswith("text_")]
        assert len(img_paths) == 6
        assert len(txt_paths) == 6


# ---------------------------------------------------------------------------
# TestQuantProxy
# ---------------------------------------------------------------------------

class TestQuantProxy:

    def _proxy_and_soft_weight(self, out=4, inp=4):
        layer = _make_linear(out, inp)
        W_np = np.array(layer.weight)
        s_np = compute_per_channel_scale(W_np, bits=4)
        alpha_np = init_alpha(W_np, s_np)
        alpha = mx.array(alpha_np)
        s_mx = mx.array(s_np)
        r = rectified_sigmoid(alpha)
        W_floor = mx.floor(mx.array(W_np) / s_mx)
        soft_w = mx.clip(W_floor + r, -8, 7) * s_mx
        a_scale = mx.array([1.0])
        return _QuantProxy(layer, soft_w, a_scale, -128, 127), soft_w, layer

    def test_output_shape_matches_linear(self):
        proxy, _, _ = self._proxy_and_soft_weight(out=4, inp=4)
        x = mx.array(np.ones((3, 4), dtype=np.float32))
        y = proxy(x)
        mx.eval(y)
        assert np.array(y).shape == (3, 4)

    def test_getattr_delegates_to_original(self):
        proxy, _, layer = self._proxy_and_soft_weight()
        # Bias should come from original layer
        if layer.bias is not None:
            assert proxy.bias is layer.bias

    def test_gradient_flows_to_soft_weight(self):
        layer = _make_linear(4, 4)
        W_np = np.array(layer.weight)
        s_np = compute_per_channel_scale(W_np, bits=4)
        alpha = mx.array(init_alpha(W_np, s_np))
        a_scale = mx.array([1.0])

        def compute_output(alpha):
            s_mx = mx.array(s_np)
            r = rectified_sigmoid(alpha)
            W_q = mx.clip(mx.floor(mx.array(W_np) / s_mx) + r, -8, 7) * s_mx
            proxy = _QuantProxy(layer, W_q, a_scale, -128, 127)
            x = mx.array(np.ones((2, 4), dtype=np.float32))
            return proxy(x).sum()

        grad = mx.grad(compute_output)(alpha)
        mx.eval(grad)
        grad_np = np.array(grad)
        assert np.any(grad_np != 0.0), "Expected non-zero gradient w.r.t. alpha"


# ---------------------------------------------------------------------------
# TestAdaRoundParams
# ---------------------------------------------------------------------------

class TestAdaRoundParams:

    def test_creates_one_alpha_per_linear(self):
        W_list = [_make_weight(4, 4) for _ in range(6)]
        p = AdaRoundParams(W_list, bits_w=4, bits_a=8)
        assert len(p.alphas) == 6

    def test_alpha_shape_matches_weight(self):
        W_np = _make_weight(8, 4)
        p = AdaRoundParams([W_np], bits_w=4)
        assert np.array(p.alphas[0]).shape == (8, 4)

    def test_a_scale_initialised_to_one(self):
        W_np = _make_weight(4, 4)
        p = AdaRoundParams([W_np])
        assert float(np.array(p.a_scales[0])[0]) == pytest.approx(1.0)

    def test_qmin_qmax_4bit(self):
        p = AdaRoundParams([_make_weight(4, 4)], bits_w=4, bits_a=8)
        assert p.qmin_w == -8
        assert p.qmax_w == 7
        assert p.qmin_a == -128
        assert p.qmax_a == 127

    def test_parameters_traversable_by_mlx(self):
        W_list = [_make_weight(4, 4) for _ in range(3)]
        p = AdaRoundParams(W_list)
        params = p.trainable_parameters()
        # Should have alphas and a_scales in the parameter tree
        assert "alphas" in params or len(params) > 0


# ---------------------------------------------------------------------------
# TestLpLoss
# ---------------------------------------------------------------------------

class TestLpLoss:

    def test_zero_for_identical_inputs(self):
        x = mx.array(np.ones((4, 8), dtype=np.float32))
        loss = _lp_loss(x, x, p=2.0)
        mx.eval(loss)
        assert float(np.array(loss)) == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different_inputs(self):
        pred = mx.array(np.ones((4, 8), dtype=np.float32))
        tgt = mx.array(np.zeros((4, 8), dtype=np.float32))
        loss = _lp_loss(pred, tgt, p=2.0)
        mx.eval(loss)
        assert float(np.array(loss)) > 0.0

    def test_3d_input_flattened_correctly(self):
        pred = mx.array(np.ones((2, 3, 4), dtype=np.float32))
        tgt = mx.array(np.zeros((2, 3, 4), dtype=np.float32))
        loss = _lp_loss(pred, tgt, p=2.0)
        mx.eval(loss)
        # 2 batch, 12 channels each; each element diff=1 → sum = 12 per batch → mean = 12
        assert float(np.array(loss)) == pytest.approx(12.0, abs=1e-5)


# ---------------------------------------------------------------------------
# TestRoundLoss
# ---------------------------------------------------------------------------

class TestRoundLoss:

    def test_zero_when_alpha_very_positive(self):
        # r → 1.0, round_loss per element = 1 - |2*(1-0.5)|^b = 1 - 1 = 0
        alpha = [mx.array(np.full((4, 4), 100.0, dtype=np.float32))]
        loss = _round_loss(alpha, b=2.0)
        mx.eval(loss)
        assert float(np.array(loss)) == pytest.approx(0.0, abs=1e-3)

    def test_zero_when_alpha_very_negative(self):
        # r → 0.0, round_loss per element = 1 - |2*(0-0.5)|^b = 1 - 1 = 0
        alpha = [mx.array(np.full((4, 4), -100.0, dtype=np.float32))]
        loss = _round_loss(alpha, b=2.0)
        mx.eval(loss)
        assert float(np.array(loss)) == pytest.approx(0.0, abs=1e-3)

    def test_maximum_when_alpha_zero(self):
        # r = 0.5 → round_loss = 1 - |2*(0.5-0.5)|^b = 1 - 0 = 1 per element
        alpha = [mx.array(np.zeros((2, 2), dtype=np.float32))]
        loss = _round_loss(alpha, b=2.0)
        mx.eval(loss)
        # 4 elements, each contributes 1.0
        assert float(np.array(loss)) == pytest.approx(4.0, abs=1e-4)

    def test_sums_over_multiple_alphas(self):
        alpha_list = [
            mx.array(np.zeros((2, 2), dtype=np.float32)),  # 4 elements, each = 1
            mx.array(np.zeros((3, 3), dtype=np.float32)),  # 9 elements, each = 1
        ]
        loss = _round_loss(alpha_list, b=2.0)
        mx.eval(loss)
        assert float(np.array(loss)) == pytest.approx(13.0, abs=1e-4)


# ---------------------------------------------------------------------------
# TestBlockLossFn (integration: patches a mock block, checks loss shape/sign)
# ---------------------------------------------------------------------------

class TestBlockLossFn:

    def _make_minimal_block(self):
        """Mock block with transformer_block.attn.[q,k,v,o]_proj and mlp.[fc1,fc2].

        hidden=8, expansion=16: fc1 maps 8→16, fc2 maps 16→8.
        _make_linear(out, inp) so fc1=_make_linear(16,8), fc2=_make_linear(8,16).
        """
        class _FFN:
            def __call__(self, x):
                return self.fc2(mx.tanh(self.fc1(x)))
            fc1 = _make_linear(16, 8)   # 8 → 16
            fc2 = _make_linear(8, 16)   # 16 → 8

        class _Attn:
            q_proj = _make_linear(8, 8)
            k_proj = _make_linear(8, 8, bias=False)
            v_proj = _make_linear(8, 8)
            o_proj = _make_linear(8, 8)

        class _Inner:
            attn = _Attn()
            mlp = _FFN()

            def __call__(self, x):
                # trivial block: x → attn → mlp → output
                q = self.attn.q_proj(x)
                return self.mlp(q)

        class _Block:
            transformer_block = _Inner()

            def __call__(self, x):
                return self.transformer_block(x)

        return _Block()

    def test_loss_is_scalar_non_negative(self):
        block = self._make_minimal_block()
        linears = get_block_linears(block, is_mm=False)
        linear_paths = [p for p, _, _ in linears]
        linear_layers = [l for _, l, _ in linears]
        W_fps_np = [np.array(l.weight) for l in linear_layers]
        w_scales_np = [compute_per_channel_scale(W, 4) for W in W_fps_np]

        params = AdaRoundParams(W_fps_np)

        x = mx.array(np.random.randn(2, 8).astype(np.float32))
        fp_out = block(x)
        mx.eval(fp_out)

        loss = block_loss_fn(
            params, block, is_mm=False,
            linear_paths=linear_paths,
            linear_layers=linear_layers,
            W_fps_np=W_fps_np,
            w_scales_np=w_scales_np,
            sample_inputs=[x],
            sample_kwargs={},
            fp_outputs=[fp_out],
            b_val=20.0,
        )
        mx.eval(loss)
        loss_val = float(np.array(loss))
        assert np.isfinite(loss_val)
        assert loss_val >= 0.0

    def test_block_linears_restored_after_loss(self):
        """Block's original linears should be restored after block_loss_fn returns."""
        block = self._make_minimal_block()
        linears = get_block_linears(block, is_mm=False)
        linear_paths = [p for p, _, _ in linears]
        linear_layers = [l for _, l, _ in linears]
        orig_ids = {p: id(_get_nested(block, p)) for p in linear_paths}

        W_fps_np = [np.array(l.weight) for l in linear_layers]
        w_scales_np = [compute_per_channel_scale(W, 4) for W in W_fps_np]
        params = AdaRoundParams(W_fps_np)
        x = mx.array(np.ones((2, 8), dtype=np.float32))
        fp_out = block(x)
        mx.eval(fp_out)

        block_loss_fn(
            params, block, is_mm=False,
            linear_paths=linear_paths,
            linear_layers=linear_layers,
            W_fps_np=W_fps_np,
            w_scales_np=w_scales_np,
            sample_inputs=[x],
            sample_kwargs={},
            fp_outputs=[fp_out],
            b_val=5.0,
        )

        # Originals must be restored
        for p in linear_paths:
            assert id(_get_nested(block, p)) == orig_ids[p]

    def test_gradient_flows_to_alpha(self):
        block = self._make_minimal_block()
        linears = get_block_linears(block, is_mm=False)
        linear_paths = [p for p, _, _ in linears]
        linear_layers = [l for _, l, _ in linears]
        W_fps_np = [np.array(l.weight) for l in linear_layers]
        w_scales_np = [compute_per_channel_scale(W, 4) for W in W_fps_np]
        params = AdaRoundParams(W_fps_np)
        x = mx.array(np.random.randn(2, 8).astype(np.float32))
        fp_out = block(x)
        mx.eval(fp_out)

        def loss_wrapper(params):
            return block_loss_fn(
                params, block, is_mm=False,
                linear_paths=linear_paths,
                linear_layers=linear_layers,
                W_fps_np=W_fps_np,
                w_scales_np=w_scales_np,
                sample_inputs=[x],
                sample_kwargs={},
                fp_outputs=[fp_out],
                b_val=20.0,
            )

        loss_and_grad = nn.value_and_grad(params, loss_wrapper)
        loss_val, grads = loss_and_grad(params)
        mx.eval(loss_val, grads["alphas"][0])

        grad_np = np.array(grads["alphas"][0])
        assert np.any(grad_np != 0.0), "Expected non-zero gradient w.r.t. alpha"


# ---------------------------------------------------------------------------
# TestFinalizeBlock
# ---------------------------------------------------------------------------

class TestFinalizeBlock:

    def test_weight_int_dtype_is_int8(self):
        W_np = _make_weight(8, 4)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = AdaRoundParams([W_np])
        result = finalize_block(params, [W_np], [s_np], ["path.layer"])
        assert result["path.layer"]["weight_int"].dtype == np.int8

    def test_weight_int_within_4bit_range(self):
        W_np = _make_weight(8, 4)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = AdaRoundParams([W_np])
        result = finalize_block(params, [W_np], [s_np], ["path.layer"])
        w_int = result["path.layer"]["weight_int"]
        assert w_int.min() >= -8
        assert w_int.max() <= 7

    def test_scale_shape_matches_out_features(self):
        out, inp = 6, 4
        W_np = _make_weight(out, inp)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = AdaRoundParams([W_np])
        result = finalize_block(params, [W_np], [s_np], ["fc1"])
        assert result["fc1"]["scale"].shape == (out, 1)

    def test_a_scale_is_positive_float(self):
        W_np = _make_weight(4, 4)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = AdaRoundParams([W_np])
        result = finalize_block(params, [W_np], [s_np], ["fc2"])
        assert isinstance(result["fc2"]["a_scale"], float)
        assert result["fc2"]["a_scale"] > 0.0

    def test_multiple_paths_all_returned(self):
        W_list = [_make_weight(4, 4) for _ in range(3)]
        s_list = [compute_per_channel_scale(W, 4) for W in W_list]
        params = AdaRoundParams(W_list)
        paths = ["a.b", "c.d", "e.f"]
        result = finalize_block(params, W_list, s_list, paths)
        assert set(result.keys()) == set(paths)

    def test_hard_rounding_alpha_positive_rounds_up(self):
        """If all alpha >> 0 then rectified_sigmoid ≈ 1 (round up)."""
        W_np = np.array([[2.3, -1.7]], dtype=np.float32)  # (1, 2)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = AdaRoundParams([W_np])
        # Force alpha to be very large (round up)
        params.alphas[0] = mx.array(np.full_like(W_np, 100.0))
        result = finalize_block(params, [W_np], [s_np], ["test"])
        w_int = result["test"]["weight_int"]
        # ceil(W / s) should equal floor(W/s) + 1 (clamped to [-8, 7])
        expected = np.clip(np.ceil(W_np / s_np), -8, 7).astype(np.int8)
        np.testing.assert_array_equal(w_int, expected)

    def test_hard_rounding_alpha_negative_rounds_down(self):
        """If all alpha << 0 then rectified_sigmoid ≈ 0 (round down)."""
        W_np = np.array([[2.3, -1.7]], dtype=np.float32)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = AdaRoundParams([W_np])
        params.alphas[0] = mx.array(np.full_like(W_np, -100.0))
        result = finalize_block(params, [W_np], [s_np], ["test"])
        w_int = result["test"]["weight_int"]
        expected = np.clip(np.floor(W_np / s_np), -8, 7).astype(np.int8)
        np.testing.assert_array_equal(w_int, expected)
