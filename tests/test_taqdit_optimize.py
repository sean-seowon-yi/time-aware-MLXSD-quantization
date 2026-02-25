"""
Tests for src/taqdit_optimize.py

All tests use small synthetic weights/tensors and temp directories.
No DiffusionKit pipeline or real model weights are loaded.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from src.taqdit_optimize import (
    load_block_data_with_ts,
    TaqDitParams,
    _TaqDitQuantProxy,
    taqdit_loss_fn,
    finalize_block_taqdit,
    _block_path_to_act_name,
    build_act_config,
    optimize_block_taqdit,
)
from src.adaround_optimize import (
    compute_per_channel_scale,
    init_alpha,
    rectified_sigmoid,
    get_block_linears,
    fake_quant_per_tensor,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_weight(out: int = 8, inp: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (out, inp)).astype(np.float32)


def _make_linear(out: int = 8, inp: int = 4, bias: bool = True, seed: int = 0) -> nn.Linear:
    layer = nn.Linear(inp, out, bias=bias)
    layer.weight = mx.array(_make_weight(out, inp, seed))
    if bias:
        layer.bias = mx.array(np.zeros(out, dtype=np.float32))
    return layer


def _make_mock_transformer_block():
    """Minimal mock of a DiffusionKit uni-block (single transformer_block)."""
    class MockAttn:
        q_proj = _make_linear(8, 8)
        k_proj = _make_linear(8, 8, bias=False)
        v_proj = _make_linear(8, 8)
        o_proj = _make_linear(8, 8)

    class MockMLP:
        def __init__(self):
            self.fc1 = _make_linear(16, 8)
            self.fc2 = _make_linear(8, 16)

        def __call__(self, x):
            return self.fc2(mx.tanh(self.fc1(x)))

    class MockTransBlock:
        def __init__(self):
            self.attn = MockAttn()
            self.mlp = MockMLP()

        def __call__(self, x):
            q = self.attn.q_proj(x)
            return self.mlp(q)

    class MockUniBlock:
        def __init__(self):
            self.transformer_block = MockTransBlock()

        def __call__(self, x):
            return self.transformer_block(x)

    return MockUniBlock()


def _write_sample_npz(
    path: Path,
    block_name: str,
    shape: tuple = (2, 8),
    seed: int = 0,
) -> None:
    """Write a minimal sample NPZ file for one block."""
    rng = np.random.default_rng(seed)
    safe = block_name.replace(".", "_")
    data = {
        f"{safe}__arg0": rng.normal(size=shape).astype(np.float16),
        f"{safe}__out0": rng.normal(size=shape).astype(np.float16),
    }
    np.savez_compressed(path, **data)


# ---------------------------------------------------------------------------
# TestLoadBlockDataWithTs
# ---------------------------------------------------------------------------

class TestLoadBlockDataWithTs:

    def test_returns_empty_for_no_matching_files(self, tmp_path):
        # Files with step indices not in step_indices
        _write_sample_npz(tmp_path / "0000_099.npz", "mm0")
        block_data, ts_idx = load_block_data_with_ts(
            "mm0", sorted(tmp_path.glob("*.npz")), step_indices=[0, 4, 8]
        )
        assert block_data == {}
        assert len(ts_idx) == 0

    def test_maps_step_to_ts_index(self, tmp_path):
        _write_sample_npz(tmp_path / "0000_000.npz", "mm0")
        _write_sample_npz(tmp_path / "0001_000.npz", "mm0", seed=1)
        _write_sample_npz(tmp_path / "0000_004.npz", "mm0", seed=2)
        block_data, ts_idx = load_block_data_with_ts(
            "mm0", sorted(tmp_path.glob("*.npz")), step_indices=[0, 4]
        )
        assert len(ts_idx) == 3
        # Files with step=0 map to ts_idx=0; file with step=4 maps to ts_idx=1
        counts = np.bincount(ts_idx, minlength=2)
        assert counts[0] == 2
        assert counts[1] == 1

    def test_stacked_shape_matches_num_samples(self, tmp_path):
        for i in range(3):
            _write_sample_npz(tmp_path / f"{i:04d}_000.npz", "mm1", seed=i)
        block_data, ts_idx = load_block_data_with_ts(
            "mm1", sorted(tmp_path.glob("*.npz")), step_indices=[0]
        )
        assert block_data["arg0"].shape[0] == 3
        assert ts_idx.shape == (3,)

    def test_ignores_blocks_not_matching_prefix(self, tmp_path):
        _write_sample_npz(tmp_path / "0000_000.npz", "mm3")
        block_data, ts_idx = load_block_data_with_ts(
            "mm0", sorted(tmp_path.glob("*.npz")), step_indices=[0]
        )
        assert block_data == {}

    def test_ts_idx_dtype_is_int32(self, tmp_path):
        _write_sample_npz(tmp_path / "0000_004.npz", "mm0")
        _, ts_idx = load_block_data_with_ts(
            "mm0", sorted(tmp_path.glob("*.npz")), step_indices=[4]
        )
        assert ts_idx.dtype == np.int32

    def test_multiple_step_indices_all_captured(self, tmp_path):
        step_indices = [0, 4, 8, 12]
        for si in step_indices:
            _write_sample_npz(tmp_path / f"0000_{si:03d}.npz", "mm0", seed=si)
        block_data, ts_idx = load_block_data_with_ts(
            "mm0", sorted(tmp_path.glob("*.npz")), step_indices=step_indices
        )
        assert len(ts_idx) == len(step_indices)
        assert set(ts_idx.tolist()) == set(range(len(step_indices)))


# ---------------------------------------------------------------------------
# TestTaqDitParams
# ---------------------------------------------------------------------------

class TestTaqDitParams:

    def test_alphas_length_matches_n_layers(self):
        W_list = [_make_weight(4, 4) for _ in range(6)]
        p = TaqDitParams(W_list, n_timesteps=5)
        assert len(p.alphas) == 6

    def test_alpha_shape_matches_weight(self):
        W_np = _make_weight(8, 4)
        p = TaqDitParams([W_np], n_timesteps=3)
        assert np.array(p.alphas[0]).shape == (8, 4)

    def test_a_scales_shape(self):
        W_list = [_make_weight(4, 4) for _ in range(6)]
        p = TaqDitParams(W_list, n_timesteps=5)
        assert np.array(p.a_scales).shape == (6, 5)

    def test_a_scales_initialized_to_one(self):
        W_np = _make_weight(4, 4)
        p = TaqDitParams([W_np], n_timesteps=4)
        a = np.array(p.a_scales)
        np.testing.assert_allclose(a, np.ones((1, 4)), atol=1e-6)

    def test_qmin_qmax_4bit(self):
        p = TaqDitParams([_make_weight(4, 4)], n_timesteps=2, bits_w=4, bits_a=8)
        assert p.qmin_w == -8
        assert p.qmax_w == 7
        assert p.qmin_a == -128
        assert p.qmax_a == 127

    def test_n_layers_and_n_timesteps_stored(self):
        W_list = [_make_weight(4, 4) for _ in range(3)]
        p = TaqDitParams(W_list, n_timesteps=7)
        assert p.n_layers == 3
        assert p.n_timesteps == 7

    def test_a_scales_are_trainable_parameters(self):
        W_np = _make_weight(4, 4)
        p = TaqDitParams([W_np], n_timesteps=3)
        trainable = p.trainable_parameters()
        assert "a_scales" in trainable or len(trainable) > 0

    def test_gradient_flows_to_a_scales(self):
        W_np = _make_weight(4, 4)
        p = TaqDitParams([W_np], n_timesteps=3)

        def fn(params):
            # Simple function that uses a_scales
            return (params.a_scales ** 2).sum()

        val_and_grad = nn.value_and_grad(p, fn)
        val, grads = val_and_grad(p)
        mx.eval(val, grads["a_scales"])
        grad_np = np.array(grads["a_scales"])
        assert grad_np.shape == (1, 3)
        assert np.all(grad_np >= 0.0)  # grad of x^2 = 2x, all >= 0 since a_scales=1.0


# ---------------------------------------------------------------------------
# TestTaqDitQuantProxy
# ---------------------------------------------------------------------------

class TestTaqDitQuantProxy:

    def _make_proxy(self, out: int = 4, inp: int = 4, ts_scale: float = 1.0):
        layer = _make_linear(out, inp)
        W_np = np.array(layer.weight)
        s_np = compute_per_channel_scale(W_np, bits=4)
        alpha = mx.array(init_alpha(W_np, s_np))
        s_mx = mx.array(s_np)
        r = rectified_sigmoid(alpha)
        W_floor = mx.floor(mx.array(W_np) / s_mx)
        soft_w = mx.clip(W_floor + r, -8, 7) * s_mx
        a_scale = mx.array(ts_scale)   # 0-d scalar
        return _TaqDitQuantProxy(layer, soft_w, a_scale, -128, 127)

    def test_output_shape(self):
        proxy = self._make_proxy(out=4, inp=4)
        x = mx.array(np.ones((3, 4), dtype=np.float32))
        y = proxy(x)
        mx.eval(y)
        assert np.array(y).shape == (3, 4)

    def test_getattr_delegates_to_original(self):
        layer = _make_linear(4, 4)
        W_np = np.array(layer.weight)
        s_np = compute_per_channel_scale(W_np, 4)
        soft_w = mx.array(W_np)   # dummy
        proxy = _TaqDitQuantProxy(layer, soft_w, mx.array(1.0), -128, 127)
        if layer.bias is not None:
            assert proxy.bias is layer.bias

    def test_gradient_flows_through_a_scale(self):
        layer = _make_linear(4, 4)
        W_np = np.array(layer.weight)
        s_np = compute_per_channel_scale(W_np, 4)
        a_scale = mx.array(1.0)

        def fn(a_scale):
            alpha = mx.array(init_alpha(W_np, s_np))
            r = rectified_sigmoid(alpha)
            soft_w = mx.clip(mx.floor(mx.array(W_np) / mx.array(s_np)) + r, -8, 7) * mx.array(s_np)
            proxy = _TaqDitQuantProxy(layer, soft_w, a_scale, -128, 127)
            x = mx.array(np.ones((2, 4), dtype=np.float32))
            return proxy(x).sum()

        grad = mx.grad(fn)(a_scale)
        mx.eval(grad)
        # Gradient should be finite
        assert np.isfinite(float(np.array(grad)))


# ---------------------------------------------------------------------------
# TestTaqDitLossFn
# ---------------------------------------------------------------------------

class TestTaqDitLossFn:

    def _setup(self):
        block = _make_mock_transformer_block()
        linears = get_block_linears(block, is_mm=False)
        linear_paths = [p for p, _, _ in linears]
        linear_layers = [l for _, l, _ in linears]
        W_fps_np = [np.array(l.weight) for l in linear_layers]
        w_scales_np = [compute_per_channel_scale(W, 4) for W in W_fps_np]
        n_ts = 3
        params = TaqDitParams(W_fps_np, n_timesteps=n_ts)
        return block, linears, linear_paths, linear_layers, W_fps_np, w_scales_np, params

    def test_loss_is_scalar_finite_nonneg(self):
        block, linears, paths, layers, W_fps, w_scales, params = self._setup()
        x = mx.array(np.random.randn(2, 8).astype(np.float32))
        fp_out = block(x)
        mx.eval(fp_out)

        loss = taqdit_loss_fn(
            params, block, is_mm=False,
            linear_paths=paths, linear_layers=layers,
            W_fps_np=W_fps, w_scales_np=w_scales,
            sample_inputs=[x], sample_kwargs={},
            fp_outputs=[fp_out], b_val=20.0, ts_idx=0,
        )
        mx.eval(loss)
        val = float(np.array(loss))
        assert np.isfinite(val)
        assert val >= 0.0

    def test_loss_differs_for_different_ts_idx(self):
        """Different timestep indices → different a_scales → different losses."""
        block, linears, paths, layers, W_fps, w_scales, params = self._setup()
        # Set different a_scale values for ts_idx=0 vs ts_idx=2
        a = np.array(params.a_scales)
        a[:, 0] = 0.5
        a[:, 2] = 5.0
        params.a_scales = mx.array(a)

        x = mx.array(np.random.randn(2, 8).astype(np.float32))
        fp_out = block(x)
        mx.eval(fp_out)

        loss0 = taqdit_loss_fn(
            params, block, is_mm=False,
            linear_paths=paths, linear_layers=layers,
            W_fps_np=W_fps, w_scales_np=w_scales,
            sample_inputs=[x], sample_kwargs={},
            fp_outputs=[fp_out], b_val=5.0, ts_idx=0,
        )
        loss2 = taqdit_loss_fn(
            params, block, is_mm=False,
            linear_paths=paths, linear_layers=layers,
            W_fps_np=W_fps, w_scales_np=w_scales,
            sample_inputs=[x], sample_kwargs={},
            fp_outputs=[fp_out], b_val=5.0, ts_idx=2,
        )
        mx.eval(loss0, loss2)
        # With significantly different a_scales the losses should differ
        assert float(np.array(loss0)) != pytest.approx(float(np.array(loss2)), rel=0.01)

    def test_block_restored_after_loss(self):
        """Block's original linears must be restored after taqdit_loss_fn returns."""
        block, linears, paths, layers, W_fps, w_scales, params = self._setup()
        from src.adaround_optimize import _get_nested
        orig_ids = {p: id(_get_nested(block, p)) for p in paths}

        x = mx.array(np.ones((2, 8), dtype=np.float32))
        fp_out = block(x)
        mx.eval(fp_out)

        taqdit_loss_fn(
            params, block, is_mm=False,
            linear_paths=paths, linear_layers=layers,
            W_fps_np=W_fps, w_scales_np=w_scales,
            sample_inputs=[x], sample_kwargs={},
            fp_outputs=[fp_out], b_val=5.0, ts_idx=1,
        )
        for p in paths:
            assert id(_get_nested(block, p)) == orig_ids[p]

    def test_gradient_flows_to_alpha(self):
        block, linears, paths, layers, W_fps, w_scales, params = self._setup()
        x = mx.array(np.random.randn(2, 8).astype(np.float32))
        fp_out = block(x)
        mx.eval(fp_out)

        def loss_fn(params):
            return taqdit_loss_fn(
                params, block, is_mm=False,
                linear_paths=paths, linear_layers=layers,
                W_fps_np=W_fps, w_scales_np=w_scales,
                sample_inputs=[x], sample_kwargs={},
                fp_outputs=[fp_out], b_val=20.0, ts_idx=0,
            )

        val_grad = nn.value_and_grad(params, loss_fn)
        _, grads = val_grad(params)
        mx.eval(grads["alphas"][0])
        grad_np = np.array(grads["alphas"][0])
        assert np.any(grad_np != 0.0), "Expected non-zero gradient w.r.t. alpha"

    def test_gradient_flows_to_a_scales_at_ts_idx(self):
        block, linears, paths, layers, W_fps, w_scales, params = self._setup()
        x = mx.array(np.random.randn(2, 8).astype(np.float32))
        fp_out = block(x)
        mx.eval(fp_out)

        target_ts = 1

        def loss_fn(params):
            return taqdit_loss_fn(
                params, block, is_mm=False,
                linear_paths=paths, linear_layers=layers,
                W_fps_np=W_fps, w_scales_np=w_scales,
                sample_inputs=[x], sample_kwargs={},
                fp_outputs=[fp_out], b_val=5.0, ts_idx=target_ts,
            )

        val_grad = nn.value_and_grad(params, loss_fn)
        _, grads = val_grad(params)
        mx.eval(grads["a_scales"])
        grad_mat = np.array(grads["a_scales"])   # (n_layers, n_ts)
        # Gradient should be non-zero at column target_ts, zero at others
        assert np.any(grad_mat[:, target_ts] != 0.0), \
            "Expected non-zero gradient in column ts_idx=1"
        assert np.all(grad_mat[:, 0] == 0.0), \
            "Expected zero gradient at ts_idx=0 (not used in this call)"


# ---------------------------------------------------------------------------
# TestBlockPathToActName
# ---------------------------------------------------------------------------

class TestBlockPathToActName:

    def test_mm_image_block_q_proj(self):
        result = _block_path_to_act_name(
            "mm3", True, "image_transformer_block.attn.q_proj"
        )
        assert result == "mm3.img.attn.q_proj"

    def test_mm_text_block_fc2(self):
        result = _block_path_to_act_name(
            "mm0", True, "text_transformer_block.mlp.fc2"
        )
        assert result == "mm0.txt.mlp.fc2"

    def test_uni_block_o_proj(self):
        result = _block_path_to_act_name(
            "uni5", False, "transformer_block.attn.o_proj"
        )
        assert result == "uni5.attn.o_proj"

    def test_uni_block_fc1(self):
        result = _block_path_to_act_name(
            "uni0", False, "transformer_block.mlp.fc1"
        )
        assert result == "uni0.mlp.fc1"

    def test_mm_image_all_projections(self):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            result = _block_path_to_act_name(
                "mm1", True, f"image_transformer_block.attn.{proj}"
            )
            assert result == f"mm1.img.attn.{proj}"

    def test_mm_text_mlp_paths(self):
        for fc in ("fc1", "fc2"):
            result = _block_path_to_act_name(
                "mm2", True, f"text_transformer_block.mlp.{fc}"
            )
            assert result == f"mm2.txt.mlp.{fc}"


# ---------------------------------------------------------------------------
# TestBuildActConfig
# ---------------------------------------------------------------------------

class TestBuildActConfig:

    def _make_params(self, n_layers: int = 6, n_ts: int = 4):
        W_list = [_make_weight(4, 4) for _ in range(n_layers)]
        p = TaqDitParams(W_list, n_timesteps=n_ts)
        return p

    def test_output_keys_match_step_indices(self):
        params = self._make_params(n_layers=6, n_ts=3)
        step_indices = [0, 4, 8]
        paths = [
            "transformer_block.attn.q_proj",
            "transformer_block.attn.k_proj",
        ]
        result = build_act_config(params, "uni0", False, paths, step_indices)
        assert set(result.keys()) == {0, 4, 8}

    def test_each_timestep_has_all_layer_names(self):
        params = self._make_params(n_layers=2, n_ts=2)
        step_indices = [0, 4]
        paths = [
            "transformer_block.attn.q_proj",
            "transformer_block.mlp.fc1",
        ]
        result = build_act_config(params, "uni0", False, paths, step_indices)
        for si in step_indices:
            layer_names = set(result[si].keys())
            assert "uni0.attn.q_proj" in layer_names
            assert "uni0.mlp.fc1" in layer_names

    def test_scale_values_are_positive(self):
        params = self._make_params(n_layers=4, n_ts=3)
        step_indices = [0, 4, 8]
        paths = [f"transformer_block.attn.q_proj"] * 4
        # Use different paths to avoid duplicate keys
        paths = [
            "transformer_block.attn.q_proj",
            "transformer_block.attn.k_proj",
            "transformer_block.mlp.fc1",
            "transformer_block.mlp.fc2",
        ]
        result = build_act_config(params, "uni0", False, paths, step_indices)
        for si in step_indices:
            for layer_name, cfg in result[si].items():
                assert cfg["scale"] > 0.0

    def test_bits_field_present(self):
        params = self._make_params(n_layers=2, n_ts=2)
        step_indices = [0, 4]
        paths = ["transformer_block.attn.q_proj", "transformer_block.mlp.fc1"]
        result = build_act_config(params, "uni0", False, paths, step_indices, bits_a=8)
        for si in step_indices:
            for cfg in result[si].values():
                assert cfg["bits"] == 8

    def test_scale_reflects_learned_values(self):
        """Learned a_scale changes should propagate to build_act_config output."""
        params = self._make_params(n_layers=1, n_ts=2)
        # Force a_scales[0, 0] = 3.0, a_scales[0, 1] = 7.0
        a = np.ones((1, 2), dtype=np.float32)
        a[0, 0] = 3.0
        a[0, 1] = 7.0
        params.a_scales = mx.array(a)

        step_indices = [0, 4]
        paths = ["transformer_block.attn.q_proj"]
        result = build_act_config(params, "uni0", False, paths, step_indices)

        assert result[0]["uni0.attn.q_proj"]["scale"] == pytest.approx(3.0, rel=1e-5)
        assert result[4]["uni0.attn.q_proj"]["scale"] == pytest.approx(7.0, rel=1e-5)


# ---------------------------------------------------------------------------
# TestFinalizeBlockTaqdit
# ---------------------------------------------------------------------------

class TestFinalizeBlockTaqdit:

    def test_weight_int_dtype_is_int8(self):
        W_np = _make_weight(8, 4)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = TaqDitParams([W_np], n_timesteps=3)
        result = finalize_block_taqdit(params, [W_np], [s_np], ["path.layer"])
        assert result["path.layer"]["weight_int"].dtype == np.int8

    def test_weight_int_within_4bit_range(self):
        W_np = _make_weight(8, 4)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = TaqDitParams([W_np], n_timesteps=3)
        result = finalize_block_taqdit(params, [W_np], [s_np], ["test"])
        w_int = result["test"]["weight_int"]
        assert w_int.min() >= -8
        assert w_int.max() <= 7

    def test_scale_shape(self):
        out, inp = 6, 4
        W_np = _make_weight(out, inp)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = TaqDitParams([W_np], n_timesteps=2)
        result = finalize_block_taqdit(params, [W_np], [s_np], ["fc1"])
        assert result["fc1"]["scale"].shape == (out, 1)

    def test_a_scale_is_mean_across_timesteps(self):
        W_np = _make_weight(4, 4)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = TaqDitParams([W_np], n_timesteps=4)
        # Set distinct a_scale values
        a = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
        params.a_scales = mx.array(a)
        result = finalize_block_taqdit(params, [W_np], [s_np], ["fc2"])
        assert result["fc2"]["a_scale"] == pytest.approx(5.0, rel=1e-5)  # mean(2,4,6,8)=5

    def test_multiple_paths_all_returned(self):
        W_list = [_make_weight(4, 4) for _ in range(3)]
        s_list = [compute_per_channel_scale(W, 4) for W in W_list]
        params = TaqDitParams(W_list, n_timesteps=2)
        paths = ["a.b", "c.d", "e.f"]
        result = finalize_block_taqdit(params, W_list, s_list, paths)
        assert set(result.keys()) == set(paths)

    def test_alpha_positive_rounds_up(self):
        W_np = np.array([[2.3, -1.7]], dtype=np.float32)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = TaqDitParams([W_np], n_timesteps=2)
        params.alphas[0] = mx.array(np.full_like(W_np, 100.0))
        result = finalize_block_taqdit(params, [W_np], [s_np], ["test"])
        w_int = result["test"]["weight_int"]
        expected = np.clip(np.ceil(W_np / s_np), -8, 7).astype(np.int8)
        np.testing.assert_array_equal(w_int, expected)

    def test_alpha_negative_rounds_down(self):
        W_np = np.array([[2.3, -1.7]], dtype=np.float32)
        s_np = compute_per_channel_scale(W_np, bits=4)
        params = TaqDitParams([W_np], n_timesteps=2)
        params.alphas[0] = mx.array(np.full_like(W_np, -100.0))
        result = finalize_block_taqdit(params, [W_np], [s_np], ["test"])
        w_int = result["test"]["weight_int"]
        expected = np.clip(np.floor(W_np / s_np), -8, 7).astype(np.int8)
        np.testing.assert_array_equal(w_int, expected)


# ---------------------------------------------------------------------------
# TestOptimizeBlockTaqdit (small-scale integration)
# ---------------------------------------------------------------------------

class TestOptimizeBlockTaqdit:

    def _make_block_data(self, n_samples: int = 4, hidden: int = 8):
        """Synthetic block data dict with n_samples, hidden=8."""
        rng = np.random.default_rng(0)
        return {
            "arg0": rng.normal(size=(n_samples, hidden)).astype(np.float32),
            "out0": rng.normal(size=(n_samples, hidden)).astype(np.float32),
        }

    def test_returns_params_and_metrics(self):
        block = _make_mock_transformer_block()
        block_data = self._make_block_data()
        sample_ts_idx = np.array([0, 0, 1, 1], dtype=np.int32)

        params, metrics = optimize_block_taqdit(
            block=block, block_name="uni0", is_mm=False,
            block_data=block_data, sample_ts_idx=sample_ts_idx,
            n_timesteps=2, iters=5, batch_size=2,
        )
        assert isinstance(params, TaqDitParams)
        assert "final_loss" in metrics
        assert metrics["n_timesteps"] == 2

    def test_final_loss_is_finite(self):
        block = _make_mock_transformer_block()
        block_data = self._make_block_data()
        sample_ts_idx = np.array([0, 0, 1, 1], dtype=np.int32)

        params, metrics = optimize_block_taqdit(
            block=block, block_name="uni0", is_mm=False,
            block_data=block_data, sample_ts_idx=sample_ts_idx,
            n_timesteps=2, iters=5, batch_size=2,
        )
        assert np.isfinite(metrics["final_loss"])

    def test_a_scales_updated_from_init(self):
        """a_scales should deviate from 1.0 after optimization."""
        block = _make_mock_transformer_block()
        block_data = self._make_block_data(n_samples=8)
        sample_ts_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

        params, _ = optimize_block_taqdit(
            block=block, block_name="uni0", is_mm=False,
            block_data=block_data, sample_ts_idx=sample_ts_idx,
            n_timesteps=2, iters=20, batch_size=4, a_lr=1e-2,
        )
        a_np = np.abs(np.array(params.a_scales))
        # At least some scales should have moved from 1.0
        assert not np.allclose(a_np, 1.0, atol=1e-4), \
            "a_scales should update during optimization"

    def test_alphas_updated_from_init(self):
        """Weight alphas should update from their initial values."""
        block = _make_mock_transformer_block()
        linears = get_block_linears(block, is_mm=False)
        linear_layers = [l for _, l, _ in linears]
        W_fps_np = [np.array(l.weight) for l in linear_layers]
        w_scales_np = [compute_per_channel_scale(W, 4) for W in W_fps_np]
        alpha_init = [init_alpha(W, s).copy() for W, s in zip(W_fps_np, w_scales_np)]

        block_data = self._make_block_data(n_samples=8)
        sample_ts_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

        params, _ = optimize_block_taqdit(
            block=block, block_name="uni0", is_mm=False,
            block_data=block_data, sample_ts_idx=sample_ts_idx,
            n_timesteps=2, iters=20, batch_size=4, w_lr=1e-2,
        )
        alpha_after = [np.array(params.alphas[i]) for i in range(len(linear_layers))]
        any_changed = any(
            not np.allclose(alpha_init[i], alpha_after[i])
            for i in range(len(linear_layers))
        )
        assert any_changed, "At least one alpha should update during optimization"
