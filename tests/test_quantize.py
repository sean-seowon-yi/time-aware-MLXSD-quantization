"""Tests for phase2 shared utils (quantize.py) and static W4A8 (quantize_static.py)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from src.phase2.quantize import _navigate_to_parent, patch_pipeline_for_quantized_inference
from src.phase2.quantize_static import (
    W4A8StaticLinear,
    fake_quantize_a8_static,
    quantize_model_static,
)

H = 256
FFN_H = 512
GROUP_SIZE = 64


# ====================================================================
# fake_quantize_a8_static
# ====================================================================

class TestFakeQuantizeA8Static:

    def test_output_shape(self):
        x = mx.array(np.random.randn(4, H).astype(np.float32))
        scale = mx.array(0.1, dtype=mx.float32)
        x_hat = fake_quantize_a8_static(x, scale)
        mx.eval(x_hat)
        assert x_hat.shape == x.shape

    def test_zero_input_returns_zero(self):
        x = mx.zeros((4, H))
        scale = mx.array(0.1, dtype=mx.float32)
        x_hat = fake_quantize_a8_static(x, scale)
        mx.eval(x_hat)
        np.testing.assert_allclose(np.array(x_hat), 0.0, atol=1e-7)

    def test_symmetric_range(self):
        x = mx.array(np.random.randn(16, H).astype(np.float32) * 5.0)
        scale = mx.array(0.05, dtype=mx.float32)
        x_hat = fake_quantize_a8_static(x, scale)
        mx.eval(x_hat)
        assert np.max(np.array(x_hat)) <= 127.0 * 0.05 + 1e-6
        assert np.min(np.array(x_hat)) >= -128.0 * 0.05 - 1e-6


# ====================================================================
# W4A8StaticLinear
# ====================================================================

class TestW4A8StaticLinear:

    def _make_w4a8(self, with_b_inv=False, d_in=H, d_out=H):
        linear = nn.Linear(d_in, d_out)
        mx.eval(linear.parameters())
        qlinear = nn.QuantizedLinear.from_linear(
            linear, group_size=GROUP_SIZE, bits=4,
        )
        mx.eval(qlinear.parameters())
        b_inv = mx.ones((d_in,)) * 0.5 if with_b_inv else None
        scale = mx.array(0.1, dtype=mx.float32)
        return W4A8StaticLinear(qlinear, b_inv, scale, per_channel=False)

    def test_forward_shape(self):
        layer = self._make_w4a8()
        x = mx.random.normal((2, H))
        y = layer(x)
        mx.eval(y)
        assert y.shape == (2, H)
        assert np.all(np.isfinite(np.array(y)))

    def test_with_b_inv_attr_present(self):
        layer = self._make_w4a8(with_b_inv=True)
        assert hasattr(layer, "b_inv")

    def test_without_b_inv_attr_absent(self):
        layer = self._make_w4a8(with_b_inv=False)
        assert not hasattr(layer, "b_inv")

    def test_b_inv_changes_output(self):
        linear = nn.Linear(H, H)
        mx.eval(linear.parameters())
        qlinear = nn.QuantizedLinear.from_linear(
            linear, group_size=GROUP_SIZE, bits=4,
        )
        mx.eval(qlinear.parameters())
        scale = mx.array(0.1, dtype=mx.float32)

        layer_no = W4A8StaticLinear(qlinear, b_inv=None, scale=scale)
        layer_yes = W4A8StaticLinear(
            qlinear, b_inv=mx.ones((H,)) * 2.0, scale=scale,
        )

        x = mx.array(np.random.randn(2, H).astype(np.float32))
        y_no = np.array(layer_no(x))
        y_yes = np.array(layer_yes(x))

        assert not np.allclose(y_no, y_yes)

    def test_fc2_dimensions(self):
        layer = self._make_w4a8(d_in=FFN_H, d_out=H)
        x = mx.random.normal((2, FFN_H))
        y = layer(x)
        mx.eval(y)
        assert y.shape == (2, H)


# ====================================================================
# _navigate_to_parent
# ====================================================================

class TestNavigateToParent:

    def test_q_proj(self, mock_mmdit):
        parent, attr = _navigate_to_parent(mock_mmdit, "blocks.0.image.attn.q_proj")
        assert attr == "q_proj"
        assert isinstance(getattr(parent, attr), nn.Linear)

    def test_k_proj(self, mock_mmdit):
        parent, attr = _navigate_to_parent(mock_mmdit, "blocks.0.image.attn.k_proj")
        assert attr == "k_proj"
        layer = getattr(parent, attr)
        assert isinstance(layer, nn.Linear)
        assert not hasattr(layer, "bias") or layer.bias is None

    def test_fc1(self, mock_mmdit):
        parent, attr = _navigate_to_parent(mock_mmdit, "blocks.0.image.mlp.fc1")
        assert attr == "fc1"
        assert isinstance(getattr(parent, attr), nn.Linear)

    def test_fc2(self, mock_mmdit):
        parent, attr = _navigate_to_parent(mock_mmdit, "blocks.0.image.mlp.fc2")
        assert attr == "fc2"

    def test_final_layer(self, mock_mmdit):
        parent, attr = _navigate_to_parent(mock_mmdit, "final_layer.linear")
        assert attr == "linear"
        assert parent is mock_mmdit.final_layer

    def test_context_embedder(self, mock_mmdit):
        parent, attr = _navigate_to_parent(mock_mmdit, "context_embedder")
        assert attr == "context_embedder"
        assert parent is mock_mmdit

    def test_text_side(self, mock_mmdit):
        parent, attr = _navigate_to_parent(mock_mmdit, "blocks.0.text.attn.v_proj")
        assert attr == "v_proj"


# ====================================================================
# quantize_model_static
# ====================================================================

class TestQuantizeModelStatic:

    def test_replaces_layers(self, mock_mmdit, registry, test_config):
        excluded = set(test_config["exclude_layers"])
        b_inv_map = {}
        static_scales = {}
        for entry in registry:
            if entry["name"] in excluded:
                continue
            d_in = FFN_H if entry["family"] == "fc2" else H
            static_scales[entry["name"]] = 0.05
            if entry["family"] in ("o_proj", "fc2") and entry["name"] not in excluded:
                b_inv_map[entry["name"]] = np.ones(d_in, dtype=np.float32)

        layer_meta = quantize_model_static(
            mock_mmdit, registry, b_inv_map, static_scales, test_config,
        )

        for entry in registry:
            if entry["name"] in excluded:
                continue
            parent, attr = _navigate_to_parent(mock_mmdit, entry["name"])
            assert isinstance(getattr(parent, attr), W4A8StaticLinear), (
                f"{entry['name']} should be W4A8StaticLinear"
            )

        assert len(layer_meta) > 0

    def test_excludes_context_embedder(self, mock_mmdit, registry, test_config):
        static_scales = {
            e["name"]: 0.05 for e in registry if e["name"] != "context_embedder"
        }
        quantize_model_static(mock_mmdit, registry, {}, static_scales, test_config)
        assert isinstance(mock_mmdit.context_embedder, nn.Linear)

    def test_layer_meta_has_correct_info(self, mock_mmdit, registry, test_config):
        b_inv_map = {}
        static_scales = {e["name"]: 0.05 for e in registry if e["name"] != "context_embedder"}
        for entry in registry:
            if entry["family"] == "o_proj":
                b_inv_map[entry["name"]] = np.ones(H, dtype=np.float32)

        meta = quantize_model_static(
            mock_mmdit, registry, b_inv_map, static_scales, test_config,
        )

        for name, info in meta.items():
            assert "d_in" in info
            assert "d_out" in info
            assert "has_bias" in info
            assert "bits" in info
            assert "has_b_inv" in info
            assert "per_channel" in info

        o_proj_name = "blocks.0.image.attn.o_proj"
        if o_proj_name in meta:
            assert meta[o_proj_name]["has_b_inv"] is True

    def test_quantized_layer_forward_pass(self, mock_mmdit, registry, test_config):
        static_scales = {
            e["name"]: 0.05 for e in registry if e["name"] != "context_embedder"
        }
        quantize_model_static(mock_mmdit, registry, {}, static_scales, test_config)

        parent, attr = _navigate_to_parent(mock_mmdit, "blocks.0.image.attn.q_proj")
        q_proj = getattr(parent, attr)
        x = mx.random.normal((2, 4, H))
        y = q_proj(x)
        mx.eval(y)
        assert y.shape == (2, 4, H)


# ====================================================================
# patch_pipeline_for_quantized_inference
# ====================================================================

class TestPatchPipeline:

    def test_modulation_dict_returns_adaln(self, mock_mmdit):
        from conftest import MockPipeline

        pipeline = MockPipeline(mock_mmdit)
        patch_pipeline_for_quantized_inference(pipeline)

        result = pipeline.load_mmdit(only_modulation_dict=True)
        assert isinstance(result, list)
        assert len(result) > 0
        for k, v in result:
            assert "adaLN" in k

    def test_full_load_falls_through(self, mock_mmdit):
        from conftest import MockPipeline

        pipeline = MockPipeline(mock_mmdit)
        original_result = pipeline.load_mmdit(only_modulation_dict=False)

        patch_pipeline_for_quantized_inference(pipeline)

        assert pipeline.load_mmdit(only_modulation_dict=False) == original_result
