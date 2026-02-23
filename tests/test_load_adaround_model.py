"""
Tests for src/load_adaround_model.py

All tests use temp directories and mock objects — no DiffusionKit pipeline
or real model weights required.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from src.load_adaround_model import (
    load_adaround_weights,
    dequantize,
    inject_weights,
    weight_diff_stats,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_block_npz(
    path: Path,
    linear_paths: list,
    out: int = 8,
    inp: int = 4,
    bits_w: int = 4,
    bits_a: int = 8,
) -> None:
    """Write a fake block .npz in the format produced by adaround_optimize.py."""
    data: Dict[str, np.ndarray] = {}
    qmax = 2 ** (bits_w - 1) - 1  # 7 for 4-bit
    rng = np.random.default_rng(0)

    for lpath in linear_paths:
        safe = lpath.replace(".", "_")
        weight_int = rng.integers(-qmax - 1, qmax + 1, (out, inp), dtype=np.int8)
        scale = rng.random((out, 1), dtype=np.float32) + 0.01
        a_scale = np.array([0.5], dtype=np.float32)
        data[f"{safe}__weight_int"] = weight_int
        data[f"{safe}__scale"] = scale
        data[f"{safe}__a_scale"] = a_scale

    np.savez_compressed(path, **data)


def _make_output_dir(
    tmp_path: Path,
    block_names=("mm0", "mm1", "uni0"),
    linear_paths=("attn.q_proj", "mlp.fc1"),
    bits_w: int = 4,
    bits_a: int = 8,
) -> Path:
    """Create a minimal adaround_optimize.py output directory."""
    output_dir = tmp_path / "quant_out"
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True)

    config = {
        "format": "adaround_v1",
        "bits_w": bits_w,
        "bits_a": bits_a,
        "iters": 20000,
        "batch_size": 16,
        "w_lr": 1e-3,
        "a_lr": 4e-5,
        "n_blocks_quantised": len(block_names),
        # Include quant_paths so load_adaround_weights can resolve paths unambiguously
        "block_metrics": [
            {"block_name": bname, "quant_paths": list(linear_paths)}
            for bname in block_names
        ],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)

    for bname in block_names:
        _make_block_npz(weights_dir / f"{bname}.npz", list(linear_paths))

    return output_dir


# Minimal mock of a DiffusionKit-like block hierarchy

class _MockLinear:
    def __init__(self, out=8, inp=4):
        self.weight = mx.array(
            np.random.default_rng(42).random((out, inp), dtype=np.float32)
        )
        self.bias = None


def _make_mock_transformer_block():
    class _Attn:
        q_proj = _MockLinear(8, 8)
        k_proj = _MockLinear(8, 8)
        v_proj = _MockLinear(8, 8)
        o_proj = _MockLinear(8, 8)

    class _MLP:
        fc1 = _MockLinear(16, 8)
        fc2 = _MockLinear(8, 16)

    class _TB:
        attn = _Attn()
        mlp = _MLP()

    return _TB()


def _make_mock_pipeline(n_mm=2, n_uni=2):
    class _MMDiT:
        pass

    mmdit = _MMDiT()
    mmdit.multimodal_transformer_blocks = [
        type("MM", (), {
            "image_transformer_block": _make_mock_transformer_block(),
            "text_transformer_block": _make_mock_transformer_block(),
        })()
        for _ in range(n_mm)
    ]
    mmdit.unified_transformer_blocks = [
        type("Uni", (), {
            "transformer_block": _make_mock_transformer_block(),
        })()
        for _ in range(n_uni)
    ]

    class _Pipeline:
        pass

    p = _Pipeline()
    p.mmdit = mmdit
    return p


# ---------------------------------------------------------------------------
# TestLoadAdaRoundWeights
# ---------------------------------------------------------------------------

class TestLoadAdaRoundWeights:

    def test_returns_config_dict(self, tmp_path):
        out_dir = _make_output_dir(tmp_path)
        config, _ = load_adaround_weights(out_dir)
        assert isinstance(config, dict)
        assert config["bits_w"] == 4
        assert config["bits_a"] == 8

    def test_returns_all_blocks(self, tmp_path):
        blocks = ("mm0", "mm1", "uni0")
        out_dir = _make_output_dir(tmp_path, block_names=blocks)
        _, qw = load_adaround_weights(out_dir)
        assert set(qw.keys()) == set(blocks)

    def test_weight_int_is_int8(self, tmp_path):
        out_dir = _make_output_dir(tmp_path, block_names=("mm0",),
                                   linear_paths=("attn.q_proj",))
        _, qw = load_adaround_weights(out_dir)
        assert qw["mm0"]["attn.q_proj"]["weight_int"].dtype == np.int8

    def test_scale_is_float32(self, tmp_path):
        out_dir = _make_output_dir(tmp_path, block_names=("uni0",),
                                   linear_paths=("mlp.fc2",))
        _, qw = load_adaround_weights(out_dir)
        assert qw["uni0"]["mlp.fc2"]["scale"].dtype == np.float32

    def test_a_scale_is_float(self, tmp_path):
        out_dir = _make_output_dir(tmp_path, block_names=("mm0",),
                                   linear_paths=("attn.q_proj",))
        _, qw = load_adaround_weights(out_dir)
        assert isinstance(qw["mm0"]["attn.q_proj"]["a_scale"], float)

    def test_bits_propagated_from_config(self, tmp_path):
        out_dir = _make_output_dir(tmp_path, block_names=("mm0",),
                                   linear_paths=("attn.q_proj",),
                                   bits_w=8)
        _, qw = load_adaround_weights(out_dir)
        assert qw["mm0"]["attn.q_proj"]["bits_w"] == 8

    def test_multiple_linears_per_block(self, tmp_path):
        paths = ("attn.q_proj", "attn.k_proj", "mlp.fc1", "mlp.fc2")
        out_dir = _make_output_dir(tmp_path, block_names=("mm0",),
                                   linear_paths=paths)
        _, qw = load_adaround_weights(out_dir)
        assert set(qw["mm0"].keys()) == set(paths)

    def test_raises_if_config_missing(self, tmp_path):
        out_dir = tmp_path / "empty"
        out_dir.mkdir()
        (out_dir / "weights").mkdir()
        with pytest.raises(FileNotFoundError):
            load_adaround_weights(out_dir)

    def test_raises_if_weights_dir_missing(self, tmp_path):
        out_dir = tmp_path / "no_weights"
        out_dir.mkdir()
        with open(out_dir / "config.json", "w") as f:
            json.dump({"bits_w": 4, "bits_a": 8}, f)
        with pytest.raises(FileNotFoundError):
            load_adaround_weights(out_dir)

    def test_empty_weights_dir_returns_empty(self, tmp_path):
        out_dir = tmp_path / "empty_weights"
        (out_dir / "weights").mkdir(parents=True)
        with open(out_dir / "config.json", "w") as f:
            json.dump({"bits_w": 4, "bits_a": 8}, f)
        _, qw = load_adaround_weights(out_dir)
        assert qw == {}

    def test_partial_npz_keys_skipped(self, tmp_path):
        """NPZ file with only weight_int (no scale) should not produce an entry."""
        out_dir = tmp_path / "partial"
        weights_dir = out_dir / "weights"
        weights_dir.mkdir(parents=True)
        with open(out_dir / "config.json", "w") as f:
            json.dump({"bits_w": 4, "bits_a": 8}, f)
        # Write NPZ with only weight_int, no scale
        np.savez_compressed(
            weights_dir / "mm0.npz",
            **{"attn_q_proj__weight_int": np.zeros((4, 4), dtype=np.int8)}
        )
        _, qw = load_adaround_weights(out_dir)
        # Entry should be absent (incomplete data)
        assert "mm0" not in qw


# ---------------------------------------------------------------------------
# TestDequantize
# ---------------------------------------------------------------------------

class TestDequantize:

    def test_output_dtype_is_float16(self):
        w = np.array([[1, -2], [3, -4]], dtype=np.int8)
        s = np.array([[1.0], [0.5]], dtype=np.float32)
        out = dequantize(w, s)
        assert out.dtype == np.float16

    def test_output_shape_preserved(self):
        w = np.zeros((6, 4), dtype=np.int8)
        s = np.ones((6, 1), dtype=np.float32)
        out = dequantize(w, s)
        assert out.shape == (6, 4)

    def test_values_are_weight_times_scale(self):
        w = np.array([[2, -3]], dtype=np.int8)
        s = np.array([[0.5]], dtype=np.float32)
        out = dequantize(w, s)
        expected = np.array([[1.0, -1.5]], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-3)

    def test_zero_weight_gives_zero(self):
        w = np.zeros((4, 4), dtype=np.int8)
        s = np.ones((4, 1), dtype=np.float32)
        out = dequantize(w, s)
        np.testing.assert_array_equal(out, np.zeros((4, 4), dtype=np.float16))

    def test_per_channel_scale_applied_row_wise(self):
        w = np.array([[1, 1], [1, 1]], dtype=np.int8)
        s = np.array([[2.0], [3.0]], dtype=np.float32)
        out = dequantize(w, s)
        # Row 0 all → 2.0; Row 1 all → 3.0
        np.testing.assert_allclose(out[0], [2.0, 2.0], atol=1e-3)
        np.testing.assert_allclose(out[1], [3.0, 3.0], atol=1e-3)


# ---------------------------------------------------------------------------
# TestInjectWeights
# ---------------------------------------------------------------------------

def _simple_quant_weights(block_name: str, linear_path: str,
                           out: int = 8, inp: int = 8) -> dict:
    """Build a minimal quant_weights dict for inject_weights()."""
    w_int = np.ones((out, inp), dtype=np.int8)  # all 1s
    scale = np.full((out, 1), 0.5, dtype=np.float32)
    return {
        block_name: {
            linear_path: {
                "weight_int": w_int,
                "scale": scale,
                "a_scale": 1.0,
                "bits_w": 4,
                "bits_a": 8,
            }
        }
    }


class TestInjectWeights:

    def test_returns_count_of_injected_layers(self, tmp_path):
        pipeline = _make_mock_pipeline(n_mm=2, n_uni=2)
        qw = _simple_quant_weights(
            "mm0", "image_transformer_block.attn.q_proj"
        )
        n = inject_weights(pipeline, qw)
        assert n == 1

    def test_multiple_linears_counted(self, tmp_path):
        pipeline = _make_mock_pipeline(n_mm=2, n_uni=2)
        qw = {
            "mm0": {
                "image_transformer_block.attn.q_proj": {
                    "weight_int": np.ones((8, 8), dtype=np.int8),
                    "scale": np.ones((8, 1), dtype=np.float32),
                    "a_scale": 1.0, "bits_w": 4, "bits_a": 8,
                },
                "image_transformer_block.mlp.fc1": {
                    "weight_int": np.ones((16, 8), dtype=np.int8),
                    "scale": np.ones((16, 1), dtype=np.float32),
                    "a_scale": 1.0, "bits_w": 4, "bits_a": 8,
                },
            }
        }
        n = inject_weights(pipeline, qw)
        assert n == 2

    def test_weight_value_matches_dequantized(self):
        pipeline = _make_mock_pipeline(n_mm=1, n_uni=0)
        w_int = np.array([[3, -2], [1, 7]], dtype=np.int8)
        scale = np.array([[0.25], [0.5]], dtype=np.float32)
        qw = {
            "mm0": {
                "image_transformer_block.attn.q_proj": {
                    "weight_int": w_int,
                    "scale": scale,
                    "a_scale": 1.0, "bits_w": 4, "bits_a": 8,
                }
            }
        }
        # Give the mock layer matching dimensions
        pipeline.mmdit.multimodal_transformer_blocks[0].image_transformer_block\
            .attn.q_proj = _MockLinear(out=2, inp=2)

        inject_weights(pipeline, qw)

        layer = pipeline.mmdit.multimodal_transformer_blocks[0]\
            .image_transformer_block.attn.q_proj
        w_after = np.array(layer.weight)
        expected = dequantize(w_int, scale).astype(np.float32)
        np.testing.assert_allclose(w_after, expected, atol=1e-4)

    def test_uninjected_block_weight_unchanged(self):
        pipeline = _make_mock_pipeline(n_mm=2, n_uni=0)
        orig_w = np.array(
            pipeline.mmdit.multimodal_transformer_blocks[1]
            .image_transformer_block.attn.q_proj.weight
        ).copy()

        # Only inject mm0
        qw = _simple_quant_weights("mm0", "image_transformer_block.attn.q_proj")
        inject_weights(pipeline, qw)

        after_w = np.array(
            pipeline.mmdit.multimodal_transformer_blocks[1]
            .image_transformer_block.attn.q_proj.weight
        )
        np.testing.assert_array_equal(orig_w, after_w)

    def test_unknown_block_name_skipped_gracefully(self):
        pipeline = _make_mock_pipeline(n_mm=1, n_uni=0)
        qw = _simple_quant_weights("mm99", "image_transformer_block.attn.q_proj")
        # Should not raise; returns 0
        n = inject_weights(pipeline, qw)
        assert n == 0

    def test_unknown_linear_path_skipped_gracefully(self):
        pipeline = _make_mock_pipeline(n_mm=1, n_uni=0)
        qw = _simple_quant_weights("mm0", "image_transformer_block.attn.no_such_layer")
        n = inject_weights(pipeline, qw)
        assert n == 0

    def test_uni_block_injected(self):
        pipeline = _make_mock_pipeline(n_mm=0, n_uni=1)
        pipeline.mmdit.unified_transformer_blocks[0].transformer_block\
            .attn.q_proj = _MockLinear(out=8, inp=8)

        w_int = np.full((8, 8), 2, dtype=np.int8)
        scale = np.full((8, 1), 1.0, dtype=np.float32)
        qw = {
            "uni0": {
                "transformer_block.attn.q_proj": {
                    "weight_int": w_int,
                    "scale": scale,
                    "a_scale": 1.0, "bits_w": 4, "bits_a": 8,
                }
            }
        }
        n = inject_weights(pipeline, qw)
        assert n == 1

        w_after = np.array(
            pipeline.mmdit.unified_transformer_blocks[0]
            .transformer_block.attn.q_proj.weight
        )
        np.testing.assert_allclose(w_after, np.full((8, 8), 2.0, dtype=np.float32),
                                   atol=1e-4)

    def test_empty_quant_weights_returns_zero(self):
        pipeline = _make_mock_pipeline()
        n = inject_weights(pipeline, {})
        assert n == 0


# ---------------------------------------------------------------------------
# TestWeightDiffStats
# ---------------------------------------------------------------------------

class TestWeightDiffStats:

    def test_returns_dict_with_expected_keys(self):
        pipeline = _make_mock_pipeline(n_mm=1, n_uni=0)
        # Use weights that differ from mock by a known amount
        layer = pipeline.mmdit.multimodal_transformer_blocks[0]\
            .image_transformer_block.attn.q_proj
        # Mock layer has shape (8, 8); make matching quant weights
        layer.weight = mx.array(np.zeros((8, 8), dtype=np.float32))
        w_int = np.ones((8, 8), dtype=np.int8)
        scale = np.ones((8, 1), dtype=np.float32)
        qw = {
            "mm0": {
                "image_transformer_block.attn.q_proj": {
                    "weight_int": w_int, "scale": scale,
                    "a_scale": 1.0, "bits_w": 4, "bits_a": 8,
                }
            }
        }
        stats = weight_diff_stats(pipeline, qw)
        assert "n_layers" in stats
        assert "mean_abs_diff" in stats
        assert "max_abs_diff" in stats

    def test_diff_is_zero_when_weights_match(self):
        pipeline = _make_mock_pipeline(n_mm=1, n_uni=0)
        # Set layer weight to exactly the dequantized value
        w_int = np.full((8, 8), 3, dtype=np.int8)
        scale = np.full((8, 1), 0.5, dtype=np.float32)
        expected_w = dequantize(w_int, scale).astype(np.float32)

        layer = pipeline.mmdit.multimodal_transformer_blocks[0]\
            .image_transformer_block.attn.q_proj
        layer.weight = mx.array(expected_w)

        qw = {
            "mm0": {
                "image_transformer_block.attn.q_proj": {
                    "weight_int": w_int, "scale": scale,
                    "a_scale": 1.0, "bits_w": 4, "bits_a": 8,
                }
            }
        }
        stats = weight_diff_stats(pipeline, qw)
        assert stats["mean_abs_diff"] == pytest.approx(0.0, abs=1e-5)

    def test_returns_empty_for_unknown_block(self):
        pipeline = _make_mock_pipeline(n_mm=1, n_uni=0)
        qw = _simple_quant_weights("mm99", "image_transformer_block.attn.q_proj")
        stats = weight_diff_stats(pipeline, qw)
        assert stats == {}


# ---------------------------------------------------------------------------
# TestPartialInjection (only a subset of blocks in output dir)
# ---------------------------------------------------------------------------

class TestPartialInjection:

    def test_only_present_blocks_are_injected(self, tmp_path):
        # Create output dir for mm0 only (not mm1)
        out_dir = _make_output_dir(tmp_path, block_names=("mm0",),
                                   linear_paths=("image_transformer_block.attn.q_proj",))
        _, qw = load_adaround_weights(out_dir)
        assert "mm0" in qw
        assert "mm1" not in qw

    def test_load_then_inject_subset(self, tmp_path):
        out_dir = _make_output_dir(
            tmp_path,
            block_names=("mm0",),
            linear_paths=("image_transformer_block.attn.q_proj",),
        )
        _, qw = load_adaround_weights(out_dir)
        pipeline = _make_mock_pipeline(n_mm=2, n_uni=2)

        # Give mm0 the right shape
        pipeline.mmdit.multimodal_transformer_blocks[0]\
            .image_transformer_block.attn.q_proj = _MockLinear(out=8, inp=8)

        n = inject_weights(pipeline, qw)
        assert n == 1

    def test_config_returns_metadata(self, tmp_path):
        out_dir = _make_output_dir(tmp_path, block_names=("mm0", "uni0"))
        config, _ = load_adaround_weights(out_dir)
        assert config["iters"] == 20000
        assert config["format"] == "adaround_v1"
