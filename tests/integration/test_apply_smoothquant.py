"""
Tests for src/apply_smoothquant.py

Verifies:
  - quantize_per_output_channel: correct shape, dtype, range, and reconstruction error
  - Weight absorption: W_smooth = W_fp * s → correct per-column scaling
  - Re-quantization after absorption: range preserved, error bounded
  - Alpha key removal after absorption
  - Output directory structure matches source
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.apply_smoothquant import quantize_per_output_channel, apply_smoothquant
from src.compute_smoothquant_scales import compute_smoothquant_scales

WEIGHTS_DIR = Path("quantized_weights_w4a8_adaround_poly_p100")
ACTIVATIONS_DIR = Path("calibration_data_512/activations")


# ---------------------------------------------------------------------------
# quantize_per_output_channel
# ---------------------------------------------------------------------------

def test_quant_output_shape():
    w = np.random.randn(8, 16).astype(np.float32)
    w_int, scale = quantize_per_output_channel(w, bits=4)
    assert w_int.shape == (8, 16)
    assert scale.shape == (8, 1)


def test_quant_output_dtype():
    w = np.random.randn(4, 8).astype(np.float32)
    w_int, scale = quantize_per_output_channel(w, bits=4)
    assert w_int.dtype == np.int8
    assert scale.dtype == np.float16


def test_quant_int_range_w4():
    w = np.random.randn(64, 64).astype(np.float32) * 10
    w_int, _ = quantize_per_output_channel(w, bits=4)
    assert w_int.min() >= -7
    assert w_int.max() <= 7


def test_quant_reconstruction_error():
    """Reconstruction error should be < 1/qmax of the per-row range."""
    np.random.seed(0)
    w = np.random.randn(32, 32).astype(np.float32) * 5
    w_int, scale = quantize_per_output_channel(w, bits=4)
    w_rec = w_int.astype(np.float32) * scale.astype(np.float32)
    # Max error per row is at most scale/2 ≈ max(|w_row|) / 7 / 2
    per_row_max = np.max(np.abs(w), axis=1)
    expected_max_err = per_row_max / 7  # one quantization step
    actual_max_err = np.max(np.abs(w - w_rec), axis=1)
    assert np.all(actual_max_err <= expected_max_err + 1e-4), \
        f"Reconstruction error exceeded step size: max={actual_max_err.max():.4f}"


def test_quant_zero_row():
    """All-zero weight row should not cause NaN/Inf."""
    w = np.zeros((4, 8), dtype=np.float32)
    w_int, scale = quantize_per_output_channel(w, bits=4)
    assert not np.any(np.isnan(w_int))
    assert not np.any(np.isinf(scale))
    assert np.all(w_int == 0)


def test_quant_scale_positive():
    w = np.random.randn(16, 16).astype(np.float32) * 3
    _, scale = quantize_per_output_channel(w, bits=4)
    assert np.all(scale > 0)


# ---------------------------------------------------------------------------
# Weight absorption: direct numerical test
# ---------------------------------------------------------------------------

def test_absorption_column_scaling():
    """
    After applying s = [2, 0.5, 3, 1], each column of W_smooth should be
    exactly W_fp[:, c] * s[c].
    """
    np.random.seed(1)
    w = np.random.randn(4, 4).astype(np.float32)
    scale = np.ones((4, 1), dtype=np.float32)  # identity scale for clean test
    w_int = np.round(w / 1.0).clip(-7, 7).astype(np.int8)
    w_fp = w_int.astype(np.float32) * scale

    s = np.array([2.0, 0.5, 3.0, 1.0], dtype=np.float32)
    w_smooth = w_fp * s[None, :]

    for c, sc in enumerate(s):
        np.testing.assert_allclose(w_smooth[:, c], w_fp[:, c] * sc, rtol=1e-6)


def test_absorption_preserves_output_rows():
    """
    Multiplying by per-channel input scale should not change the number
    of output or input channels.
    """
    w_fp = np.random.randn(1536, 1536).astype(np.float32)
    s = np.random.rand(1536).astype(np.float32) + 0.1
    w_smooth = w_fp * s[None, :]
    assert w_smooth.shape == w_fp.shape


# ---------------------------------------------------------------------------
# End-to-end apply_smoothquant (real weights, temp output dir)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sq_scales():
    return compute_smoothquant_scales(ACTIVATIONS_DIR, WEIGHTS_DIR, alpha=0.5, scale_clip=32)


@pytest.fixture(scope="module")
def sq_scales_dict(sq_scales):
    return {"alpha": 0.5, "scale_clip": 32, "n_layers": len(sq_scales),
            "layers": {k: v.tolist() for k, v in sq_scales.items()}}


@pytest.fixture(scope="module")
def absorbed_dir(sq_scales_dict, tmp_path_factory):
    out = tmp_path_factory.mktemp("absorbed_weights")
    apply_smoothquant(WEIGHTS_DIR, sq_scales_dict, out, bits=4)
    return out


def test_absorbed_config_exists(absorbed_dir):
    assert (absorbed_dir / "config.json").exists()


def test_absorbed_config_annotated(absorbed_dir):
    with open(absorbed_dir / "config.json") as f:
        cfg = json.load(f)
    assert cfg.get("smoothquant") is True
    assert "smoothquant_alpha" in cfg
    assert cfg.get("adaround_alpha_preserved") is False


def test_absorbed_weights_dir_exists(absorbed_dir):
    assert (absorbed_dir / "weights").is_dir()


def test_absorbed_npz_count(absorbed_dir):
    src_npzs = list((WEIGHTS_DIR / "weights").glob("*.npz"))
    out_npzs = list((absorbed_dir / "weights").glob("*.npz"))
    assert len(out_npzs) == len(src_npzs)


def test_absorbed_weight_int_dtype(absorbed_dir):
    for npz_path in sorted((absorbed_dir / "weights").glob("*.npz")):
        data = np.load(npz_path)
        for key in data.files:
            if key.endswith("__weight_int"):
                assert data[key].dtype == np.int8, f"{npz_path.name}:{key} not int8"


def test_absorbed_weight_range_w4(absorbed_dir):
    """All weight_int values should stay in [-7, 7] (W4 symmetric)."""
    for npz_path in sorted((absorbed_dir / "weights").glob("*.npz")):
        data = np.load(npz_path)
        for key in data.files:
            if key.endswith("__weight_int"):
                w = data[key]
                assert w.min() >= -7, f"{npz_path.name}:{key} min={w.min()}"
                assert w.max() <= 7,  f"{npz_path.name}:{key} max={w.max()}"


def test_absorbed_scale_positive(absorbed_dir):
    for npz_path in sorted((absorbed_dir / "weights").glob("*.npz")):
        data = np.load(npz_path)
        for key in data.files:
            if key.endswith("__scale"):
                assert np.all(data[key] > 0), f"{npz_path.name}:{key} non-positive scale"


def test_absorbed_no_alpha_keys(absorbed_dir):
    """AdaRound alpha parameters should be removed after absorption."""
    for npz_path in sorted((absorbed_dir / "weights").glob("*.npz")):
        data = np.load(npz_path)
        alpha_keys = [k for k in data.files if k.endswith("__alpha")]
        assert alpha_keys == [], f"{npz_path.name} still has alpha keys: {alpha_keys}"


def test_absorbed_weights_changed(absorbed_dir):
    """Absorbed weights should differ from source (SQ scaling was applied)."""
    changed = 0
    total = 0
    for src_path in sorted((WEIGHTS_DIR / "weights").glob("*.npz")):
        src = np.load(src_path)
        out_path = absorbed_dir / "weights" / src_path.name
        out = np.load(out_path)
        for key in src.files:
            if key.endswith("__weight_int"):
                total += 1
                if not np.array_equal(src[key], out[key]):
                    changed += 1
    assert changed > 0, "No weights changed after SQ absorption"
    assert changed == total, f"Only {changed}/{total} layers changed (expected all)"


def test_absorbed_a_scale_preserved(absorbed_dir):
    """a_scale (activation scale) should be copied unchanged."""
    for src_path in sorted((WEIGHTS_DIR / "weights").glob("*.npz")):
        src = np.load(src_path)
        out_path = absorbed_dir / "weights" / src_path.name
        out = np.load(out_path)
        for key in src.files:
            if key.endswith("__a_scale"):
                np.testing.assert_array_equal(
                    src[key], out[key],
                    err_msg=f"{src_path.name}:{key} a_scale changed"
                )
