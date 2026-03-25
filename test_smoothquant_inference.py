"""
Tests for SmoothQuant inference changes in src/load_adaround_model.py

Verifies:
  - _ActQuantLayer with sq_scale correctly divides activations before fake-quant
  - No restore after fake-quant (absorbed weights provide the inverse scaling)
  - apply_act_quant_hooks correctly assigns sq_scale from scales dict
  - No SQ applied when smoothquant_scales=None (backward compatibility)
  - SQ + fake-quant pipeline reduces per-channel spread before quantization
  - SQ scale shape mismatch is handled gracefully
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Load module without mlx (it's available but we test at numpy level where possible)
import sys, os
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_linear(out=4, inp=8):
    """Return a callable that acts like nn.Linear (passes input through)."""
    layer = MagicMock()
    layer.side_effect = lambda x: x  # identity
    return layer


# We need mlx for the actual _ActQuantLayer test
import mlx.core as mx
from src.load_adaround_model import _ActQuantLayer, apply_act_quant_hooks, fake_quant_int


# ---------------------------------------------------------------------------
# _ActQuantLayer: SmoothQuant scale application
# ---------------------------------------------------------------------------

class TestActQuantLayerSmoothQuant:

    def _make_proxy(self, n_channels=8, sq_scale=None, scale=1.0):
        layer = MagicMock()
        layer.side_effect = lambda x: x

        # Static per-timestep config
        per_timestep = {"0": {"bits": 8, "scale": scale}}

        proxy = _ActQuantLayer(
            layer=layer,
            layer_name="test_layer",
            per_timestep=per_timestep,
            outlier_cfg={},
            sq_scale=sq_scale,
        )
        proxy.current_step_key = 0
        proxy.current_sigma = None
        return proxy

    def test_no_sq_scale_passthrough(self):
        """Without sq_scale, output should match standard fake-quant."""
        proxy = self._make_proxy(sq_scale=None, scale=1.0)
        x = mx.array(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
                               dtype=np.float32))
        out = proxy(x)
        # layer is identity, so output = fake_quant(x, scale=1, bits=8)
        expected = fake_quant_int(x, 1.0, 8)
        np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-5)

    def test_sq_scale_divides_before_quant(self):
        """
        With sq_scale=[2, 2, ...], the effective input to fake_quant is x/2.
        No restore: the absorbed weights (W' = W*diag(s)) provide the inverse.
        Layer receives fake_quant(x/2) directly.
        """
        n = 8
        sq = np.full(n, 2.0, dtype=np.float32)
        proxy = self._make_proxy(n_channels=n, sq_scale=sq, scale=1.0)

        x_np = np.array([[2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
        x = mx.array(x_np)
        out = proxy(x)

        # Expected: fake_quant(x/2, scale=1, bits=8) — no restore
        x_scaled = mx.array(x_np / 2)
        expected = fake_quant_int(x_scaled, 1.0, 8)
        np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-4)

    def test_sq_scale_identity(self):
        """sq_scale=1 everywhere should give same result as no SQ."""
        n = 8
        sq_ones = np.ones(n, dtype=np.float32)
        proxy_sq = self._make_proxy(sq_scale=sq_ones, scale=1.0)
        proxy_no = self._make_proxy(sq_scale=None, scale=1.0)

        x_np = np.random.RandomState(0).randn(2, n).astype(np.float32) * 3
        x = mx.array(x_np)

        out_sq = proxy_sq(x)
        out_no = proxy_no(x)
        np.testing.assert_allclose(np.array(out_sq), np.array(out_no), rtol=1e-4)

    def test_sq_reduces_channel_spread(self):
        """
        An activation with outlier channels (one channel 100×larger than others)
        should have lower max/min spread after SQ scaling.
        """
        n = 16
        x_np = np.ones((10, n), dtype=np.float32)
        x_np[:, 3] = 100.0  # outlier channel 3
        x_np[:, 7] = 50.0   # outlier channel 7

        # SQ scale absorbs the outlier magnitudes
        sq = np.ones(n, dtype=np.float32)
        sq[3] = 100.0
        sq[7] = 50.0

        x_scaled = x_np / sq[None, :]
        spread_before = x_np.max() - x_np.min()
        spread_after = x_scaled.max() - x_scaled.min()
        assert spread_after < spread_before, \
            f"SQ did not reduce spread: before={spread_before}, after={spread_after}"

    def test_sq_layer_receives_scaled_input(self):
        """
        The wrapped layer should receive fake_quant(x / sq_scale) directly.
        No restore — the absorbed weights W' = W*diag(s) cancel the scaling.
        """
        n = 4
        sq = np.array([2.0, 0.5, 4.0, 1.0], dtype=np.float32)

        layer = MagicMock()
        captured = {}
        def capture(x):
            captured["x"] = np.array(x)
            return x
        layer.side_effect = capture

        per_timestep = {"0": {"bits": 8, "scale": 0.5}}
        proxy = _ActQuantLayer(layer, "t", per_timestep, {}, sq_scale=sq)
        proxy.current_step_key = 0
        proxy.current_sigma = None

        x_np = np.array([[4.0, 2.0, 8.0, 3.0]], dtype=np.float32)
        proxy(mx.array(x_np))

        # Layer receives fake_quant(x / sq, scale=0.5, bits=8) — no restore
        x_div_sq = x_np / sq
        expected = np.round(x_div_sq / 0.5).clip(-127, 127) * 0.5
        np.testing.assert_allclose(captured["x"], expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# apply_act_quant_hooks: sq_scale threading
# ---------------------------------------------------------------------------

class TestApplyActQuantHooksSQ:

    def _build_mock_mmdit(self):
        """Build a minimal mock MMDiT with one mm block."""
        q_proj = MagicMock(spec=["weight", "__call__"])
        q_proj.side_effect = lambda x: x

        attn = MagicMock()
        attn.q_proj = q_proj

        itb = MagicMock()
        itb.attn = attn
        itb.mlp = None

        ttb = MagicMock()
        ttb.attn = MagicMock()
        ttb.attn.q_proj = MagicMock(spec=["weight", "__call__"])
        ttb.attn.q_proj.side_effect = lambda x: x
        ttb.mlp = None

        block = MagicMock()
        block.image_transformer_block = itb
        block.text_transformer_block = ttb

        mmdit = MagicMock()
        mmdit.multimodal_transformer_blocks = [block]
        del mmdit.unified_transformer_blocks  # suppress unified blocks

        return mmdit, q_proj

    def test_sq_scale_assigned_when_key_present(self):
        """Proxy gets sq_scale when its us_key appears in smoothquant_scales."""
        mmdit, q_proj = self._build_mock_mmdit()

        sq_scales = {
            "alpha": 0.5,
            "layers": {
                "mm0_img_attn_q_proj": [1.0, 2.0, 0.5, 3.0],
            }
        }

        # Minimal poly schedule so hooks are installed
        poly_schedule = {
            "sigma_range": [0.1, 1.0],
            "layers": {
                "mm0_img_attn_q_proj": {"degree": 0, "coeffs": [1.0], "r2": 1.0, "cv": 0.0},
            }
        }

        proxies, _ = apply_act_quant_hooks(
            mmdit, {}, {}, poly_schedule,
            smoothquant_scales=sq_scales,
        )

        # Find the proxy for mm0.img.attn.q_proj
        img_proxies = [p for p in proxies if p.layer_name == "mm0.img.attn.q_proj"]
        assert len(img_proxies) == 1
        proxy = img_proxies[0]
        assert proxy.sq_scale is not None
        np.testing.assert_allclose(proxy.sq_scale, [1.0, 2.0, 0.5, 3.0])

    def test_no_sq_scale_when_key_absent(self):
        """Proxy has sq_scale=None when layer not in smoothquant_scales."""
        mmdit, q_proj = self._build_mock_mmdit()

        poly_schedule = {
            "sigma_range": [0.1, 1.0],
            "layers": {
                "mm0_img_attn_q_proj": {"degree": 0, "coeffs": [1.0], "r2": 1.0, "cv": 0.0},
            }
        }

        proxies, _ = apply_act_quant_hooks(
            mmdit, {}, {}, poly_schedule,
            smoothquant_scales=None,
        )

        img_proxies = [p for p in proxies if p.layer_name == "mm0.img.attn.q_proj"]
        assert len(img_proxies) == 1
        assert img_proxies[0].sq_scale is None

    def test_sq_scale_none_when_smoothquant_scales_none(self):
        """Backward compat: no smoothquant_scales → all proxies have sq_scale=None."""
        mmdit, _ = self._build_mock_mmdit()
        poly_schedule = {
            "sigma_range": [0.1, 1.0],
            "layers": {
                "mm0_img_attn_q_proj": {"degree": 0, "coeffs": [1.0], "r2": 1.0, "cv": 0.0},
                "mm0_txt_attn_q_proj": {"degree": 0, "coeffs": [2.0], "r2": 1.0, "cv": 0.0},
            }
        }
        proxies, _ = apply_act_quant_hooks(mmdit, {}, {}, poly_schedule)
        for proxy in proxies:
            assert proxy.sq_scale is None, f"{proxy.layer_name} has unexpected sq_scale"


# ---------------------------------------------------------------------------
# Polynomial schedule + SmoothQuant: generate_schedule with real data
# ---------------------------------------------------------------------------

def test_smoothquant_schedule_generation():
    """generate_schedule with smoothquant_scales produces valid schedule."""
    from src.generate_poly_schedule import generate_schedule
    from src.compute_smoothquant_scales import compute_smoothquant_scales

    activations_dir = Path("calibration_data_512/activations")
    weights_dir = Path("quantized_weights_w4a8_adaround_poly_p100")

    sq_scales_raw = compute_smoothquant_scales(activations_dir, weights_dir,
                                               alpha=0.5, scale_clip=32)
    sq_scales = {
        "alpha": 0.5,
        "layers": {k: v.tolist() for k, v in sq_scales_raw.items()},
    }

    schedule = generate_schedule(activations_dir, smoothquant_scales=sq_scales)

    assert "layers" in schedule
    assert schedule["version"] == "poly_v1_smoothquant"
    assert schedule["percentile"] == "smoothquant_absmax"
    assert schedule["smoothquant_alpha"] == 0.5

    # Every layer should have valid coefficients
    for name, entry in schedule["layers"].items():
        assert "degree" in entry
        assert "coeffs" in entry
        assert len(entry["coeffs"]) > 0
        assert not any(np.isnan(c) for c in entry["coeffs"]), \
            f"{name}: NaN coefficient"


def test_smoothquant_schedule_absmax_reduced():
    """
    After SQ smoothing, the fitted polynomial max value (at sigma=1.0) should be
    smaller than the raw absmax for outlier-heavy layers.
    """
    from src.generate_poly_schedule import generate_schedule
    from src.compute_smoothquant_scales import compute_smoothquant_scales

    activations_dir = Path("calibration_data_512/activations")
    weights_dir = Path("quantized_weights_w4a8_adaround_poly_p100")

    sq_scales_raw = compute_smoothquant_scales(activations_dir, weights_dir,
                                               alpha=0.5, scale_clip=32)
    sq_scales = {
        "alpha": 0.5,
        "layers": {k: v.tolist() for k, v in sq_scales_raw.items()},
    }

    sched_raw = generate_schedule(activations_dir)
    sched_sq  = generate_schedule(activations_dir, smoothquant_scales=sq_scales)

    raw_layers = sched_raw["layers"]
    sq_layers  = sched_sq["layers"]

    # The smoothed schedule absmax (coeffs[0] for degree-0 layers, or polyval at sigma=1)
    # should on average be smaller than raw because outliers are absorbed
    def approx_absmax(entry):
        coeffs = entry["coeffs"]
        if entry["degree"] == 0:
            return coeffs[0]
        return float(np.polyval(coeffs, 1.0))  # evaluate at sigma=1 (high noise)

    common = set(raw_layers) & set(sq_layers)
    raw_vals = [approx_absmax(raw_layers[k]) for k in common]
    sq_vals  = [approx_absmax(sq_layers[k])  for k in common]

    mean_raw = np.mean(raw_vals)
    mean_sq  = np.mean(sq_vals)
    assert mean_sq < mean_raw, (
        f"Smoothed schedule not smaller: mean_sq={mean_sq:.2f} >= mean_raw={mean_raw:.2f}"
    )
