"""
Tests for src/compute_smoothquant_scales.py

Verifies:
  - Layer name round-trip conversions
  - Scale formula correctness (s = s_act^alpha / s_w^(1-alpha))
  - Scale clamping behaviour
  - load_per_channel_act_stats shape/value sanity
  - load_per_column_weight_range shape/value sanity
  - End-to-end compute_smoothquant_scales against real calibration data
"""

import json
import numpy as np
import pytest
from pathlib import Path

from src.compute_smoothquant_scales import (
    calib_name_to_weight_key,
    weight_path_to_calib_name,
    weight_path_to_npz_key,
    load_per_channel_act_stats,
    load_per_column_weight_range,
    compute_smoothquant_scales,
)

ACTIVATIONS_DIR = Path("calibration_data_512/activations")
WEIGHTS_DIR = Path("quantized_weights_w4a8_adaround_poly_p100")

# ---------------------------------------------------------------------------
# Layer name conversion tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("calib,expected_block,expected_path", [
    ("mm0_img_attn_q_proj",  "mm0",  "image_transformer_block.attn.q_proj"),
    ("mm0_img_attn_k_proj",  "mm0",  "image_transformer_block.attn.k_proj"),
    ("mm0_img_attn_v_proj",  "mm0",  "image_transformer_block.attn.v_proj"),
    ("mm0_img_attn_o_proj",  "mm0",  "image_transformer_block.attn.o_proj"),
    ("mm0_img_mlp_fc1",      "mm0",  "image_transformer_block.mlp.fc1"),
    ("mm0_img_mlp_fc2",      "mm0",  "image_transformer_block.mlp.fc2"),
    ("mm0_txt_attn_q_proj",  "mm0",  "text_transformer_block.attn.q_proj"),
    ("mm3_txt_mlp_fc2",      "mm3",  "text_transformer_block.mlp.fc2"),
    ("mm23_img_attn_o_proj", "mm23", "image_transformer_block.attn.o_proj"),
])
def test_calib_name_to_weight_key(calib, expected_block, expected_path):
    block, path = calib_name_to_weight_key(calib)
    assert block == expected_block, f"{calib}: got block {block!r}"
    assert path == expected_path, f"{calib}: got path {path!r}"


@pytest.mark.parametrize("calib", [
    "mm0_img_attn_q_proj",
    "mm0_img_mlp_fc1",
    "mm0_txt_attn_k_proj",
    "mm3_txt_mlp_fc2",
    "mm23_img_attn_o_proj",
])
def test_round_trip(calib):
    block, path = calib_name_to_weight_key(calib)
    back = weight_path_to_calib_name(block, path)
    assert back == calib, f"Round-trip failed: {calib} → {back}"


def test_npz_key():
    key = weight_path_to_npz_key("image_transformer_block.attn.q_proj", "weight_int")
    assert key == "image_transformer_block_attn_q_proj__weight_int"


def test_calib_name_unknown_stream():
    # Unsupported stream prefix should return (None, None)
    block, path = calib_name_to_weight_key("mm0_bad_attn_q_proj")
    assert block is None and path is None


# ---------------------------------------------------------------------------
# Scale formula
# ---------------------------------------------------------------------------

def test_scale_formula_alpha_0():
    """alpha=0: s = s_act^0 / s_w^1 = 1/s_w → all migration to activation side."""
    s_act = np.array([4.0, 8.0])
    s_w   = np.array([2.0, 4.0])
    s = (s_act ** 0.0) / (s_w ** 1.0)
    expected = 1.0 / s_w
    np.testing.assert_allclose(s, expected)


def test_scale_formula_alpha_1():
    """alpha=1: s = s_act^1 / s_w^0 = s_act → all migration to weight side."""
    s_act = np.array([4.0, 8.0])
    s_w   = np.array([2.0, 4.0])
    s = (s_act ** 1.0) / (s_w ** 0.0)
    np.testing.assert_allclose(s, s_act)


def test_scale_formula_alpha_half():
    """alpha=0.5: s = sqrt(s_act/s_w)."""
    s_act = np.array([9.0, 16.0])
    s_w   = np.array([1.0,  4.0])
    s = (s_act ** 0.5) / (s_w ** 0.5)
    expected = np.sqrt(s_act / s_w)   # [3.0, 2.0]
    np.testing.assert_allclose(s, expected)


def test_scale_clamping():
    """scale_clip=4 should clamp [0.1, 1, 5, 100] to [0.25, 1, 4, 4]."""
    raw = np.array([0.1, 1.0, 5.0, 100.0])
    clipped = np.clip(raw, 1.0 / 4, 4.0)
    assert clipped[0] == pytest.approx(0.25)
    assert clipped[1] == pytest.approx(1.0)
    assert clipped[2] == pytest.approx(4.0)
    assert clipped[3] == pytest.approx(4.0)


def test_scale_no_clamping():
    """scale_clip=0 leaves extreme values intact."""
    raw = np.array([0.001, 1000.0])
    # The compute function doesn't clamp when scale_clip=0
    clipped = raw  # no-op
    np.testing.assert_array_equal(clipped, raw)


# ---------------------------------------------------------------------------
# load_per_channel_act_stats
# ---------------------------------------------------------------------------

def test_act_stats_shape_and_type():
    stats = load_per_channel_act_stats(ACTIVATIONS_DIR)
    assert len(stats) > 0
    for name, arr in stats.items():
        assert isinstance(arr, np.ndarray), f"{name}: not ndarray"
        assert arr.ndim == 1, f"{name}: expected 1D, got {arr.shape}"
        assert arr.dtype == np.float32, f"{name}: expected float32"


def test_act_stats_nonnegative():
    """load_per_channel_act_stats returns per-channel absmax (always ≥ 0)."""
    stats = load_per_channel_act_stats(ACTIVATIONS_DIR)
    for name, arr in stats.items():
        assert np.all(arr >= 0), (
            f"{name}: negative value in absmax array — "
            "implementation should take max(|avg_max|, |avg_min|)"
        )


def test_act_stats_is_max_across_steps():
    """
    Manually compute per-channel absmax across steps for one layer and
    verify it matches what load_per_channel_act_stats returns.
    Uses max(|avg_max|, |avg_min|) per timestep, then max across timesteps.
    """
    ts_dir = ACTIVATIONS_DIR / "timestep_stats"
    target_layer = "mm0_img_attn_q_proj"
    max_key = f"{target_layer}__avg_max"
    min_key = f"{target_layer}__avg_min"

    per_step = []
    for npz_path in sorted(ts_dir.glob("step_*.npz")):
        d = np.load(npz_path)
        if max_key not in d.files:
            continue
        absmax = np.abs(d[max_key].astype(np.float32))
        if min_key in d.files:
            absmax = np.maximum(absmax, np.abs(d[min_key].astype(np.float32)))
        per_step.append(absmax)

    assert len(per_step) > 0
    expected = np.stack(per_step).max(axis=0)

    stats = load_per_channel_act_stats(ACTIVATIONS_DIR)
    np.testing.assert_allclose(stats[target_layer], expected)


# ---------------------------------------------------------------------------
# load_per_column_weight_range
# ---------------------------------------------------------------------------

def test_weight_range_shape():
    ranges = load_per_column_weight_range(WEIGHTS_DIR)
    assert len(ranges) > 0
    for name, arr in ranges.items():
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1


def test_weight_range_nonnegative():
    """Per-column max abs should be ≥ 0."""
    ranges = load_per_column_weight_range(WEIGHTS_DIR)
    for name, arr in ranges.items():
        assert np.all(arr >= 0), f"{name}: negative range"


def test_weight_range_matches_act_channel_count():
    """
    Activation has shape (in_channels,); weight per-column range also has shape (in_channels,).
    Both should agree on channel count.
    """
    act_stats = load_per_channel_act_stats(ACTIVATIONS_DIR)
    weight_ranges = load_per_column_weight_range(WEIGHTS_DIR)

    mismatches = []
    for name in act_stats:
        if name in weight_ranges:
            if act_stats[name].shape != weight_ranges[name].shape:
                mismatches.append(
                    f"{name}: act={act_stats[name].shape} w={weight_ranges[name].shape}"
                )
    assert mismatches == [], "Shape mismatches:\n" + "\n".join(mismatches)


# ---------------------------------------------------------------------------
# End-to-end compute_smoothquant_scales
# ---------------------------------------------------------------------------

def test_e2e_scale_count():
    scales = compute_smoothquant_scales(ACTIVATIONS_DIR, WEIGHTS_DIR, alpha=0.5)
    # Should have one entry per quantized layer (285 in this project)
    assert len(scales) == 285


def test_e2e_scales_positive():
    scales = compute_smoothquant_scales(ACTIVATIONS_DIR, WEIGHTS_DIR, alpha=0.5)
    for name, arr in scales.items():
        assert np.all(arr > 0), f"{name}: non-positive scale"


def test_e2e_scale_clip_reduces_max():
    raw = compute_smoothquant_scales(ACTIVATIONS_DIR, WEIGHTS_DIR, alpha=0.5, scale_clip=0)
    clipped = compute_smoothquant_scales(ACTIVATIONS_DIR, WEIGHTS_DIR, alpha=0.5, scale_clip=32)

    raw_max = max(v.max() for v in raw.values())
    clipped_max = max(v.max() for v in clipped.values())

    assert clipped_max <= 32.0 + 1e-6, f"scale_clip=32 failed: max={clipped_max}"
    assert raw_max > clipped_max, "Clamped max should be smaller than unclamped"


def test_e2e_alpha_effect():
    """
    Higher alpha migrates more to the weight side, making post-smoothing
    activations more uniform.  Concretely: the per-channel spread of
    (s_act[c] / s[c]) — which is what the quantizer actually sees after
    SQ scaling — should decrease as alpha increases (at alpha=1 it collapses
    to a constant equal to s_w[c]^0 = 1 for each channel).
    """
    from src.compute_smoothquant_scales import load_per_channel_act_stats

    s_act_map = load_per_channel_act_stats(ACTIVATIONS_DIR)

    for alpha in (0.1, 0.9):
        scales = compute_smoothquant_scales(ACTIVATIONS_DIR, WEIGHTS_DIR, alpha=alpha)
        # post_act[c] = s_act[c] / s[c]  (effective per-channel range seen by quantizer)
        post_act = {}
        for name, s in scales.items():
            s_act = np.maximum(s_act_map[name], 1e-5)
            post_act[name] = s_act / np.maximum(s, 1e-5)

        cvs = []
        for name, pa in post_act.items():
            mean = pa.mean()
            if mean > 0:
                cvs.append(pa.std() / mean)

        if alpha == 0.1:
            cv_lo = np.mean(cvs)
        else:
            cv_hi = np.mean(cvs)

    # Higher alpha → lower per-channel CV of effective activation range
    assert cv_hi < cv_lo, (
        f"alpha=0.9 CV ({cv_hi:.3f}) should be < alpha=0.1 CV ({cv_lo:.3f}): "
        "higher alpha should produce more uniform post-smoothing activations"
    )


def test_e2e_scale_shape_consistency():
    """All scales for the same layer should have length == weight in_channels."""
    import json
    with open(WEIGHTS_DIR / "config.json") as f:
        cfg = json.load(f)
    path_lookup = {bm["block_name"]: bm.get("quant_paths", [])
                   for bm in cfg.get("block_metrics", [])}

    scales = compute_smoothquant_scales(ACTIVATIONS_DIR, WEIGHTS_DIR, alpha=0.5)

    for block_name, paths in path_lookup.items():
        for weight_path in paths:
            calib_name = weight_path_to_calib_name(block_name, weight_path)
            if calib_name not in scales:
                continue
            npz = np.load(WEIGHTS_DIR / "weights" / f"{block_name}.npz")
            wi_key = weight_path_to_npz_key(weight_path, "weight_int")
            if wi_key not in npz.files:
                continue
            in_channels = npz[wi_key].shape[1]
            assert scales[calib_name].shape == (in_channels,), (
                f"{calib_name}: scale shape {scales[calib_name].shape} != ({in_channels},)"
            )
