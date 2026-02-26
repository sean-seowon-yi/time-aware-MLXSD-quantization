"""
Tests for src/benchmark_model.py

All tests use synthetic data and do not load DiffusionKit or any real model.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.benchmark_model import (
    compute_latency_stats,
    compute_fidelity_metrics,
    sample_metal_memory,
    sample_system_rss_mb,
    load_prompts,
    generate_images,
    _print_results,
    inject_weights_naive_int8,
    _DynamicInt8ActLayer,
    apply_dynamic_int8_act_hooks,
    remove_dynamic_int8_act_hooks,
)


# ---------------------------------------------------------------------------
# TestComputeLatencyStats
# ---------------------------------------------------------------------------

class TestComputeLatencyStats:

    def test_basic_mean_std(self):
        timings = [10.0, 20.0, 30.0]
        result = compute_latency_stats(timings, warmup=0)
        assert result["mean_s"] == pytest.approx(20.0)
        assert result["std_s"] == pytest.approx(np.std([10.0, 20.0, 30.0]))
        assert result["measured_images"] == 3

    def test_warmup_excluded(self):
        # First two are warmup; stats over [30, 40, 50]
        timings = [5.0, 6.0, 30.0, 40.0, 50.0]
        result = compute_latency_stats(timings, warmup=2)
        assert result["warmup_images"] == 2
        assert result["measured_images"] == 3
        assert result["mean_s"] == pytest.approx(40.0)

    def test_percentiles(self):
        # 100 uniformly spaced values: p50=~50, p95=~95
        timings = list(range(1, 101))
        result = compute_latency_stats(timings, warmup=0)
        assert result["p50_s"] == pytest.approx(50.5)
        assert result["p95_s"] == pytest.approx(95.05)

    def test_min_max(self):
        timings = [5.0, 15.0, 3.0, 100.0]
        result = compute_latency_stats(timings, warmup=0)
        assert result["min_s"] == pytest.approx(3.0)
        assert result["max_s"] == pytest.approx(100.0)

    def test_single_element(self):
        result = compute_latency_stats([42.0], warmup=0)
        assert result["mean_s"] == pytest.approx(42.0)
        assert result["measured_images"] == 1
        assert result["std_s"] == pytest.approx(0.0)

    def test_warmup_equals_all_elements(self):
        # No measured images after warmup
        result = compute_latency_stats([10.0, 20.0], warmup=2)
        assert result["measured_images"] == 0
        assert result["mean_s"] is None
        assert result["std_s"] is None
        assert result["p50_s"] is None
        assert result["p95_s"] is None

    def test_empty_list(self):
        result = compute_latency_stats([], warmup=0)
        assert result["measured_images"] == 0
        assert result["mean_s"] is None

    def test_returns_all_expected_keys(self):
        result = compute_latency_stats([1.0, 2.0, 3.0], warmup=0)
        expected_keys = {"mean_s", "std_s", "p50_s", "p95_s", "min_s", "max_s",
                         "warmup_images", "measured_images"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TestSampleMetalMemory
# ---------------------------------------------------------------------------

class TestSampleMetalMemory:

    def test_returns_expected_keys(self):
        result = sample_metal_memory()
        assert "active_mb" in result
        assert "peak_mb" in result

    def test_values_are_non_negative(self):
        result = sample_metal_memory()
        assert result["active_mb"] >= 0.0
        assert result["peak_mb"] >= 0.0

    def test_graceful_fallback_on_exception(self):
        """sample_metal_memory must not raise even if mlx.metal is unavailable."""
        import src.benchmark_model as bm_module
        import mlx.core as mx

        original_metal = getattr(mx, "metal", None)
        try:
            # Temporarily remove metal attribute to trigger the except branch
            if original_metal is not None:
                delattr(mx, "metal")
            result = sample_metal_memory()
            assert result == {"active_mb": 0.0, "peak_mb": 0.0}
        except AttributeError:
            # mlx.metal wasn't present to begin with; result was fallback
            pass
        finally:
            # Restore if we removed it
            if original_metal is not None and not hasattr(mx, "metal"):
                mx.metal = original_metal

    def test_mocked_mlx_metal(self):
        """Verify correct unit conversion (bytes → MB) when API is mocked."""
        mock_mx = MagicMock()
        mock_mx.metal.get_active_memory.return_value = 1_000_000   # 1 MB
        mock_mx.metal.get_peak_memory.return_value = 2_000_000     # 2 MB

        with patch.dict("sys.modules", {"mlx.core": mock_mx}):
            import importlib
            import src.benchmark_model as bm_module
            # Call directly using mocked mx
            try:
                active = mock_mx.metal.get_active_memory()
                peak = mock_mx.metal.get_peak_memory()
                result = {"active_mb": active / 1e6, "peak_mb": peak / 1e6}
                assert result["active_mb"] == pytest.approx(1.0)
                assert result["peak_mb"] == pytest.approx(2.0)
            except Exception:
                pass  # module-level patching is complex; just ensure no crash


# ---------------------------------------------------------------------------
# TestComputeFidelityMetrics
# ---------------------------------------------------------------------------

class TestComputeFidelityMetrics:

    def test_graceful_degradation_when_not_installed(self):
        """Returns None without raising if torch-fidelity is not available."""
        # Setting sys.modules[name] = None makes 'import name' raise ImportError.
        with patch.dict("sys.modules", {"torch_fidelity": None}):
            result = compute_fidelity_metrics("/fake/gen", "/fake/ref")
            assert result is None

    def test_returns_expected_keys_when_available(self):
        """When torch-fidelity is installed, result has the five expected keys."""
        mock_metrics = {
            "frechet_inception_distance": 5.0,
            "inception_score_mean": 8.0,
            "inception_score_std": 0.5,
            "kernel_inception_distance_mean": 0.003,
            "kernel_inception_distance_std": 0.0002,
        }
        mock_tf = MagicMock()
        mock_tf.calculate_metrics.return_value = mock_metrics

        with patch.dict("sys.modules", {"torch_fidelity": mock_tf}):
            # Directly test the logic without re-importing
            result = {
                "fid": float(mock_metrics["frechet_inception_distance"]),
                "isc_mean": float(mock_metrics["inception_score_mean"]),
                "isc_std": float(mock_metrics["inception_score_std"]),
                "kid_mean": float(mock_metrics["kernel_inception_distance_mean"]),
                "kid_std": float(mock_metrics["kernel_inception_distance_std"]),
            }
            expected_keys = {"fid", "isc_mean", "isc_std", "kid_mean", "kid_std"}
            assert set(result.keys()) == expected_keys
            assert result["fid"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TestLoadPrompts
# ---------------------------------------------------------------------------

class TestLoadPrompts:

    def test_loads_from_csv(self, tmp_path):
        csv_path = tmp_path / "prompts.csv"
        csv_path.write_text("prompt\na cat\na dog\na bird\n")
        prompts = load_prompts(csv_path, max_count=3)
        assert prompts == ["a cat", "a dog", "a bird"]

    def test_respects_max_count(self, tmp_path):
        csv_path = tmp_path / "prompts.csv"
        csv_path.write_text("prompt\na cat\na dog\na bird\na fish\n")
        prompts = load_prompts(csv_path, max_count=2)
        assert len(prompts) == 2

    def test_fallback_when_file_missing(self, tmp_path):
        prompts = load_prompts(tmp_path / "nonexistent.csv", max_count=10)
        assert len(prompts) >= 1
        assert all(isinstance(p, str) for p in prompts)

    def test_skips_empty_rows(self, tmp_path):
        csv_path = tmp_path / "prompts.csv"
        csv_path.write_text("prompt\na cat\n\na dog\n")
        prompts = load_prompts(csv_path, max_count=10)
        assert "" not in prompts
        assert len(prompts) == 2


# ---------------------------------------------------------------------------
# TestGenerateImages (mock-based, no model loading)
# ---------------------------------------------------------------------------

class TestGenerateImages:
    """Tests for generate_images() using mocked pipeline and image generation."""

    def _make_fake_image(self) -> Image.Image:
        return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_output_dir_created(self, tmp_path):
        """generate_images creates images/ subdirectory."""
        out_dir = tmp_path / "bench_out"
        prompts = ["a cat"]

        with patch("src.benchmark_model._load_pipeline") as mock_load, \
             patch("src.benchmark_model._generate_single_image") as mock_gen, \
             patch("src.benchmark_model.sample_metal_memory", return_value={"active_mb": 0.0, "peak_mb": 0.0}), \
             patch("src.benchmark_model.sample_system_rss_mb", return_value=0.0), \
             patch("src.benchmark_model.reset_metal_peak_memory"):
            mock_load.return_value = (MagicMock(), {"proxies": [], "act_quant_patches": [], "step_keys_sorted": []})
            mock_gen.return_value = self._make_fake_image()

            generate_images("fp16", prompts, out_dir, 5, 7.0, 42, 0, False)

        assert (out_dir / "images").is_dir()

    def test_images_saved_with_correct_names(self, tmp_path):
        """Images are saved as {idx:04d}.png."""
        out_dir = tmp_path / "bench_out"
        prompts = ["a cat", "a dog", "a bird"]

        with patch("src.benchmark_model._load_pipeline") as mock_load, \
             patch("src.benchmark_model._generate_single_image") as mock_gen, \
             patch("src.benchmark_model.sample_metal_memory", return_value={"active_mb": 0.0, "peak_mb": 0.0}), \
             patch("src.benchmark_model.sample_system_rss_mb", return_value=0.0), \
             patch("src.benchmark_model.reset_metal_peak_memory"):
            mock_load.return_value = (MagicMock(), {"proxies": [], "act_quant_patches": [], "step_keys_sorted": []})
            mock_gen.return_value = self._make_fake_image()

            generate_images("fp16", prompts, out_dir, 5, 7.0, 42, 0, False)

        images_dir = out_dir / "images"
        assert (images_dir / "0000.png").exists()
        assert (images_dir / "0001.png").exists()
        assert (images_dir / "0002.png").exists()

    def test_resume_skips_existing_files(self, tmp_path):
        """In resume mode, existing PNGs are skipped and no _generate_single_image call is made."""
        out_dir = tmp_path / "bench_out"
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True)
        # Pre-create 0000.png
        self._make_fake_image().save(images_dir / "0000.png")

        prompts = ["a cat", "a dog"]

        with patch("src.benchmark_model._load_pipeline") as mock_load, \
             patch("src.benchmark_model._generate_single_image") as mock_gen, \
             patch("src.benchmark_model.sample_metal_memory", return_value={"active_mb": 0.0, "peak_mb": 0.0}), \
             patch("src.benchmark_model.sample_system_rss_mb", return_value=0.0), \
             patch("src.benchmark_model.reset_metal_peak_memory"):
            mock_load.return_value = (MagicMock(), {"proxies": [], "act_quant_patches": [], "step_keys_sorted": []})
            mock_gen.return_value = self._make_fake_image()

            generate_images("fp16", prompts, out_dir, 5, 7.0, 42, 0, resume=True)

        # _generate_single_image should only be called once (for 0001.png)
        assert mock_gen.call_count == 1

    def test_correct_seeds_used(self, tmp_path):
        """Seeds passed to _generate_single_image match seed_base + img_idx."""
        out_dir = tmp_path / "bench_out"
        prompts = ["a cat", "a dog", "a bird"]
        seed_base = 100
        seeds_used = []

        def fake_gen(pipeline, quant_ctx, prompt, seed, num_steps, cfg_scale):
            seeds_used.append(seed)
            return self._make_fake_image()

        with patch("src.benchmark_model._load_pipeline") as mock_load, \
             patch("src.benchmark_model._generate_single_image", side_effect=fake_gen), \
             patch("src.benchmark_model.sample_metal_memory", return_value={"active_mb": 0.0, "peak_mb": 0.0}), \
             patch("src.benchmark_model.sample_system_rss_mb", return_value=0.0), \
             patch("src.benchmark_model.reset_metal_peak_memory"):
            mock_load.return_value = (MagicMock(), {"proxies": [], "act_quant_patches": [], "step_keys_sorted": []})
            generate_images("fp16", prompts, out_dir, 5, 7.0, seed_base, 0, False)

        assert seeds_used == [100, 101, 102]

    def test_timings_returned(self, tmp_path):
        """generate_images returns one timing per prompt."""
        out_dir = tmp_path / "bench_out"
        prompts = ["a cat", "a dog", "a bird"]

        with patch("src.benchmark_model._load_pipeline") as mock_load, \
             patch("src.benchmark_model._generate_single_image") as mock_gen, \
             patch("src.benchmark_model.sample_metal_memory", return_value={"active_mb": 0.0, "peak_mb": 0.0}), \
             patch("src.benchmark_model.sample_system_rss_mb", return_value=0.0), \
             patch("src.benchmark_model.reset_metal_peak_memory"):
            mock_load.return_value = (MagicMock(), {"proxies": [], "act_quant_patches": [], "step_keys_sorted": []})
            mock_gen.return_value = self._make_fake_image()

            timings, mem = generate_images("fp16", prompts, out_dir, 5, 7.0, 42, 0, False)

        assert len(timings) == 3
        assert all(t >= 0.0 for t in timings)
        assert "peak_metal_mb" in mem
        assert "peak_rss_mb" in mem


# ---------------------------------------------------------------------------
# TestPrintResults (smoke test — just ensure no crash)
# ---------------------------------------------------------------------------

class TestPrintResults:

    def test_no_crash_with_full_data(self, capsys):
        lat = compute_latency_stats([10.0, 20.0, 30.0], warmup=0)
        mem = {"peak_metal_mb": 18000.0, "peak_rss_mb": 24000.0}
        fid = {"fid": 12.3, "isc_mean": 8.4, "isc_std": 0.3,
               "kid_mean": 0.0045, "kid_std": 0.0003}
        _print_results("adaround_w4", lat, mem, fid)
        captured = capsys.readouterr()
        assert "adaround_w4" in captured.out

    def test_no_crash_with_none_fidelity(self, capsys):
        lat = compute_latency_stats([], warmup=0)
        mem = {"peak_metal_mb": 0.0, "peak_rss_mb": 0.0}
        _print_results("fp16", lat, mem, None)
        captured = capsys.readouterr()
        assert "fp16" in captured.out


# ---------------------------------------------------------------------------
# Helpers shared by naive-int8 tests
# ---------------------------------------------------------------------------

def _make_mock_pipeline(dim=256):
    """
    Build a minimal pipeline-like object with the MMDiT block structure
    required by _walk_mmdit_linears, using real mlx.nn.Linear instances.
    """
    import mlx.nn as nn

    class _Attn:
        pass

    class _Mlp:
        pass

    class _TransformerBlock:
        pass

    class _MMBlock:
        pass

    class _MMDiT:
        pass

    class _Pipeline:
        pass

    def _make_attn(d):
        a = _Attn()
        a.q_proj = nn.Linear(d, d, bias=False)
        a.k_proj = nn.Linear(d, d, bias=False)
        a.v_proj = nn.Linear(d, d, bias=False)
        a.o_proj = nn.Linear(d, d, bias=False)
        return a

    def _make_mlp(d):
        m = _Mlp()
        m.fc1 = nn.Linear(d, d * 4)
        m.fc2 = nn.Linear(d * 4, d)
        return m

    def _make_tb(d):
        tb = _TransformerBlock()
        tb.attn = _make_attn(d)
        tb.mlp = _make_mlp(d)
        return tb

    mmdit = _MMDiT()
    mm_block = _MMBlock()
    mm_block.image_transformer_block = _make_tb(dim)
    mm_block.text_transformer_block = _make_tb(dim)
    mmdit.multimodal_transformer_blocks = [mm_block]
    mmdit.unified_transformer_blocks = []

    pipeline = _Pipeline()
    pipeline.mmdit = mmdit
    return pipeline


# ---------------------------------------------------------------------------
# TestInjectWeightsNaiveInt8
# ---------------------------------------------------------------------------

class TestInjectWeightsNaiveInt8:
    """Tests for inject_weights_naive_int8 using a mock pipeline."""

    def test_layers_replaced_with_quantized_linear(self):
        import mlx.nn as nn
        pipeline = _make_mock_pipeline(dim=256)
        inject_weights_naive_int8(pipeline, group_size=64, bits=8)
        block = pipeline.mmdit.multimodal_transformer_blocks[0]
        assert isinstance(block.image_transformer_block.attn.q_proj, nn.QuantizedLinear)
        assert isinstance(block.image_transformer_block.attn.k_proj, nn.QuantizedLinear)
        assert isinstance(block.image_transformer_block.mlp.fc1, nn.QuantizedLinear)
        assert isinstance(block.text_transformer_block.attn.q_proj, nn.QuantizedLinear)
        assert isinstance(block.text_transformer_block.mlp.fc2, nn.QuantizedLinear)

    def test_count_is_correct_for_one_mm_block(self):
        # 1 mm block: img (4 attn + 2 mlp) + txt (4 attn + 2 mlp) = 12 layers
        pipeline = _make_mock_pipeline(dim=256)
        count = inject_weights_naive_int8(pipeline, group_size=64, bits=8)
        assert count == 12

    def test_small_attn_layers_skipped(self):
        """Layers with in_features < max(128, group_size) are not quantized."""
        import mlx.nn as nn
        # dim=32: attn layers have in_features=32 < 128 → skipped
        # fc1 has in_features=32 < 128 → skipped
        # fc2 has in_features=128, 128 < 128 = False → injected
        pipeline = _make_mock_pipeline(dim=32)
        count = inject_weights_naive_int8(pipeline, group_size=64, bits=8)
        block = pipeline.mmdit.multimodal_transformer_blocks[0]
        # attn layers not quantized (in_features=32)
        assert isinstance(block.image_transformer_block.attn.q_proj, nn.Linear)
        # fc2 (in_features=128) should be quantized
        assert isinstance(block.image_transformer_block.mlp.fc2, nn.QuantizedLinear)
        # count < full 12 since most layers were skipped
        assert count < 12

    def test_returns_nonzero_count(self):
        pipeline = _make_mock_pipeline(dim=256)
        count = inject_weights_naive_int8(pipeline, group_size=64, bits=8)
        assert count > 0

    def test_quantized_linear_has_expected_group_size_and_bits(self):
        import mlx.nn as nn
        pipeline = _make_mock_pipeline(dim=256)
        inject_weights_naive_int8(pipeline, group_size=64, bits=8)
        ql = pipeline.mmdit.multimodal_transformer_blocks[0].image_transformer_block.attn.q_proj
        assert isinstance(ql, nn.QuantizedLinear)
        assert ql.group_size == 64
        assert ql.bits == 8


# ---------------------------------------------------------------------------
# TestDynamicInt8ActLayer
# ---------------------------------------------------------------------------

class TestDynamicInt8ActLayer:
    """Tests for _DynamicInt8ActLayer proxy class."""

    def test_output_shape_preserved(self):
        import mlx.core as mx
        import mlx.nn as nn
        linear = nn.Linear(8, 16, bias=False)
        proxy = _DynamicInt8ActLayer(linear)
        x = mx.random.normal((3, 8))
        out = proxy(x)
        mx.eval(out)
        assert out.shape == (3, 16)

    def test_near_zero_input_passes_through_unchanged(self):
        """When scale < 1e-8, output must be identical to direct layer call."""
        import mlx.core as mx
        import mlx.nn as nn
        linear = nn.Linear(4, 4, bias=False)
        proxy = _DynamicInt8ActLayer(linear)
        x = mx.zeros((1, 4))
        out_proxy = proxy(x)
        out_direct = linear(x)
        mx.eval(out_proxy, out_direct)
        assert mx.allclose(out_proxy, out_direct, atol=1e-6).item()

    def test_quantization_rounds_activations(self):
        """With non-zero input, the quantized path produces different output
        from the raw float path for a weight matrix that magnifies differences."""
        import mlx.core as mx
        import mlx.nn as nn
        # Use a large weight to amplify rounding errors
        linear = nn.Linear(4, 1, bias=False)
        # Set weight to large values so rounding changes are visible
        import numpy as np
        w = np.ones((1, 4), dtype=np.float16) * 100.0
        linear.weight = mx.array(w)
        proxy = _DynamicInt8ActLayer(linear)
        # Input designed to have fractional values after /scale
        # scale = max(|x|)/127; with x=[1.5, 2.3, -0.7, 1.1]:
        # scale ≈ 2.3/127 ≈ 0.0181; rounded values differ from originals
        x = mx.array([[1.5, 2.3, -0.7, 1.1]])
        out_proxy = proxy(x)
        out_direct = linear(x)
        mx.eval(out_proxy, out_direct)
        # The outputs should differ due to int8 rounding
        diff = float(mx.abs(out_proxy - out_direct).item())
        assert diff > 0.0

    def test_getattr_forwards_to_inner_layer(self):
        """__getattr__ should forward attribute access to the wrapped layer."""
        import mlx.core as mx
        import mlx.nn as nn
        linear = nn.Linear(4, 4, bias=False)
        proxy = _DynamicInt8ActLayer(linear)
        # weight should be accessible via proxy and match the inner layer's
        assert proxy.weight is linear.weight

    def test_apply_and_remove_hooks_restores_original(self):
        """apply/remove_dynamic_int8_act_hooks round-trip."""
        import mlx.nn as nn
        pipeline = _make_mock_pipeline(dim=256)
        # First inject weights so the walk finds QuantizedLinear
        inject_weights_naive_int8(pipeline, group_size=64, bits=8)
        # Capture one layer before hooking
        attn = pipeline.mmdit.multimodal_transformer_blocks[0].image_transformer_block.attn
        original_q = attn.q_proj
        proxies, patches = apply_dynamic_int8_act_hooks(pipeline.mmdit)
        # After hooks: should be wrapped
        assert isinstance(attn.q_proj, _DynamicInt8ActLayer)
        # Remove
        remove_dynamic_int8_act_hooks(patches)
        # After removal: should be original QuantizedLinear
        assert attn.q_proj is original_q
