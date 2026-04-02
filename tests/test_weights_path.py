"""Tests for --weights-path loading pattern and cache_adaround_data helpers."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from tests.conftest import H, FFN_H, MockMMDiT


# ---------------------------------------------------------------------------
# Unit: mx.load + load_weights(strict=False) pattern
# ---------------------------------------------------------------------------

class TestWeightsPathLoading:

    def test_load_weights_modifies_model(self, tmp_path):
        model = MockMMDiT()
        mx.eval(model.parameters())

        # Save modified weights
        original_w = np.array(model.context_embedder.weight)
        new_w = mx.ones_like(model.context_embedder.weight) * 99.0
        mx.save_safetensors(str(tmp_path / "modified.safetensors"), {
            "context_embedder.weight": new_w,
        })

        # Reload
        weights = mx.load(str(tmp_path / "modified.safetensors"))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())

        loaded_w = np.array(model.context_embedder.weight)
        np.testing.assert_allclose(loaded_w, 99.0)
        assert not np.allclose(loaded_w, original_w)

    def test_partial_load_preserves_unmodified(self, tmp_path):
        model = MockMMDiT()
        mx.eval(model.parameters())

        # Snapshot a layer we won't modify
        original_fl = np.array(model.final_layer.linear.weight).copy()

        # Save only context_embedder
        mx.save_safetensors(str(tmp_path / "partial.safetensors"), {
            "context_embedder.weight": mx.zeros((H, H)),
        })

        weights = mx.load(str(tmp_path / "partial.safetensors"))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())

        # Context embedder changed
        np.testing.assert_allclose(np.array(model.context_embedder.weight), 0.0)
        # Final layer unchanged
        np.testing.assert_allclose(
            np.array(model.final_layer.linear.weight), original_fl,
        )

    def test_load_multiple_layers(self, tmp_path):
        model = MockMMDiT()
        mx.eval(model.parameters())

        block0_img_q = model.multimodal_transformer_blocks[0].image_transformer_block.attn.q_proj
        new_weights = {
            "context_embedder.weight": mx.ones((H, H)) * 1.0,
            "multimodal_transformer_blocks.0.image_transformer_block.attn.q_proj.weight": mx.ones_like(block0_img_q.weight) * 2.0,
        }
        mx.save_safetensors(str(tmp_path / "multi.safetensors"), new_weights)

        weights = mx.load(str(tmp_path / "multi.safetensors"))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())

        np.testing.assert_allclose(np.array(model.context_embedder.weight), 1.0)
        np.testing.assert_allclose(np.array(block0_img_q.weight), 2.0)


# ---------------------------------------------------------------------------
# Integration: registry and hooks work on reloaded model
# ---------------------------------------------------------------------------

class TestRegistryAfterReload:

    def test_registry_works_after_weight_reload(self, tmp_path):
        from src.phase1.registry import build_layer_registry

        model = MockMMDiT()
        mx.eval(model.parameters())

        mx.save_safetensors(str(tmp_path / "weights.safetensors"), {
            "context_embedder.weight": mx.ones((H, H)) * 5.0,
        })
        weights = mx.load(str(tmp_path / "weights.safetensors"))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())

        registry = build_layer_registry(model)
        assert len(registry) == 23

        ce = [e for e in registry if e["name"] == "context_embedder"][0]
        np.testing.assert_allclose(np.array(ce["module"].weight), 5.0)

    def test_hooks_work_after_weight_reload(self, tmp_path):
        from src.phase1.hooks import (
            ChannelStatsCollector, install_hooks, remove_hooks,
        )
        from src.phase1.registry import build_layer_registry

        model = MockMMDiT()
        mx.eval(model.parameters())

        mx.save_safetensors(str(tmp_path / "weights.safetensors"), {
            "context_embedder.weight": mx.ones((H, H)) * 3.0,
        })
        weights = mx.load(str(tmp_path / "weights.safetensors"))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())

        registry = build_layer_registry(model)
        collector = ChannelStatsCollector()
        collector.set_context(step_idx=0, sigma=14.0, prompt_id="0", seed=42)
        hooks = install_hooks(registry, collector)

        # Fire one layer
        x = mx.ones((1, H))
        _ = model.context_embedder(x)
        assert collector.call_count == 1

        remove_hooks(hooks)


# ---------------------------------------------------------------------------
# Integration: weight salience changes after reload
# ---------------------------------------------------------------------------

class TestWeightSalienceAfterReload:

    def test_salience_changes_with_modified_weights(self, tmp_path):
        from src.phase1.collect import compute_weight_salience
        from src.phase1.registry import build_layer_registry

        model = MockMMDiT()
        mx.eval(model.parameters())
        registry = build_layer_registry(model)
        ws_before = compute_weight_salience(registry)

        # Modify context_embedder to have large weights
        mx.save_safetensors(str(tmp_path / "big.safetensors"), {
            "context_embedder.weight": mx.ones((H, H)) * 100.0,
        })
        weights = mx.load(str(tmp_path / "big.safetensors"))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())

        # Rebuild registry (module refs are the same objects, now with new weights)
        ws_after = compute_weight_salience(registry)

        before_max = ws_before["context_embedder"]["w_channel_max"]
        after_max = ws_after["context_embedder"]["w_channel_max"]
        assert np.all(after_max > before_max)
        np.testing.assert_allclose(after_max, 100.0)


# ---------------------------------------------------------------------------
# Unit: cache_adaround_data helpers
# ---------------------------------------------------------------------------

class TestCacheAdaroundHelpers:

    def test_block_hook_capture(self):
        from src.cache_adaround_data import BlockHook

        block = nn.Linear(8, 16)
        mx.eval(block.parameters())
        hook = BlockHook(block, "mm0", is_mm=True, list_idx=0)

        x = mx.ones((1, 8))
        out = hook(x)
        mx.eval(out)

        assert hook._last_args is not None
        assert len(hook._last_args) == 1
        assert hook._last_output is not None

    def test_block_hook_clear(self):
        from src.cache_adaround_data import BlockHook

        block = nn.Linear(8, 16)
        mx.eval(block.parameters())
        hook = BlockHook(block, "mm0", is_mm=True, list_idx=0)

        x = mx.ones((1, 8))
        hook(x)
        hook.clear()

        assert hook._last_args is None
        assert hook._last_kwargs is None
        assert hook._last_output is None

    def test_pack_unpack_roundtrip(self):
        from src.cache_adaround_data import pack_sample, load_block_data

        block_data = {
            "mm0": {
                "args": [np.ones((2, 4), dtype=np.float16), np.ones((2, 8), dtype=np.float16)],
                "kwargs": {"pe": np.ones((2, 4), dtype=np.float16)},
                "output": [np.ones((2, 4), dtype=np.float16), np.ones((2, 8), dtype=np.float16)],
            },
        }

        flat = pack_sample(block_data)
        assert "mm0__arg0" in flat
        assert "mm0__arg1" in flat
        assert "mm0__kw_pe" in flat
        assert "mm0__out0" in flat
        assert "mm0__out1" in flat

    def test_flush_hooks(self):
        from src.cache_adaround_data import BlockHook, flush_hooks

        block = nn.Linear(8, 16)
        mx.eval(block.parameters())
        hook = BlockHook(block, "mm0", is_mm=True, list_idx=0)

        x = mx.ones((1, 8))
        hook(x)

        result = flush_hooks([hook])
        assert "mm0" in result
        assert result["mm0"] is not None
        assert len(result["mm0"]["args"]) == 1
        assert isinstance(result["mm0"]["args"][0], np.ndarray)
        # Hook should be cleared after flush
        assert hook._last_args is None

    def test_load_block_data_from_npz(self, tmp_path):
        from src.cache_adaround_data import pack_sample, load_block_data

        block_data = {
            "mm0": {
                "args": [np.ones((2, 4), dtype=np.float16)],
                "kwargs": {},
                "output": np.ones((2, 8), dtype=np.float16),
            },
        }
        flat = pack_sample(block_data)
        path = tmp_path / "0000_000.npz"
        np.savez_compressed(path, **flat)

        loaded = load_block_data("mm0", [path])
        assert len(loaded) == 1
        assert "arg0" in loaded[0]
        assert "out0" in loaded[0]
        np.testing.assert_allclose(loaded[0]["arg0"], 1.0)
