"""Tests for src.phase1.collect: weight salience and persistence."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from src.phase1.collect import (
    compute_weight_salience,
    load_activation_stats,
    load_weight_stats,
    save_activation_stats,
    save_weight_stats,
)
from src.phase1.hooks import ChannelStatsCollector
from tests.conftest import H, FFN_H


# ---------------------------------------------------------------------------
# compute_weight_salience
# ---------------------------------------------------------------------------

class TestComputeWeightSalience:

    def test_returns_all_layers(self, registry):
        ws = compute_weight_salience(registry)
        assert set(ws.keys()) == {e["name"] for e in registry}

    def test_correct_shapes(self, registry):
        ws = compute_weight_salience(registry)
        for entry in registry:
            stats = ws[entry["name"]]
            assert stats["w_channel_max"].shape == (entry["d_in"],)
            assert stats["w_channel_mean"].shape == (entry["d_in"],)

    def test_max_geq_mean(self, registry):
        ws = compute_weight_salience(registry)
        for name, stats in ws.items():
            assert np.all(stats["w_channel_max"] >= stats["w_channel_mean"] - 1e-6)

    def test_values_are_nonnegative(self, registry):
        ws = compute_weight_salience(registry)
        for name, stats in ws.items():
            assert np.all(stats["w_channel_max"] >= 0)
            assert np.all(stats["w_channel_mean"] >= 0)


# ---------------------------------------------------------------------------
# Weight stats persistence
# ---------------------------------------------------------------------------

class TestWeightStatsPersistence:

    def test_save_load_roundtrip(self, registry, tmp_path):
        ws = compute_weight_salience(registry)
        save_weight_stats(ws, output_dir=tmp_path)

        loaded = load_weight_stats(output_dir=tmp_path)
        assert set(loaded.keys()) == set(ws.keys())

        for name in ws:
            for key in ("w_channel_max", "w_channel_mean"):
                np.testing.assert_allclose(
                    loaded[name][key], ws[name][key], atol=1e-6,
                )

    def test_npz_key_format(self, registry, tmp_path):
        ws = compute_weight_salience(registry)
        save_weight_stats(ws, output_dir=tmp_path)

        data = np.load(tmp_path / "weight_stats.npz")
        for key in data.files:
            parts = key.rsplit("/", 1)
            assert len(parts) == 2
            assert parts[1] in ("w_channel_max", "w_channel_mean")


# ---------------------------------------------------------------------------
# Activation stats persistence
# ---------------------------------------------------------------------------

class TestActivationStatsPersistence:

    def _make_collector(self, registry, n_steps=3):
        collector = ChannelStatsCollector()
        for step in range(n_steps):
            collector.set_context(
                step_idx=step, sigma=14.0 - step * 4.0,
                prompt_id="0", seed=42,
            )
            for entry in registry:
                X = mx.ones((1, 4, entry["d_in"]))
                W = entry["module"].weight
                collector.record(entry["name"], X, W)
        return collector

    def test_save_load_roundtrip(self, registry, tmp_path):
        n_steps = 3
        collector = self._make_collector(registry, n_steps)
        save_activation_stats(collector, registry, output_dir=tmp_path)

        for entry in registry:
            stats = load_activation_stats(entry["name"], output_dir=tmp_path)
            assert "sigma_values" in stats
            assert "act_channel_max" in stats
            assert "act_channel_mean" in stats
            assert stats["act_channel_max"].shape == (n_steps, entry["d_in"])
            assert stats["act_channel_mean"].shape == (n_steps, entry["d_in"])
            assert stats["sigma_values"].shape == (n_steps,)

    def test_per_layer_npz_files_created(self, registry, tmp_path):
        collector = self._make_collector(registry)
        save_activation_stats(collector, registry, output_dir=tmp_path)

        for entry in registry:
            path = tmp_path / f"{entry['name']}.npz"
            assert path.exists(), f"Missing {path}"
