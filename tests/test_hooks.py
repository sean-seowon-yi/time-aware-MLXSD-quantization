"""Tests for src.phase1.hooks: ChannelStatsCollector, LinearHook, install/remove."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from src.phase1.hooks import (
    ChannelStatsCollector,
    LinearHook,
    install_hooks,
    remove_hooks,
)


# ---------------------------------------------------------------------------
# ChannelStatsCollector
# ---------------------------------------------------------------------------

class TestChannelStatsCollector:

    def test_record_increments_call_count(self):
        c = ChannelStatsCollector()
        c.set_context(step_idx=0, sigma=14.0, prompt_id="0", seed=42)
        X = mx.ones((2, 4, 8))
        W = mx.ones((16, 8))
        c.record("layer_a", X, W)
        assert c.call_count == 1
        c.record("layer_a", X, W)
        assert c.call_count == 2

    def test_layer_names_and_num_steps(self):
        c = ChannelStatsCollector()
        X = mx.ones((1, 4, 8))
        W = mx.ones((16, 8))
        for step in range(3):
            c.set_context(step_idx=step, sigma=14.0 - step, prompt_id="0", seed=42)
            c.record("layer_a", X, W)
            c.record("layer_b", X, W)
        assert c.layer_names() == {"layer_a", "layer_b"}
        assert c.num_steps() == 3

    def test_trajectory_shape(self):
        c = ChannelStatsCollector()
        d_in = 16
        X = mx.ones((1, 4, d_in))
        W = mx.ones((32, d_in))
        n_steps = 5
        for step in range(n_steps):
            c.set_context(step_idx=step, sigma=14.0 - step, prompt_id="0", seed=42)
            c.record("layer_a", X, W)
        traj = c.get_trajectory("layer_a")
        assert traj.shape == (n_steps, d_in)

    def test_abs_max_reduction_across_prompts(self):
        c = ChannelStatsCollector()
        d_in = 4
        W = mx.ones((8, d_in))

        c.set_context(step_idx=0, sigma=14.0, prompt_id="0", seed=42)
        X1 = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
        c.record("layer_a", X1, W)

        c.set_context(step_idx=0, sigma=14.0, prompt_id="1", seed=43)
        X2 = mx.array([[[5.0, 1.0, 1.0, 1.0]]])
        c.record("layer_a", X2, W)

        traj = c.get_trajectory("layer_a")
        # Should take elementwise max: [5, 2, 3, 4]
        np.testing.assert_allclose(traj[0], [5.0, 2.0, 3.0, 4.0])

    def test_mean_trajectory(self):
        c = ChannelStatsCollector()
        d_in = 4
        W = mx.ones((8, d_in))

        c.set_context(step_idx=0, sigma=14.0, prompt_id="0", seed=42)
        X1 = mx.array([[[2.0, 4.0, 6.0, 8.0]]])
        c.record("layer_a", X1, W)

        c.set_context(step_idx=0, sigma=14.0, prompt_id="1", seed=43)
        X2 = mx.array([[[4.0, 2.0, 2.0, 2.0]]])
        c.record("layer_a", X2, W)

        mean_traj = c.get_mean_trajectory("layer_a")
        # mean of abs means: ([2,4,6,8] + [4,2,2,2]) / 2 = [3,3,4,5]
        np.testing.assert_allclose(mean_traj[0], [3.0, 3.0, 4.0, 5.0])

    def test_sigma_values_ordering(self):
        c = ChannelStatsCollector()
        X = mx.ones((1, 4, 8))
        W = mx.ones((16, 8))
        sigmas = [14.6, 7.0, 0.03]
        for step, sigma in enumerate(sigmas):
            c.set_context(step_idx=step, sigma=sigma, prompt_id="0", seed=42)
            c.record("layer_a", X, W)
        np.testing.assert_allclose(c.sigma_values(), sigmas)


# ---------------------------------------------------------------------------
# LinearHook
# ---------------------------------------------------------------------------

class TestLinearHook:

    def test_hook_fires_and_preserves_output(self):
        layer = nn.Linear(8, 16)
        mx.eval(layer.parameters())
        collector = ChannelStatsCollector()
        collector.set_context(step_idx=0, sigma=14.0, prompt_id="0", seed=42)

        hook = LinearHook(layer, "test_layer", collector)
        x = mx.ones((1, 8))
        out = layer(x)
        mx.eval(out)

        assert collector.call_count == 1
        assert out.shape == (1, 16)

    def test_remove_restores_class(self):
        layer = nn.Linear(8, 16)
        original_cls = layer.__class__
        collector = ChannelStatsCollector()

        hook = LinearHook(layer, "test_layer", collector)
        assert layer.__class__ is not original_cls
        assert "Hooked" in layer.__class__.__name__

        hook.remove()
        assert layer.__class__ is original_cls


# ---------------------------------------------------------------------------
# install_hooks / remove_hooks
# ---------------------------------------------------------------------------

class TestInstallRemoveHooks:

    def test_install_count(self, registry):
        collector = ChannelStatsCollector()
        hooks = install_hooks(registry, collector)
        assert len(hooks) == len(registry)
        remove_hooks(hooks)

    def test_all_layers_hooked(self, mock_mmdit, registry):
        collector = ChannelStatsCollector()
        hooks = install_hooks(registry, collector)

        for entry in registry:
            assert "Hooked" in entry["module"].__class__.__name__

        remove_hooks(hooks)

    def test_all_layers_restored_after_remove(self, registry):
        collector = ChannelStatsCollector()
        hooks = install_hooks(registry, collector)
        remove_hooks(hooks)

        for entry in registry:
            assert "Hooked" not in entry["module"].__class__.__name__

    def test_hooks_fire_during_forward(self, mock_mmdit, registry):
        collector = ChannelStatsCollector()
        collector.set_context(step_idx=0, sigma=14.0, prompt_id="0", seed=42)
        hooks = install_hooks(registry, collector)

        # Forward pass through one linear layer
        first_entry = registry[0]
        x = mx.ones((1, first_entry["d_in"]))
        _ = first_entry["module"](x)

        assert collector.call_count == 1
        assert first_entry["name"] in collector.layer_names()

        remove_hooks(hooks)
