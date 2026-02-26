"""
Tests for src/htg_cluster.py

All tests use synthetic small tensors — no model loading, no filesystem I/O.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.htg_cluster import (
    adjacent_agglomerative,
    _per_layer_boundaries,
    derive_consensus_partition,
    compute_per_layer_z_bar,
    build_htg_groups,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_z(T: int, D: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (T, D)).astype(np.float32)


def _make_gradual_z(T: int, D: int) -> np.ndarray:
    """Gradual drift — adjacent pairs should be closer than distant pairs."""
    z = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        z[t] = float(t) * 0.1  # monotone increasing shift
    return z


def _make_two_regime_z(T: int, D: int) -> np.ndarray:
    """First half near 0, second half near 1 — natural two-group partition."""
    z = np.zeros((T, D), dtype=np.float32)
    z[T // 2 :] = 5.0  # large gap at midpoint
    return z


# ---------------------------------------------------------------------------
# TestAdjacentAgglomerative
# ---------------------------------------------------------------------------

class TestAdjacentAgglomerative:

    def test_returns_correct_number_of_groups(self):
        z = _make_z(10, 4)
        for n_groups in range(1, 6):
            assgn = adjacent_agglomerative(z, n_groups)
            assert len(np.unique(assgn)) == n_groups, \
                f"Expected {n_groups} groups, got {len(np.unique(assgn))}"

    def test_assignments_length_matches_input(self):
        z = _make_z(8, 6)
        assgn = adjacent_agglomerative(z, 3)
        assert len(assgn) == 8

    def test_assignments_are_contiguous_and_monotone(self):
        """Group labels should be non-decreasing (adjacency constraint preserved)."""
        z = _make_z(12, 4)
        assgn = adjacent_agglomerative(z, 4)
        # Must be non-decreasing
        for i in range(1, len(assgn)):
            assert assgn[i] >= assgn[i - 1], \
                f"Non-monotone at {i}: {assgn[i-1]} -> {assgn[i]}"

    def test_single_group_all_same_label(self):
        z = _make_z(6, 3)
        assgn = adjacent_agglomerative(z, 1)
        assert np.all(assgn == 0)

    def test_n_groups_equals_t_each_own_group(self):
        T = 5
        z = _make_z(T, 4)
        assgn = adjacent_agglomerative(z, T)
        # Each timestep gets its own group label
        assert len(assgn) == T
        assert np.all(np.sort(assgn) == np.arange(T))

    def test_two_regime_splits_at_gap(self):
        """A two-regime signal should cluster into first-half / second-half."""
        T, D = 10, 8
        z = _make_two_regime_z(T, D)
        assgn = adjacent_agglomerative(z, 2)
        # All timesteps in first half should be group 0; second half group 1
        assert np.all(assgn[: T // 2] == 0)
        assert np.all(assgn[T // 2 :] == 1)

    def test_empty_input_returns_empty(self):
        z = np.zeros((0, 4), dtype=np.float32)
        assgn = adjacent_agglomerative(z, 3)
        assert len(assgn) == 0

    def test_n_groups_clamped_to_T(self):
        """Requesting more groups than timesteps should clamp to T."""
        T = 3
        z = _make_z(T, 4)
        assgn = adjacent_agglomerative(z, 100)
        assert len(assgn) == T
        assert len(np.unique(assgn)) == T

    def test_gradual_drift_produces_ordered_groups(self):
        """Gradual drift: cluster boundaries should form a regular partition."""
        T, D = 20, 4
        z = _make_gradual_z(T, D)
        n_groups = 4
        assgn = adjacent_agglomerative(z, n_groups)
        # Labels must be monotone
        assert np.all(np.diff(assgn) >= 0), "Labels must be non-decreasing"
        assert len(np.unique(assgn)) == n_groups

    def test_dtype_is_int(self):
        z = _make_z(8, 4)
        assgn = adjacent_agglomerative(z, 3)
        assert assgn.dtype in (np.int32, np.int64, int)


# ---------------------------------------------------------------------------
# TestPerLayerBoundaries
# ---------------------------------------------------------------------------

class TestPerLayerBoundaries:

    def test_two_group_boundary(self):
        # [0,0,0,1,1] → boundary at index 3
        assgn = np.array([0, 0, 0, 1, 1])
        bounds = _per_layer_boundaries(assgn)
        assert bounds == [3]

    def test_no_boundary_single_group(self):
        assgn = np.array([0, 0, 0, 0])
        bounds = _per_layer_boundaries(assgn)
        assert bounds == []

    def test_three_groups(self):
        assgn = np.array([0, 0, 1, 1, 2, 2])
        bounds = _per_layer_boundaries(assgn)
        assert bounds == [2, 4]

    def test_every_timestep_own_group(self):
        assgn = np.array([0, 1, 2, 3, 4])
        bounds = _per_layer_boundaries(assgn)
        assert bounds == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# TestDeriveConsensusPartition
# ---------------------------------------------------------------------------

class TestDeriveConsensusPartition:

    def _make_layer_assignments(self, n_layers: int, T: int, n_groups: int,
                                 seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        result = {}
        for i in range(n_layers):
            z = rng.normal(0, 1, (T, 4)).astype(np.float32)
            assgn = adjacent_agglomerative(z, n_groups)
            result[f"layer_{i}"] = assgn
        return result

    def test_output_length_matches_T(self):
        T, n_groups = 10, 3
        per_layer = self._make_layer_assignments(5, T, n_groups)
        global_assgn = derive_consensus_partition(per_layer, n_groups, T)
        assert len(global_assgn) == T

    def test_output_has_correct_n_groups(self):
        T, n_groups = 15, 4
        per_layer = self._make_layer_assignments(8, T, n_groups)
        global_assgn = derive_consensus_partition(per_layer, n_groups, T)
        assert len(np.unique(global_assgn)) == n_groups

    def test_output_is_monotone(self):
        T, n_groups = 12, 3
        per_layer = self._make_layer_assignments(6, T, n_groups)
        global_assgn = derive_consensus_partition(per_layer, n_groups, T)
        assert np.all(np.diff(global_assgn) >= 0), "Global assignments must be non-decreasing"

    def test_single_group_all_zeros(self):
        T = 8
        per_layer = {"l": np.zeros(T, dtype=int)}
        global_assgn = derive_consensus_partition(per_layer, 1, T)
        assert np.all(global_assgn == 0)

    def test_empty_layers_dict_returns_uniform_partition(self):
        T, n_groups = 10, 3
        global_assgn = derive_consensus_partition({}, n_groups, T)
        assert len(global_assgn) == T
        assert len(np.unique(global_assgn)) <= n_groups


# ---------------------------------------------------------------------------
# TestComputePerLayerZBar
# ---------------------------------------------------------------------------

class TestComputePerLayerZBar:

    def _make_per_step_full(self, step_keys, layer_names, C_in=8, seed=0):
        """Synthetic per_step_full dict matching the format from load_stats_v2."""
        rng = np.random.default_rng(seed)
        per_step_full = {}
        for sk in step_keys:
            per_step_full[sk] = {}
            for layer in layer_names:
                avg_min = rng.uniform(-1, 0, C_in).astype(np.float32)
                avg_max = rng.uniform(0, 1, C_in).astype(np.float32)
                per_step_full[sk][layer] = {
                    "avg_min": avg_min,
                    "avg_max": avg_max,
                    "shift": None,
                    "tensor_absmax": float(np.abs(avg_max).max()),
                    "hist_p999": float(np.abs(avg_max).max()) * 0.999,
                    "has_shift": False,
                }
        return per_step_full

    def test_all_layers_have_z_bar_for_all_groups(self):
        step_keys = [str(i) for i in range(0, 25, 5)]  # 5 timesteps
        layer_names = ["layer_A", "layer_B"]
        n_groups = 2

        per_step_full = self._make_per_step_full(step_keys, layer_names)
        assgn = np.array([0, 0, 0, 1, 1])
        per_layer_assgn = {ln: assgn for ln in layer_names}

        z_bar = compute_per_layer_z_bar(
            per_step_full, step_keys, layer_names,
            per_layer_assgn, assgn, n_groups,
        )

        for layer in layer_names:
            assert layer in z_bar
            for g in range(n_groups):
                assert str(g) in z_bar[layer], f"Missing group {g} for {layer}"

    def test_z_bar_shape_matches_channels(self):
        C_in = 16
        step_keys = ["0", "4", "8", "12", "16"]
        layer_names = ["test_layer"]
        n_groups = 2

        per_step_full = self._make_per_step_full(step_keys, layer_names, C_in)
        assgn = np.array([0, 0, 0, 1, 1])
        per_layer_assgn = {ln: assgn for ln in layer_names}

        z_bar = compute_per_layer_z_bar(
            per_step_full, step_keys, layer_names,
            per_layer_assgn, assgn, n_groups,
        )

        assert len(z_bar["test_layer"]["0"]) == C_in
        assert len(z_bar["test_layer"]["1"]) == C_in

    def test_z_bar_is_mean_of_group_shifts(self):
        """Verify z̄ = mean((avg_max + avg_min)/2) for timesteps in group."""
        step_keys = ["0", "4", "8"]
        layer = "my_layer"
        C_in = 4

        # Fixed values for exact verification
        per_step_full = {
            "0": {layer: {"avg_min": np.array([-1.0, -2.0, -1.0, -2.0], dtype=np.float32),
                           "avg_max": np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32),
                           "shift": None, "tensor_absmax": 2.0, "hist_p999": 1.99, "has_shift": False}},
            "4": {layer: {"avg_min": np.array([-3.0, -4.0, -3.0, -4.0], dtype=np.float32),
                           "avg_max": np.array([3.0, 4.0, 3.0, 4.0], dtype=np.float32),
                           "shift": None, "tensor_absmax": 4.0, "hist_p999": 3.99, "has_shift": False}},
            "8": {layer: {"avg_min": np.array([-5.0, -6.0, -5.0, -6.0], dtype=np.float32),
                           "avg_max": np.array([5.0, 6.0, 5.0, 6.0], dtype=np.float32),
                           "shift": None, "tensor_absmax": 6.0, "hist_p999": 5.99, "has_shift": False}},
        }
        # Group 0: steps 0,4 → z_t = (avg_max + avg_min)/2 = 0 for all (symmetric)
        # Expected z_bar group 0: all zeros (since symmetric about 0)
        assgn = np.array([0, 0, 1])
        per_layer_assgn = {layer: assgn}

        z_bar = compute_per_layer_z_bar(
            per_step_full, step_keys, [layer],
            per_layer_assgn, assgn, 2,
        )

        expected_g0 = np.zeros(C_in, dtype=np.float32)
        expected_g1 = np.zeros(C_in, dtype=np.float32)  # (5+(-5))/2 + (6+(-6))/2 / 1 = 0

        np.testing.assert_allclose(z_bar[layer]["0"], expected_g0, atol=1e-5)
        np.testing.assert_allclose(z_bar[layer]["1"], expected_g1, atol=1e-5)


# ---------------------------------------------------------------------------
# TestBuildHtgGroupsEndToEnd (uses temp filesystem)
# ---------------------------------------------------------------------------

class TestBuildHtgGroupsEndToEnd:
    """
    End-to-end test of build_htg_groups using a temporary layer_statistics.json
    and synthetic NPZ files matching the per_timestep_npz_v2 format.
    """

    def _write_fake_stats(self, tmp_dir: Path, n_timesteps: int = 8,
                           n_layers: int = 4, C_in: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        ts_dir = tmp_dir / "timestep_stats"
        ts_dir.mkdir()

        step_keys = [str(i * 4) for i in range(n_timesteps)]
        layer_names = [f"mm0.image_transformer_block.attn.{p}"
                       for p in ["q_proj", "k_proj", "v_proj", "o_proj"][:n_layers]]

        sigma_map = {sk: float(1.0 - i / (n_timesteps - 1))
                     for i, sk in enumerate(step_keys)}

        for sk in step_keys:
            npz_data = {}
            index_data = {}
            for layer in layer_names:
                safe = layer.replace(".", "_")
                avg_min = rng.uniform(-1, 0, C_in).astype(np.float32)
                avg_max = rng.uniform(0, 1, C_in).astype(np.float32)
                npz_data[f"{safe}__avg_min"] = avg_min
                npz_data[f"{safe}__avg_max"] = avg_max
                npz_data[f"{safe}__hist_counts"] = np.zeros(256, dtype=np.int32)
                npz_data[f"{safe}__hist_edges"] = np.linspace(-1, 1, 257)
                index_data[layer] = {
                    "tensor_absmax": float(np.abs(avg_max).max()),
                    "hist_p999": float(np.abs(avg_max).max()) * 0.999,
                    "has_shift": False,
                }
            np.savez(ts_dir / f"step_{sk}.npz", **npz_data)
            with open(ts_dir / f"step_{sk}_index.json", "w") as f:
                json.dump(index_data, f)

        manifest = {
            "format": "per_timestep_npz_v2",
            "hist_bins": 256,
            "timestep_dir": str(ts_dir),
            "sigma_map": sigma_map,
            "step_keys": step_keys,
            "metadata": {
                "num_images": 2,
                "num_timesteps": n_timesteps,
                "total_processed": n_timesteps * 2,
            },
        }
        stats_path = tmp_dir / "layer_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(manifest, f)

        return stats_path, step_keys, layer_names

    def test_output_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_path, _, _ = self._write_fake_stats(tmp_path)
            result = build_htg_groups(stats_path, n_groups=3)

        assert "n_groups" in result
        assert "global_groups" in result
        assert "per_layer_z_bar" in result
        assert "sigma_map" in result

    def test_correct_n_groups_in_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_path, _, _ = self._write_fake_stats(tmp_path, n_timesteps=10)
            result = build_htg_groups(stats_path, n_groups=4)

        assert result["n_groups"] == 4
        assert len(result["global_groups"]) == 4

    def test_global_groups_cover_all_timesteps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_path, step_keys, _ = self._write_fake_stats(tmp_path, n_timesteps=8)
            result = build_htg_groups(stats_path, n_groups=3)

        all_indices = []
        for g_info in result["global_groups"].values():
            all_indices.extend(g_info["timestep_indices"])
        assert sorted(all_indices) == list(range(len(step_keys)))

    def test_per_layer_z_bar_populated_for_all_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_path, _, layer_names = self._write_fake_stats(tmp_path)
            result = build_htg_groups(stats_path, n_groups=2)

        for layer in layer_names:
            assert layer in result["per_layer_z_bar"]

    def test_sigma_range_within_global_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_path, _, _ = self._write_fake_stats(tmp_path, n_timesteps=10)
            result = build_htg_groups(stats_path, n_groups=3)

        for g_info in result["global_groups"].values():
            lo, hi = g_info["sigma_range"]
            assert lo <= hi, "sigma_range must be [min, max]"

    def test_n_groups_greater_than_t_clamped(self):
        """Requesting more groups than timesteps should not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_path, step_keys, _ = self._write_fake_stats(tmp_path, n_timesteps=4)
            result = build_htg_groups(stats_path, n_groups=20)

        # Should produce at most T groups
        assert result["n_groups"] <= len(step_keys)

    def test_output_is_json_serialisable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stats_path, _, _ = self._write_fake_stats(tmp_path)
            result = build_htg_groups(stats_path, n_groups=2)
            # Must not raise
            json.dumps(result)
