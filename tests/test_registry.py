"""Tests for src.phase1.registry.build_layer_registry."""

import mlx.nn as nn
import pytest


class TestBuildLayerRegistry:
    """Validate registry construction against MockMMDiT."""

    def test_registry_count(self, registry):
        # 2 blocks × 2 sides × (3 qkv + o_proj + fc1 + fc2) = 24
        # minus: last block text skips o_proj, fc1, fc2 = -3
        # plus: context_embedder + final_layer.linear = +2
        # total = 24 - 3 + 2 = 23
        assert len(registry) == 23

    def test_registry_entry_schema(self, registry):
        required_keys = {"name", "module", "block", "family", "side", "d_in"}
        for entry in registry:
            assert set(entry.keys()) == required_keys

    def test_names_are_unique(self, registry):
        names = [e["name"] for e in registry]
        assert len(names) == len(set(names))

    def test_block_indices(self, registry):
        block_indices = {e["block"] for e in registry}
        assert block_indices == {0, 1, -1}

    def test_sides(self, registry):
        sides = {e["side"] for e in registry}
        assert sides == {"image", "text", "shared"}

    def test_families(self, registry):
        families = {e["family"] for e in registry}
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2",
                    "context_embedder", "final_linear"}
        assert families == expected

    def test_d_in_values(self, registry):
        from tests.conftest import H, FFN_H
        for entry in registry:
            if entry["family"] == "fc2":
                assert entry["d_in"] == FFN_H
            else:
                assert entry["d_in"] == H

    def test_module_references_are_linear(self, registry):
        for entry in registry:
            assert isinstance(entry["module"], nn.Linear)

    def test_skip_post_sdpa_excludes_text_layers(self, registry):
        """Last block's text side should skip o_proj, fc1, fc2."""
        last_block_text = [
            e for e in registry if e["block"] == 1 and e["side"] == "text"
        ]
        families = {e["family"] for e in last_block_text}
        assert "o_proj" not in families
        assert "fc1" not in families
        assert "fc2" not in families
        assert families == {"q_proj", "k_proj", "v_proj"}

    def test_context_embedder_entry(self, registry):
        ce = [e for e in registry if e["name"] == "context_embedder"]
        assert len(ce) == 1
        assert ce[0]["block"] == -1
        assert ce[0]["side"] == "shared"

    def test_final_layer_entry(self, registry):
        fl = [e for e in registry if e["name"] == "final_layer.linear"]
        assert len(fl) == 1
        assert fl[0]["block"] == -1
        assert fl[0]["side"] == "image"
