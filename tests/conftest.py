"""Shared fixtures for Phase 2 tests.

Builds a lightweight mock MMDiT with the same module hierarchy as real SD3
Medium (2 transformer blocks, final layer, context embedder), but with tiny
dimensions so tests run in seconds without GPU pressure.
"""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Dimensions (must satisfy: H % GROUP_SIZE == 0, FFN_H % GROUP_SIZE == 0)
# ---------------------------------------------------------------------------

H = 256
FFN_H = 512
N_BLOCKS = 2
T = 5
GROUP_SIZE = 64


# ---------------------------------------------------------------------------
# Mock MMDiT module hierarchy
# ---------------------------------------------------------------------------

class MockAttn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q_proj = nn.Linear(h, h)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h)
        self.o_proj = nn.Linear(h, h)


class MockMLP(nn.Module):
    def __init__(self, h, ffn_h):
        super().__init__()
        self.fc1 = nn.Linear(h, ffn_h)
        self.fc2 = nn.Linear(ffn_h, h)


class MockTransformerBlock(nn.Module):
    def __init__(self, h, ffn_h, n_mod=6, skip=False):
        super().__init__()
        self.attn = MockAttn(h)
        self.mlp = MockMLP(h, ffn_h)
        adaln = nn.Module()
        adaln.layers = [nn.SiLU(), nn.Linear(h, n_mod * h)]
        self.adaLN_modulation = adaln
        self.skip_post_sdpa = skip


class MockMultimodalBlock(nn.Module):
    def __init__(self, h, ffn_h, is_last=False):
        super().__init__()
        self.image_transformer_block = MockTransformerBlock(h, ffn_h)
        if is_last:
            self.text_transformer_block = MockTransformerBlock(
                h, ffn_h, n_mod=2, skip=True,
            )
        else:
            self.text_transformer_block = MockTransformerBlock(h, ffn_h)


class MockFinalLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.linear = nn.Linear(h, h)
        adaln = nn.Module()
        adaln.layers = [nn.SiLU(), nn.Linear(h, 2 * h)]
        self.adaLN_modulation = adaln


class MockMMDiT(nn.Module):
    def __init__(self, h=H, ffn_h=FFN_H, n_blocks=N_BLOCKS):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(
                MockMultimodalBlock(h, ffn_h, is_last=(i == n_blocks - 1))
            )
        self.multimodal_transformer_blocks = blocks
        self.context_embedder = nn.Linear(h, h)
        self.final_layer = MockFinalLayer(h)


class MockPipeline:
    """Minimal stand-in for DiffusionPipeline (avoids model download)."""

    def __init__(self, mmdit):
        self.mmdit = mmdit

    def load_mmdit(self, only_modulation_dict=False):
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mmdit():
    model = MockMMDiT()
    mx.eval(model.parameters())
    return model


@pytest.fixture
def registry(mock_mmdit):
    from src.phase1.registry import build_layer_registry
    return build_layer_registry(mock_mmdit)


@pytest.fixture
def layer_names(registry):
    return [e["name"] for e in registry]


@pytest.fixture
def mock_diagnostics(tmp_path, layer_names):
    """Create synthetic Phase 1 data (activation trajectories + weight stats)."""
    diag_dir = tmp_path / "diagnostics"
    act_dir = diag_dir / "activation_stats"
    act_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    wt_flat: dict[str, np.ndarray] = {}

    for name in layer_names:
        d_in = FFN_H if "fc2" in name else H

        act_max = rng.exponential(1.0, size=(T, d_in)).astype(np.float32)
        sigma_values = np.linspace(14.6, 0.03, T).astype(np.float32)
        np.savez_compressed(
            act_dir / f"{name}.npz",
            act_channel_max=act_max,
            act_channel_mean=act_max * 0.5,
            sigma_values=sigma_values,
        )

        w_max = rng.exponential(0.5, size=(d_in,)).astype(np.float32)
        w_max[:2] = 0.0  # dead channels for edge-case testing
        wt_flat[f"{name}/w_channel_max"] = w_max
        wt_flat[f"{name}/w_channel_mean"] = w_max * 0.3

    np.savez_compressed(diag_dir / "weight_stats.npz", **wt_flat)

    config = {"layer_names": layer_names, "num_steps": T}
    (diag_dir / "config.json").write_text(json.dumps(config))

    return diag_dir


@pytest.fixture
def test_config():
    return {
        "alpha": 0.5,
        "b_min": 1e-5,
        "b_max": 1e5,
        "w_eps": 1e-12,
        "group_size": GROUP_SIZE,
        "bits": 4,
        "a_bits": 8,
        "qkv_method": "max",
        "final_layer_bits": 16,
        "exclude_layers": ["context_embedder", "final_layer.linear"],
    }
