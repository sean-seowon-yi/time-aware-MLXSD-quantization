"""End-to-end integration test: calibrate → CSB → static quantize → save → load."""

import mlx.core as mx
import numpy as np
import pytest

from conftest import H, FFN_H, MockMMDiT, MockPipeline

from src.phase2.balance import apply_csb_to_model
from src.phase2.calibrate import calibrate_all_layers, save_calibration, load_calibration
from src.phase2.quantize import (
    _navigate_to_parent,
    patch_pipeline_for_quantized_inference,
)
from src.phase2.quantize_static import (
    W4A8StaticLinear,
    compute_static_scales,
    load_quantized_model_static,
    quantize_model_static,
    save_quantized_model_static,
)


class TestFullPipeline:

    def test_calibrate_csb_quantize_save_load(
        self, mock_mmdit, registry, mock_diagnostics, test_config, tmp_path,
    ):
        """Run the full Phase 2 pipeline on the mock model and verify
        the loaded model has the correct structure and can forward."""

        # === 1. Calibrate ===
        cal = calibrate_all_layers(registry, mock_diagnostics, test_config)
        assert len(cal["balancing_vectors"]) > 0

        # === 2. Apply CSB ===
        b_inv_map = apply_csb_to_model(
            mock_mmdit, registry, cal, hidden_size=H,
        )
        assert len(b_inv_map) > 0

        # === 3. Patch pipeline for adaLN preservation ===
        pipeline = MockPipeline(mock_mmdit)
        patch_pipeline_for_quantized_inference(pipeline)

        light_registry = [
            {"name": e["name"], "block": e["block"], "family": e["family"], "side": e["side"]}
            for e in registry
        ]

        # === 4. Static scales + quantize ===
        static_scales = compute_static_scales(
            light_registry,
            mock_diagnostics,
            cal,
            test_config,
            mode="ssc_weighted",
            granularity="per_tensor",
        )
        layer_meta = quantize_model_static(
            mock_mmdit, registry, b_inv_map, static_scales, test_config,
        )
        assert len(layer_meta) > 0

        # === 5. Save ===
        out_dir = tmp_path / "quantized"
        cfg_save = {**test_config, "model_version": "test-mock-model"}
        save_quantized_model_static(
            mock_mmdit, out_dir, cfg_save, layer_meta, cal["b_inv_layers"],
            static_scales, granularity="per_tensor", mode="ssc_weighted",
        )
        assert (out_dir / "mmdit_quantized.safetensors").exists()
        assert (out_dir / "quantize_config.json").exists()

        # === 6. Load into a fresh model ===
        fresh_mmdit = MockMMDiT()
        mx.eval(fresh_mmdit.parameters())
        fresh_pipeline = MockPipeline(fresh_mmdit)

        loaded_meta = load_quantized_model_static(fresh_pipeline, out_dir)
        assert loaded_meta["model_version"] == "test-mock-model"
        assert len(loaded_meta["quantized_layers"]) == len(layer_meta)

        # === 7. Verify structure ===
        excluded = set(test_config["exclude_layers"])
        for entry in registry:
            if entry["name"] in excluded:
                continue
            parent, attr = _navigate_to_parent(fresh_mmdit, entry["name"])
            layer = getattr(parent, attr)
            assert isinstance(layer, W4A8StaticLinear), (
                f"{entry['name']} should be W4A8StaticLinear after load"
            )

        # context_embedder should remain nn.Linear
        import mlx.nn as nn
        assert isinstance(fresh_mmdit.context_embedder, nn.Linear)

        # === 8. Forward pass on loaded model ===
        parent, attr = _navigate_to_parent(fresh_mmdit, "blocks.0.image.attn.q_proj")
        q_proj = getattr(parent, attr)
        x = mx.random.normal((1, 4, H))
        y = q_proj(x)
        mx.eval(y)
        assert y.shape == (1, 4, H)
        assert np.all(np.isfinite(np.array(y)))

        # fc2 with b_inv
        parent, attr = _navigate_to_parent(fresh_mmdit, "blocks.0.image.mlp.fc2")
        fc2 = getattr(parent, attr)
        x_ffn = mx.random.normal((1, 4, FFN_H))
        y_ffn = fc2(x_ffn)
        mx.eval(y_ffn)
        assert y_ffn.shape == (1, 4, H)

        # === 9. adaLN patch works ===
        result = fresh_pipeline.load_mmdit(only_modulation_dict=True)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_calibration_persistence_in_pipeline(
        self, registry, mock_diagnostics, test_config, tmp_path,
    ):
        """Calibrate → save → load calibration → use in CSB."""
        cal = calibrate_all_layers(registry, mock_diagnostics, test_config)

        cal_dir = tmp_path / "cal"
        save_calibration(cal, cal_dir)
        loaded_cal = load_calibration(cal_dir)

        mmdit = MockMMDiT()
        mx.eval(mmdit.parameters())

        b_inv_map = apply_csb_to_model(
            mmdit, registry, loaded_cal, hidden_size=H,
        )
        assert len(b_inv_map) == len(loaded_cal["b_inv_layers"])


class TestHyperparameterSweep:
    """Test the pipeline with different hyperparameter combinations."""

    @pytest.mark.parametrize(
        "alpha,qkv_method",
        [
            (0.3, "max"),
            (0.5, "max"),
            (0.7, "max"),
            (0.3, "geomean"),
            (0.5, "geomean"),
            (0.7, "geomean"),
        ],
    )
    def test_calibrate_and_csb_with_config(
        self, registry, mock_diagnostics, test_config, alpha, qkv_method,
    ):
        cfg = {**test_config, "alpha": alpha, "qkv_method": qkv_method}

        cal = calibrate_all_layers(registry, mock_diagnostics, cfg)
        assert len(cal["balancing_vectors"]) > 0

        mmdit = MockMMDiT()
        mx.eval(mmdit.parameters())

        b_inv_map = apply_csb_to_model(mmdit, registry, cal, hidden_size=H)
        assert isinstance(b_inv_map, dict)

        light_registry = [
            {"name": e["name"], "block": e["block"], "family": e["family"], "side": e["side"]}
            for e in registry
        ]
        static_scales = compute_static_scales(
            light_registry, mock_diagnostics, cal, cfg,
            mode="ssc_weighted", granularity="per_tensor",
        )
        layer_meta = quantize_model_static(mmdit, registry, b_inv_map, static_scales, cfg)
        assert len(layer_meta) > 0
