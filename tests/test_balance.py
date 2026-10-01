"""Tests for src.phase2.balance — adaLN absorption, weight balancing, CSB cancellation."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from src.phase2.balance import (
    absorb_into_adaln,
    absorb_into_final_adaln,
    apply_csb_to_model,
    balance_weight,
)

H = 256
FFN_H = 512


# ====================================================================
# balance_weight
# ====================================================================

class TestBalanceWeight:

    def test_weight_scaled_correctly(self):
        linear = nn.Linear(H, H)
        mx.eval(linear.parameters())

        W_orig = np.array(linear.weight).copy()
        b = np.abs(np.random.default_rng(0).normal(size=(H,))) + 0.1

        balance_weight(linear, b)

        W_new = np.array(linear.weight)
        np.testing.assert_allclose(W_new, W_orig * b[None, :], rtol=1e-6)

    def test_bias_unchanged(self):
        linear = nn.Linear(H, H)
        mx.eval(linear.parameters())

        bias_orig = np.array(linear.bias).copy()
        b = np.abs(np.random.default_rng(1).normal(size=(H,))) + 0.1

        balance_weight(linear, b)

        np.testing.assert_array_equal(np.array(linear.bias), bias_orig)

    def test_no_bias_layer(self):
        """k_proj has no bias — balance_weight should not crash."""
        linear = nn.Linear(H, H, bias=False)
        mx.eval(linear.parameters())
        b = np.ones(H, dtype=np.float32)
        balance_weight(linear, b)


# ====================================================================
# absorb_into_adaln  (standard 6-param block)
# ====================================================================

class TestAbsorbIntoAdaln:

    def _make_adaln(self, n_mod=6):
        linear = nn.Linear(H, n_mod * H)
        mx.eval(linear.parameters())
        return linear

    def test_beta1_rows(self):
        linear = self._make_adaln()
        W0 = np.array(linear.weight)[:H].copy()
        b0 = np.array(linear.bias)[:H].copy()

        b_qkv = np.abs(np.random.default_rng(0).normal(size=(H,))) + 0.1
        absorb_into_adaln(linear, b_qkv, hidden_size=H)

        np.testing.assert_allclose(np.array(linear.weight)[:H], W0 / b_qkv[:, None], rtol=1e-5)
        np.testing.assert_allclose(np.array(linear.bias)[:H], b0 / b_qkv, rtol=1e-5)

    def test_gamma1_bias_correction(self):
        """gamma1 bias uses (1+gamma)/b - 1 for the SD3 (1+gamma) formulation."""
        linear = self._make_adaln()
        W0 = np.array(linear.weight)[H:2*H].copy()
        b0 = np.array(linear.bias)[H:2*H].copy()

        b_qkv = np.abs(np.random.default_rng(1).normal(size=(H,))) + 0.1
        absorb_into_adaln(linear, b_qkv, hidden_size=H)

        np.testing.assert_allclose(
            np.array(linear.weight)[H:2*H], W0 / b_qkv[:, None], rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(linear.bias)[H:2*H], (1 + b0) / b_qkv - 1, rtol=1e-5,
        )

    def test_alpha1_unchanged(self):
        linear = self._make_adaln()
        W0 = np.array(linear.weight)[2*H:3*H].copy()
        b0 = np.array(linear.bias)[2*H:3*H].copy()

        b_qkv = np.abs(np.random.default_rng(2).normal(size=(H,))) + 0.1
        absorb_into_adaln(linear, b_qkv, hidden_size=H)

        np.testing.assert_allclose(np.array(linear.weight)[2*H:3*H], W0, rtol=1e-7)
        np.testing.assert_allclose(np.array(linear.bias)[2*H:3*H], b0, rtol=1e-7)

    def test_fc1_beta2_gamma2(self):
        linear = self._make_adaln()
        W0 = np.array(linear.weight).copy()
        b0 = np.array(linear.bias).copy()

        rng = np.random.default_rng(3)
        b_qkv = np.abs(rng.normal(size=(H,))) + 0.1
        b_fc1 = np.abs(rng.normal(size=(H,))) + 0.1

        absorb_into_adaln(linear, b_qkv, b_fc1, hidden_size=H)

        np.testing.assert_allclose(
            np.array(linear.weight)[3*H:4*H], W0[3*H:4*H] / b_fc1[:, None], rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(linear.bias)[3*H:4*H], b0[3*H:4*H] / b_fc1, rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(linear.weight)[4*H:5*H], W0[4*H:5*H] / b_fc1[:, None], rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(linear.bias)[4*H:5*H], (1 + b0[4*H:5*H]) / b_fc1 - 1, rtol=1e-5,
        )

    def test_alpha2_unchanged_with_fc1(self):
        linear = self._make_adaln()
        W0 = np.array(linear.weight)[5*H:6*H].copy()
        b0 = np.array(linear.bias)[5*H:6*H].copy()

        rng = np.random.default_rng(4)
        b_qkv = np.abs(rng.normal(size=(H,))) + 0.1
        b_fc1 = np.abs(rng.normal(size=(H,))) + 0.1

        absorb_into_adaln(linear, b_qkv, b_fc1, hidden_size=H)

        np.testing.assert_allclose(np.array(linear.weight)[5*H:6*H], W0, rtol=1e-7)
        np.testing.assert_allclose(np.array(linear.bias)[5*H:6*H], b0, rtol=1e-7)

    def test_no_fc1_only_modifies_first_2h(self):
        """Block 23 text: n_mod=2, b_fc1=None — only beta1/gamma1 modified."""
        linear = self._make_adaln(n_mod=2)
        W0 = np.array(linear.weight).copy()
        b0 = np.array(linear.bias).copy()

        b_qkv = np.abs(np.random.default_rng(5).normal(size=(H,))) + 0.1
        absorb_into_adaln(linear, b_qkv, b_fc1=None, hidden_size=H)

        W_new = np.array(linear.weight)
        np.testing.assert_allclose(W_new[:H], W0[:H] / b_qkv[:, None], rtol=1e-5)
        np.testing.assert_allclose(W_new[H:2*H], W0[H:2*H] / b_qkv[:, None], rtol=1e-5)


# ====================================================================
# absorb_into_final_adaln
# ====================================================================

class TestAbsorbIntoFinalAdaln:

    def test_beta_and_gamma_rows(self):
        linear = nn.Linear(H, 2 * H)
        mx.eval(linear.parameters())

        W0 = np.array(linear.weight).copy()
        b0 = np.array(linear.bias).copy()

        b_final = np.abs(np.random.default_rng(6).normal(size=(H,))) + 0.1
        absorb_into_final_adaln(linear, b_final, hidden_size=H)

        np.testing.assert_allclose(np.array(linear.weight)[:H], W0[:H] / b_final[:, None], rtol=1e-5)
        np.testing.assert_allclose(np.array(linear.bias)[:H], b0[:H] / b_final, rtol=1e-5)
        np.testing.assert_allclose(np.array(linear.weight)[H:], W0[H:] / b_final[:, None], rtol=1e-5)
        np.testing.assert_allclose(np.array(linear.bias)[H:], (1 + b0[H:]) / b_final - 1, rtol=1e-5)


# ====================================================================
# CSB mathematical cancellation proofs
# ====================================================================

class TestCSBCancellation:

    def test_absorbed_layer_exact(self):
        """Y = (X*(1+gamma)+beta) @ W^T  should equal  Y' after absorption + balancing."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(4, H)).astype(np.float64)
        beta = rng.normal(size=(H,)).astype(np.float64)
        gamma = rng.normal(size=(H,)).astype(np.float64) * 0.1
        W = rng.normal(size=(H, H)).astype(np.float64) * 0.02
        bias_w = rng.normal(size=(H,)).astype(np.float64) * 0.01
        b = np.abs(rng.normal(size=(H,))) + 0.1

        Z = X * (1 + gamma) + beta
        Y = Z @ W.T + bias_w

        beta_new = beta / b
        gamma_new = (1 + gamma) / b - 1
        W_new = W * b[None, :]

        Z_new = X * (1 + gamma_new) + beta_new
        Y_new = Z_new @ W_new.T + bias_w

        np.testing.assert_allclose(Y, Y_new, rtol=1e-10)

    def test_online_balancing_exact(self):
        """Y = X @ W^T  should equal  Y' = (X*b_inv) @ (W*b)^T."""
        rng = np.random.default_rng(43)
        X = rng.normal(size=(4, H)).astype(np.float64)
        W = rng.normal(size=(H, H)).astype(np.float64) * 0.02
        b = np.abs(rng.normal(size=(H,))) + 0.1

        Y = X @ W.T

        W_new = W * b[None, :]
        X_new = X * (1.0 / b)[None, :]
        Y_new = X_new @ W_new.T

        np.testing.assert_allclose(Y, Y_new, rtol=1e-10)

    def test_online_balancing_fc2(self):
        """fc2 has d_in=FFN_H — verify cancellation with non-square weight."""
        rng = np.random.default_rng(44)
        X = rng.normal(size=(4, FFN_H)).astype(np.float64)
        W = rng.normal(size=(H, FFN_H)).astype(np.float64) * 0.02
        b = np.abs(rng.normal(size=(FFN_H,))) + 0.1

        Y = X @ W.T
        Y_new = (X / b[None, :]) @ (W * b[None, :]).T

        np.testing.assert_allclose(Y, Y_new, rtol=1e-10)


# ====================================================================
# apply_csb_to_model (integration)
# ====================================================================

class TestApplyCSBToModel:

    def test_returns_b_inv_map(
        self, mock_mmdit, registry, mock_diagnostics, test_config,
    ):
        from src.phase2.calibrate import calibrate_all_layers

        cal = calibrate_all_layers(registry, mock_diagnostics, test_config)
        b_inv_map = apply_csb_to_model(mock_mmdit, registry, cal, hidden_size=H)

        assert isinstance(b_inv_map, dict)
        for name in cal["b_inv_layers"]:
            assert name in b_inv_map
            assert b_inv_map[name].dtype == np.float32

    def test_weights_actually_modified(
        self, mock_mmdit, registry, mock_diagnostics, test_config,
    ):
        from src.phase2.calibrate import calibrate_all_layers

        q_layer = mock_mmdit.multimodal_transformer_blocks[0].image_transformer_block.attn.q_proj
        W_before = np.array(q_layer.weight).copy()

        cal = calibrate_all_layers(registry, mock_diagnostics, test_config)
        apply_csb_to_model(mock_mmdit, registry, cal, hidden_size=H)

        W_after = np.array(q_layer.weight)
        assert not np.allclose(W_before, W_after), "q_proj weight should change"

    def test_adaln_modified(
        self, mock_mmdit, registry, mock_diagnostics, test_config,
    ):
        from src.phase2.calibrate import calibrate_all_layers

        adaln = mock_mmdit.multimodal_transformer_blocks[0].image_transformer_block.adaLN_modulation.layers[1]
        bias_before = np.array(adaln.bias).copy()

        cal = calibrate_all_layers(registry, mock_diagnostics, test_config)
        apply_csb_to_model(mock_mmdit, registry, cal, hidden_size=H)

        bias_after = np.array(adaln.bias)
        assert not np.allclose(bias_before, bias_after), "adaLN bias should change"
