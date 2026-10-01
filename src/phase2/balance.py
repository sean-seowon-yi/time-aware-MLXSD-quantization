"""CSB re-parameterization: absorb balancing into adaLN, balance weights.

For layers preceded by adaLN (q/k/v_proj, fc1, final_layer.linear), the
B^{-1} activation scaling is absorbed into the adaLN modulation weights,
incurring zero runtime overhead.  Layers without a preceding adaLN (o_proj,
fc2) store a b_inv vector for online element-wise multiplication.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import numpy as np

from .config import HIDDEN_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# adaLN absorption
# ---------------------------------------------------------------------------

def absorb_into_adaln(
    adaln_linear,
    b_qkv: np.ndarray,
    b_fc1: np.ndarray | None = None,
    hidden_size: int = HIDDEN_SIZE,
) -> None:
    """Modify a block's ``adaLN_modulation`` Linear in-place.

    For a standard block (num_modulation_params=6), the output layout is:

        [beta1 | gamma1 | alpha1 | beta2 | gamma2 | alpha2]
         0:H     H:2H    2H:3H    3H:4H   4H:5H   5H:6H

    For block-23 text (num_modulation_params=2), the layout is:

        [beta1 | gamma1]
         0:H     H:2H

    Parameters
    ----------
    adaln_linear : nn.Linear
        The ``adaLN_modulation.layers[1]`` module (shape [N*H, 1536]).
    b_qkv : ndarray [H]
        Shared balancing vector for q/k/v_proj.
    b_fc1 : ndarray [H] or None
        Balancing vector for fc1.  Pass None for block-23 text.
    """
    H = hidden_size
    W = np.array(adaln_linear.weight)
    bias = np.array(adaln_linear.bias)

    # --- q/k/v_proj absorption (pre-attention shift & scale) ---
    # Shift beta1: rows [0:H]
    W[:H] /= b_qkv[:, None]
    bias[:H] /= b_qkv

    # Scale gamma1: rows [H:2H]  —  SD3's (1+gamma) formulation
    W[H : 2 * H] /= b_qkv[:, None]
    bias[H : 2 * H] = (1 + bias[H : 2 * H]) / b_qkv - 1

    # Gate alpha1: rows [2H:3H] — unchanged

    # --- fc1 absorption (pre-FFN shift & scale, only if present) ---
    if b_fc1 is not None:
        # Shift beta2: rows [3H:4H]
        W[3 * H : 4 * H] /= b_fc1[:, None]
        bias[3 * H : 4 * H] /= b_fc1

        # Scale gamma2: rows [4H:5H]
        W[4 * H : 5 * H] /= b_fc1[:, None]
        bias[4 * H : 5 * H] = (1 + bias[4 * H : 5 * H]) / b_fc1 - 1

        # Gate alpha2: rows [5H:6H] — unchanged

    adaln_linear.weight = mx.array(W, dtype=adaln_linear.weight.dtype)
    adaln_linear.bias = mx.array(bias, dtype=adaln_linear.bias.dtype)


def absorb_into_final_adaln(
    adaln_linear,
    b_final: np.ndarray,
    hidden_size: int = HIDDEN_SIZE,
) -> None:
    """Modify ``final_layer.adaLN_modulation`` Linear in-place.

    Layout: [beta | gamma]  (shape [2H, 1536]).
    """
    H = hidden_size
    W = np.array(adaln_linear.weight)
    bias = np.array(adaln_linear.bias)

    # Shift: rows [0:H]
    W[:H] /= b_final[:, None]
    bias[:H] /= b_final

    # Scale: rows [H:2H]
    W[H : 2 * H] /= b_final[:, None]
    bias[H : 2 * H] = (1 + bias[H : 2 * H]) / b_final - 1

    adaln_linear.weight = mx.array(W, dtype=adaln_linear.weight.dtype)
    adaln_linear.bias = mx.array(bias, dtype=adaln_linear.bias.dtype)


# ---------------------------------------------------------------------------
# Weight balancing
# ---------------------------------------------------------------------------

def balance_weight(linear, b_vector: np.ndarray) -> None:
    """Modify ``linear.weight`` in-place: W_new = W * b[None, :].

    This is the weight-side half of CSB.  The bias (if present) is unaffected
    because it operates on the output dimension.
    """
    W = np.array(linear.weight)        # [d_out, d_in]
    W *= b_vector[None, :]
    linear.weight = mx.array(W, dtype=linear.weight.dtype)


# ---------------------------------------------------------------------------
# Full CSB application
# ---------------------------------------------------------------------------

def apply_csb_to_model(
    mmdit,
    registry: list[dict],
    calibration: dict,
    hidden_size: int = HIDDEN_SIZE,
) -> dict[str, np.ndarray]:
    """Apply CSB re-parameterization to the entire MMDiT model.

    For each block and modality:
      1. Absorb b_qkv and b_fc1 into the adaLN modulation linear.
      2. Balance q/k/v/o_proj and fc1/fc2 weights.
      3. Collect b_inv vectors for online-balanced layers (o_proj, fc2).

    Returns
    -------
    dict mapping layer_name → b_inv (ndarray [d_in]) for online-balanced layers.
    """
    bv = calibration["balancing_vectors"]
    b_inv_set = set(calibration["b_inv_layers"])
    b_inv_map: dict[str, np.ndarray] = {}
    absorbed: set[tuple[int, str]] = set()

    for entry in registry:
        name = entry["name"]
        if name not in bv:
            continue

        b = bv[name]
        family = entry["family"]
        block = entry["block"]
        side = entry["side"]

        # --- adaLN absorption (triggered once per block/side by q_proj) ---
        if family == "q_proj" and block >= 0:
            key = (block, side)
            if key not in absorbed:
                absorbed.add(key)
                b_qkv = b
                fc1_name = f"blocks.{block}.{side}.mlp.fc1"
                b_fc1 = bv.get(fc1_name)

                mmdit_block = mmdit.multimodal_transformer_blocks[block]
                tb = (
                    mmdit_block.image_transformer_block
                    if side == "image"
                    else mmdit_block.text_transformer_block
                )
                adaln_linear = tb.adaLN_modulation.layers[1]
                absorb_into_adaln(adaln_linear, b_qkv, b_fc1, hidden_size=hidden_size)
                logger.info("Absorbed CSB into block %d %s adaLN", block, side)

        if family == "final_linear":
            adaln_linear = mmdit.final_layer.adaLN_modulation.layers[1]
            absorb_into_final_adaln(adaln_linear, b, hidden_size=hidden_size)
            logger.info("Absorbed CSB into final_layer adaLN")

        # --- Weight balancing ---
        balance_weight(entry["module"], b)

        # --- Store b_inv for online-balanced layers ---
        if name in b_inv_set:
            b_inv_map[name] = (1.0 / b).astype(np.float32)

    logger.info(
        "CSB applied: %d weights balanced, %d b_inv stored, %d adaLN absorbed",
        len([n for n in bv if n not in set(calibration["b_inv_layers"])
             and n in {e["name"] for e in registry}]),
        len(b_inv_map),
        len(absorbed) + (1 if "final_layer.linear" in bv else 0),
    )
    return b_inv_map
