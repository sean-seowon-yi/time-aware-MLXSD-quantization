"""QuantizedLinear: drop-in replacement for nn.Linear with AdaRound + activation quantization."""

import mlx.core as mx
import mlx.nn as nn

from .quantizer import (
    ActivationQuantizer,
    adaround_quantize,
    adaround_hard_quantize,
    compute_weight_scale,
    init_v_from_weights,
)


class QuantizedLinear(nn.Module):
    """Fake-quantized linear layer with AdaRound weight rounding and optional activation quantization.

    During AdaRound optimization (Step 1):
        - v_param is trainable (soft rounding)
        - act_quantizer is disabled (activations pass through in FP)
        - Forward: x_q = x (FP), w_q = adaround_quantize(weight, v, scale, bits)
                   output = x_q @ w_q.T + bias

    After freeze_rounding() + activation calibration (Step 2):
        - v_param is frozen (hard rounded to 0/1)
        - act_quantizer is enabled with calibrated alpha
        - Forward: x_q = act_quantizer(x), w_q = hard_quantized_weight
                   output = x_q @ w_q.T + bias
    """

    def __init__(
        self,
        weight: mx.array,
        bias: mx.array | None,
        weight_scale: mx.array,
        v_param: mx.array,
        weight_bits: int,
        act_bits: int | None = None,
        act_symmetric: bool = True,
    ):
        super().__init__()
        # Frozen parameters (not trained)
        self.weight = weight
        self.bias = bias
        self.weight_scale = weight_scale
        self.weight_bits = weight_bits

        # Trainable: AdaRound V parameter
        self.v_param = v_param

        # State
        self._frozen = False

        # Activation quantizer (created but disabled until calibration)
        if act_bits is not None:
            self.act_quantizer = ActivationQuantizer(
                bits=act_bits, symmetric=act_symmetric
            )
        else:
            self.act_quantizer = None

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        weight_bits: int,
        act_bits: int | None = None,
        act_symmetric: bool = True,
        per_channel: bool = True,
    ) -> "QuantizedLinear":
        """Create a QuantizedLinear from an existing nn.Linear.

        Initializes weight scales and V parameters for AdaRound.
        """
        weight = linear.weight
        bias = getattr(linear, "bias", None)

        # Compute per-channel weight scale
        weight_scale = compute_weight_scale(
            weight, weight_bits, symmetric=True, per_channel=per_channel
        )

        # Initialize V from fractional part of w/s
        v_param = init_v_from_weights(weight, weight_scale)

        return QuantizedLinear(
            weight=weight,
            bias=bias,
            weight_scale=weight_scale,
            v_param=v_param,
            weight_bits=weight_bits,
            act_bits=act_bits,
            act_symmetric=act_symmetric,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Fake-quantized forward pass.

        Quantizes INPUT activation (if act_quantizer enabled), then
        quantizes weight via AdaRound, then matmul with both quantized operands.
        """
        # 1. Quantize input activation (disabled during Step 1, enabled after Step 2)
        if self.act_quantizer is not None:
            x_q = self.act_quantizer(x)
        else:
            x_q = x

        # 2. Quantize weight via AdaRound
        if self._frozen:
            w_q = adaround_hard_quantize(
                self.weight, self.v_param, self.weight_scale, self.weight_bits
            )
        else:
            w_q = adaround_quantize(
                self.weight, self.v_param, self.weight_scale, self.weight_bits
            )

        # 3. Matmul with both quantized operands + bias
        out = x_q @ w_q.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def freeze_rounding(self):
        """Freeze V parameters: soft rounding -> hard rounding (threshold at 0.5)."""
        self._frozen = True

    def trainable_parameters(self) -> dict:
        """Return only the V parameters for AdaRound optimization."""
        return {"v_param": self.v_param}
