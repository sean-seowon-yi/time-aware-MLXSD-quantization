Summary: W4A8 (int8 activations + 4-bit weights) path added to MLX Metal backend

What was added
- New core op: quantized_matmul_a8(x_q, x_scales, w_q, scales, biases, transpose=True)
- New primitive: QuantizedMatmulA8 (Metal GPU eval; CPU/CUDA throw)
- New Metal kernel: affine_qmm_t_a8 (int8 activations, affine 4-bit weights)
- New Python layer: W4A8Linear
- Linear.to_quantized(quantize_input=True, mode="affine") now maps to W4A8Linear

Key behavior/limits
- Metal only for fast path; non-Metal falls back to dequantize-then-matmul
- Affine weights only; biases required (from mx.quantize)
- Transpose=True only (x @ w.T) for now
- Activations: symmetric int8, per-group scale
- Group size must divide K and should be >= 32 (BK=32 kernel)
- Kernel uses float MMA (activations dequantized in-kernel); no integer MAC on Apple GPU

How to use (DiffusionKit)
- Enable W4A8 for Linear layers:
  nn.quantize(
      model,
      group_size=64,  # or 32/128
      bits=4,
      mode="affine",
      quantize_input=True,
      class_predicate=lambda _, m: isinstance(m, nn.Linear),
  )

Files touched
- mlx/mlx/ops.h
- mlx/mlx/ops.cpp
- mlx/mlx/primitives.h
- mlx/mlx/primitives.cpp
- mlx/mlx/export.cpp
- mlx/mlx/backend/metal/quantized.cpp
- mlx/mlx/backend/metal/kernels/quantized.h
- mlx/mlx/backend/metal/kernels/quantized.metal
- mlx/mlx/backend/cpu/quantized.cpp
- mlx/mlx/backend/cuda/quantized/quantized.cpp
- mlx/mlx/backend/no_cpu/primitives.cpp
- mlx/mlx/backend/no_gpu/primitives.cpp
- mlx/python/src/ops.cpp
- mlx/python/mlx/nn/layers/quantized.py
- mlx/python/mlx/nn/layers/linear.py
- mlx/python/mlx/nn/layers/__init__.py

Notes
- Not tested.
- If you want better perf, next steps include QMV (vector) A8 kernels, qmm_n (non-transpose) support, and fused quantization of activations.
