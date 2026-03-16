# naive_int8 benchmark config

## Goal
Add a `naive_int8` config to `benchmark_model.py` that applies W8A8 quantization
without AdaRound — plain symmetric int8 on weights (via `mx.quantize`) and
dynamic per-tensor int8 fake-quant on activations at inference time.

## Method
1. Walk all `nn.Linear` / `nn.QuantizedLinear` layers in the DiT blocks via
   `_walk_mmdit_linears(mmdit)`.
2. `inject_weights_naive_int8` quantizes each eligible layer's weight tensor
   with `mx.quantize(bits=8)` and replaces it with `nn.QuantizedLinear`.
3. `_DynamicInt8ActLayer` proxy applies fake-quant (round→clip→rescale) on
   the input activation before forwarding to the wrapped layer.
4. `apply_dynamic_int8_act_hooks` wraps every walked layer; cleanup via
   `remove_dynamic_int8_act_hooks`.
5. `_load_pipeline` branches on `config == "naive_int8"`, calls (2) then (3/4).

## Key design decisions
- Skip layers where `in_features < max(128, group_size)` (MLX min-col limit).
- `_DynamicInt8ActLayer` is a plain Python proxy (not `nn.Module`) — MLX stores
  it in `_other` so it fires on forward without disrupting parameter traversal
  (weights are already in the QuantizedLinear).
- Cleanup function stored in `quant_ctx["remove_act_fn"]` to avoid importing
  `load_adaround_model` (unavailable on this branch) for naive_int8 cleanup.

## Files changed
- `src/benchmark_model.py` — 5 new callables, `_load_pipeline` update, argparse
- `tests/test_benchmark_model.py` — `TestInjectWeightsNaiveInt8`, `TestDynamicInt8ActLayer`
