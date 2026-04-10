"""Per-layer activation alpha search against fixed RTN W4 weights.

Loads the Phase 2 W4A8 checkpoint, poly schedule, and calibration prompts.
The denoiser runs unchanged Phase 2 W4A8 modules; the search minimizes MSE
between a poly-scaled int8 activation **proxy** (``@ W_dequant``) and a
**reference linear** whose matmul uses **FP16 activations × FP16 weights**
(and FP16 bias add), then compares in float32 for MSE. When
``fp_pre_rtn_weights.npz`` is present, ``W_ref`` is the post-CSB float
weights captured before RTN (stored FP32, cast to FP16 for the reference);
otherwise ``W_ref`` is dequantized W4. Pass ``--reference-fp32`` for a full
FP32 reference matmul instead.

- **RTN / Phase 2:** :func:`extract_rtn_weight_dequant_map` + :func:`collect_alpha_mse_global`.

CLI::

    python -m src.phase4_1.alpha_search \\
        --quantized-dir quantized/<tag>/ \\
        --prompts-file src/settings/coco_100_calibration_prompts.txt

Writes ``alpha_search_results.json`` and, by default, merges ``alpha_multiplier``
into ``poly_schedule.json`` (backup ``*.pre_alpha_search.bak``). Then call
``load_quantized_model_poly`` to obtain the final σ-aware W4A8 model.

**Resume:** pass ``--checkpoint PATH`` (state saved after each prompt) or
``--resume`` (default path ``alpha_search_checkpoint.json`` under the quantized
directory). Re-run with the same flags and prompts file; mismatch raises a clear
error. The checkpoint is deleted after a full successful run unless
``--keep-checkpoint``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .utils import (
    _get_block_linears,
    _reset_modulation_cache,
    _set_nested,
    full_path_to_poly_key,
    get_poly_alpha_raw,
    load_prompt_file,
)
from ..phase3.poly_clipping import POLY_SCHEDULE_FILENAME

logger = logging.getLogger(__name__)

ALPHA_SEARCH_RESULTS_FILENAME = "alpha_search_results.json"
ALPHA_SEARCH_CHECKPOINT_FILENAME = "alpha_search_checkpoint.json"
# Bump when checkpoint semantics change (invalidates old resume files).
ALPHA_CHECKPOINT_VERSION = 4
# MSE target for alpha search: linear ref (no A8 fake-quant on ref path); W_ref from snapshot or dequant W.
ALPHA_MSE_REFERENCE = "fp_linear"
# Reference matmul precision (stored in checkpoint JSON).
REF_LINEAR_FP16 = "fp16"
REF_LINEAR_FP32 = "fp32"
DEFAULT_PROMPTS_FILE = Path("src/settings/coco_100_calibration_prompts.txt")

# Symmetric int8 fake-quant range — must match W4A8StaticLinear / W4A8PolyLinear (mlx.clip …, -128, 127).
_INT8_QMIN, _INT8_QMAX = -128, 127

# -----------------------------------------------------------------------------
# Poly schedule + pipeline
# -----------------------------------------------------------------------------


def load_poly_schedule(quantized_dir: Path) -> dict:
    """Load ``poly_schedule.json`` from a Phase 2/3 checkpoint directory if present."""
    poly_path = Path(quantized_dir) / POLY_SCHEDULE_FILENAME
    if poly_path.exists():
        with open(poly_path) as f:
            schedule = json.load(f)
        logger.info(
            "Loaded poly schedule: %d layers (version: %s)",
            len(schedule.get("layers", {})),
            schedule.get("version"),
        )
        return schedule
    logger.warning(
        "No poly_schedule.json in %s — alpha search needs per-layer poly entries "
        "or pass static_alphas; results may default to multiplier=1.0",
        quantized_dir,
    )
    return {"layers": {}}


def build_pipeline():
    """Construct ``DiffusionPipeline`` using Phase 2 config."""
    from ..phase2.config import DIFFUSIONKIT_SRC, MODEL_VERSION, PIPELINE_KWARGS

    sys.path.insert(0, DIFFUSIONKIT_SRC)
    from diffusionkit.mlx import DiffusionPipeline

    return DiffusionPipeline(**PIPELINE_KWARGS, model_version=MODEL_VERSION)


def build_denoiser(pipeline):
    """CFGDenoiser wrapper."""
    from diffusionkit.mlx import CFGDenoiser

    return CFGDenoiser(pipeline)


def load_phase2_checkpoint(pipeline, quantized_dir: Path) -> dict:
    """Load W4A8 weights from ``quantized_dir``; return ``quantize_config.json`` dict."""
    from ..phase2.config import QUANTIZE_CONFIG_FILENAME
    from ..phase2.quantize_static import load_quantized_model_static

    meta_path = quantized_dir / QUANTIZE_CONFIG_FILENAME
    meta = json.loads(meta_path.read_text())
    load_quantized_model_static(pipeline, quantized_dir)
    logger.info(
        "Loaded Phase 2 static checkpoint (mode=%s, granularity=%s)",
        meta.get("static_mode"),
        meta.get("static_granularity"),
    )
    return meta


# -----------------------------------------------------------------------------
# Weight extraction (RTN / Phase 2)
# -----------------------------------------------------------------------------

WeightResult = np.ndarray
"""Dequantized W (float32) from RTN / Phase 2 checkpoint."""


def extract_rtn_weight_dequant_map(pipeline) -> Dict[str, np.ndarray]:
    """``{poly_key: W_dequant}`` from a pipeline already loaded with Phase 2 W4A8."""
    mmdit = pipeline.mmdit
    out: Dict[str, np.ndarray] = {}

    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            ql = layer.qlinear if hasattr(layer, "qlinear") else layer
            if not isinstance(ql, nn.QuantizedLinear):
                continue
            W_deq = mx.dequantize(
                ql.weight,
                ql.scales,
                ql.biases,
                bits=ql.bits,
                group_size=ql.group_size,
            )
            poly_key = full_path_to_poly_key(block_idx, full_path)
            out[poly_key] = np.asarray(W_deq, dtype=np.float32)

    return out


def _dequant_weight(entry: WeightResult) -> np.ndarray:
    return np.asarray(entry, dtype=np.float32)


_ALPHA_CANDIDATES = [
    0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0,
    3.0, 4.0, 5.0, 6, 8, 10,
]


def _eval_poly_raw_preserving_channels(
    poly_entry: dict, sigma: float
) -> Union[float, np.ndarray]:
    """Evaluate poly α(σ) without floor, preserving per-channel shape.

    Unlike :func:`get_poly_alpha_raw` (which reduces per-channel to scalar
    via ``max``), this returns a 1-D array of shape ``[d_in]`` when the
    schedule entry has per-channel coefficients.  Per-tensor entries return
    a plain float.
    """
    coeffs = poly_entry["coeffs"]
    if isinstance(coeffs[0], (list, np.ndarray)):
        arr = np.asarray(coeffs, dtype=np.float64)
        return np.polynomial.polynomial.polyval(sigma, arr.T).astype(np.float32)
    return float(np.polynomial.polynomial.polyval(sigma, coeffs))


class _AlphaAccumulator:
    """Accumulates MSE over ``alpha_multiplier`` candidates (**MLX GPU**).

    The **denoiser still runs** through ``wrapped`` (Phase 2 W4A8) so the
    activation trajectory matches deployment. The MSE objective compares the
    poly-scaled int8 **proxy** (``x_sub`` → fake-quant → ``@ W_dequant``) to a
    **float linear reference** with the same balanced activations and
    ``W_ref`` / ``bias_ref``—**no activation fake-quant** on the reference path.

    All heavy computation (fake-quant, matmuls, SE reduction) runs on the
    Metal GPU via MLX — **no** ``np.asarray`` sync in the hot path.  The SE
    accumulator ``_total_se`` is an MLX float32 array that must be evaluated
    with ``mx.eval`` at the end of each denoising step (alongside the latent
    ``x``) to keep the lazy-evaluation graph bounded.

    Weights (``_W_q_dequant_mx``, ``_W_ref_mx``) are stored in **FP16** to
    halve memory and eliminate per-call dtype casts.  Both the reference and
    proxy matmuls execute in FP16 on Metal (2× throughput vs FP32); the SE
    diff is widened to FP32 before squaring.
    """

    _ALPHA_SCALES_MX = mx.array(_ALPHA_CANDIDATES, dtype=mx.float32)

    def __init__(
        self,
        wrapped,
        W_q_dequant_mx: mx.array,
        bias_mx: Optional[mx.array],
        b_inv_mx: Optional[mx.array],
        poly_entry: Optional[dict] = None,
        static_alpha: Optional[float] = None,
        subsample_rows: int = 128,
        *,
        W_ref_mx: Optional[mx.array] = None,
        bias_ref_mx: Optional[mx.array] = None,
        w_ref_is_fp_snapshot: bool = False,
        reference_fp16: bool = True,
    ):
        self._wrapped = wrapped
        self._W_q_dequant_mx = W_q_dequant_mx.astype(mx.float16)
        self._bias_mx = bias_mx
        if W_ref_mx is None:
            self._W_ref_mx = self._W_q_dequant_mx
            self._bias_ref_mx = self._bias_mx
            self._w_ref_is_fp_snapshot = False
        else:
            self._W_ref_mx = W_ref_mx.astype(mx.float16)
            self._bias_ref_mx = bias_ref_mx
            self._w_ref_is_fp_snapshot = bool(w_ref_is_fp_snapshot)
        self._reference_fp16 = bool(reference_fp16)
        self._b_inv_mx = b_inv_mx
        self._poly_entry = poly_entry
        self._static_alpha = static_alpha
        self._sigma: Optional[float] = None
        self._subsample_rows = subsample_rows

        n_cands = len(_ALPHA_CANDIDATES)
        self._total_se = mx.zeros((n_cands,), dtype=mx.float32)
        self._total_elements = 0

    # ------------------------------------------------------------------

    def __call__(self, x):
        out = self._wrapped(x)

        if self._static_alpha is not None:
            raw_alpha: Union[float, np.ndarray] = float(self._static_alpha)
        elif self._poly_entry is not None and self._sigma is not None:
            raw_alpha = _eval_poly_raw_preserving_channels(
                self._poly_entry, self._sigma
            )
        else:
            return out

        # Subsample in MLX (stays on GPU).
        x_flat = x.reshape(-1, x.shape[-1])
        n_rows = x_flat.shape[0]
        if n_rows > self._subsample_rows:
            step = max(1, n_rows // self._subsample_rows)
            x_sub = x_flat[::step][: self._subsample_rows]
        else:
            x_sub = x_flat

        if self._b_inv_mx is not None:
            x_sub = x_sub * self._b_inv_mx

        # --- Reference matmul (FP16 or FP32, on GPU) ---
        # _W_ref_mx is already stored as FP16.
        if self._reference_fp16:
            x_h = x_sub.astype(mx.float16)
            o_ref = (x_h @ self._W_ref_mx.T).astype(mx.float16)
            if self._bias_ref_mx is not None:
                o_ref = (o_ref + self._bias_ref_mx.astype(mx.float16)).astype(
                    mx.float16
                )
            o_ref = o_ref.astype(mx.float32)
        else:
            o_ref = x_sub.astype(mx.float32) @ self._W_ref_mx.astype(mx.float32).T
            if self._bias_ref_mx is not None:
                o_ref = o_ref + self._bias_ref_mx

        rows_actual = x_sub.shape[0]
        d_out = self._W_ref_mx.shape[0]
        self._total_elements += rows_actual * d_out

        # --- Proxy: vectorised over all alpha candidates (FP16 matmul) ---
        if isinstance(raw_alpha, np.ndarray) and raw_alpha.ndim == 1:
            ra_mx = mx.array(raw_alpha, dtype=mx.float32)
            eff_alpha = mx.maximum(
                self._ALPHA_SCALES_MX[:, None] * ra_mx[None, :], 0.01
            )
            scales_broad = (eff_alpha / 127.0).reshape(
                len(_ALPHA_CANDIDATES), 1, -1
            )
        else:
            eff_alpha = mx.maximum(
                self._ALPHA_SCALES_MX * float(raw_alpha), 0.01
            )
            scales_broad = (eff_alpha / 127.0).reshape(
                len(_ALPHA_CANDIDATES), 1, 1
            )

        x_exp = mx.expand_dims(x_sub, 0)
        x_all = mx.clip(mx.round(x_exp / scales_broad), _INT8_QMIN, _INT8_QMAX) * scales_broad

        # Proxy matmul in FP16 (2× Metal throughput); _W_q_dequant_mx is already FP16.
        y_all = (x_all.astype(mx.float16) @ self._W_q_dequant_mx.T).astype(mx.float32)
        if self._bias_mx is not None:
            y_all = y_all + self._bias_mx

        diff = y_all - mx.expand_dims(o_ref, 0)
        se = mx.sum(diff * diff, axis=(1, 2))
        self._total_se = self._total_se + se

        return out

    # ------------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def set_sigma(self, sigma: float):
        self._sigma = sigma

    def get_best_alpha(self) -> Tuple[float, float]:
        if self._total_elements == 0:
            return 1.0, float("inf")
        se_np = np.asarray(self._total_se, dtype=np.float64)
        mses = se_np / self._total_elements
        best_idx = int(np.argmin(mses))
        return _ALPHA_CANDIDATES[best_idx], float(mses[best_idx])

    def checkpoint_state(self) -> Dict[str, Any]:
        se_np = np.asarray(self._total_se, dtype=np.float64)
        return {
            "total_se": [float(x) for x in se_np.tolist()],
            "total_elements": int(self._total_elements),
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        se = state["total_se"]
        n = len(_ALPHA_CANDIDATES)
        if len(se) != n:
            raise ValueError(
                f"checkpoint total_se length {len(se)} != {n} alpha candidates",
            )
        self._total_se = mx.array(
            np.asarray(se, dtype=np.float32), dtype=mx.float32
        )
        self._total_elements = int(state["total_elements"])


def _accumulator_w_ref_mode(accumulators: Dict[str, _AlphaAccumulator]) -> str:
    """How reference weights were chosen: snapshot-only, dequant-only, or mixed."""
    if not accumulators:
        return "dequant"
    flags = [acc._w_ref_is_fp_snapshot for acc in accumulators.values()]
    if all(flags):
        return "snapshot"
    if not any(flags):
        return "dequant"
    return "mixed"


def _poly_schedule_fingerprint(schedule: dict) -> str:
    payload = json.dumps(schedule, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _poly_source_path(
    quantized_dir: Path, poly_schedule_path: Optional[Path]
) -> str:
    if poly_schedule_path is not None:
        return str(Path(poly_schedule_path).resolve())
    return str((Path(quantized_dir) / POLY_SCHEDULE_FILENAME).resolve())


def _atomic_write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _checkpoint_alpha_candidates_match(stored: Any) -> bool:
    """JSON may use int or float for whole numbers; compare numerically."""
    if not isinstance(stored, list):
        return False
    cur = list(_ALPHA_CANDIDATES)
    if len(stored) != len(cur):
        return False
    for a, b in zip(stored, cur):
        try:
            if float(a) != float(b):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _ckpt_int_eq(raw_val: Any, current: int) -> bool:
    try:
        return int(raw_val) == int(current)
    except (TypeError, ValueError):
        return False


def _ckpt_float_eq(raw_val: Any, current: float) -> bool:
    try:
        return float(raw_val) == float(current)
    except (TypeError, ValueError):
        return False


def _canonical_path_str(p: Union[str, Path]) -> str:
    return str(Path(p).resolve())


def _poly_sources_equivalent(stored: str, current: str) -> bool:
    """Same file after resolve (handles equivalent relative vs absolute paths)."""
    try:
        return Path(stored).resolve() == Path(current).resolve()
    except OSError:
        return stored == current


def _save_alpha_checkpoint(
    path: Path,
    *,
    next_prompt_index: int,
    n_prompts: int,
    accumulators: Dict[str, _AlphaAccumulator],
    w_ref_mode: str,
    ref_linear_dtype: str,
    poly_fp: str,
    poly_source: str,
    prompts_resolved: str,
    quantized_resolved: str,
    num_steps: int,
    cfg_weight: float,
    latent_size: int,
    subsample_rows: int,
) -> None:
    layers = {k: acc.checkpoint_state() for k, acc in accumulators.items()}
    payload = {
        "version": ALPHA_CHECKPOINT_VERSION,
        "mse_reference": ALPHA_MSE_REFERENCE,
        "w_ref_mode": w_ref_mode,
        "ref_linear_dtype": ref_linear_dtype,
        "next_prompt_index": int(next_prompt_index),
        "n_prompts": int(n_prompts),
        "alpha_candidates": list(_ALPHA_CANDIDATES),
        "num_steps": int(num_steps),
        "cfg_weight": float(cfg_weight),
        "latent_size": int(latent_size),
        "subsample_rows": int(subsample_rows),
        "prompts_file": prompts_resolved,
        "quantized_dir": quantized_resolved,
        "poly_source": poly_source,
        "poly_schedule_sha256": poly_fp,
        "layers": layers,
    }
    _atomic_write_json(Path(path), payload)


def _load_alpha_checkpoint_for_resume(
    path: Path,
    *,
    accumulators: Dict[str, _AlphaAccumulator],
    w_ref_mode: str,
    ref_linear_dtype: str,
    prompt_entries: List[Tuple[int, str]],
    poly_schedule: dict,
    quantized_dir: Path,
    prompts_file: Path,
    poly_schedule_path: Optional[Path],
    num_steps: int,
    cfg_weight: float,
    latent_size: int,
    subsample_rows: int,
) -> int:
    """Return index of first prompt to run. Raises ValueError if checkpoint invalid."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Checkpoint is not valid JSON: {e}") from e
    if raw.get("version") != ALPHA_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {raw.get('version')!r} "
            f"(expected {ALPHA_CHECKPOINT_VERSION}; older checkpoints used a "
            f"different MSE target — delete the file and restart alpha search)",
        )
    if raw.get("mse_reference") != ALPHA_MSE_REFERENCE:
        raise ValueError(
            f"Checkpoint mse_reference={raw.get('mse_reference')!r} != "
            f"{ALPHA_MSE_REFERENCE!r} — delete checkpoint and restart",
        )
    if raw.get("w_ref_mode") != w_ref_mode:
        raise ValueError(
            f"Checkpoint w_ref_mode={raw.get('w_ref_mode')!r} != current {w_ref_mode!r} "
            "(FP snapshot coverage changed vs this run — delete checkpoint and restart)",
        )
    if raw.get("ref_linear_dtype") != ref_linear_dtype:
        raise ValueError(
            f"Checkpoint ref_linear_dtype={raw.get('ref_linear_dtype')!r} != current "
            f"{ref_linear_dtype!r} — delete checkpoint and restart (or match --reference-fp32)",
        )
    if not _checkpoint_alpha_candidates_match(raw.get("alpha_candidates")):
        raise ValueError("Checkpoint alpha_candidates do not match this code version")
    try:
        n = int(raw["n_prompts"])
        start = int(raw["next_prompt_index"])
    except KeyError as e:
        raise ValueError(f"Checkpoint missing required field: {e.args[0]}") from e
    if n != len(prompt_entries):
        raise ValueError(
            f"Checkpoint n_prompts={n} but current run has {len(prompt_entries)} prompts",
        )
    pr = _canonical_path_str(prompts_file)
    st_pf = raw.get("prompts_file")
    if not st_pf:
        raise ValueError("Checkpoint missing prompts_file")
    if _canonical_path_str(st_pf) != pr:
        raise ValueError(
            f"Checkpoint prompts_file mismatch:\n  checkpoint: {st_pf}\n  current:    {pr}",
        )
    qr = _canonical_path_str(quantized_dir)
    st_qd = raw.get("quantized_dir")
    if not st_qd:
        raise ValueError("Checkpoint missing quantized_dir")
    if _canonical_path_str(st_qd) != qr:
        raise ValueError(
            f"Checkpoint quantized_dir mismatch:\n  checkpoint: {st_qd}\n  current:    {qr}",
        )
    ps = _poly_source_path(quantized_dir, poly_schedule_path)
    st_poly = raw.get("poly_source")
    if not st_poly:
        raise ValueError("Checkpoint missing poly_source")
    if not _poly_sources_equivalent(str(st_poly), ps):
        raise ValueError(
            f"Checkpoint poly_source mismatch:\n  checkpoint: {st_poly}\n  current:    {ps}",
        )
    fp = _poly_schedule_fingerprint(poly_schedule)
    if raw.get("poly_schedule_sha256") != fp:
        raise ValueError(
            "Checkpoint poly_schedule_sha256 does not match current poly schedule "
            "(regenerate schedule or use the same poly_schedule.json).",
        )
    if not _ckpt_int_eq(raw.get("num_steps"), num_steps):
        raise ValueError(
            f"Checkpoint num_steps={raw.get('num_steps')!r} != current run {num_steps!r}",
        )
    if not _ckpt_float_eq(raw.get("cfg_weight"), cfg_weight):
        raise ValueError(
            f"Checkpoint cfg_weight={raw.get('cfg_weight')!r} != current run {cfg_weight!r}",
        )
    if not _ckpt_int_eq(raw.get("latent_size"), latent_size):
        raise ValueError(
            f"Checkpoint latent_size={raw.get('latent_size')!r} != current run {latent_size!r}",
        )
    if not _ckpt_int_eq(raw.get("subsample_rows"), subsample_rows):
        raise ValueError(
            f"Checkpoint subsample_rows={raw.get('subsample_rows')!r} "
            f"!= current run {subsample_rows!r}",
        )
    saved_layers = raw.get("layers") or {}
    if set(saved_layers) != set(accumulators):
        raise ValueError(
            f"Checkpoint layer keys differ from current model "
            f"(checkpoint {len(saved_layers)} vs current {len(accumulators)})",
        )
    if start < 0 or start > n:
        raise ValueError(f"Invalid next_prompt_index {start} for n_prompts={n}")
    for k, acc in accumulators.items():
        acc.restore_checkpoint_state(saved_layers[k])
    return start


def _linear_bias_mx(layer) -> Optional[mx.array]:
    ql = layer.qlinear if hasattr(layer, "qlinear") else layer
    if hasattr(ql, "bias") and ql.bias is not None:
        return ql.bias.astype(mx.float32)
    return None


def _layer_b_inv_mx(layer) -> Optional[mx.array]:
    if hasattr(layer, "b_inv"):
        return layer.b_inv.astype(mx.float32)
    return None


def install_alpha_accumulators(
    mmdit,
    poly_schedule: dict,
    weight_results: Mapping[str, WeightResult],
    static_alphas: Optional[Dict[str, float]] = None,
    subsample_rows: int = 128,
    fp_pre_rtn: Optional[Mapping[str, Tuple[np.ndarray, np.ndarray]]] = None,
    reference_fp16: bool = True,
) -> Dict[str, _AlphaAccumulator]:
    """Install accumulators on block linears listed in ``weight_results``."""
    layers_dict = poly_schedule.get("layers", {})
    accumulators: Dict[str, _AlphaAccumulator] = {}

    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, layer in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            if poly_key not in weight_results:
                continue

            W_q_np = _dequant_weight(weight_results[poly_key])
            W_q_mx = mx.array(W_q_np, dtype=mx.float16)
            bias_mx = _linear_bias_mx(layer)
            b_inv_mx = _layer_b_inv_mx(layer)

            W_ref_mx: Optional[mx.array] = None
            bias_ref_mx: Optional[mx.array] = None
            w_snap = False
            if fp_pre_rtn:
                tup = fp_pre_rtn.get(poly_key)
                if tup is not None:
                    W_fp, b_fp = tup
                    W_fp = np.asarray(W_fp, dtype=np.float32)
                    b_fp = np.asarray(b_fp, dtype=np.float32)
                    if W_fp.shape == W_q_np.shape:
                        W_ref_mx = mx.array(W_fp, dtype=mx.float16)
                        bias_ref_mx = mx.array(b_fp, dtype=mx.float32)
                        w_snap = True
                    else:
                        logger.warning(
                            "FP pre-RTN snapshot shape mismatch for %s %s vs %s — dequant ref",
                            poly_key,
                            W_fp.shape,
                            W_q_np.shape,
                        )

            poly_entry = layers_dict.get(poly_key) if static_alphas is None else None
            static_alpha = static_alphas.get(poly_key) if static_alphas else None

            acc = _AlphaAccumulator(
                layer,
                W_q_mx,
                bias_mx,
                b_inv_mx=b_inv_mx,
                poly_entry=poly_entry,
                static_alpha=static_alpha,
                subsample_rows=subsample_rows,
                W_ref_mx=W_ref_mx,
                bias_ref_mx=bias_ref_mx,
                w_ref_is_fp_snapshot=w_snap,
                reference_fp16=reference_fp16,
            )
            _set_nested(block, full_path, acc)
            accumulators[poly_key] = acc

    return accumulators


def remove_alpha_accumulators(
    mmdit, accumulators: Dict[str, _AlphaAccumulator]
) -> None:
    for block_idx, block in enumerate(mmdit.multimodal_transformer_blocks):
        for full_path, _ in _get_block_linears(block, is_mm=True):
            poly_key = full_path_to_poly_key(block_idx, full_path)
            if poly_key in accumulators:
                _set_nested(block, full_path, accumulators[poly_key]._wrapped)


def collect_alpha_mse_global(
    pipeline,
    denoiser,
    prompt_entries: List[Tuple[int, str]],
    poly_schedule: dict,
    weight_results: Mapping[str, WeightResult],
    num_steps: int,
    cfg_weight: float = 4.0,
    latent_size: int = 64,
    static_alphas: Optional[Dict[str, float]] = None,
    subsample_rows: int = 128,
    *,
    quantized_dir: Optional[Path] = None,
    prompts_file: Optional[Path] = None,
    poly_schedule_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    keep_checkpoint: bool = False,
    reference_fp16: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """Run calibration prompts; return ``{poly_key: (best_multiplier, mse)}``.

    If *checkpoint_path* is set, state is written after each prompt. If that file
    already exists at entry, load it and resume from ``next_prompt_index``.
    On a full successful run the checkpoint is removed unless *keep_checkpoint*.

    *reference_fp16*
        If True (default), MSE reference is ``matmul(fp16(x_bal), fp16(W_ref).T)``
        (+ bias in FP16). If False, reference uses FP32 matmul.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    mmdit = pipeline.mmdit
    fp_pre_rtn = None
    if quantized_dir is not None:
        from ..phase2.fp_snapshot import load_fp_pre_rtn_weights

        fp_pre_rtn = load_fp_pre_rtn_weights(Path(quantized_dir))
        if fp_pre_rtn:
            logger.info(
                "Loaded FP pre-RTN snapshot: %d layer tensors (reference W where keys match)",
                len(fp_pre_rtn),
            )

    accumulators = install_alpha_accumulators(
        mmdit,
        poly_schedule,
        weight_results,
        static_alphas=static_alphas,
        subsample_rows=subsample_rows,
        fp_pre_rtn=fp_pre_rtn,
        reference_fp16=reference_fp16,
    )
    w_ref_mode = _accumulator_w_ref_mode(accumulators)
    ref_linear_dtype = REF_LINEAR_FP16 if reference_fp16 else REF_LINEAR_FP32
    logger.info(
        "Alpha search reference: w_ref_mode=%s, ref_linear_dtype=%s",
        w_ref_mode,
        ref_linear_dtype,
    )

    n_layers = len(accumulators)
    n_prompts = len(prompt_entries)
    poly_fp = _poly_schedule_fingerprint(poly_schedule)

    start_idx = 0
    cp: Optional[Path] = Path(checkpoint_path).resolve() if checkpoint_path else None
    if cp is not None:
        if quantized_dir is None or prompts_file is None:
            raise ValueError(
                "checkpoint_path requires quantized_dir and prompts_file for validation",
            )
        qdir = Path(quantized_dir).resolve()
        if cp.is_file():
            start_idx = _load_alpha_checkpoint_for_resume(
                cp,
                accumulators=accumulators,
                w_ref_mode=w_ref_mode,
                ref_linear_dtype=ref_linear_dtype,
                prompt_entries=prompt_entries,
                poly_schedule=poly_schedule,
                quantized_dir=qdir,
                prompts_file=Path(prompts_file),
                poly_schedule_path=poly_schedule_path,
                num_steps=num_steps,
                cfg_weight=cfg_weight,
                latent_size=latent_size,
                subsample_rows=subsample_rows,
            )
            logger.info(
                "Resuming alpha search: prompt %d / %d (checkpoint %s)",
                start_idx,
                n_prompts,
                cp,
            )
        else:
            logger.info("Starting alpha search with new checkpoint file %s", cp)

    logger.info(
        "Alpha search: %d layers, %d prompts, %d candidates/layer%s",
        n_layers,
        n_prompts,
        len(_ALPHA_CANDIDATES),
        f", checkpoint={cp}" if cp else "",
    )

    if start_idx > n_prompts:
        raise ValueError(f"Invalid start_idx {start_idx} > n_prompts {n_prompts}")
    if start_idx == n_prompts:
        logger.info("Checkpoint indicates all %d prompts already processed", n_prompts)

    remaining = prompt_entries[start_idx:]
    if tqdm is not None and remaining:
        iterator = tqdm(
            remaining,
            desc="Alpha search prompts",
            initial=start_idx,
            total=n_prompts,
        )
    else:
        iterator = remaining

    qdir_r = Path(quantized_dir).resolve() if quantized_dir else Path(".")
    pf_r = str(Path(prompts_file).resolve()) if prompts_file else ""
    poly_src = _poly_source_path(qdir_r, poly_schedule_path)

    for offset, (seed, prompt) in enumerate(iterator):
        idx = start_idx + offset
        conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
        mx.eval(conditioning, pooled)

        sigmas = pipeline.get_sigmas(pipeline.sampler, num_steps)
        timesteps = pipeline.sampler.timestep(sigmas).astype(
            pipeline.activation_dtype
        )
        denoiser.cache_modulation_params(pooled, timesteps)

        mx.random.seed(seed)
        latent_shape = (1, latent_size, latent_size, 16)
        noise = mx.random.normal(latent_shape).astype(pipeline.activation_dtype)
        x = pipeline.sampler.noise_scaling(
            sigmas[0],
            noise,
            mx.zeros(latent_shape),
            pipeline.max_denoise(sigmas),
        )
        mx.eval(x)

        for i in range(len(sigmas) - 1):
            sigma_val = float(sigmas[i])
            for acc in accumulators.values():
                acc.set_sigma(sigma_val)

            denoised = denoiser(
                x,
                timesteps[i],
                sigmas[i],
                conditioning=conditioning,
                cfg_weight=cfg_weight,
            )
            d = (x - denoised) / sigmas[i]
            x = x + d * (sigmas[i + 1] - sigmas[i])
            # Evaluate both the latent AND all SE accumulators so the
            # lazy graph is trimmed each step (keeps memory bounded).
            mx.eval(x, *[acc._total_se for acc in accumulators.values()])

        _reset_modulation_cache(pipeline)

        if cp is not None:
            _save_alpha_checkpoint(
                cp,
                next_prompt_index=idx + 1,
                n_prompts=n_prompts,
                accumulators=accumulators,
                w_ref_mode=w_ref_mode,
                ref_linear_dtype=ref_linear_dtype,
                poly_fp=poly_fp,
                poly_source=poly_src,
                prompts_resolved=pf_r,
                quantized_resolved=str(qdir_r),
                num_steps=num_steps,
                cfg_weight=cfg_weight,
                latent_size=latent_size,
                subsample_rows=subsample_rows,
            )

    if cp is not None and not keep_checkpoint and cp.is_file():
        cp.unlink()
        logger.info("Removed checkpoint %s (run complete)", cp)

    results = {k: acc.get_best_alpha() for k, acc in accumulators.items()}
    remove_alpha_accumulators(mmdit, accumulators)
    return results


# -----------------------------------------------------------------------------
# High-level entry
# -----------------------------------------------------------------------------


def run_alpha_search_on_checkpoint(
    quantized_dir: Path,
    prompts_file: Path,
    *,
    num_prompts: int = 16,
    num_steps: int = 30,
    cfg_weight: float = 4.0,
    latent_size: int = 64,
    subsample_rows: int = 128,
    poly_schedule_path: Optional[Path] = None,
    weight_results: Optional[Mapping[str, WeightResult]] = None,
    static_alphas: Optional[Dict[str, float]] = None,
    merge_poly_schedule: bool = True,
    poly_schedule_backup: bool = True,
    checkpoint_path: Optional[Path] = None,
    keep_checkpoint: bool = False,
    reference_fp16: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """End-to-end alpha search: load Phase 2 W4A8, poly schedule, run collection.

    When ``merge_poly_schedule`` is True (default), writes updated
    ``poly_schedule.json`` with per-layer ``alpha_multiplier`` so
    :func:`src.phase3.quantize_poly.load_quantized_model_poly` builds the final
    σ-aware W4A8 model (RTN weights unchanged).

    Parameters
    ----------
    quantized_dir
        Directory with ``quantize_config.json`` and ``mmdit_quantized.safetensors``.
    prompts_file
        Tab-separated ``seed<TAB>prompt`` file.
    poly_schedule_path
        Optional JSON override; default is ``quantized_dir / poly_schedule.json``.
    weight_results
        Optional precomputed weights; default is RTN map from the loaded checkpoint.
    merge_poly_schedule
        If True, merge ``alpha_multiplier`` into ``quantized_dir/poly_schedule.json``.
    poly_schedule_backup
        If True, copy existing ``poly_schedule.json`` to ``*.pre_alpha_search.bak``.
    checkpoint_path
        If set, save/load per-prompt state here for resume. If the file exists at
        start, resume from ``next_prompt_index`` after validating hyperparameters
        and layer keys.
    keep_checkpoint
        If False (default), delete *checkpoint_path* after a full successful run.
    reference_fp16
        If True (default), MSE reference linear uses FP16 matmul on ``x_bal`` and ``W_ref``.
    """
    quantized_dir = Path(quantized_dir).resolve()
    t0 = time.time()
    pipeline = build_pipeline()
    logger.info("Built pipeline in %.1fs", time.time() - t0)

    t1 = time.time()
    load_phase2_checkpoint(pipeline, quantized_dir)
    logger.info("Loaded checkpoint in %.1fs", time.time() - t1)

    if poly_schedule_path is not None:
        with open(poly_schedule_path) as f:
            poly_schedule = json.load(f)
        logger.info("Loaded poly schedule from %s", poly_schedule_path)
    else:
        poly_schedule = load_poly_schedule(quantized_dir)

    if weight_results is None:
        weight_results = extract_rtn_weight_dequant_map(pipeline)

    prompt_entries = load_prompt_file(Path(prompts_file))[:num_prompts]
    logger.info("Using %d prompt(s) from %s", len(prompt_entries), prompts_file)

    denoiser = build_denoiser(pipeline)
    results = collect_alpha_mse_global(
        pipeline,
        denoiser,
        prompt_entries,
        poly_schedule,
        weight_results,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        latent_size=latent_size,
        static_alphas=static_alphas,
        subsample_rows=subsample_rows,
        quantized_dir=quantized_dir,
        prompts_file=Path(prompts_file),
        poly_schedule_path=poly_schedule_path,
        checkpoint_path=checkpoint_path,
        keep_checkpoint=keep_checkpoint,
        reference_fp16=reference_fp16,
    )

    if merge_poly_schedule:
        if not poly_schedule.get("layers"):
            logger.warning(
                "merge_poly_schedule set but no poly layers loaded — "
                "skipping %s update",
                POLY_SCHEDULE_FILENAME,
            )
        else:
            if poly_schedule_path is not None:
                alt = Path(poly_schedule_path).resolve()
                if alt.parent != quantized_dir:
                    logger.warning(
                        "Merged schedule is written to %s (not alongside --poly-schedule %s)",
                        quantized_dir / POLY_SCHEDULE_FILENAME,
                        alt,
                    )
            save_poly_schedule_with_alpha_multipliers(
                quantized_dir,
                poly_schedule,
                results,
                backup=poly_schedule_backup,
            )

    return results


def save_alpha_search_results(
    path: Path,
    results: Dict[str, Tuple[float, float]],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write JSON: ``layers`` map ``poly_key -> {multiplier, mse}``."""
    path = Path(path)
    layers_out: Dict[str, Dict[str, float]] = {}
    for k, (mult, mse) in results.items():
        layers_out[k] = {"multiplier": float(mult), "mse": float(mse)}

    payload: Dict[str, Any] = {
        "version": 1,
        "n_layers": len(layers_out),
        "alpha_candidates": list(_ALPHA_CANDIDATES),
        "layers": layers_out,
    }
    if extra_meta:
        payload["meta"] = extra_meta

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote alpha search results to %s", path)


def merge_alpha_multipliers_into_poly_schedule(
    poly_schedule: dict,
    results: Dict[str, Tuple[float, float]],
) -> dict:
    """Return a deep copy of *poly_schedule* with ``alpha_multiplier`` / ``alpha_search_mse`` set.

    :func:`load_quantized_model_poly` applies ``alpha_multiplier`` inside
    :class:`~src.phase3.quantize_poly.W4A8PolyLinear` so the checkpoint + schedule
    define the final σ-aware W4A8 path after Phase 4.1.
    """
    out = copy.deepcopy(poly_schedule)
    layers = out.setdefault("layers", {})
    for key, (mult, mse) in results.items():
        if key not in layers:
            continue
        layers[key]["alpha_multiplier"] = float(mult)
        layers[key]["alpha_search_mse"] = float(mse)
    return out


def save_poly_schedule_with_alpha_multipliers(
    quantized_dir: Path,
    poly_schedule: dict,
    results: Dict[str, Tuple[float, float]],
    *,
    backup: bool = True,
) -> Path:
    """Write ``poly_schedule.json`` under *quantized_dir* with merged multipliers."""
    qdir = Path(quantized_dir).resolve()
    path = qdir / POLY_SCHEDULE_FILENAME
    merged = merge_alpha_multipliers_into_poly_schedule(poly_schedule, results)
    if backup and path.exists():
        bak = path.with_suffix(".json.pre_alpha_search.bak")
        shutil.copy2(path, bak)
        logger.info("Backed up previous poly schedule to %s", bak)
    path.write_text(json.dumps(merged, indent=2))
    logger.info(
        "Updated %s with alpha_multiplier for %d layers (run load_quantized_model_poly to deploy)",
        path,
        sum(1 for k in results if k in poly_schedule.get("layers", {})),
    )
    return path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4.1: per-layer activation alpha search (RTN / Phase 2 checkpoint)",
    )
    p.add_argument(
        "--quantized-dir",
        type=Path,
        required=True,
        help="Phase 2 checkpoint (quantize_config.json + weights)",
    )
    p.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help=f"seed<TAB>prompt file (default: {DEFAULT_PROMPTS_FILE})",
    )
    p.add_argument(
        "--poly-schedule",
        type=Path,
        default=None,
        help="Override poly_schedule.json path (default: <quantized-dir>/poly_schedule.json)",
    )
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--cfg-weight", type=float, default=4.0)
    p.add_argument("--latent-size", type=int, default=64)
    p.add_argument("--subsample-rows", type=int, default=128)
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=f"Output path (default: <quantized-dir>/{ALPHA_SEARCH_RESULTS_FILENAME})",
    )
    p.add_argument(
        "--no-merge-poly-schedule",
        action="store_true",
        help="Do not write alpha_multiplier into poly_schedule.json (default: merge)",
    )
    p.add_argument(
        "--no-poly-schedule-backup",
        action="store_true",
        help="Do not backup existing poly_schedule.json before merge",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Save/load resume state after each prompt. If the file exists, resume "
            f"after validation. Default: <quantized-dir>/{ALPHA_SEARCH_CHECKPOINT_FILENAME} "
            "when --resume is set"
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=f"Use <quantized-dir>/{ALPHA_SEARCH_CHECKPOINT_FILENAME} as --checkpoint",
    )
    p.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep checkpoint file after a full successful run (default: delete)",
    )
    p.add_argument(
        "--reference-fp32",
        action="store_true",
        help="Use FP32 matmul for the MSE reference (default: FP16 matmul × FP16 weights)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    prompts_file = args.prompts_file or DEFAULT_PROMPTS_FILE
    if not prompts_file.exists():
        raise SystemExit(f"Prompts file not found: {prompts_file}")

    out_json = args.output_json or (
        Path(args.quantized_dir) / ALPHA_SEARCH_RESULTS_FILENAME
    )

    qdir = Path(args.quantized_dir).resolve()
    checkpoint: Optional[Path] = None
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint).resolve()
    elif args.resume:
        checkpoint = qdir / ALPHA_SEARCH_CHECKPOINT_FILENAME

    logger.info("=" * 60)
    logger.info("Phase 4.1: alpha search")
    logger.info("Quantized dir : %s", args.quantized_dir)
    logger.info("Prompts file  : %s", prompts_file)
    logger.info("Output JSON   : %s", out_json)
    if checkpoint is not None:
        logger.info("Checkpoint    : %s", checkpoint)
    logger.info("=" * 60)

    t0 = time.time()
    try:
        results = run_alpha_search_on_checkpoint(
            args.quantized_dir,
            prompts_file,
            num_prompts=args.num_prompts,
            num_steps=args.num_steps,
            cfg_weight=args.cfg_weight,
            latent_size=args.latent_size,
            subsample_rows=args.subsample_rows,
            poly_schedule_path=args.poly_schedule,
            merge_poly_schedule=not args.no_merge_poly_schedule,
            poly_schedule_backup=not args.no_poly_schedule_backup,
            checkpoint_path=checkpoint,
            keep_checkpoint=args.keep_checkpoint,
            reference_fp16=not args.reference_fp32,
        )
    except ValueError as e:
        # Includes json.JSONDecodeError from a corrupted checkpoint file.
        raise SystemExit(f"Alpha search checkpoint error: {e}") from e
    elapsed = time.time() - t0

    save_alpha_search_results(
        out_json,
        results,
        extra_meta={
            "quantized_dir": str(Path(args.quantized_dir).resolve()),
            "prompts_file": str(Path(prompts_file).resolve()),
            "num_prompts": args.num_prompts,
            "num_steps": args.num_steps,
            "cfg_weight": args.cfg_weight,
            "latent_size": args.latent_size,
            "subsample_rows": args.subsample_rows,
            "reference_fp16": not args.reference_fp32,
            "elapsed_sec": round(elapsed, 2),
        },
    )

    finite = [m for _, m in results.values() if m < float("inf")]
    if finite:
        logger.info(
            "Mean MSE (finite layers): %.6g over %d layers",
            sum(finite) / len(finite),
            len(finite),
        )
    logger.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()
