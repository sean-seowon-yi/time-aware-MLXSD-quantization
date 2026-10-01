# Phase 4.1 pipeline: calibration → SSC/CSB → RTN W4A8 → polynomial clipping → alpha search

This document describes an **optional pipeline variant** that stops **before GPTQ (Phase 4)** and instead adds **per-layer activation alpha search** on top of **round-to-nearest (RTN) weight quantization**. Implementation lives primarily in `src/phase4_1/alpha_search.py` (draft); Phases 1–3 follow the existing project layout.

**Execution order vs title:** weights are quantized in **Phase 2**; **Phase 3** fits the poly schedule **from** Phase 1 diagnostics plus the Phase 2 checkpoint (see §1 and §2.5).

---

## 1. End-to-end flow (what runs, in order)

| Step | Phase | Role |
|------|--------|------|
| 1 | **Phase 1** | **Calibrate / diagnose:** run calibration prompts through the FP16 denoiser with hooks. Collect activation trajectories, weight salience, adaLN stats, and configuration needed downstream. |
| 2 | **Phase 2** | **SSC → CSB → RTN W4A8:** SSC-weighted calibration produces statistics; **channel-scale balancing (CSB)** absorbs scaling into adaLN and per-layer balancing where applicable; **RTN** quantizes weights to 4-bit groups and wires **8-bit static activation fake-quantization**. Saves checkpoint + `calibration.npz` / metadata. |
| 3 | **Phase 3** | **Polynomial clipping schedule:** read Phase 1 diagnostics and the Phase 2 calibration directory; fit **per-layer (or hybrid per-channel) polynomials** that map noise level (e.g. σ) to an **activation clipping scale** (“alpha”). Writes `poly_schedule.json` next to the checkpoint. |
| 4 | **Phase 4.1 (alpha search)** | **Calibration pass:** with the **same RTN weights** loaded and the **poly schedule** giving **base α(σ)**, search **per-layer multipliers** so a **poly-style A8 proxy × dequant W** best matches **that layer’s actual W4A8 forward** under sampling (see §5.1). |

**Important ordering note (matches the repo today):** **RTN weight quantization happens inside Phase 2**, not after Phase 3. Phase 3 **depends** on Phase 2 artifacts (`calibration.npz`, balancing metadata, etc.). Conceptually you still get:

**calibrate → SSC/CSB → RTN W4 → (poly fit on those artifacts) → alpha search → W4A8 model**

The **polynomial** is fit **after** the first quantized checkpoint exists; alpha search then **refines how aggressively you quantize activations** relative to that poly, **per layer**, using the **fixed RTN weights**.

### 1.1 Data flow (artifacts)

Clean dependency chain in the repo:

| Stage | Reads | Writes |
|-------|--------|--------|
| **Phase 1** | Calibration prompt file, FP16 model | `diagnostics/activation_stats/*.npz`, weight / adaLN stats, `diagnostics/config.json` |
| **Phase 2** | `diagnostics/` (if collecting or calibrating) | `quantized/<tag>/mmdit_quantized.safetensors`, `quantize_config.json`, **`calibration.npz`**, **`calibration_meta.json`** |
| **Phase 3** | **`diagnostics/`** (`activation_stats`, σ grid) + **`calibration.npz`** / **`calibration_meta.json`** from `--calibration-dir` | **`poly_schedule.json`** (default: next to checkpoint) |
| **Phase 4.1** | Loaded **Phase 2** pipeline (MLX) + **`poly_schedule.json`** + prompt list | **`alpha_search_results.json`**; updated **`poly_schedule.json`** with per-layer **`alpha_multiplier`** (optional backup). Deploy σ-aware W4A8 via **`load_quantized_model_poly`** (same RTN weights, poly A8 + multipliers). |

Phase 3 implementation detail (see `src/phase3/poly_clipping.py`): polynomials are fit to **post-CSB absmax trajectories** — Phase 1 stores **pre-quant FP16** per-timestep channel maxima; Phase 3 combines them with **per-layer balancing vectors `b`** from **`calibration.npz`** so the schedule matches what **CSB-conditioned** activations look like before A8.

---

## 2. Phase 1 — calibration (diagnostic collection)

**Purpose:** Build a **time-resolved picture** of activations and **which layers** drive quantization difficulty (e.g. Spearman ρ across timesteps, weight salience).

**Typical entrypoint:** `python -m src.phase1.run_collection`

**Outputs (under your diagnostics directory, default `diagnostics/`):**

- Activation statistics aligned with layer names and timesteps.
- Weight statistics and adaLN-related stats.
- `config.json` (layer lists, etc.) consumed by Phase 3.

**Why it matters for later stages:** Phase 3 fits schedules from **Phase 1 trajectories** once **Phase 2’s `b` vectors** are applied (post-CSB absmax — see §1.1). SSC in Phase 2 also consumes this data for **time-aware** scale aggregation when using static activations.

---

## 3. Phase 2 — SSC, CSB, and RTN W4A8

**Purpose:** Produce a **first full W4A8 checkpoint**: 4-bit weights + 8-bit activation path, with **CSB** reducing cross-channel dynamic range before quantizing activations.

**Typical entrypoint:** `python -m src.phase2.run_e2e` (collection + quantize in one process), or `run_quantize` if diagnostics already exist.

**Internal ordering (high level):**

1. **SSC calibration** — aggregate activation information in a way that respects **which timesteps matter** (SSC weighting / temperature).
2. **CSB** — absorb channel scaling into **adaLN** and **online `b_inv`** factors on selected projections so the tensor seen by the quantizer is **better conditioned**.
3. **RTN W4 quantization** — group-wise (or as configured) **round-to-nearest** weights; wrap linears as **`W4A8StaticLinear`** with **`nn.QuantizedLinear`** inside.
4. **Save** — `mmdit_quantized.safetensors`, `quantize_config.json`, calibration payloads used by Phase 3.

At the end of Phase 2 you already have a **valid W4A8 model** (the “RTN baseline”). Phase 4.1 does **not** replace this; it proposes **extra per-layer activation scaling knobs** calibrated against that baseline.

---

## 4. Phase 3 — polynomial clipping on activations

**Purpose:** Replace (or augment) a **single global clipping heuristic** with a **schedule α(σ)** per layer (or per-channel coefficients later reduced to a scalar for some consumers), learned from:

- Phase 1 **trajectories**, and  
- Phase 2 **CSB / calibration** artifacts.

**Typical entrypoint:** `python -m src.phase3.generate_schedule --diagnostics-dir … --calibration-dir …`

**Output:** `poly_schedule.json` — for each poly key (`blocks.{i}.{image|text}.…`), polynomial coefficients and metadata so that at inference (or during later calibration) you can evaluate **how wide the activation quantizer’s effective range should be** as noise level changes.

**Relationship to W4A8:** The poly schedule is an **activation-side** artifact. Weights remain the RTN tensors from Phase 2 until you optionally run **GPTQ (Phase 4)** or other refinements.

---

## 5. Phase 4.1 — alpha search (draft implementation)

**Code:** `src/phase4_1/alpha_search.py`

### 5.1 What it optimizes

For each target linear (same enumeration as poly keys / GPTQ tooling):

1. The **forward pass** still uses the **real loaded module** — **`W4A8StaticLinear`** (Phase 2 static A8 + **`nn.QuantizedLinear`**). The denoiser’s activation **trajectory** is therefore the **RTN W4A8** graph, **not** the poly-σ A8 graph you get after `load_quantized_model_poly`.
2. In parallel, on a **CPU NumPy path**, the code:
   - Builds a **reference** output: **FP16 matmul** **`matmul(fp16(x_bal), fp16(W_ref).T)`** plus bias in FP16 (then promoted to float32 for MSE), unless **`--reference-fp32`** is set (full FP32 matmul). **`W_ref`** comes from **`fp_pre_rtn_weights.npz`** (post-CSB float weights saved immediately before RTN) when present; otherwise **`W_ref`** is RTN-dequantized weights (same as the proxy branch).
   - Builds a **proxy** output: symmetric int8-style fake-quant on subsampled **`x_bal`** using scales **`max(α_raw(σ) · m, 0.01) / 127`** for each candidate multiplier **`m`** (see source for the candidate list). Here **`α_raw(σ)`** is the poly schedule value **before** the 0.01 floor — this matches **`W4A8PolyLinear`**, which uses **`max(poly(σ) · multiplier, 0.01)`**, not **`multiplier · max(poly(σ), 0.01)`**.
   - Multiplies fake-quant activations by **`W_{\mathrm{RTN}}`** (dequantized) + current layer bias, and accumulates **MSE(proxy − reference)**.

After all calibration prompts and denoising steps, each layer picks the **multiplier with lowest average MSE**.

So alpha search is **layer-local**: it tunes **`m`** so a **poly-style** int8 activation proxy (with RTN dequant matmul) tracks a **low-precision linear reference** (default **FP16** matmul on activations and weights) — **not** full end-to-end FP16 activations from an all-FP16 teacher, and **not** minimizing error to the **wrapped** W4A8 output `out = wrapped(x)`. Inputs **`x_bal`** are still whatever the Phase 2 W4A8 graph produces upstream.

**`b_inv` (CSB):** Applied on the NumPy side before both reference and proxy, matching **`W4A8StaticLinear`**.

**Per-token A8 layers** (high-ρ): Skipped — the live forward uses **per-row** scales; the proxy assumes a **single** poly-based scale per tensor.

### 5.2 Memory and compute characteristics

- **No huge activation cache:** only **running sums** of squared error per candidate (~17 floats per layer in the default list).
- **Row subsampling** (default 128 rows per forward) keeps the NumPy matmul cheap.
- **Vectorization:** all candidates are evaluated in **one batched matmul** per hook invocation.

### 5.3 API sketch (RTN path)

```text
extract_rtn_weight_dequant_map(pipeline)  ->  { poly_key: W_float32 }
collect_alpha_mse_global(
    pipeline, denoiser, prompt_entries, poly_schedule,
    weight_results=that_map, ...
)  ->  { poly_key: (best_multiplier, mse) }
```

`weight_results` can also accept GPTQ-style `(W_int, scales, _)` tuples if you extend the pipeline later; the **Phase 4.1 doc path** emphasizes **RTN only**.

### 5.4 Integration status

**Persistence:** `run_alpha_search_on_checkpoint` (and the CLI) merge **`alpha_multiplier`** and **`alpha_search_mse`** into **`poly_schedule.json`** by default (with a backup). **`W4A8PolyLinear`** multiplies the evaluated α(σ) by this factor before building the int8 scale.

**Final W4A8 model:** Phase 2 already saved **RTN W4** weights. After Phase 4.1, call **`load_quantized_model_poly(pipeline, quantized_dir)`** to replace scheduled linears with **`W4A8PolyLinear`** using the updated JSON — same safetensors, σ-dependent A8 aligned with the search.

---

## 6. Why alpha search helps (benefits)

### 6.1 Closes the gap between “schedule from offline stats” and “this W4A8 layer”

The polynomial is fit from **population statistics** over calibration prompts and timesteps. A **specific** RTN weight realization and the **existing static A8 path** change what error looks like in practice. Alpha search is a **cheap second pass** that tunes **poly-scaled** activation fake-quant so an **explicit RTN matmul proxy** tracks the **running W4A8 forward** more closely **without retraining weights** (subject to the `b_inv` caveat in §5.1).

### 6.2 Per-layer adaptivity without per-tensor grid search at runtime

Instead of hand-tuning one global constant, you obtain **per-layer multipliers** tied to the **poly’s σ-dependent α**. That addresses **heterogeneous layers** (attention vs MLP, image vs text branch) where a single global scale is always wrong for some blocks.

### 6.3 Conservative poly → practical poly

Phase 3 may use **conservative** clipping (e.g. max over channels when collapsing per-channel polynomials to a scalar, or high-degree fits that generalize safely). A **multiplier \(m < 1\)** effectively **tightens** quantization; **\(m > 1\)** **relaxes** it. Search discovers **where** the schedule is too tight or too loose **for MSE under RTN**.

### 6.4 Composes with RTN before committing to GPTQ

GPTQ (Phase 4) is **much more expensive** and optimizes **weights** given a Hessian proxy. Alpha search targets **activations** given **fixed RTN weights**, with **lightweight** forward-only accumulation. It is a sensible **middle rung**: improve W4A8 quality **before** paying for GPTQ, or **instead of** GPTQ when latency or engineering complexity matters.

### 6.5 Explainability and debugging

Per-layer **(multiplier, MSE)** curves are easy to log. Spikes identify layers where **either** the poly is a poor match **or** RTN weights distort the activation distribution the poly assumed—useful for **iterating Phase 2/3 hyperparameters** (ρ thresholds, degree, CSB coverage).

---

## 7. Comparison to the full repo pipeline (with GPTQ)

| Aspect | Phases 1–3 + **Phase 4.1** (this doc) | Phases 1–4 (existing **GPTQ**) |
|--------|--------------------------------------|--------------------------------|
| Weight update | **RTN only** (Phase 2) | RTN then **GPTQ** error compensation |
| Primary extra calibration | **Alpha multipliers** (activation-side) | **Hessian** collection + column-wise GPTQ |
| Cost | Moderate extra forwards, light CPU | Heavier (Hessian + optimization) |
| Best when | You want **better A8 alignment** under RTN quickly | You need **weight-level** error reduction |

Phase 4.1 **does not exclude** GPTQ later: you could run alpha search on the **RTN** model before GPTQ, or **re-run** search (optionally with GPTQ weight tensors) after Phase 4; the code path for GPTQ tuples exists in `alpha_search.py` for that reason.

---

## 8. Suggested commands (reference)

See `src/settings/commands.md` for authoritative flags. At a high level:

1. **Phase 1:** `python -m src.phase1.run_collection`
2. **Phase 2:** `python -m src.phase2.run_e2e …` (or `run_quantize` with existing diagnostics)
3. **Phase 3:** `python -m src.phase3.generate_schedule --diagnostics-dir … --calibration-dir …`
4. **Phase 4.1:** load pipeline + Phase 2 checkpoint + `poly_schedule.json` in code, then call `extract_rtn_weight_dequant_map` and `collect_alpha_mse_global` (CLI wiring optional).

---

## 9. Summary

- **Phase 1** supplies **rich calibration signals**.  
- **Phase 2** delivers **SSC/CSB-conditioned RTN W4A8**.  
- **Phase 3** turns trajectories + calibration into a **σ-dependent activation clipping schedule**.  
- **Phase 4.1 (alpha search)** calibrates **per-layer multipliers** so that, **for the fixed RTN weights**, a **poly-driven A8 proxy** better matches **the same layer’s W4A8 output** under the sampling distribution—with **low memory**, **batched candidates**, and a clear path toward **persisted overrides** and **improved W4A8 quality** without immediately entering **GPTQ**.
