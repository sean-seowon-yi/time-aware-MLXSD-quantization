"""Finalize a partial alpha-search checkpoint into the production poly schedule.

Usage::

    python -m src.finalize_alpha_checkpoint \
        --quantized-dir quantized/w4a8_l2_a0.50_gs32_static \
        --cutoff 50

This reads ``alpha_search_checkpoint.json`` from the quantized directory,
verifies at least *cutoff* prompts have been accumulated, computes the best
alpha multiplier per layer from the accumulated SE, and merges the results
into ``poly_schedule.json`` (with a ``.pre_alpha_search.bak`` backup).

No GPU, no model loading — pure JSON manipulation.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ALPHA_CANDIDATES = [
    0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0,
    3.0, 4.0, 5.0, 6, 8, 10,
]

ALPHA_SEARCH_RESULTS_FILENAME = "alpha_search_results.json"
ALPHA_SEARCH_CHECKPOINT_FILENAME = "alpha_search_checkpoint.json"
POLY_SCHEDULE_FILENAME = "poly_schedule.json"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _get_best_alpha(
    total_se: List[float], total_elements: int
) -> Tuple[float, float]:
    """Replicate _AlphaAccumulator.get_best_alpha from checkpoint data."""
    if total_elements == 0:
        return 1.0, float("inf")
    se = np.array(total_se, dtype=np.float64)
    mses = se / total_elements
    best_idx = int(np.argmin(mses))
    return _ALPHA_CANDIDATES[best_idx], float(mses[best_idx])


def finalize_checkpoint(
    quantized_dir: Path,
    cutoff: int,
    *,
    checkpoint_path: Path | None = None,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, Tuple[float, float]]:
    qdir = Path(quantized_dir).resolve()
    cp_path = Path(checkpoint_path) if checkpoint_path else qdir / ALPHA_SEARCH_CHECKPOINT_FILENAME

    if not cp_path.is_file():
        print(f"ERROR: checkpoint not found at {cp_path}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(cp_path.read_text())

    version = raw.get("version")
    next_idx = raw.get("next_prompt_index", 0)
    n_prompts = raw.get("n_prompts", 0)
    layers = raw.get("layers", {})

    print(f"Checkpoint version  : {version}")
    print(f"Prompts completed   : {next_idx} / {n_prompts}")
    print(f"Layers              : {len(layers)}")
    print(f"Cutoff requested    : {cutoff}")
    print()

    if next_idx < cutoff:
        print(
            f"ERROR: only {next_idx} prompts accumulated — need at least {cutoff}. "
            f"Let the run continue.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_cands = len(_ALPHA_CANDIDATES)
    results: Dict[str, Tuple[float, float]] = {}
    for key, state in layers.items():
        se = state["total_se"]
        elems = state["total_elements"]
        if len(se) != n_cands:
            print(
                f"WARNING: layer {key} has {len(se)} SE entries (expected {n_cands}), skipping",
                file=sys.stderr,
            )
            continue
        results[key] = _get_best_alpha(se, elems)

    if not results:
        print("ERROR: no valid layers found in checkpoint", file=sys.stderr)
        sys.exit(1)

    mults = [m for m, _ in results.values()]
    mses = [mse for _, mse in results.values()]
    print(f"Computed best alpha for {len(results)} layers")
    print(f"  Multiplier distribution: min={min(mults):.2f}  median={np.median(mults):.2f}  max={max(mults):.2f}")
    print(f"  MSE distribution:        min={min(mses):.6f}  median={np.median(mses):.6f}  max={max(mses):.6f}")
    print()

    if dry_run:
        print("DRY RUN — not writing any files.")
        return results

    # --- Write alpha_search_results.json ---
    results_path = qdir / ALPHA_SEARCH_RESULTS_FILENAME
    layers_out: Dict[str, Dict[str, float]] = {}
    for k, (mult, mse) in results.items():
        layers_out[k] = {"multiplier": float(mult), "mse": float(mse)}
    results_payload: Dict[str, Any] = {
        "version": 1,
        "n_layers": len(layers_out),
        "alpha_candidates": list(_ALPHA_CANDIDATES),
        "finalized_from_checkpoint": True,
        "prompts_used": next_idx,
        "cutoff_requested": cutoff,
        "layers": layers_out,
    }
    _atomic_write(results_path, json.dumps(results_payload, indent=2))
    print(f"Wrote {results_path}")

    # --- Merge into poly_schedule.json ---
    poly_path = qdir / POLY_SCHEDULE_FILENAME
    if not poly_path.is_file():
        print(f"WARNING: {poly_path} not found — skipping merge.", file=sys.stderr)
        return results

    poly_schedule = json.loads(poly_path.read_text())
    merged = copy.deepcopy(poly_schedule)
    sched_layers = merged.setdefault("layers", {})
    n_merged = 0
    for key, (mult, mse) in results.items():
        if key not in sched_layers:
            continue
        sched_layers[key]["alpha_multiplier"] = float(mult)
        sched_layers[key]["alpha_search_mse"] = float(mse)
        n_merged += 1

    if backup and poly_path.exists():
        bak = poly_path.with_suffix(".json.pre_alpha_search.bak")
        shutil.copy2(poly_path, bak)
        print(f"Backed up {poly_path} -> {bak}")

    _atomic_write(poly_path, json.dumps(merged, indent=2))
    print(f"Merged alpha_multiplier into {poly_path} ({n_merged} layers)")
    print()
    print("Done. Load the model with load_quantized_model_poly to deploy.")

    return results


def main():
    p = argparse.ArgumentParser(
        description="Finalize a partial alpha-search checkpoint into the production poly schedule.",
    )
    p.add_argument(
        "--quantized-dir",
        type=Path,
        required=True,
        help="Directory with quantize_config.json, poly_schedule.json, and checkpoint",
    )
    p.add_argument(
        "--cutoff",
        type=int,
        required=True,
        help="Minimum number of prompts to require (e.g. 50). "
             "Fails if the checkpoint has fewer completed prompts.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint path (default: <quantized-dir>/alpha_search_checkpoint.json)",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup poly_schedule.json before overwriting",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing any files",
    )
    args = p.parse_args()

    finalize_checkpoint(
        args.quantized_dir,
        args.cutoff,
        checkpoint_path=args.checkpoint,
        backup=not args.no_backup,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
