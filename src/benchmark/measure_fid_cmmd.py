#!/usr/bin/env python3
"""FID + CMMD only between two image directories (no generation).

Assumes ``--eval-dir`` and ``--reference-dir`` already contain PNG/JPEG images.

Example::

    python -m src.benchmark.measure_fid_cmmd \\
        --eval-dir benchmark_results/run/images \\
        --reference-dir results/gt/images \\
        --output-json benchmark_results/run/fid_cmmd.json

Requires: torch-fidelity (FID), open_clip_torch (CMMD).
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path
from typing import Any, Dict, Optional


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def run_measure(
    eval_dir: Path,
    reference_dir: Path,
    *,
    fidelity_cuda: bool = False,
    clip_arch: str = "ViT-L-14-336",
    clip_pretrained: str = "openai",
    clip_batch_size: int = 16,
) -> Dict[str, Any]:
    from src.benchmark.metric_clip import (
        compute_clip_embeddings,
        compute_cmmd_from_embeddings,
    )
    from src.benchmark.metric_fidelity_torch import compute_fidelity_metrics

    eval_s = str(eval_dir.resolve())
    ref_s = str(reference_dir.resolve())

    fid_block: Optional[Dict] = compute_fidelity_metrics(
        eval_s,
        ref_s,
        use_cuda=fidelity_cuda,
        isc=False,
        kid=False,
        prc=False,
    )

    emb_eval = compute_clip_embeddings(
        eval_s,
        batch_size=clip_batch_size,
        arch=clip_arch,
        pretrained=clip_pretrained,
    )
    emb_ref = compute_clip_embeddings(
        ref_s,
        batch_size=clip_batch_size,
        arch=clip_arch,
        pretrained=clip_pretrained,
    )

    cmmd_val: Optional[float] = None
    if emb_eval is not None and emb_ref is not None:
        cmmd_val = compute_cmmd_from_embeddings(
            emb_eval["embeddings"],
            emb_ref["embeddings"],
        )

    fid_scalar: Optional[float] = None
    if fid_block is not None:
        v = float(fid_block["fid"])
        fid_scalar = None if math.isnan(v) else v

    return {
        "eval_dir": eval_s,
        "reference_dir": ref_s,
        "fid": fid_scalar,
        "cmmd": cmmd_val,
        "clip_model_id": emb_eval.get("model_id") if emb_eval else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure FID and CMMD between two image directories (no generation).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Generated or evaluated images (FID input1 / CMMD first set).",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Reference images (FID input2 / CMMD second set).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write results JSON to this path (default: print JSON to stdout only).",
    )
    parser.add_argument(
        "--fidelity-cuda",
        action="store_true",
        help="Run torch-fidelity on CUDA.",
    )
    parser.add_argument("--clip-arch", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--clip-batch-size", type=int, default=16)

    args = parser.parse_args()
    eval_dir = args.eval_dir
    reference_dir = args.reference_dir
    if not eval_dir.is_dir():
        print(f"ERROR: --eval-dir is not a directory: {eval_dir}", file=sys.stderr)
        sys.exit(1)
    if not reference_dir.is_dir():
        print(
            f"ERROR: --reference-dir is not a directory: {reference_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    results = run_measure(
        eval_dir,
        reference_dir,
        fidelity_cuda=args.fidelity_cuda,
        clip_arch=args.clip_arch,
        clip_pretrained=args.clip_pretrained,
        clip_batch_size=args.clip_batch_size,
    )

    out = {
        "eval_dir": results["eval_dir"],
        "reference_dir": results["reference_dir"],
        "fid": results["fid"],
        "cmmd": results["cmmd"],
        "clip_model_id": results["clip_model_id"],
    }

    text = json.dumps(_json_safe(out), indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
        print(f"Wrote {args.output_json.resolve()}", file=sys.stderr)
    else:
        print(text)

    fid = results.get("fid")
    cmmd = results.get("cmmd")
    print("\n--- Summary ---", file=sys.stderr)
    if fid is not None:
        print(f"  FID (eval vs ref): {fid:.4f}", file=sys.stderr)
    else:
        print("  FID: unavailable (torch-fidelity missing or failed)", file=sys.stderr)
    if cmmd is not None:
        print(f"  CMMD:              {cmmd:.6f}", file=sys.stderr)
    else:
        print("  CMMD: unavailable (open_clip missing or <2 images per set)", file=sys.stderr)


if __name__ == "__main__":
    main()
