"""
Merge multiple partial adaround_optimize output directories into one.

Usage:
    python scripts/merge_adaround_outputs.py \
        --inputs out_0 out_1 out_2 out_3 \
        --output quantized_weights_poly_merged

Each input directory must have the structure:
    <dir>/
        config.json
        weights/
            mmN.npz ...

The merged output uses the config.json from the first input dir.
If any block .npz appears in more than one input, the script errors
rather than silently overwriting.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge partial adaround output dirs")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="Partial output directories to merge")
    parser.add_argument("--output", type=Path, required=True,
                        help="Merged output directory (must not exist unless --force)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directory")
    args = parser.parse_args()

    # Validate inputs
    for d in args.inputs:
        if not d.is_dir():
            print(f"ERROR: input directory not found: {d}")
            sys.exit(1)
        if not (d / "weights").is_dir():
            print(f"ERROR: no weights/ subdirectory in {d}")
            sys.exit(1)

    # Collect all .npz files, check for duplicates
    seen: dict[str, Path] = {}
    for d in args.inputs:
        for npz in sorted((d / "weights").glob("*.npz")):
            if npz.name in seen:
                print(f"ERROR: block {npz.name} appears in both {seen[npz.name].parent.parent} and {d}")
                sys.exit(1)
            seen[npz.name] = npz

    if not seen:
        print("ERROR: no .npz files found in any input directory")
        sys.exit(1)

    # Set up output directory
    if args.output.exists():
        if not args.force:
            print(f"ERROR: output {args.output} already exists. Use --force to overwrite.")
            sys.exit(1)
        shutil.rmtree(args.output)

    out_weights = args.output / "weights"
    out_weights.mkdir(parents=True)

    # Copy all .npz files
    for name, src in sorted(seen.items()):
        shutil.copy2(src, out_weights / name)
        print(f"  {src.parent.parent.name}/weights/{name} -> weights/{name}")

    # Copy config.json from first input, update block count
    config_src = args.inputs[0] / "config.json"
    if config_src.exists():
        with open(config_src) as f:
            config = json.load(f)
        # Update n_blocks_quantised to reflect the full merged set
        config["n_blocks_quantised"] = len(seen)
        config["merged_from"] = [str(d) for d in args.inputs]
        # Merge block_metrics from all inputs
        merged_metrics = []
        for d in args.inputs:
            cfg_path = d / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    c = json.load(f)
                merged_metrics.extend(c.get("block_metrics", []))
        config["block_metrics"] = merged_metrics
        with open(args.output / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n  config.json written (merged from {len(args.inputs)} runs)")

    print(f"\nMerged {len(seen)} blocks into {args.output}/")


if __name__ == "__main__":
    main()
