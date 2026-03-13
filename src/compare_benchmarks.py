"""
Compare benchmark results across multiple quantization configs.

Reads ``benchmark.json`` from each directory, prints a markdown table to
stdout, and writes ``comparison.csv`` to the output directory.

Usage
-----
    python -m src.compare_benchmarks dir1 dir2 ... [--baseline fp16]

Example
-------
    python -m src.compare_benchmarks \\
        benchmark_results/fp16 benchmark_results/adaround_w4 \\
        --baseline fp16
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark(directory: Path) -> Dict:
    """Load benchmark.json from a directory."""
    json_path = directory / "benchmark.json"
    if not json_path.exists():
        raise FileNotFoundError(f"benchmark.json not found in {directory}")
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def _safe_get(d: Any, *keys, default: Any = None) -> Any:
    """Safely traverse a nested dict; return default if any key is missing."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
        if d is None:
            return default
    return d


def build_rows(benchmarks: List[Dict], baseline_config: str) -> List[Dict]:
    """
    Build comparison rows from a list of benchmark dicts.

    Parameters
    ----------
    benchmarks : list of dict
        Loaded benchmark.json contents.
    baseline_config : str
        Config name (e.g. ``"fp16"``) used as the denominator for speedup
        and compression ratio.

    Returns
    -------
    list of dict with keys: config, fid, sfid, is, precision, mean_s,
    speedup, metal_gb, size_gb, compression.
    """
    # Locate baseline
    baseline: Optional[Dict] = None
    for bm in benchmarks:
        if bm.get("config") == baseline_config:
            baseline = bm
            break

    baseline_mean_s: Optional[float] = _safe_get(baseline, "latency", "mean_s")
    baseline_size_gb: Optional[float] = _safe_get(baseline, "model", "size_gb")

    rows = []
    for bm in benchmarks:
        config: str = bm.get("config", "?")
        fid: Optional[float] = _safe_get(bm, "fidelity", "fid")
        sfid: Optional[float] = _safe_get(bm, "fidelity", "sfid")
        isc_mean: Optional[float] = _safe_get(bm, "fidelity", "isc_mean")
        precision: Optional[float] = _safe_get(bm, "fidelity", "precision")
        mean_s: Optional[float] = _safe_get(bm, "latency", "mean_s")
        peak_metal_mb: float = _safe_get(bm, "memory", "peak_metal_mb") or 0.0
        metal_gb: float = peak_metal_mb / 1000.0
        size_gb: Optional[float] = _safe_get(bm, "model", "size_gb")

        speedup: Optional[float] = None
        if mean_s is not None and baseline_mean_s is not None and mean_s > 0:
            speedup = baseline_mean_s / mean_s

        compression: Optional[float] = None
        if size_gb is not None and baseline_size_gb is not None and size_gb > 0:
            compression = baseline_size_gb / size_gb

        rows.append({
            "config": config,
            "fid": fid,
            "sfid": sfid,
            "is": isc_mean,
            "precision": precision,
            "mean_s": mean_s,
            "speedup": speedup,
            "metal_gb": metal_gb,
            "size_gb": size_gb,
            "compression": compression,
        })

    return rows


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val: Any, fmt: str = ".2f", default: str = "--") -> str:
    """Format a numeric value, returning ``default`` for None / NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        if f != f:  # NaN check
            return default
        return format(f, fmt)
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

_HEADERS = [
    "config", "FID", "sFID", "IS", "Precision",
    "mean_s", "speedup", "metal_GB", "size_GB", "compression",
]

_FMT_MAP = {
    "FID": ".2f", "sFID": ".2f", "IS": ".2f", "Precision": ".4f",
    "mean_s": ".1f", "speedup": ".2f", "metal_GB": ".2f",
    "size_GB": ".3f", "compression": ".2f",
}


def _row_cells(row: Dict) -> List[str]:
    return [
        row["config"],
        _fmt(row["fid"], _FMT_MAP["FID"]),
        _fmt(row["sfid"], _FMT_MAP["sFID"]),
        _fmt(row["is"], _FMT_MAP["IS"]),
        _fmt(row["precision"], _FMT_MAP["Precision"]),
        _fmt(row["mean_s"], _FMT_MAP["mean_s"]),
        _fmt(row["speedup"], _FMT_MAP["speedup"]),
        _fmt(row["metal_gb"], _FMT_MAP["metal_GB"]),
        _fmt(row["size_gb"], _FMT_MAP["size_GB"]),
        _fmt(row["compression"], _FMT_MAP["compression"]),
    ]


def print_markdown_table(rows: List[Dict]) -> None:
    """Print a GitHub-flavored markdown table to stdout."""
    all_cells = [_HEADERS] + [_row_cells(r) for r in rows]

    col_widths = [len(h) for h in _HEADERS]
    for cells in all_cells[1:]:
        for i, c in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(c))

    def _fmt_row(cells: List[str]) -> str:
        padded = [c.ljust(col_widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    def _sep_row() -> str:
        return "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    print(_fmt_row(_HEADERS))
    print(_sep_row())
    for cells in all_cells[1:]:
        print(_fmt_row(cells))


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "config", "fid", "sfid", "is", "precision",
    "mean_s", "speedup", "metal_gb", "size_gb", "compression",
]


def write_csv(rows: List[Dict], output_path: Path) -> None:
    """Write comparison rows to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in _CSV_FIELDS})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across multiple quantization configs"
    )
    parser.add_argument(
        "dirs", nargs="+", type=Path,
        help="Directories containing benchmark.json",
    )
    parser.add_argument(
        "--baseline", type=str, default="fp16",
        help="Config name to use as baseline for speedup/compression (default: fp16)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for comparison.csv (default: first input directory)",
    )

    args = parser.parse_args()

    benchmarks: List[Dict] = []
    for d in args.dirs:
        try:
            bm = load_benchmark(d)
            benchmarks.append(bm)
        except FileNotFoundError as e:
            print(f"WARNING: {e}", file=sys.stderr)

    if not benchmarks:
        print("ERROR: No benchmark.json files could be loaded.", file=sys.stderr)
        sys.exit(1)

    rows = build_rows(benchmarks, args.baseline)
    print_markdown_table(rows)

    output_dir = args.output_dir or args.dirs[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    write_csv(rows, csv_path)
    print(f"\n✓ comparison.csv → {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
