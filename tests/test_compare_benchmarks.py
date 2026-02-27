"""
Tests for src/compare_benchmarks.py

All tests use synthetic benchmark.json data; no model loading required.
"""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.compare_benchmarks import (
    load_benchmark,
    build_rows,
    print_markdown_table,
    write_csv,
    _safe_get,
    _fmt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_benchmark(
    config="fp16",
    fid=10.0,
    sfid=12.0,
    isc_mean=8.0,
    precision=0.65,
    recall=0.55,
    mean_s=30.0,
    peak_metal_mb=18000.0,
    size_gb=5.5,
) -> dict:
    """Build a minimal benchmark dict matching the benchmark.json schema."""
    return {
        "config": config,
        "num_images": 150,
        "num_steps": 28,
        "cfg_scale": 7.0,
        "seed": 42,
        "latency": {
            "mean_s": mean_s, "std_s": 1.0,
            "p50_s": mean_s, "p95_s": mean_s + 2.0,
            "min_s": mean_s - 2.0, "max_s": mean_s + 3.0,
            "warmup_images": 2, "measured_images": 148,
        },
        "memory": {"peak_metal_mb": peak_metal_mb, "peak_rss_mb": 24000.0},
        "model": {"size_gb": size_gb, "total_params_M": 2050.0},
        "fidelity": {
            "fid": fid, "sfid": sfid,
            "isc_mean": isc_mean, "isc_std": 0.3,
            "kid_mean": 0.003, "kid_std": 0.0002,
            "precision": precision, "recall": recall,
            "reference_dir": "/tmp/ref", "num_reference_images": 1000,
            "num_generated_images": 150,
        },
    }


# ---------------------------------------------------------------------------
# TestLoadBenchmark
# ---------------------------------------------------------------------------

class TestLoadBenchmark:

    def test_loads_valid_json(self, tmp_path):
        bm = _make_benchmark()
        (tmp_path / "benchmark.json").write_text(json.dumps(bm))
        result = load_benchmark(tmp_path)
        assert result["config"] == "fp16"
        assert result["fidelity"]["fid"] == pytest.approx(10.0)

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_benchmark(tmp_path)  # no benchmark.json present

    def test_config_field_preserved(self, tmp_path):
        bm = _make_benchmark(config="adaround_w4")
        (tmp_path / "benchmark.json").write_text(json.dumps(bm))
        result = load_benchmark(tmp_path)
        assert result["config"] == "adaround_w4"


# ---------------------------------------------------------------------------
# TestSafeGet
# ---------------------------------------------------------------------------

class TestSafeGet:

    def test_nested_key_access(self):
        d = {"a": {"b": {"c": 42}}}
        assert _safe_get(d, "a", "b", "c") == 42

    def test_missing_key_returns_default(self):
        d = {"a": {"b": 1}}
        assert _safe_get(d, "a", "x") is None
        assert _safe_get(d, "a", "x", default=-1) == -1

    def test_none_value_returns_default(self):
        d = {"a": None}
        assert _safe_get(d, "a", "b") is None

    def test_non_dict_intermediate_returns_default(self):
        d = {"a": 5}
        assert _safe_get(d, "a", "b") is None


# ---------------------------------------------------------------------------
# TestFmt
# ---------------------------------------------------------------------------

class TestFmt:

    def test_formats_float(self):
        assert _fmt(3.14159, ".2f") == "3.14"

    def test_none_returns_default(self):
        assert _fmt(None) == "--"

    def test_nan_returns_default(self):
        assert _fmt(float("nan")) == "--"

    def test_custom_default(self):
        assert _fmt(None, default="N/A") == "N/A"

    def test_integer_value(self):
        assert _fmt(5, ".1f") == "5.0"


# ---------------------------------------------------------------------------
# TestBuildRows
# ---------------------------------------------------------------------------

class TestBuildRows:

    def test_basic_row_structure(self):
        bm = _make_benchmark(config="fp16")
        rows = build_rows([bm], baseline_config="fp16")
        assert len(rows) == 1
        row = rows[0]
        assert row["config"] == "fp16"
        assert row["fid"] == pytest.approx(10.0)
        assert row["sfid"] == pytest.approx(12.0)
        assert row["precision"] == pytest.approx(0.65)

    def test_speedup_relative_to_baseline(self):
        fp16 = _make_benchmark(config="fp16", mean_s=30.0)
        w4 = _make_benchmark(config="adaround_w4", mean_s=15.0)
        rows = build_rows([fp16, w4], baseline_config="fp16")
        # fp16 speedup should be 1.0 (baseline / baseline)
        assert rows[0]["speedup"] == pytest.approx(1.0)
        # w4 is 2x faster
        assert rows[1]["speedup"] == pytest.approx(2.0)

    def test_compression_relative_to_baseline(self):
        fp16 = _make_benchmark(config="fp16", size_gb=8.0)
        w4 = _make_benchmark(config="adaround_w4", size_gb=2.0)
        rows = build_rows([fp16, w4], baseline_config="fp16")
        # fp16 compression = 8.0 / 8.0 = 1.0
        assert rows[0]["compression"] == pytest.approx(1.0)
        # w4 compression = 8.0 / 2.0 = 4.0
        assert rows[1]["compression"] == pytest.approx(4.0)

    def test_missing_baseline_gives_none_speedup(self):
        bm = _make_benchmark(config="adaround_w4", mean_s=20.0)
        rows = build_rows([bm], baseline_config="fp16")  # fp16 not present
        assert rows[0]["speedup"] is None

    def test_multiple_configs_order_preserved(self):
        configs = ["fp16", "naive_int8", "adaround_w4"]
        benchmarks = [_make_benchmark(config=c) for c in configs]
        rows = build_rows(benchmarks, baseline_config="fp16")
        assert [r["config"] for r in rows] == configs

    def test_metal_gb_conversion(self):
        bm = _make_benchmark(peak_metal_mb=20000.0)
        rows = build_rows([bm], baseline_config="fp16")
        assert rows[0]["metal_gb"] == pytest.approx(20.0)

    def test_missing_fidelity_gives_none_metrics(self):
        bm = _make_benchmark()
        bm["fidelity"] = None
        rows = build_rows([bm], baseline_config="fp16")
        assert rows[0]["fid"] is None
        assert rows[0]["sfid"] is None
        assert rows[0]["precision"] is None

    def test_missing_model_gives_none_size(self):
        bm = _make_benchmark()
        bm["model"] = None
        rows = build_rows([bm], baseline_config="fp16")
        assert rows[0]["size_gb"] is None
        assert rows[0]["compression"] is None


# ---------------------------------------------------------------------------
# TestPrintMarkdownTable
# ---------------------------------------------------------------------------

class TestPrintMarkdownTable:

    def test_no_crash_single_row(self, capsys):
        bm = _make_benchmark(config="fp16")
        rows = build_rows([bm], baseline_config="fp16")
        print_markdown_table(rows)
        out = capsys.readouterr().out
        assert "fp16" in out

    def test_no_crash_multiple_rows(self, capsys):
        benchmarks = [
            _make_benchmark(config="fp16", mean_s=30.0, size_gb=8.0),
            _make_benchmark(config="adaround_w4", mean_s=15.0, size_gb=2.0),
        ]
        rows = build_rows(benchmarks, baseline_config="fp16")
        print_markdown_table(rows)
        out = capsys.readouterr().out
        assert "fp16" in out
        assert "adaround_w4" in out

    def test_output_has_header_separator(self, capsys):
        bm = _make_benchmark()
        rows = build_rows([bm], baseline_config="fp16")
        print_markdown_table(rows)
        out = capsys.readouterr().out
        lines = out.strip().splitlines()
        # Line 0: header, Line 1: separator, Line 2+: data
        assert len(lines) >= 3
        assert "---" in lines[1]

    def test_output_has_pipe_delimiters(self, capsys):
        bm = _make_benchmark()
        rows = build_rows([bm], baseline_config="fp16")
        print_markdown_table(rows)
        out = capsys.readouterr().out
        for line in out.strip().splitlines():
            assert "|" in line

    def test_none_values_show_dashes(self, capsys):
        bm = _make_benchmark()
        bm["fidelity"] = None
        bm["model"] = None
        rows = build_rows([bm], baseline_config="fp16")
        print_markdown_table(rows)
        out = capsys.readouterr().out
        assert "--" in out


# ---------------------------------------------------------------------------
# TestWriteCsv
# ---------------------------------------------------------------------------

class TestWriteCsv:

    def test_creates_csv_file(self, tmp_path):
        bm = _make_benchmark()
        rows = build_rows([bm], baseline_config="fp16")
        csv_path = tmp_path / "comparison.csv"
        write_csv(rows, csv_path)
        assert csv_path.exists()

    def test_csv_has_expected_columns(self, tmp_path):
        bm = _make_benchmark()
        rows = build_rows([bm], baseline_config="fp16")
        csv_path = tmp_path / "comparison.csv"
        write_csv(rows, csv_path)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        expected = {"config", "fid", "sfid", "is", "precision",
                    "mean_s", "speedup", "metal_gb", "size_gb", "compression"}
        assert expected.issubset(set(fieldnames))

    def test_csv_row_count_matches(self, tmp_path):
        benchmarks = [_make_benchmark(config=c) for c in ["fp16", "adaround_w4"]]
        rows = build_rows(benchmarks, baseline_config="fp16")
        csv_path = tmp_path / "comparison.csv"
        write_csv(rows, csv_path)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            data_rows = list(reader)
        assert len(data_rows) == 2

    def test_csv_none_written_as_empty_string(self, tmp_path):
        bm = _make_benchmark()
        bm["model"] = None
        rows = build_rows([bm], baseline_config="fp16")
        csv_path = tmp_path / "comparison.csv"
        write_csv(rows, csv_path)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # size_gb should be empty string for None
        assert row["size_gb"] == ""

    def test_csv_values_match_rows(self, tmp_path):
        bm = _make_benchmark(config="fp16", fid=12.34)
        rows = build_rows([bm], baseline_config="fp16")
        csv_path = tmp_path / "comparison.csv"
        write_csv(rows, csv_path)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["config"] == "fp16"
        assert float(row["fid"]) == pytest.approx(12.34)


# ---------------------------------------------------------------------------
# TestMainCLI (smoke test using argparse + subprocess mock)
# ---------------------------------------------------------------------------

class TestMainCLI:

    def test_main_runs_without_error(self, tmp_path, capsys):
        """main() should print a table and write comparison.csv."""
        bm = _make_benchmark(config="fp16")
        bm_dir = tmp_path / "fp16"
        bm_dir.mkdir()
        (bm_dir / "benchmark.json").write_text(json.dumps(bm))

        with patch("sys.argv", ["compare_benchmarks", str(bm_dir), "--baseline", "fp16",
                                 "--output-dir", str(tmp_path)]):
            from src.compare_benchmarks import main
            main()

        assert (tmp_path / "comparison.csv").exists()
        out = capsys.readouterr().out
        assert "fp16" in out

    def test_main_warns_on_missing_dir(self, tmp_path, capsys):
        """Directories without benchmark.json emit a warning and continue."""
        bm = _make_benchmark(config="fp16")
        bm_dir = tmp_path / "fp16"
        bm_dir.mkdir()
        (bm_dir / "benchmark.json").write_text(json.dumps(bm))

        missing_dir = tmp_path / "nonexistent"

        with patch("sys.argv", ["compare_benchmarks",
                                 str(bm_dir), str(missing_dir),
                                 "--baseline", "fp16",
                                 "--output-dir", str(tmp_path)]):
            from src.compare_benchmarks import main
            main()

        err = capsys.readouterr().err
        assert "WARNING" in err

    def test_main_exits_with_no_valid_dirs(self, tmp_path):
        """main() should sys.exit(1) when no benchmark.json can be loaded."""
        missing = tmp_path / "nodata"
        missing.mkdir()
        with patch("sys.argv", ["compare_benchmarks", str(missing)]):
            from src.compare_benchmarks import main
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
