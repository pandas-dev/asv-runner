from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ci.process_results import (
    BASE_COLUMNS,
    DERIVED_COLUMNS,
    PARQUET_DIRNAME,
    build_new_rows,
    detect_regression,
    load_existing,
    run,
)


def _write_benchmarks(input_path: Path, benchmarks: dict[str, Any]) -> None:
    results_dir = input_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "benchmarks.json").write_text(json.dumps(benchmarks))


def _write_result(
    input_path: Path,
    sha: str,
    when: dt.datetime,
    results: dict[str, Any],
    result_columns: list[str] | None = None,
) -> None:
    asvrunner_dir = input_path / "results" / "asvrunner"
    asvrunner_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "commit_hash": sha,
        "date": int(when.timestamp() * 1000),
        "result_columns": result_columns or ["result", "params"],
        "results": results,
    }
    (asvrunner_dir / f"{sha}.json").write_text(json.dumps(payload))


def test_load_existing_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_existing(tmp_path / "does-not-exist.parquet") is None


def test_load_existing_normalizes_added_date_dtype(tmp_path: Path) -> None:
    parquet_path = tmp_path / "results.parquet"
    df = pd.DataFrame(
        {
            "sha": pd.array(["a", "b"], dtype="string[pyarrow]"),
            "result": pd.array([1.0, 2.0], dtype="float64[pyarrow]"),
            "added_date": ["2024-01-01", "2024-01-02"],
        }
    )
    df.to_parquet(
        parquet_path,
        index=False,
        partition_cols=["added_date"],
        basename_template="part-{i}.parquet",
    )

    loaded = load_existing(parquet_path)

    assert loaded is not None
    assert str(loaded["added_date"].dtype) == "string"


def test_build_new_rows_produces_expected_rows(tmp_path: Path) -> None:
    _write_benchmarks(
        tmp_path,
        {
            "version": "1.0",
            "bench.foo": {"param_names": ["a", "b"]},
        },
    )
    _write_result(
        tmp_path,
        "deadbeef",
        dt.datetime(2024, 1, 1),
        {
            "bench.foo": [
                [1.0, 2.0],
                [["1", "2"], ["3"]],
            ],
        },
    )

    df = build_new_rows(tmp_path, skip_shas=set(), added_date="2024-01-02")

    assert len(df) == 2
    assert set(df["sha"]) == {"deadbeef"}
    assert set(df["params"]) == {"a=1, b=3", "a=2, b=3"}
    assert set(df["name"]) == {"bench.foo"}
    assert sorted(df["result"].tolist()) == [1.0, 2.0]
    assert set(df["added_date"]) == {"2024-01-02"}


def test_build_new_rows_skips_machine_json(tmp_path: Path) -> None:
    _write_benchmarks(
        tmp_path,
        {
            "version": "1.0",
            "bench.foo": {"param_names": ["a"]},
        },
    )
    _write_result(
        tmp_path,
        "deadbeef",
        dt.datetime(2024, 1, 1),
        {"bench.foo": [[1.0], [["1"]]]},
    )
    # machine.json must be ignored even though it lives alongside results.
    (tmp_path / "results" / "asvrunner" / "machine.json").write_text(
        json.dumps({"unrelated": "data"})
    )

    df = build_new_rows(tmp_path, skip_shas=set(), added_date="2024-01-02")

    assert len(df) == 1


def test_build_new_rows_respects_skip_shas(tmp_path: Path) -> None:
    _write_benchmarks(
        tmp_path,
        {
            "version": "1.0",
            "bench.foo": {"param_names": ["a"]},
        },
    )
    _write_result(
        tmp_path,
        "skipme",
        dt.datetime(2024, 1, 1),
        {"bench.foo": [[1.0], [["1"]]]},
    )
    _write_result(
        tmp_path,
        "keepme",
        dt.datetime(2024, 1, 2),
        {"bench.foo": [[2.0], [["1"]]]},
    )

    df = build_new_rows(tmp_path, skip_shas={"skipme"}, added_date="2024-01-02")

    assert set(df["sha"]) == {"keepme"}


def test_build_new_rows_empty_when_no_result_files(tmp_path: Path) -> None:
    _write_benchmarks(
        tmp_path,
        {
            "version": "1.0",
            "bench.foo": {"param_names": ["a"]},
        },
    )
    (tmp_path / "results" / "asvrunner").mkdir(parents=True)

    df = build_new_rows(tmp_path, skip_shas=set(), added_date="2024-01-02")

    assert len(df) == 0
    assert "name" in df.columns
    assert "added_date" in df.columns


def _stable_frame(n: int, value: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "name": ["bench.foo"] * n,
            "params": ["a=1"] * n,
            "date": dates,
            "result": [value] * n,
        }
    )


def test_detect_regression_adds_derived_columns() -> None:
    df = _stable_frame(30)

    out = detect_regression(df, window_size=5)

    for col in DERIVED_COLUMNS:
        assert col in out.columns


def test_detect_regression_no_regressions_on_constant_series() -> None:
    df = _stable_frame(30, value=1.0)

    out = detect_regression(df, window_size=5)

    assert not out["is_regression"].any()


def test_detect_regression_flags_step_change() -> None:
    # 30 points stable, then a sharp slowdown that holds — established_worst
    # from window_size ago is well under 0.95 * established_best now.
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    values = [1.0] * (n // 2) + [10.0] * (n // 2)
    df = pd.DataFrame(
        {
            "name": ["bench.foo"] * n,
            "params": ["a=1"] * n,
            "date": dates,
            "result": values,
        }
    )

    out = detect_regression(df, window_size=5)

    assert out["is_regression"].sum() >= 1


def test_detect_regression_drops_null_result_rows() -> None:
    df = _stable_frame(10)
    df.loc[0, "result"] = None

    out = detect_regression(df, window_size=5)

    assert len(out) == 9


def test_run_writes_parquet_with_expected_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    output_path.mkdir()

    _write_benchmarks(
        input_path,
        {
            "version": "1.0",
            "bench.foo": {"param_names": ["a"]},
        },
    )
    base = dt.datetime(2024, 1, 1)
    for i in range(30):
        sha = f"sha{i:03d}"
        _write_result(
            input_path,
            sha,
            base + dt.timedelta(days=i),
            {"bench.foo": [[1.0], [["1"]]]},
        )

    run(input_path, output_path)

    parquet_path = output_path / PARQUET_DIRNAME
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    for col in BASE_COLUMNS:
        assert col in df.columns
    for col in DERIVED_COLUMNS:
        assert col in df.columns
    assert len(df) == 30


def test_run_appends_to_existing_results(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    output_path.mkdir()

    _write_benchmarks(
        input_path,
        {
            "version": "1.0",
            "bench.foo": {"param_names": ["a"]},
        },
    )
    base = dt.datetime(2024, 1, 1)
    for i in range(15):
        _write_result(
            input_path,
            f"sha{i:03d}",
            base + dt.timedelta(days=i),
            {"bench.foo": [[1.0], [["1"]]]},
        )
    run(input_path, output_path)

    # Add new result files for additional shas; existing ones should be skipped
    # and the new ones appended.
    for i in range(15, 25):
        _write_result(
            input_path,
            f"sha{i:03d}",
            base + dt.timedelta(days=i),
            {"bench.foo": [[1.0], [["1"]]]},
        )
    run(input_path, output_path)

    df = pd.read_parquet(output_path / PARQUET_DIRNAME)
    assert len(df) == 25
    assert df["sha"].nunique() == 25


if __name__ == "__main__":
    pytest.main([__file__])
