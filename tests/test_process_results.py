from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas._testing as tm
import pyarrow as pa
import pytest

from asv_runner.process_results import (
    BASE_COLUMNS,
    DERIVED_COLUMNS,
    PARQUET_DIRNAME,
    build_new_rows,
    build_parquet,
    detect_regression,
    get_commit_range,
    make_body,
    make_envs_diff,
    read_rows,
    read_sha_dates,
    stage_asv_inputs,
    update_parquet,
    write_rows,
)

# === get_commit_range / make_body ===


def _benchmarks_with_shas(shas: list[str], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"sha": shas, "date": pd.to_datetime(dates)})


def test_get_commit_range_returns_prev_sha_to_sha() -> None:
    df = _benchmarks_with_shas(
        ["a", "b", "c"], ["2024-01-01", "2024-01-02", "2024-01-03"]
    )
    assert get_commit_range(benchmarks=df, sha="c") == "b...c"
    assert get_commit_range(benchmarks=df, sha="b") == "a...b"


def test_get_commit_range_orders_by_date_not_input_order() -> None:
    df = _benchmarks_with_shas(
        ["c", "a", "b"], ["2024-01-03", "2024-01-01", "2024-01-02"]
    )
    assert get_commit_range(benchmarks=df, sha="c") == "b...c"


def _regression_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sha": ["abc123", "abc123", "abc123", "def456"],
            "is_regression": [True, True, True, True],
            "name": ["bench.foo", "bench.foo", "bench.bar", "bench.foo"],
            "params": ["x=1", "x=2", "", "x=1"],
            "abs_change": [0.5, 0.0005, 1.5, 0.1],
            "pct_change": [0.10, 0.20, 0.30, 0.40],
        }
    )


def test_make_body_includes_commit_range_link() -> None:
    body = make_body(
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
    )
    assert (
        "[Commit Range](https://github.com/pandas-dev/pandas/compare/aaa...bbb)" in body
    )


def test_make_body_only_includes_target_sha_regressions() -> None:
    body = make_body(
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
    )
    assert "bench.foo" in body
    assert "bench.bar" in body
    assert "def456" not in body


def test_make_body_renders_param_sublist_for_nonempty_params() -> None:
    body = make_body(
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
    )
    assert "   - [ ] [x=1]" in body
    assert "   - [ ] [x=2]" in body
    assert "10.000% (500.000ms)" in body
    assert "20.000% (500.000us)" in body


def test_make_body_inlines_severity_when_params_empty() -> None:
    body = make_body(
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
    )
    msg = (
        " - [ ] [bench.bar](https://pandas-dev.github.io/asv-runner/#bench.bar)"
        " - 30.000% (1.500s)"
    )
    assert msg in body


def test_make_body_shorten_collapses_param_sublist() -> None:
    full = make_body(
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
    )
    short = make_body(
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
        shorten=True,
    )
    assert len(short) < len(full)
    assert "   - [ ]" not in short
    assert "10.000% (500.000ms)" in short


def test_make_body_with_pr_info_links_pr_and_pings_author_and_approvers() -> None:
    body = make_body(
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
        pr_info={
            "number": 1234,
            "author": "alice",
            "approvers": ["bob", "carol"],
        },
    )
    assert "[PR #1234](https://github.com/pandas-dev/pandas/pull/1234)" in body
    assert "cc @alice @bob @carol" in body
    assert "[Commit Range]" not in body
    assert "Subsequent benchmarks may have skipped" not in body
    assert "bench.foo" in body


def test_make_body_with_pr_info_no_approvers_only_pings_author() -> None:
    body = make_body(
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
        pr_info={"number": 1234, "author": "alice", "approvers": []},
    )
    assert "cc @alice\n" in body


def test_make_body_excludes_non_regression_rows() -> None:
    df = pd.DataFrame(
        {
            "sha": ["abc", "abc"],
            "is_regression": [True, False],
            "name": ["bench.foo", "bench.bar"],
            "params": ["", ""],
            "abs_change": [0.5, 0.5],
            "pct_change": [0.10, 0.10],
        }
    )
    body = make_body(
        commit_range="x...y",
        benchmarks=df,
        sha="abc",
    )
    assert "bench.foo" in body
    assert "bench.bar" not in body


# === detect_regression / build_new_rows / load_existing / build_parquet ===


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
        {"bench.foo": [[1.0, 2.0], [["1", "2"], ["3"]]]},
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
        {"version": "1.0", "bench.foo": {"param_names": ["a"]}},
    )
    _write_result(
        tmp_path,
        "deadbeef",
        dt.datetime(2024, 1, 1),
        {"bench.foo": [[1.0], [["1"]]]},
    )
    (tmp_path / "results" / "asvrunner" / "machine.json").write_text(
        json.dumps({"unrelated": "data"})
    )

    df = build_new_rows(tmp_path, skip_shas=set(), added_date="2024-01-02")
    assert len(df) == 1


def test_build_new_rows_respects_skip_shas(tmp_path: Path) -> None:
    _write_benchmarks(
        tmp_path,
        {"version": "1.0", "bench.foo": {"param_names": ["a"]}},
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


def test_build_new_rows_skips_benchmarks_missing_from_benchmarks_json(
    tmp_path: Path,
) -> None:
    _write_benchmarks(
        tmp_path,
        {"version": "1.0", "bench.known": {"param_names": ["a"]}},
    )
    _write_result(
        tmp_path,
        "deadbeef",
        dt.datetime(2024, 1, 1),
        {
            "bench.known": [[1.0], [["1"]]],
            "bench.renamed_away": [[2.0], [["1"]]],
        },
    )

    df = build_new_rows(tmp_path, skip_shas=set(), added_date="2024-01-02")

    assert set(df["name"]) == {"bench.known"}
    assert len(df) == 1


def test_build_new_rows_empty_when_no_result_files(tmp_path: Path) -> None:
    _write_benchmarks(
        tmp_path,
        {"version": "1.0", "bench.foo": {"param_names": ["a"]}},
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


def test_build_parquet_writes_expected_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    output_path.mkdir()
    _write_benchmarks(
        input_path,
        {"version": "1.0", "bench.foo": {"param_names": ["a"]}},
    )
    base = dt.datetime(2024, 1, 1)
    for i in range(30):
        _write_result(
            input_path,
            f"sha{i:03d}",
            base + dt.timedelta(days=i),
            {"bench.foo": [[1.0], [["1"]]]},
        )

    build_parquet(input_path, output_path=output_path)

    parquet_path = output_path / PARQUET_DIRNAME
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    for col in BASE_COLUMNS:
        assert col in df.columns
    for col in DERIVED_COLUMNS:
        assert col in df.columns
    assert len(df) == 30


def test_build_parquet_appends_to_existing(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    output_path.mkdir()
    _write_benchmarks(
        input_path,
        {"version": "1.0", "bench.foo": {"param_names": ["a"]}},
    )
    base = dt.datetime(2024, 1, 1)
    for i in range(15):
        _write_result(
            input_path,
            f"sha{i:03d}",
            base + dt.timedelta(days=i),
            {"bench.foo": [[1.0], [["1"]]]},
        )
    build_parquet(input_path, output_path=output_path)

    for i in range(15, 25):
        _write_result(
            input_path,
            f"sha{i:03d}",
            base + dt.timedelta(days=i),
            {"bench.foo": [[1.0], [["1"]]]},
        )
    build_parquet(input_path, output_path=output_path)

    df = pd.read_parquet(output_path / PARQUET_DIRNAME)
    assert len(df) == 25
    assert df["sha"].nunique() == 25


def test_build_parquet_idempotent_on_rerun(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    output_path.mkdir()
    _write_benchmarks(
        input_path,
        {"version": "1.0", "bench.foo": {"param_names": ["a"]}},
    )
    base = dt.datetime(2024, 1, 1)
    for i in range(20):
        _write_result(
            input_path,
            f"sha{i:03d}",
            base + dt.timedelta(days=i),
            {"bench.foo": [[1.0], [["1"]]]},
        )

    build_parquet(input_path, output_path=output_path)
    build_parquet(input_path, output_path=output_path)

    df = pd.read_parquet(output_path / PARQUET_DIRNAME)
    assert len(df) == 20
    assert df["sha"].nunique() == 20


def test_detect_regression_isolates_groups() -> None:
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    step = [1.0] * (n // 2) + [10.0] * (n // 2)
    flat = [1.0] * n
    df = pd.DataFrame(
        {
            "name": ["bench.foo"] * (2 * n),
            "params": ["a=1"] * n + ["a=2"] * n,
            "date": list(dates) + list(dates),
            "result": step + flat,
        }
    )
    out = detect_regression(df, window_size=5)

    flagged_a = out[(out["params"] == "a=1") & out["is_regression"]]
    flagged_b = out[(out["params"] == "a=2") & out["is_regression"]]
    assert len(flagged_a) >= 1
    assert len(flagged_b) == 0


def _random_rows(n_shas: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_shas, freq="D")
    frames = []
    series = [
        *(("bench.a", f"x={i}", 0, 0.0) for i in range(100)),
        *(("bench.b", f"x={i}", 0, 0.2) for i in range(100)),
        ("bench.c", "", 0, 0.7),
        ("bench.d", "", n_shas - 12, 0.0),
    ]
    for name, params, start, missing in series:
        n = n_shas - start
        steps = rng.random(n) < 0.08
        result = np.where(steps, rng.choice([1.3, 1 / 1.3], n), 1.0).cumprod()
        result[rng.random(n) < missing] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "date": dates[start:],
                    "sha": [f"sha{i:04d}" for i in range(start, n_shas)],
                    "name": name,
                    "params": params,
                    "result": result,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["added_date"] = "2024-" + (df["date"].dt.month).astype(str).str.zfill(2)
    return pd.DataFrame(
        {
            "name": pd.array(df["name"], dtype="string[pyarrow]"),
            "params": pd.array(df["params"], dtype="string[pyarrow]"),
            "result": pd.array(df["result"], dtype="float64[pyarrow]"),
            "date": pd.array(df["date"], dtype=pd.ArrowDtype(pa.timestamp("us"))),
            "sha": pd.array(df["sha"], dtype="string[pyarrow]"),
            "added_date": pd.array(df["added_date"], dtype="string[pyarrow]"),
        }
    )


@pytest.mark.parametrize("slack", [0, 50])
@pytest.mark.parametrize("seed", [0, 12, 21])
def test_update_parquet_matches_full_recompute(
    tmp_path: Path, slack: int, seed: int
) -> None:
    window_size = 5
    rows = _random_rows(n_shas=120, seed=seed)
    new_shas = {
        "sha0060",
        "sha0061",
        "sha0090",
        *(f"sha{i:04d}" for i in range(115, 120)),
    }
    is_new = rows["sha"].isin(new_shas)
    new_rows = rows[is_new].assign(added_date="2024-12-31")
    all_rows = pd.concat([rows[~is_new], new_rows], ignore_index=True)

    path = tmp_path / "incremental"
    write_rows(detect_regression(rows[~is_new], window_size=window_size), path)
    update_parquet(
        path,
        new_rows,
        sha_dates=read_sha_dates(path),
        window_size=window_size,
        slack=slack,
    )

    expected_path = tmp_path / "full"
    write_rows(detect_regression(all_rows, window_size=window_size), expected_path)

    def load(p: Path) -> pd.DataFrame:
        df = read_rows(p, filters=None)
        return df.sort_values(["sha", "name", "params"]).reset_index(drop=True)

    expected = load(expected_path)
    assert expected["is_regression"].any()
    tm.assert_frame_equal(load(path), expected, check_like=True)


# === make_envs_diff ===


def test_make_envs_diff_includes_diff_output(tmp_path: Path) -> None:
    envs_dir = tmp_path / "envs"
    envs_dir.mkdir()
    (envs_dir / "prev.yml").write_text("pkg 1.0\n")
    (envs_dir / "curr.yml").write_text("pkg 2.0\n")
    benchmarks = pd.DataFrame({"sha": ["prev", "curr"]})

    result = make_envs_diff(envs_dir=envs_dir, benchmarks=benchmarks, sha="curr")

    assert result.startswith("Environment changes from previous commit:\n```\n")
    assert result.endswith("\n```")
    assert "1.0" in result
    assert "2.0" in result


# === stage_asv_inputs ===


def test_stage_asv_inputs_copies_metadata_when_no_compressed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "storage"
    asv = tmp_path / "asv"
    asvrunner_storage = storage / "data" / "results" / "asvrunner"
    asvrunner_storage.mkdir(parents=True)
    (storage / "data" / "envs").mkdir(parents=True)
    (storage / "data" / "results" / "benchmarks.json").write_text("{}")
    (asvrunner_storage / "machine.json").write_text("{}")

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    envs_dir = stage_asv_inputs(storage, asv=asv)

    assert (asv / "results" / "benchmarks.json").exists()
    assert (asv / "results" / "asvrunner" / "machine.json").exists()
    assert calls == []
    assert envs_dir.is_dir()


def test_stage_asv_inputs_decompresses_zst_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "storage"
    asv = tmp_path / "asv"
    asvrunner_storage = storage / "data" / "results" / "asvrunner"
    envs_storage = storage / "data" / "envs"
    asvrunner_storage.mkdir(parents=True)
    envs_storage.mkdir(parents=True)
    (storage / "data" / "results" / "benchmarks.json").write_text("{}")
    (asvrunner_storage / "machine.json").write_text("{}")
    (asvrunner_storage / "abc.json.zst").write_bytes(b"")
    (envs_storage / "abc.yml.zst").write_bytes(b"")

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    envs_dir = stage_asv_inputs(storage, asv=asv)

    assert len(calls) == 2
    json_call, yml_call = calls
    assert json_call == [
        "zstd",
        "-d",
        "-q",
        "--output-dir-flat",
        str(asv / "results" / "asvrunner"),
        str(asvrunner_storage / "abc.json.zst"),
    ]
    assert yml_call == [
        "zstd",
        "-d",
        "-q",
        "--output-dir-flat",
        str(envs_dir),
        str(envs_storage / "abc.yml.zst"),
    ]
    # Ephemeral envs dir must not live under storage, or it'll get committed
    # to the orphan branch by the subsequent push.
    assert storage not in envs_dir.parents
