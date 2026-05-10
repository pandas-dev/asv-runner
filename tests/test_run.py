from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ci.run import (
    BASE_COLUMNS,
    DERIVED_COLUMNS,
    PARQUET_DIRNAME,
    build_new_rows,
    build_parquet,
    cmd_claim,
    detect_regression,
    escape_ansi,
    get_commit_range,
    load_existing,
    make_body,
    orphan_push_with_retry,
    pick_next_sha,
    read_existing_shas,
    time_to_str,
)

# === pick_next_sha ===


class _FakeCompleted:
    def __init__(self, stdout: bytes) -> None:
        self.stdout = stdout


def _fake_subprocess_run(shas: list[str]) -> Callable[..., _FakeCompleted]:
    out = "\n".join(f"{sha} commit message {sha}" for sha in shas).encode()

    def _run(cmd: Any, **kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(out)

    return _run


def test_pick_next_sha_no_existing_picks_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(["sha1111", "sha2222"]))
    assert pick_next_sha(tmp_path, set()) == "sha1111"


def test_pick_next_sha_skips_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_subprocess_run(["sha1111", "sha2222", "sha3333"]),
    )
    assert pick_next_sha(tmp_path, {"sha1111", "sha2222"}) == "sha3333"


def test_pick_next_sha_all_existing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(["sha1111", "sha2222"]))
    assert pick_next_sha(tmp_path, {"sha1111", "sha2222"}) is None


def test_read_existing_shas_missing_file(tmp_path: Path) -> None:
    assert read_existing_shas(tmp_path / "missing.txt") == set()


def test_read_existing_shas_ignores_blank_lines(tmp_path: Path) -> None:
    shas_path = tmp_path / "shas.txt"
    shas_path.write_text("sha1111\n\nsha2222\n")
    # Blank line collapses to empty string in the set; pick_next_sha filters
    # those out via the non-empty guard.
    assert "sha1111" in read_existing_shas(shas_path)
    assert "sha2222" in read_existing_shas(shas_path)


# === time_to_str / escape_ansi ===


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.5, "1.500s"),
        (1.0, "1.000s"),
        (0.5, "500.000ms"),
        (0.001, "1.000ms"),
        (0.0005, "500.000us"),
        (0.000001, "1.000us"),
        (0.0000005, "500.000ns"),
    ],
)
def test_time_to_str_positive(value: float, expected: str) -> None:
    assert time_to_str(value) == expected


def test_escape_ansi_passes_plain_text() -> None:
    assert escape_ansi("hello world") == "hello world"


def test_escape_ansi_strips_color_codes() -> None:
    colored = "\x1b[31mhello\x1b[0m world"
    assert escape_ansi(colored) == "hello world"


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
        base_url="https://github.com/pandas-dev/pandas/compare/",
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
    )
    assert (
        "[Commit Range](https://github.com/pandas-dev/pandas/compare/aaa...bbb)" in body
    )


def test_make_body_only_includes_target_sha_regressions() -> None:
    body = make_body(
        base_url="https://github.com/pandas-dev/pandas/compare/",
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
    )
    assert "bench.foo" in body
    assert "bench.bar" in body
    assert "def456" not in body


def test_make_body_renders_param_sublist_for_nonempty_params() -> None:
    body = make_body(
        base_url="https://github.com/pandas-dev/pandas/compare/",
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
        base_url="https://github.com/pandas-dev/pandas/compare/",
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
        base_url="https://github.com/pandas-dev/pandas/compare/",
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
    )
    short = make_body(
        base_url="https://github.com/pandas-dev/pandas/compare/",
        commit_range="aaa...bbb",
        benchmarks=_regression_frame(),
        sha="abc123",
        shorten=True,
    )
    assert len(short) < len(full)
    assert "   - [ ]" not in short
    assert "10.000% (500.000ms)" in short


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
        base_url="https://example.com/",
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

    build_parquet(input_path, output_path)

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
    build_parquet(input_path, output_path)

    for i in range(15, 25):
        _write_result(
            input_path,
            f"sha{i:03d}",
            base + dt.timedelta(days=i),
            {"bench.foo": [[1.0], [["1"]]]},
        )
    build_parquet(input_path, output_path)

    df = pd.read_parquet(output_path / PARQUET_DIRNAME)
    assert len(df) == 25
    assert df["sha"].nunique() == 25


# === orphan_push_with_retry / cmd_claim integration ===


def _init_remote_and_storage(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare 'remote' with an initial commit on `pandas_test`,
    then clone it into `storage` so tests can push/refetch.
    """
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    storage = tmp_path / "storage"

    subprocess.run(["git", "init", "--bare", str(remote)], check=True)

    seed.mkdir()
    subprocess.run(["git", "init", "-b", "pandas_test", str(seed)], check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=seed, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=seed, check=True)
    (seed / "data").mkdir()
    (seed / "data" / "shas.txt").write_text("")
    subprocess.run(["git", "add", "-A"], cwd=seed, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=seed, check=True)
    subprocess.run(["git", "push", str(remote), "pandas_test"], cwd=seed, check=True)

    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(storage)],
        check=True,
    )
    return remote, storage


def _init_pandas_repo(tmp_path: Path, n_commits: int = 2) -> tuple[Path, list[str]]:
    """Create a tiny git repo with n synthetic commits and return its path
    and the resulting SHAs (most-recent first)."""
    repo = tmp_path / "pandas"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
    shas: list[str] = []
    for i in range(n_commits):
        (repo / "f").write_text(str(i))
        subprocess.run(["git", "add", "f"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", f"commit {i}"], cwd=repo, check=True)
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        shas.append(sha)
    return repo, list(reversed(shas))


def test_orphan_push_with_retry_roundtrip(tmp_path: Path) -> None:
    remote, storage = _init_remote_and_storage(tmp_path)

    def modify(repo: Path) -> bool:
        (repo / "data" / "shas.txt").write_text("hello\n")
        return True

    pushed = orphan_push_with_retry(storage, "pandas_test", "msg", modify)
    assert pushed

    verify = tmp_path / "verify"
    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(verify)],
        check=True,
    )
    assert (verify / "data" / "shas.txt").read_text() == "hello\n"


def test_orphan_push_with_retry_skip_when_modify_returns_false(
    tmp_path: Path,
) -> None:
    _, storage = _init_remote_and_storage(tmp_path)

    def modify(repo: Path) -> bool:
        return False

    pushed = orphan_push_with_retry(storage, "pandas_test", "msg", modify)
    assert pushed is False


def test_cmd_claim_picks_sha_and_pushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, storage = _init_remote_and_storage(tmp_path)
    pandas_repo, shas = _init_pandas_repo(tmp_path, n_commits=2)
    output_file = tmp_path / "github_output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    args = argparse.Namespace(
        storage_dir=str(storage),
        repo_dir=str(pandas_repo),
        branch="pandas_test",
    )
    cmd_claim(args)

    out = output_file.read_text()
    head_sha = shas[0]
    assert f"sha={head_sha}" in out
    assert "new_commit=yes" in out

    verify = tmp_path / "verify"
    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(verify)],
        check=True,
    )
    assert (verify / "data" / "shas.txt").read_text().strip() == head_sha


def test_cmd_claim_no_new_commit_when_all_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, storage = _init_remote_and_storage(tmp_path)
    pandas_repo, shas = _init_pandas_repo(tmp_path, n_commits=2)
    (storage / "data" / "shas.txt").write_text("\n".join(shas) + "\n")
    output_file = tmp_path / "github_output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    args = argparse.Namespace(
        storage_dir=str(storage),
        repo_dir=str(pandas_repo),
        branch="pandas_test",
    )
    cmd_claim(args)

    out = output_file.read_text()
    assert "sha=NONE" in out
    assert "new_commit=no" in out
