from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from asv_runner import store as store_mod
from asv_runner.store import (
    ENV_SUFFIX,
    RESULT_SUFFIX,
    FileTooLargeError,
    Store,
    read_zst,
    week_key,
)
from tests._results_helpers import PIXI_ENV, add_commit, write_benchmark_defs

# Commit dates in ms: two in ISO week 2026-W10, one in 2026-W12.
WEEK10_A = 1772932225000  # 2026-03-08 (Sunday, still W10)
WEEK10_B = 1772845825000  # 2026-03-07
WEEK12 = 1773900000000  # 2026-03-19


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    d = tmp_path / "input"
    add_commit(d, "aaa", WEEK10_A)
    add_commit(d, "bbb", WEEK10_B, env=PIXI_ENV)
    write_benchmark_defs(d, {"a.b": {"version": "v1", "param_names": ["x"]}})
    return d


def assert_round_trips(store: Store, input_dir: Path) -> None:
    """Every stored result and env file rebuilds to the original exactly."""
    results_dir = input_dir / "data" / "results" / "asvrunner"
    rebuilt = {
        sha: result
        for week in store.weeks()
        for sha, result in store.rebuild_week(week)
    }
    originals = {
        path.name.removesuffix(RESULT_SUFFIX): json.loads(read_zst(path))
        for path in results_dir.glob(f"*{RESULT_SUFFIX}")
    }
    assert rebuilt.keys() == originals.keys()
    for sha, result in rebuilt.items():
        # Compare serialized text so int vs float, NaN vs null, and key order
        # all count as differences.
        assert json.dumps(result) == json.dumps(originals[sha]), sha

    env_originals = {
        path.name.removesuffix(ENV_SUFFIX): read_zst(path).decode()
        for path in (input_dir / "data" / "envs").glob(f"*{ENV_SUFFIX}")
    }
    assert store.env_text() == env_originals


def test_update_round_trips(tmp_path: Path, input_dir: Path) -> None:
    store = Store(tmp_path / "store")
    report = store.update(input_dir, workers=1)
    assert sorted(report.results_added) == ["aaa", "bbb"]
    assert report.weeks_written == ["2026-W10"]
    assert sorted(report.envs_added) == ["aaa", "bbb"]
    assert report.benchmark_defs_added == 1
    assert store.weeks() == ["2026-W10"]

    assert_round_trips(store, input_dir)


def test_rows_sorted_within_partition(tmp_path: Path, input_dir: Path) -> None:
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)
    values = store.read_partition("values", "2026-W10")
    assert values is not None
    name = "algorithms.Duplicated.time_duplicated"
    combo0 = values.filter((pl.col("name") == name) & (pl.col("combo") == 0))
    # bbb is the earlier commit.
    assert combo0["sha"].to_list() == ["bbb", "aaa"]


def test_second_update_is_noop(tmp_path: Path, input_dir: Path) -> None:
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)
    report = store.update(input_dir, workers=1)
    assert report.results_added == []
    assert report.weeks_written == []
    assert report.envs_added == []
    assert report.benchmark_defs_added == 0


def test_update_rewrites_only_touched_weeks(tmp_path: Path, input_dir: Path) -> None:
    store = Store(tmp_path / "store")
    add_commit(input_dir, "ccc", WEEK12)
    store.update(input_dir, workers=1)
    w12 = store.partition_path("values", "2026-W12")
    mtime = w12.stat().st_mtime_ns

    # A backfilled commit in week 10 leaves week 12 alone.
    add_commit(input_dir, "ddd", WEEK10_A - 1000)
    report = store.update(input_dir, workers=1)
    assert report.results_added == ["ddd"]
    assert report.weeks_written == ["2026-W10"]
    assert w12.stat().st_mtime_ns == mtime
    assert_round_trips(store, input_dir)


def test_interrupted_update_does_not_duplicate(tmp_path: Path, input_dir: Path) -> None:
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)
    # Simulate a crash after the partitions were written but before runs.parquet.
    runs = store.runs().filter(pl.col("sha") != "aaa")
    runs.write_parquet(store.runs_path)

    report = store.update(input_dir, workers=1)
    assert report.results_added == ["aaa"]
    values = store.read_partition("values", "2026-W10")
    assert values is not None
    assert values.filter(pl.col("sha") == "aaa").height == 6
    assert_round_trips(store, input_dir)


def test_mismatched_commit_hash_raises(tmp_path: Path, input_dir: Path) -> None:
    src = input_dir / "data" / "results" / "asvrunner" / "aaa.json.zst"
    src.rename(src.with_name("zzz.json.zst"))
    with pytest.raises(ValueError, match="holds results for aaa"):
        Store(tmp_path / "store").update(input_dir, workers=1)


def test_file_size_limit(
    tmp_path: Path, input_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_mod, "MAX_FILE_BYTES", 100)
    store = Store(tmp_path / "store")
    with pytest.raises(FileTooLargeError, match="Partition more finely"):
        store.update(input_dir, workers=1)
    # Nothing is marked stored, and no partial files are left behind.
    assert not store.runs_path.exists()
    assert not list(store.root.rglob("*.tmp"))


def test_benchmark_defs_keep_old_entries(tmp_path: Path, input_dir: Path) -> None:
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)
    write_benchmark_defs(input_dir, {"a.b": {"version": "v2", "param_names": ["y"]}})
    assert store.update(input_dir, workers=1).benchmark_defs_added == 1
    defs = pl.read_parquet(store.benchmark_defs_path)
    assert defs["version"].to_list() == ["v1", "v2"]
    assert json.loads(defs["definition"][0])["param_names"] == ["x"]


def test_week_key_uses_iso_year() -> None:
    import datetime as dt

    # 2027-01-01 is a Friday in ISO week 2026-W53.
    assert week_key(dt.datetime(2027, 1, 1, tzinfo=dt.UTC)) == "2026-W53"


def _git(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_add_commit_info(tmp_path: Path) -> None:
    repo = tmp_path / "pandas"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@test")
    _git(
        repo,
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "PERF: faster thing (#123)\n\nBody.",
    )
    first = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "RLS: 3.0.0")
    second = _git(repo, "rev-parse", "HEAD")

    input_dir = tmp_path / "input"
    add_commit(input_dir, first, WEEK10_B)
    add_commit(input_dir, second, WEEK10_A)
    add_commit(input_dir, "f" * 40, WEEK12)  # not in the clone
    store = Store(tmp_path / "store")
    report = store.update(input_dir, workers=1, pandas_repo=repo)
    assert report.commit_info_added == 2
    info = {sha: (title, pr) for sha, title, pr in store.commit_info().iter_rows()}
    assert info == {
        first: ("PERF: faster thing (#123)", 123),
        second: ("RLS: 3.0.0", None),
    }
    # Already stored: nothing to add; the missing commit is retried and skipped.
    assert store.add_commit_info(repo) == 0
