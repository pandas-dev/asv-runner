from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from asv_runner.check_store import check
from asv_runner.store import Store
from tests._results_helpers import PIXI_ENV, add_commit, write_benchmark_defs

WEEK10_A = 1772932225000  # 2026-03-08
WEEK10_B = 1772845825000  # 2026-03-07
WEEK12 = 1773900000000  # 2026-03-19
DUPLICATED = "algorithms.Duplicated.time_duplicated"


@pytest.fixture
def stores(tmp_path: Path) -> tuple[Store, Store]:
    """A store with two commits, and a copy updated with a third in a new week."""
    input_dir = tmp_path / "input"
    add_commit(input_dir, "aaa", WEEK10_A)
    add_commit(input_dir, "bbb", WEEK10_B, env=PIXI_ENV)
    write_benchmark_defs(input_dir, {"a.b": {"version": "v1"}})
    old = Store(tmp_path / "old")
    old.update(input_dir, workers=1)

    shutil.copytree(old.root, tmp_path / "new")
    new = Store(tmp_path / "new")
    add_commit(input_dir, "ccc", WEEK12)
    write_benchmark_defs(input_dir, {"a.b": {"version": "v2"}})
    new.update(input_dir, workers=1)
    return old, new


def rewrite(path: Path, func: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
    pl.read_parquet(path).pipe(func).write_parquet(path)


def test_update_passes(stores: tuple[Store, Store]) -> None:
    old, new = stores
    assert check(old, new) == []


def test_empty_old_store_passes(tmp_path: Path, stores: tuple[Store, Store]) -> None:
    _, new = stores
    assert check(Store(tmp_path / "empty"), new) == []


def test_unchanged_store_passes(stores: tuple[Store, Store]) -> None:
    old, _ = stores
    assert check(old, old) == []


def set_result(sha: str, value: float) -> Callable[[pl.DataFrame], pl.DataFrame]:
    def func(df: pl.DataFrame) -> pl.DataFrame:
        hit = (
            (pl.col("sha") == sha)
            & (pl.col("name") == DUPLICATED)
            & (pl.col("combo") == 0)
        )
        return df.with_columns(
            result=pl.when(hit).then(pl.lit(value)).otherwise(pl.col("result"))
        )

    return func


@pytest.mark.parametrize("value", [2.0, float("nan")])
def test_changed_value_fails(stores: tuple[Store, Store], value: float) -> None:
    old, new = stores
    rewrite(new.partition_path("values", "2026-W10"), set_result("aaa", value))
    assert check(old, new) == [
        "values/week=2026-W10: rows of commits already stored changed"
    ]


def test_negative_zero_is_not_zero(stores: tuple[Store, Store]) -> None:
    old, new = stores
    for store, value in ((old, -0.0), (new, 0.0)):
        rewrite(store.partition_path("values", "2026-W10"), set_result("aaa", value))
    assert check(old, new) == [
        "values/week=2026-W10: rows of commits already stored changed"
    ]


def test_removed_partition_row_fails(stores: tuple[Store, Store]) -> None:
    old, new = stores
    rewrite(
        new.partition_path("benchmarks", "2026-W10"),
        lambda df: df.filter(
            ~((pl.col("sha") == "aaa") & (pl.col("name") == DUPLICATED))
        ),
    )
    assert check(old, new) == [
        "benchmarks/week=2026-W10: rows of commits already stored changed"
    ]


def test_missing_partition_fails(stores: tuple[Store, Store]) -> None:
    old, new = stores
    new.partition_path("values", "2026-W10").unlink()
    problems = check(old, new)
    assert "values/week=2026-W10: missing from the new store" in problems
    assert "values/week=2026-W10: missing" in problems


def test_changed_schema_fails(stores: tuple[Store, Store]) -> None:
    old, new = stores
    rewrite(
        new.partition_path("values", "2026-W10"),
        lambda df: df.with_columns(pl.col("combo").cast(pl.Int64)),
    )
    [problem] = check(old, new)
    assert problem.startswith("values/week=2026-W10: schema changed")


@pytest.mark.parametrize(
    ("name", "func"),
    [
        (
            "runs.parquet",
            lambda df: df.with_columns(
                meta=pl.when(pl.col("sha") == "aaa")
                .then(pl.col("meta").str.replace("existing", "other"))
                .otherwise(pl.col("meta"))
            ),
        ),
        ("env_files.parquet", lambda df: df.filter(pl.col("sha") != "bbb")),
        (
            "envs.parquet",
            lambda df: df.with_columns(
                version=pl.when(pl.col("package") == "numpy")
                .then(pl.lit("0.0"))
                .otherwise(pl.col("version"))
            ),
        ),
        ("benchmark_defs.parquet", lambda df: df.filter(pl.col("version") != "v1")),
    ],
)
def test_changed_table_fails(
    stores: tuple[Store, Store],
    name: str,
    func: Callable[[pl.DataFrame], pl.DataFrame],
) -> None:
    old, new = stores
    rewrite(new.root / name, func)
    problems = check(old, new)
    assert problems
    assert problems[0].startswith(f"{name}: ")
    assert "stored rows changed or removed" in problems[0]


def test_missing_table_fails(stores: tuple[Store, Store]) -> None:
    old, new = stores
    (new.root / "envs.parquet").unlink()
    assert check(old, new) == ["envs.parquet: missing from the new store"]


def test_duplicate_rows_fail(stores: tuple[Store, Store]) -> None:
    old, new = stores
    rewrite(
        new.partition_path("values", "2026-W12"),
        lambda df: pl.concat([df, df.head(1)]),
    )
    [problem] = check(old, new)
    assert problem.startswith("values/week=2026-W12: 2 rows share a sha, name, combo")


def test_commit_in_wrong_week_fails(stores: tuple[Store, Store]) -> None:
    old, new = stores
    w12 = new.read_partition("benchmarks", "2026-W12")
    assert w12 is not None
    rewrite(
        new.partition_path("benchmarks", "2026-W10"), lambda df: pl.concat([df, w12])
    )
    problems = check(old, new)
    assert (
        "benchmarks/week=2026-W10: commits not in runs.parquet for this week: ['ccc']"
        in problems
    )


def test_values_without_benchmarks_fail(stores: tuple[Store, Store]) -> None:
    old, new = stores
    rewrite(
        new.partition_path("benchmarks", "2026-W12"),
        lambda df: df.filter(pl.col("sha") != "ccc"),
    )
    assert check(old, new) == [
        "values/week=2026-W12: commits with no benchmarks rows: ['ccc']"
    ]


def test_rebuild_failure_fails(stores: tuple[Store, Store]) -> None:
    old, new = stores
    # A new commit's row can change freely, but its week must still rebuild.
    rewrite(
        new.runs_path,
        lambda df: df.with_columns(
            meta=pl.when(pl.col("sha") == "ccc")
            .then(pl.lit("not json"))
            .otherwise(pl.col("meta"))
        ),
    )
    [problem] = check(old, new)
    assert problem.startswith("week=2026-W12: rebuilding results failed: ")
