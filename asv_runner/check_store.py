"""Check that an update to the Parquet store changed no stored rows.

Commits are only ever added to the store: once a commit is in ``runs.parquet``,
none of its rows change. ``check`` compares a store with an earlier copy of
itself and reports any stored row that changed or went missing, plus anything
in the new store that is inconsistent with itself.
"""

from __future__ import annotations

import filecmp
from pathlib import Path

import polars as pl
import pyarrow as pa

from asv_runner.store import Store, week_key

# Tables stored whole, with the columns that identify a row.
WHOLE_TABLES = {
    "runs.parquet": None,  # duplicates are checked in check()
    "env_files.parquet": ["sha"],
    "envs.parquet": None,
    "commit_info.parquet": ["sha"],
    "benchmark_defs.parquet": None,
}
PARTITION_KEYS = {"benchmarks": ["sha", "name"], "values": ["sha", "name", "combo"]}


def _exact(df: pl.DataFrame) -> pl.DataFrame:
    """``df`` with float columns replaced by their bit patterns.

    Polars treats -0.0 and 0.0 as equal, and the store must keep them apart.
    """
    table = df.to_arrow()
    for i, f in enumerate(table.schema):
        if pa.types.is_floating(f.type):
            as_int = pa.int64() if f.type.bit_width == 64 else pa.int32()
            col = pa.chunked_array(
                [chunk.view(as_int) for chunk in table.column(i).chunks], as_int
            )
            table = table.set_column(i, f.name, col)
    return pl.DataFrame(table)


def _schema(path: Path) -> dict:
    return dict(pl.read_parquet_schema(path))


def _check_whole_table(
    old: Store, new: Store, name: str, key: list[str] | None
) -> list[str]:
    old_path, new_path = old.root / name, new.root / name
    if not old_path.exists():
        return []
    if not new_path.exists():
        return [f"{name}: missing from the new store"]
    if _schema(old_path) != _schema(new_path):
        return [
            f"{name}: schema changed from {_schema(old_path)} to {_schema(new_path)}"
        ]
    problems = []
    old_df = _exact(pl.read_parquet(old_path))
    new_df = _exact(pl.read_parquet(new_path))
    lost = old_df.join(new_df, on=old_df.columns, how="anti", nulls_equal=True)
    if lost.height:
        problems.append(
            f"{name}: {lost.height} stored rows changed or removed, e.g. "
            f"{lost.head(3).to_dicts()}"
        )
    if key is not None:
        dupes = new_df.filter(pl.struct(key).is_duplicated())
        if dupes.height:
            problems.append(
                f"{name}: {dupes.height} rows share a {', '.join(key)} with "
                f"another row, e.g. {dupes.head(3).select(key).to_dicts()}"
            )
    return problems


def _check_partition(old: Store, new: Store, table: str, week: str) -> list[str]:
    old_path = old.partition_path(table, week)
    new_path = new.partition_path(table, week)
    label = f"{table}/week={week}"
    if not new_path.exists():
        return [f"{label}: missing from the new store"]
    if _schema(old_path) != _schema(new_path):
        return [
            f"{label}: schema changed from {_schema(old_path)} to {_schema(new_path)}"
        ]
    key = PARTITION_KEYS[table]
    old_df = pl.read_parquet(old_path)
    old_shas = old_df["sha"].unique().implode()
    new_df = pl.read_parquet(new_path).filter(pl.col("sha").is_in(old_shas))
    if not _exact(old_df.sort(key)).equals(_exact(new_df.sort(key))):
        return [f"{label}: rows of commits already stored changed"]
    return []


def _check_consistent(new: Store, week_shas: set[str], week: str) -> list[str]:
    """Problems within one week of the new store.

    ``week_shas`` are the commits that ``runs.parquet`` puts in ``week``.
    """
    problems = []
    shas = {}
    for table, key in PARTITION_KEYS.items():
        label = f"{table}/week={week}"
        df = new.read_partition(table, week)
        if df is None:
            problems.append(f"{label}: missing")
            continue
        shas[table] = set(df["sha"].unique().to_list())
        if stray := sorted(shas[table] - week_shas):
            problems.append(
                f"{label}: commits not in runs.parquet for this week: {stray[:5]}"
            )
        dupes = df.filter(pl.struct(key).is_duplicated())
        if dupes.height:
            problems.append(
                f"{label}: {dupes.height} rows share a {', '.join(key)} with "
                f"another row, e.g. {dupes.head(3).select(key).to_dicts()}"
            )
    if len(shas) == 2 and (extra := sorted(shas["values"] - shas["benchmarks"])):
        problems.append(
            f"values/week={week}: commits with no benchmarks rows: {extra[:5]}"
        )
    return problems


def _week_changed(old: Store, new: Store, week: str) -> bool:
    for table in PARTITION_KEYS:
        old_path = old.partition_path(table, week)
        new_path = new.partition_path(table, week)
        if not old_path.exists() or not new_path.exists():
            return True
        if not filecmp.cmp(old_path, new_path, shallow=False):
            return True
    return False


def check(old: Store, new: Store) -> list[str]:
    """Problems found comparing ``new`` with the earlier ``old``; empty if none.

    Every row stored in ``old`` must be unchanged in ``new``, and ``new`` must be
    consistent with itself. Weeks whose files differ from ``old`` are also
    rebuilt into result files, which is what a reader of the store relies on.
    """
    problems = []
    for name, key in WHOLE_TABLES.items():
        problems += _check_whole_table(old, new, name, key)

    old_weeks = {
        p.parent.name.removeprefix("week=")
        for table in PARTITION_KEYS
        for p in (old.root / table).glob("week=*/data.parquet")
    }
    for week in sorted(old_weeks):
        for table in PARTITION_KEYS:
            if old.partition_path(table, week).exists():
                problems += _check_partition(old, new, table, week)

    runs = new.runs()
    if dupes := sorted(runs.filter(pl.col("sha").is_duplicated())["sha"].unique()):
        problems.append(f"runs.parquet: commits stored more than once: {dupes[:5]}")
    shas_by_week: dict[str, set[str]] = {}
    for sha, date in runs.select("sha", "date").iter_rows():
        shas_by_week.setdefault(week_key(date), set()).add(sha)
    new_weeks = {
        p.parent.name.removeprefix("week=")
        for table in PARTITION_KEYS
        for p in (new.root / table).glob("week=*/data.parquet")
    }
    for week in sorted(new_weeks):
        problems += _check_consistent(new, shas_by_week.get(week, set()), week)

    for week in sorted(new_weeks):
        if not _week_changed(old, new, week):
            continue
        try:
            for _ in new.rebuild_week(week):
                pass
        except Exception as e:
            problems.append(f"week={week}: rebuilding results failed: {e!r}")
    return problems


def check_cmd(args) -> int:
    problems = check(Store(Path(args.old)), Store(Path(args.new)))
    for problem in problems:
        print(problem)
    if problems:
        print(f"check-store found {len(problems)} problems.")
        return 1
    print("check-store: no stored rows changed.")
    return 0
