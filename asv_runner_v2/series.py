"""Load stored results as time series, one per (benchmark, parameter combination).

asv numbers parameter combinations in ``itertools.product`` order of the
benchmark's parameter grid. A grid can change over time, so the same
combination can have a different index in different commits; series are keyed
by the parameter values instead and numbered per benchmark.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import polars as pl

from asv_runner_v2.store import Store


def commits(store: Store) -> pl.DataFrame:
    """Stored commits in date order, with their position ``i`` in that order."""
    return (
        store.runs()
        .select("sha", "date")
        .sort("date", "sha")
        .with_row_index("i")
        .with_columns(pl.col("i").cast(pl.Int32))
    )


def _series_keys(grids: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[tuple]]]:
    rows: dict[str, list] = {"grid": [], "combo": [], "series": []}
    series: dict[str, dict[tuple, int]] = {}
    for grid_id, name, params in grids.sort("grid").iter_rows():
        grid = json.loads(params) if params else None
        ids = series.setdefault(name, {})
        for combo, values in enumerate(itertools.product(*grid) if grid else [()]):
            rows["grid"].append(grid_id)
            rows["combo"].append(combo)
            rows["series"].append(ids.setdefault(values, len(ids)))
    keys = pl.DataFrame(
        rows, schema={"grid": pl.UInt32, "combo": pl.Int32, "series": pl.UInt32}
    )
    return keys, {name: list(ids) for name, ids in series.items()}


@dataclass
class Series:
    store: Store
    commits: pl.DataFrame
    """sha, date, i for every stored commit."""
    runs: pl.DataFrame
    """i, name, grid, version, started_at: one row per (commit, benchmark) run."""
    keys: pl.DataFrame
    """grid, combo, series: maps a grid's combination index to the series id."""
    params: dict[str, list[tuple]]
    """Each benchmark's series, as tuples of parameter values, by series id."""

    def values(self, names: list[str] | None = None) -> pl.DataFrame:
        """name, series, i, version, value: one row per stored result.

        ``value`` is null for a failed or skipped combination; integer results
        are cast to float. ``names`` limits the result to those benchmarks.
        """
        scan = pl.scan_parquet(self.store.root / "values" / "*" / "data.parquet")
        if names is not None:
            scan = scan.filter(pl.col("name").is_in(names))
        # Collected before joining `keys`: as one lazy query, Polars 2.1 joins
        # `runs` to `keys` first, which is one row per result for the whole
        # history regardless of `names` (~25 GB on the full store).
        with_grid = (
            scan.select(
                "sha",
                "name",
                "combo",
                pl.coalesce("result", pl.col("result_int").cast(pl.Float64))
                .fill_nan(None)
                .alias("value"),
            )
            .join(self.commits.lazy().select("sha", "i"), on="sha")
            .join(
                self.runs.lazy().select("i", "name", "grid", "version"),
                on=["i", "name"],
            )
            .collect()
        )
        return with_grid.join(self.keys, on=["grid", "combo"]).select(
            "name", "series", "i", "version", "value"
        )


def load(store: Store) -> Series:
    all_commits = commits(store)
    bench_scan = pl.scan_parquet(store.root / "benchmarks" / "*" / "data.parquet")
    grids = (
        bench_scan.select("name", pl.col("params").fill_null(""))
        .unique()
        .collect()
        .with_row_index("grid")
        .select("grid", "name", "params")
    )
    keys, params = _series_keys(grids)
    runs = (
        bench_scan.select(
            "sha", "name", pl.col("params").fill_null(""), "version", "started_at"
        )
        .join(grids.lazy(), on=["name", "params"])
        .join(all_commits.lazy().select("sha", "i"), on="sha")
        .select("i", "name", "grid", "version", "started_at")
        .collect()
    )
    return Series(store=store, commits=all_commits, runs=runs, keys=keys, params=params)
