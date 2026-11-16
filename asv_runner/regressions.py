"""Detect step changes (regressions and improvements) in benchmark series.

Same method as ``detect_regression`` in ``asv_runner/process_results.py``:
compare a window of ``window`` results with the ``window`` results right
before it. A regression is flagged when every result in the newer window is
slower than every result in the older one by more than ``1 / tol``; an
improvement is the mirror image. The flag is placed on the first commit of the
newer window, and only the first commit of a run of flags is kept.

Differences from the old code:

- Windows never span a change of benchmark code ``version``. A version change
  can shift results by itself; at ``4606f19379`` (pandas #65681, ``gc.collect``
  in the shared setup) a third of the affected series moved by more than 5%.
- Windows never span a break point listed in ``breaks.toml`` (a change of how
  results are measured).
- Improvements are detected too.
- Each flag carries ``step``: the median of the newer window over the median of
  the older one, minus one. It is less noisy than the change from the single
  previous result (``pct_change``, kept for the issue text).

A change can only be flagged once ``(window - 1) / 2`` later results exist, so
the latest ~10 commits are never flagged yet.
"""

from __future__ import annotations

import polars as pl

from asv_runner import breaks as breaks_mod
from asv_runner.series import Series

WINDOW = 21
TOL = 0.95
# Benchmarks processed together; bounds memory to a slice of the history.
BATCH = 100
REGRESSION = "regression"
IMPROVEMENT = "improvement"


def flag(values: pl.DataFrame, window: int = WINDOW, tol: float = TOL) -> pl.DataFrame:
    """Add detection columns to one row per result.

    Parameters
    ----------
    values
        Columns ``name``, ``series``, ``version``, ``i`` (commit position) and
        ``value``, plus optionally ``segment`` (see ``breaks.segments``). Rows
        with a null value are dropped first, so windows count results, not
        commits.

    Returns
    -------
    pl.DataFrame
        The non-null rows, sorted, with ``established_worst``,
        ``established_best``, ``is_regression``, ``is_improvement``, ``step``,
        ``pct_change`` and ``abs_change``.
    """
    half = (window - 1) // 2
    v = pl.col("value")
    gid = pl.col("gid")

    def same_group(k: int) -> pl.Expr:
        """The row ``k`` rows earlier (later, for negative k) is in the same series."""
        return (gid.shift(k) == gid).fill_null(False)

    # Rows are sorted by series, then commit. Rolling over the whole column and
    # discarding windows that cross a series boundary is one linear pass with
    # no per-group work.
    centered = same_group(half) & same_group(-half)

    def rolling(expr: pl.Expr) -> pl.Expr:
        return pl.when(centered).then(expr)

    def earlier(expr: pl.Expr, k: int) -> pl.Expr:
        return pl.when(same_group(k)).then(expr.shift(k))

    def first_of_run(m: pl.Expr) -> pl.Expr:
        m = m.fill_null(False)
        m = m & ~(m.shift(1).fill_null(False) & same_group(1))
        # Move the flag from the window's center to its first commit.
        return pl.when(same_group(-half)).then(m.shift(-half)).otherwise(False)

    group_cols = ["name", "series", "version"]
    if "segment" in values.columns:
        group_cols.append("segment")
    return (
        values.filter(v.is_not_null())
        .sort(*group_cols, "i")
        .with_columns(
            gid=pl.any_horizontal([pl.col(c) != pl.col(c).shift(1) for c in group_cols])
            .fill_null(True)
            .cum_sum()
        )
        .with_columns(
            established_worst=rolling(v.rolling_max(window, center=True)),
            established_best=rolling(v.rolling_min(window, center=True)),
            median=rolling(v.rolling_median(window, center=True)),
        )
        .with_columns(
            up=earlier(pl.col("established_worst"), window)
            < tol * pl.col("established_best"),
            down=pl.col("established_worst")
            < tol * earlier(pl.col("established_best"), window),
            step=pl.col("median") / earlier(pl.col("median"), window) - 1,
        )
        .with_columns(
            is_regression=first_of_run(pl.col("up")),
            is_improvement=first_of_run(pl.col("down")),
            step=earlier(pl.col("step"), -half),
            pct_change=v / earlier(v, 1) - 1,
            abs_change=v - earlier(v, 1),
        )
        .drop("up", "down", "median", "gid")
    )


def detect(
    series: Series,
    breaks: list[breaks_mod.Break] | None = None,
    window: int = WINDOW,
    tol: float = TOL,
) -> pl.DataFrame:
    """Flagged rows for every stored series.

    ``breaks`` defaults to the list in ``breaks.toml``.

    Returns
    -------
    pl.DataFrame
        One row per detection: ``kind``, ``name``, ``series``, ``params``
        (parameter values), ``i``, ``sha``, ``version``, ``value``, ``step``,
        ``pct_change``, ``abs_change``, ``established_worst`` and
        ``established_best``.
    """
    if breaks is None:
        breaks = breaks_mod.load()
    segments = breaks_mod.segments(series.commits, breaks)
    names = sorted(series.params)
    parts = []
    for start in range(0, len(names), BATCH):
        batch = names[start : start + BATCH]
        values = series.values(batch).join(segments, on="i")
        parts.append(
            flag(values, window=window, tol=tol).filter(
                pl.col("is_regression") | pl.col("is_improvement")
            )
        )
    flagged = (
        pl.concat(parts)
        .with_columns(
            kind=pl.when(pl.col("is_regression"))
            .then(pl.lit(REGRESSION))
            .otherwise(pl.lit(IMPROVEMENT))
        )
        .join(series.commits.select("i", "sha"), on="i")
    )
    params = [
        list(series.params[name][sid])
        for name, sid in flagged.select("name", "series").iter_rows()
    ]
    return (
        flagged.with_columns(params=pl.Series(params, dtype=pl.List(pl.String)))
        .select(
            "kind",
            "name",
            "series",
            "params",
            "i",
            "sha",
            "version",
            "value",
            "step",
            "pct_change",
            "abs_change",
            "established_worst",
            "established_best",
        )
        .sort("i", "name", "series")
    )
