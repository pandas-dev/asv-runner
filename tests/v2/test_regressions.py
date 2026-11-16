from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from asv_runner.process_results import detect_regression
from asv_runner_v2.regressions import flag


def _random_series(seed: int, n: int = 300) -> list[float | None]:
    rng = np.random.default_rng(seed)
    level = np.ones(n)
    for at in rng.choice(np.arange(30, n - 30), size=4, replace=False):
        level[at:] *= rng.choice([0.7, 0.85, 1.2, 1.5])
    values = level * (1 + rng.normal(0, 0.02, n))
    out: list[float | None] = list(values)
    for at in rng.choice(n, size=5, replace=False):
        out[at] = None
    return out


@pytest.mark.parametrize("seed", range(5))
def test_matches_old_detect_regression(seed: int) -> None:
    names = ["a.b", "a.c"]
    rows: list[dict] = []
    for k, name in enumerate(names):
        for series in range(2):
            vals = _random_series(seed * 10 + k * 2 + series)
            rows.extend(
                {"name": name, "series": series, "i": i, "value": v}
                for i, v in enumerate(vals)
            )
    new = flag(pl.DataFrame(rows).with_columns(version=pl.lit("v1")))

    old_input = pd.DataFrame(
        {
            "name": [r["name"] for r in rows],
            "params": [str(r["series"]) for r in rows],
            "date": pd.Timestamp("2026-01-01")
            + pd.to_timedelta([r["i"] for r in rows], unit="h"),
            "result": pd.array([r["value"] for r in rows], dtype="float64[pyarrow]"),
        }
    )
    old = detect_regression(old_input)
    old_flags = {
        (n, int(p), int((d - pd.Timestamp("2026-01-01")) / pd.Timedelta("1h")))
        for n, p, d in old.loc[
            old["is_regression"], ["name", "params", "date"]
        ].itertuples(index=False)
    }
    new_flags = set(
        new.filter("is_regression").select("name", "series", "i").iter_rows()
    )
    assert new_flags == old_flags
    assert new_flags, "the random series should contain regressions"

    # pct_change and abs_change match on every row.
    old_sorted = old.sort_values(["name", "params", "date"])
    np.testing.assert_allclose(
        new["pct_change"].to_numpy()[1:],
        old_sorted["pct_change"].to_numpy(dtype=float, na_value=np.nan)[1:],
        equal_nan=True,
    )


def _step(before: float, after: float, n: int = 40, at: int = 20, version_at=None):
    rows = []
    for i in range(n):
        value = (before if i < at else after) * (1 + 0.001 * (i % 3))
        version = "v2" if version_at is not None and i >= version_at else "v1"
        rows.append(
            {"name": "x", "series": 0, "version": version, "i": i, "value": value}
        )
    return pl.DataFrame(rows)


def test_regression_flagged_at_first_slow_commit() -> None:
    # A full window before the step (21 results) and after it (10 more than
    # the flagged commit) are needed; 40 on each side is plenty.
    out = flag(_step(1.0, 1.5, n=80, at=40))
    assert out.filter("is_regression")["i"].to_list() == [40]
    assert out.filter("is_improvement").is_empty()
    step = out.filter("is_regression")["step"][0]
    assert step == pytest.approx(0.5, abs=0.01)


def test_improvement_flagged() -> None:
    out = flag(_step(1.0, 0.5, n=80, at=40))
    assert out.filter("is_improvement")["i"].to_list() == [40]
    assert out.filter("is_regression").is_empty()


def test_version_change_splits_windows() -> None:
    # The step coincides with a benchmark code change: not a regression.
    out = flag(_step(1.0, 1.5, n=80, at=40, version_at=40))
    assert out.filter("is_regression").is_empty()


def test_too_recent_to_flag() -> None:
    # Only 9 results after the step: the newer window is incomplete.
    out = flag(_step(1.0, 1.5, n=49, at=40))
    assert out.filter("is_regression").is_empty()


def test_null_values_are_skipped() -> None:
    df = _step(1.0, 1.5, n=80, at=40).with_columns(
        value=pl.when(pl.col("i") == 10).then(None).otherwise(pl.col("value"))
    )
    out = flag(df)
    assert 10 not in out["i"].to_list()
    assert out.filter("is_regression")["i"].to_list() == [40]
