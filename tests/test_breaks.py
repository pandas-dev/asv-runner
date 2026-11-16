from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from asv_runner import breaks
from asv_runner.regressions import flag

SHA = "c" * 40


def _commits(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sha": [f"{i:040x}" for i in range(n)],
            "i": pl.Series(range(n), dtype=pl.Int32),
        }
    )


def test_shipped_list_loads() -> None:
    assert breaks.load()


def test_load_validates(tmp_path: Path) -> None:
    path = tmp_path / "breaks.toml"
    path.write_text('[[break]]\nsha = "cae1cd626f"\nreason = "x"\n')
    with pytest.raises(ValueError, match="full 40-character SHA"):
        breaks.load(path)
    path.write_text(f'[[break]]\nsha = "{SHA}"\n')
    with pytest.raises(ValueError, match="needs a reason"):
        breaks.load(path)


def test_unknown_commit_raises() -> None:
    with pytest.raises(ValueError, match="not benchmarked"):
        breaks.positions(_commits(3), [breaks.Break(SHA, "x")])


def test_segments() -> None:
    commits = _commits(5)
    b = [breaks.Break(f"{3:040x}", "x"), breaks.Break(f"{1:040x}", "y")]
    seg = breaks.segments(commits, b)
    assert seg["segment"].to_list() == [0, 1, 1, 2, 2]
    assert breaks.positions(commits, b) == [(1, "y"), (3, "x")]


def test_break_splits_detection_windows() -> None:
    rows = [
        {
            "name": "x",
            "series": 0,
            "version": "v1",
            "i": i,
            "value": (1.0 if i < 40 else 5.0) * (1 + 0.001 * (i % 3)),
            "segment": int(i >= 40),
        }
        for i in range(80)
    ]
    df = pl.DataFrame(rows)
    assert flag(df.drop("segment")).filter("is_regression")["i"].to_list() == [40]
    assert flag(df).filter("is_regression").is_empty()
