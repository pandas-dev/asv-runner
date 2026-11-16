"""Break points: commits where results before and after must not be compared.

See ``breaks.toml`` for the list and why it exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tomllib

BREAKS_PATH = Path(__file__).parent / "breaks.toml"


@dataclass(frozen=True)
class Break:
    sha: str
    reason: str


def load(path: Path = BREAKS_PATH) -> list[Break]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    out = []
    for entry in data.get("break", []):
        sha, reason = entry.get("sha", ""), entry.get("reason", "")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(
                f"{path}: break sha must be a full 40-character SHA: {sha!r}"
            )
        if not reason:
            raise ValueError(f"{path}: break {sha} needs a reason")
        out.append(Break(sha, reason))
    return out


def positions(commits: pl.DataFrame, breaks: list[Break]) -> list[tuple[int, str]]:
    """(commit position, reason) for each break, in commit order.

    Raises if a break is not a stored commit, so a typo cannot silently
    disable it.
    """
    index = dict(commits.select("sha", "i").iter_rows())
    missing = [b.sha for b in breaks if b.sha not in index]
    if missing:
        raise ValueError(
            "breaks.toml lists commits that were not benchmarked: "
            + ", ".join(missing)
            + ". Use the first benchmarked commit measured the new way."
        )
    return sorted((index[b.sha], b.reason) for b in breaks)


def segments(commits: pl.DataFrame, breaks: list[Break]) -> pl.DataFrame:
    """Map each commit position ``i`` to its segment: the number of breaks at or
    before it."""
    starts = [i for i, _ in positions(commits, breaks)]
    return commits.select("i").with_columns(
        segment=pl.lit(pl.Series(starts, dtype=pl.Int32))
        .search_sorted(pl.col("i"), side="right")
        .cast(pl.Int32)
        if starts
        else pl.lit(0, dtype=pl.Int32)
    )
