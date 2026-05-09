from __future__ import annotations

import pandas as pd
import pytest

from ci.make_issues import (
    escape_ansi,
    get_commit_range,
    make_body,
    time_to_str,
)


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


def _benchmarks_with_shas(shas: list[str], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"sha": shas, "date": pd.to_datetime(dates)})


def test_get_commit_range_returns_prev_sha_to_sha() -> None:
    df = _benchmarks_with_shas(
        ["a", "b", "c"], ["2024-01-01", "2024-01-02", "2024-01-03"]
    )
    assert get_commit_range(benchmarks=df, sha="c") == "b...c"
    assert get_commit_range(benchmarks=df, sha="b") == "a...b"


def test_get_commit_range_orders_by_date_not_input_order() -> None:
    # Rows are not in date order; the function should still pick the previous
    # sha by date.
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
    # Both benchmarks for abc123 should appear; def456 should not.
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
    # Non-empty params should produce indented sub-bullets with severity.
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
    # bench.bar has empty params; severity should be on the benchmark line.
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
    # In shorten mode, no indented sub-bullets are emitted.
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
