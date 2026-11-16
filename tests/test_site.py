from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from asv_runner import series, site
from asv_runner.breaks import Break
from asv_runner.store import Store
from tests._results_helpers import (
    PIXI_ENV,
    add_commit,
    make_result,
    write_benchmark_defs,
    write_zst,
)

DAY_MS = 86_400_000
START = 1772845825000  # 2026-03-07

DUPLICATED = "algorithms.Duplicated.time_duplicated"


@pytest.fixture
def built(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    add_commit(input_dir, "a" * 40, START)
    add_commit(input_dir, "b" * 40, START + DAY_MS)
    add_commit(input_dir, "c" * 40, START + 2 * DAY_MS, env=PIXI_ENV)
    write_benchmark_defs(
        input_dir,
        {
            DUPLICATED: {
                "version": "v1",
                "param_names": ["unique", "keep"],
                "unit": "seconds",
                "type": "time",
            }
        },
    )
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)
    out = tmp_path / "site"
    site.build(store, out, breaks=[Break("b" * 40, "runner changed")])
    return out


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def test_index(built: Path) -> None:
    index = _load(built / "data" / "index.json")
    assert index["commits"]["sha"] == ["a" * 40, "b" * 40, "c" * 40]
    assert index["breaks"] == [{"i": 1, "reason": "runner changed"}]
    assert index["benchmarks"][DUPLICATED] == {
        "param_names": ["unique", "keep"],
        "param_values": [["True", "False"], ["'first'", "'last'"]],
        "unit": "seconds",
        "type": "time",
    }
    # No definition stored: parameter names are made up from the grid size.
    assert index["benchmarks"]["series_methods.Peakmem.peakmem_x"]["param_names"] == []
    # Switching from conda export to pixi list is one format change, not a
    # list of every package.
    assert index["env_changes"] == [
        {"i": 2, "format_change": ["conda-export", "pixi-list"], "changes": []}
    ]
    for name in ("index.html", "app.js", "style.css"):
        assert (built / name).exists()


def test_graph(built: Path) -> None:
    graph = _load(built / "data" / "graphs" / f"{DUPLICATED}.json")
    assert graph["commits"] == [0, 1, 2]
    assert graph["series"] == [
        ["True", "'first'"],
        ["True", "'last'"],
        ["False", "'first'"],
        ["False", "'last'"],
    ]
    # NaN and None results become null; values are rounded for display.
    assert graph["values"][0] == [1.5e-06] * 3
    assert graph["values"][1] == [None] * 3
    assert graph["values"][3] == [0.0031] * 3
    assert graph["version_changes"] == []


def test_int_results(built: Path) -> None:
    graph = _load(built / "data" / "graphs" / "series_methods.Peakmem.peakmem_x.json")
    # Integer results are plotted like any other, rounded for display.
    assert graph["values"] == [[123460000.0] * 3]


def test_env_changes_lists_key_packages_first(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    add_commit(input_dir, "aaa", START)
    newer = (
        "name: test\ndependencies:\n"
        "  - aaa-lib=1.0=h_0\n"
        "  - numpy=2.4.0=py311h_0\n"
        "  - python=3.11.9=hb806964_0\n"
        "  - pip:\n"
        "      - pandas==3.1.0.dev0+277.gbbb\n"
    )
    add_commit(input_dir, "bbb", START + DAY_MS, env=newer)
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)
    changes = site.env_changes(store, series.commits(store))
    # pandas is rebuilt at every commit, so its version change is ignored.
    assert changes == [
        {
            "i": 1,
            "changes": [["numpy", "2.3.1", "2.4.0"], ["aaa-lib", None, "1.0"]],
            "n_key": 1,
        }
    ]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([[1.0, 2.0, 3.0]], None),  # latest commit has a result
        ([[1.0, None, None], [None, None, None]], 11),  # failing for 2 commits
        ([[None, None, None]], 10),  # never had a result
        ([[None, 2.0, None], [1.0, None, None]], 12),
    ],
)
def test_failing_since(values: list[list], expected: int | None) -> None:
    assert site._failing_since([10, 11, 12], values) == expected


def test_detections_in_site_data(tmp_path: Path) -> None:
    # 40 commits at one level, then 40 at 1.5x: one regression at commit 40.
    input_dir = tmp_path / "input"
    for i in range(80):
        level = 1.0 if i < 40 else 1.5
        results = {
            "a.time_x": [
                [level * (1 + 0.001 * (i % 3)), 2.0],
                [["1", "2"]],
                "v1",
                1772940858665,
                1.0,
            ]
        }
        sha = f"{i:040x}"
        d = make_result(sha, START + i * DAY_MS, results)
        write_zst(
            input_dir / "data" / "results" / "asvrunner" / f"{sha}.json.zst",
            json.dumps(d).encode(),
        )
    write_benchmark_defs(
        input_dir,
        {"a.time_x": {"version": "v1", "param_names": ["n"], "unit": "seconds"}},
    )
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)
    out = tmp_path / "site"
    stats = site.build(store, out, breaks=[])
    assert stats["detections"] == 1

    index = _load(out / "data" / "index.json")
    assert index["detections"]["i"] == [40]
    assert index["detections"]["name"] == ["a.time_x"]
    assert index["detections"]["params"] == [["1"]]
    assert index["detections"]["kind"] == [1]
    assert index["detections"]["step"][0] == pytest.approx(0.5, abs=0.01)

    graph = _load(out / "data" / "graphs" / "a.time_x.json")
    assert graph["series"] == [["1"], ["2"]]
    [[i, sid, kind, step]] = graph["detections"]
    assert (i, sid, kind) == (40, 0, 1)
