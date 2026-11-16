from __future__ import annotations

import json

import pyarrow as pa
import pytest

from asv_runner_v2 import results
from tests.v2._helpers import make_result


def _round_trip(d: dict) -> dict:
    run, bench, values = results.parse(d)
    bench_rows = pa.table(bench, schema=results.BENCHMARKS_SCHEMA).to_pylist()
    values_table = pa.table(values, schema=results.VALUES_SCHEMA)
    by_name: dict[str, dict[str, list]] = {}
    for row in values_table.to_pylist():
        cols = by_name.setdefault(row["name"], {})
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    return results.rebuild(run["meta"], bench_rows, by_name)


def test_round_trip_is_exact() -> None:
    d = make_result("abc", 1772932225000)
    assert json.dumps(_round_trip(d)) == json.dumps(d)


def test_round_trip_keeps_benchmark_order() -> None:
    d = make_result("abc", 1772932225000)
    d["results"] = dict(reversed(list(d["results"].items())))
    assert list(_round_trip(d)["results"]) == list(d["results"])


def test_parse_types() -> None:
    d = make_result("abc", 1772932225000)
    run, bench, values = results.parse(d)
    assert run["date"].isoformat() == "2026-03-08T01:10:25+00:00"

    names = bench["name"]
    peakmem = names.index("series_methods.Peakmem.peakmem_x")
    assert bench["result_is_int"][peakmem]
    # The int duration does not fit the float column.
    assert json.loads(bench["raw"][peakmem]) == {"duration": 3}

    odd = names.index("frame_methods.Odd.time_odd")
    assert set(json.loads(bench["raw"][odd])) == {
        "stats_ci_99_a",
        "stats_ci_99_b",
        "samples",
    }

    # One values row per parameter combination: 4 + 1 + 1.
    assert len(values["sha"]) == 6
    assert values["combo"][:4] == [0, 1, 2, 3]
    assert values["stats_q_25_m"][0] == 14907
    assert values["stats_q_25_e"][0] == -10


@pytest.mark.parametrize(
    "v",
    [0.0, 1.0, -1.5, 100.0, 1.8873e-06, 0.0030268, 1e16, 1.2345e300, 5e-324, 0.1 + 0.2],
)
def test_decimal_round_trip(v: float) -> None:
    m, e = results.to_decimal(v)
    out = results.from_decimal(m, e)
    assert repr(out) == repr(v)
