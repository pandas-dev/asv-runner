"""Convert asv result JSONs to table rows and back without loss.

An asv result file is split into three tables:

- runs: one row per file. Everything except ``results``, as a JSON string, plus
  the commit date.
- benchmarks: one row per (commit, benchmark). The parameter grid, the
  benchmark code ``version``, ``started_at``, ``duration``, and raw JSON for any
  value that does not fit a typed column.
- values: one row per (commit, benchmark, parameter combination). The result
  and the stats columns. asv writes the stats columns rounded to 5 significant
  digits; they are stored as an integer mantissa and a power-of-ten exponent
  (``<col>_m``, ``<col>_e``), which compresses about 40% better than float64
  and rebuilds the same float.

``rebuild`` inverts ``parse`` exactly: ``json.dumps`` of the rebuilt dict equals
``json.dumps`` of the original, including int vs float, NaN vs null, and key
order.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import pyarrow as pa

# Per-combination columns stored in values when they are a list of the right
# length holding only numbers of the expected type and None.
FLOAT_COLS = ["result"]
DECIMAL_COLS = ["stats_ci_99_a", "stats_ci_99_b", "stats_q_25", "stats_q_75"]
INT_COLS = ["stats_number", "stats_repeat"]
# peakmem_/mem_ benchmarks report integer results.
RESULT_INT = "result_int"
# Per-benchmark scalar columns stored natively when they have the expected type.
SCALARS: dict[str, type] = {"version": str, "started_at": int, "duration": float}

RUNS_SCHEMA = pa.schema(
    [
        ("sha", pa.string()),
        ("date", pa.timestamp("ms", tz="UTC")),
        ("meta", pa.string()),
    ]
)
BENCHMARKS_SCHEMA = pa.schema(
    [
        ("sha", pa.string()),
        ("name", pa.string()),
        ("position", pa.int32()),  # index within the file's "results"
        ("n_cols", pa.int8()),  # length of the file's row for this benchmark
        ("params", pa.string()),  # JSON of the parameter grid
        ("version", pa.string()),
        ("started_at", pa.int64()),
        ("duration", pa.float64()),
        ("result_is_int", pa.bool_()),
        ("raw", pa.string()),  # JSON {column: value} for values not stored natively
    ]
)
VALUES_SCHEMA = pa.schema(
    [("sha", pa.string()), ("name", pa.string()), ("combo", pa.int32())]
    + [(c, pa.float64()) for c in FLOAT_COLS]
    + [(RESULT_INT, pa.int64())]
    + [
        field
        for c in DECIMAL_COLS
        for field in ((f"{c}_m", pa.int64()), (f"{c}_e", pa.int16()))
    ]
    + [(c, pa.int64()) for c in INT_COLS]
)


def n_combos(params: list | None) -> int:
    return math.prod(len(p) for p in params) if params else 1


def _fits(values: Any, n: int, kind: type) -> bool:
    if not isinstance(values, list) or len(values) != n:
        return False
    # `type(v) is kind`: an int stored as float would come back as 1.0, and a
    # bool is an int subclass.
    return all(v is None or type(v) is kind for v in values)


def to_decimal(v: float) -> tuple[int, int]:
    """Split a finite float into (mantissa, exponent) with ``repr(v) == m * 10**e``.

    ``float(f"{m}e{e}")`` is ``v`` again: ``repr`` is the shortest decimal string
    that parses to ``v``, and this is the same decimal number.
    """
    mant, _, exp = repr(v).partition("e")
    whole, _, frac = mant.partition(".")
    return int(whole + frac), (int(exp) if exp else 0) - len(frac)


def from_decimal(m: int | None, e: int | None) -> float | None:
    return None if m is None else float(f"{m}e{e}")


def _fits_decimal(values: Any, n: int) -> bool:
    # -0.0 would come back as 0.0; NaN and inf have no decimal form.
    return _fits(values, n, float) and all(
        v is None or (math.isfinite(v) and (v != 0 or math.copysign(1, v) > 0))
        for v in values
    )


def commit_date(d: dict) -> dt.datetime:
    """Commit date of a result file; asv stores it as milliseconds since epoch."""
    return dt.datetime.fromtimestamp(d["date"] / 1000, tz=dt.timezone.utc)


def parse(d: dict) -> tuple[dict, dict[str, list], dict[str, list]]:
    """Split one loaded result file into a runs row and benchmarks/values columns."""
    sha = d["commit_hash"]
    cols = d["result_columns"]
    # Keep "results" as a None placeholder so key order survives the round trip.
    meta = json.dumps({k: (None if k == "results" else v) for k, v in d.items()})
    run = {"sha": sha, "date": commit_date(d), "meta": meta}

    bench: dict[str, list] = {f.name: [] for f in BENCHMARKS_SCHEMA}
    values: dict[str, list] = {f.name: [] for f in VALUES_SCHEMA}
    for position, (name, row_values) in enumerate(d["results"].items()):
        row = dict(zip(cols, row_values))
        params = row.get("params")
        n = n_combos(params)
        native: dict[str, Any] = {}
        raw: dict[str, Any] = {}
        result_is_int = False
        for c, v in row.items():
            if c == "params":
                continue
            if c == "result" and _fits(v, n, int) and not _fits(v, n, float):
                native[RESULT_INT] = v
                result_is_int = True
            elif c in FLOAT_COLS and _fits(v, n, float):
                native[c] = v
            elif c in DECIMAL_COLS and _fits_decimal(v, n):
                pairs = [(None, None) if x is None else to_decimal(x) for x in v]
                native[f"{c}_m"] = [m for m, _ in pairs]
                native[f"{c}_e"] = [e for _, e in pairs]
            elif (
                c in INT_COLS
                and _fits(v, n, int)
                or c in SCALARS
                and type(v) is SCALARS[c]
            ):
                native[c] = v
            else:
                raw[c] = v
        bench["sha"].append(sha)
        bench["name"].append(name)
        bench["position"].append(position)
        bench["n_cols"].append(len(row_values))
        bench["params"].append(json.dumps(params) if "params" in row else None)
        for c in SCALARS:
            bench[c].append(native.get(c))
        bench["result_is_int"].append(result_is_int)
        bench["raw"].append(json.dumps(raw) if raw else None)
        if native.keys() - SCALARS.keys():
            values["sha"].extend([sha] * n)
            values["name"].extend([name] * n)
            values["combo"].extend(range(n))
            for c in VALUES_SCHEMA.names[3:]:
                values[c].extend(native.get(c, [None] * n))
    return run, bench, values


def rebuild(meta: str, bench: list[dict], values: dict[str, dict[str, list]]) -> dict:
    """Rebuild one result file.

    Parameters
    ----------
    meta
        The ``meta`` column of the file's runs row.
    bench
        The file's benchmarks rows, as dicts, in any order.
    values
        Maps benchmark name to that benchmark's values columns as lists,
        ordered by ``combo``.

    Returns
    -------
    dict
        Equal to ``json.load`` of the original file.
    """
    d = json.loads(meta)
    cols = d["result_columns"]
    results = {}
    for b in sorted(bench, key=lambda b: b["position"]):
        raw = json.loads(b["raw"]) if b["raw"] else {}
        grid = json.loads(b["params"]) if b["params"] is not None else None
        v = values.get(b["name"], {})
        out = []
        for c in cols[: b["n_cols"]]:
            if c in raw:
                out.append(raw[c])
            elif c == "params":
                out.append(grid)
            elif c in SCALARS:
                out.append(b[c])
            elif c == "result" and b["result_is_int"]:
                out.append(v[RESULT_INT])
            elif c in DECIMAL_COLS:
                out.append(list(map(from_decimal, v[f"{c}_m"], v[f"{c}_e"])))
            else:
                out.append(v[c])
        results[b["name"]] = out
    d["results"] = results  # fills the placeholder, keeping key order
    return d
