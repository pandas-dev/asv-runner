from __future__ import annotations

import argparse
import datetime as dt
import faulthandler
import itertools as it
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa

# Print a Python stack trace on SIGABRT / SIGSEGV / etc. so a crash inside
# pyarrow / pandas C extensions still surfaces a useful traceback.
faulthandler.enable()


def _step(msg: str) -> None:
    print(f"[process_results] {msg}", flush=True)


PARQUET_DIRNAME = "results.parquet"
BASE_COLUMNS = ["date", "sha", "name", "params", "result", "added_date"]
DERIVED_COLUMNS = [
    "established_worst",
    "established_best",
    "is_regression",
    "pct_change",
    "abs_change",
]


def detect_regression(data: pd.DataFrame, window_size: int = 21) -> pd.DataFrame:
    _step(f"detect_regression: input rows={len(data)}")
    data = (
        data[data["result"].notnull()]
        .set_index(["name", "params", "date"])
        .sort_index()
    )
    _step(f"detect_regression: after notnull+sort rows={len(data)}")
    keys = ["name", "params"]
    tol = 0.95

    _step("detect_regression: rolling max (established_worst)")
    data["established_worst"] = (
        data.groupby(keys, as_index=False)["result"]
        .rolling(window_size, center=True)
        .max()[["result"]]
    )
    _step("detect_regression: rolling min (established_best)")
    data["established_best"] = (
        data.groupby(keys, as_index=False)["result"]
        .rolling(window_size, center=True)
        .min()[["result"]]
    )

    _step("detect_regression: building mask")
    mask = (
        # TODO: is the arg to shift right?
        data["established_worst"].groupby(keys).shift(window_size)
        < tol * data["established_best"]
    )
    mask = mask & ~mask.groupby(keys).shift(1, fill_value=False)
    mask = mask.groupby(keys).shift(-(window_size - 1) // 2, fill_value=False)

    data["is_regression"] = mask
    _step("detect_regression: pct_change/abs_change")
    data["pct_change"] = data.groupby(keys)["result"].pct_change()
    data["abs_change"] = data["result"] - data.groupby(keys)["result"].shift(1)
    _step("detect_regression: done, resetting index")
    return data.reset_index()


def load_existing(parquet_path: Path) -> pd.DataFrame | None:
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    # Partition columns round-trip as categorical strings; normalize to a plain
    # string dtype so concat with freshly-built rows doesn't trip over dtypes.
    df["added_date"] = df["added_date"].astype("string[pyarrow]")
    return df


def build_new_rows(
    input_path: Path, skip_shas: set[str], added_date: str
) -> pd.DataFrame:
    with open(input_path / "results" / "benchmarks.json") as fh:
        benchmarks = json.load(fh)
    benchmark_to_param_names = {
        k: v["param_names"] for k, v in benchmarks.items() if k != "version"
    }

    result_path = input_path / "results" / "asvrunner"
    buf: dict[str, list] = {
        "date": [],
        "sha": [],
        "name": [],
        "params": [],
        "result": [],
    }
    for result_json in result_path.glob("*.json"):
        if result_json.name == "machine.json":
            continue
        with open(result_json) as fh:
            results = json.load(fh)
        commit_hash = results["commit_hash"]
        if commit_hash in skip_shas:
            continue
        columns = results["result_columns"]
        timestamp = dt.datetime.fromtimestamp(results["date"] / 1000)
        for name, benchmark in results["results"].items():
            data = dict(zip(columns, benchmark))
            result = data["result"]
            param_names = benchmark_to_param_names[name]
            params = [
                ", ".join(f"{k}={v}" for k, v in zip(param_names, e))
                for e in it.product(*data["params"])
            ]
            buf["name"].extend([name] * len(result))
            buf["params"].extend(params)
            buf["result"].extend(result)
            buf["date"].extend([timestamp] * len(result))
            buf["sha"].extend([commit_hash] * len(result))

    df = pd.DataFrame(
        {
            "name": pd.array(buf["name"], dtype="string[pyarrow]"),
            "params": pd.array(buf["params"], dtype="string[pyarrow]"),
            "result": pd.array(buf["result"], dtype="float64[pyarrow]"),
            "date": pd.array(buf["date"], dtype=pd.ArrowDtype(pa.timestamp("us"))),
            "sha": pd.array(buf["sha"], dtype="string[pyarrow]"),
        }
    )
    df["added_date"] = pd.array([added_date] * len(df), dtype="string[pyarrow]")
    return df


def run(input_path: str | Path, output_path: str | Path):
    _step(
        f"run: pandas={pd.__version__} pyarrow={pa.__version__} python={sys.version.split()[0]}"
    )
    input_path = Path(input_path)
    output_path = Path(output_path)
    parquet_path = output_path / PARQUET_DIRNAME

    _step(f"run: load_existing({parquet_path})")
    existing = load_existing(parquet_path)
    _step(f"run: existing rows={0 if existing is None else len(existing)}")
    skip_shas: set[str] = (
        set(existing["sha"].dropna().unique()) if existing is not None else set()
    )
    _step(f"run: skip_shas count={len(skip_shas)}")

    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    _step(f"run: build_new_rows (added_date={today})")
    new_rows = build_new_rows(input_path, skip_shas, today)
    _step(f"run: new rows={len(new_rows)}")

    if existing is not None:
        _step("run: concat existing + new")
        existing_base = existing.drop(
            columns=[c for c in DERIVED_COLUMNS if c in existing.columns]
        )
        df = pd.concat(
            [existing_base[BASE_COLUMNS], new_rows[BASE_COLUMNS]], ignore_index=True
        )
    else:
        df = new_rows[BASE_COLUMNS]
    _step(f"run: total rows={len(df)}")

    _step("run: detect_regression")
    result = detect_regression(df, window_size=21)
    _step(f"run: detected, regressions={int(result['is_regression'].sum())}")

    _step(f"run: to_parquet({parquet_path})")
    result.to_parquet(
        parquet_path,
        index=False,
        partition_cols=["added_date"],
        existing_data_behavior="delete_matching",
        basename_template="part-{i}.parquet",
    )
    _step("run: done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    args = parser.parse_args()
    run(input_path=args.input_path, output_path=args.output_path)
