from __future__ import annotations

import argparse
import datetime as dt
import itertools as it
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa

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
    data = (
        data[data["result"].notnull()]
        .set_index(["name", "params", "date"])
        .sort_index()
    )
    keys = ["name", "params"]
    tol = 0.95

    data["established_worst"] = (
        data.groupby(keys, as_index=False)["result"]
        .rolling(window_size, center=True)
        .max()[["result"]]
    )
    data["established_best"] = (
        data.groupby(keys, as_index=False)["result"]
        .rolling(window_size, center=True)
        .min()[["result"]]
    )

    mask = (
        # TODO: is the arg to shift right?
        data["established_worst"].groupby(keys).shift(window_size)
        < tol * data["established_best"]
    )
    mask = mask & ~mask.groupby(keys).shift(1, fill_value=False)
    mask = mask.groupby(keys).shift(-(window_size - 1) // 2, fill_value=False)

    data["is_regression"] = mask
    data["pct_change"] = data.groupby(keys)["result"].pct_change()
    data["abs_change"] = data["result"] - data.groupby(keys)["result"].shift(1)
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
    input_path = Path(input_path)
    output_path = Path(output_path)
    parquet_path = output_path / PARQUET_DIRNAME

    existing = load_existing(parquet_path)
    skip_shas: set[str] = (
        set(existing["sha"].dropna().unique()) if existing is not None else set()
    )

    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    new_rows = build_new_rows(input_path, skip_shas, today)

    if existing is not None:
        existing_base = existing.drop(
            columns=[c for c in DERIVED_COLUMNS if c in existing.columns]
        )
        df = pd.concat(
            [existing_base[BASE_COLUMNS], new_rows[BASE_COLUMNS]], ignore_index=True
        )
    else:
        df = new_rows[BASE_COLUMNS]

    result = detect_regression(df, window_size=21)

    result.to_parquet(
        parquet_path,
        index=False,
        partition_cols=["added_date"],
        existing_data_behavior="delete_matching",
        basename_template="part-{i}.parquet",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    args = parser.parse_args()
    run(input_path=args.input_path, output_path=args.output_path)
