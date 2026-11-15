"""Decompress, asv publish, build parquet, raise issues, push."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools as it
import json
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path

from asv_runner.util import execute, orphan_push_with_retry, time_to_str

PARQUET_DIRNAME = "results.parquet"
WINDOW_SIZE = 21
SERIES_KEYS = ["name", "params"]
ROW_KEYS = ["sha", "name", "params"]
BASE_COLUMNS = ["date", "sha", "name", "params", "result", "added_date"]
DERIVED_COLUMNS = [
    "established_worst",
    "established_best",
    "is_regression",
    "pct_change",
    "abs_change",
]
GITHUB_ISSUE_LENGTH = 65000
COMPARE_URL_BASE = "https://github.com/pandas-dev/pandas/compare/"


def detect_regression(data, window_size: int = 21):
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
        data["established_worst"].groupby(keys).shift(window_size)
        < tol * data["established_best"]
    )
    mask = mask & ~mask.groupby(keys).shift(1, fill_value=False)
    mask = mask.groupby(keys).shift(-(window_size - 1) // 2, fill_value=False)

    data["is_regression"] = mask
    data["pct_change"] = data.groupby(keys)["result"].pct_change()
    data["abs_change"] = data["result"] - data.groupby(keys)["result"].shift(1)
    return data.reset_index()


def build_new_rows(input_path: Path, skip_shas: set[str], added_date: str):
    import pandas as pd
    import pyarrow as pa

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
            if name not in benchmark_to_param_names:
                # benchmarks.json reflects only the latest benchmarked commit;
                # older or concurrently-pushed result files can reference names
                # that have since been renamed or removed in pandas.
                print(
                    f"Skipping {name!r} for {commit_hash}: "
                    "not in current benchmarks.json."
                )
                continue
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


def regression_reach(window_size: int) -> tuple[int, int]:
    """Return (lookback, lookahead) of detect_regression in rows.

    The derived columns of row i in a (name, params) series depend only on
    rows i - lookback through i + lookahead of that series.
    """
    half = (window_size - 1) // 2
    return window_size + 1, 2 * half


def read_sha_dates(parquet_path: Path):
    """Return the unique (sha, date) pairs in the dataset, sorted by date."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_path, format="parquet", partitioning="hive")
    tables = [
        fragment.to_table(columns=["sha", "date"])
        .group_by(["sha", "date"])
        .aggregate([])
        for fragment in dataset.get_fragments()
    ]
    if not tables:
        tables = [dataset.head(0, columns=["sha", "date"])]
    table = pa.concat_tables(tables).group_by(["sha", "date"]).aggregate([])
    return table.to_pandas().sort_values("date").reset_index(drop=True)


def read_rows(parquet_path: Path, filters, columns=None):
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=columns, filters=filters)
    if "added_date" in df.columns:
        df["added_date"] = df["added_date"].astype("string[pyarrow]")
    return df


def write_rows(df, parquet_path: Path) -> None:
    df.to_parquet(
        parquet_path,
        index=False,
        partition_cols=["added_date"],
        existing_data_behavior="delete_matching",
        basename_template="part-{i}.parquet",
    )


def update_parquet(
    parquet_path: Path,
    new_rows,
    sha_dates,
    window_size: int = WINDOW_SIZE,
    slack: int = WINDOW_SIZE,
) -> None:
    """Add new_rows to the dataset, recomputing only derived values that change.

    A row inserted at position p of a series changes the derived values of
    rows p - lookahead onward, and recomputing those needs the rows from
    p - lookahead - lookback onward. Only the added_date partitions holding
    changed rows are rewritten.
    """
    import pandas as pd

    new_rows = new_rows[new_rows["result"].notna()]
    if new_rows.empty:
        return
    lookback, lookahead = regression_reach(window_size)
    context = lookback + lookahead
    first_new_date = new_rows["date"].min()

    n_before = int((sha_dates["date"] < first_new_date).sum())
    start = n_before - context - slack
    if start <= 0:
        existing = read_rows(parquet_path, filters=None, columns=BASE_COLUMNS)
    else:
        cutoff = sha_dates["date"].iloc[start]
        existing = read_rows(
            parquet_path, filters=[("date", ">=", cutoff)], columns=BASE_COLUMNS
        )
        # Series with missing results have fewer rows than shas in the slice;
        # read the full history of any series left without enough context.
        candidates = pd.concat(
            [existing.loc[existing["date"] >= first_new_date, SERIES_KEYS], new_rows]
        )[SERIES_KEYS].drop_duplicates()
        n_context = (
            existing[existing["date"] < first_new_date]
            .groupby(SERIES_KEYS)
            .size()
            .rename("n")
            .reset_index()
        )
        candidates = candidates.merge(n_context, on=SERIES_KEYS, how="left")
        short = candidates[candidates["n"].fillna(0) < context]
        if not short.empty:
            older = read_rows(
                parquet_path,
                filters=[
                    ("date", "<", cutoff),
                    ("name", "in", short["name"].unique().tolist()),
                ],
                columns=BASE_COLUMNS,
            )
            existing = pd.concat([older, existing], ignore_index=True)

    combined = pd.concat(
        [existing[BASE_COLUMNS], new_rows[BASE_COLUMNS]], ignore_index=True
    )
    result = detect_regression(combined, window_size=window_size)

    groups = [result["name"], result["params"]]
    rank = result.groupby(groups).cumcount()
    first_changed = (
        rank.where(result["date"] >= first_new_date).groupby(groups).transform("min")
        - lookahead
    )
    changed = result[rank >= first_changed]

    partitions = changed["added_date"].unique().tolist()
    old = read_rows(parquet_path, filters=[("added_date", "in", partitions)])
    old = old.merge(changed[ROW_KEYS], on=ROW_KEYS, how="left", indicator=True)
    unchanged = old[old["_merge"] == "left_only"][result.columns]
    write_rows(pd.concat([unchanged, changed], ignore_index=True), parquet_path)


def build_parquet(input_path: Path, output_path: Path) -> None:
    parquet_path = output_path / PARQUET_DIRNAME
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()

    if not parquet_path.exists():
        new_rows = build_new_rows(input_path, skip_shas=set(), added_date=today)
        write_rows(detect_regression(new_rows[BASE_COLUMNS], WINDOW_SIZE), parquet_path)
        return

    sha_dates = read_sha_dates(parquet_path)
    new_rows = build_new_rows(
        input_path, skip_shas=set(sha_dates["sha"]), added_date=today
    )
    update_parquet(parquet_path, new_rows, sha_dates=sha_dates)


def get_commit_range(*, benchmarks, sha: str) -> str:
    shas = benchmarks.sort_values("date")["sha"].unique().tolist()
    idx = shas.index(sha)
    prev_sha = shas[idx - 1]
    return f"{prev_sha}...{sha}"


def fetch_pr_info(*, commit_range: str, sha: str) -> dict | None:
    """Return PR metadata when commit_range contains exactly one commit.

    Returns dict with keys {number, author, approvers}, or None when the range
    spans more than one commit or no PR is associated with sha.
    """
    cmd = f'gh api "repos/pandas-dev/pandas/compare/{commit_range}"'
    compare = json.loads(execute(cmd))
    if compare.get("ahead_by") != 1:
        return None

    cmd = f'gh api "repos/pandas-dev/pandas/commits/{sha}/pulls"'
    prs = json.loads(execute(cmd))
    if not prs:
        return None
    pr = prs[0]
    number = pr["number"]
    author = (pr.get("user") or {}).get("login")
    if not author:
        return None

    cmd = f'gh api "repos/pandas-dev/pandas/pulls/{number}/reviews"'
    reviews = json.loads(execute(cmd))
    approvers: set[str] = set()
    for review in reviews:
        if review.get("state") != "APPROVED":
            continue
        login = (review.get("user") or {}).get("login")
        if login and login != author:
            approvers.add(login)
    return {"number": number, "author": author, "approvers": sorted(approvers)}


def make_body(
    commit_range: str,
    benchmarks,
    sha: str,
    pr_info: dict | None = None,
    shorten: bool = False,
) -> str:
    if pr_info is not None:
        pr_url = f"https://github.com/pandas-dev/pandas/pull/{pr_info['number']}"
        body = f"[PR #{pr_info['number']}]({pr_url})\n\n"
        mentions = [f"@{pr_info['author']}", *(f"@{a}" for a in pr_info["approvers"])]
        body += f"cc {' '.join(mentions)}\n\n"
    else:
        body = f"[Commit Range]({COMPARE_URL_BASE + commit_range})\n\n"
        body += (
            "Subsequent benchmarks may have skipped some commits. The link"
            " above lists the commits that are"
            " between the two benchmark runs where the regression was identified."
            "\n\n"
        )

    regressions = benchmarks[benchmarks["sha"].eq(sha) & benchmarks["is_regression"]]
    prev_benchmark = ""
    for _, regression in regressions.iterrows():
        benchmark = regression["name"]
        params = regression["params"]
        site_base = "https://pandas-dev.github.io/asv-runner/#"
        url = f"{site_base}{benchmark}"
        abs_change = time_to_str(regression["abs_change"])
        severity = f"{regression['pct_change']:0.3%} ({abs_change})"
        if prev_benchmark != benchmark:
            body += f" - [ ] [{benchmark}]({url})"
        prev_benchmark = benchmark
        if params == "" or shorten:
            body += f" - {severity}\n"
            continue
        body += "\n"
        params_list = list(params.split(", "))
        params_suffix = "?p-" + "&p-".join(params_list)
        url = f"{site_base}{benchmark}{params_suffix}"
        url = urllib.parse.quote(url, safe="/:?=&#")
        body += f"   - [ ] [{params}]({url}) - {severity}\n"
    body += "\n"
    return body


def make_envs_diff(*, envs_dir: Path, benchmarks, sha: str) -> str:
    prev_sha = benchmarks["sha"][
        benchmarks["sha"].eq(sha).shift(-1, fill_value=False)
    ].iloc[0]
    curr_env = envs_dir / f"{sha}.yml"
    prev_env = envs_dir / f"{prev_sha}.yml"
    result = subprocess.run(
        ["diff", str(prev_env), str(curr_env)],
        capture_output=True,
        check=False,
    ).stdout.decode()
    return f"Environment changes from previous commit:\n```\n{result}\n```"


def raise_issues(input_path: Path, envs_dir: Path) -> None:
    parquet_path = input_path / PARQUET_DIRNAME
    benchmarks = read_rows(parquet_path, filters=[("is_regression", "==", True)])
    sha_dates = read_sha_dates(parquet_path)
    regression_shas = (
        benchmarks[benchmarks["is_regression"]]
        .drop_duplicates(subset="sha")
        .sort_values("date")["sha"]
        .unique()
        .tolist()[-40:]
    )
    print("Number of regressions to raise issues for:", len(regression_shas))
    for sha in regression_shas:
        time.sleep(2)
        needle = f"Commit {sha}"
        cmd = f'gh search issues --repo pandas-dev/asv-runner "{needle}"'
        result = execute(cmd)
        if result != "":
            continue

        title = f"Commit {sha}"
        commit_range = get_commit_range(benchmarks=sha_dates, sha=sha)
        pr_info = fetch_pr_info(commit_range=commit_range, sha=sha)

        body = make_body(
            commit_range=commit_range,
            benchmarks=benchmarks,
            sha=sha,
            pr_info=pr_info,
        )
        if len(body) >= GITHUB_ISSUE_LENGTH:
            body = make_body(
                commit_range=commit_range,
                benchmarks=benchmarks,
                sha=sha,
                pr_info=pr_info,
                shorten=True,
            )
        if len(body) >= GITHUB_ISSUE_LENGTH:
            body = body[:GITHUB_ISSUE_LENGTH]
            body += "\nWARNING: Body has been clipped due to length."

        cmd = (
            "gh issue create"
            " --repo pandas-dev/asv-runner"
            f' --title "{title}"'
            " --body-file -"
        )
        issue_url = execute(cmd, input=body)

        issue_number = issue_url[issue_url.rfind("/") + 1 :].strip()
        envs_diff = make_envs_diff(envs_dir=envs_dir, benchmarks=sha_dates, sha=sha)
        if len(envs_diff) >= GITHUB_ISSUE_LENGTH:
            envs_diff = envs_diff[:GITHUB_ISSUE_LENGTH]
            envs_diff += "\n```\n\nWARNING: Body has been clipped due to length."
        cmd = (
            f"gh issue comment {issue_number}"
            " --repo pandas-dev/asv-runner"
            " --body-file -"
        )
        execute(cmd, input=envs_diff)


def stage_asv_inputs(storage: Path, asv: Path) -> Path:
    """Rehydrate the asv tree from compressed per-sha files in storage.

    Decompresses env yml files into an ephemeral directory (returned) rather
    than back into storage, so they don't end up in the orphan-pushed commit.
    """
    asvrunner = asv / "results" / "asvrunner"
    asvrunner.mkdir(parents=True, exist_ok=True)
    envs_dir = Path(tempfile.mkdtemp(prefix="asv-runner-envs-"))

    shutil.copy(
        storage / "data" / "results" / "benchmarks.json",
        asv / "results" / "benchmarks.json",
    )
    shutil.copy(
        storage / "data" / "results" / "asvrunner" / "machine.json",
        asvrunner / "machine.json",
    )
    json_zsts = list((storage / "data" / "results" / "asvrunner").glob("*.json.zst"))
    if json_zsts:
        subprocess.run(
            [
                "zstd",
                "-d",
                "-q",
                "--output-dir-flat",
                str(asvrunner),
                *(str(p) for p in json_zsts),
            ],
            check=True,
        )
    yml_zsts = list((storage / "data" / "envs").glob("*.yml.zst"))
    if yml_zsts:
        subprocess.run(
            [
                "zstd",
                "-d",
                "-q",
                "--output-dir-flat",
                str(envs_dir),
                *(str(p) for p in yml_zsts),
            ],
            check=True,
        )
    return envs_dir


def run(args: argparse.Namespace) -> None:
    storage = Path(args.storage_dir)
    asv = Path(args.asv_dir)

    envs_dir = stage_asv_inputs(storage, asv=asv)

    subprocess.run(["asv", "publish"], cwd=asv, check=True)

    build_parquet(input_path=asv, output_path=storage / "data")
    raise_issues(input_path=storage / "data", envs_dir=envs_dir)

    save = Path(tempfile.mkdtemp())
    shutil.copytree(storage / "data" / "results.parquet", save / "results.parquet")
    shutil.copytree(asv / "html", save / "docs")

    def modify_tree(repo: Path) -> bool:
        parquet_dir = repo / "data" / "results.parquet"
        docs_dir = repo / "docs"
        if parquet_dir.exists():
            shutil.rmtree(parquet_dir)
        if docs_dir.exists():
            shutil.rmtree(docs_dir)
        shutil.copytree(save / "results.parquet", parquet_dir)
        shutil.copytree(save / "docs", docs_dir)
        return True

    orphan_push_with_retry(
        storage,
        branch=args.branch,
        message="Results",
        modify_tree=modify_tree,
    )
