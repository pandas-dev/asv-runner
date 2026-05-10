"""CI orchestration for asv-runner.

Subcommands invoked from .github/workflows/*.yaml:
  claim   - pick next pandas SHA, append to shas.txt, push storage branch
  push    - stage asv outputs (zstd-compress per-sha files) and push to storage
  process - decompress, asv publish, build parquet, raise issues, push
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

PARQUET_DIRNAME = "results.parquet"
BASE_COLUMNS = ["date", "sha", "name", "params", "result", "added_date"]
DERIVED_COLUMNS = [
    "established_worst",
    "established_best",
    "is_regression",
    "pct_change",
    "abs_change",
]
GITHUB_ISSUE_LENGTH = 65000


# === Generic helpers ===


def execute(cmd: str, *, input: str | None = None) -> str:
    print("Executing command")
    print(f"`{cmd}`")
    response = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        check=False,
        input=input,
        text=True,
    )
    if response.returncode != 0:
        raise ValueError(f"{response.stdout}\n\n{response.stderr}")
    return response.stdout


def git(args: list[str], *, cwd: Path, capture: bool = False) -> str:
    if capture:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    subprocess.run(["git", *args], cwd=cwd, check=True)
    return ""


def write_github_output(**kwargs: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        for k, v in kwargs.items():
            print(f"{k}={v}")
        return
    with open(output_file, "a") as f:
        for k, v in kwargs.items():
            f.write(f"{k}={v}\n")


def time_to_str(x: float) -> str:
    is_negative = x < 0.0
    if x >= 1.0:
        result = f"{x:0.3f}s"
    elif x >= 0.001:
        result = f"{x * 1000:0.3f}ms"
    elif x >= 0.000001:
        result = f"{x * (1000**2):0.3f}us"
    else:
        result = f"{x * (1000**3):0.3f}ns"
    if is_negative:
        result = "-" + result
    return result


def escape_ansi(line: str) -> str:
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


def configure_git_user(repo: Path) -> None:
    git(["config", "user.name", "github-actions[bot]"], cwd=repo)
    git(
        ["config", "user.email", "github-actions[bot]@users.noreply.github.com"],
        cwd=repo,
    )


def orphan_push_with_retry(
    repo: Path,
    branch: str,
    message: str,
    modify_tree: Callable[[Path], bool],
    attempts: int = 5,
) -> bool:
    """Fetch branch, let caller mutate the tree, then orphan + force-push.

    modify_tree is called after each fetch with the repo path; return False
    to abort (no push, no retries). On lease failure refetches and retries
    with backoff. Returns True if pushed, False if modify_tree opted out.
    """
    for attempt in range(1, attempts + 1):
        git(["fetch", "origin", branch], cwd=repo)
        git(["checkout", "-B", branch, f"origin/{branch}"], cwd=repo)
        configure_git_user(repo)

        if not modify_tree(repo):
            return False

        expected = git(
            ["rev-parse", f"origin/{branch}"], cwd=repo, capture=True
        ).strip()
        git(["checkout", "--orphan", "fresh"], cwd=repo)
        git(["add", "-A"], cwd=repo)
        git(["commit", "-m", message], cwd=repo)
        git(["branch", "-M", "fresh", branch], cwd=repo)

        result = subprocess.run(
            [
                "git",
                "push",
                f"--force-with-lease={branch}:{expected}",
                "origin",
                branch,
            ],
            cwd=repo,
            check=False,
        )
        if result.returncode == 0:
            return True

        delay = attempt * 5 + random.randint(0, 9)
        print(
            f"Push race lost on attempt {attempt}; retrying in {delay}s",
            file=sys.stderr,
        )
        time.sleep(delay)

    raise RuntimeError(f"Failed to push to {branch} after {attempts} attempts")


# === claim subcommand ===


def read_existing_shas(shas_path: Path) -> set[str]:
    if not shas_path.exists():
        return set()
    return {line.strip() for line in shas_path.read_text().splitlines()}


def pick_next_sha(repo: Path, existing_shas: set[str]) -> str | None:
    response = subprocess.run(
        ["git", "log", "-40", "--oneline", "--no-abbrev-commit"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    recent_shas = [
        line[: line.find(" ")] for line in response.stdout.decode().strip().split("\n")
    ]
    for sha in recent_shas:
        if sha and sha not in existing_shas:
            return sha
    return None


def cmd_claim(args: argparse.Namespace) -> None:
    storage = Path(args.storage_dir)
    repo = Path(args.repo_dir)
    shas_path = storage / "data" / "shas.txt"

    last_picked: list[str | None] = [None]

    def modify_tree(_: Path) -> bool:
        existing = read_existing_shas(shas_path)
        sha = pick_next_sha(repo, existing)
        if sha is None:
            last_picked[0] = None
            return False
        with shas_path.open("a") as f:
            f.write(f"{sha}\n")
        last_picked[0] = sha
        return True

    pushed = orphan_push_with_retry(
        storage, args.branch, "Update shas.txt", modify_tree
    )
    if pushed:
        assert last_picked[0] is not None
        write_github_output(sha=last_picked[0], new_commit="yes")
    else:
        write_github_output(sha="NONE", new_commit="no")


# === push subcommand ===


def cmd_push(args: argparse.Namespace) -> None:
    import tempfile

    storage = Path(args.storage_dir)
    asv = Path(args.asv_dir)
    sha = args.sha
    short_sha = sha[:8]

    asv_results = asv / "results" / "asvrunner"
    matches = list(asv_results.glob(f"{short_sha}-existing*.json"))
    if not matches:
        raise RuntimeError(f"No matching result file for short sha {short_sha}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matches for short sha {short_sha}: {matches}")

    benchmarks_json = asv / "results" / "benchmarks.json"
    machine_json = asv_results / "machine.json"

    workdir = Path(tempfile.mkdtemp())
    sha_json = workdir / f"{sha}.json"
    sha_yml = workdir / f"{sha}.yml"
    shutil.copy(matches[0], sha_json)
    with sha_yml.open("w") as f:
        subprocess.run(
            ["pixi", "list", "-e", "asv", "--fields", "name,version"],
            stdout=f,
            check=True,
        )
    subprocess.run(["zstd", "--rm", "-19", "-q", str(sha_json)], check=True)
    subprocess.run(["zstd", "--rm", "-19", "-q", str(sha_yml)], check=True)
    sha_json_zst = workdir / f"{sha}.json.zst"
    sha_yml_zst = workdir / f"{sha}.yml.zst"

    def modify_tree(repo: Path) -> bool:
        data = repo / "data"
        (data / "results" / "asvrunner").mkdir(parents=True, exist_ok=True)
        (data / "envs").mkdir(parents=True, exist_ok=True)
        shutil.copy(benchmarks_json, data / "results" / "benchmarks.json")
        shutil.copy(machine_json, data / "results" / "asvrunner" / "machine.json")
        shutil.copy(sha_json_zst, data / "results" / "asvrunner" / f"{sha}.json.zst")
        shutil.copy(sha_yml_zst, data / "envs" / f"{sha}.yml.zst")
        return True

    orphan_push_with_retry(storage, args.branch, "Results", modify_tree)


# === process subcommand ===


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


def load_existing(parquet_path: Path):
    import pandas as pd

    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    df["added_date"] = df["added_date"].astype("string[pyarrow]")
    return df


def build_new_rows(input_path: Path, skip_shas: set[str], added_date: str):
    import datetime as dt
    import itertools as it
    import json

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


def build_parquet(input_path: Path, output_path: Path) -> None:
    import datetime as dt

    import pandas as pd

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
            [existing_base[BASE_COLUMNS], new_rows[BASE_COLUMNS]],
            ignore_index=True,
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


def get_commit_range(*, benchmarks, sha: str) -> str:
    shas = benchmarks.sort_values("date")["sha"].unique().tolist()
    idx = shas.index(sha)
    prev_sha = shas[idx - 1]
    return f"{prev_sha}...{sha}"


def make_body(
    base_url: str,
    commit_range: str,
    benchmarks,
    sha: str,
    shorten: bool = False,
) -> str:
    import urllib.parse

    body = f"[Commit Range]({base_url + commit_range})\n\n"
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


def make_envs_diff(*, input_path: Path, benchmarks, sha: str) -> str:
    prev_sha = benchmarks["sha"][benchmarks["sha"].eq(sha).shift(-1)].iloc[0]
    curr_env = input_path / "envs" / f"{sha}.yml"
    prev_env = input_path / "envs" / f"{prev_sha}.yml"
    result = subprocess.run(
        ["diff", str(prev_env), str(curr_env)],
        capture_output=True,
        check=False,
    ).stdout.decode()
    return f"Environment changes from previous commit:\n```\n{result}\n```"


def raise_issues(input_path: Path) -> None:
    import pandas as pd

    benchmarks = pd.read_parquet(input_path / "results.parquet")
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
        base_url = "https://github.com/pandas-dev/pandas/compare/"
        commit_range = get_commit_range(benchmarks=benchmarks, sha=sha)

        body = make_body(
            base_url=base_url,
            commit_range=commit_range,
            benchmarks=benchmarks,
            sha=sha,
        )
        if len(body) >= GITHUB_ISSUE_LENGTH:
            body = make_body(
                base_url=base_url,
                commit_range=commit_range,
                benchmarks=benchmarks,
                sha=sha,
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
        envs_diff = make_envs_diff(
            input_path=input_path, benchmarks=benchmarks, sha=sha
        )
        if len(envs_diff) >= GITHUB_ISSUE_LENGTH:
            envs_diff = envs_diff[:GITHUB_ISSUE_LENGTH]
            envs_diff += "\n```\n\nWARNING: Body has been clipped due to length."
        cmd = (
            f"gh issue comment {issue_number}"
            " --repo pandas-dev/asv-runner"
            " --body-file -"
        )
        execute(cmd, input=envs_diff)


def cmd_process(args: argparse.Namespace) -> None:
    import tempfile

    storage = Path(args.storage_dir)
    asv = Path(args.asv_dir)
    asvrunner = asv / "results" / "asvrunner"
    asvrunner.mkdir(parents=True, exist_ok=True)

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
                str(storage / "data" / "envs"),
                *(str(p) for p in yml_zsts),
            ],
            check=True,
        )

    subprocess.run(["asv", "publish"], cwd=asv, check=True)

    build_parquet(input_path=asv, output_path=storage / "data")
    raise_issues(input_path=storage / "data")

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

    orphan_push_with_retry(storage, args.branch, "Results", modify_tree)


# === entry point ===


def main() -> None:
    parser = argparse.ArgumentParser(description="CI orchestration for asv-runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("claim")
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--branch", required=True)
    p.set_defaults(func=cmd_claim)

    p = sub.add_parser("push")
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--asv-dir", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--branch", required=True)
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("process")
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--asv-dir", required=True)
    p.add_argument("--branch", required=True)
    p.set_defaults(func=cmd_process)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
