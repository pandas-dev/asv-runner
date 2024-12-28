import urllib
import pandas as pd
from pathlib import Path
import re
import subprocess
import argparse

def get_commit_range(*, benchmarks: pd.DataFrame, sha: str) -> str:
    """Get commit range between a hash and the previous hash that has a benchmark.

    Args:
        benchmarks: Benchmark data.
        sha: SHA of a commit. This must be from a commit that had benchmarks run.

    Returns:
        The commit range in the form of "{prev_git_hash}...{git_hash}" from the
        previous commit that has a benchmark to the provided git_hash.
    """
    # We're interested in the hashes, so just grab a single benchmark to get
    # the time series.
    shas = benchmarks.sort_values("date")["sha"].unique().tolist()
    idx = shas.index(sha)
    prev_sha = shas[idx - 1]
    result = f"{prev_sha}...{sha}"
    return result

def execute(cmd):
    response = subprocess.run(cmd, shell=True, capture_output=True)
    if response.returncode != 0:
        raise ValueError(f"{response.stdout.decode()}\n\n{response.stderr.decode()}")
    return response.stdout.decode()

# TODO: Try without this
def escape_ansi(line):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)

def time_to_str(x: float) -> str:
    is_negative = x < 0.0
    if x >= 1.0:
        result = f"{x:0.3f}s"
    elif x >= 0.001:
        result = f"{x * 1000:0.3f}ms"
    elif x >= 0.000001:
        result = f"{x * (1000 ** 2):0.3f}us"
    else:
        result = f"{x * (1000 ** 3):0.3f}ns"
    if is_negative:
        result = "-" + result
    return result

def run(input_path: str | Path):
    if not isinstance(input_path, Path):
        input_path = Path(input_path)
    benchmarks = pd.read_parquet(input_path / "results.parquet")
    regression_shas = (
        benchmarks[benchmarks["is_regression"]]
        .drop_duplicates(subset="sha")
        .sort_values("date")
        ["sha"]
        .unique()
        .tolist()
        [-20:]
    )
    for sha in regression_shas:
        needle = f"Commit {sha}"
        cmd = f'gh search issues --repo rhshadrach/asv-runner "{needle}"'
        result = execute(cmd)
        if result != "":
            continue

        commit_range = get_commit_range(benchmarks=benchmarks, sha=sha)
        # TODO: Use checkout path
        response = execute(
            f"cd /home/richard/dev/pandas"
            f" && git rev-list --ancestry-path {commit_range}"
        )

        title = f"Commit {sha}"
        base_url = "https://github.com/pandas-dev/pandas/compare/"
        body = f"[Commit Range]({base_url + get_commit_range(benchmarks=benchmarks, sha=sha)})"
        body += "\n\n"
        body += (
            "Subsequent benchmarks may have skipped some commits. The link"
            " above lists the commits that are"
            " between the two benchmark runs where the regression was identified."
            "\n\n"
        )

        regressions = benchmarks[benchmarks["sha"].eq(sha) & benchmarks["is_regression"]]
        for idx, regression in regressions.iterrows():
            benchmark = regression["name"]
            params = regression["params"]
            base_url = "https://rhshadrach.github.io/asv-runner/#"
            url = f"{base_url}{benchmark}"
            severity = f"{regression['pct_change']:0.3%} ({time_to_str(regression['abs_change'])})"
            body += f" - [ ] [{benchmark}]({url})"
            if params == "":
                result += f" - {severity}\n"
                continue
            body += "\n"
            params_list = [param for param in params.split(", ")]
            params_suffix = "?p-" + "&p-".join(params_list)
            url = f"{base_url}{benchmark}{params_suffix}"
            url = urllib.parse.quote(url, safe="/:?=&#")
            body += f"   - [ ] [{params}]({url}) - {severity}\n"
        body += "\n"

        cmd = (
            f"gh issue create"
            rf' --repo rhshadrach/asv-runner'
            rf' --title "{title}"'
            rf' --body "{body}"'
        )
        execute(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    args = parser.parse_args()
    run(input_path=args.input_path)
