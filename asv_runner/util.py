"""Lower-level utilities shared across asv_runner step modules."""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path


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
    magnitude = abs(x)
    if magnitude >= 1.0:
        result = f"{magnitude:0.3f}s"
    elif magnitude >= 0.001:
        result = f"{magnitude * 1000:0.3f}ms"
    elif magnitude >= 0.000001:
        result = f"{magnitude * (1000**2):0.3f}us"
    else:
        result = f"{magnitude * (1000**3):0.3f}ns"
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
