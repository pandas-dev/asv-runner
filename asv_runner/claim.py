"""Claim the next pandas SHA, append to shas.txt, push storage branch."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from asv_runner.util import orphan_push_with_retry, write_github_output

LOOKBACK_COMMITS = 40


def read_existing_shas(shas_path: Path) -> set[str]:
    if not shas_path.exists():
        return set()
    return {line.strip() for line in shas_path.read_text().splitlines()}


def pick_next_sha(repo: Path, existing_shas: set[str]) -> str | None:
    response = subprocess.run(
        ["git", "log", f"-{LOOKBACK_COMMITS}", "--oneline", "--no-abbrev-commit"],
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


def run(args: argparse.Namespace) -> None:
    storage = Path(args.storage_dir)
    repo = Path(args.repo_dir)
    shas_path = storage / "data" / "shas.txt"

    last_picked: list[str | None] = [None]

    # Append the next unclaimed SHA to shas.txt; the mutable cell smuggles
    # the picked SHA back out so the caller can report it once the push lands.
    def modify_tree(_: Path) -> bool:
        existing = read_existing_shas(shas_path)
        sha = pick_next_sha(repo, existing_shas=existing)
        if sha is None:
            last_picked[0] = None
            return False
        with shas_path.open("a") as f:
            f.write(f"{sha}\n")
        last_picked[0] = sha
        return True

    pushed = orphan_push_with_retry(
        storage,
        branch=args.branch,
        message="Update shas.txt",
        modify_tree=modify_tree,
    )
    if pushed:
        assert last_picked[0] is not None
        write_github_output(sha=last_picked[0], new_commit="yes")
    else:
        write_github_output(sha="NONE", new_commit="no")
