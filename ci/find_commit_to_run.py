from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(*, input_path: str | Path, repo_path: str | Path):
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)

    shas_path = input_path / "shas.txt"
    existing_shas: set[str]
    if not shas_path.exists():
        existing_shas = set()
    else:
        with open(input_path / "shas.txt") as fh:
            existing_shas = {line.strip() for line in fh.readlines()}

    response = subprocess.run(
        f"cd {repo_path} && git log -40 --oneline --no-abbrev-commit",
        capture_output=True,
        shell=True,
        check=False,
    )
    recent_shas = [
        line[: line.find(" ")] for line in response.stdout.decode().strip().split("\n")
    ]
    for sha in recent_shas:
        if sha not in existing_shas:
            print(sha)
            return
    print("NONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path")
    parser.add_argument("--repo-path")
    args = parser.parse_args()
    run(input_path=args.input_path, repo_path=args.repo_path)
