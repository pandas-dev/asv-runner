"""Run asv benchmarks for a target SHA."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(args: argparse.Namespace) -> None:
    asv = Path(args.asv_dir)
    subprocess.run(
        ["python", "-m", "asv", "machine", "--machine=asvrunner", "--yes"],
        cwd=asv,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "asv",
            "run",
            "--machine=asvrunner",
            "--python=same",
            f"--set-commit-hash={args.sha}",
            "--show-stderr",
        ],
        cwd=asv,
        check=True,
    )
