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
            # forkserver imports pandas once and forks per benchmark; asv
            # >=0.6.6 defaults to "spawn", which re-imports pandas in every
            # benchmark process and blows past the 6h job limit (GH#150).
            "--launch-method=forkserver",
            "--show-stderr",
        ],
        cwd=asv,
        check=True,
    )
