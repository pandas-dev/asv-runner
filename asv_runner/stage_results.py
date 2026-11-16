"""Stage asv outputs (zstd-compressed per-sha files) for upload as an artifact.

The output directory has the layout of the old results branch, which
``python -m asv_runner update`` reads::

    data/results/benchmarks.json
    data/results/asvrunner/<sha>.json.zst
    data/envs/<sha>.yml.zst

The hourly benchmark run uploads it as artifact ``result-<sha>``; the nightly
run downloads the artifacts of commits not yet in the store.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(args: argparse.Namespace) -> None:
    asv = Path(args.asv_dir)
    out = Path(args.out_dir)
    sha = args.sha
    short_sha = sha[:8]

    asv_results = asv / "results" / "asvrunner"
    matches = list(asv_results.glob(f"{short_sha}-existing*.json"))
    if not matches:
        raise RuntimeError(f"No matching result file for short sha {short_sha}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matches for short sha {short_sha}: {matches}")

    results = out / "data" / "results"
    (results / "asvrunner").mkdir(parents=True, exist_ok=True)
    envs = out / "data" / "envs"
    envs.mkdir(parents=True, exist_ok=True)

    shutil.copy(asv / "results" / "benchmarks.json", results / "benchmarks.json")
    sha_json = results / "asvrunner" / f"{sha}.json"
    sha_yml = envs / f"{sha}.yml"
    shutil.copy(matches[0], sha_json)
    with sha_yml.open("w") as f:
        subprocess.run(
            ["pixi", "list", "-e", "asv", "--fields", "name,version"],
            stdout=f,
            check=True,
        )
    subprocess.run(["zstd", "--rm", "-19", "-q", str(sha_json)], check=True)
    subprocess.run(["zstd", "--rm", "-19", "-q", str(sha_yml)], check=True)
