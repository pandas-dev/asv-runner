"""Stage asv outputs (zstd-compress per-sha files) and push to storage."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from asv_runner.util import orphan_push_with_retry


def run(args: argparse.Namespace) -> None:
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

    orphan_push_with_retry(
        storage,
        branch=args.branch,
        message="Results",
        modify_tree=modify_tree,
    )
