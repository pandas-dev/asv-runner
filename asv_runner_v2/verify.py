"""Check that the store rebuilds the original result and env files exactly."""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from asv_runner_v2.store import ENV_SUFFIX, RESULT_SUFFIX, Store, read_zst


@dataclass
class VerifyReport:
    results_checked: int = 0
    envs_checked: int = 0
    mismatches: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.missing


def _verify_week(args: tuple[Path, Path, str]) -> tuple[list[str], list[str]]:
    root, results_dir, week = args
    checked, bad = [], []
    for sha, rebuilt in Store(root).rebuild_week(week):
        checked.append(sha)
        original = json.loads(read_zst(results_dir / f"{sha}{RESULT_SUFFIX}"))
        # Compare serialized text so int vs float, NaN vs null, and key order
        # all count as differences.
        if json.dumps(original) != json.dumps(rebuilt):
            bad.append(sha)
    return checked, bad


def verify(store: Store, input_dir: Path, workers: int | None = None) -> VerifyReport:
    """Rebuild every stored file and compare it to the original under ``input_dir``.

    Also reports original files that are not in the store.
    """
    input_dir = Path(input_dir)
    results_dir = input_dir / "data" / "results" / "asvrunner"
    envs_dir = input_dir / "data" / "envs"
    report = VerifyReport()

    ctx = multiprocessing.get_context("forkserver")
    seen: set[str] = set()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        jobs = [(store.root, results_dir, week) for week in store.weeks()]
        for checked, bad in pool.map(_verify_week, jobs):
            seen.update(checked)
            report.mismatches.extend(f"result {sha}" for sha in bad)
    report.results_checked = len(seen)
    for path in results_dir.glob(f"*{RESULT_SUFFIX}"):
        if path.name[: -len(RESULT_SUFFIX)] not in seen:
            report.missing.append(f"result {path.name}")

    stored_env = store.env_text() if store.env_files_path.exists() else {}
    for path in envs_dir.glob(f"*{ENV_SUFFIX}"):
        sha = path.name[: -len(ENV_SUFFIX)]
        if sha not in stored_env:
            report.missing.append(f"env {path.name}")
            continue
        report.envs_checked += 1
        if read_zst(path).decode() != stored_env[sha]:
            report.mismatches.append(f"env {sha}")
    return report
