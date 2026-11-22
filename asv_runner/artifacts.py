"""Hand results from the hourly benchmark runs to the nightly run.

Each hourly run uploads its staged results as a GitHub Actions artifact named
``result-<sha>`` (see ``stage_results``), kept for 90 days. The nightly run
downloads the artifacts of commits not yet in the store into a directory with
the results-branch layout that ``Store.update`` reads, and checks that no
completed lease has lost its results.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from asv_runner.claim import read_claims
from asv_runner.store import Store
from asv_runner.util import gh_bytes, github_repo

ARTIFACT_PREFIX = "result-"
# A done lease older than this whose results are neither stored nor uploaded
# has been lost: the nightly run should have stored it by now.
LOST_AFTER = timedelta(days=2)


def list_result_artifacts(repo: str) -> list[dict]:
    """Unexpired ``result-<sha>`` artifacts, oldest first."""
    out = gh_bytes(
        [
            *("api", "--paginate", f"repos/{repo}/actions/artifacts?per_page=100"),
            *("--jq", ".artifacts[] | {id, name, expired, created_at}"),
        ]
    )
    artifacts = [json.loads(line) for line in out.decode().splitlines() if line]
    return sorted(
        (
            a
            for a in artifacts
            if a["name"].startswith(ARTIFACT_PREFIX) and not a["expired"]
        ),
        key=lambda a: a["created_at"],
    )


def download(repo: str, artifact_id: int) -> bytes:
    return gh_bytes(["api", f"repos/{repo}/actions/artifacts/{artifact_id}/zip"])


def fetch(store: Store, out: Path, repo: str) -> list[str]:
    """Extract the artifacts of commits not in the store into ``out``.

    Artifacts are extracted oldest first, so ``benchmarks.json`` ends up from
    the newest run. Returns the commits fetched.
    """
    stored = store.stored_shas()
    fetched = []
    for artifact in list_result_artifacts(repo):
        sha = artifact["name"].removeprefix(ARTIFACT_PREFIX)
        if sha in stored:
            continue
        with zipfile.ZipFile(io.BytesIO(download(repo, artifact["id"]))) as zf:
            zf.extractall(out)
        fetched.append(sha)
    return fetched


def lost_results(
    store: Store, shas_path: Path, now: datetime | None = None
) -> list[str]:
    """Done leases older than LOST_AFTER whose results are not in the store."""
    now = now or datetime.now(UTC)
    stored = store.stored_shas()
    return [
        claim.sha
        for claim in read_claims(shas_path)
        if claim.done
        and claim.sha not in stored
        and now - claim.claimed_at > LOST_AFTER
    ]


def fetch_cmd(args: argparse.Namespace) -> int:
    fetched = fetch(Store(Path(args.store)), Path(args.out), github_repo())
    print(f"Downloaded results for {len(fetched)} commits.")
    return 0


def check_cmd(args: argparse.Namespace) -> int:
    lost = lost_results(Store(Path(args.store)), Path(args.claims))
    if not lost:
        print("Every completed lease has its results stored.")
        return 0
    print(
        f"{len(lost)} commits were benchmarked more than {LOST_AFTER.days} days ago "
        "but their results are not in the store, and their artifacts are gone "
        "or were never uploaded:"
    )
    for sha in lost:
        print(f"  {sha}")
    return 1
