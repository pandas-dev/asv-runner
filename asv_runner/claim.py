"""Lease the next pandas SHA to benchmark, update shas.txt, push storage branch.

Each line of shas.txt is a lease: ``<sha> <claimed_at> <attempts> [abandoned]``.
A SHA is complete when its results file exists on the storage branch; a lease
with no results whose age exceeds CLAIM_TTL belonged to a run that died
(timeout, build failure, runner loss) and is retried up to MAX_ATTEMPTS times.
Legacy bare-sha lines parse as expired leases with no attempts on record.
Detected failures are reported as comments on a single tracking issue.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from asv_runner.util import execute, orphan_push_with_retry, write_github_output

LOOKBACK_COMMITS = 40
# GitHub kills a job after 6 hours, so a lease older than this with no
# results is guaranteed to belong to a dead run.
CLAIM_TTL = timedelta(hours=7)
MAX_ATTEMPTS = 3
FAILURE_ISSUE_TITLE = "Benchmark run failures"
EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)


@dataclass
class Claim:
    sha: str
    claimed_at: datetime
    attempts: int
    abandoned: bool = False

    @classmethod
    def from_line(cls, line: str) -> Claim:
        parts = line.split()
        if len(parts) == 1:
            # Legacy bare-sha line: an expired lease with no attempts on
            # record. If its results exist it is complete; otherwise it is
            # eligible for retry.
            return cls(sha=parts[0], claimed_at=EPOCH, attempts=0)
        return cls(
            sha=parts[0],
            claimed_at=datetime.fromisoformat(parts[1]),
            attempts=int(parts[2]),
            abandoned=len(parts) > 3 and parts[3] == "abandoned",
        )

    def to_line(self) -> str:
        line = f"{self.sha} {self.claimed_at.isoformat()} {self.attempts}"
        if self.abandoned:
            line += " abandoned"
        return line


def read_claims(shas_path: Path) -> list[Claim]:
    if not shas_path.exists():
        return []
    return [
        Claim.from_line(line)
        for line in shas_path.read_text().splitlines()
        if line.strip()
    ]


def write_claims(shas_path: Path, claims: list[Claim]) -> None:
    shas_path.write_text("".join(f"{claim.to_line()}\n" for claim in claims))


def has_results(storage: Path, sha: str) -> bool:
    return (storage / "data" / "results" / "asvrunner" / f"{sha}.json.zst").exists()


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


def find_failure_issue() -> str | None:
    result = execute(
        "gh issue list"
        " --repo pandas-dev/asv-runner"
        " --state open"
        " --limit 1000"
        " --json number,title"
    )
    for issue in json.loads(result):
        if issue["title"] == FAILURE_ISSUE_TITLE:
            return str(issue["number"])
    return None


def notify_failures(events: list[str]) -> None:
    """Report detected run failures on a single tracking issue."""
    body = "\n".join(f"- {event}" for event in events)
    issue_number = find_failure_issue()
    if issue_number is None:
        cmd = (
            "gh issue create"
            " --repo pandas-dev/asv-runner"
            f' --title "{FAILURE_ISSUE_TITLE}"'
            " --body-file -"
        )
    else:
        cmd = (
            f"gh issue comment {issue_number}"
            " --repo pandas-dev/asv-runner"
            " --body-file -"
        )
    execute(cmd, input=body)


def run(args: argparse.Namespace) -> None:
    storage = Path(args.storage_dir)
    repo = Path(args.repo_dir)
    shas_path = storage / "data" / "shas.txt"

    # Mutable cells smuggle results out of modify_tree so the caller can
    # report them once the push lands. Rebuilt on every call because each
    # push attempt refetches the branch.
    picked: list[str | None] = [None]
    events: list[str] = []

    def modify_tree(tree: Path) -> bool:
        picked[0] = None
        events.clear()
        claims = read_claims(shas_path)
        now = datetime.now(timezone.utc)
        changed = False

        sha = pick_next_sha(repo, existing_shas={claim.sha for claim in claims})
        if sha is not None:
            claims.append(Claim(sha=sha, claimed_at=now, attempts=1))
            picked[0] = sha
            changed = True
        else:
            # No fresh commit: spend the idle slot retrying the newest stale
            # lease, marking exhausted ones abandoned along the way.
            for claim in reversed(claims):
                if claim.abandoned or has_results(tree, claim.sha):
                    continue
                if now - claim.claimed_at < CLAIM_TTL:
                    # A runner may still be working on it.
                    continue
                if claim.attempts >= MAX_ATTEMPTS:
                    claim.abandoned = True
                    events.append(
                        f"Giving up on `{claim.sha}`:"
                        f" no results after {claim.attempts} attempts."
                    )
                    changed = True
                    continue
                claim.attempts += 1
                claim.claimed_at = now
                if claim.attempts > 1:
                    # attempts == 1 is a legacy bare line being backfilled,
                    # not a newly detected death.
                    ttl_hours = int(CLAIM_TTL.total_seconds() // 3600)
                    events.append(
                        f"Run for `{claim.sha}` produced no results within"
                        f" {ttl_hours} hours; retrying"
                        f" (attempt {claim.attempts}/{MAX_ATTEMPTS})."
                    )
                picked[0] = claim.sha
                changed = True
                break

        if changed:
            write_claims(shas_path, claims)
        return changed

    pushed = orphan_push_with_retry(
        storage,
        branch=args.branch,
        message="Update shas.txt",
        modify_tree=modify_tree,
    )
    if pushed and picked[0] is not None:
        write_github_output(sha=picked[0], new_commit="yes")
    else:
        write_github_output(sha="NONE", new_commit="no")
    if pushed and events:
        try:
            notify_failures(events)
        except Exception as err:
            # Notification is best-effort; never fail the job (and thereby
            # waste the lease we just pushed) over it.
            print(f"Failed to post failure notification: {err}", file=sys.stderr)
