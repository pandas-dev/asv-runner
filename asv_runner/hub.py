"""Keep the Parquet store in a Hugging Face dataset repo.

The store lives under ``data/`` in the dataset repo, so the dataset card
(``README.md``) and anything else at the top level are never touched.

- ``pull-store`` downloads the store at the commit at the head of ``main`` and
  records that commit.
- ``push-store`` uploads the store as a single commit whose parent must be the
  recorded commit, so it fails if anything else pushed in the meantime.
- ``squash-store`` squashes the repo's history to one commit once the oldest
  commit is old enough, and only if ``check-store`` passes between the oldest
  commit and the store just pushed.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from asv_runner import check_store
from asv_runner.store import Store

REPO_TYPE = "dataset"
BRANCH = "main"
PREFIX = "data"


def head(api: HfApi, dataset: str) -> str:
    """The commit at the head of ``main``."""
    sha = api.dataset_info(dataset, revision=BRANCH).sha
    if sha is None:
        raise RuntimeError(f"Hugging Face did not report the head of {dataset}")
    return sha


def _download(api: HfApi, dataset: str, revision: str, local_dir: Path) -> Path:
    """Download the store at ``revision`` and return its root."""
    api.snapshot_download(
        dataset,
        repo_type=REPO_TYPE,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=f"{PREFIX}/**",
    )
    root = Path(local_dir) / PREFIX
    root.mkdir(parents=True, exist_ok=True)  # an empty repo has no data/
    return root


def pull(api: HfApi, dataset: str, local_dir: Path, revision_file: Path) -> str:
    """Download the store into ``local_dir/data``; return the commit downloaded."""
    sha = head(api, dataset)
    _download(api, dataset, sha, local_dir)
    Path(revision_file).write_text(f"{sha}\n")
    return sha


def push(
    api: HfApi, dataset: str, store_root: Path, revision_file: Path, message: str
) -> str:
    """Upload the store at ``store_root`` on top of the commit ``pull`` recorded.

    Files that did not change are skipped, and files no longer in the store are
    deleted, all in one commit. Records and returns the new commit.
    """
    parent = Path(revision_file).read_text().strip()
    store_root = Path(store_root)
    local = {
        f"{PREFIX}/{p.relative_to(store_root).as_posix()}": p
        for p in sorted(store_root.rglob("*"))
        if p.is_file()
    }
    remote = {
        f
        for f in api.list_repo_files(dataset, repo_type=REPO_TYPE, revision=parent)
        if f.startswith(f"{PREFIX}/")
    }
    operations = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(file))
        for path, file in local.items()
    ] + [
        CommitOperationDelete(path_in_repo=path)
        for path in sorted(remote - local.keys())
    ]
    info = api.create_commit(
        dataset,
        operations,
        commit_message=message,
        repo_type=REPO_TYPE,
        revision=BRANCH,
        parent_commit=parent,
    )
    Path(revision_file).write_text(f"{info.oid}\n")
    return info.oid


def squash(
    api: HfApi,
    dataset: str,
    store_root: Path,
    revision_file: Path,
    max_age: dt.timedelta,
    now: dt.datetime | None = None,
) -> list[str]:
    """Squash history if the oldest commit is older than ``max_age``.

    ``store_root`` and ``revision_file`` are the store and commit ``push`` left.
    Returns the problems that stopped a squash; empty if it squashed or had
    nothing to do. Problems leave history in place, to be retried next time.
    """
    now = now or dt.datetime.now(dt.UTC)
    commits = api.list_repo_commits(dataset, repo_type=REPO_TYPE, revision=BRANCH)
    oldest = commits[-1]
    if len(commits) == 1 or now - oldest.created_at <= max_age:
        print(
            f"Not squashing: {len(commits)} commits, the oldest from "
            f"{oldest.created_at:%Y-%m-%d}."
        )
        return []
    pushed = Path(revision_file).read_text().strip()
    if (current := head(api, dataset)) != pushed:
        return [f"The head of {dataset} is {current}, not the commit pushed, {pushed}"]
    with tempfile.TemporaryDirectory(prefix="asv-runner-oldest-") as tmp:
        old = _download(api, dataset, oldest.commit_id, Path(tmp))
        problems = check_store.check(Store(old), Store(Path(store_root)))
    if problems:
        return [f"Checking against {oldest.commit_id}: {p}" for p in problems]
    api.super_squash_history(
        dataset,
        branch=BRANCH,
        repo_type=REPO_TYPE,
        commit_message=f"Squash history before {now:%Y-%m-%d}",
    )
    print(
        f"Squashed {len(commits)} commits back to {oldest.created_at:%Y-%m-%d} "
        "into one."
    )
    return []


def pull_cmd(args) -> int:
    sha = pull(HfApi(), args.dataset, Path(args.out), Path(args.revision_file))
    print(f"Downloaded {args.dataset} at {sha}.")
    return 0


def push_cmd(args) -> int:
    sha = push(
        HfApi(),
        args.dataset,
        Path(args.store),
        Path(args.revision_file),
        args.message,
    )
    print(f"Pushed {args.dataset} at {sha}.")
    return 0


def squash_cmd(args) -> int:
    problems = squash(
        HfApi(),
        args.dataset,
        Path(args.store),
        Path(args.revision_file),
        dt.timedelta(days=args.max_age_days),
    )
    for problem in problems:
        print(problem)
    return 1 if problems else 0
