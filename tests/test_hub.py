from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import polars as pl
import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from asv_runner import hub
from asv_runner.store import Store
from tests._results_helpers import add_commit

DATASET = "pandas-dev/asv-runner"
WEEK10 = 1772932225000  # 2026-03-08
WEEK12 = 1773900000000  # 2026-03-19
NOW = dt.datetime(2026, 4, 1, tzinfo=dt.UTC)
MONTH = dt.timedelta(days=30)


class ParentMismatchError(Exception):
    pass


@dataclass
class Commit:
    commit_id: str
    created_at: dt.datetime
    files: dict[str, bytes]


class FakeHub:
    """The parts of HfApi that asv_runner.hub uses, over an in-memory repo."""

    def __init__(self) -> None:
        self.clock = NOW - dt.timedelta(days=60)
        self.commits: list[Commit] = []  # oldest first
        self._commit({".gitattributes": b"", "README.md": b"card"})

    @property
    def api(self) -> HfApi:
        return cast("HfApi", self)

    def _commit(self, files: dict[str, bytes]) -> str:
        sha = hashlib.sha1(f"{len(self.commits)}".encode()).hexdigest()
        self.commits.append(Commit(sha, self.clock, files))
        self.clock += dt.timedelta(days=1)
        return sha

    def _at(self, revision: str) -> Commit:
        if revision == hub.BRANCH:
            return self.commits[-1]
        return next(c for c in self.commits if c.commit_id == revision)

    def dataset_info(self, dataset: str, revision: str) -> SimpleNamespace:
        return SimpleNamespace(sha=self._at(revision).commit_id)

    def snapshot_download(
        self, dataset, repo_type, revision, local_dir, allow_patterns
    ) -> None:
        for path, data in self._at(revision).files.items():
            if fnmatch.fnmatch(path, allow_patterns):
                (Path(local_dir) / path).parent.mkdir(parents=True, exist_ok=True)
                (Path(local_dir) / path).write_bytes(data)

    def list_repo_files(self, dataset, repo_type, revision) -> list[str]:
        return list(self._at(revision).files)

    def create_commit(
        self, dataset, operations, commit_message, repo_type, revision, parent_commit
    ) -> SimpleNamespace:
        if parent_commit != self.commits[-1].commit_id:
            raise ParentMismatchError(parent_commit)
        files = dict(self.commits[-1].files)
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
            else:
                assert isinstance(op, CommitOperationDelete)
                del files[op.path_in_repo]
        return SimpleNamespace(oid=self._commit(files))

    def list_repo_commits(self, dataset, repo_type, revision) -> list[Commit]:
        return self.commits[::-1]

    def super_squash_history(self, dataset, branch, repo_type, commit_message) -> None:
        files = self.commits[-1].files
        self.commits = []
        self._commit(files)


@pytest.fixture
def repo() -> FakeHub:
    return FakeHub()


def make_store(tmp_path: Path, *shas_dates: tuple[str, int]) -> Store:
    input_dir = tmp_path / "input"
    for sha, date in shas_dates:
        add_commit(input_dir, sha, date)
    store = Store(tmp_path / "work" / "data")
    store.update(input_dir, workers=1)
    return store


def files_under(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_pull_empty_repo(tmp_path: Path, repo: FakeHub) -> None:
    rev = tmp_path / "rev"
    sha = hub.pull(repo.api, DATASET, tmp_path / "store", rev)
    assert sha == repo.commits[-1].commit_id
    assert rev.read_text() == f"{sha}\n"
    assert (tmp_path / "store" / "data").is_dir()
    assert files_under(tmp_path / "store" / "data") == {}


def test_push_then_pull_round_trips(tmp_path: Path, repo: FakeHub) -> None:
    rev = tmp_path / "rev"
    hub.pull(repo.api, DATASET, tmp_path / "work", rev)
    store = make_store(tmp_path, ("aaa", WEEK10))
    sha = hub.push(repo.api, DATASET, store.root, rev, "Update store")
    assert rev.read_text() == f"{sha}\n"
    assert repo.commits[-1].files["README.md"] == b"card"

    hub.pull(repo.api, DATASET, tmp_path / "again", tmp_path / "rev2")
    assert files_under(tmp_path / "again" / "data") == files_under(store.root)


def test_push_deletes_removed_files(tmp_path: Path, repo: FakeHub) -> None:
    rev = tmp_path / "rev"
    hub.pull(repo.api, DATASET, tmp_path / "work", rev)
    store = make_store(tmp_path, ("aaa", WEEK10), ("ccc", WEEK12))
    hub.push(repo.api, DATASET, store.root, rev, "Update store")
    shutil.rmtree(store.root / "values" / "week=2026-W12")
    hub.push(repo.api, DATASET, store.root, rev, "Update store")
    files = repo.commits[-1].files
    assert "data/values/week=2026-W12/data.parquet" not in files
    assert "data/values/week=2026-W10/data.parquet" in files
    assert "README.md" in files


def test_push_fails_if_repo_moved(tmp_path: Path, repo: FakeHub) -> None:
    rev = tmp_path / "rev"
    hub.pull(repo.api, DATASET, tmp_path / "work", rev)
    repo._commit({**repo.commits[-1].files, "data/other": b"x"})
    store = make_store(tmp_path, ("aaa", WEEK10))
    with pytest.raises(ParentMismatchError):
        hub.push(repo.api, DATASET, store.root, rev, "Update store")


def pushed_twice(tmp_path: Path, repo: FakeHub) -> tuple[Store, Path]:
    """Push a store, then push it again with a commit added; history starts 60
    days before NOW."""
    rev = tmp_path / "rev"
    hub.pull(repo.api, DATASET, tmp_path / "work", rev)
    store = make_store(tmp_path, ("aaa", WEEK10))
    hub.push(repo.api, DATASET, store.root, rev, "Update store")
    add_commit(tmp_path / "input", "ccc", WEEK12)
    store.update(tmp_path / "input", workers=1)
    hub.push(repo.api, DATASET, store.root, rev, "Update store")
    return store, rev


def test_squash(tmp_path: Path, repo: FakeHub) -> None:
    store, rev = pushed_twice(tmp_path, repo)
    head = repo.commits[-1].files
    assert hub.squash(repo.api, DATASET, store.root, rev, MONTH, now=NOW) == []
    assert len(repo.commits) == 1
    assert repo.commits[0].files == head


def test_no_squash_while_history_is_recent(tmp_path: Path, repo: FakeHub) -> None:
    store, rev = pushed_twice(tmp_path, repo)
    year = dt.timedelta(days=365)
    assert hub.squash(repo.api, DATASET, store.root, rev, year, now=NOW) == []
    assert len(repo.commits) == 3


def test_no_squash_if_check_fails(tmp_path: Path, repo: FakeHub) -> None:
    rev = tmp_path / "rev"
    hub.pull(repo.api, DATASET, tmp_path / "work", rev)
    store = make_store(tmp_path, ("aaa", WEEK10))
    # Make the oldest commit hold the store, then change a stored row.
    repo.commits[0].files.update(
        {f"data/{k}": v for k, v in files_under(store.root).items()}
    )
    runs = store.runs().with_columns(meta=pl.lit("{}"))
    runs.write_parquet(store.runs_path)
    hub.push(repo.api, DATASET, store.root, rev, "Update store")

    problems = hub.squash(repo.api, DATASET, store.root, rev, MONTH, now=NOW)
    assert problems
    assert problems[0].startswith(f"Checking against {repo.commits[0].commit_id}: ")
    assert "runs.parquet: 1 stored rows changed or removed" in problems[0]
    assert len(repo.commits) == 2


def test_no_squash_if_head_is_not_what_was_pushed(
    tmp_path: Path, repo: FakeHub
) -> None:
    store, rev = pushed_twice(tmp_path, repo)
    repo._commit({**repo.commits[-1].files, "data/other": b"x"})
    [problem] = hub.squash(repo.api, DATASET, store.root, rev, MONTH, now=NOW)
    assert problem.startswith(f"The head of {DATASET} is ")
    assert len(repo.commits) == 4
