from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

import pytest

from asv_runner_v2 import issues
from asv_runner_v2.store import Store
from tests.v2._helpers import CONDA_ENV, make_result, write_benchmark_defs, write_zst

DAY_MS = 86_400_000
START = 1772845825000
NAMES = [f"mod{k}.Bench.time_x" for k in range(6)]


class FakeGitHub(issues.GitHub):
    def __init__(self, existing: Collection[str] = (), pr: dict | None = None) -> None:
        super().__init__(dry_run=True)
        self.existing = existing
        self.pr = pr

    def issue_titles(self, sha: str) -> list[str]:
        return [t for t in self.existing if sha in t]

    def pr_info(self, prev_sha: str, sha: str) -> dict | None:
        return self.pr

    def changed_files(self, prev_sha: str, sha: str) -> list[str] | None:
        return ["pandas/core/frame.py"]


def _sha(i: int) -> str:
    return f"{i:040x}"


def _store(tmp_path: Path, slow: list[str], newer_env_at: int | None = None) -> Store:
    """80 commits; benchmarks in ``slow`` get 1.5x slower from commit 40."""
    input_dir = tmp_path / "input"
    for i in range(80):
        results = {}
        for name in NAMES:
            level = 1.5 if (name in slow and i >= 40) else 1.0
            results[name] = [
                [level * (1 + 0.001 * (i % 3)), 2.0],
                [["1", "2"]],
                "v1",
                1772940858665,
                1.0,
            ]
        sha = _sha(i)
        d = make_result(sha, START + i * DAY_MS, results)
        write_zst(
            input_dir / "data" / "results" / "asvrunner" / f"{sha}.json.zst",
            json.dumps(d).encode(),
        )
        env = CONDA_ENV
        if newer_env_at is not None and i >= newer_env_at:
            env = CONDA_ENV.replace("numpy=2.3.1", "numpy=2.4.0")
        write_zst(input_dir / "data" / "envs" / f"{sha}.yml.zst", env.encode())
    write_benchmark_defs(
        input_dir, {n: {"version": "v1", "param_names": ["n"]} for n in NAMES}
    )
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)
    return store


def test_regression_issue(tmp_path: Path) -> None:
    store = _store(tmp_path, slow=NAMES[:2], newer_env_at=40)
    gh = FakeGitHub(pr={"number": 123, "author": "alice", "approvers": ["bob"]})
    raised = issues.raise_issues(
        store, gh, site_url="https://x.test/", breaks=[], pause=0
    )
    assert raised == [_sha(40)]
    [issue] = gh.would_create
    assert issue["title"] == f"Commit {_sha(40)}"
    assert issue["labels"] == []
    body = issue["body"]
    assert "[PR #123](https://github.com/pandas-dev/pandas/pull/123)" in body
    assert "cc @alice @bob" in body
    assert f"(https://x.test/#commit/{_sha(40)})" in body
    assert "Environment changed at this commit: numpy: 2.3.1 → 2.4.0" in body
    # One line per benchmark, one per regressed parameter combination, with a
    # link in the issue-link format the site parses.
    assert " - [ ] [mod0.Bench.time_x](https://x.test/#mod0.Bench.time_x)" in body
    assert "[n=1](https://x.test/#mod0.Bench.time_x?p-n=1) - +50.0%" in body
    assert "mod2" not in body
    assert "numpy: 2.3.1 → 2.4.0" in issue["comment"]


def test_existing_issue_is_skipped(tmp_path: Path) -> None:
    store = _store(tmp_path, slow=NAMES[:1])
    gh = FakeGitHub(existing={f"Commit {_sha(40)}"})
    assert issues.raise_issues(store, gh, breaks=[], pause=0) == []
    assert gh.would_create == []


def test_mass_change_issue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(issues, "MASS_CHANGE", 3)
    store = _store(tmp_path, slow=NAMES, newer_env_at=40)
    gh = FakeGitHub(pr={"number": 123, "author": "alice", "approvers": []})
    issues.raise_issues(store, gh, site_url="https://x.test/", breaks=[], pause=0)
    [issue] = gh.would_create
    assert issue["title"].startswith(f"Commit {_sha(40)}: 6 benchmarks slower")
    assert issue["labels"] == ["measurement-change"]
    assert issue["comment"] is None
    body = issue["body"]
    assert "@" not in body.replace("@github", "")  # nobody is pinged
    assert "6 benchmarks across 6 modules got slower (median +50%" in body
    assert "| Key packages | `numpy` 2.3.1 → 2.4.0 |" in body
    assert "| Benchmark code (`asv_bench/`) | unchanged |" in body
    assert f'sha = "{_sha(40)}"' in body
    assert "<summary>All 6 benchmarks</summary>" in body


def test_dry_run_file(tmp_path: Path) -> None:
    store = _store(tmp_path, slow=NAMES[:1])
    gh = FakeGitHub()
    issues.raise_issues(store, gh, breaks=[], pause=0)
    out = tmp_path / "issues.md"
    issues.write_dry_run(gh, out)
    text = out.read_text()
    assert text.startswith("# Issues the new pipeline would open (1)")
    assert f"## Commit {_sha(40)}" in text
    assert "[Commit Range](https://github.com/pandas-dev/pandas/compare/" in text


@pytest.mark.parametrize(
    ("titles", "expected"),
    [
        ([f"Commit {'a' * 40}"], True),
        ([f"Commit {'a' * 40}: 120 benchmarks slower; check"], True),
        # Another commit's issue that only mentions this SHA does not count.
        ([f"Commit {'b' * 40}"], False),
        ([f"Re: Commit {'a' * 40}"], False),
        ([], False),
    ],
)
def test_is_issue_for(titles: list[str], expected: bool) -> None:
    assert issues.is_issue_for("a" * 40, titles) is expected


def test_mass_change_issue_counts_as_existing(tmp_path: Path) -> None:
    store = _store(tmp_path, slow=NAMES[:1])
    gh = FakeGitHub(existing={f"Commit {_sha(40)}: 6 benchmarks slower; check"})
    assert issues.raise_issues(store, gh, breaks=[], pause=0) == []
