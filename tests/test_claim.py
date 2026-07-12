from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from asv_runner import claim as claim_mod
from asv_runner.claim import (
    EPOCH,
    LOOKBACK_COMMITS,
    MAX_ATTEMPTS,
    Claim,
    find_failure_issue,
    notify_failures,
    pick_next_sha,
    read_claims,
)
from asv_runner.claim import run as cmd_claim
from tests._helpers import init_remote_and_storage

OLD_TS = "2020-01-01T00:00:00+00:00"

# === pick_next_sha ===


class _FakeCompleted:
    def __init__(self, stdout: bytes) -> None:
        self.stdout = stdout


def _fake_subprocess_run(shas: list[str]) -> Callable[..., _FakeCompleted]:
    out = "\n".join(f"{sha} commit message {sha}" for sha in shas).encode()

    def _run(cmd: Any, **kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(out)

    return _run


def test_pick_next_sha_no_existing_picks_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(["sha1111", "sha2222"]))
    assert pick_next_sha(tmp_path, existing_shas=set()) == "sha1111"


def test_pick_next_sha_skips_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_subprocess_run(["sha1111", "sha2222", "sha3333"]),
    )
    assert pick_next_sha(tmp_path, existing_shas={"sha1111", "sha2222"}) == "sha3333"


def test_pick_next_sha_all_existing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(["sha1111", "sha2222"]))
    assert pick_next_sha(tmp_path, existing_shas={"sha1111", "sha2222"}) is None


def test_pick_next_sha_invokes_git_log_with_expected_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd: Any, **kwargs: Any) -> _FakeCompleted:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        return _FakeCompleted(b"sha1111 msg\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pick_next_sha(tmp_path, existing_shas=set())

    assert captured["cmd"] == [
        "git",
        "log",
        f"-{LOOKBACK_COMMITS}",
        "--oneline",
        "--no-abbrev-commit",
    ]
    assert captured["cwd"] == tmp_path


# === Claim parsing ===


def test_claim_from_bare_line_is_expired_lease() -> None:
    result = Claim.from_line("sha1111")
    assert result == Claim(sha="sha1111", claimed_at=EPOCH, attempts=0)


def test_claim_from_full_line() -> None:
    result = Claim.from_line(f"sha1111 {OLD_TS} 2")
    assert result == Claim(
        sha="sha1111",
        claimed_at=datetime.fromisoformat(OLD_TS),
        attempts=2,
    )


def test_claim_from_abandoned_line() -> None:
    assert Claim.from_line(f"sha1111 {OLD_TS} 3 abandoned").abandoned


@pytest.mark.parametrize("abandoned", [False, True])
def test_claim_line_roundtrip(abandoned: bool) -> None:
    original = Claim(
        sha="sha1111",
        claimed_at=datetime.fromisoformat(OLD_TS),
        attempts=2,
        abandoned=abandoned,
    )
    assert Claim.from_line(original.to_line()) == original


def test_read_claims_missing_file(tmp_path: Path) -> None:
    assert read_claims(tmp_path / "missing.txt") == []


def test_read_claims_ignores_blank_lines(tmp_path: Path) -> None:
    shas_path = tmp_path / "shas.txt"
    shas_path.write_text(f"sha1111\n\nsha2222 {OLD_TS} 2\n")
    assert [claim.sha for claim in read_claims(shas_path)] == ["sha1111", "sha2222"]


# === cmd_claim integration ===


def _init_pandas_repo(tmp_path: Path, n_commits: int = 2) -> tuple[Path, list[str]]:
    """Create a tiny git repo with n synthetic commits and return its path
    and the resulting SHAs (most-recent first)."""
    repo = tmp_path / "pandas"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
    shas: list[str] = []
    for i in range(n_commits):
        (repo / "f").write_text(str(i))
        subprocess.run(["git", "add", "f"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", f"commit {i}"], cwd=repo, check=True)
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        shas.append(sha)
    return repo, list(reversed(shas))


def _add_results(storage: Path, sha: str) -> None:
    results = storage / "data" / "results" / "asvrunner"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{sha}.json.zst").write_text("")


@pytest.fixture
def notifications(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture failure notifications instead of shelling out to gh."""
    captured: list[str] = []
    monkeypatch.setattr(claim_mod, "notify_failures", captured.extend)
    return captured


def _claim_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_commits: int = 2,
) -> tuple[Path, Path, list[str], argparse.Namespace, Path]:
    remote, storage = init_remote_and_storage(tmp_path)
    pandas_repo, shas = _init_pandas_repo(tmp_path, n_commits=n_commits)
    output_file = tmp_path / "github_output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    args = argparse.Namespace(
        storage_dir=str(storage),
        repo_dir=str(pandas_repo),
        branch="pandas_test",
    )
    return remote, storage, shas, args, output_file


def _pushed_claims(tmp_path: Path, remote: Path) -> list[Claim]:
    verify = tmp_path / "verify"
    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(verify)],
        check=True,
    )
    return read_claims(verify / "data" / "shas.txt")


def test_cmd_claim_picks_sha_and_pushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notifications: list[str]
) -> None:
    remote, _, shas, args, output_file = _claim_scenario(tmp_path, monkeypatch)
    cmd_claim(args)

    out = output_file.read_text()
    head_sha = shas[0]
    assert f"sha={head_sha}" in out
    assert "new_commit=yes" in out

    (pushed,) = _pushed_claims(tmp_path, remote)
    assert pushed.sha == head_sha
    assert pushed.attempts == 1
    assert not pushed.abandoned
    assert datetime.now(timezone.utc) - pushed.claimed_at < timedelta(minutes=5)
    assert notifications == []


def test_cmd_claim_appends_next_unseen_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notifications: list[str]
) -> None:
    remote, storage, shas, args, output_file = _claim_scenario(
        tmp_path, monkeypatch, n_commits=3
    )
    head_sha, second_sha, third_sha = shas
    # Pre-seed shas.txt with everything except the most recent commit.
    (storage / "data" / "shas.txt").write_text(f"{third_sha}\n{second_sha}\n")
    cmd_claim(args)

    assert f"sha={head_sha}" in output_file.read_text()
    assert [claim.sha for claim in _pushed_claims(tmp_path, remote)] == [
        third_sha,
        second_sha,
        head_sha,
    ]


def test_cmd_claim_no_new_commit_when_all_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notifications: list[str]
) -> None:
    _, storage, shas, args, output_file = _claim_scenario(tmp_path, monkeypatch)
    (storage / "data" / "shas.txt").write_text("\n".join(reversed(shas)) + "\n")
    for sha in shas:
        _add_results(storage, sha)
    cmd_claim(args)

    out = output_file.read_text()
    assert "sha=NONE" in out
    assert "new_commit=no" in out
    assert notifications == []


def test_cmd_claim_retries_legacy_bare_sha_without_notifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notifications: list[str]
) -> None:
    remote, storage, shas, args, output_file = _claim_scenario(tmp_path, monkeypatch)
    head_sha, second_sha = shas
    (storage / "data" / "shas.txt").write_text(f"{second_sha}\n{head_sha}\n")
    _add_results(storage, second_sha)
    cmd_claim(args)

    out = output_file.read_text()
    assert f"sha={head_sha}" in out
    assert "new_commit=yes" in out

    retried = {claim.sha: claim for claim in _pushed_claims(tmp_path, remote)}
    assert retried[head_sha].attempts == 1
    assert retried[second_sha].attempts == 0
    # Legacy backfill is not a newly detected death.
    assert notifications == []


def test_cmd_claim_retries_stale_lease_and_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notifications: list[str]
) -> None:
    remote, storage, shas, args, output_file = _claim_scenario(tmp_path, monkeypatch)
    head_sha, second_sha = shas
    (storage / "data" / "shas.txt").write_text(f"{second_sha}\n{head_sha} {OLD_TS} 1\n")
    _add_results(storage, second_sha)
    cmd_claim(args)

    out = output_file.read_text()
    assert f"sha={head_sha}" in out
    assert "new_commit=yes" in out

    retried = {claim.sha: claim for claim in _pushed_claims(tmp_path, remote)}
    assert retried[head_sha].attempts == 2
    (event,) = notifications
    assert head_sha in event
    assert f"attempt 2/{MAX_ATTEMPTS}" in event


def test_cmd_claim_skips_in_flight_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notifications: list[str]
) -> None:
    _, storage, shas, args, output_file = _claim_scenario(tmp_path, monkeypatch)
    head_sha, second_sha = shas
    now = datetime.now(timezone.utc).isoformat()
    (storage / "data" / "shas.txt").write_text(f"{second_sha}\n{head_sha} {now} 1\n")
    _add_results(storage, second_sha)
    cmd_claim(args)

    out = output_file.read_text()
    assert "sha=NONE" in out
    assert "new_commit=no" in out
    assert notifications == []


def test_cmd_claim_abandons_exhausted_lease_and_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notifications: list[str]
) -> None:
    remote, storage, shas, args, output_file = _claim_scenario(tmp_path, monkeypatch)
    head_sha, second_sha = shas
    (storage / "data" / "shas.txt").write_text(
        f"{second_sha}\n{head_sha} {OLD_TS} {MAX_ATTEMPTS}\n"
    )
    _add_results(storage, second_sha)
    cmd_claim(args)

    out = output_file.read_text()
    assert "sha=NONE" in out
    assert "new_commit=no" in out

    abandoned = {claim.sha: claim for claim in _pushed_claims(tmp_path, remote)}
    assert abandoned[head_sha].abandoned
    (event,) = notifications
    assert head_sha in event
    assert "Giving up" in event


def test_cmd_claim_skips_abandoned_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notifications: list[str]
) -> None:
    _, storage, shas, args, output_file = _claim_scenario(tmp_path, monkeypatch)
    head_sha, second_sha = shas
    (storage / "data" / "shas.txt").write_text(
        f"{second_sha}\n{head_sha} {OLD_TS} {MAX_ATTEMPTS} abandoned\n"
    )
    _add_results(storage, second_sha)
    cmd_claim(args)

    out = output_file.read_text()
    assert "sha=NONE" in out
    assert "new_commit=no" in out
    assert notifications == []


def test_cmd_claim_prefers_fresh_commit_over_stale_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notifications: list[str]
) -> None:
    remote, storage, shas, args, output_file = _claim_scenario(tmp_path, monkeypatch)
    head_sha, second_sha = shas
    (storage / "data" / "shas.txt").write_text(f"{second_sha} {OLD_TS} 1\n")
    cmd_claim(args)

    assert f"sha={head_sha}" in output_file.read_text()
    stale = {claim.sha: claim for claim in _pushed_claims(tmp_path, remote)}
    # The stale lease is left for an idle slot.
    assert stale[second_sha].attempts == 1
    assert notifications == []


def test_cmd_claim_survives_notification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, storage, shas, args, output_file = _claim_scenario(tmp_path, monkeypatch)
    head_sha, second_sha = shas
    (storage / "data" / "shas.txt").write_text(f"{second_sha}\n{head_sha} {OLD_TS} 1\n")
    _add_results(storage, second_sha)

    def boom(events: list[str]) -> None:
        raise RuntimeError("gh unavailable")

    monkeypatch.setattr(claim_mod, "notify_failures", boom)
    cmd_claim(args)

    out = output_file.read_text()
    assert f"sha={head_sha}" in out
    assert "new_commit=yes" in out


# === failure notifications ===


def _fake_execute(
    calls: list[tuple[str, str | None]], list_result: str
) -> Callable[..., str]:
    def _execute(cmd: str, *, input: str | None = None) -> str:
        calls.append((cmd, input))
        if cmd.startswith("gh issue list"):
            return list_result
        return "https://github.com/pandas-dev/asv-runner/issues/99\n"

    return _execute


def test_find_failure_issue_matches_title(monkeypatch: pytest.MonkeyPatch) -> None:
    listed = (
        '[{"number": 7, "title": "other"},'
        ' {"number": 42, "title": "Benchmark run failures"}]'
    )
    monkeypatch.setattr(claim_mod, "execute", _fake_execute([], listed))
    assert find_failure_issue() == "42"


def test_find_failure_issue_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        claim_mod, "execute", _fake_execute([], '[{"number": 7, "title": "other"}]')
    )
    assert find_failure_issue() is None


def test_notify_failures_creates_issue_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(claim_mod, "execute", _fake_execute(calls, "[]"))
    notify_failures(["event one", "event two"])

    cmd, body = calls[-1]
    assert cmd.startswith("gh issue create")
    assert '--title "Benchmark run failures"' in cmd
    assert body == "- event one\n- event two"


def test_notify_failures_comments_on_existing_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    listed = '[{"number": 42, "title": "Benchmark run failures"}]'
    monkeypatch.setattr(claim_mod, "execute", _fake_execute(calls, listed))
    notify_failures(["event one"])

    cmd, body = calls[-1]
    assert cmd.startswith("gh issue comment 42")
    assert body == "- event one"
