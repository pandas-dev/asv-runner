from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from asv_runner.claim import LOOKBACK_COMMITS, pick_next_sha, read_existing_shas
from asv_runner.claim import run as cmd_claim
from tests._helpers import init_remote_and_storage

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


def test_read_existing_shas_missing_file(tmp_path: Path) -> None:
    assert read_existing_shas(tmp_path / "missing.txt") == set()


def test_read_existing_shas_ignores_blank_lines(tmp_path: Path) -> None:
    shas_path = tmp_path / "shas.txt"
    shas_path.write_text("sha1111\n\nsha2222\n")
    # Blank line collapses to empty string in the set; pick_next_sha filters
    # those out via the non-empty guard.
    assert "sha1111" in read_existing_shas(shas_path)
    assert "sha2222" in read_existing_shas(shas_path)


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


def test_cmd_claim_picks_sha_and_pushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, storage = init_remote_and_storage(tmp_path)
    pandas_repo, shas = _init_pandas_repo(tmp_path, n_commits=2)
    output_file = tmp_path / "github_output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    args = argparse.Namespace(
        storage_dir=str(storage),
        repo_dir=str(pandas_repo),
        branch="pandas_test",
    )
    cmd_claim(args)

    out = output_file.read_text()
    head_sha = shas[0]
    assert f"sha={head_sha}" in out
    assert "new_commit=yes" in out

    verify = tmp_path / "verify"
    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(verify)],
        check=True,
    )
    assert (verify / "data" / "shas.txt").read_text().strip() == head_sha


def test_cmd_claim_appends_next_unseen_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, storage = init_remote_and_storage(tmp_path)
    pandas_repo, shas = _init_pandas_repo(tmp_path, n_commits=3)
    head_sha, second_sha, third_sha = shas
    # Pre-seed shas.txt with everything except the most recent commit.
    (storage / "data" / "shas.txt").write_text(f"{third_sha}\n{second_sha}\n")
    output_file = tmp_path / "github_output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    args = argparse.Namespace(
        storage_dir=str(storage),
        repo_dir=str(pandas_repo),
        branch="pandas_test",
    )
    cmd_claim(args)

    assert f"sha={head_sha}" in output_file.read_text()

    verify = tmp_path / "verify"
    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(verify)],
        check=True,
    )
    lines = (verify / "data" / "shas.txt").read_text().splitlines()
    assert lines == [third_sha, second_sha, head_sha]


def test_cmd_claim_no_new_commit_when_all_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, storage = init_remote_and_storage(tmp_path)
    pandas_repo, shas = _init_pandas_repo(tmp_path, n_commits=2)
    (storage / "data" / "shas.txt").write_text("\n".join(shas) + "\n")
    output_file = tmp_path / "github_output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    args = argparse.Namespace(
        storage_dir=str(storage),
        repo_dir=str(pandas_repo),
        branch="pandas_test",
    )
    cmd_claim(args)

    out = output_file.read_text()
    assert "sha=NONE" in out
    assert "new_commit=no" in out
