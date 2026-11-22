from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from asv_runner.util import (
    DEFAULT_REPO,
    gh,
    github_repo,
    orphan_push_with_retry,
    time_to_str,
    write_github_output,
)
from tests._helpers import init_remote_and_storage

# === time_to_str ===


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.5, "1.500s"),
        (1.0, "1.000s"),
        (0.5, "500.000ms"),
        (0.001, "1.000ms"),
        (0.0005, "500.000us"),
        (0.000001, "1.000us"),
        (0.0000005, "500.000ns"),
    ],
)
def test_time_to_str_positive(value: float, expected: str) -> None:
    assert time_to_str(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (-1.5, "-1.500s"),
        (-0.5, "-500.000ms"),
        (-0.0005, "-500.000us"),
        (-0.0000005, "-500.000ns"),
    ],
)
def test_time_to_str_negative(value: float, expected: str) -> None:
    assert time_to_str(value) == expected


# === gh / github_repo ===


@pytest.fixture
def fake_gh_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``gh`` on PATH that echoes its arguments and stdin, failing on ``fail``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "gh"
    script.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = fail ]; then echo "bad request" >&2; exit 1; fi\n'
        'echo "args: $*"\n'
        "cat\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")


@pytest.mark.usefixtures("fake_gh_cli")
def test_gh_passes_args_and_stdin() -> None:
    # Arguments are passed as a list, so spaces and quotes survive unchanged.
    out = gh(["issue", "create", "--title", 'a "b" c'], input="body\n")
    assert out == 'args: issue create --title a "b" c\nbody\n'


@pytest.mark.usefixtures("fake_gh_cli")
def test_gh_raises_with_stderr() -> None:
    with pytest.raises(RuntimeError, match="bad request"):
        gh(["fail"])


def test_github_repo_from_actions_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/fork")
    assert github_repo() == "someone/fork"


def test_github_repo_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert github_repo() == DEFAULT_REPO


# === write_github_output ===


def test_write_github_output_prints_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    write_github_output(sha="abc", new_commit="yes")
    out = capsys.readouterr().out
    assert "sha=abc" in out
    assert "new_commit=yes" in out


# === orphan_push_with_retry ===


def test_orphan_push_with_retry_roundtrip(tmp_path: Path) -> None:
    remote, storage = init_remote_and_storage(tmp_path)

    def modify(repo: Path) -> bool:
        (repo / "data" / "shas.txt").write_text("hello\n")
        return True

    pushed = orphan_push_with_retry(
        storage, branch="pandas_test", message="msg", modify_tree=modify
    )
    assert pushed

    verify = tmp_path / "verify"
    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(verify)],
        check=True,
    )
    assert (verify / "data" / "shas.txt").read_text() == "hello\n"


def test_orphan_push_with_retry_skip_when_modify_returns_false(
    tmp_path: Path,
) -> None:
    _, storage = init_remote_and_storage(tmp_path)

    def modify(repo: Path) -> bool:
        return False

    pushed = orphan_push_with_retry(
        storage, branch="pandas_test", message="msg", modify_tree=modify
    )
    assert pushed is False


def test_orphan_push_with_retry_recovers_from_lost_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, storage = init_remote_and_storage(tmp_path)

    # Second clone acts as a competing writer.
    competitor = tmp_path / "competitor"
    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(competitor)],
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "other"], cwd=competitor, check=True)
    subprocess.run(
        ["git", "config", "user.email", "other@other"], cwd=competitor, check=True
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    calls: list[int] = []

    def modify(repo: Path) -> bool:
        calls.append(1)
        if len(calls) == 1:
            # Race in a competing push between this fetch and our upcoming push.
            (competitor / "data" / "shas.txt").write_text("competing\n")
            subprocess.run(["git", "add", "-A"], cwd=competitor, check=True)
            subprocess.run(
                ["git", "commit", "-m", "competing"], cwd=competitor, check=True
            )
            subprocess.run(
                ["git", "push", str(remote), "pandas_test"],
                cwd=competitor,
                check=True,
            )
        (repo / "data" / "shas.txt").write_text("ours\n")
        return True

    pushed = orphan_push_with_retry(
        storage,
        branch="pandas_test",
        message="msg",
        modify_tree=modify,
        attempts=3,
    )
    assert pushed
    assert len(calls) >= 2

    verify = tmp_path / "verify"
    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(verify)],
        check=True,
    )
    assert (verify / "data" / "shas.txt").read_text() == "ours\n"
