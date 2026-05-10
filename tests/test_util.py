from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from asv_runner.util import (
    escape_ansi,
    execute,
    orphan_push_with_retry,
    time_to_str,
    write_github_output,
)
from tests._helpers import init_remote_and_storage

# === time_to_str / escape_ansi ===


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


def test_escape_ansi_passes_plain_text() -> None:
    assert escape_ansi("hello world") == "hello world"


def test_escape_ansi_strips_color_codes() -> None:
    colored = "\x1b[31mhello\x1b[0m world"
    assert escape_ansi(colored) == "hello world"


# === execute ===


def test_execute_returns_stdout() -> None:
    assert execute("echo hello").strip() == "hello"


def test_execute_passes_stdin_through() -> None:
    assert execute("cat", input="ping\n").strip() == "ping"


def test_execute_raises_with_stdout_and_stderr_on_nonzero_exit() -> None:
    with pytest.raises(ValueError) as exc_info:
        execute("echo out; echo err >&2; exit 1")
    msg = str(exc_info.value)
    assert "out" in msg
    assert "err" in msg


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
