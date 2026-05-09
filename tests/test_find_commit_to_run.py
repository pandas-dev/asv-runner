from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ci.find_commit_to_run import run


class _FakeCompleted:
    def __init__(self, stdout: bytes) -> None:
        self.stdout = stdout


def _fake_subprocess_run(shas: list[str]) -> Callable[..., _FakeCompleted]:
    out = "\n".join(f"{sha} commit message {sha}" for sha in shas).encode()

    def _run(cmd: Any, **kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(out)

    return _run


def test_run_no_existing_shas_picks_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(["sha1111", "sha2222"]))

    run(input_path=tmp_path, repo_path=repo)

    assert capsys.readouterr().out.strip() == "sha1111"


def test_run_skips_existing_shas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "shas.txt").write_text("sha1111\nsha2222\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_subprocess_run(["sha1111", "sha2222", "sha3333"]),
    )

    run(input_path=tmp_path, repo_path=repo)

    assert capsys.readouterr().out.strip() == "sha3333"


def test_run_all_existing_prints_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "shas.txt").write_text("sha1111\nsha2222\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(["sha1111", "sha2222"]))

    run(input_path=tmp_path, repo_path=repo)

    assert capsys.readouterr().out.strip() == "NONE"


def test_run_accepts_string_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(["abcdef0"]))

    run(input_path=str(tmp_path), repo_path=str(repo))

    assert capsys.readouterr().out.strip() == "abcdef0"


def test_run_ignores_blank_lines_in_shas_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "shas.txt").write_text("sha1111\n\nsha2222\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_subprocess_run(["sha1111", "sha2222", "sha3333"]),
    )

    run(input_path=tmp_path, repo_path=repo)

    assert capsys.readouterr().out.strip() == "sha3333"
