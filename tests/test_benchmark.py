from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import pytest

from asv_runner import benchmark


def test_run_invokes_asv_machine_then_asv_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = argparse.Namespace(asv_dir=str(tmp_path), sha="abc123")
    benchmark.run(args)

    assert len(calls) == 2

    machine_cmd, machine_kwargs = calls[0]
    assert machine_cmd == [
        "python",
        "-m",
        "asv",
        "machine",
        "--machine=asvrunner",
        "--yes",
    ]
    assert machine_kwargs["cwd"] == tmp_path
    assert machine_kwargs["check"] is True

    run_cmd, run_kwargs = calls[1]
    assert run_cmd == [
        "python",
        "-m",
        "asv",
        "run",
        "--machine=asvrunner",
        "--python=same",
        "--set-commit-hash=abc123",
        "--show-stderr",
    ]
    assert run_kwargs["cwd"] == tmp_path
    assert run_kwargs["check"] is True
