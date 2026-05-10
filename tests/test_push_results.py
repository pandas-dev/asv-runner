from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from asv_runner import push_results


def _make_asv_dir(tmp_path: Path) -> Path:
    asv = tmp_path / "asv"
    asv_results = asv / "results" / "asvrunner"
    asv_results.mkdir(parents=True)
    (asv / "results" / "benchmarks.json").write_text(json.dumps({"version": "1.0"}))
    (asv_results / "machine.json").write_text(json.dumps({"machine": "asvrunner"}))
    return asv


def _args(tmp_path: Path, asv: Path, sha: str) -> argparse.Namespace:
    return argparse.Namespace(
        storage_dir=str(tmp_path / "storage"),
        asv_dir=str(asv),
        sha=sha,
        branch="pandas_test",
    )


def test_run_raises_when_no_matching_result(tmp_path: Path) -> None:
    asv = _make_asv_dir(tmp_path)
    with pytest.raises(RuntimeError, match="No matching result file"):
        push_results.run(_args(tmp_path, asv, "abcd1234deadbeef"))


def test_run_raises_when_multiple_matches(tmp_path: Path) -> None:
    asv = _make_asv_dir(tmp_path)
    asv_results = asv / "results" / "asvrunner"
    (asv_results / "abcd1234-existing-foo.json").write_text("{}")
    (asv_results / "abcd1234-existing-bar.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Multiple matches"):
        push_results.run(_args(tmp_path, asv, "abcd1234deadbeef"))


def test_run_compresses_and_stages_for_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asv = _make_asv_dir(tmp_path)
    asv_results = asv / "results" / "asvrunner"
    sha = "abcd1234deadbeef"
    (asv_results / "abcd1234-existing-uuid.json").write_text(
        json.dumps({"commit_hash": sha})
    )

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if cmd[0] == "pixi":
            kwargs["stdout"].write("pkg 1.0\n")
        elif cmd[0] == "zstd":
            target = Path(cmd[-1])
            target.with_name(target.name + ".zst").write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    captured: dict[str, Any] = {}

    def fake_push(
        repo: Path,
        *,
        branch: str,
        message: str,
        modify_tree: Any,
        attempts: int = 5,
    ) -> bool:
        storage_repo = tmp_path / "fake_storage"
        storage_repo.mkdir()
        captured["repo"] = repo
        captured["branch"] = branch
        captured["message"] = message
        captured["modify_result"] = modify_tree(storage_repo)
        captured["storage_repo"] = storage_repo
        return True

    monkeypatch.setattr(push_results, "orphan_push_with_retry", fake_push)

    push_results.run(_args(tmp_path, asv, sha))

    assert captured["branch"] == "pandas_test"
    assert captured["message"] == "Results"
    assert captured["modify_result"] is True

    storage_repo = captured["storage_repo"]
    data = storage_repo / "data"
    assert (data / "results" / "benchmarks.json").exists()
    assert (data / "results" / "asvrunner" / "machine.json").exists()
    assert (data / "results" / "asvrunner" / f"{sha}.json.zst").exists()
    assert (data / "envs" / f"{sha}.yml.zst").exists()
