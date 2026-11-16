from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from asv_runner import stage_results


def _make_asv_dir(tmp_path: Path) -> Path:
    asv = tmp_path / "asv"
    asv_results = asv / "results" / "asvrunner"
    asv_results.mkdir(parents=True)
    (asv / "results" / "benchmarks.json").write_text(json.dumps({"version": "1.0"}))
    (asv_results / "machine.json").write_text(json.dumps({"machine": "asvrunner"}))
    return asv


def _args(tmp_path: Path, asv: Path, sha: str) -> argparse.Namespace:
    return argparse.Namespace(
        asv_dir=str(asv),
        sha=sha,
        out_dir=str(tmp_path / "staged"),
    )


def test_run_raises_when_no_matching_result(tmp_path: Path) -> None:
    asv = _make_asv_dir(tmp_path)
    with pytest.raises(RuntimeError, match="No matching result file"):
        stage_results.run(_args(tmp_path, asv, "abcd1234deadbeef"))


def test_run_raises_when_multiple_matches(tmp_path: Path) -> None:
    asv = _make_asv_dir(tmp_path)
    asv_results = asv / "results" / "asvrunner"
    (asv_results / "abcd1234-existing-foo.json").write_text("{}")
    (asv_results / "abcd1234-existing-bar.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Multiple matches"):
        stage_results.run(_args(tmp_path, asv, "abcd1234deadbeef"))


def test_run_compresses_and_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asv = _make_asv_dir(tmp_path)
    asv_results = asv / "results" / "asvrunner"
    sha = "abcd1234deadbeef"
    (asv_results / "abcd1234-existing-uuid.json").write_text(
        json.dumps({"commit_hash": sha})
    )
    real_run = subprocess.run

    def fake_pixi(cmd: list[str], **kwargs: Any) -> Any:
        if cmd[0] == "pixi":
            kwargs["stdout"].write("Name  Version\npkg   1.0\n")
            return subprocess.CompletedProcess(cmd, 0)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_pixi)

    stage_results.run(_args(tmp_path, asv, sha))

    data = tmp_path / "staged" / "data"
    assert (data / "results" / "benchmarks.json").exists()
    staged = sorted(p.relative_to(data).as_posix() for p in data.rglob("*.zst"))
    assert staged == [f"envs/{sha}.yml.zst", f"results/asvrunner/{sha}.json.zst"]
    # zstd --rm removes the uncompressed copies.
    assert not (data / "results" / "asvrunner" / f"{sha}.json").exists()
    assert not (data / "envs" / f"{sha}.yml").exists()
