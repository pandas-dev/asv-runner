from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from asv_runner import artifacts
from asv_runner.store import Store
from tests._results_helpers import add_commit, make_result

START = 1772845825000
DAY_MS = 86_400_000


def _zip_for(sha: str, benchmarks_version: str) -> bytes:
    """An artifact as stage_results lays it out."""
    import zstandard

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        result = json.dumps(make_result(sha, START)).encode()
        zf.writestr(
            f"data/results/asvrunner/{sha}.json.zst",
            zstandard.ZstdCompressor().compress(result),
        )
        zf.writestr(
            "data/results/benchmarks.json",
            json.dumps({"a.b": {"version": benchmarks_version}, "version": 2}),
        )
    return buf.getvalue()


@pytest.fixture
def fake_gh(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Artifacts listed and served from memory instead of the GitHub API."""
    state: dict = {"listing": [], "zips": {}, "downloaded": []}

    def gh(args: list[str]) -> bytes:
        path = args[2] if args[1] == "--paginate" else args[1]
        if path.endswith("actions/artifacts?per_page=100"):
            return "".join(json.dumps(a) + "\n" for a in state["listing"]).encode()
        artifact_id = int(path.split("/")[-2])
        state["downloaded"].append(artifact_id)
        return state["zips"][artifact_id]

    monkeypatch.setattr(artifacts, "_gh", gh)
    return state


def _artifact(artifact_id: int, name: str, created: str, expired: bool = False) -> dict:
    return {"id": artifact_id, "name": name, "expired": expired, "created_at": created}


def test_fetch_extracts_unstored_results(tmp_path: Path, fake_gh: dict) -> None:
    input_dir = tmp_path / "input"
    stored_sha = "a" * 40
    add_commit(input_dir, stored_sha, START)
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)

    new_old, new_new = "b" * 40, "c" * 40
    fake_gh["listing"] = [
        _artifact(3, f"result-{new_new}", "2026-11-16T03:00:00Z"),
        _artifact(1, f"result-{stored_sha}", "2026-11-14T03:00:00Z"),
        _artifact(2, f"result-{new_old}", "2026-11-15T03:00:00Z"),
        _artifact(4, f"result-{'d' * 40}", "2026-08-01T03:00:00Z", expired=True),
        _artifact(5, "github-pages", "2026-11-16T03:00:00Z"),
    ]
    fake_gh["zips"] = {2: _zip_for(new_old, "v-old"), 3: _zip_for(new_new, "v-new")}

    out = tmp_path / "fetched"
    assert artifacts.fetch(store, out, "owner/repo") == [new_old, new_new]
    # Stored, expired, and non-result artifacts are not downloaded.
    assert fake_gh["downloaded"] == [2, 3]
    results = out / "data" / "results"
    assert sorted(p.name for p in (results / "asvrunner").iterdir()) == [
        f"{new_old}.json.zst",
        f"{new_new}.json.zst",
    ]
    # Oldest first, so the newest run's benchmarks.json wins.
    assert json.loads((results / "benchmarks.json").read_text())["a.b"] == {
        "version": "v-new"
    }

    report = store.update(out, workers=1)
    assert sorted(report.results_added) == [new_old, new_new]


def test_lost_results(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    stored = "a" * 40
    add_commit(input_dir, stored, START)
    store = Store(tmp_path / "store")
    store.update(input_dir, workers=1)

    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    old = (now - timedelta(days=3)).isoformat()
    recent = (now - timedelta(hours=5)).isoformat()
    shas = tmp_path / "shas.txt"
    shas.write_text(
        f"{stored} {old} 1 done\n"  # stored: fine
        f"{'b' * 40} {old} 1 done\n"  # done, old, not stored: lost
        f"{'c' * 40} {recent} 1 done\n"  # done recently: tonight's run will get it
        f"{'d' * 40} {old} 3 abandoned\n"  # never produced results
        f"{'e' * 40} {old} 1\n"  # not done: claim retries it
    )
    assert artifacts.lost_results(store, shas, now=now) == ["b" * 40]
