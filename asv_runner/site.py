"""Build the static benchmark site from the Parquet store.

Output layout::

    index.html, app.js, style.css,    copied from site_static/
    uPlot.*
    data/index.json                   commits, benchmark list, env changes,
                                      break points, run status, detections
    data/graphs/<benchmark>.json      one file per benchmark
    data/graphs/<benchmark>.chunk<k>.json
                                      values of series k*64 .. k*64+63, for
                                      benchmarks with more than 64 series

Files refer to commits by their position in ``index.json``'s commit list, which
keeps them small. Results are rounded to ``SIG_DIGITS`` significant digits for
display; the store keeps full precision.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from asv_runner import breaks as breaks_mod
from asv_runner import regressions, series
from asv_runner.store import Store

SIG_DIGITS = 5
# Benchmarks with more series than this split their values into chunk files so
# the page only downloads the series it shows.
SERIES_PER_FILE = 64
# Benchmarks loaded together while writing graph files; bounds memory.
BATCH = 100
STATIC_DIR = Path(__file__).parent / "site_static"
# Shown first in env-change details; everything else is behind "show all".
KEY_PACKAGES = [
    # The benchmark tooling itself: a change can shift every timing.
    "asv",
    "asv_runner",
    "python",
    "numpy",
    "cython",
    "meson",
    "meson-python",
    "pyarrow",
    "gcc",
    "gxx",
    "clang",
    "libblas",
    "libopenblas",
    "mkl",
]
# Rebuilt for every commit, so its version always changes.
IGNORED_PACKAGES = {"pandas"}
# Detection kinds as stored in the site data.
KIND_CODES = {regressions.REGRESSION: 1, regressions.IMPROVEMENT: -1}


def _definitions(store: Store) -> dict[str, dict]:
    """Latest known definition fields per benchmark from benchmark_defs."""
    if not store.benchmark_defs_path.exists():
        return {}
    out = {}
    for name, definition in (
        pl.read_parquet(store.benchmark_defs_path)
        .select("name", "definition")
        .iter_rows()
    ):
        d = json.loads(definition)
        out[name] = {
            "param_names": d.get("param_names", []),
            "unit": d.get("unit"),
            "type": d.get("type"),
        }
    return out


def env_changes(store: Store, commits: pl.DataFrame) -> list[dict]:
    """Package changes between consecutive benchmarked commits."""
    if not store.envs_path.exists():
        return []
    envs = (
        pl.read_parquet(store.envs_path)
        .filter(~pl.col("package").is_in(list(IGNORED_PACKAGES)))
        .join(commits.select("sha", "i"), on="sha")
    )
    by_commit = {i: g for (i,), g in envs.partition_by("i", as_dict=True).items()}
    key_rank = {p: n for n, p in enumerate(KEY_PACKAGES)}
    out = []
    prev_pkgs: dict[str, str] | None = None
    prev_format = None
    for i in sorted(by_commit):
        g = by_commit[i]
        pkgs = dict(zip(g["package"].to_list(), g["version"].to_list()))
        fmt = g["format"][0]
        if prev_pkgs is not None:
            if fmt != prev_format:
                out.append({"i": i, "format_change": [prev_format, fmt], "changes": []})
            elif pkgs != prev_pkgs:
                names = sorted(
                    (
                        p
                        for p in prev_pkgs.keys() | pkgs.keys()
                        if prev_pkgs.get(p) != pkgs.get(p)
                    ),
                    key=lambda p: (key_rank.get(p, len(key_rank)), p),
                )
                changes = [[p, prev_pkgs.get(p), pkgs.get(p)] for p in names]
                out.append(
                    {
                        "i": i,
                        "changes": changes,
                        "n_key": sum(c[0] in key_rank for c in changes),
                    }
                )
        prev_pkgs, prev_format = pkgs, fmt
    return out


def _failing_since(commits: list[int], values: list[list]) -> int | None:
    """First commit of the run of commits, ending at the latest, where every
    series failed (no result). None if the latest commit has a result."""
    since = None
    for pos in range(len(commits) - 1, -1, -1):
        if any(row[pos] is not None for row in values):
            break
        since = commits[pos]
    return since


def _round_step(step: float | None) -> float | None:
    return None if step is None or not math.isfinite(step) else round(step, 4)


def build(
    store: Store, out_dir: Path, breaks: list[breaks_mod.Break] | None = None
) -> dict:
    """Write the site to ``out_dir``. Returns size statistics.

    ``breaks`` defaults to the list in ``breaks.toml``.
    """
    if breaks is None:
        breaks = breaks_mod.load()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    graphs_dir = out_dir / "data" / "graphs"
    graphs_dir.mkdir(parents=True)
    for p in STATIC_DIR.iterdir():
        shutil.copy(p, out_dir / p.name)

    s = series.load(store)
    commits = s.commits
    last_commit = commits.height - 1
    version_changes = dict(
        s.runs.sort("name", "i")
        .filter(pl.col("version") != pl.col("version").shift(1).over("name"))
        .group_by("name")
        .agg(pl.col("i"))
        .iter_rows()
    )
    detections = regressions.detect(s, breaks=breaks)
    detections_by_name = {
        name: g for (name,), g in detections.partition_by("name", as_dict=True).items()
    }
    defs = _definitions(store)

    index_benchmarks = {}
    failing = []
    sizes = []
    names = sorted(s.params)
    for start in range(0, len(names), BATCH):
        batch = (
            s.values(names[start : start + BATCH])
            .with_columns(pl.col("value").round_sig_figs(SIG_DIGITS))
            .group_by("name", "series")
            .agg(pl.col("i", "value").sort_by("i"))
        )
        for (name,), g in batch.partition_by("name", as_dict=True).items():
            graph, info, size = _write_graph(
                graphs_dir,
                name,
                g,
                s.params[name],
                defs.get(name, {}),
                version_changes.get(name, []),
                detections_by_name.get(name),
            )
            sizes.extend(size)
            index_benchmarks[name] = info
            if graph["commits"][-1] == last_commit:
                since = _failing_since(graph["commits"], graph["_values"])
                if since is not None:
                    failing.append({"name": name, "since": since})

    last_run = s.runs.filter(pl.col("i") == last_commit)["started_at"].max()
    info = commits.join(store.commit_info(), on="sha", how="left").sort("i")
    index = {
        "status": {
            "last_commit": last_commit,
            "last_run": last_run,
            "failing": sorted(failing, key=lambda f: (f["since"], f["name"])),
        },
        "commits": {
            "sha": commits["sha"].to_list(),
            "date": [int(d.timestamp() * 1000) for d in commits["date"].to_list()],
            # null where the title is not stored (no pandas clone given yet)
            "title": info["title"].to_list(),
            "pr": info["pr"].to_list(),
        },
        "benchmarks": dict(sorted(index_benchmarks.items())),
        "env_changes": env_changes(store, commits),
        "breaks": [
            {"i": i, "reason": reason}
            for i, reason in breaks_mod.positions(commits, breaks)
        ],
        # Columnar: one entry per detection, ordered by commit.
        "detections": {
            "i": detections["i"].to_list(),
            "name": detections["name"].to_list(),
            "params": detections["params"].to_list(),
            "kind": [KIND_CODES[k] for k in detections["kind"].to_list()],
            "step": [_round_step(v) for v in detections["step"].to_list()],
        },
    }
    index_bytes = _write_json(out_dir / "data" / "index.json", index)
    return {
        "benchmarks": len(index_benchmarks),
        "detections": detections.height,
        "graphs_bytes": sum(sizes),
        "largest_graph_bytes": max(sizes, default=0),
        "index_bytes": index_bytes,
    }


def _write_graph(
    graphs_dir: Path,
    name: str,
    g: pl.DataFrame,
    series_params: list[tuple],
    definition: dict,
    version_changes: list[int],
    detections: pl.DataFrame | None,
) -> tuple[dict, dict, list[int]]:
    rows = g.select("series", "i", "value").rows()
    bench_commits = sorted({i for _, idx, _ in rows for i in idx})
    pos = {c: n for n, c in enumerate(bench_commits)}
    dense = {}
    for sid, idx, vals in rows:
        row = [None] * len(bench_commits)
        for i, v in zip(idx, vals):
            row[pos[i]] = v
        dense[sid] = row
    params = [list(p) for p in series_params]
    values = [dense.get(sid, [None] * len(bench_commits)) for sid in range(len(params))]
    n_params = len(params[0]) if params else 0
    param_names = definition.get("param_names") or []
    if len(param_names) != n_params:
        param_names = [f"param{n + 1}" for n in range(n_params)]
    param_values: list[list[str]] = [[] for _ in range(n_params)]
    for key in params:
        for n, v in enumerate(key):
            if v not in param_values[n]:
                param_values[n].append(v)

    n_chunks = math.ceil(len(params) / SERIES_PER_FILE)
    graph: dict[str, Any] = {
        "name": name,
        "param_names": param_names,
        "commits": bench_commits,
        "series": params,
        "version_changes": version_changes,
        # [commit position, series id, kind (1 regression, -1 improvement), step]
        "detections": []
        if detections is None
        else [
            [i, sid, KIND_CODES[kind], _round_step(step)]
            for i, sid, kind, step in detections.select(
                "i", "series", "kind", "step"
            ).iter_rows()
        ],
        "chunk_size": SERIES_PER_FILE,
        "chunks": n_chunks,
    }
    sizes = []
    if len(params) <= SERIES_PER_FILE:
        graph["values"] = values
    else:
        for k in range(n_chunks):
            chunk = values[k * SERIES_PER_FILE : (k + 1) * SERIES_PER_FILE]
            sizes.append(_write_json(graphs_dir / f"{name}.chunk{k}.json", chunk))
    sizes.append(_write_json(graphs_dir / f"{name}.json", graph))
    info = {
        "param_names": param_names,
        "param_values": param_values,
        "unit": definition.get("unit"),
        "type": definition.get("type"),
    }
    # Kept out of the written file; used for the failing check.
    graph["_values"] = values
    return graph, info, sizes


def _write_json(path: Path, data: object) -> int:
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    path.write_text(text)
    return len(text)
