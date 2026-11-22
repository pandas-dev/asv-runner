"""Parquet store for asv results, env files, and benchmark definitions.

Layout under the store root::

    runs.parquet                         one row per result file
    benchmarks/week=<YYYY-Www>/data.parquet
    values/week=<YYYY-Www>/data.parquet
    env_files.parquet                    raw env file text, one row per commit
    envs.parquet                         parsed env files, one row per package
    benchmark_defs.parquet               every distinct benchmarks.json entry seen
    commit_info.parquet                  pandas commit title and PR number

``benchmarks`` and ``values`` are partitioned by ISO week of the commit date so
that every file stays under GitHub's file size limits while the store is
committed to a git branch, an update rewrites only the weeks its commits fall
in, and readers that need recent history read only recent weeks. Within a
partition, rows are sorted by benchmark, parameter combination, and commit date,
which is what makes the noisy float columns compress well.

Updates write the week partitions first and ``runs.parquet`` last. A commit is
stored once it is in ``runs.parquet``; merging a week drops any existing rows for
the commits being added, so an interrupted update can be rerun.
"""

from __future__ import annotations

import datetime as dt
import json
import multiprocessing
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from asv_runner import envs, results

# GitHub rejects files over 100 MB and warns over 50 MB.
MAX_FILE_BYTES = 50_000_000
RESULT_SUFFIX = ".json.zst"
ENV_SUFFIX = ".yml.zst"

BENCHMARK_DEFS_SCHEMA = pa.schema(
    [("name", pa.string()), ("version", pa.string()), ("definition", pa.string())]
)


COMMIT_INFO_SCHEMA = pa.schema(
    [("sha", pa.string()), ("title", pa.string()), ("pr", pa.int32())]
)
# Squash-merged pandas commits end their title with the PR number.
PR_SUFFIX = re.compile(r"\(#(\d+)\)\s*$")


class FileTooLargeError(RuntimeError):
    pass


def week_key(date: dt.datetime) -> str:
    year, week, _ = date.isocalendar()
    return f"{year}-W{week:02d}"


def read_zst(path: Path) -> bytes:
    with open(path, "rb") as fh:
        return zstandard.ZstdDecompressor().stream_reader(fh).read()


def sha_from_path(path: Path, suffix: str) -> str:
    assert path.name.endswith(suffix), path
    return path.name[: -len(suffix)]


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically, refusing files over MAX_FILE_BYTES."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=9)
    size = tmp.stat().st_size
    if size > MAX_FILE_BYTES:
        tmp.unlink()
        raise FileTooLargeError(
            f"{path} would be {size / 1e6:.1f} MB; the limit is "
            f"{MAX_FILE_BYTES / 1e6:.0f} MB. Partition more finely."
        )
    os.replace(tmp, path)


def read_commit_messages(repo: Path, shas: list[str]) -> dict[str, str]:
    """Commit messages by SHA, read with ``git cat-file --batch``.

    SHAs that are not commits in ``repo`` are left out.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input="".join(f"{sha}\n" for sha in shas).encode(),
        capture_output=True,
        check=True,
    )
    out, pos, data = {}, 0, proc.stdout
    while pos < len(data):
        end = data.index(b"\n", pos)
        header = data[pos:end].decode().split()
        pos = end + 1
        if len(header) != 3:  # "<sha> missing"
            continue
        sha, kind, size = header[0], header[1], int(header[2])
        body = data[pos : pos + size]
        pos += size + 1  # content is followed by a newline
        if kind == "commit":
            # Headers, a blank line, then the message.
            out[sha] = body.split(b"\n\n", 1)[1].decode(errors="replace")
    return out


def _parse_result_file(path: Path) -> tuple[str, pa.Table, pa.Table, dict]:
    d = json.loads(read_zst(path))
    sha = sha_from_path(path, RESULT_SUFFIX)
    if d["commit_hash"] != sha:
        raise ValueError(f"{path} holds results for {d['commit_hash']}")
    run, bench, values = results.parse(d)
    return (
        week_key(run["date"]),
        pa.table(bench, schema=results.BENCHMARKS_SCHEMA),
        pa.table(values, schema=results.VALUES_SCHEMA),
        run,
    )


@dataclass
class UpdateReport:
    results_added: list[str] = field(default_factory=list)
    weeks_written: list[str] = field(default_factory=list)
    envs_added: list[str] = field(default_factory=list)
    benchmark_defs_added: int = 0
    commit_info_added: int = 0


class Store:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------

    @property
    def runs_path(self) -> Path:
        return self.root / "runs.parquet"

    @property
    def env_files_path(self) -> Path:
        return self.root / "env_files.parquet"

    @property
    def envs_path(self) -> Path:
        return self.root / "envs.parquet"

    @property
    def benchmark_defs_path(self) -> Path:
        return self.root / "benchmark_defs.parquet"

    @property
    def commit_info_path(self) -> Path:
        return self.root / "commit_info.parquet"

    def partition_path(self, table: str, week: str) -> Path:
        return self.root / table / f"week={week}" / "data.parquet"

    def weeks(self) -> list[str]:
        return sorted(
            p.parent.name.removeprefix("week=")
            for p in (self.root / "values").glob("week=*/data.parquet")
        )

    # -- reads ---------------------------------------------------------------

    def runs(self) -> pl.DataFrame:
        if not self.runs_path.exists():
            return pl.DataFrame(results.RUNS_SCHEMA.empty_table())
        return pl.read_parquet(self.runs_path)

    def stored_shas(self) -> set[str]:
        return set(self.runs()["sha"].to_list())

    def read_partition(self, table: str, week: str) -> pl.DataFrame | None:
        path = self.partition_path(table, week)
        return pl.read_parquet(path) if path.exists() else None

    # -- updates -------------------------------------------------------------

    def update(
        self,
        input_dir: Path,
        workers: int | None = None,
        pandas_repo: Path | None = None,
    ) -> UpdateReport:
        """Add everything new under ``input_dir`` (the output of fetch-results).

        With ``pandas_repo`` (a clone of pandas; commit objects are enough), also
        store the title and PR number of commits that do not have them yet.

        Reads ``data/results/asvrunner/<sha>.json.zst``,
        ``data/envs/<sha>.yml.zst``, and ``data/results/benchmarks.json``.
        """
        input_dir = Path(input_dir)
        report = UpdateReport()
        result_files = sorted(
            (input_dir / "data" / "results" / "asvrunner").glob(f"*{RESULT_SUFFIX}")
        )
        report.results_added, report.weeks_written = self.add_results(
            result_files, workers=workers
        )
        report.envs_added = self.add_envs(
            sorted((input_dir / "data" / "envs").glob(f"*{ENV_SUFFIX}"))
        )
        defs = input_dir / "data" / "results" / "benchmarks.json"
        if defs.exists():
            report.benchmark_defs_added = self.add_benchmark_defs(defs)
        if pandas_repo is not None:
            report.commit_info_added = self.add_commit_info(Path(pandas_repo))
        return report

    def commit_info(self) -> pl.DataFrame:
        if not self.commit_info_path.exists():
            return pl.DataFrame(COMMIT_INFO_SCHEMA.empty_table())
        return pl.read_parquet(self.commit_info_path)

    def add_commit_info(self, pandas_repo: Path) -> int:
        """Store title and PR number for stored commits that lack them.

        Commits missing from the clone are skipped and tried again next time.
        """
        info = self.commit_info()
        missing = sorted(self.stored_shas() - set(info["sha"].to_list()))
        if not missing:
            return 0
        rows: dict[str, list] = {"sha": [], "title": [], "pr": []}
        for sha, message in read_commit_messages(pandas_repo, missing).items():
            title = message.split("\n", 1)[0].strip()
            m = PR_SUFFIX.search(title)
            rows["sha"].append(sha)
            rows["title"].append(title)
            rows["pr"].append(int(m.group(1)) if m else None)
        if not rows["sha"]:
            return 0
        new = pl.DataFrame(pa.table(rows, schema=COMMIT_INFO_SCHEMA))
        write_parquet(pl.concat([info, new]).sort("sha"), self.commit_info_path)
        return new.height

    def add_results(
        self, paths: Iterable[Path], workers: int | None = None
    ) -> tuple[list[str], list[str]]:
        """Add result files whose commit is not stored yet.

        Returns
        -------
        tuple[list[str], list[str]]
            The commits added and the weeks rewritten.
        """
        stored = self.stored_shas()
        # Skip by filename before reading: parsing is most of the cost.
        new = [p for p in paths if sha_from_path(p, RESULT_SUFFIX) not in stored]
        if not new:
            return [], []

        self.root.mkdir(parents=True, exist_ok=True)
        new_runs: list[dict] = []
        # The spill directory is outside the store so a killed run cannot leave
        # it in the tree that gets committed.
        with tempfile.TemporaryDirectory(prefix="asv-runner-spill-") as spill:
            # Parsed rows are spilled to one file per (table, week) so a full
            # rebuild never holds more than one week in memory.
            writers: dict[tuple[str, str], pq.ParquetWriter] = {}

            def spill_path(table: str, week: str) -> Path:
                return Path(spill) / f"{table}-{week}.parquet"

            for week, bench, values, run in _map(_parse_result_file, new, workers):
                new_runs.append(run)
                for table, data in (("benchmarks", bench), ("values", values)):
                    key = (table, week)
                    if key not in writers:
                        writers[key] = pq.ParquetWriter(
                            spill_path(table, week), data.schema
                        )
                    writers[key].write_table(data)
            for w in writers.values():
                w.close()

            runs_new = pl.DataFrame(
                pa.Table.from_pylist(new_runs, schema=results.RUNS_SCHEMA)
            )
            runs_all = pl.concat(
                [
                    self.runs().filter(~pl.col("sha").is_in(runs_new["sha"].to_list())),
                    runs_new,
                ]
            )
            dates = runs_all.select("sha", "date")
            new_shas = runs_new["sha"].to_list()
            weeks = sorted({week for _, week in writers})
            for week in weeks:
                for table, sort_cols in (
                    ("benchmarks", ["name", "date", "sha"]),
                    ("values", ["name", "combo", "date", "sha"]),
                ):
                    added = pl.read_parquet(spill_path(table, week))
                    frames = [added]
                    existing = self.read_partition(table, week)
                    if existing is not None:
                        frames.insert(
                            0, existing.filter(~pl.col("sha").is_in(new_shas))
                        )
                    df = (
                        pl.concat(frames)
                        .join(dates, on="sha", how="left")
                        .sort(sort_cols)
                        .drop("date")
                    )
                    write_parquet(df, self.partition_path(table, week))

        write_parquet(runs_all.sort("date", "sha"), self.runs_path)
        return runs_new["sha"].to_list(), weeks

    def add_envs(self, paths: Iterable[Path]) -> list[str]:
        """Add env files whose commit is not stored yet."""
        if self.env_files_path.exists():
            files = pl.read_parquet(self.env_files_path)
        else:
            files = pl.DataFrame(envs.ENV_FILES_SCHEMA.empty_table())
        stored = set(files["sha"].to_list())
        new = [p for p in paths if sha_from_path(p, ENV_SUFFIX) not in stored]
        if not new:
            return []

        new_files: dict[str, list] = {"sha": [], "text": []}
        new_rows: dict[str, list] = {f.name: [] for f in envs.ENVS_SCHEMA}
        for path in new:
            sha = sha_from_path(path, ENV_SUFFIX)
            text = read_zst(path).decode()
            new_files["sha"].append(sha)
            new_files["text"].append(text)
            for k, v in envs.parse(sha, text).items():
                new_rows[k].extend(v)

        parsed = (
            pl.read_parquet(self.envs_path)
            if self.envs_path.exists()
            else pl.DataFrame(envs.ENVS_SCHEMA.empty_table())
        )
        new_parsed = pl.DataFrame(pa.table(new_rows, schema=envs.ENVS_SCHEMA))
        # Write parsed rows first: env_files.parquet marks a commit as stored.
        write_parquet(
            pl.concat(
                [parsed.filter(~pl.col("sha").is_in(new_files["sha"])), new_parsed]
            ).sort("sha", "manager", "package"),
            self.envs_path,
        )
        new_files_df = pl.DataFrame(pa.table(new_files, schema=envs.ENV_FILES_SCHEMA))
        write_parquet(pl.concat([files, new_files_df]).sort("sha"), self.env_files_path)
        return new_files["sha"]

    def add_benchmark_defs(self, path: Path) -> int:
        """Add the entries of a ``benchmarks.json`` not seen before.

        ``benchmarks.json`` describes only the most recently benchmarked commit,
        so keeping every distinct entry preserves parameter names and code for
        benchmarks that were later changed, renamed, or removed.
        """
        with open(path) as fh:
            data = json.load(fh)
        rows: dict[str, list] = {"name": [], "version": [], "definition": []}
        for name, definition in data.items():
            if name == "version":  # benchmarks.json format version
                continue
            rows["name"].append(name)
            rows["version"].append(definition.get("version"))
            rows["definition"].append(json.dumps(definition, sort_keys=True))
        new = pl.DataFrame(pa.table(rows, schema=BENCHMARK_DEFS_SCHEMA))
        if self.benchmark_defs_path.exists():
            existing = pl.read_parquet(self.benchmark_defs_path)
            new = new.join(existing, on=new.columns, how="anti")
            if new.is_empty():
                return 0
            new_all = pl.concat([existing, new])
        else:
            new_all = new
        write_parquet(
            new_all.sort("name", "version", "definition"), self.benchmark_defs_path
        )
        return new.height

    # -- rebuild -------------------------------------------------------------

    def rebuild_week(self, week: str) -> Iterator[tuple[str, dict]]:
        """Yield (sha, result file dict) for every commit stored in ``week``."""
        bench = self.read_partition("benchmarks", week)
        values = self.read_partition("values", week)
        if bench is None or values is None:
            return
        shas = bench["sha"].unique()
        meta = dict(
            self.runs()
            .filter(pl.col("sha").is_in(shas))
            .select("sha", "meta")
            .iter_rows()
        )
        values_by_sha = values.sort("sha", "name", "combo").partition_by(
            "sha", as_dict=True
        )
        for (sha,), b in bench.partition_by("sha", as_dict=True).items():
            v = values_by_sha.get((sha,))
            by_name: dict[str, dict[str, list]] = {}
            if v is not None:
                for (name,), g in v.partition_by("name", as_dict=True).items():
                    by_name[name] = g.to_dict(as_series=False)
            yield sha, results.rebuild(meta[sha], b.to_dicts(), by_name)

    def env_text(self) -> dict[str, str]:
        return dict(pl.read_parquet(self.env_files_path).iter_rows())


def _map(func, items: list, workers: int | None):
    if workers == 1 or len(items) < 8:
        yield from map(func, items)
        return
    # Polars starts threads on import; forking a threaded process can deadlock.
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        yield from pool.map(func, items, chunksize=4)
