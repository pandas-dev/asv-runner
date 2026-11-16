from __future__ import annotations

import json
from pathlib import Path

import zstandard

RESULT_COLUMNS = [
    "result",
    "params",
    "version",
    "started_at",
    "duration",
    "stats_ci_99_a",
    "stats_ci_99_b",
    "stats_q_25",
    "stats_q_75",
    "stats_number",
    "stats_repeat",
    "samples",
    "profile",
]


def make_result(sha: str, date_ms: int, results: dict | None = None) -> dict:
    """A result file shaped like asv's output, with a few edge cases."""
    if results is None:
        results = {
            "algorithms.Duplicated.time_duplicated": [
                [1.5e-06, float("nan"), None, 0.0031],
                [["True", "False"], ["'first'", "'last'"]],
                "v1",
                1772940858665,
                21.031,
                [1.4873e-06, None, None, 0.0030268],
                [1.597e-06, None, None, 0.0032],
                [1.4907e-06, None, None, 0.0030901],
                [1.5941e-06, None, None, 0.00313],
                [5657, None, None, 3],
                [10, None, None, 10],
            ],
            "series_methods.Peakmem.peakmem_x": [
                [123456789],
                None,
                "v2",
                1772940858999,
                3,  # an int duration falls back to raw JSON
            ],
            "frame_methods.Odd.time_odd": [
                [2.0],
                [],
                "v3",
                1772940859000,
                1.0,
                [-0.0],  # -0.0 has no decimal form
                [float("inf")],
                [1.0],
                [1.0],
                [1],
                [1],
                [[1.0, 2.0]],  # samples
            ],
        }
    return {
        "commit_hash": sha,
        "env_name": "existing-py_test",
        "date": date_ms,
        "params": {"machine": "asvrunner", "python": "/usr/bin/python"},
        "python": "/usr/bin/python",
        "requirements": {},
        "env_vars": {},
        "result_columns": RESULT_COLUMNS,
        "results": results,
        "durations": {"<build>": 6.7e-05},
        "version": 2,
    }


CONDA_ENV = """\
name: /home/runner/micromamba/envs/test
channels:
  - conda-forge
dependencies:
  - numpy=2.3.1=py311h_0
  - python=3.11.9=hb806964_0
  - pip:
      - pandas==3.1.0.dev0+276.g0039f168fe
prefix: /home/runner/micromamba/envs/test
"""

PIXI_ENV = """\
Name                              Version
numpy                             2.4.3
python                            3.13.13
"""


def write_zst(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstandard.ZstdCompressor().compress(data))


def add_commit(input_dir: Path, sha: str, date_ms: int, env: str = CONDA_ENV) -> dict:
    """Write a result file and env file for ``sha`` as the results branch holds them."""
    d = make_result(sha, date_ms)
    write_zst(
        input_dir / "data" / "results" / "asvrunner" / f"{sha}.json.zst",
        json.dumps(d).encode(),
    )
    write_zst(input_dir / "data" / "envs" / f"{sha}.yml.zst", env.encode())
    return d


def write_benchmark_defs(input_dir: Path, defs: dict) -> None:
    path = input_dir / "data" / "results" / "benchmarks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**defs, "version": 2}))
