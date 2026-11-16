"""CLI entry point for asv_runner.

Subcommands invoked from .github/workflows/*.yaml:

Hourly benchmark run (run_asvs_*.yaml), standard library only:
  claim     - lease next pandas SHA (new or stale retry), update shas.txt, push
  benchmark - run asv machine + asv run for a target SHA
  push      - stage asv outputs (zstd-compress per-sha files) and push to storage

Nightly processing (process_results.yaml), needs the package's dependencies:
  update    - add new result files, env files, benchmark definitions, and commit
              titles from a checkout of the results branch to the Parquet store
  verify    - rebuild every stored file and compare it to the originals
  site      - build the static benchmark site from the store
  issues    - open GitHub issues for new regressions (or write them to a file)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# The hourly benchmark run executes this module from a source checkout without
# installing the package, so only standard-library modules are imported here;
# the nightly commands import theirs when they run.
from asv_runner import benchmark, claim, push_results

DEFAULT_SITE_URL = "https://pandas-dev.github.io/asv-runner/"


def _update(args: argparse.Namespace) -> int:
    from asv_runner.store import Store

    t0 = time.perf_counter()
    report = Store(Path(args.store)).update(
        Path(args.input),
        workers=args.workers,
        pandas_repo=Path(args.pandas_repo) if args.pandas_repo else None,
    )
    print(
        f"Added {len(report.results_added)} result files "
        f"({len(report.weeks_written)} weeks rewritten), "
        f"{len(report.envs_added)} env files, "
        f"{report.benchmark_defs_added} benchmark definitions, "
        f"{report.commit_info_added} commit titles "
        f"in {time.perf_counter() - t0:.1f}s."
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    from asv_runner.store import Store
    from asv_runner.verify import verify

    t0 = time.perf_counter()
    report = verify(Store(Path(args.store)), Path(args.input), workers=args.workers)
    print(
        f"Checked {report.results_checked} result files and {report.envs_checked} "
        f"env files in {time.perf_counter() - t0:.1f}s."
    )
    for line in report.mismatches[:20]:
        print(f"MISMATCH {line}")
    for line in report.missing[:20]:
        print(f"MISSING {line}")
    print(f"{len(report.mismatches)} mismatches, {len(report.missing)} missing.")
    return 0 if report.ok else 1


def _site(args: argparse.Namespace) -> int:
    from asv_runner import site
    from asv_runner.store import Store

    t0 = time.perf_counter()
    stats = site.build(Store(Path(args.store)), Path(args.out))
    print(
        f"Wrote {stats['benchmarks']} benchmark files "
        f"({stats['graphs_bytes'] / 1e6:.1f} MB, largest "
        f"{stats['largest_graph_bytes'] / 1e6:.2f} MB) and index.json "
        f"({stats['index_bytes'] / 1e6:.2f} MB) in {time.perf_counter() - t0:.1f}s."
    )
    return 0


def _issues(args: argparse.Namespace) -> int:
    from asv_runner import issues
    from asv_runner.store import Store

    github = issues.GitHub(dry_run=args.dry_run is not None)
    raised = issues.raise_issues(
        Store(Path(args.store)), github, site_url=args.site_url
    )
    if args.dry_run is not None:
        issues.write_dry_run(github, Path(args.dry_run))
        print(f"Wrote {len(raised)} would-be issues to {args.dry_run}.")
    else:
        print(f"Opened {len(raised)} issues.")
    return 0


def main() -> int | None:
    parser = argparse.ArgumentParser(description="CI orchestration for asv-runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("claim")
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--branch", required=True)
    p.set_defaults(func=claim.run)

    p = sub.add_parser("benchmark")
    p.add_argument("--asv-dir", required=True)
    p.add_argument("--sha", required=True)
    p.set_defaults(func=benchmark.run)

    p = sub.add_parser("push")
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--asv-dir", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--branch", required=True)
    p.set_defaults(func=push_results.run)

    for name, func in (("update", _update), ("verify", _verify)):
        p = sub.add_parser(name)
        p.add_argument("--input", required=True, help="Checkout of the results branch.")
        p.add_argument("--store", required=True, help="Root of the Parquet store.")
        p.add_argument("--workers", type=int, default=None)
        if name == "update":
            p.add_argument(
                "--pandas-repo",
                help="Clone of pandas, to store commit titles and PR numbers.",
            )
        p.set_defaults(func=func)

    p = sub.add_parser("site")
    p.add_argument("--store", required=True, help="Root of the Parquet store.")
    p.add_argument("--out", required=True, help="Output directory for the site.")
    p.set_defaults(func=_site)

    p = sub.add_parser("issues")
    p.add_argument("--store", required=True, help="Root of the Parquet store.")
    p.add_argument("--site-url", default=DEFAULT_SITE_URL)
    p.add_argument(
        "--dry-run",
        metavar="FILE",
        help="Write the issues that would be opened to FILE instead of opening them. "
        "Read-only GitHub calls (duplicate search, PR info) still happen.",
    )
    p.set_defaults(func=_issues)

    args = parser.parse_args()
    return args.func(args)


sys.exit(main())
