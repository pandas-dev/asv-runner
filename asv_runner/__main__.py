"""CLI entry point for asv_runner.

Subcommands invoked from .github/workflows/*.yaml:

Hourly benchmark run (run_asvs.yaml), standard library only:
  claim       - lease next pandas SHA (new or stale retry), update shas.txt, push
  benchmark   - run asv machine + asv run for a target SHA
  stage       - zstd-compress the run's result and env files for upload as an
                artifact
  complete    - mark the SHA's lease done once its artifact is uploaded

Nightly processing (process_results.yaml), needs the package's dependencies:
  update    - add new result files, env files, benchmark definitions, and commit
              titles from the downloaded results to the Parquet store
  site      - build the static benchmark site from the store
  issues    - open GitHub issues for new regressions (or write them to a file)
  fetch-results - download result artifacts of commits not yet stored
  check-results - fail if a completed lease's results were never stored
  check-store - fail if an update changed rows already in the store
  pull-store  - download the store from its Hugging Face dataset
  push-store  - upload the store to its Hugging Face dataset as one commit
  squash-store - squash the dataset's history once it is old enough
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# The hourly benchmark run executes this module from a source checkout without
# installing the package, so only standard-library modules are imported here;
# the nightly commands import theirs when they run.
from asv_runner import benchmark, claim, stage_results

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


def _fetch_results(args: argparse.Namespace) -> int:
    from asv_runner import artifacts

    return artifacts.fetch_cmd(args)


def _check_results(args: argparse.Namespace) -> int:
    from asv_runner import artifacts

    return artifacts.check_cmd(args)


def _check_store(args: argparse.Namespace) -> int:
    from asv_runner import check_store

    return check_store.check_cmd(args)


def _pull_store(args: argparse.Namespace) -> int:
    from asv_runner import hub

    return hub.pull_cmd(args)


def _push_store(args: argparse.Namespace) -> int:
    from asv_runner import hub

    return hub.push_cmd(args)


def _squash_store(args: argparse.Namespace) -> int:
    from asv_runner import hub

    return hub.squash_cmd(args)


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

    p = sub.add_parser("stage")
    p.add_argument("--asv-dir", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=stage_results.run)

    p = sub.add_parser("complete")
    p.add_argument(
        "--storage-dir", required=True, help="Checkout of the claims branch."
    )
    p.add_argument("--sha", required=True)
    p.add_argument("--branch", required=True)
    p.set_defaults(func=claim.complete)

    p = sub.add_parser("update")
    p.add_argument(
        "--input",
        required=True,
        help="Directory holding data/results/asvrunner and data/envs, "
        "such as the output of fetch-results.",
    )
    p.add_argument("--store", required=True, help="Root of the Parquet store.")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument(
        "--pandas-repo",
        help="Clone of pandas, to store commit titles and PR numbers.",
    )
    p.set_defaults(func=_update)

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

    p = sub.add_parser("fetch-results")
    p.add_argument("--store", required=True, help="Root of the Parquet store.")
    p.add_argument("--out", required=True, help="Directory to extract results into.")
    p.set_defaults(func=_fetch_results)

    p = sub.add_parser("check-results")
    p.add_argument("--store", required=True, help="Root of the Parquet store.")
    p.add_argument("--claims", required=True, help="The claims branch's shas.txt.")
    p.set_defaults(func=_check_results)

    p = sub.add_parser("check-store")
    p.add_argument("--old", required=True, help="Root of the store before the update.")
    p.add_argument("--new", required=True, help="Root of the store after the update.")
    p.set_defaults(func=_check_store)

    dataset_help = (
        "Hugging Face dataset holding the store, such as pandas-dev/asv-runner."
    )
    revision_help = (
        "File recording the dataset commit the store was downloaded at "
        "(written by pull-store, updated by push-store)."
    )

    p = sub.add_parser("pull-store")
    p.add_argument("--dataset", required=True, help=dataset_help)
    p.add_argument(
        "--out",
        required=True,
        help="Directory to download into; the store is OUT/data.",
    )
    p.add_argument("--revision-file", required=True, help=revision_help)
    p.set_defaults(func=_pull_store)

    p = sub.add_parser("push-store")
    p.add_argument("--dataset", required=True, help=dataset_help)
    p.add_argument("--store", required=True, help="Root of the Parquet store.")
    p.add_argument("--revision-file", required=True, help=revision_help)
    p.add_argument("--message", default="Update store", help="Commit message.")
    p.set_defaults(func=_push_store)

    p = sub.add_parser("squash-store")
    p.add_argument("--dataset", required=True, help=dataset_help)
    p.add_argument(
        "--store", required=True, help="Root of the Parquet store just pushed."
    )
    p.add_argument("--revision-file", required=True, help=revision_help)
    p.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Squash once the oldest commit is older than this.",
    )
    p.set_defaults(func=_squash_store)

    args = parser.parse_args()
    return args.func(args)


sys.exit(main())
