"""CLI entry point for asv_runner.

Subcommands invoked from .github/workflows/*.yaml:
  claim     - pick next pandas SHA, append to shas.txt, push storage branch
  benchmark - run asv machine + asv run for a target SHA
  push      - stage asv outputs (zstd-compress per-sha files) and push to storage
  process   - decompress, asv publish, build parquet, raise issues, push
"""

from __future__ import annotations

import argparse

from asv_runner import benchmark, claim, process_results, push_results


def main() -> None:
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

    p = sub.add_parser("process")
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--asv-dir", required=True)
    p.add_argument("--branch", required=True)
    p.set_defaults(func=process_results.run)

    args = parser.parse_args()
    args.func(args)


main()
