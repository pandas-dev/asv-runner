"""Open GitHub issues for detected regressions.

One issue per commit with regressions, titled ``Commit <sha>`` as the old
pipeline did, so the duplicate check finds issues it already opened.

When one commit has more than ``MASS_CHANGE`` regressions, a single summary
issue labelled ``measurement-change`` is opened instead, pinging nobody. Such a
change is usually the benchmark environment or tooling; the issue lays out
what changed and how to add the commit to ``breaks.toml``.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from asv_runner_v2 import breaks as breaks_mod
from asv_runner_v2 import regressions, series, site
from asv_runner_v2.store import Store

REPO = "pandas-dev/asv-runner"
PANDAS = "pandas-dev/pandas"
COMMIT_URL = f"https://github.com/{PANDAS}/commit/"
COMPARE_URL = f"https://github.com/{PANDAS}/compare/"
PR_URL = f"https://github.com/{PANDAS}/pull/"
DEFAULT_SITE_URL = "https://pandas-dev.github.io/asv-runner/"
GITHUB_ISSUE_LENGTH = 65000
# Only the latest commits with regressions are considered each run.
MAX_COMMITS = 40
MASS_CHANGE = 100
MASS_CHANGE_LABEL = "measurement-change"
TOOLING = {"asv", "asv_runner"}
COMPILERS = {"gcc", "gxx", "clang", "clangxx", "c-compiler", "cxx-compiler"}


def time_to_str(x: float) -> str:
    magnitude = abs(x)
    if magnitude >= 1.0:
        result = f"{magnitude:0.3f}s"
    elif magnitude >= 0.001:
        result = f"{magnitude * 1000:0.3f}ms"
    elif magnitude >= 0.000001:
        result = f"{magnitude * (1000**2):0.3f}us"
    else:
        result = f"{magnitude * (1000**3):0.3f}ns"
    return ("-" if x < 0 else "") + result


# -- GitHub access ------------------------------------------------------------


class GitHub:
    """GitHub through the ``gh`` CLI. ``dry_run`` makes only read calls and
    collects what would be created."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.would_create: list[dict] = []

    def _gh(self, args: list[str], input: str | None = None) -> str:
        return subprocess.run(
            ["gh", *args], input=input, capture_output=True, text=True, check=True
        ).stdout

    def api(self, path: str) -> Any:
        return json.loads(self._gh(["api", path]))

    def issue_exists(self, needle: str) -> bool:
        return self._gh(["search", "issues", "--repo", REPO, needle]).strip() != ""

    def create_issue(
        self, title: str, body: str, comment: str | None, labels: list[str]
    ) -> None:
        if self.dry_run:
            self.would_create.append(
                {"title": title, "body": body, "comment": comment, "labels": labels}
            )
            return
        args = ["issue", "create", "--repo", REPO, "--title", title, "--body-file", "-"]
        for label in labels:
            args += ["--label", label]
        url = self._gh(args, input=body).strip()
        if comment:
            number = url.rsplit("/", 1)[-1]
            self._gh(
                ["issue", "comment", number, "--repo", REPO, "--body-file", "-"],
                input=comment,
            )

    def pr_info(self, prev_sha: str, sha: str) -> dict | None:
        """PR number, author, and approvers when the range is exactly one commit."""
        compare = self.api(f"repos/{PANDAS}/compare/{prev_sha}...{sha}")
        if compare.get("ahead_by") != 1:
            return None
        prs = self.api(f"repos/{PANDAS}/commits/{sha}/pulls")
        if not prs:
            return None
        pr = prs[0]
        author = (pr.get("user") or {}).get("login")
        if not author:
            return None
        reviews = self.api(f"repos/{PANDAS}/pulls/{pr['number']}/reviews")
        approvers = sorted(
            {
                login
                for r in reviews
                if r.get("state") == "APPROVED"
                and (login := (r.get("user") or {}).get("login"))
                and login != author
            }
        )
        return {"number": pr["number"], "author": author, "approvers": approvers}

    def changed_files(self, prev_sha: str, sha: str) -> list[str] | None:
        """Files changed in the range; None if GitHub truncates the list."""
        compare = self.api(f"repos/{PANDAS}/compare/{prev_sha}...{sha}")
        files = compare.get("files") or []
        if len(files) >= 300:  # the compare API returns at most 300 files
            return None
        return [f["filename"] for f in files]


# -- issue content ------------------------------------------------------------


@dataclass
class Context:
    """Everything the issue text needs for one commit."""

    sha: str
    i: int
    prev_sha: str
    regressions: pl.DataFrame
    improvements: pl.DataFrame
    env: dict | None
    version_changes: int
    runner_changes: list[str] = field(default_factory=list)


def bench_url(
    site_url: str, name: str, params: list[str], param_names: list[str]
) -> str:
    parts = []
    for n, v in enumerate(params):
        param = param_names[n] if n < len(param_names) else f"param{n + 1}"
        parts.append(f"p-{urllib.parse.quote(param)}={urllib.parse.quote(v)}")
    return f"{site_url}#{name}" + ("?" + "&".join(parts) if parts else "")


def params_label(params: list[str], param_names: list[str]) -> str:
    return ", ".join(
        f"{param_names[n] if n < len(param_names) else f'param{n + 1}'}={v}"
        for n, v in enumerate(params)
    )


def env_lines(env: dict) -> list[str]:
    if env.get("format_change"):
        old, new = env["format_change"]
        return [f"environment tool changed: {old} → {new}"]
    return [f"{p}: {a or '—'} → {b or '—'}" for p, a, b in env["changes"]]


def env_comment(env: dict | None) -> str:
    if env is None:
        return "No environment changes from the previous benchmarked commit."
    lines = "\n".join(env_lines(env))
    return (
        f"Environment changes from the previous benchmarked commit:\n```\n{lines}\n```"
    )


def regression_body(
    ctx: Context,
    pr: dict | None,
    site_url: str,
    param_names: dict[str, list[str]],
    shorten: bool = False,
) -> str:
    if pr is not None:
        body = f"[PR #{pr['number']}]({PR_URL}{pr['number']})\n\n"
        mentions = [f"@{pr['author']}", *(f"@{a}" for a in pr["approvers"])]
        body += f"cc {' '.join(mentions)}\n\n"
    else:
        body = f"[Commit Range]({COMPARE_URL}{ctx.prev_sha}...{ctx.sha})\n\n"
        body += (
            "Subsequent benchmarks may have skipped some commits. The link above lists"
            " the commits that are between the two benchmark runs where the regression"
            " was identified.\n\n"
        )
    body += f"[All changes at this commit]({site_url}#commit/{ctx.sha})\n\n"
    if ctx.env is not None and (ctx.env.get("format_change") or ctx.env.get("n_key")):
        key = env_lines(ctx.env)[: max(ctx.env.get("n_key", 0), 1)]
        body += "Environment changed at this commit: " + "; ".join(key) + "\n\n"

    prev_name = None
    for row in ctx.regressions.sort("name", "series").iter_rows(named=True):
        names = param_names.get(row["name"], [])
        change = time_to_str(row["abs_change"])
        severity = f"{row['step']:+.1%} ({change} from the previous result)"
        detailed = bool(row["params"]) and not shorten
        if row["name"] != prev_name:
            body += f" - [ ] [{row['name']}]({site_url}#{row['name']})"
            body += "\n" if detailed else ""
            prev_name = row["name"]
        if not detailed:
            body += f" - {severity}\n"
            continue
        url = bench_url(site_url, row["name"], row["params"], names)
        body += f"   - [ ] [{params_label(row['params'], names)}]({url}) - {severity}\n"
    return body + "\n"


def mass_change_body(
    ctx: Context,
    changed_files: list[str] | None,
    site_url: str,
    param_names: dict[str, list[str]],
) -> str:
    reg = ctx.regressions
    steps = reg["step"].drop_nulls()
    modules = Counter(n.split(".")[0] for n in reg["name"].to_list())
    n_modules = len(modules)
    over_85 = (steps > 0.85).mean() if len(steps) else 0.0
    body = (
        f"At [{ctx.sha[:10]}]({COMMIT_URL}{ctx.sha}), {reg.height} benchmarks across "
        f"{n_modules} modules got slower (median {steps.median():+.0%}, "
        f"{over_85:.0%} of them by more than 85%). That is more than this repo opens "
        f"individual issues for ({MASS_CHANGE}). A change this large is usually the "
        "benchmark environment or tooling, not pandas.\n\n"
    )
    if ctx.improvements.height:
        body += (
            f"{ctx.improvements.height} benchmarks also got faster at this commit.\n\n"
        )

    body += (
        f"**What changed between the previous benchmarked commit "
        f"[{ctx.prev_sha[:10]}]({COMMIT_URL}{ctx.prev_sha}) and {ctx.sha[:10]}** "
        f"([compare]({COMPARE_URL}{ctx.prev_sha}...{ctx.sha}))\n\n| | |\n|---|---|\n"
    )
    changes = (
        [] if ctx.env is None or ctx.env.get("format_change") else ctx.env["changes"]
    )

    def fmt(rows: list) -> str:
        return (
            ", ".join(f"`{p}` {a or '—'} → {b or '—'}" for p, a, b in rows)
            or "unchanged"
        )

    tooling = [c for c in changes if c[0] in TOOLING]
    compilers = [c for c in changes if c[0] in COMPILERS]
    key = [
        c
        for c in changes
        if c[0] in site.KEY_PACKAGES and c[0] not in TOOLING | COMPILERS
    ]
    other = len(changes) - len(tooling) - len(compilers) - len(key)
    body += f"| Benchmark tooling | {fmt(tooling)} |\n"
    body += f"| Compilers | {fmt(compilers)} |\n"
    body += f"| Key packages | {fmt(key)} |\n"
    body += f"| Other packages | {other} changed (details below) |\n"
    if ctx.env is not None and ctx.env.get("format_change"):
        old, new = ctx.env["format_change"]
        body += f"| Environment tool | {old} → {new} |\n"
    if changed_files is None:
        asv_bench = "unknown (too many files changed to list)"
    else:
        touched = [f for f in changed_files if f.startswith("asv_bench/")]
        asv_bench = ", ".join(f"`{f}`" for f in touched[:5]) or "unchanged"
        if len(touched) > 5:
            asv_bench += f" and {len(touched) - 5} more"
    body += f"| Benchmark code (`asv_bench/`) | {asv_bench}"
    if ctx.version_changes:
        body += f"; {ctx.version_changes} benchmark versions changed"
    body += " |\n"
    body += f"| Runner | {'; '.join(ctx.runner_changes) or 'unchanged'} |\n\n"

    spread = ", ".join(f"{m} {n}" for m, n in modules.most_common(8))
    more = f" and {n_modules - 8} more modules" if n_modules > 8 else ""
    body += (
        f"**Spread:** {spread}{more}. A pandas change usually affects a few related "
        "modules; every module moving at once points to the environment.\n\n"
    )

    ordered = reg.filter(pl.col("step").is_not_null()).sort("step")
    mid = ordered.height // 2
    examples = [
        ("largest", ordered.tail(3).reverse()),
        ("near the median", ordered.slice(max(mid - 1, 0), 3)),
        ("smallest", ordered.head(3)),
    ]
    body += "**Examples** (each opens the chart):\n"
    for label, rows in examples:
        links = []
        for r in rows.iter_rows(named=True):
            names = param_names.get(r["name"], [])
            url = bench_url(site_url, r["name"], r["params"], names)
            links.append(f"[{r['name']}]({url}) {r['step']:+.0%}")
        body += f"- {label}: {', '.join(links)}\n"
    body += f"\n[All changes at this commit]({site_url}#commit/{ctx.sha})\n\n"

    breaks_url = f"https://github.com/{REPO}/blob/main/asv_runner_v2/breaks.toml"
    body += (
        "**What to do**\n"
        "1. If the cause is the environment or tooling (most likely when a row in "
        '"Benchmark tooling" or "Compilers" changed, `asv_bench/` did not, and every '
        f"module moved), add this to [`breaks.toml`]({breaks_url}) in a PR. Results "
        "before and after are then never compared:\n"
        "   ```toml\n   [[break]]\n"
        f'   sha = "{ctx.sha}"\n'
        f'   reason = "{_suggested_reason(tooling, compilers)}"\n'
        "   ```\n"
        "2. If it looks like a real pandas change (the compare range touches `pandas/` "
        "build or core code), comment here. A maintainer can then ask for the full "
        "per-benchmark list below to be split into normal issues.\n\n"
    )

    packages = ""
    if ctx.env is not None:
        packages = "<details><summary>All package changes</summary>\n\n```\n"
        packages += "\n".join(env_lines(ctx.env)) + "\n```\n\n</details>\n"
    # The benchmark list goes last and stops early enough that the body stays
    # under GitHub's limit; the commit view on the site has the full list.
    budget = GITHUB_ISSUE_LENGTH - len(body) - len(packages) - 1000
    lines: list[str] = []
    for row in reg.sort("name", "series").iter_rows(named=True):
        label = params_label(row["params"], param_names.get(row["name"], []))
        line = f"- {row['name']}{f' ({label})' if label else ''} {row['step']:+.0%}\n"
        if budget - len(line) < 0:
            lines.append(
                f"- … and {reg.height - len(lines)} more: see "
                f"[all changes at this commit]({site_url}#commit/{ctx.sha})\n"
            )
            break
        budget -= len(line)
        lines.append(line)
    body += f"<details><summary>All {reg.height} benchmarks</summary>\n\n"
    body += "".join(lines) + "\n</details>\n\n"
    return body + packages


def _suggested_reason(tooling: list, compilers: list) -> str:
    parts = [f"{p} {a} → {b}" for p, a, b in tooling + compilers]
    return "; ".join(parts) or "describe what changed"


def clip(body: str) -> str:
    if len(body) < GITHUB_ISSUE_LENGTH:
        return body
    return (
        body[:GITHUB_ISSUE_LENGTH] + "\n\nWARNING: Body has been clipped due to length."
    )


def runner_changes(store: Store, prev_sha: str, sha: str) -> list[str]:
    meta = dict(
        store.runs()
        .filter(pl.col("sha").is_in([prev_sha, sha]))
        .select("sha", "meta")
        .iter_rows()
    )
    if prev_sha not in meta or sha not in meta:
        return []
    old, new = json.loads(meta[prev_sha]), json.loads(meta[sha])
    out = [
        f"{key}: {old.get(key)} → {new.get(key)}"
        for key in ("env_name", "python")
        if old.get(key) != new.get(key)
    ]
    old_machine = (old.get("params") or {}).get("machine")
    new_machine = (new.get("params") or {}).get("machine")
    if old_machine != new_machine:
        out.append(f"machine: {old_machine} → {new_machine}")
    return out


# -- driver -------------------------------------------------------------------


def raise_issues(
    store: Store,
    github: GitHub,
    site_url: str = DEFAULT_SITE_URL,
    breaks: list[breaks_mod.Break] | None = None,
    pause: float = 2.0,
) -> list[str]:
    """Open issues for the latest commits with regressions. Returns their SHAs."""
    s = series.load(store)
    detections = regressions.detect(s, breaks=breaks)
    reg = detections.filter(pl.col("kind") == regressions.REGRESSION)
    imp = detections.filter(pl.col("kind") == regressions.IMPROVEMENT)
    commits = s.commits
    shas = commits["sha"].to_list()
    envs = {e["i"]: e for e in site.env_changes(store, commits)}
    version_changed = (
        s.runs.sort("name", "i")
        .filter(pl.col("version") != pl.col("version").shift(1).over("name"))
        .group_by("i")
        .len()
    )
    version_changes = dict(version_changed.iter_rows())
    defs = {name: d["param_names"] for name, d in site._definitions(store).items()}

    latest = sorted(set(reg["i"].to_list()))[-MAX_COMMITS:]
    print(f"Commits with regressions to check: {len(latest)}")
    raised = []
    for i in latest:
        sha, prev_sha = shas[i], shas[i - 1]
        if github.issue_exists(f"Commit {sha}"):
            continue
        time.sleep(pause)
        ctx = Context(
            sha=sha,
            i=i,
            prev_sha=prev_sha,
            regressions=reg.filter(pl.col("i") == i),
            improvements=imp.filter(pl.col("i") == i),
            env=envs.get(i),
            version_changes=version_changes.get(i, 0),
        )
        if ctx.regressions.height > MASS_CHANGE:
            ctx.runner_changes = runner_changes(store, prev_sha, sha)
            title = (
                f"Commit {sha}: {ctx.regressions.height} benchmarks slower; "
                "check for a measurement change"
            )
            body = mass_change_body(
                ctx, github.changed_files(prev_sha, sha), site_url, defs
            )
            github.create_issue(title, clip(body), None, [MASS_CHANGE_LABEL])
        else:
            pr = github.pr_info(prev_sha, sha)
            body = regression_body(ctx, pr, site_url, defs)
            if len(body) >= GITHUB_ISSUE_LENGTH:
                body = regression_body(ctx, pr, site_url, defs, shorten=True)
            github.create_issue(
                f"Commit {sha}", clip(body), clip(env_comment(ctx.env)), []
            )
        raised.append(sha)
    return raised


def write_dry_run(github: GitHub, path: Path) -> None:
    """Write the issues a dry run would have created, for review."""
    parts = [f"# Issues the new pipeline would open ({len(github.would_create)})\n"]
    for issue in github.would_create:
        labels = f" — labels: {', '.join(issue['labels'])}" if issue["labels"] else ""
        parts.append(f"\n---\n\n## {issue['title']}{labels}\n\n{issue['body']}\n")
        if issue["comment"]:
            parts.append(f"\n**Comment:**\n\n{issue['comment']}\n")
    Path(path).write_text("".join(parts))
