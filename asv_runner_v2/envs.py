"""Parse the per-commit environment files into package rows.

Two formats occur:

- ``conda env export`` YAML: ``- name=version=build`` under ``dependencies:``,
  plus ``- name==version`` under a nested ``- pip:`` list.
- ``pixi list`` table: a header line starting with ``Name`` followed by
  whitespace-separated rows.
"""

from __future__ import annotations

import re

import pyarrow as pa

CONDA_EXPORT = "conda-export"
PIXI_LIST = "pixi-list"

ENV_FILES_SCHEMA = pa.schema([("sha", pa.string()), ("text", pa.string())])
ENVS_SCHEMA = pa.schema(
    [
        ("sha", pa.string()),
        ("format", pa.string()),
        ("manager", pa.string()),  # "conda" or "pypi"
        ("package", pa.string()),
        ("version", pa.string()),
        ("build", pa.string()),
    ]
)


class EnvFormatError(ValueError):
    pass


def detect_format(text: str) -> str:
    first = text.lstrip().split("\n", 1)[0]
    if first.startswith("name:") or "\ndependencies:" in text:
        return CONDA_EXPORT
    if first.split()[:1] == ["Name"]:
        return PIXI_LIST
    raise EnvFormatError(f"Unrecognized env file format; first line: {first!r}")


def _parse_conda_export(text: str) -> list[tuple[str, str, str, str | None]]:
    rows: list[tuple[str, str, str, str | None]] = []
    in_deps = in_pip = False
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            in_deps = line.startswith("dependencies:")
            in_pip = False
            continue
        if not in_deps:
            continue
        item = line.strip()
        if not item.startswith("- "):
            raise EnvFormatError(f"Unexpected line in dependencies: {line!r}")
        item = item[2:].strip()
        if item == "pip:":
            in_pip = True
            continue
        # The pip list is indented deeper than the conda list.
        if in_pip and len(line) - len(line.lstrip()) <= 2:
            in_pip = False
        if in_pip:
            name, sep, version = item.partition("==")
            if not sep:
                raise EnvFormatError(f"Unexpected pip line: {line!r}")
            rows.append(("pypi", name, version, None))
        else:
            parts = item.split("=")
            if len(parts) != 3:
                raise EnvFormatError(f"Unexpected conda line: {line!r}")
            rows.append(("conda", parts[0], parts[1], parts[2]))
    return rows


def _parse_pixi_list(text: str) -> list[tuple[str, str, str, str | None]]:
    lines = [line for line in text.splitlines() if line.strip()]
    # Columns are aligned under the header; a value such as "8 MiB" can contain
    # a space, so slice at the header's column starts instead of splitting.
    starts = [m.start() for m in re.finditer(r"\S+", lines[0])]
    header = [
        lines[0][i:j].strip().lower() for i, j in zip(starts, [*starts[1:], None])
    ]
    if header[:2] != ["name", "version"]:
        raise EnvFormatError(f"Unexpected pixi list header: {lines[0]!r}")
    rows: list[tuple[str, str, str, str | None]] = []
    for line in lines[1:]:
        fields = {
            h: line[i:j].strip() for h, i, j in zip(header, starts, [*starts[1:], None])
        }
        kind = fields.get("kind", "conda")
        rows.append(
            (kind, fields["name"], fields["version"], fields.get("build") or None)
        )
    return rows


def parse(sha: str, text: str) -> dict[str, list]:
    """Parse one env file into columns matching ``ENVS_SCHEMA``."""
    fmt = detect_format(text)
    parser = _parse_conda_export if fmt == CONDA_EXPORT else _parse_pixi_list
    out: dict[str, list] = {f.name: [] for f in ENVS_SCHEMA}
    for manager, package, version, build in parser(text):
        out["sha"].append(sha)
        out["format"].append(fmt)
        out["manager"].append(manager)
        out["package"].append(package)
        out["version"].append(version)
        out["build"].append(build)
    return out
