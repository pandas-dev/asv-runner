from __future__ import annotations

import pytest

from asv_runner import envs
from tests._results_helpers import CONDA_ENV, PIXI_ENV


def _rows(out: dict[str, list]) -> list[tuple]:
    return list(zip(out["manager"], out["package"], out["version"], out["build"]))


def test_conda_export() -> None:
    out = envs.parse("abc", CONDA_ENV)
    assert set(out["format"]) == {envs.CONDA_EXPORT}
    assert set(out["sha"]) == {"abc"}
    assert _rows(out) == [
        ("conda", "numpy", "2.3.1", "py311h_0"),
        ("conda", "python", "3.11.9", "hb806964_0"),
        ("pypi", "pandas", "3.1.0.dev0+276.g0039f168fe", None),
    ]


def test_pixi_list() -> None:
    out = envs.parse("abc", PIXI_ENV)
    assert set(out["format"]) == {envs.PIXI_LIST}
    assert _rows(out) == [
        ("conda", "numpy", "2.4.3", None),
        ("conda", "python", "3.13.13", None),
    ]


def test_pixi_list_with_kind_and_build() -> None:
    text = (
        "Name     Version  Build       Size  Kind   Source\n"
        "numpy    2.4.3    py313h_0    8 MiB conda  numpy\n"
        "pandas   3.1.0    pypi_0      1 MiB pypi   pandas\n"
    )
    out = envs.parse("abc", text)
    assert _rows(out) == [
        ("conda", "numpy", "2.4.3", "py313h_0"),
        ("pypi", "pandas", "3.1.0", "pypi_0"),
    ]


def test_unknown_format() -> None:
    with pytest.raises(envs.EnvFormatError, match="Unrecognized"):
        envs.parse("abc", "something else\n")
