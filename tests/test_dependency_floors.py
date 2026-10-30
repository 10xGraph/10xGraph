"""Extras keep security floors for libraries that handle untrusted input (G-71)."""

import re
import tomllib
from pathlib import Path


def _extras():
    data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    return data["project"]["optional-dependencies"]


def _floor(specs, name):
    for spec in specs:
        m = re.match(rf"{re.escape(name)}>=([0-9.]+)", spec)
        if m:
            return tuple(int(p) for p in m.group(1).split("."))
    raise AssertionError(f"{name} has no lower bound in {specs}")


def test_pillow_floor():
    assert _floor(_extras()["images"], "Pillow") >= (13, 0, 0)


def test_mcp_floor():
    assert _floor(_extras()["mcp"], "mcp") >= (1, 30, 0)


def test_opentelemetry_floor():
    for extra in ("otel", "langsmith"):
        assert _floor(_extras()[extra], "opentelemetry-api") >= (1, 45, 1)
