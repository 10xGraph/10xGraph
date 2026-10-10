"""Tests for prebuilt AgentFlow tools."""

from __future__ import annotations

import json
import sys
import types as pytypes
from types import SimpleNamespace

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.prebuilt.tools import (
    fetch_url,
    file_read,
    file_search,
    file_write,
    google_web_search,
    safe_calculator,
    vertex_ai_search,
)
from tenxgraph.prebuilt.tools import fetch as fetch_module


def test_safe_calculator_evaluates_basic_math() -> None:
    result = json.loads(safe_calculator("(2 + 3) * 4"))

    assert result == {"result": 20}


def test_safe_calculator_rejects_unsupported_expressions() -> None:
    result = json.loads(safe_calculator("__import__('os').system('echo nope')"))

    assert "error" in result


def test_file_read_write_and_search_are_workspace_scoped(tmp_path) -> None:
    config = {"file_tool_root": str(tmp_path)}

    write_result = json.loads(
        file_write(
            "notes/info.txt",
            "hello agentflow\nsecond line",
            create_dirs=True,
            config=config,
        )
    )
    assert write_result["status"] == "written"

    read_result = json.loads(file_read("notes/info.txt", start_line=1, end_line=1, config=config))
    assert read_result["content"] == "hello agentflow"

    search_result = json.loads(file_search("agentflow", path="notes", config=config))
    assert search_result["results"][0]["path"] == "notes/info.txt"
    assert search_result["results"][0]["line"] == 1

    blocked = json.loads(file_read("../outside.txt", config=config))
    assert "configured root" in blocked["error"]


@pytest.mark.asyncio
async def test_fetch_url_returns_normalized_html(monkeypatch) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size: int) -> bytes:
            return (
                b"<html><body><h1>Hello</h1><script>skip()</script>"
                b"<p>AgentFlow</p></body></html>"
            )

        def getcode(self) -> int:
            return 200

        def geturl(self) -> str:
            return "https://example.com/"

    monkeypatch.setattr(fetch_module, "_is_public_hostname", lambda hostname: True)
    monkeypatch.setattr(fetch_module.request, "urlopen", lambda req, timeout: FakeResponse())

    result = json.loads(await fetch_url("https://example.com"))

    assert result["status_code"] == 200
    assert "Hello" in result["content"]
    assert "AgentFlow" in result["content"]
    assert "skip" not in result["content"]


def _install_fake_google_genai(monkeypatch) -> None:
    response = SimpleNamespace(
        text="grounded answer",
        candidates=[
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(
                    model_dump=lambda: {"sources": [{"uri": "https://example.com"}]}
                )
            )
        ],
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return response

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = FakeModels()

    class SimpleType:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    google_mod = pytypes.ModuleType("google")
    genai_mod = pytypes.ModuleType("google.genai")
    genai_types_mod = pytypes.ModuleType("google.genai.types")

    genai_mod.Client = FakeClient
    genai_types_mod.GenerateContentConfig = SimpleType
    genai_types_mod.GoogleSearch = SimpleType
    genai_types_mod.HttpOptions = SimpleType
    genai_types_mod.Retrieval = SimpleType
    genai_types_mod.Tool = SimpleType
    genai_types_mod.VertexAISearch = SimpleType
    genai_mod.types = genai_types_mod
    google_mod.genai = genai_mod

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types_mod)


@pytest.mark.asyncio
async def test_google_web_search_uses_google_genai(monkeypatch) -> None:
    _install_fake_google_genai(monkeypatch)

    result = json.loads(await google_web_search("latest agent frameworks"))

    assert result["content"] == "grounded answer"
    assert result["grounding_metadata"]["sources"][0]["uri"] == "https://example.com"


@pytest.mark.asyncio
async def test_vertex_ai_search_requires_datastore() -> None:
    result = json.loads(await vertex_ai_search("policies", datastore=""))

    assert result["error"] == "datastore is required"


@pytest.mark.asyncio
async def test_vertex_ai_search_uses_google_genai(monkeypatch) -> None:
    _install_fake_google_genai(monkeypatch)

    result = json.loads(
        await vertex_ai_search(
            "policies",
            datastore=(
                "projects/demo/locations/global/collections/default_collection/"
                "dataStores/policies"
            ),
        )
    )

    assert result["content"] == "grounded answer"


@pytest.mark.asyncio
async def test_tool_node_hides_injected_file_config_from_schema() -> None:
    node = ToolNode([file_read, file_write, file_search])

    schemas = await node.all_tools()
    params_by_name = {
        item["function"]["name"]: item["function"]["parameters"]["properties"]
        for item in schemas
    }

    assert "config" not in params_by_name["file_read"]
    assert "config" not in params_by_name["file_write"]
    assert "config" not in params_by_name["file_search"]
