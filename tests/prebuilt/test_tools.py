"""Tests for prebuilt AgentFlow tools."""

from __future__ import annotations

import json
import sys
import types as pytypes
from types import SimpleNamespace

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.prebuilt.tools import (
    create_file_tools,
    fetch_url,
    file_read,
    file_search,
    file_write,
    google_web_search,
    safe_calculator,
    vertex_ai_search,
)


def test_safe_calculator_evaluates_basic_math() -> None:
    result = json.loads(safe_calculator("(2 + 3) * 4"))

    assert result == {"result": 20}


def test_safe_calculator_rejects_unsupported_expressions() -> None:
    result = json.loads(safe_calculator("__import__('os').system('echo nope')"))

    assert "error" in result


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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


# --- Security review 01-F1: the file tool root must not come from the run config ---


def _scoped_tools(root, **kwargs):
    return {fn.__name__: fn for fn in create_file_tools(root, **kwargs)}


def test_create_file_tools_are_scoped_to_their_root(tmp_path) -> None:
    tools = _scoped_tools(tmp_path)

    written = json.loads(tools["scoped_file_write"]("notes/info.txt", "hello", create_dirs=True))
    assert written["status"] == "written"
    assert (tmp_path / "notes" / "info.txt").read_text() == "hello"

    read = json.loads(tools["scoped_file_read"]("notes/info.txt"))
    assert read["content"] == "hello"

    found = json.loads(tools["scoped_file_search"]("hello"))
    assert found["results"][0]["path"] == "notes/info.txt"

    blocked = json.loads(tools["scoped_file_read"]("../outside.txt"))
    assert "configured root" in blocked["error"]


@pytest.mark.asyncio
async def test_create_file_tools_ignore_root_in_run_config(tmp_path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET")

    node = ToolNode(create_file_tools(root))
    schemas = await node.all_tools()
    for item in schemas:
        assert "config" not in item["function"]["parameters"]["properties"]

    tools = _scoped_tools(root)
    # The factory tools take no config, so a client value has nowhere to go
    with pytest.raises(TypeError):
        tools["scoped_file_read"]("secret.txt", config={"file_tool_root": str(outside)})
    result = json.loads(tools["scoped_file_read"]("secret.txt"))
    assert result["error"] == "file does not exist"


def test_create_file_tools_selects_tools(tmp_path) -> None:
    tools = create_file_tools(tmp_path, write=False)
    names = [fn._py_tool_name for fn in tools]

    # Same names the model sees for the module-level tools
    assert names == ["file_read", "file_search"]


def test_create_file_tools_rejects_missing_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        create_file_tools(tmp_path / "missing")


def test_run_config_file_root_is_deprecated(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("hi")

    with pytest.warns(DeprecationWarning, match="create_file_tools"):
        result = json.loads(file_read("a.txt", config={"file_tool_root": str(tmp_path)}))
    assert result["content"] == "hi"

    with pytest.warns(DeprecationWarning):
        file_read("a.txt", config={"workspace_root": str(tmp_path)})


# --- Security review 01-F8: file_search must not follow symlinks out of the root ---


def _workspace_with_escaping_links(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET-VALUE")
    (root / "inside.txt").write_text("SECRET-VALUE is not here, this is fine")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "linkdir").symlink_to(outside, target_is_directory=True)
    return root


def test_file_search_skips_symlinks_that_leave_the_root(tmp_path) -> None:
    root = _workspace_with_escaping_links(tmp_path)
    tools = _scoped_tools(root)

    found = json.loads(tools["scoped_file_search"]("SECRET-VALUE"))
    paths = {item["path"] for item in found["results"]}

    assert paths == {"inside.txt"}
    assert all("is not here" in item["preview"] for item in found["results"])

    by_name = json.loads(tools["scoped_file_search"]("link"))
    assert by_name["results"] == []


def test_file_search_follows_symlinks_that_stay_inside_the_root(tmp_path) -> None:
    root = tmp_path / "ws"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "real.txt").write_text("needle")
    (root / "alias.txt").symlink_to(root / "docs" / "real.txt")
    tools = _scoped_tools(root)

    found = json.loads(tools["scoped_file_search"]("needle"))
    paths = sorted(item["path"] for item in found["results"])

    assert paths == ["alias.txt", "docs/real.txt"]
