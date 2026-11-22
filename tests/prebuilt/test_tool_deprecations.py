"""The prebuilt tools slated for removal in 2.0 warn when used; the kept ones do not."""

from __future__ import annotations

import inspect
import subprocess
import sys
import warnings
from types import SimpleNamespace

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.prebuilt.tools import (
    create_fetch_url,
    create_file_tools,
    create_google_web_search,
    create_handoff_tool,
    create_vertex_ai_search,
    fetch_url,
    file_read,
    file_search,
    file_write,
    google_web_search,
    make_agent_memory_tool,
    make_user_memory_tool,
    safe_calculator,
    vertex_ai_search,
)


MSG = "is deprecated and will be removed in 10xgraph 2.0"


def test_calculator_warns() -> None:
    with pytest.warns(DeprecationWarning, match=f"safe_calculator {MSG}"):
        safe_calculator("1 + 1")


@pytest.mark.asyncio
async def test_fetch_url_warns() -> None:
    with pytest.warns(DeprecationWarning, match=f"fetch_url {MSG}"):
        await fetch_url("http://127.0.0.1/")


@pytest.mark.asyncio
async def test_google_web_search_warns() -> None:
    with pytest.warns(DeprecationWarning, match=f"google_web_search {MSG}"):
        await google_web_search("")


@pytest.mark.asyncio
async def test_vertex_ai_search_warns() -> None:
    with pytest.warns(DeprecationWarning, match=f"vertex_ai_search {MSG}"):
        await vertex_ai_search("", "datastore")


def test_file_read_warns(tmp_path) -> None:
    with pytest.warns(DeprecationWarning, match=f"file_read {MSG}"):
        file_read("a.txt", config={"file_tool_root": str(tmp_path)})


def test_file_write_warns(tmp_path) -> None:
    with pytest.warns(DeprecationWarning, match=f"file_write {MSG}"):
        file_write("a.txt", "x", config={"file_tool_root": str(tmp_path)})


def test_file_search_warns(tmp_path) -> None:
    with pytest.warns(DeprecationWarning, match=f"file_search {MSG}"):
        file_search("x", config={"file_tool_root": str(tmp_path)})


def test_create_fetch_url_warns() -> None:
    with pytest.warns(DeprecationWarning, match=f"create_fetch_url {MSG}"):
        create_fetch_url(["example.com"])


def test_create_google_web_search_warns() -> None:
    with pytest.warns(DeprecationWarning, match=f"create_google_web_search {MSG}"):
        create_google_web_search()


def test_create_vertex_ai_search_warns() -> None:
    with pytest.warns(DeprecationWarning, match=f"create_vertex_ai_search {MSG}"):
        create_vertex_ai_search("projects/p/datastores/d")


def test_create_file_tools_warns(tmp_path) -> None:
    with pytest.warns(DeprecationWarning, match=f"create_file_tools {MSG}"):
        create_file_tools(tmp_path)


def test_tools_from_factories_do_not_warn_again(tmp_path) -> None:
    with pytest.warns(DeprecationWarning):
        read_tool = create_file_tools(tmp_path, write=False, search=False)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        read_tool("a.txt")


def test_import_emits_no_warning() -> None:
    code = (
        "import warnings; warnings.simplefilter('error', DeprecationWarning); "
        "import tenxgraph.prebuilt.tools"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_kept_tools_do_not_warn() -> None:
    memory_config = SimpleNamespace(user_memory=None, agent_memory=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        create_handoff_tool("worker")
        make_user_memory_tool(memory_config)
        make_agent_memory_tool(memory_config)


def test_toolnode_schema_is_unchanged_and_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        schemas = ToolNode([fetch_url, file_read, safe_calculator]).get_local_tool()
    by_name = {s["function"]["name"]: s["function"] for s in schemas}

    fetch = by_name["fetch_url"]
    assert fetch["description"].startswith("Fetch a public http/https URL")
    assert set(fetch["parameters"]["properties"]) == {"url", "timeout", "max_chars"}
    assert fetch["parameters"]["required"] == ["url"]
    assert fetch["parameters"]["properties"]["timeout"]["type"] == "number"
    assert fetch["parameters"]["properties"]["max_chars"]["type"] == "integer"

    read = by_name["file_read"]
    assert set(read["parameters"]["properties"]) == {"path", "start_line", "end_line", "max_chars"}
    assert read["parameters"]["required"] == ["path"]

    calc = by_name["safe_calculator"]
    assert set(calc["parameters"]["properties"]) == {"expression", "precision"}
    assert calc["parameters"]["required"] == ["expression"]


def test_wrappers_keep_sync_and_async_kind() -> None:
    assert inspect.iscoroutinefunction(fetch_url)
    assert inspect.iscoroutinefunction(google_web_search)
    assert inspect.iscoroutinefunction(vertex_ai_search)
    assert not inspect.iscoroutinefunction(file_read)
    assert not inspect.iscoroutinefunction(safe_calculator)
