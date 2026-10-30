"""File writes, fetch host allow-list and bound search tools (G-53, G-54)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tenxgraph.prebuilt.tools import (
    create_fetch_url,
    create_file_tools,
    create_google_web_search,
    create_vertex_ai_search,
)
from tenxgraph.prebuilt.tools import files as files_mod
from tenxgraph.prebuilt.tools import search as search_mod


def _tools(root):
    return {fn.__name__: fn for fn in create_file_tools(root)}


def test_overwrite_is_atomic_when_the_write_fails(tmp_path):
    (tmp_path / "a.txt").write_text("original")
    tools = _tools(tmp_path)
    with patch.object(files_mod.os, "fdopen", side_effect=OSError("disk full")):
        reply = json.loads(tools["scoped_file_write"]("a.txt", "new", mode="overwrite"))
    assert "error" in reply
    assert (tmp_path / "a.txt").read_text() == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]  # no temp file left behind


def test_overwrite_replaces_content_and_keeps_mode(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("original")
    target.chmod(0o640)
    tools = _tools(tmp_path)
    assert json.loads(tools["scoped_file_write"]("a.txt", "new", mode="overwrite"))["status"]
    assert target.read_text() == "new"
    assert (target.stat().st_mode & 0o777) == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]


def test_create_is_exclusive(tmp_path):
    target = tmp_path / "a.txt"
    # Simulate a file that appears after the exists() check.
    with pytest.raises(FileExistsError):
        target.write_text("other")
        files_mod._write_file_content(target, "mine", "create")
    assert target.read_text() == "other"


def test_os_error_text_does_not_leak_paths(tmp_path):
    tools = _tools(tmp_path)
    (tmp_path / "d").mkdir()
    with patch.object(Path, "read_text", side_effect=PermissionError(f"{tmp_path}/secret")):
        (tmp_path / "f.txt").write_text("x")
        reply = json.loads(tools["scoped_file_read"]("f.txt"))
    assert str(tmp_path) not in json.dumps(reply)


@pytest.mark.asyncio
async def test_fetch_allow_list_blocks_other_hosts_without_a_request():
    tool = create_fetch_url(["docs.example.com", "*.allowed.org"])
    with patch("tenxgraph.prebuilt.tools.fetch._resolve_public_ip") as resolve:
        reply = json.loads(await tool("http://evil.example.net/x"))
    assert "error" in reply
    resolve.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_allow_list_lets_listed_host_reach_resolution():
    tool = create_fetch_url(["*.allowed.org"])
    with patch("tenxgraph.prebuilt.tools.fetch._resolve_public_ip", return_value=None) as resolve:
        await tool("http://a.allowed.org/x")
    resolve.assert_called()


def test_fetch_requires_hosts():
    with pytest.raises(ValueError):
        create_fetch_url([])


@pytest.mark.asyncio
async def test_bound_search_tools_hide_datastore_and_model_from_the_model():
    vertex = create_vertex_ai_search("projects/p/datastores/d", model="gemini-x")
    web = create_google_web_search(model="gemini-y")
    import inspect

    assert "datastore" not in inspect.signature(vertex).parameters
    assert "model" not in inspect.signature(vertex).parameters
    assert "model" not in inspect.signature(web).parameters

    with patch.object(search_mod, "_vertex_ai_search_sync", return_value={"ok": 1}) as sync:
        await vertex("q")
    assert sync.call_args.args[1] == "projects/p/datastores/d"
    assert sync.call_args.args[2] == "gemini-x"


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("file_read", ("x.txt",)),
        ("file_write", ("x.txt", "hi")),
        ("file_search", ("hi",)),
    ],
)
def test_module_level_file_tools_warn_and_name_the_factory(name, args, tmp_path, monkeypatch):
    from tenxgraph.prebuilt import tools as tools_pkg

    monkeypatch.chdir(tmp_path)
    with pytest.warns(DeprecationWarning, match="create_file_tools"):
        getattr(tools_pkg, name)(*args)
