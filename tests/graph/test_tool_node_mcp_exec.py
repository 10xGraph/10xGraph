from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tenxgraph.core.graph.tool_node.mcp_exec import MCPMixin
from tenxgraph.core.state import Message


class _Node(MCPMixin):
    def __init__(self, client=None, pass_user=False):
        self._client = client
        self.mcp_tools = []
        self._pass_user_info_to_mcp = pass_user


class _Client:
    def __init__(self, ping_ok=True, tools=None, call_result=None):
        self._ping_ok = ping_ok
        self._tools = tools or []
        self._call_result = call_result
        self.called_with = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def ping(self):
        return self._ping_ok

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, args):
        self.called_with = (name, args)
        return self._call_result


class _MCPTool:
    def __init__(self, name, tags=None):
        self.name = name
        self.description = f"{name} description"
        self.inputSchema = {"type": "object", "properties": {}}
        self.meta = {"_fastmcp": {"tags": tags or []}}


class _ResWithModelDump:
    def __init__(self):
        self.content = [self]

    def model_dump(self):
        return {
            "type": "resource",
            "resource": {"uri": Path("/tmp/mcp.txt")},
        }


def _callback_manager():
    cb = SimpleNamespace()
    cb.execute_before_invoke = AsyncMock(side_effect=lambda ctx, data: data)
    cb.execute_after_invoke = AsyncMock(side_effect=lambda ctx, data, out: out)
    cb.execute_on_error = AsyncMock(return_value=None)
    return cb


def test_serialize_result_uses_content_and_normalizes_uri():
    node = _Node()
    blocks = node._serialize_result("tc-1", _ResWithModelDump())

    dumped = blocks[0].output[0]
    assert dumped["type"] == "resource"
    assert dumped["resource"]["uri"] == "/tmp/mcp.txt"


def test_serialize_result_falls_back_to_string_representation():
    node = _Node()
    blocks = node._serialize_result("tc-2", object())
    assert blocks[0].output[0]["type"] == "fallback"


@pytest.mark.asyncio
async def test_get_mcp_tool_returns_empty_without_client():
    node = _Node(client=None)
    tools = await node._get_mcp_tool()
    assert tools == []


@pytest.mark.asyncio
async def test_get_mcp_tool_filters_by_tags_and_tracks_names():
    client = _Client(
        ping_ok=True,
        tools=[_MCPTool("a", tags=["core"]), _MCPTool("b", tags=["extra"])],
    )
    node = _Node(client=client)

    tools = await node._get_mcp_tool(tags={"core"})

    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "a"
    assert node.mcp_tools == ["a"]


@pytest.mark.asyncio
async def test_mcp_execute_returns_error_message_when_client_missing():
    node = _Node(client=None)
    cb = _callback_manager()

    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        result = await node._mcp_execute(
            name="tool",
            args={"x": 1},
            tool_call_id="tc-3",
            config={},
            callback_mgr=cb,
        )

    assert isinstance(result, Message)
    assert "No MCP client configured" in result.text()


@pytest.mark.asyncio
async def test_mcp_execute_calls_tool_and_injects_user_info_when_enabled():
    client = _Client(ping_ok=True, call_result=SimpleNamespace(content={"ok": True}))
    node = _Node(client=client, pass_user=True)
    cb = _callback_manager()

    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        result = await node._mcp_execute(
            name="tool",
            args={"x": 1},
            tool_call_id="tc-4",
            config={"user_id": "u-1"},
            callback_mgr=cb,
        )

    assert isinstance(result, Message)
    assert client.called_with is not None
    _, passed_args = client.called_with
    assert passed_args["x"] == 1
    assert passed_args["user"] == {"user_id": "u-1"}
