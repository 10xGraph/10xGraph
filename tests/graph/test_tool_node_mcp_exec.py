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
        self._funcs = {}
        self.remote_tool_names = []
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

    async def call_tool(self, name, args, *, meta=None):
        self.called_with = (name, args)
        self.meta = meta
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
    assert node.mcp_tools == ["a", "b"]


@pytest.mark.asyncio
async def test_get_mcp_tool_raises_on_local_name_collision():
    node = _Node(client=_Client(tools=[_MCPTool("calc")]))
    node._funcs = {"calc": object()}

    with pytest.raises(ValueError, match="calc"):
        await node._get_mcp_tool()


@pytest.mark.asyncio
async def test_get_mcp_tool_raises_on_remote_name_collision():
    node = _Node(client=_Client(tools=[_MCPTool("geo", tags=["x"])]))
    node.remote_tool_names = ["geo"]

    with pytest.raises(ValueError, match="geo"):
        await node._get_mcp_tool(tags={"other"})


@pytest.mark.asyncio
async def test_mcp_tools_stays_bounded_and_covers_filtered_tools():
    client = _Client(
        tools=[_MCPTool("a", tags=["core"]), _MCPTool("b", tags=["extra"]), _MCPTool("a")]
    )
    node = _Node(client=client)

    for _ in range(50):
        tools = await node._get_mcp_tool(tags={"core"})

    assert [x["function"]["name"] for x in tools] == ["a"]
    assert node.mcp_tools == ["a", "b"]


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
    assert passed_args == {"x": 1}  # user info is request metadata, not a tool argument
    assert client.meta == {"user": {"user_id": "u-1"}}


@pytest.mark.asyncio
async def test_mcp_include_and_exclude_filter_listed_tools():
    client = _Client(tools=[_MCPTool("a"), _MCPTool("b"), _MCPTool("c")])
    node = _Node(client=client)
    node._mcp_include = frozenset({"a", "b"})
    node._mcp_exclude = frozenset({"b"})
    tools = await node._get_mcp_tool()
    assert [t["function"]["name"] for t in tools] == ["a"]
    assert node.mcp_tools == ["a"]  # a hidden tool is not dispatchable either


@pytest.mark.asyncio
async def test_user_info_not_sent_when_client_cannot_take_meta():
    class _OldClient(_Client):
        async def call_tool(self, name, args):
            self.called_with = (name, args)
            return self._call_result

    client = _OldClient(call_result=SimpleNamespace(content={"ok": True}))
    node = _Node(client=client, pass_user=True)
    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        await node._mcp_execute(
            name="tool",
            args={"x": 1},
            tool_call_id="tc-5",
            config={"user_id": "u-1"},
            callback_mgr=_callback_manager(),
        )
    assert client.called_with[1] == {"x": 1}


# G-50 / G-51 / G-43: caps, prefix, listing cache, connect timeout
class _CountingClient(_Client):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.list_calls = 0

    async def list_tools(self):
        self.list_calls += 1
        return self._tools


@pytest.mark.asyncio
async def test_mcp_description_cap_is_opt_in():
    long = _MCPTool("a")
    long.description = "d" * 50
    node = _Node(client=_Client(tools=[long]))
    assert (await node._get_mcp_tool())[0]["function"]["description"] == "d" * 50
    node._mcp_max_description_chars = 10
    assert (await node._get_mcp_tool())[0]["function"]["description"] == "d" * 10 + "..."


def test_mcp_result_cap_is_opt_in():
    res = SimpleNamespace(content=[{"text": "x" * 100}])
    node = _Node()
    assert isinstance(node._serialize_result("c", res)[0].output, list)
    node._mcp_max_result_chars = 20
    out = node._serialize_result("c", res)[0].output
    assert isinstance(out, str) and "output cut" in out and out.startswith('[{"text": "xxx')


@pytest.mark.asyncio
async def test_mcp_prefix_is_exposed_and_stripped_on_call():
    client = _Client(tools=[_MCPTool("search")], call_result=SimpleNamespace(content={"ok": 1}))
    node = _Node(client=client)
    node._mcp_prefix = "gh_"
    tools = await node._get_mcp_tool()
    assert tools[0]["function"]["name"] == "gh_search"
    assert node.mcp_tools == ["gh_search"]
    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        await node._mcp_execute("gh_search", {}, "tc", {}, _callback_manager())
    assert client.called_with[0] == "search"


@pytest.mark.asyncio
async def test_mcp_prefix_lets_server_tool_share_a_local_name():
    node = _Node(client=_Client(tools=[_MCPTool("search")]))
    node._funcs = {"search": lambda: None}
    with pytest.raises(ValueError):
        await node._get_mcp_tool()
    node._mcp_prefix = "gh_"
    assert (await node._get_mcp_tool())[0]["function"]["name"] == "gh_search"


@pytest.mark.asyncio
async def test_mcp_tool_list_cache_and_refresh():
    client = _CountingClient(tools=[_MCPTool("a")])
    node = _Node(client=client)
    node._mcp_cache_tools = True
    await node._get_mcp_tool()
    await node._get_mcp_tool()
    assert client.list_calls == 1
    await node.refresh_mcp_tools()
    assert client.list_calls == 2


@pytest.mark.asyncio
async def test_mcp_failed_listing_is_not_cached():
    client = _CountingClient(ping_ok=False, tools=[_MCPTool("a")])
    node = _Node(client=client)
    node._mcp_cache_tools = True
    assert await node._get_mcp_tool() == []
    client._ping_ok = True
    assert len(await node._get_mcp_tool()) == 1


@pytest.mark.asyncio
async def test_mcp_connect_timeout_bounds_connect_only():
    import asyncio

    class _Slow(_Client):
        async def __aenter__(self):
            await asyncio.sleep(30)

    node = _Node(client=_Slow())
    node._mcp_connect_timeout = 0.05
    with pytest.raises(TimeoutError, match="MCP connect timed out"):
        async with node._mcp_session():
            pass

    # A slow body is not cut.
    node2 = _Node(client=_Client())
    node2._mcp_connect_timeout = 0.05
    async with node2._mcp_session():
        await asyncio.sleep(0.15)


def test_tool_node_mcp_defaults():
    from tenxgraph.core.graph.tool_node import ToolNode

    n = ToolNode([])
    assert n._mcp_connect_timeout == 30.0
    assert n._mcp_cache_tools is True
    assert n._mcp_prefix is None and n._mcp_max_result_chars is None
    assert ToolNode([], mcp_connect_timeout=None)._mcp_connect_timeout is None


@pytest.mark.asyncio
async def test_published_end_event_carries_after_invoke_replacement():
    client = _Client(call_result=SimpleNamespace(content={"secret": "RAW"}))
    node = _Node(client=client)
    cb = _callback_manager()
    replacement = Message.tool_message(content=[], meta={"scrubbed": True})
    cb.execute_after_invoke = AsyncMock(return_value=replacement)

    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event") as pub:
        result = await node._mcp_execute("tool", {}, "tc-9", {}, cb)

    assert result is replacement
    end = pub.call_args.args[0]
    assert end.data["message"] == replacement.model_dump()
    assert "RAW" not in str(end.data)
