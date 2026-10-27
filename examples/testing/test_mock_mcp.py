"""MockMCPClient: a stand-in for an MCP server.

Register tools with handlers, check the MCP protocol surface (list_tools and
call_tool), and assert on recorded calls. It also shows what ToolNode accepts.
Run: pytest examples/testing/test_mock_mcp.py -q
"""

import pytest

from tenxgraph.core.graph import ToolNode
from tenxgraph.qa.testing import MockMCPClient


@pytest.fixture
def mcp() -> MockMCPClient:
    client = MockMCPClient()
    client.add_tool(
        name="search",
        description="Search the web",
        parameters={"query": {"type": "string"}},
        handler=lambda query: f"Results for: {query}",
    )

    async def add(a: int, b: int) -> int:
        return a + b

    client.add_tool(
        name="add",
        description="Add two numbers",
        parameters={"a": {"type": "integer"}, "b": {"type": "integer"}},
        handler=add,
    )
    return client


async def test_list_tools(mcp: MockMCPClient):
    tools = await mcp.list_tools()

    assert [t["name"] for t in tools] == ["search", "add"]
    assert tools[0]["inputSchema"]["properties"] == {"query": {"type": "string"}}


async def test_call_tool_sync_and_async_handlers(mcp: MockMCPClient):
    assert await mcp.call_tool("search", {"query": "10xGraph"}) == "Results for: 10xGraph"
    assert await mcp.call_tool("add", {"a": 2, "b": 3}) == 5  # async handlers are awaited


async def test_unknown_tool_raises(mcp: MockMCPClient):
    with pytest.raises(ValueError, match="Unknown MCP tool"):
        await mcp.call_tool("nope", {})


async def test_call_assertions(mcp: MockMCPClient):
    await mcp.call_tool("search", {"query": "a"})
    await mcp.call_tool("search", {"query": "b"})

    assert mcp.was_called("search")
    assert not mcp.was_called("add")
    assert mcp.call_count("search") == 2
    assert mcp.get_last_call("search") == {"arguments": {"query": "b"}}
    mcp.assert_called("search")
    mcp.assert_called_with("search", query="b")

    mcp.reset()
    assert mcp.call_count("search") == 0


def test_tool_node_accepts_the_mock(mcp: MockMCPClient):
    # ToolNode takes the mock wherever it takes an MCP client (needs the `mcp` extra).
    # Note: ToolNode talks to a real fastmcp client (async context manager, ping(),
    # tool objects with .name). MockMCPClient only mirrors list_tools/call_tool, so
    # use it to test your own code around the client, not ToolNode's MCP execution.
    node = ToolNode([], client=mcp)
    assert node._client is mcp
