"""ToolNode against a real FastMCP server over streamable HTTP, with the default client mode.

fastmcp 4.x negotiates MCP protocol 2026-07-28 by default, which has no ``ping`` method, so a
ToolNode that pinged before each call failed every call there. This runs with whichever fastmcp
is installed (3.x and 4.x are both supported).
"""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import patch

import pytest
import pytest_asyncio


fastmcp = pytest.importorskip("fastmcp")
uvicorn = pytest.importorskip("uvicorn")

from fastmcp import Context, FastMCP
from fastmcp.client.transports import StreamableHttpTransport

from tenxgraph.core.authz import Identity, Reach
from tenxgraph.core.graph.tool_node import ToolNode

from .test_tool_node_mcp_exec import _callback_manager


def _server() -> FastMCP:
    mcp = FastMCP("demo")

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @mcp.tool
    def whoami(ctx: Context) -> dict:
        """Return the user metadata the client sent."""
        meta = ctx.request_context.meta  # a dict on fastmcp 4.x, a pydantic model on 3.x
        if meta is not None and not isinstance(meta, dict):
            meta = meta.model_dump()
        return {"user": (meta or {}).get("user")}

    return mcp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture
async def mcp_url():
    port = _free_port()
    config = uvicorn.Config(
        _server().http_app(path="/mcp"), host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_tool_node_lists_and_calls_over_streamable_http(mcp_url):
    client = fastmcp.Client(StreamableHttpTransport(mcp_url))
    node = ToolNode([], client=client)

    names = [x["function"]["name"] for x in await node._get_mcp_tool()]
    assert {"add", "whoami"} <= set(names)

    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        result = await node._mcp_execute("add", {"a": 2, "b": 3}, "tc-1", {}, _callback_manager())
    block = result.content[0]
    assert not block.is_error, result.text()
    assert "5" in result.text()


@pytest.mark.asyncio
async def test_identity_reaches_mcp_server_over_streamable_http(mcp_url):
    client = fastmcp.Client(StreamableHttpTransport(mcp_url))
    node = ToolNode([], client=client, pass_user_info_to_mcp=True)
    await node._get_mcp_tool()
    identity = Identity(
        user_id="alice", tenant={"org_id": "acme"}, roles=frozenset({"member"}), reach=Reach.OWN
    )

    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        result = await node._mcp_execute(
            "whoami", {}, "tc-2", {"user": identity, "user_id": "alice"}, _callback_manager()
        )
    text = result.text()
    assert not result.content[0].is_error, text
    assert "acme" in text and "alice" in text and "member" in text
