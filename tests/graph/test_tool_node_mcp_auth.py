"""ToolNode(mcp_auth=...) against a real FastMCP 4 server over streamable HTTP.

The server's tool echoes the headers it received, so each test sees exactly what went over
the wire. The concurrency test runs two users through ONE shared ToolNode and client with
overlapping calls: each must only ever see its own Authorization header.
"""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import patch

import pytest
import pytest_asyncio


fastmcp = pytest.importorskip("fastmcp")
uvicorn = pytest.importorskip("uvicorn")
if int(fastmcp.__version__.split(".")[0]) < 4:
    pytest.skip("mcp_auth needs fastmcp 4", allow_module_level=True)

import httpx2
from fastmcp import Context, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from fastmcp.server.dependencies import get_http_headers

from tenxgraph.core.authz import Identity, UserCredential
from tenxgraph.core.graph.tool_node import (
    MCPAuthContext,
    MCPCallAuth,
    ToolNode,
    forward_credential,
)
from tenxgraph.core.state import AgentState

from .test_tool_node_mcp_exec import _callback_manager


TOKEN = "tok-mcp-secret-987654321"  # noqa: S105 - test value


def _server() -> FastMCP:
    mcp = FastMCP("auth-demo")

    @mcp.tool
    async def echo_auth(delay: float = 0.0) -> dict:
        """Return the auth headers this request carried."""
        await asyncio.sleep(delay)
        headers = get_http_headers(include_all=True)
        return {"authorization": headers.get("authorization"), "m2m": headers.get("x-m2m")}

    @mcp.tool
    def whoami(ctx: Context) -> dict:
        """Return the user metadata the client sent."""
        meta = ctx.request_context.meta
        if meta is not None and not isinstance(meta, dict):
            meta = meta.model_dump()
        return {"meta": meta}

    return mcp


class _RecordAuthorization:
    """ASGI wrapper recording the Authorization header of every request."""

    def __init__(self, app, seen: list):
        self.app = app
        self.seen = seen

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            self.seen.append(headers.get(b"authorization", b"").decode())
        await self.app(scope, receive, send)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture
async def server():
    port = _free_port()
    seen: list[str] = []
    app = _RecordAuthorization(_server().http_app(path="/mcp"), seen)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    for _ in range(100):
        if srv.started:
            break
        await asyncio.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp", seen
    finally:
        srv.should_exit = True
        await task


class _M2MAuth(httpx2.Auth):
    def auth_flow(self, request):
        request.headers["X-M2M"] = "m2m-token"
        yield request


def _config(user: str, token: str | None) -> dict:
    config = {"thread_id": f"t-{user}", "run_id": f"r-{user}", "user": Identity(user_id=user)}
    if token is not None:
        config["credential"] = UserCredential(token)
    return config


async def _call(node: ToolNode, config: dict, name: str = "echo_auth", args=None) -> str:
    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        result = await node._mcp_execute(name, args or {}, "tc", config, _callback_manager())
    assert not result.content[0].is_error, result.text()
    return result.text()


@pytest.mark.asyncio
async def test_two_users_on_one_shared_client_each_get_their_own_header(server):
    url, _ = server
    client = fastmcp.Client(StreamableHttpTransport(url))
    node = ToolNode([], client=client, mcp_auth=forward_credential)
    await node._get_mcp_tool(config=_config("alice", "alice-token"))

    async def as_user(user: str) -> list[str]:
        config = _config(user, f"{user}-token")
        return [await _call(node, config, args={"delay": 0.02}) for _ in range(5)]

    # Keep the session open across calls so all requests share one MCP session.
    async with client:
        results = await asyncio.gather(*(as_user(u) for u in ["alice", "bob"] * 6))

    for user, texts in zip(["alice", "bob"] * 6, results, strict=True):
        other = "bob" if user == "alice" else "alice"
        for text in texts:
            assert f"Bearer {user}-token" in text
            assert f"{other}-token" not in text


@pytest.mark.asyncio
async def test_existing_transport_auth_still_applies(server):
    url, _ = server
    client = fastmcp.Client(StreamableHttpTransport(url, auth=_M2MAuth()))
    node = ToolNode([], client=client, mcp_auth=forward_credential)
    assert isinstance(client.transport.auth, MCPCallAuth)
    assert isinstance(client.transport.auth.inner, _M2MAuth)
    await node._get_mcp_tool()

    text = await _call(node, _config("alice", TOKEN))
    assert f"Bearer {TOKEN}" in text
    assert "m2m-token" in text


@pytest.mark.asyncio
async def test_listing_runs_mcp_auth_and_sends_its_headers(server):
    url, seen = server
    calls: list[MCPAuthContext] = []

    async def mcp_auth(ctx: MCPAuthContext) -> dict[str, str]:
        calls.append(ctx)
        return {"Authorization": f"Bearer {ctx.tool_name or 'listing'}-{ctx.identity.user_id}"}

    client = fastmcp.Client(StreamableHttpTransport(url))
    node = ToolNode([], client=client, mcp_auth=mcp_auth, mcp_cache_tools=False)
    tools = await node.all_tools(config=_config("carol", None))
    assert {"echo_auth", "whoami"} <= {t["function"]["name"] for t in tools}

    listing = calls[0]
    assert listing.tool_name is None
    assert listing.server == url
    assert listing.thread_id == "t-carol" and listing.run_id == "r-carol"
    assert listing.credential is None
    assert "Bearer listing-carol" in seen

    text = await _call(node, _config("carol", None))
    assert "Bearer echo_auth-carol" in text
    assert calls[-1].tool_name == "echo_auth"


@pytest.mark.asyncio
async def test_without_mcp_auth_nothing_changes(server):
    url, seen = server
    transport = StreamableHttpTransport(url)
    client = fastmcp.Client(transport)
    node = ToolNode([], client=client)
    assert transport.auth is None
    await node._get_mcp_tool()
    text = await _call(node, _config("alice", TOKEN))
    assert TOKEN not in text
    assert all(TOKEN not in h for h in seen)


@pytest.mark.asyncio
async def test_pass_user_info_meta_never_holds_the_credential(server):
    url, seen = server
    client = fastmcp.Client(StreamableHttpTransport(url))
    node = ToolNode([], client=client, pass_user_info_to_mcp=True)
    await node._get_mcp_tool()
    text = await _call(node, _config("alice", TOKEN), name="whoami")
    assert "alice" in text
    assert TOKEN not in text
    assert all(TOKEN not in h for h in seen)


@pytest.mark.asyncio
async def test_mcp_auth_failure_is_a_tool_error(server):
    url, _ = server

    async def broken(ctx):
        return {"Authorization": 42}

    node = ToolNode([], client=fastmcp.Client(StreamableHttpTransport(url)), mcp_auth=broken)
    node.mcp_tools = ["echo_auth"]
    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        result = await node.invoke(
            "echo_auth", {}, "tc", _config("a", TOKEN), AgentState(), _callback_manager()
        )
    assert result.content[0].is_error is True


@pytest.mark.asyncio
async def test_forward_credential():
    ctx = MCPAuthContext(
        identity=None,
        credential=UserCredential(TOKEN, scheme="DPoP"),
        tool_name="t",
        server=None,
        thread_id=None,
        run_id=None,
    )
    assert await forward_credential(ctx) == {"Authorization": f"DPoP {TOKEN}"}
    assert TOKEN not in repr(ctx)
    empty = MCPAuthContext(None, None, None, None, None, None)
    assert await forward_credential(empty) == {}


def test_stdio_and_in_memory_transports_are_refused():
    stdio = fastmcp.Client(StdioTransport(command="python", args=["-c", "pass"]))
    with pytest.raises(ValueError, match="HTTP MCP transport"):
        ToolNode([], client=stdio, mcp_auth=forward_credential)
    with pytest.raises(ValueError, match="HTTP MCP transport"):
        ToolNode([], client=fastmcp.Client(_server()), mcp_auth=forward_credential)


def test_mcp_auth_needs_a_client():
    with pytest.raises(ValueError, match="needs an MCP client"):
        ToolNode([], mcp_auth=forward_credential)


def test_fastmcp_older_than_4_is_refused():
    client = fastmcp.Client(StreamableHttpTransport("http://127.0.0.1:1/mcp"))
    with (
        patch.object(fastmcp, "__version__", "3.3.1"),
        pytest.raises(ImportError, match="fastmcp>=4"),
    ):
        ToolNode([], client=client, mcp_auth=forward_credential)
    assert client.transport.auth is None


def test_installing_twice_does_not_double_wrap():
    client = fastmcp.Client(StreamableHttpTransport("http://127.0.0.1:1/mcp"))
    ToolNode([], client=client, mcp_auth=forward_credential)
    first = client.transport.auth
    ToolNode([], client=client, mcp_auth=forward_credential)
    assert client.transport.auth is first
    assert first.inner is None
