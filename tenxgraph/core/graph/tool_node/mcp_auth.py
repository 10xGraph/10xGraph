"""Per-call MCP authentication: headers computed for each list and call, per user.

``ToolNode(client=..., mcp_auth=fn)`` calls ``fn(MCPAuthContext)`` before every MCP tool
listing and tool call and sends the returned headers with that request only. The headers sit
in a ContextVar for the duration of the call; :class:`MCPCallAuth`, installed on the client's
HTTP transport, adds them to each request the call makes. mcp 2.x (fastmcp 4) sends every
POST with the contextvars of the task that made the call, so one shared client serves many
users at once without mixing their headers. mcp 1.x does not, hence the fastmcp 4 floor.
"""

from __future__ import annotations

import contextvars
import typing as t
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass

from tenxgraph.core.authz import Identity, UserCredential, get_credential, get_identity


try:  # fastmcp 4 transports use httpx2
    import httpx2

    _AuthBase: type = httpx2.Auth
except ImportError:  # pragma: no cover - optional
    httpx2 = None  # type: ignore[assignment]
    _AuthBase = object


@dataclass(frozen=True)
class MCPAuthContext:
    """What ``mcp_auth`` knows about the MCP request it is asked to authenticate.

    Attributes:
        identity: The caller from ``config["user"]``, or None.
        credential: The caller's credential from ``config["credential"]``, or None. Its repr
            is redacted, so printing the context never shows the token.
        tool_name: The server's tool name, or None when the tool list is being fetched.
        server: The server URL of the client's transport, when known.
        thread_id: The run's thread id, or None.
        run_id: The run id, or None.
    """

    identity: Identity | None
    credential: UserCredential | None
    tool_name: str | None
    server: str | None
    thread_id: str | None
    run_id: str | None


# mcp 2.x (fastmcp 4) is the first to send each request with the caller's contextvars.
_MIN_FASTMCP_MAJOR = 4

MCPAuth = Callable[[MCPAuthContext], Awaitable[dict[str, str]]]

# Headers for the MCP call running in this context; None outside one.
mcp_call_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "tenxgraph_mcp_call_headers", default=None
)


async def forward_credential(ctx: MCPAuthContext) -> dict[str, str]:
    """``mcp_auth`` that sends the caller's own credential as ``Authorization``.

    Only for an MCP server that accepts the same token (same audience). Sending a token to
    a server it was not issued for is token passthrough, which the MCP spec forbids; use a
    token exchange or a service token in your own ``mcp_auth`` instead. Returns no headers
    when the run has no credential.
    """
    if ctx.credential is None:
        return {}
    return {"Authorization": f"{ctx.credential.scheme} {ctx.credential.token.get_secret_value()}"}


class MCPCallAuth(_AuthBase):  # type: ignore[misc]
    """``httpx2.Auth`` that adds the current MCP call's headers to each request.

    An existing transport ``auth=`` (a fixed M2M token, OAuth) still runs underneath; the
    per-call headers are set on every request it yields, so they win on the same header name.
    Async only: fastmcp transports use an async client.
    """

    def __init__(self, inner: t.Any = None) -> None:
        self.inner = inner

    def _apply(self, request: t.Any) -> t.Any:
        headers = mcp_call_headers.get()
        if headers:
            request.headers.update(headers)
        return request

    async def async_auth_flow(self, request: t.Any) -> AsyncGenerator[t.Any, t.Any]:
        if self.inner is None:
            yield self._apply(request)
            return
        flow = self.inner.async_auth_flow(request)
        try:
            request = await flow.__anext__()
            while True:
                response = yield self._apply(request)
                request = await flow.asend(response)
        except StopAsyncIteration:
            return


def install_mcp_auth(client: t.Any) -> str | None:
    """Wrap the client's transport ``auth`` in :class:`MCPCallAuth`. Returns the server URL.

    Raises:
        ImportError: fastmcp is older than 4 (its mcp 1.x would mix users' headers).
        ValueError: the transport cannot carry HTTP headers (stdio, in-memory, multi-server
            config).
    """
    import fastmcp  # noqa: PLC0415 - only reached with an MCP client

    major = int(str(getattr(fastmcp, "__version__", "0")).split(".")[0] or 0)
    if major < _MIN_FASTMCP_MAJOR or httpx2 is None:
        raise ImportError(
            "mcp_auth needs fastmcp>=4.1.7 (mcp 2.x sends each request with the caller's "
            f"context; fastmcp {getattr(fastmcp, '__version__', '?')} would mix users' "
            "headers). Upgrade with: pip install -U '10xgraph[mcp]'"
        )
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport  # noqa: PLC0415

    transport = getattr(client, "transport", None)
    if not isinstance(transport, StreamableHttpTransport | SSETransport):
        raise ValueError(
            "mcp_auth needs an HTTP MCP transport (streamable HTTP or SSE); "
            f"{type(transport).__name__} cannot send per-call headers."
        )
    if not isinstance(transport.auth, MCPCallAuth):
        transport.auth = MCPCallAuth(transport.auth)
    return str(transport.url)


async def resolve_mcp_headers(
    mcp_auth: MCPAuth,
    config: dict[str, t.Any] | None,
    tool_name: str | None,
    server: str | None,
) -> dict[str, str]:
    """Call ``mcp_auth`` for one MCP request and check what it returned."""
    config = config or {}
    ctx = MCPAuthContext(
        identity=get_identity(config),
        credential=get_credential(config),
        tool_name=tool_name,
        server=server,
        thread_id=config.get("thread_id"),
        run_id=config.get("run_id"),
    )
    headers = await mcp_auth(ctx)
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise TypeError("mcp_auth must return a dict of header names to string values")
    return headers
