"""Dependency flags and optional imports for ToolNode.

This module manages optional third-party dependencies for the ToolNode implementation,
providing clean import handling and feature flags. It isolates optional imports
to prevent ImportError cascades when optional dependencies are not installed.

The module handles two main optional dependency groups:
1. MCP (Model Context Protocol) support via 'fastmcp' and 'mcp' packages
2. Future extensibility for other optional tool providers

By centralizing optional imports here, other modules can safely import the
flags and types without triggering ImportError exceptions, allowing graceful
degradation when optional features are not available.

Typical usage:
    ```python
    from .deps import HAS_FASTMCP, HAS_MCP, Client

    if HAS_FASTMCP and HAS_MCP:
        # Use MCP functionality
        client = Client(...)
    else:
        # Graceful fallback or error message
        client = None
    ```


FastMCP integration support.

HAS_FASTMCP: Boolean flag indicating whether FastMCP is available.
    True if 'fastmcp' package is installed and imports successfully.

Client: FastMCP Client class for connecting to MCP servers.
    None if FastMCP is not available.

CallToolResult: Result type for MCP tool executions.
    None if FastMCP is not available.

"""

from __future__ import annotations


try:  # Optional fastmcp
    from fastmcp import Client  # type: ignore
    from fastmcp.client.client import CallToolResult  # type: ignore

    HAS_FASTMCP = True
except Exception:  # pragma: no cover - optional
    HAS_FASTMCP = False
    Client = None  # type: ignore
    CallToolResult = None  # type: ignore


try:  # Optional MCP protocol
    from mcp import Tool  # type: ignore
    from mcp.types import ContentBlock  # type: ignore

    HAS_MCP = True
except Exception:  # pragma: no cover - optional
    HAS_MCP = False
    Tool = None  # type: ignore
    ContentBlock = None  # type: ignore


__all__ = ["HAS_FASTMCP", "HAS_MCP", "CallToolResult", "Client", "ContentBlock", "Tool"]
