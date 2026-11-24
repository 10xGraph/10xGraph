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

import importlib
import importlib.util
from typing import Any


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# Only the flags are computed here. ``Client``, ``CallToolResult``, ``Tool`` and
# ``ContentBlock`` are imported on first access (``__getattr__`` below): importing
# fastmcp at module load costs about half a second at startup.
HAS_FASTMCP = _installed("fastmcp")
HAS_MCP = _installed("mcp")

_LAZY = {
    "Client": ("fastmcp", "Client"),
    "CallToolResult": ("fastmcp.client.client", "CallToolResult"),
    "Tool": ("mcp", "Tool"),
    "ContentBlock": ("mcp.types", "ContentBlock"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _LAZY[name]
    try:
        value = getattr(importlib.import_module(module), attr)
    except Exception:  # optional dependency missing or broken
        value = None
    globals()[name] = value
    return value


__all__ = ["HAS_FASTMCP", "HAS_MCP", "CallToolResult", "Client", "ContentBlock", "Tool"]
