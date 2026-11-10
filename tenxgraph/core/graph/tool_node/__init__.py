"""ToolNode package.

This package provides a modularized implementation of ToolNode. Public API:

- ToolNode
- MCPAuthContext, MCPCallAuth, forward_credential (per-call MCP auth, ``mcp_auth=``)
- HAS_FASTMCP, HAS_MCP
- UnsupportedToolParameterError
"""

from tenxgraph.core.state.tool_result import ToolResult

from .base import ToolNode
from .deps import HAS_FASTMCP, HAS_MCP
from .mcp_auth import MCPAuthContext, MCPCallAuth, forward_credential
from .schema import UnsupportedToolParameterError


__all__ = [
    "HAS_FASTMCP",
    "HAS_MCP",
    "MCPAuthContext",
    "MCPCallAuth",
    "ToolNode",
    "ToolResult",
    "UnsupportedToolParameterError",
    "forward_credential",
]
