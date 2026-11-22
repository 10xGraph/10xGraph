"""Prebuilt tools for agentflow graphs.

Kept: ``create_handoff_tool``, ``is_handoff_tool``, ``make_agent_memory_tool`` and
``make_user_memory_tool``.

Deprecated, will be removed in 2.0 (calling them emits a ``DeprecationWarning``):
``safe_calculator``, ``fetch_url``, ``create_fetch_url``, ``google_web_search``,
``create_google_web_search``, ``vertex_ai_search``, ``create_vertex_ai_search``,
``file_read``, ``file_write``, ``file_search`` and ``create_file_tools``. Write your own
``@tool`` or use an MCP server.
"""

from .calculator import safe_calculator
from .fetch import create_fetch_url, fetch_url
from .files import create_file_tools, file_read, file_search, file_write
from .handoff import create_handoff_tool, is_handoff_tool
from .memory import make_agent_memory_tool, make_user_memory_tool
from .search import (
    create_google_web_search,
    create_vertex_ai_search,
    google_web_search,
    vertex_ai_search,
)


__all__ = [
    "create_fetch_url",
    "create_file_tools",
    "create_google_web_search",
    "create_vertex_ai_search",
    "create_handoff_tool",
    "fetch_url",
    "file_read",
    "file_search",
    "file_write",
    "google_web_search",
    "is_handoff_tool",
    "make_agent_memory_tool",
    "make_user_memory_tool",
    "safe_calculator",
    "vertex_ai_search",
]
