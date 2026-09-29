"""Client-executed tools supplied per run, through the run config.

Remote tools attached with ``CompiledGraph.attach_remote_tools`` are fixed for the life of the
graph. A frontend (for example an AG-UI client such as CopilotKit) can also bring its own tools
for a single run, without mutating the shared graph::

    await app.ainvoke(
        {"messages": [...]},
        config={
            "thread_id": "t1",
            "remote_tools": [
                {
                    "name": "change_background",
                    "description": "Change the page background color.",
                    "parameters": {"type": "object", "properties": {"color": {"type": "string"}}},
                }
            ],
        },
    )

The model sees them next to the ``ToolNode``'s own tools for that run only. A call to one of them
is not executed on the server: the ``ToolNode`` hands it back to the client exactly like a
configured remote tool, and the run resumes when the client sends the result. A per-run tool
never shadows a server tool: a name the ``ToolNode`` already has is ignored.
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger("agentflow.tool_node")

# Run-config key holding per-run client tool schemas.
RUN_REMOTE_TOOLS_KEY = "remote_tools"


def run_remote_tools(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Per-run client tool schemas from ``config``, in OpenAI function-calling format.

    Accepts OpenAI-style entries (``{"type": "function", "function": {...}}``) and flat entries
    (``{"name", "description", "parameters"}``, the AG-UI ``Tool`` shape). Entries without a
    name are skipped, and a repeated name keeps its first definition.
    """
    raw = (config or {}).get(RUN_REMOTE_TOOLS_KEY) or []
    if not isinstance(raw, list | tuple):
        return []

    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        spec = item.get("function") if isinstance(item.get("function"), dict) else item
        name = spec.get("name")
        if not isinstance(name, str) or not name.strip() or name in seen:
            continue
        seen.add(name)
        parameters = spec.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(spec.get("description") or ""),
                    "parameters": parameters,
                },
            }
        )
    return tools


def tool_name(schema: dict[str, Any]) -> str:
    """The name in an OpenAI function-calling schema."""
    return str((schema.get("function") or {}).get("name") or "")
