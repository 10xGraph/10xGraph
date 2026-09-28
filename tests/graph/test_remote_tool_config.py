"""Remote tool schema validation and static graph attachment."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentflow.core.graph import CompiledGraph, Node, RemoteToolConfig, ToolNode


def _compiled_with_tool_nodes(*names: str) -> CompiledGraph:
    graph = CompiledGraph.__new__(CompiledGraph)
    graph._state_graph = SimpleNamespace(
        nodes={name: Node(name, ToolNode([])) for name in names}
    )
    return graph


def test_remote_tool_config_accepts_node_aliases_and_normalizes_parameters():
    by_alias = RemoteToolConfig(
        node="tools",
        name="read_clipboard",
        description="Read clipboard text.",
        parameters={"type": "object"},
    )
    by_field_name = RemoteToolConfig(
        node_name="tools",
        name="read_clipboard",
        description="Read clipboard text.",
    )

    assert by_alias.node_name == "tools"
    assert by_field_name.parameters == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_remote_tool_config_rejects_unknown_keys():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RemoteToolConfig.model_validate(
            {
                "nod": "tools",
                "name": "read_clipboard",
                "description": "Read clipboard text.",
            }
        )


def test_attach_remote_tools_groups_typed_configs_by_node():
    graph = _compiled_with_tool_nodes("browser", "desktop")

    graph.attach_remote_tools(
        [
            RemoteToolConfig(
                node="browser",
                name="read_clipboard",
                description="Read clipboard text.",
            ),
            RemoteToolConfig(
                node="desktop",
                name="write_report",
                description="Write a report.",
            ),
        ]
    )

    browser = graph._state_graph.nodes["browser"].func
    desktop = graph._state_graph.nodes["desktop"].func
    assert browser.remote_tool_names == ["read_clipboard"]
    assert desktop.remote_tool_names == ["write_report"]


def test_attach_remote_tools_rejects_duplicate_name_within_node():
    graph = _compiled_with_tool_nodes("tools")
    tool = {
        "node": "tools",
        "name": "read_clipboard",
        "description": "Read clipboard text.",
    }

    with pytest.raises(ValueError, match="Duplicate remote tool name"):
        graph.attach_remote_tools([tool, tool])
