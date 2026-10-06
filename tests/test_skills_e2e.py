"""End-to-end skill activation through a compiled graph.

A scripted node plays the model: it issues ``activate_skill`` /
``read_skill_resource`` tool calls, and the real ToolNode executes them. This
checks that activations are recorded on the graph state for single and
parallel tool calls, and that bundled files are returned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.skills import SkillConfig, SkillsRegistry
from tenxgraph.core.skills.activation import (
    get_active_skills,
    make_activate_skill_tool,
    make_read_skill_resource_tool,
    skill_content_marker,
)
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import CallbackManager, InvocationType
from tenxgraph.utils.constants import END


def _write_skill(root: Path, name: str, files: dict[str, str] | None = None) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: The {name} skill.\n---\n{name} instructions",
        encoding="utf-8",
    )
    for rel_path, content in (files or {}).items():
        target = skill_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _tool_call(call_id: str, name: str, **args) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def get_time() -> str:
    """Return the current time."""
    return "12:00"


def _build_tool_node(tmp_path: Path) -> ToolNode:
    _write_skill(tmp_path, "alpha", files={"scripts/run.py": "print('alpha')"})
    _write_skill(tmp_path, "beta")
    registry = SkillsRegistry()
    registry.discover(str(tmp_path))
    config = SkillConfig(skills_dir=str(tmp_path))
    return ToolNode(
        [
            make_activate_skill_tool(registry, config, embed_catalog=False, can_read_files=True),
            make_read_skill_resource_tool(registry, config),
            get_time,
        ]
    )


def _build_app(tmp_path: Path, calls: list[dict]):
    tool_node = _build_tool_node(tmp_path)

    def scripted_model(state: AgentState) -> Message:
        if state.context and state.context[-1].role == "tool":
            return Message.text_message("done", role="assistant")
        message = Message.text_message("using skills", role="assistant")
        message.tools_calls = calls
        return message

    def route(state: AgentState) -> str:
        last = state.context[-1]
        return "TOOL" if getattr(last, "tools_calls", None) else END

    graph = StateGraph()
    graph.add_node("MAIN", scripted_model)
    graph.add_node("TOOL", tool_node)
    graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")
    return graph.compile()


async def _run(app) -> AgentState:
    result = await app.ainvoke(
        {"messages": [Message.text_message("hi")]},
        config={"thread_id": "t1"},
        response_granularity="full",
    )
    return result["state"]


@pytest.mark.asyncio
async def test_single_activation_is_recorded(tmp_path: Path):
    app = _build_app(tmp_path, [_tool_call("c1", "activate_skill", skill_name="alpha")])
    state = await _run(app)

    assert get_active_skills(state) == ["alpha"]
    tool_texts = [m.text() for m in state.context if m.role == "tool"]
    assert any(skill_content_marker("alpha") in text for text in tool_texts)
    assert any("<file>scripts/run.py</file>" in text for text in tool_texts)


@pytest.mark.asyncio
async def test_parallel_activations_are_all_recorded(tmp_path: Path):
    app = _build_app(
        tmp_path,
        [
            _tool_call("c1", "activate_skill", skill_name="alpha"),
            _tool_call("c2", "activate_skill", skill_name="beta"),
            _tool_call("c3", "read_skill_resource", skill_name="alpha", path="scripts/run.py"),
        ],
    )
    state = await _run(app)

    assert sorted(get_active_skills(state)) == ["alpha", "beta"]
    tool_texts = [m.text() for m in state.context if m.role == "tool"]
    assert any("print('alpha')" in text for text in tool_texts)


@pytest.mark.asyncio
async def test_skill_tools_fire_skill_callbacks(tmp_path: Path):
    seen: list[tuple[InvocationType, str]] = []

    def record(context, input_data):
        seen.append((context.invocation_type, context.function_name))
        return input_data

    manager = CallbackManager()
    manager.register_before_invoke(InvocationType.SKILL, record)
    manager.register_before_invoke(InvocationType.TOOL, record)

    tool_node = _build_tool_node(tmp_path)
    calls = [
        ("activate_skill", {"skill_name": "alpha"}),
        ("read_skill_resource", {"skill_name": "alpha", "path": "scripts/run.py"}),
        ("get_time", {}),
    ]
    for index, (name, args) in enumerate(calls):
        await tool_node.invoke(name, args, f"c{index}", {}, AgentState(), callback_manager=manager)

    assert seen == [
        (InvocationType.SKILL, "activate_skill"),
        (InvocationType.SKILL, "read_skill_resource"),
        (InvocationType.TOOL, "get_time"),
    ]
