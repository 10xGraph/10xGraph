"""ToolResultBlock.name carries the name of the tool that produced the result."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.core.state.message import Message, ToolResultBlock
from tenxgraph.utils import CallbackManager
from tenxgraph.utils.converter import _convert_dict, convert_messages

from .test_tool_node_mcp_exec import _callback_manager, _Client, _Node


def echo(text: str) -> str:
    """Echo."""
    return text


def boom() -> str:
    """Always fails."""
    raise RuntimeError("nope")


def _result_blocks(message: Message) -> list[ToolResultBlock]:
    return [b for b in message.content if isinstance(b, ToolResultBlock)]


async def _invoke(node: ToolNode, name: str, args: dict) -> Message:
    return await node.invoke(
        name,
        args,
        tool_call_id="c1",
        config={},
        state=AgentState(),
        callback_manager=CallbackManager(),
    )


@pytest.mark.asyncio
async def test_local_tool_result_has_name():
    result = await _invoke(ToolNode([echo]), "echo", {"text": "hi"})
    blocks = _result_blocks(result)
    assert blocks
    assert all(b.name == "echo" for b in blocks)


@pytest.mark.asyncio
async def test_raising_tool_result_has_name_and_is_error():
    result = await _invoke(ToolNode([boom]), "boom", {})
    blocks = _result_blocks(result)
    assert blocks
    assert all(b.name == "boom" and b.is_error for b in blocks)


@pytest.mark.asyncio
async def test_unknown_tool_result_has_name_and_is_error():
    result = await _invoke(ToolNode([echo]), "missing", {})
    blocks = _result_blocks(result)
    assert blocks
    assert all(b.name == "missing" and b.is_error for b in blocks)


@pytest.mark.asyncio
async def test_mcp_tool_result_has_name():
    client = _Client(call_result=SimpleNamespace(content={"ok": True}))
    node = _Node(client=client)
    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        result = await node._mcp_execute("search", {}, "tc", {}, _callback_manager())
    blocks = _result_blocks(result)
    assert blocks
    assert all(b.name == "search" for b in blocks)


@pytest.mark.asyncio
async def test_mcp_without_client_result_has_name():
    node = _Node(client=None)
    with patch("tenxgraph.core.graph.tool_node.mcp_exec.publish_event"):
        result = await node._mcp_execute("search", {}, "tc", {}, _callback_manager())
    blocks = _result_blocks(result)
    assert blocks
    assert all(b.name == "search" and b.is_error for b in blocks)


def test_result_saved_without_name_loads_as_none():
    saved = {
        "message_id": "m1",
        "role": "tool",
        "content": [{"type": "tool_result", "call_id": "c1", "output": "x"}],
    }
    block = _result_blocks(Message.model_validate(saved))[0]
    assert block.name is None
    assert block.call_id == "c1"


def _tool_message(name: str | None) -> Message:
    return Message.tool_message(
        [ToolResultBlock(call_id="c1", name=name, output="42")], message_id="t1"
    )


def test_wire_format_ignores_name():
    assert _convert_dict(_tool_message("echo")) == _convert_dict(_tool_message(None))
    assert _convert_dict(_tool_message("echo")) == {
        "role": "tool",
        "content": "42",
        "tool_call_id": "c1",
    }
    named = convert_messages(system_prompts=[], state=AgentState(context=[_tool_message("echo")]))
    plain = convert_messages(system_prompts=[], state=AgentState(context=[_tool_message(None)]))
    assert named == plain
