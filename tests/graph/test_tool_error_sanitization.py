"""Unexpected tool exceptions reach the model and events as a generic message only."""

import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from tenxgraph.core.exceptions import NodeTimeoutError
from tenxgraph.core.graph import ToolNode
from tenxgraph.core.graph.tool_node.mcp_exec import MCPMixin
from tenxgraph.core.graph.utils.invoke_node_handler import InvokeNodeHandler
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ErrorBlock, ToolResultBlock
from tenxgraph.utils import CallbackManager


DSN = "postgres://user:pw@db/x"
LOGGER_NAME = "tenxgraph.graph.tool_node"


def _texts(msg: Message) -> list[str]:
    out = [b.output for b in msg.content if isinstance(b, ToolResultBlock)]
    out += [b.message for b in msg.content if isinstance(b, ErrorBlock)]
    return [str(o) for o in out]


def _reference(texts: list[str]) -> str:
    match = re.search(r"reference ([0-9a-f]+)", texts[0])
    assert match, texts[0]
    return match.group(1)


def _assert_generic(msg: Message, name: str, events: list[str], caplog) -> None:
    texts = _texts(msg)
    assert len(texts) == 2 and texts[0] == texts[1]
    ref = _reference(texts)
    assert texts[0] == f"Tool '{name}' failed with an internal error (reference {ref})."
    assert DSN not in " ".join(texts)
    assert all(DSN not in e for e in events)
    assert any(ref in e for e in events)

    records = [r for r in caplog.records if ref in r.getMessage()]
    assert records
    record = records[0]
    assert record.exc_info and DSN in str(record.exc_info[1])
    assert name in record.getMessage() and "call_1" in record.getMessage()
    block = next(b for b in msg.content if isinstance(b, ToolResultBlock))
    assert block.is_error is True and block.status == "failed"


def _capture(module: str, events: list[str]):
    def _record(event):
        events.append(repr(event.data) + repr(event.metadata) + repr(event.content))

    return patch(f"tenxgraph.core.graph.tool_node.{module}.publish_event", side_effect=_record)


class TestLocalTool:
    @pytest.mark.asyncio
    async def test_unexpected_exception_is_generic(self, caplog):
        def boom(x: int) -> str:
            raise RuntimeError(f"cannot connect to {DSN}")

        events: list[str] = []
        caplog.set_level(logging.ERROR, logger=LOGGER_NAME)
        with _capture("local_exec", events):
            result = await ToolNode([boom]).invoke(
                name="boom",
                args={"x": 1},
                tool_call_id="call_1",
                config={},
                state=AgentState(),
                callback_manager=CallbackManager(),
            )

        _assert_generic(result, "boom", events, caplog)

    @pytest.mark.asyncio
    async def test_argument_errors_reach_the_model(self):
        class Point(BaseModel):
            x: int

        def needs(a: int, p: Point | None = None) -> str:
            return "ok"

        node = ToolNode([needs])
        for args, expected in (
            ({}, "Missing required parameter 'a' for function 'needs'"),
            ({"a": 1, "p": {"x": "nope"}}, "Invalid argument 'p' for tool 'needs'"),
        ):
            result = await node.invoke(
                name="needs",
                args=args,
                tool_call_id="call_1",
                config={},
                state=AgentState(),
                callback_manager=CallbackManager(),
            )
            texts = _texts(result)
            assert texts[0] == texts[1]
            assert texts[0].startswith("Internal execution error: ")
            assert expected in texts[0]
            assert "internal error (reference" not in texts[0]

    @pytest.mark.asyncio
    async def test_tool_not_found_is_unchanged(self):
        result = await ToolNode([]).invoke(
            name="ghost",
            args={},
            tool_call_id="call_1",
            config={},
            state=AgentState(),
            callback_manager=CallbackManager(),
        )
        assert "Tool 'ghost' not found." in _texts(result)[0]


class _Client:
    def __init__(self, error):
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def ping(self):
        return True

    async def call_tool(self, name, args):
        raise self._error


class _MCPNode(MCPMixin):
    def __init__(self, client):
        self._client = client
        self.mcp_tools = []
        self._funcs = {}
        self.remote_tool_names = []
        self._pass_user_info_to_mcp = False


class TestMCPTool:
    @pytest.mark.asyncio
    async def test_unexpected_exception_is_generic(self, caplog):
        node = _MCPNode(_Client(RuntimeError(f"upstream refused {DSN}")))
        cb = SimpleNamespace(
            execute_before_invoke=AsyncMock(side_effect=lambda ctx, data: data),
            execute_after_invoke=AsyncMock(side_effect=lambda ctx, data, out: out),
            execute_on_error=AsyncMock(return_value=None),
        )
        events: list[str] = []
        caplog.set_level(logging.ERROR, logger=LOGGER_NAME)
        with _capture("mcp_exec", events):
            result = await node._mcp_execute(
                name="remote",
                args={"q": 1},
                tool_call_id="call_1",
                config={},
                callback_mgr=cb,
            )

        _assert_generic(result, "remote", events, caplog)


class TestNodeHandlerCatchAll:
    @pytest.mark.asyncio
    async def test_exception_escaping_a_tool_is_generic(self, caplog):
        msg = Message.text_message("calling", role="assistant")
        msg.tools_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]
        handler = InvokeNodeHandler("tools", ToolNode([]))
        caplog.set_level(logging.ERROR, logger="tenxgraph.graph")
        with patch.object(handler, "_handle_single_tool", AsyncMock(side_effect=RuntimeError(DSN))):
            result = await handler._call_tools(msg, AgentState(context=[msg]), {})

        text = _texts(result[0])[0]
        assert DSN not in text
        ref = _reference([text])
        assert text == f"Tool 't' failed with an internal error (reference {ref})."
        assert any(ref in r.getMessage() and DSN in str(r.exc_info[1]) for r in caplog.records)

    @pytest.mark.asyncio
    async def test_timeout_message_is_kept(self):
        msg = Message.text_message("calling", role="assistant")
        msg.tools_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]
        handler = InvokeNodeHandler("tools", ToolNode([]))
        err = NodeTimeoutError(message="Tool 't' exceeded its timeout of 1s and was cancelled")
        with patch.object(handler, "_handle_single_tool", AsyncMock(side_effect=err)):
            result = await handler._call_tools(msg, AgentState(context=[msg]), {})

        assert "exceeded its timeout of 1s" in _texts(result[0])[0]
