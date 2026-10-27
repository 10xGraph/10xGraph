"""Dispatch guards for ToolNode calls: handoff targets, hidden tools, parallel cap."""

import asyncio

import pytest

from tenxgraph.core.exceptions import GraphError
from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.graph.utils.invoke_node_handler import InvokeNodeHandler
from tenxgraph.core.graph.utils.stream_node_handler import StreamNodeHandler
from tenxgraph.core.graph.utils.tool_dispatch import (
    resolve_max_parallel_tool_calls,
    validate_goto,
)
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ErrorBlock, ToolResultBlock
from tenxgraph.core.state.stream_chunks import StreamEvent
from tenxgraph.prebuilt.tools.handoff import create_handoff_tool
from tenxgraph.utils.command import Command
from tenxgraph.utils.constants import (
    DEFAULT_MAX_PARALLEL_TOOL_CALLS,
    END,
    MAX_PARALLEL_TOOL_CALLS_ENV,
)


def _calls(*names: str) -> list[dict]:
    return [
        {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
        for i, n in enumerate(names)
    ]


def _assistant(*names: str, metadata: dict | None = None) -> Message:
    msg = Message.text_message("calling", role="assistant")
    msg.tools_calls = _calls(*names)
    if metadata:
        msg.metadata = metadata
    return msg


def _tool_result_text(msg: Message) -> str:
    return " ".join(b.output for b in msg.content if isinstance(b, ToolResultBlock))


def _graph(tool_node: ToolNode, assistant: Message) -> StateGraph:
    """START -> model (emits `assistant`) -> TOOL -> END; `admin_node` has no edge in."""
    ran: list[str] = []

    def model(state: AgentState):
        return assistant.model_copy(deep=True, update={"message_id": "assistant_1"})

    def admin_node(state: AgentState):
        ran.append("admin_node")
        return Message.text_message("admin ran", role="assistant")

    graph = StateGraph()
    graph.add_node("MAIN", model)
    graph.add_node("TOOL", tool_node)
    graph.add_node("admin_node", admin_node)
    graph.add_edge("MAIN", "TOOL")
    graph.add_edge("TOOL", END)
    graph.add_edge("admin_node", END)
    graph.set_entry_point("MAIN")
    graph.ran = ran  # type: ignore[attr-defined]
    return graph


async def _astream_texts(compiled, messages) -> list[str]:
    texts = []
    async for chunk in compiled.astream({"messages": messages}, config={"thread_id": "t"}):
        if chunk.message is not None and not chunk.message.delta:
            texts.append(chunk.message.text())
    return texts


class TestHandoffRegistration:
    @pytest.mark.asyncio
    async def test_unregistered_transfer_to_does_not_jump_invoke(self):
        graph = _graph(ToolNode([]), _assistant("transfer_to_admin_node"))
        compiled = graph.compile()
        result = await compiled.ainvoke({"messages": [Message.text_message("hi")]})
        assert graph.ran == []
        tool_msgs = [m for m in result["messages"] if m.role == "tool"]
        assert tool_msgs and "not found" in _tool_result_text(tool_msgs[0])

    @pytest.mark.asyncio
    async def test_unregistered_transfer_to_does_not_jump_stream(self):
        graph = _graph(ToolNode([]), _assistant("transfer_to_admin_node"))
        compiled = graph.compile()
        await _astream_texts(compiled, [Message.text_message("hi")])
        assert graph.ran == []

    @pytest.mark.asyncio
    async def test_registered_handoff_jumps_in_ainvoke(self):
        graph = _graph(
            ToolNode([create_handoff_tool("admin_node")]), _assistant("transfer_to_admin_node")
        )
        compiled = graph.compile()
        result = await compiled.ainvoke({"messages": [Message.text_message("hi")]})
        assert graph.ran == ["admin_node"]
        assert any(m.text() == "admin ran" for m in result["messages"])

    @pytest.mark.asyncio
    async def test_registered_handoff_jumps_in_astream(self):
        graph = _graph(
            ToolNode([create_handoff_tool("admin_node")]), _assistant("transfer_to_admin_node")
        )
        compiled = graph.compile()
        texts = await _astream_texts(compiled, [Message.text_message("hi")])
        assert graph.ran == ["admin_node"]
        assert "admin ran" in texts

    @pytest.mark.asyncio
    async def test_unknown_goto_raises_graph_error_invoke(self):
        graph = _graph(
            ToolNode([create_handoff_tool("nowhere")]), _assistant("transfer_to_nowhere")
        )
        compiled = graph.compile()
        with pytest.raises(GraphError, match="nowhere"):
            await compiled.ainvoke({"messages": [Message.text_message("hi")]})

    @pytest.mark.asyncio
    async def test_unknown_goto_raises_graph_error_stream(self):
        graph = _graph(
            ToolNode([create_handoff_tool("nowhere")]), _assistant("transfer_to_nowhere")
        )
        compiled = graph.compile()
        with pytest.raises(GraphError, match="nowhere"):
            await _astream_texts(compiled, [Message.text_message("hi")])

    def test_validate_goto_accepts_nodes_and_end(self):
        validate_goto(None, ["a"], "a")
        validate_goto(END, ["a"], "a")
        validate_goto("a", ["a"], "a")
        with pytest.raises(GraphError):
            validate_goto("b", ["a"], "a")

    @pytest.mark.asyncio
    async def test_handler_returns_command_only_for_registered_tool(self):
        state = AgentState(context=[_assistant("transfer_to_x")])
        plain = InvokeNodeHandler("tools", ToolNode([]))
        result = await plain._call_tools(state.context[-1], state, {})
        assert not isinstance(result, Command)

        wired = InvokeNodeHandler("tools", ToolNode([create_handoff_tool("x")]))
        result = await wired._call_tools(state.context[-1], state, {})
        assert isinstance(result, Command) and result.goto == "x"


class TestHiddenTools:
    def _tool_node(self, ran: list[str]) -> ToolNode:
        def safe_tool() -> str:
            ran.append("safe_tool")
            return "ok"

        def admin_reset() -> str:
            ran.append("admin_reset")
            return "reset"

        return ToolNode([safe_tool, admin_reset])

    @pytest.mark.asyncio
    async def test_invoke_hidden_call_not_executed(self):
        ran: list[str] = []
        msg = _assistant("safe_tool", "admin_reset", metadata={"allowed_tools": ["safe_tool"]})
        state = AgentState(context=[msg])
        handler = InvokeNodeHandler("tools", self._tool_node(ran))

        result = await handler._call_tools(msg, state, {})

        assert ran == ["safe_tool"]
        assert len(result) == 2
        hidden = result[1]
        assert isinstance(hidden, Message)
        assert "Tool 'admin_reset' not found." in _tool_result_text(hidden)
        assert any(isinstance(b, ErrorBlock) for b in hidden.content)
        assert hidden.content[-1].call_id == "call_1"

    @pytest.mark.asyncio
    async def test_stream_hidden_call_not_executed(self):
        ran: list[str] = []
        msg = _assistant("safe_tool", "admin_reset", metadata={"allowed_tools": ["safe_tool"]})
        state = AgentState(context=[msg])
        handler = StreamNodeHandler("tools", self._tool_node(ran))

        items = [i async for i in handler._call_tools(msg, state, {"thread_id": "t"})]

        assert ran == ["safe_tool"]
        errors = [i for i in items if getattr(i, "event", None) == StreamEvent.ERROR]
        assert len(errors) == 1
        assert errors[0].data["tool_call_id"] == "call_1"
        hidden = [
            i
            for i in items
            if isinstance(i, Message) and "admin_reset" in _tool_result_text(i)
        ]
        assert hidden and "not found" in _tool_result_text(hidden[0])

    @pytest.mark.asyncio
    async def test_no_allowed_tools_means_no_filtering(self):
        ran: list[str] = []
        msg = _assistant("safe_tool", "admin_reset")
        handler = InvokeNodeHandler("tools", self._tool_node(ran))
        await handler._call_tools(msg, AgentState(context=[msg]), {})
        assert sorted(ran) == ["admin_reset", "safe_tool"]

    @pytest.mark.asyncio
    async def test_converter_merges_message_metadata(self):
        from tenxgraph.runtime.adapters.llm.base_converter import BaseConverter
        from tenxgraph.runtime.adapters.llm.model_response_converter import (
            ModelResponseConverter,
        )

        class Stub(BaseConverter):
            async def convert_response(self, response):
                return Message.text_message("hi", role="assistant")

            async def convert_streaming_response(self, config, node_name, response, meta=None):
                yield Message.text_message("hi", role="assistant")

        conv = ModelResponseConverter(object(), converter=Stub())
        assert (await conv.invoke()).metadata == {}

        conv.message_metadata["allowed_tools"] = ["a"]
        assert (await conv.invoke()).metadata["allowed_tools"] == ["a"]
        streamed = [m async for m in conv.stream({"x": 1}, "node")]
        assert streamed[0].metadata["allowed_tools"] == ["a"]


class TestParallelLimit:
    def test_resolution_order(self, monkeypatch):
        monkeypatch.delenv(MAX_PARALLEL_TOOL_CALLS_ENV, raising=False)
        assert resolve_max_parallel_tool_calls({}) == DEFAULT_MAX_PARALLEL_TOOL_CALLS
        assert resolve_max_parallel_tool_calls(None) == DEFAULT_MAX_PARALLEL_TOOL_CALLS

        monkeypatch.setenv(MAX_PARALLEL_TOOL_CALLS_ENV, "4")
        assert resolve_max_parallel_tool_calls({}) == 4
        assert resolve_max_parallel_tool_calls({"max_parallel_tool_calls": 2}) == 2

    @pytest.mark.parametrize("bad", [0, -3, "abc", True, 1.5e-9])
    def test_invalid_config_uses_default(self, monkeypatch, bad, caplog):
        monkeypatch.delenv(MAX_PARALLEL_TOOL_CALLS_ENV, raising=False)
        with caplog.at_level("WARNING"):
            assert (
                resolve_max_parallel_tool_calls({"max_parallel_tool_calls": bad})
                == DEFAULT_MAX_PARALLEL_TOOL_CALLS
            )
        assert "max_parallel_tool_calls" in caplog.text

    def test_invalid_env_uses_default(self, monkeypatch, caplog):
        monkeypatch.setenv(MAX_PARALLEL_TOOL_CALLS_ENV, "0")
        with caplog.at_level("WARNING"):
            assert resolve_max_parallel_tool_calls({}) == DEFAULT_MAX_PARALLEL_TOOL_CALLS
        assert MAX_PARALLEL_TOOL_CALLS_ENV in caplog.text

    def _slow_node(self):
        stats = {"now": 0, "peak": 0}

        async def slow() -> str:
            stats["now"] += 1
            stats["peak"] = max(stats["peak"], stats["now"])
            await asyncio.sleep(0.05)
            stats["now"] -= 1
            return "done"

        return ToolNode([slow]), stats

    @pytest.mark.asyncio
    async def test_invoke_respects_limit(self):
        tool_node, stats = self._slow_node()
        msg = _assistant(*["slow"] * 6)
        state = AgentState(context=[msg])
        handler = InvokeNodeHandler("tools", tool_node)

        result = await handler._call_tools(msg, state, {"max_parallel_tool_calls": 2})

        assert stats["peak"] == 2
        assert len(result) == 6
        assert [r.content[-1].call_id for r in result] == [f"call_{i}" for i in range(6)]

    @pytest.mark.asyncio
    async def test_stream_respects_limit(self):
        tool_node, stats = self._slow_node()
        msg = _assistant(*["slow"] * 6)
        state = AgentState(context=[msg])
        handler = StreamNodeHandler("tools", tool_node)

        items = [
            i async for i in handler._call_tools(msg, state, {"max_parallel_tool_calls": 2})
        ]

        assert stats["peak"] == 2
        assert len([i for i in items if isinstance(i, Message)]) == 6

    @pytest.mark.asyncio
    async def test_branch_copies_are_lazy(self, monkeypatch):
        """Copies are taken inside the semaphore, so live copies never exceed the limit."""
        import copy as copy_mod

        tool_node, _ = self._slow_node()
        msg = _assistant(*["slow"] * 6)
        state = AgentState(context=[msg])
        handler = InvokeNodeHandler("tools", tool_node)

        started = {"copies": 0, "at_first_tool": None}
        real_deepcopy = copy_mod.deepcopy

        def counting(obj, *a, **k):
            if isinstance(obj, AgentState):
                started["copies"] += 1
            return real_deepcopy(obj, *a, **k)

        monkeypatch.setattr(
            "tenxgraph.core.graph.utils.invoke_node_handler.copy.deepcopy", counting
        )
        original = tool_node.invoke

        async def spying(*a, **k):
            if started["at_first_tool"] is None:
                started["at_first_tool"] = started["copies"]
            return await original(*a, **k)

        monkeypatch.setattr(tool_node, "invoke", spying)

        await handler._call_tools(msg, state, {"max_parallel_tool_calls": 2})

        # baseline + at most 2 branch copies exist when the first tool starts
        assert started["at_first_tool"] <= 3
