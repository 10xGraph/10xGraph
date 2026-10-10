"""``interrupt()`` pauses a graph from a node or tool; ``{"resume": value}`` continues it."""

import json

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message, StreamEvent, ToolCallBlock
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END
from tenxgraph.utils.interrupt import GraphInterrupt, interrupt, pending_interrupt


def _config(thread_id: str) -> dict:
    return {"thread_id": thread_id}


FULL = ResponseGranularity.FULL


def _approval_graph(calls: list):
    async def ask(state: AgentState):
        calls.append("before")
        answer = interrupt({"amount": 50}, message="Approve refund?", reason="approval")
        calls.append(("after", answer))
        return Message.text_message(f"answer={answer}", role="assistant")

    graph = StateGraph()
    graph.add_node("ASK", ask)
    graph.add_edge("ASK", END)
    graph.set_entry_point("ASK")
    return graph.compile(checkpointer=InMemoryCheckpointer())


def _texts(result) -> list[str]:
    return [m.text() for m in result["messages"] if m.role == "assistant"]


@pytest.mark.asyncio
async def test_invoke_pauses_then_resumes_with_the_value():
    calls: list = []
    app = _approval_graph(calls)

    first = await app.ainvoke(
        {"messages": [Message.text_message("refund")]}, _config("t1"), response_granularity=FULL
    )
    request = pending_interrupt(first["state"])
    assert request is not None
    assert request.value == {"amount": 50}
    assert request.message == "Approve refund?"
    assert request.reason == "approval"
    assert request.node == "ASK"
    assert _texts(first) == []

    second = await app.ainvoke(
        {"resume": {"approved": True}}, _config("t1"), response_granularity=FULL
    )
    assert pending_interrupt(second["state"]) is None
    assert _texts(second) == ["answer={'approved': True}"]
    # The node ran twice: once to pause, once to finish.
    assert calls == ["before", "before", ("after", {"approved": True})]


@pytest.mark.asyncio
async def test_stream_reports_the_interrupt_and_resumes():
    app = _approval_graph([])
    chunks = [
        c
        async for c in app.astream(
            {"messages": [Message.text_message("refund")]},
            _config("t2"),
            response_granularity=ResponseGranularity.FULL,
        )
    ]
    interrupted = [
        c
        for c in chunks
        if c.event == StreamEvent.UPDATES and (c.data or {}).get("status") == "interrupted"
    ]
    assert len(interrupted) == 1
    payload = interrupted[0].data["interrupt"]
    assert payload["value"] == {"amount": 50}
    assert payload["id"].startswith("int_")
    assert not any(c.event == StreamEvent.ERROR for c in chunks)

    resumed = [
        c.message.text()
        async for c in app.astream({"resume": "yes"}, _config("t2"))
        if c.event == StreamEvent.MESSAGE and c.message and c.message.role == "assistant"
    ]
    assert resumed == ["answer=yes"]


@pytest.mark.asyncio
async def test_resuming_without_a_value_is_rejected():
    app = _approval_graph([])
    await app.ainvoke(
        {"messages": [Message.text_message("refund")]}, _config("t3"), response_granularity=FULL
    )
    with pytest.raises(ValueError, match="resume"):
        await app.ainvoke(
            {"messages": [Message.text_message("hello?")]}, _config("t3"), response_granularity=FULL
        )


@pytest.mark.asyncio
async def test_two_interrupts_in_one_node_are_answered_in_order():
    async def two_questions(state: AgentState):
        first = interrupt("first?")
        second = interrupt("second?")
        return Message.text_message(f"{first}+{second}", role="assistant")

    graph = StateGraph()
    graph.add_node("Q", two_questions)
    graph.add_edge("Q", END)
    graph.set_entry_point("Q")
    app = graph.compile(checkpointer=InMemoryCheckpointer())

    r1 = await app.ainvoke(
        {"messages": [Message.text_message("go")]}, _config("t4"), response_granularity=FULL
    )
    assert pending_interrupt(r1["state"]).value == "first?"
    r2 = await app.ainvoke({"resume": "a"}, _config("t4"), response_granularity=FULL)
    assert pending_interrupt(r2["state"]).value == "second?"
    r3 = await app.ainvoke({"resume": "b"}, _config("t4"), response_granularity=FULL)
    assert _texts(r3) == ["a+b"]


@pytest.mark.asyncio
async def test_a_later_turn_asks_again():
    app = _approval_graph([])
    await app.ainvoke(
        {"messages": [Message.text_message("one")]}, _config("t5"), response_granularity=FULL
    )
    await app.ainvoke({"resume": "first"}, _config("t5"), response_granularity=FULL)
    # The old answer must not leak into the next turn's interrupt.
    again = await app.ainvoke(
        {"messages": [Message.text_message("two")]}, _config("t5"), response_granularity=FULL
    )
    assert pending_interrupt(again["state"]) is not None


def test_interrupt_outside_a_graph_raises():
    with pytest.raises(RuntimeError):
        interrupt("nope")


def test_graph_interrupt_is_not_an_exception():
    # Tool and node error handlers catch Exception; the pause must pass through them.
    assert not issubclass(GraphInterrupt, Exception)


# ---------------------------------------------------------------------------- tools


def _tool_graph(tools: list, calls: list[dict]):
    async def model(state: AgentState):
        last = state.context[-1]
        if last.role == "tool":
            done = [m for m in state.context if m.role == "tool"]
            if len(done) >= len(calls):
                return Message.text_message("|".join(m.text() for m in done), role="assistant")
            return Message.text_message("waiting", role="assistant")
        return Message(
            role="assistant",
            content=[ToolCallBlock(id=c["id"], name=c["name"], args=c["args"]) for c in calls],
            tools_calls=[
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                }
                for c in calls
            ],
        )

    def route(state: AgentState) -> str:
        last = state.context[-1]
        if last.role == "assistant" and last.tools_calls:
            return "TOOL"
        return END

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode(tools))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    return graph.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invoke", "stream"])
async def test_tool_can_ask_for_approval(mode):
    async def refund(amount: int) -> str:
        decision = interrupt({"amount": amount}, reason="tool_approval")
        return "refunded" if decision == "approve" else "declined"

    app = _tool_graph([refund], [{"id": "c1", "name": "refund", "args": {"amount": 5}}])
    config = _config(f"t-tool-{mode}")

    if mode == "invoke":
        first = await app.ainvoke(
            {"messages": [Message.text_message("refund 5")]}, config, response_granularity=FULL
        )
        request = pending_interrupt(first["state"])
        # A pause is not a tool failure: no error result was recorded.
        assert not [m for m in first["state"].context if m.role == "tool"]
        final = await app.ainvoke({"resume": "approve"}, config, response_granularity=FULL)
        texts = _texts(final)
    else:
        chunks = [
            c
            async for c in app.astream(
                {"messages": [Message.text_message("refund 5")]},
                config,
                response_granularity=ResponseGranularity.FULL,
            )
        ]
        assert not any(c.event == StreamEvent.ERROR for c in chunks)
        state = await app._checkpointer.aget_state(config)
        request = pending_interrupt(state)
        texts = [
            c.message.text()
            async for c in app.astream({"resume": "approve"}, config)
            if c.event == StreamEvent.MESSAGE and c.message and c.message.role == "assistant"
        ]

    assert request is not None
    assert request.tool_call_id == "c1"
    assert request.node == "TOOL"
    assert request.value == {"amount": 5}
    assert texts[-1] == "refunded"


@pytest.mark.asyncio
async def test_finished_parallel_tools_do_not_run_again_on_resume():
    runs: list[str] = []

    async def lookup() -> str:
        runs.append("lookup")
        return "found"

    async def refund() -> str:
        runs.append("refund")
        return "refunded" if interrupt("ok?") == "yes" else "declined"

    app = _tool_graph(
        [lookup, refund],
        [{"id": "a", "name": "lookup", "args": {}}, {"id": "b", "name": "refund", "args": {}}],
    )
    await app.ainvoke(
        {"messages": [Message.text_message("go")]}, _config("t-par"), response_granularity=FULL
    )
    final = await app.ainvoke({"resume": "yes"}, _config("t-par"), response_granularity=FULL)

    assert _texts(final)[-1] == "found|refunded"
    assert runs.count("lookup") == 1


def test_sync_stream_pauses_and_sync_invoke_resumes():
    # The sync wrapper drives the async generator across contexts; the pause must still work.
    calls: list = []
    app = _approval_graph(calls)
    chunks = list(
        app.stream(
            {"messages": [Message.text_message("refund")]},
            _config("t-sync"),
            response_granularity=FULL,
        )
    )
    assert any((c.data or {}).get("status") == "interrupted" for c in chunks)
    result = app.invoke({"resume": "ok"}, _config("t-sync"), response_granularity=FULL)
    assert _texts(result) == ["answer=ok"]
