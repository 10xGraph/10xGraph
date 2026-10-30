"""The stream path does what the invoke path does (G-09, G-160, T-1.1, C-S1).

Streaming is the production mode. It needs the tool ledger, isolated parallel tools, tool
results in call order, cancel handling, the stop check inside a node, the AI callbacks, and the
same developer-set timeouts as ``ainvoke``.
"""

import asyncio
import json
from typing import Annotated

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, ExecutionStatus, Message, ToolCallBlock
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.qa.testing.test_agent import TestAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import CallbackManager, InvocationType, ResponseGranularity, constants
from tenxgraph.utils.constants import END


FULL = ResponseGranularity.FULL
MODES = ["ainvoke", "astream"]


def _sum(left: int, right: int) -> int:
    return left + right


class PState(AgentState):
    counter: Annotated[int, _sum] = 0
    winner: str = ""
    alpha: str = ""
    beta: str = ""


def assistant_call(calls: list[tuple[str, str, dict]], message_id: str = "asst-1") -> Message:
    return Message(
        role="assistant",
        message_id=message_id,
        content=[ToolCallBlock(id=cid, name=name, args=args) for cid, name, args in calls],
        tools_calls=[
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for cid, name, args in calls
        ],
    )


def build(tools, calls, *, checkpointer=None, state=None, callback_manager=None):
    async def model(st: AgentState):
        if st.context[-1].role == "tool":
            return Message.text_message("done", role="assistant")
        return assistant_call(calls)

    def route(st: AgentState) -> str:
        last = st.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph(state) if state is not None else StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode(tools))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    kwargs = {}
    if callback_manager is not None:
        kwargs["callback_manager"] = callback_manager
    return graph.compile(checkpointer=checkpointer, **kwargs)


async def run(app, mode, config, text="go"):
    data = {"messages": [Message.text_message(text)]}
    if mode == "ainvoke":
        return await app.ainvoke(data, config, response_granularity=FULL)
    return [c async for c in app.astream(data, config, response_granularity=FULL)]


def tool_outputs(context: list[Message]) -> list[str]:
    out = []
    for msg in context:
        if msg.role == "tool":
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    out.append(str(block.output))
    return out


# ------------------------------------------------------------------ tool ledger (C-S2)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_recorded_tool_result_is_reused_not_run_again(mode):
    ran: list[str] = []

    async def send_email(tool_call_id: str | None = None) -> str:
        ran.append("sent")
        return "sent"

    cp = InMemoryCheckpointer()
    app = build([send_email], [("c1", "send_email", {})], checkpointer=cp)
    config = {"thread_id": f"t-ledger-{mode}", "user_id": "u1"}
    recorded = Message.tool_message(
        content=[
            ToolResultBlock(call_id="c1", output="recorded", status="completed", is_error=False)
        ]
    )
    await cp.aput_tool_result(
        config, "asst-1:c1", {"__kind__": "message", "value": recorded.model_dump(mode="json")}
    )

    await run(app, mode, config)

    assert ran == []
    state = await cp.aget_state(config)
    assert "recorded" in tool_outputs(state.context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_finished_tool_call_is_recorded(mode):
    async def work(tool_call_id: str | None = None) -> str:
        return "worked"

    cp = InMemoryCheckpointer()
    app = build([work], [("c1", "work", {})], checkpointer=cp)
    config = {"thread_id": f"t-record-{mode}", "user_id": "u1"}

    await run(app, mode, config)

    entry = await cp.aget_tool_result(config, "asst-1:c1")
    assert entry is not None
    assert entry["__kind__"] == "message"


# ------------------------------------------------------- parallel isolation and order


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_parallel_tools_do_not_clobber_each_others_state(mode):
    async def bump(state: AgentState, tool_call_id: str | None = None) -> Message:
        seen = state.counter  # type: ignore[attr-defined]
        await asyncio.sleep(0.05)
        state.counter = seen + 1  # type: ignore[attr-defined]
        return Message.tool_message(
            content=[ToolResultBlock(call_id=tool_call_id or "", output="ok", status="completed")]
        )

    cp = InMemoryCheckpointer()
    app = build(
        [bump],
        [("c1", "bump", {}), ("c2", "bump", {})],
        checkpointer=cp,
        state=PState(),
    )
    config = {"thread_id": f"t-iso-{mode}", "user_id": "u1"}

    await run(app, mode, config)

    state = await cp.aget_state(config)
    assert state.counter == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_parallel_tool_results_are_saved_in_call_order(mode):
    async def slow(label: str, delay: float, tool_call_id: str | None = None) -> str:
        await asyncio.sleep(delay)
        return label

    cp = InMemoryCheckpointer()
    app = build(
        [slow],
        [
            ("c1", "slow", {"label": "first", "delay": 0.2}),
            ("c2", "slow", {"label": "second", "delay": 0.1}),
            ("c3", "slow", {"label": "third", "delay": 0.0}),
        ],
        checkpointer=cp,
    )
    config = {"thread_id": f"t-order-{mode}", "user_id": "u1"}

    await run(app, mode, config)

    state = await cp.aget_state(config)
    assert tool_outputs(state.context) == ["first", "second", "third"]
    stored = await cp.alist_messages(config)
    assert tool_outputs(stored) == ["first", "second", "third"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_parallel_write_to_one_plain_field_keeps_the_last_call(mode):
    async def set_winner(
        name: str, delay: float, state: AgentState, tool_call_id: str | None = None
    ):
        await asyncio.sleep(delay)
        state.winner = name  # type: ignore[attr-defined]
        return Message.tool_message(
            content=[ToolResultBlock(call_id=tool_call_id or "", output=name, status="completed")]
        )

    cp = InMemoryCheckpointer()
    app = build(
        [set_winner],
        [
            ("c1", "set_winner", {"name": "call-1", "delay": 0.2}),
            ("c2", "set_winner", {"name": "call-2", "delay": 0.0}),
        ],
        checkpointer=cp,
        state=PState(),
    )
    config = {"thread_id": f"t-winner-{mode}", "user_id": "u1"}

    await run(app, mode, config)

    state = await cp.aget_state(config)
    assert state.winner == "call-2"


# ------------------------------------------------------------------------ cancel


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_cancelled_run_stops_its_tools(mode):
    started = asyncio.Event()
    outcome: list[str] = []

    async def hang(tool_call_id: str | None = None) -> str:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        outcome.append("finished")
        return "never"

    app = build(
        [hang],
        [("c1", "hang", {}), ("c2", "hang", {})],
        checkpointer=InMemoryCheckpointer(),
    )
    task = asyncio.create_task(run(app, mode, {"thread_id": f"t-cancel-{mode}"}))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)

    assert outcome == ["cancelled", "cancelled"]


# ---------------------------------------------------------------- stop inside a node


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_stop_request_interrupts_a_node_that_is_running(mode):
    started = asyncio.Event()

    async def hang(state: AgentState):
        started.set()
        await asyncio.sleep(30)
        return Message.text_message("never", role="assistant")

    cp = InMemoryCheckpointer()
    graph = StateGraph()
    graph.add_node("HANG", hang)
    graph.add_edge("HANG", END)
    graph.set_entry_point("HANG")
    app = graph.compile(checkpointer=cp)
    config = {"thread_id": f"t-midstop-{mode}"}

    task = asyncio.create_task(run(app, mode, config))
    await asyncio.wait_for(started.wait(), 5)
    await app.astop({"thread_id": config["thread_id"]})
    await asyncio.wait_for(task, 10)

    state = await cp.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert state.execution_meta.interrupt_reason == "stop_requested"


# -------------------------------------------------------------------- AI callbacks


def callback_manager(events: list[str], *, fail: bool = False):
    manager = CallbackManager()

    async def before(context, data):
        events.append(f"before:{context.node_name}")
        return data

    async def after(context, data, result):
        events.append(f"after:{context.node_name}")
        return result

    async def on_error(context, data, error):
        events.append(f"error:{context.node_name}")
        return Message.text_message("recovered", role="assistant")

    manager.register_before_invoke(InvocationType.AI, before)
    manager.register_after_invoke(InvocationType.AI, after)
    manager.register_on_error(InvocationType.AI, on_error)
    return manager


def agent_graph(agent, manager, checkpointer=None):
    graph = StateGraph()
    graph.add_node("MAIN", agent)
    graph.add_edge("MAIN", END)
    graph.set_entry_point("MAIN")
    return graph.compile(checkpointer=checkpointer, callback_manager=manager)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_before_and_after_callbacks_fire_for_an_agent_node(mode):
    events: list[str] = []
    app = agent_graph(TestAgent(responses=["hi"]), callback_manager(events))

    await run(app, mode, {"thread_id": f"t-cb-{mode}"})

    assert [e for e in events if e.endswith(":MAIN")] == ["before:MAIN", "after:MAIN"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_on_error_callback_can_recover_an_agent_node(mode):
    class Broken(TestAgent):
        async def _call_llm(self, messages, tools=None, **kwargs):
            raise RuntimeError("model down")

    events: list[str] = []
    cp = InMemoryCheckpointer()
    app = agent_graph(Broken(responses=["x"]), callback_manager(events), cp)
    config = {"thread_id": f"t-cb-err-{mode}"}

    await run(app, mode, config)

    assert "error:MAIN" in events
    state = await cp.aget_state(config)
    assert state.context[-1].text() == "recovered"


# ------------------------------------------------------------------------ timeouts


def test_there_is_no_default_tool_or_node_timeout():
    assert constants.DEFAULT_TOOL_TIMEOUT_SECONDS is None
    assert constants.DEFAULT_NODE_TIMEOUT_SECONDS is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_without_config_a_slow_tool_is_not_cut_off(mode, monkeypatch):
    async def slow(tool_call_id: str | None = None) -> str:
        await asyncio.sleep(0.3)
        return "slow done"

    cp = InMemoryCheckpointer()
    app = build([slow], [("c1", "slow", {})], checkpointer=cp)
    config = {"thread_id": f"t-notimeout-{mode}", "user_id": "u1"}

    await run(app, mode, config)

    state = await cp.aget_state(config)
    assert "slow done" in tool_outputs(state.context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_configured_tool_timeout_cuts_the_tool_off(mode):
    async def slow(tool_call_id: str | None = None) -> str:
        await asyncio.sleep(5)
        return "slow done"

    cp = InMemoryCheckpointer()
    app = build([slow], [("c1", "slow", {})], checkpointer=cp)
    config = {"thread_id": f"t-tooltimeout-{mode}", "user_id": "u1", "tool_timeout": 0.1}

    await asyncio.wait_for(run(app, mode, config), 3)

    state = await cp.aget_state(config)
    outputs = tool_outputs(state.context)
    assert "slow done" not in outputs
    assert any("timeout" in o for o in outputs)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_configured_node_timeout_cuts_the_node_off(mode):
    async def slow(state: AgentState):
        await asyncio.sleep(5)
        return Message.text_message("late", role="assistant")

    graph = StateGraph()
    graph.add_node("SLOW", slow)
    graph.add_edge("SLOW", END)
    graph.set_entry_point("SLOW")
    app = graph.compile(checkpointer=InMemoryCheckpointer())
    config = {"thread_id": f"t-nodetimeout-{mode}", "node_timeout": 0.1}

    with pytest.raises(Exception, match="timeout"):
        await asyncio.wait_for(run(app, mode, config), 3)
