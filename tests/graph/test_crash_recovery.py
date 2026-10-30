"""A run that died without cleanup is repaired on the next load (batch B).

A RUNNING state whose ``run_id`` has no live active-run marker belongs to a dead run (SIGKILL,
OOM, or a consumer that closed the stream early). The next load repairs it from the cached
state (decision D2): unanswered tool calls are closed, the state is marked INTERRUPTED with
reason ``crashed`` and saved durably. Then decision D1 applies: a new user message starts a
new turn; a call with no new input resumes at the saved node.
"""

import asyncio
import json

import pytest
import pytest_asyncio

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, ExecutionStatus, Message, ToolCallBlock
from tenxgraph.core.state.execution_state import ExecutionState
from tenxgraph.storage.checkpointer import (
    InMemoryCheckpointer,
    SqliteCheckpointer,
    base_checkpointer,
)
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END, START


FULL = ResponseGranularity.FULL


def tool_call_message(call_id: str = "c1") -> Message:
    return Message(
        role="assistant",
        content=[ToolCallBlock(id=call_id, name="work", args={})],
        tools_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "work", "arguments": json.dumps({})},
            }
        ],
    )


def build(checkpointer):
    histories: list[list[Message]] = []
    gate = {"event": None}

    async def work() -> str:
        return "work done"

    async def model(state: AgentState):
        histories.append(list(state.context))
        if gate["event"] is not None:
            gate["event"].set()
            await asyncio.sleep(30)
        last = state.context[-1]
        if last.role == "tool":
            return Message.text_message("finished", role="assistant")
        if last.text() == "use tool":
            return tool_call_message()
        return Message.text_message(f"ok:{last.text()}", role="assistant")

    def route(state: AgentState) -> str:
        last = state.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([work]))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    return graph.compile(checkpointer=checkpointer), histories, gate


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def checkpointer(request):
    if request.param == "memory":
        yield InMemoryCheckpointer()
    else:
        cp = SqliteCheckpointer(":memory:")
        yield cp
        await cp.arelease()


async def seed_crashed(cp, config, *, run_id="dead-run", node="TOOL", with_call=True):
    """Leave the thread as a SIGKILLed run would: RUNNING in the cache, no marker."""
    context = [Message.text_message("go")]
    if with_call:
        context.append(tool_call_message())
    state = AgentState(context=context)
    state.execution_meta = ExecutionState(
        current_node=node, step=1, status=ExecutionStatus.RUNNING, run_id=run_id
    )
    await cp.aput_state_cache(config, state)
    return state


def assert_closed(context: list[Message]) -> None:
    calls = {c["id"] for m in context for c in (m.tools_calls or [])}
    done = {
        b.call_id for m in context if m.role == "tool" for b in m.content if hasattr(b, "call_id")
    }
    assert calls <= done, f"unanswered tool calls {calls - done}"


@pytest.mark.asyncio
async def test_crashed_state_is_repaired_and_new_message_starts_a_new_turn(checkpointer):
    config = {"thread_id": "t-crash", "user_id": "u1"}
    await seed_crashed(checkpointer, config)
    app, histories, _ = build(checkpointer)

    saved: list[AgentState] = []
    real_put = checkpointer.aput_checkpoint

    async def spy(cfg, state, messages):
        saved.append(state.model_copy(deep=True))
        return await real_put(cfg, state, messages)

    checkpointer.aput_checkpoint = spy

    await app.ainvoke({"messages": [Message.text_message("again")]}, config)

    # The repair was saved durably, with the reason recorded.
    first = saved[0]
    assert first.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert first.execution_meta.interrupt_reason == "crashed"
    assert_closed(first.context)
    assert first.context[0].text() == "go"

    # The new message started a new turn on the repaired context; the model saw a valid history.
    assert histories, "the model ran"
    assert_closed(histories[0])
    assert histories[0][0].text() == "go"
    assert histories[0][-1].text() == "again"

    state = await checkpointer.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert [m.text() for m in state.context][-1] == "ok:again"
    stored = [m.text() for m in await checkpointer.alist_messages(config)]
    assert stored.count("go") == 1
    assert "again" in stored


@pytest.mark.asyncio
async def test_crashed_state_without_input_resumes_at_saved_node(checkpointer):
    config = {"thread_id": "t-crash-resume", "user_id": "u1"}
    await seed_crashed(checkpointer, config, node="MODEL", with_call=False)
    app, histories, _ = build(checkpointer)

    await app.ainvoke({}, config)

    assert [h[-1].text() for h in histories] == ["go"]
    state = await checkpointer.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert [m.text() for m in state.context] == ["go", "ok:go"]


@pytest.mark.asyncio
async def test_crash_recovery_uses_the_cache_not_the_durable_copy(checkpointer):
    """The cache holds the crashed turn's user message; the durable store does not (D2)."""
    config = {"thread_id": "t-crash-cache", "user_id": "u1"}
    old = AgentState(
        context=[Message.text_message("earlier"), Message.text_message("done", role="assistant")]
    )
    old.execution_meta = ExecutionState(
        current_node=END, step=2, status=ExecutionStatus.COMPLETED
    )
    await checkpointer.aput_state(config, old)
    await seed_crashed(checkpointer, config, node="MODEL", with_call=False)
    app, histories, _ = build(checkpointer)

    await app.ainvoke({"messages": [Message.text_message("more")]}, config)

    assert [m.text() for m in histories[0]] == ["go", "more"]


@pytest.mark.asyncio
async def test_live_run_is_left_alone():
    cp = InMemoryCheckpointer()
    config = {"thread_id": "t-live", "user_id": "u1", "run_id": "other-run"}
    await seed_crashed(cp, config, run_id="alive-run")
    await cp.aput_cache_value(base_checkpointer.ACTIVE_RUN_NAMESPACE, "t-live:u1", "alive-run", 60)
    from tenxgraph.core.graph.utils.utils import is_dead_run

    state = await cp.aget_state_cache(config)
    assert not is_dead_run(state, await cp.aget_active_run(config), config)
    assert is_dead_run(state, None, config)
    assert is_dead_run(state, "someone-else", config)
    state.execution_meta.run_id = None  # a state from before run ids existed
    assert not is_dead_run(state, None, config)
    state.execution_meta.run_id = "same"
    assert not is_dead_run(state, None, {"run_id": "same"})
    state.execution_meta.status = ExecutionStatus.COMPLETED
    state.execution_meta.run_id = "dead"
    assert not is_dead_run(state, None, config)


@pytest.mark.asyncio
async def test_stream_closed_early_is_repaired_on_next_load(checkpointer):
    """A consumer that closes the stream generator early skips the cancel branch.

    The state stays RUNNING and ``_end_run`` clears the marker; the next load must repair it.
    """
    config = {"thread_id": "t-genexit", "user_id": "u1"}
    app, histories, _ = build(checkpointer)

    agen = app.astream({"messages": [Message.text_message("use tool")]}, config)
    async for chunk in agen:
        if chunk.message is not None and chunk.message.role == "tool":
            break  # the consumer walks away mid-run
    await agen.aclose()

    cached = await checkpointer.aget_state_cache(config)
    assert cached is not None
    assert cached.execution_meta.status == ExecutionStatus.RUNNING
    assert await checkpointer.aget_active_run(config) is None

    histories.clear()
    await app.ainvoke({"messages": [Message.text_message("next")]}, config)

    assert histories[0][0].text() == "use tool"
    assert histories[0][-1].text() == "next"
    assert_closed(histories[0])
    state = await checkpointer.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_active_run_marker_expires_without_a_heartbeat(monkeypatch):
    monkeypatch.setattr(base_checkpointer, "ACTIVE_RUN_TTL_SECONDS", 1)
    cp = InMemoryCheckpointer()
    config = {"thread_id": "t-ttl", "user_id": "u1", "run_id": "r1"}
    await cp.abegin_run(config)
    assert await cp.aget_active_run(config) == "r1"
    await asyncio.sleep(1.2)
    assert await cp.aget_active_run(config) is None


@pytest.mark.asyncio
async def test_heartbeat_keeps_the_marker_alive_while_the_run_lasts(monkeypatch):
    monkeypatch.setattr(base_checkpointer, "ACTIVE_RUN_TTL_SECONDS", 1)
    monkeypatch.setattr(base_checkpointer, "ACTIVE_RUN_HEARTBEAT_SECONDS", 0.3)
    cp = InMemoryCheckpointer()
    config = {"thread_id": "t-beat", "user_id": "u1"}
    app, _, gate = build(cp)
    gate["event"] = asyncio.Event()

    task = asyncio.ensure_future(app.ainvoke({"messages": [Message.text_message("hi")]}, config))
    await asyncio.wait_for(gate["event"].wait(), 5)
    await asyncio.sleep(1.6)  # longer than the TTL
    assert await cp.aget_active_run(config) is not None

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await cp.aget_active_run(config) is None


def test_run_id_round_trips_and_old_states_load():
    meta = ExecutionState(current_node=START, run_id="r1")
    assert ExecutionState.model_validate(meta.model_dump()).run_id == "r1"
    legacy = ExecutionState.from_dict({"current_node": START})
    assert legacy.run_id is None
