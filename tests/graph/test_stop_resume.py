"""A stopped thread is usable again (G-03, G-16).

A stop saves the run INTERRUPTED_AFTER at the node that finished. The next call on the thread
must run: with no new input it resumes at the saved node; with a new user message it starts a
new turn (decision D1).
"""

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END


FULL = ResponseGranularity.FULL


def build(stop_on_first: dict):
    checkpointer = InMemoryCheckpointer()
    seen: list[list[str]] = []
    holder: dict = {}

    async def first(state: AgentState, config: dict):
        seen.append([m.text() for m in state.context])
        if stop_on_first["stop"]:
            stop_on_first["stop"] = False
            await holder["app"].astop({"thread_id": config["thread_id"]})
        return Message.text_message(f"first:{state.context[-1].text()}", role="assistant")

    async def second(state: AgentState):
        return Message.text_message("second", role="assistant")

    graph = StateGraph()
    graph.add_node("FIRST", first)
    graph.add_node("SECOND", second)
    graph.add_edge("FIRST", "SECOND")
    graph.add_edge("SECOND", END)
    graph.set_entry_point("FIRST")
    app = graph.compile(checkpointer=checkpointer)
    holder["app"] = app
    return app, checkpointer, seen


async def call(app, mode, data, thread):
    config = {"thread_id": thread}
    if mode == "ainvoke":
        return await app.ainvoke(data, config, response_granularity=FULL)
    return [c async for c in app.astream(data, config, response_granularity=FULL)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_new_message_after_stop_runs_a_new_turn(mode):
    stop = {"stop": True}
    app, cp, seen = build(stop)
    thread = f"t-stop-new-{mode}"

    await call(app, mode, {"messages": [Message.text_message("one")]}, thread)
    state = await cp.aget_state({"thread_id": thread})
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert state.execution_meta.interrupt_reason == "stop_requested"

    await call(app, mode, {"messages": [Message.text_message("two")]}, thread)

    # The agent ran again on the new message, on top of the earlier context.
    assert len(seen) == 2
    assert seen[-1][-1] == "two"
    state = await cp.aget_state({"thread_id": thread})
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    texts = [m.text() for m in state.context]
    assert "two" in texts
    assert "second" in texts
    stored = [m.text() for m in await cp.alist_messages({"thread_id": thread})]
    assert "two" in stored


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_call_without_input_after_stop_resumes_at_saved_node(mode):
    stop = {"stop": True}
    app, cp, seen = build(stop)
    thread = f"t-stop-resume-{mode}"

    await call(app, mode, {"messages": [Message.text_message("one")]}, thread)
    await call(app, mode, {}, thread)

    state = await cp.aget_state({"thread_id": thread})
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert "second" in [m.text() for m in state.context]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_resume_after_stop_does_not_rerun_the_finished_node(mode):
    """INTERRUPTED_AFTER means the node finished: the resume continues at the next one."""
    stop = {"stop": True}
    app, cp, seen = build(stop)
    thread = f"t-stop-norerun-{mode}"

    await call(app, mode, {"messages": [Message.text_message("one")]}, thread)
    await call(app, mode, {}, thread)

    assert len(seen) == 1, "FIRST ran once"
    state = await cp.aget_state({"thread_id": thread})
    texts = [m.text() for m in state.context]
    assert texts.count("first:one") == 1
    assert texts.count("second") == 1
    assert "first:first:one" not in texts


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_stop_during_the_last_node_just_completes(mode):
    """Nothing is left to stop when the node that finished was the last one."""
    checkpointer = InMemoryCheckpointer()
    holder: dict = {}

    async def only(state: AgentState, config: dict):
        await holder["app"].astop({"thread_id": config["thread_id"]})
        return Message.text_message("only", role="assistant")

    graph = StateGraph()
    graph.add_node("ONLY", only)
    graph.add_edge("ONLY", END)
    graph.set_entry_point("ONLY")
    app = graph.compile(checkpointer=checkpointer)
    holder["app"] = app
    thread = f"t-stop-last-{mode}"

    await call(app, mode, {"messages": [Message.text_message("one")]}, thread)

    state = await checkpointer.aget_state({"thread_id": thread})
    assert state.execution_meta.status == ExecutionStatus.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_stop_saves_the_live_state(mode):
    """G-16: the stop save carries the live state, not the cache copy reloaded from storage."""
    stop = {"stop": True}
    app, cp, _ = build(stop)
    thread = f"t-stop-live-{mode}"

    # Make the cache stale: it never sees the node's output, only the live state does.
    real_put = cp.aput_state_cache

    async def stale_cache(config, state):
        return None

    cp.aput_state_cache = stale_cache  # type: ignore[method-assign]
    await call(app, mode, {"messages": [Message.text_message("one")]}, thread)
    cp.aput_state_cache = real_put  # type: ignore[method-assign]

    state = await cp.aget_state({"thread_id": thread})
    assert state is not None
    assert "first:one" in [m.text() for m in state.context]


@pytest.mark.asyncio
async def test_resume_without_input_does_not_resend_the_last_user_message():
    """G-27: invoke put the last user message into the run's messages again on resume."""
    stop = {"stop": True}
    app, cp, _ = build(stop)
    thread = "t-stop-noresend"

    await call(app, "ainvoke", {"messages": [Message.text_message("one")]}, thread)
    result = await call(app, "ainvoke", {}, thread)

    assert "one" not in [m.text() for m in result["messages"]]
    stored = [m.text() for m in await cp.alist_messages({"thread_id": thread})]
    assert stored.count("one") == 1
