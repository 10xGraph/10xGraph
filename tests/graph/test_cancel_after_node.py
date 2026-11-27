"""A run cancelled after a node returned, before the engine moved on, does not re-run it.

The node's output is already in the live state then. Saving the run at that node (as
interrupted) made a call with no new input run it again and store its output twice. Like a stop,
the cancel now saves the run at the node that follows, or COMPLETED when nothing follows.
"""

from __future__ import annotations

import asyncio

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleHook
from tenxgraph.utils.constants import END


FINAL = "final reply"


def _tool_call() -> Message:
    return Message(
        role="assistant",
        content=[],
        tools_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "work", "arguments": "{}"}}
        ],
    )


def _agent_graph(checkpointer, calls: dict[str, int], **compile_kwargs):
    """agent -> tools -> agent -> END; the agent is a plain function node."""

    def work() -> str:
        """A tool."""
        calls["tool"] += 1
        return "tool-result"

    async def agent(state: AgentState) -> Message:
        if not any(m.role == "tool" for m in state.context):
            calls["agent1"] += 1
            return _tool_call()
        calls["agent2"] += 1
        await asyncio.sleep(0)
        return Message.text_message(FINAL, "assistant")

    def route(state: AgentState) -> str:
        last = state.context[-1]
        return "tools" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph()
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([work]))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer, **compile_kwargs)


def _new_calls() -> dict[str, int]:
    return {"agent1": 0, "agent2": 0, "tool": 0}


def _finals(state: AgentState) -> int:
    return sum(1 for m in state.context if m.role == "assistant" and m.text() == FINAL)


def _tool_calls(state: AgentState) -> int:
    return sum(1 for m in state.context if m.role == "assistant" and m.tools_calls)


async def _close_at(app, config, k: int, started) -> bool:
    """Stream the run and close it at chunk ``k`` counted once ``started()`` holds."""
    gen = app.astream({"messages": [Message.text_message("hi")]}, config=dict(config))
    i = 0
    async for _chunk in gen:
        if not started():
            continue
        if i >= k:
            await gen.aclose()
            return True
        i += 1
    return False


@pytest.mark.asyncio
async def test_stream_closed_after_final_reply_stores_it_once():
    closes = 0
    for k in range(40):
        calls = _new_calls()
        cp = InMemoryCheckpointer()
        app = _agent_graph(cp, calls)
        config = {"thread_id": f"t-final-{k}", "user_id": "u1"}
        if not await _close_at(app, config, k, lambda c=calls: c["agent2"] > 0):
            break
        closes += 1
        state = await cp.aget_state(dict(config))
        if state.execution_meta.status != ExecutionStatus.COMPLETED:
            await app.ainvoke({"messages": []}, config=dict(config))
            state = await cp.aget_state(dict(config))
            assert state.execution_meta.status == ExecutionStatus.COMPLETED
        assert _finals(state) == 1, k
        assert calls["agent2"] == 1, k
        await app.aclose()
    assert closes > 0


class _SlowSave(InMemoryCheckpointer):
    """Holds the first durable save whose state ``match`` accepts, so a test can cancel it."""

    def __init__(self, match) -> None:
        super().__init__()
        self.match = match
        self.hit = asyncio.Event()

    async def aput_checkpoint(self, config, state, *args, **kwargs):
        if not self.hit.is_set() and self.match(state):
            self.hit.set()
            await asyncio.sleep(0.2)
        return await super().aput_checkpoint(config, state, *args, **kwargs)


async def _cancel_invoke_during_save(holder, app, config) -> None:
    """Start the run, cancel it while ``holder`` holds it (``holder.hit`` is set)."""
    task = asyncio.create_task(
        app.ainvoke({"messages": [Message.text_message("hi")]}, config=dict(config))
    )
    await asyncio.wait_for(holder.hit.wait(), 5)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_invoke_cancelled_during_completed_save_stores_reply_once():
    calls = _new_calls()
    cp = _SlowSave(lambda s: s.execution_meta.status == ExecutionStatus.COMPLETED)
    app = _agent_graph(cp, calls)
    config = {"thread_id": "t-completed-save", "user_id": "u1"}
    await _cancel_invoke_during_save(cp, app, config)

    state = await cp.aget_state(dict(config))
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert _finals(state) == 1
    assert calls["agent2"] == 1
    await app.aclose()


class _SlowStateUpdate(GraphLifecycleHook):
    """Holds ``on_state_update`` once the final reply is merged: after the node, before
    ``_advance``."""

    def __init__(self) -> None:
        self.hit = asyncio.Event()

    async def on_state_update(self, context, node_name, old_state, new_state, step):
        if not self.hit.is_set() and _finals(new_state):
            self.hit.set()
            await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_invoke_cancelled_after_final_reply_node_completes():
    calls = _new_calls()
    hook = _SlowStateUpdate()
    callbacks = CallbackManager()
    callbacks.register_lifecycle_hook(hook)
    cp = InMemoryCheckpointer()
    app = _agent_graph(cp, calls, callback_manager=callbacks)
    config = {"thread_id": "t-after-final", "user_id": "u1"}
    await _cancel_invoke_during_save(hook, app, config)

    state = await cp.aget_state(dict(config))
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert _finals(state) == 1
    assert calls["agent2"] == 1
    await app.aclose()


@pytest.mark.asyncio
async def test_invoke_cancelled_during_step_save_resumes_at_next_node():
    """The per-step save after the tool-call reply: the step is counted once, not twice."""
    calls = _new_calls()
    cp = _SlowSave(lambda s: s.execution_meta.current_node == "tools" and _tool_calls(s) == 1)
    app = _agent_graph(cp, calls)
    config = {"thread_id": "t-step-save", "user_id": "u1"}
    await _cancel_invoke_during_save(cp, app, config)

    saved = await cp.aget_state(dict(config))
    assert saved.execution_meta.current_node == "tools"
    assert saved.execution_meta.step == 2  # the step an uncancelled run saves at tools
    await app.ainvoke({"messages": []}, config=dict(config))
    state = await cp.aget_state(dict(config))
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert _tool_calls(state) == 1
    assert _finals(state) == 1
    assert calls["agent1"] == calls["agent2"] == 1
    # The cancel closed the unanswered call ("did not finish"), so the tool node is skipped.
    assert calls["tool"] == 0
    await app.aclose()


@pytest.mark.asyncio
async def test_stream_closed_after_tool_call_reply_runs_tool_once():
    closes = 0
    for k in range(60):
        calls = _new_calls()
        cp = InMemoryCheckpointer()
        app = _agent_graph(cp, calls)
        config = {"thread_id": f"t-toolcall-{k}", "user_id": "u1"}
        if not await _close_at(app, config, k, lambda c=calls: c["agent1"] > 0):
            break
        closes += 1
        state = await cp.aget_state(dict(config))
        if state.execution_meta.status != ExecutionStatus.COMPLETED:
            await app.ainvoke({"messages": []}, config=dict(config))
            state = await cp.aget_state(dict(config))
        assert state.execution_meta.status == ExecutionStatus.COMPLETED, k
        assert _tool_calls(state) == 1, k
        assert _finals(state) == 1, k
        assert calls["agent1"] == calls["agent2"] == 1, k
        # A cancel before the tool node ran closes the call as unfinished; never twice.
        assert calls["tool"] <= 1, k
        await app.aclose()
    assert closes > 0


def _a_then_b(checkpointer, ran: list[str]):
    async def a(state: AgentState):
        ran.append("A")
        return Message.text_message("a done", role="assistant")

    async def b(state: AgentState):
        ran.append("B")
        return Message.text_message("b done", role="assistant")

    graph = StateGraph()
    graph.add_node("A", a)
    graph.add_node("B", b)
    graph.add_edge("A", "B")
    graph.add_edge("B", END)
    graph.set_entry_point("A")
    return graph.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_stream_closed_after_reply_node_resumes_at_next_node():
    closes = 0
    for k in range(40):
        ran: list[str] = []
        cp = InMemoryCheckpointer()
        app = _a_then_b(cp, ran)
        config = {"thread_id": f"t-ab-{k}", "user_id": "u1"}
        if not await _close_at(app, config, k, lambda r=ran: "A" in r):
            break
        closes += 1
        state = await cp.aget_state(dict(config))
        if "B" not in ran:
            assert state.execution_meta.current_node == "B", k
        if state.execution_meta.status != ExecutionStatus.COMPLETED:
            await app.ainvoke({"messages": []}, config=dict(config))
            state = await cp.aget_state(dict(config))
        assert ran == ["A", "B"], k
        assert [m.text() for m in state.context] == ["hi", "a done", "b done"], k
        await app.aclose()
    assert closes > 0


class _DurableLog(InMemoryCheckpointer):
    """Keeps a copy of every durable state save."""

    def __init__(self) -> None:
        super().__init__()
        self.durable: list[AgentState] = []

    async def aput_state(self, config, state):
        self.durable.append(state.model_copy(deep=True))
        return await super().aput_state(config, state)


@pytest.mark.asyncio
async def test_crash_before_completed_save_does_not_duplicate_reply():
    calls = _new_calls()
    cp = _DurableLog()
    app = _agent_graph(cp, calls)
    config = {"thread_id": "t-crash-last-step", "user_id": "u1"}
    await app.ainvoke({"messages": [Message.text_message("hi")]}, config=dict(config))
    await app.aclose()
    # The last RUNNING durable save is all a process killed before the completed save leaves.
    last_running = [s for s in cp.durable if s.execution_meta.status == ExecutionStatus.RUNNING][-1]
    assert _finals(last_running) == 0

    crashed = InMemoryCheckpointer()
    await crashed.aput_checkpoint(config, last_running, list(last_running.context))
    await crashed.aclear_state_cache(config)
    app = _agent_graph(crashed, calls)
    await app.ainvoke({}, config=dict(config))
    state = await crashed.aget_state(dict(config))
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert _finals(state) == 1
    assert _tool_calls(state) == 1
    await app.aclose()


@pytest.mark.asyncio
async def test_cancelled_after_last_node_clears_the_tool_ledger():
    """The cancel saves the run COMPLETED; its tool ledger rows are cleared like a normal end."""
    calls = _new_calls()
    hook = _SlowStateUpdate()
    callbacks = CallbackManager()
    callbacks.register_lifecycle_hook(hook)
    cp = InMemoryCheckpointer()
    app = _agent_graph(cp, calls, callback_manager=callbacks)
    config = {"thread_id": "t-cancel-ledger", "user_id": "u1"}
    await _cancel_invoke_during_save(hook, app, config)
    await app._task_manager.wait_for_all(timeout=5)

    state = await cp.aget_state(dict(config))
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert calls["tool"] == 1
    assert len(cp._tool_results) == 0
    await app.aclose()
