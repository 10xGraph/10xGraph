"""A cancelled or failed run leaves a thread the next message can use.

Cancelling ``ainvoke`` / ``astream`` mid-tool or mid-model-call used to save nothing durable
and leave a cache that held an assistant tool call with no answer. The finalizer now saves the
state as INTERRUPTED (reason ``cancelled``), closes unanswered tool calls, and re-raises the
cancel.
"""

import asyncio
import json

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, ExecutionStatus, Message, ToolCallBlock
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END


FULL = ResponseGranularity.FULL


class Harness:
    """A scripted model and a tool that can be held open, plus the graph around them."""

    def __init__(self, mode: str = "ainvoke"):
        self.mode = mode
        self.checkpointer = InMemoryCheckpointer()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.histories: list[list[Message]] = []
        self.app = self._build()

    def _build(self):
        async def slow_tool() -> str:
            self.started.set()
            await self.release.wait()
            return "tool done"

        async def model(state: AgentState):
            self.histories.append(list(state.context))
            last = state.context[-1]
            if last.role == "tool":
                return Message.text_message("finished", role="assistant")
            text = last.text()
            if text == "use tool":
                return Message(
                    role="assistant",
                    content=[ToolCallBlock(id="c1", name="slow_tool", args={})],
                    tools_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "slow_tool", "arguments": json.dumps({})},
                        }
                    ],
                )
            if text == "hang":
                self.started.set()
                await self.release.wait()
            return Message.text_message(f"ok:{text}", role="assistant")

        def route(state: AgentState) -> str:
            last = state.context[-1]
            if last.role == "assistant" and last.tools_calls:
                return "TOOL"
            return END

        graph = StateGraph()
        graph.add_node("MODEL", model)
        graph.add_node("TOOL", ToolNode([slow_tool]))
        graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
        graph.add_edge("TOOL", "MODEL")
        graph.set_entry_point("MODEL")
        return graph.compile(checkpointer=self.checkpointer)

    async def run(self, text: str, thread_id: str):
        data = {"messages": [Message.text_message(text)]}
        config = {"thread_id": thread_id}
        if self.mode == "ainvoke":
            return await self.app.ainvoke(data, config, response_granularity=FULL)
        return [c async for c in self.app.astream(data, config, response_granularity=FULL)]

    async def cancel_run(self, text: str, thread_id: str) -> None:
        """Start a run, wait until it is inside the tool or model, cancel it."""
        self.started.clear()
        self.release.clear()
        task = asyncio.create_task(self.run(text, thread_id))
        await asyncio.wait_for(self.started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        self.release.set()


def unanswered_calls(context: list[Message]) -> set[str]:
    calls = {c["id"] for m in context for c in (m.tools_calls or [])}
    done = {
        b.call_id for m in context if m.role == "tool" for b in m.content if hasattr(b, "call_id")
    }
    return calls - done


def assert_valid_history(context: list[Message]) -> None:
    """Every tool call is answered by tool messages that come right after it."""
    for i, msg in enumerate(context):
        if not msg.tools_calls:
            continue
        ids = {c["id"] for c in msg.tools_calls}
        after = set()
        for later in context[i + 1 :]:
            if later.role != "tool":
                break
            after |= {b.call_id for b in later.content if hasattr(b, "call_id")}
        assert ids <= after, f"unanswered tool calls {ids - after}"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_cancel_mid_tool_saves_interrupted_and_closes_tool_calls(mode):
    h = Harness(mode)
    await h.cancel_run("use tool", "t-tool")

    config = {"thread_id": "t-tool"}
    for state in (
        await h.checkpointer.aget_state(config),
        await h.checkpointer.aget_state_cache(config),
    ):
        assert state is not None
        assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
        assert state.execution_meta.interrupt_reason == "cancelled"
        assert unanswered_calls(state.context) == set()
        assert_valid_history(state.context)
        assert state.context[0].text() == "use tool"

    # The durable messages hold the user's message too.
    stored = await h.checkpointer.alist_messages(config)
    assert "use tool" in [m.text() for m in stored]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_cancel_mid_model_saves_interrupted(mode):
    h = Harness(mode)
    await h.cancel_run("hang", "t-model")

    config = {"thread_id": "t-model"}
    state = await h.checkpointer.aget_state(config)
    assert state is not None
    assert state.execution_meta.status == ExecutionStatus.INTERRUPTED_AFTER
    assert state.execution_meta.interrupt_reason == "cancelled"
    assert [m.text() for m in state.context] == ["hang"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
@pytest.mark.parametrize("where", ["use tool", "hang"])
async def test_next_run_after_cancel_gets_a_valid_history(mode, where):
    h = Harness(mode)
    thread = f"t-next-{mode}-{where}".replace(" ", "-")
    await h.cancel_run(where, thread)

    h.histories.clear()
    await h.run("hello", thread)

    assert h.histories, "the model was called on the next run"
    for history in h.histories:
        assert_valid_history(history)
    assert (
        h.histories[-1][-1].text() in ("hello", "tool done") or h.histories[-1][-1].role == "tool"
    )
    state = await h.checkpointer.aget_state({"thread_id": thread})
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    assert "hello" in [m.text() for m in state.context]
    assert unanswered_calls(state.context) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_failed_save_on_cancel_keeps_the_cache(mode):
    h = Harness(mode)

    async def failing_put_checkpoint(*args, **kwargs):
        raise RuntimeError("db down")

    # Let the run start normally, then make the durable save fail.
    h.started.clear()
    task = asyncio.create_task(h.run("use tool", "t-fail"))
    await asyncio.wait_for(h.started.wait(), 5)
    h.checkpointer.aput_checkpoint = failing_put_checkpoint
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    cached = await h.checkpointer.aget_state_cache({"thread_id": "t-fail"})
    assert cached is not None
    assert cached.context[0].text() == "use tool"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_error_in_tool_node_leaves_no_unanswered_tool_call(mode):
    checkpointer = InMemoryCheckpointer()

    async def model(state: AgentState):
        return Message(
            role="assistant",
            content=[ToolCallBlock(id="c1", name="boom", args={})],
            tools_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "boom", "arguments": "{}"},
                }
            ],
        )

    async def boom_node(state: AgentState):
        raise RuntimeError("tool node exploded")

    graph = StateGraph()
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", boom_node)
    graph.add_edge("MODEL", "TOOL")
    graph.add_edge("TOOL", END)
    graph.set_entry_point("MODEL")
    app = graph.compile(checkpointer=checkpointer)

    data = {"messages": [Message.text_message("go")]}
    config = {"thread_id": "t-err"}
    with pytest.raises(Exception, match="exploded"):
        if mode == "ainvoke":
            await app.ainvoke(data, config)
        else:
            _ = [c async for c in app.astream(data, config)]

    for state in (
        await checkpointer.aget_state(config),
        await checkpointer.aget_state_cache(config),
    ):
        assert state is not None
        assert state.execution_meta.status == ExecutionStatus.ERROR
        assert unanswered_calls(state.context) == set()
        assert_valid_history(state.context)


# ---------------------------------------------------------------------------- A5


@pytest.mark.asyncio
async def test_aclear_state_cache_drops_only_the_cache():
    cp = InMemoryCheckpointer()
    config = {"thread_id": "t-clear", "user_id": "u1"}
    state = AgentState(context=[Message.text_message("hi")])
    await cp.aput_state(config, state)
    await cp.aput_state_cache(config, state)

    assert await cp.aclear_state_cache(config) is True
    assert await cp.aget_state_cache(config) is None
    assert await cp.aget_state(config) is not None

    # Clearing a thread with no cache is fine.
    assert await cp.aclear_state_cache(config) is True


def test_clear_state_cache_sync_wrapper():
    cp = InMemoryCheckpointer()
    config = {"thread_id": "t-sync", "user_id": "u1"}
    cp.put_state_cache(config, AgentState(context=[Message.text_message("hi")]))
    assert cp.get_state_cache(config) is not None
    assert cp.clear_state_cache(config) is True
    assert cp.get_state_cache(config) is None


@pytest.mark.asyncio
async def test_aclear_state_cache_is_owner_checked():
    from tenxgraph.core.authz import build_authz

    def cfg(user_id):
        return {
            "thread_id": "t-own",
            "user_id": user_id,
            "authz": build_authz(user_id, scope="owner"),
        }

    cp = InMemoryCheckpointer()
    await cp.aput_state_cache(cfg("A"), AgentState(context=[Message.text_message("hi")]))

    await cp.aclear_state_cache(cfg("B"))  # not the owner: no-op
    assert await cp.aget_state_cache(cfg("A")) is not None

    await cp.aclear_state_cache(cfg("A"))
    assert await cp.aget_state_cache(cfg("A")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_call_without_input_after_cancel_resumes_at_the_saved_node(mode):
    """Decision D1: no new input resumes where the cancelled run was; no message is re-sent."""
    h = Harness(mode)
    await h.cancel_run("hang", "t-resume")

    h.histories.clear()
    config = {"thread_id": "t-resume"}
    if mode == "ainvoke":
        await h.app.ainvoke({}, config)
    else:
        _ = [c async for c in h.app.astream({}, config)]

    assert [m.text() for m in h.histories[0]] == ["hang"]
    state = await h.checkpointer.aget_state(config)
    assert state.execution_meta.status == ExecutionStatus.COMPLETED
    stored = [m.text() for m in await h.checkpointer.alist_messages(config)]
    assert stored.count("hang") == 1
