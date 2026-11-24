"""One node runner under ainvoke and astream (engine merge, step 3).

T4: a function node's ``Command(update=..., goto=...)`` routes to ``goto`` in both modes.
T5: ``Command(state=S)`` makes S the new state and records its new messages, and an
unsupported return value fails the run with ``NODE_001``, in both modes, for sync and
async node functions. Also: ``on_error`` recovery takes a ``Message`` only.
"""

from __future__ import annotations

import pytest

from tenxgraph.core.exceptions import NodeError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import CallbackManager, InvocationType
from tenxgraph.utils.command import Command
from tenxgraph.utils.constants import END

from .tool_guard_helpers import MODES, run


class NoteState(AgentState):
    note: str = ""


def _texts(state: AgentState) -> list[str]:
    return [m.text() for m in state.context or []]


def _app(first, *, manager: CallbackManager | None = None):
    """FIRST -> SECOND -> END, plus a JUMP node a Command can route to."""

    def second(state: AgentState):
        return Message.text_message("second", role="assistant")

    def jump(state: AgentState):
        return Message.text_message("jumped", role="assistant")

    graph = StateGraph(NoteState())
    graph.add_node("FIRST", first)
    graph.add_node("SECOND", second)
    graph.add_node("JUMP", jump)
    graph.add_edge("FIRST", "SECOND")
    graph.add_edge("SECOND", END)
    graph.add_edge("JUMP", END)
    graph.set_entry_point("FIRST")
    return graph.compile(checkpointer=InMemoryCheckpointer(), callback_manager=manager)


def _both(body):
    """The same node body as a sync and an async function."""

    def sync_node(state: NoteState):
        return body(state)

    async def async_node(state: NoteState):
        return body(state)

    return {"sync": sync_node, "async": async_node}


KINDS = ["sync", "async"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
async def test_command_goto_routes_in_both_modes(mode, kind):
    first = _both(
        lambda s: Command(update=Message.text_message("first", role="assistant"), goto="JUMP")
    )[kind]
    state, _ = await run(_app(first), mode, {"thread_id": f"goto-{mode}-{kind}"})
    texts = _texts(state)
    assert texts[-2:] == ["first", "jumped"]
    assert "second" not in texts


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
async def test_command_state_becomes_the_new_state(mode, kind):
    def body(state: NoteState):
        replaced = state.model_copy(deep=True)
        replaced.note = "replaced"
        replaced.context.append(Message.text_message("from S", role="assistant"))
        return Command(state=replaced)

    state, _ = await run(_app(_both(body)[kind]), mode, {"thread_id": f"state-{mode}-{kind}"})
    assert state.note == "replaced"
    texts = _texts(state)
    assert texts.count("from S") == 1
    assert texts[-2:] == ["from S", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
async def test_command_state_messages_reach_the_caller(mode, kind):
    def body(state: NoteState):
        replaced = state.model_copy(deep=True)
        replaced.context.append(Message.text_message("from S", role="assistant"))
        return Command(state=replaced, goto="JUMP")

    app = _app(_both(body)[kind])
    config = {"thread_id": f"state-msgs-{mode}-{kind}"}
    data = {"messages": [Message.text_message("go")]}
    if mode == "ainvoke":
        result = await app.ainvoke(data, config)
        texts = [m.text() for m in result["messages"]]
    else:
        chunks = [c async for c in app.astream(data, config)]
        texts = [c.message.text() for c in chunks if c.message is not None]
    assert "from S" in texts
    assert "jumped" in texts


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
async def test_unsupported_return_value_fails_with_node_001(mode, kind):
    app = _app(_both(lambda s: 42)[kind])
    with pytest.raises(NodeError) as raised:
        await run(app, mode, {"thread_id": f"bad-{mode}-{kind}"})
    assert raised.value.error_code == "NODE_001"
    assert raised.value.context["node_name"] == "FIRST"


def _failing(state: NoteState):
    raise RuntimeError("boom")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_on_error_message_recovers_the_node(mode):
    manager = CallbackManager()

    async def recover(context, input_data, error):
        return Message.text_message("recovered", role="assistant")

    manager.register_on_error(InvocationType.AI, recover)
    state, _ = await run(_app(_failing, manager=manager), mode, {"thread_id": f"rec-{mode}"})
    assert _texts(state)[-2:] == ["recovered", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_on_error_non_message_does_not_recover(mode):
    manager = CallbackManager()

    async def recover(context, input_data, error):
        return "not a message"

    manager.register_on_error(InvocationType.AI, recover)
    with pytest.raises(NodeError) as raised:
        await run(_app(_failing, manager=manager), mode, {"thread_id": f"norec-{mode}"})
    assert raised.value.error_code == "NODE_001"
