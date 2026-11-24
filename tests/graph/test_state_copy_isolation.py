"""The engine copies only what isolation needs (P-1); in-place changes still never leak.

Saved states, the ``old_state`` an ``on_state_update`` hook gets and parallel tool branches
share the message objects (messages are not changed once written) but own their message
list and every other field, so a node that appends to ``state.context`` or changes a custom
list or dict in place cannot reach them.
"""

import json

import pytest
from pydantic import Field

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message, ToolCallBlock
from tenxgraph.core.state.agent_state import copy_state
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import END
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleHook


class NotesState(AgentState):
    notes: list[str] = Field(default_factory=list)
    data: dict[str, list[str]] = Field(default_factory=dict)


def test_copy_state_shares_messages_and_owns_everything_else():
    state = NotesState(context=[Message.text_message("one")], notes=["a"], data={"k": ["v"]})
    copied = copy_state(state)

    assert copied == state
    assert type(copied) is NotesState
    assert copied.context is not state.context
    assert copied.context[0] is state.context[0]

    state.context.append(Message.text_message("two"))
    state.notes.append("b")
    state.data["k"].append("w")
    state.execution_meta.step = 7
    assert len(copied.context) == 1
    assert copied.notes == ["a"]
    assert copied.data == {"k": ["v"]}
    assert copied.execution_meta.step != 7


@pytest.mark.asyncio
async def test_in_place_changes_in_a_node_do_not_reach_the_saved_state():
    cp = InMemoryCheckpointer()
    cfg = {"thread_id": "inplace"}
    seen: dict[str, NotesState] = {}

    async def first(state: NotesState):
        state.notes.append("first")
        state.data["k"] = ["first"]
        return state

    async def second(state: NotesState):
        # The checkpoint of the first step is saved by now; change everything in place.
        state.context.append(Message.text_message("in place", role="assistant"))
        state.notes.append("second")
        state.data["k"].append("second")
        seen["saved"] = await cp.aget_state(cfg)
        seen["cached"] = await cp.aget_state_cache(cfg)

    graph = StateGraph(NotesState())
    graph.add_node("FIRST", first)
    graph.add_node("SECOND", second)
    graph.add_edge("FIRST", "SECOND")
    graph.add_edge("SECOND", END)
    graph.set_entry_point("FIRST")
    app = graph.compile(checkpointer=cp)

    await app.ainvoke({"messages": [Message.text_message("go")]}, cfg)

    for stored in (seen["saved"], seen["cached"]):
        assert stored.notes == ["first"]
        assert stored.data == {"k": ["first"]}
        assert [m.text() for m in stored.context] == ["go"]
    final = await cp.aget_state(cfg)
    assert final.notes == ["first", "second"]
    assert final.data == {"k": ["first", "second"]}


@pytest.mark.asyncio
async def test_on_state_update_old_state_is_not_changed_by_the_node():
    snapshots: list[tuple[list[str], list[str], int]] = []

    class Observe(GraphLifecycleHook):
        async def on_state_update(self, ctx, node_name, old_state, new_state, step):
            if node_name != "NODE":
                return
            snapshots.append(
                (list(old_state.notes), [m.text() for m in old_state.context], len(new_state.notes))
            )

    async def node(state: NotesState):
        state.notes.append("changed")
        state.context.append(Message.text_message("added", role="assistant"))
        return state

    mgr = CallbackManager()
    mgr.register_lifecycle_hook(Observe())
    graph = StateGraph(NotesState())
    graph.add_node("NODE", node)
    graph.add_edge("NODE", END)
    graph.set_entry_point("NODE")
    app = graph.compile(checkpointer=InMemoryCheckpointer(), callback_manager=mgr)

    await app.ainvoke({"messages": [Message.text_message("go")]}, {"thread_id": "hook"})

    assert snapshots == [([], ["go"], 1)]


@pytest.mark.asyncio
async def test_parallel_tool_branches_do_not_see_each_others_in_place_changes():
    seen: dict[str, tuple[list[str], int]] = {}
    start_len: list[int] = []

    def note(label: str, state: NotesState) -> str:
        # A sibling's in-place change would show here, whichever ran first.
        seen[label] = (list(state.notes), len(state.context))
        state.notes.append(label)
        state.context.append(Message.text_message(f"from {label}", role="assistant"))
        return label

    def call(call_id: str, label: str) -> dict:
        args = json.dumps({"label": label})
        return {"id": call_id, "type": "function", "function": {"name": "note", "arguments": args}}

    async def model(state: NotesState):
        if any(m.role == "tool" for m in state.context):
            return None
        start_len.append(len(state.context) + 1)
        return Message(
            role="assistant",
            content=[
                ToolCallBlock(id="c1", name="note", args={"label": "a"}),
                ToolCallBlock(id="c2", name="note", args={"label": "b"}),
            ],
            tools_calls=[call("c1", "a"), call("c2", "b")],
        )

    def route(state: NotesState) -> str:
        last = state.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph(NotesState())
    graph.add_node("MODEL", model)
    graph.add_node("TOOL", ToolNode([note]))
    graph.add_conditional_edges("MODEL", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MODEL")
    graph.set_entry_point("MODEL")
    cp = InMemoryCheckpointer()
    app = graph.compile(checkpointer=cp)
    cfg = {"thread_id": "parallel"}

    await app.ainvoke({"messages": [Message.text_message("go")]}, cfg)

    assert seen == {"a": ([], start_len[0]), "b": ([], start_len[0])}
    final = await cp.aget_state(cfg)
    # Branch context changes are never merged back; the tool results are.
    assert not any(m.text().startswith("from ") for m in final.context)
    assert [m.role for m in final.context].count("tool") == 2


@pytest.mark.asyncio
async def test_updating_message_metadata_does_not_change_a_shared_message():
    cp = InMemoryCheckpointer()
    cfg = {"thread_id": "meta"}
    message = Message.text_message("one")
    await cp.aput_messages(cfg, [message])
    await cp.aput_state(cfg, AgentState(context=[message]))

    updated = await cp.aupdate_message_metadata(cfg, message.message_id, {"k": "v"})

    assert updated.metadata["k"] == "v"
    assert "k" not in message.metadata
    assert "k" not in (await cp.aget_state(cfg)).context[0].metadata
    assert (await cp.aget_message(cfg, message.message_id)).metadata["k"] == "v"
