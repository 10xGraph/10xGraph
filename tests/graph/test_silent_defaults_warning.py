"""G-28: the silent defaults (in-memory checkpointer, shared anonymous user) log WARNING."""

import logging

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


def _graph() -> StateGraph:
    graph = StateGraph(AgentState())
    graph.add_node("WORK", lambda state: state)
    graph.set_entry_point("WORK")
    graph.add_edge("WORK", END)
    return graph


def _messages(caplog, needle: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and needle in r.getMessage()
    ]


def test_compile_without_checkpointer_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _graph().compile()
    assert _messages(caplog, "InMemoryCheckpointer")


def test_compile_with_checkpointer_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        _graph().compile(checkpointer=InMemoryCheckpointer())
    assert not _messages(caplog, "No checkpointer given")


@pytest.mark.asyncio
async def test_run_without_user_id_warns_once_per_graph(caplog):
    app = _graph().compile(checkpointer=InMemoryCheckpointer())
    with caplog.at_level(logging.WARNING):
        await app.ainvoke({"messages": [Message.text_message("hi")]}, config={"thread_id": "t1"})
        await app.ainvoke({"messages": [Message.text_message("hi")]}, config={"thread_id": "t2"})
    assert len(_messages(caplog, "No user_id in config")) == 1


@pytest.mark.asyncio
async def test_run_with_user_id_does_not_warn(caplog):
    app = _graph().compile(checkpointer=InMemoryCheckpointer())
    with caplog.at_level(logging.WARNING):
        await app.ainvoke(
            {"messages": [Message.text_message("hi")]},
            config={"thread_id": "t1", "user_id": "u1"},
        )
    assert not _messages(caplog, "No user_id in config")
