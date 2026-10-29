"""J-1: compile() warns when a second graph reuses a checkpointer another graph holds."""

import logging

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


def _graph() -> StateGraph:
    graph = StateGraph(AgentState())

    def work(state: AgentState):
        return state

    graph.add_node("WORK", work)
    graph.set_entry_point("WORK")
    graph.add_edge("WORK", END)
    return graph


def _warned(caplog) -> bool:
    return any(
        r.levelno == logging.WARNING and "already used by another compiled graph" in r.message
        for r in caplog.records
    )


def test_second_graph_on_same_checkpointer_logs_warning(caplog):
    shared = InMemoryCheckpointer()
    first = _graph().compile(checkpointer=shared)
    with caplog.at_level(logging.WARNING):
        second = _graph().compile(checkpointer=shared)
    assert _warned(caplog)
    assert first is not second


def test_separate_checkpointers_do_not_warn(caplog):
    first = _graph().compile(checkpointer=InMemoryCheckpointer())
    with caplog.at_level(logging.WARNING):
        second = _graph().compile(checkpointer=InMemoryCheckpointer())
    assert not _warned(caplog)
    assert first is not second
