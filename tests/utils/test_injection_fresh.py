"""``Inject[...]`` defaults resolve from the active container on every call.

The core used to take the checkpointer, publisher and callback manager as ``Inject[...]``
defaults without ``@inject``. Those proxies cache the first object they resolve, so the first
graph that ever ran in a process decided which checkpointer every later graph persisted to.
"""

import pytest
from injectq import Inject, InjectQ

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import BaseCheckpointer, InMemoryCheckpointer
from tenxgraph.utils.constants import END
from tenxgraph.utils.injection import fresh


class _Service:
    pass


def test_fresh_returns_explicit_values_unchanged():
    value = _Service()
    assert fresh(value) is value
    assert fresh(None) is None


def test_fresh_resolves_from_the_current_container_each_time():
    marker = Inject[_Service]
    first, second = _Service(), _Service()

    InjectQ.get_instance().bind_instance(_Service, first)
    assert fresh(marker) is first

    InjectQ.reset_instance()
    InjectQ.get_instance().bind_instance(_Service, second)
    assert fresh(marker) is second


def test_fresh_returns_none_for_an_unbound_service():
    assert fresh(Inject[_Service]) is None


def _graph(checkpointer: BaseCheckpointer):
    async def reply(state: AgentState):
        return Message.text_message("ok", role="assistant")

    graph = StateGraph()
    graph.add_node("MAIN", reply)
    graph.add_edge("MAIN", END)
    graph.set_entry_point("MAIN")
    return graph.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_each_graph_persists_to_its_own_checkpointer():
    first_checkpointer = InMemoryCheckpointer()
    config = {"thread_id": "t1"}
    await _graph(first_checkpointer).ainvoke(
        {"messages": [Message.text_message("hi")]}, config=dict(config)
    )
    assert await first_checkpointer.aget_state(config) is not None

    InjectQ.reset_instance()
    second_checkpointer = InMemoryCheckpointer()
    await _graph(second_checkpointer).ainvoke(
        {"messages": [Message.text_message("hi")]}, config=dict(config)
    )
    assert await second_checkpointer.aget_state(config) is not None


def test_fresh_returns_none_for_an_unbound_abstract_service():
    from tenxgraph.runtime.publisher.base_publisher import BasePublisher

    assert fresh(Inject[BasePublisher]) is None
