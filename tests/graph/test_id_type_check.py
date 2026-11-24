"""The first run refuses an id generator the checkpointer's id columns cannot hold."""

from unittest.mock import patch

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.graph.utils.id_type_check import check_id_type
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer, PgCheckpointer
from tenxgraph.utils.constants import END
from tenxgraph.utils.id_generator import (
    BaseIDGenerator,
    BigIntIDGenerator,
    DefaultIDGenerator,
    IntIDGenerator,
)


def _pg(id_type: str) -> PgCheckpointer:
    with patch("asyncpg.create_pool"), pytest.warns(DeprecationWarning, match="id_type"):
        return PgCheckpointer(
            postgres_dsn="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
            id_type=id_type,
        )


def _graph(generator: BaseIDGenerator | None = None) -> StateGraph:
    async def node(state: AgentState):
        return Message.text_message("ok", role="assistant")

    graph = StateGraph(id_generator=generator)
    graph.add_node("N", node)
    graph.add_edge("N", END)
    graph.set_entry_point("N")
    return graph


def test_declared_id_type_is_none_without_a_declaration():
    assert InMemoryCheckpointer().declared_id_type is None


@pytest.mark.parametrize(
    ("id_type", "generator"),
    [
        ("string", DefaultIDGenerator()),
        ("int", IntIDGenerator()),
        ("bigint", BigIntIDGenerator()),
        ("bigint", IntIDGenerator()),
    ],
)
def test_matching_generator_passes(id_type, generator):
    check_id_type(_pg(id_type), generator)


def test_no_checkpointer_declaration_never_raises():
    check_id_type(InMemoryCheckpointer(), BigIntIDGenerator())


def test_explicit_id_type_that_disagrees_with_the_generator_raises():
    with pytest.raises(
        ValueError, match=r"makes bigint ids.*stores string ids.*id_type=\"bigint\""
    ):
        check_id_type(_pg("string"), BigIntIDGenerator())
    with pytest.raises(ValueError, match=r"makes string ids.*stores bigint ids"):
        check_id_type(_pg("bigint"), DefaultIDGenerator())
    with pytest.raises(ValueError, match=r"id_type=\"bigint\""):
        check_id_type(_pg("int"), BigIntIDGenerator())


def test_compile_does_not_check_the_generator():
    _graph(BigIntIDGenerator()).compile(checkpointer=_pg("string"))


@pytest.mark.asyncio
async def test_first_run_checks_the_generator_bound_after_compile():
    graph = _graph()
    app = graph.compile(checkpointer=_pg("bigint"))
    with pytest.raises(ValueError, match="makes string ids"):
        await app.ainvoke({"messages": [Message.text_message("hi")]}, {"thread_id": 1})
    # a bootstrap (the API server) binds its own generator after compile
    graph._container.bind_instance(BaseIDGenerator, BigIntIDGenerator())
    app._id_type_checked = False
    app._begin_run({"thread_id": 1, "run_id": "r"}, "invoke", heartbeat=False)
