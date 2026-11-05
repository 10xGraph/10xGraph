"""D6: compile() refuses (MODE=production) or warns when the checkpointer and the store
declare different tenant columns."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tenxgraph.core.exceptions import GraphError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.graph.utils.tenant_check import check_tenant_columns
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.store.pgvector_store import PgVectorStore
from tenxgraph.utils import END


def _pg_cp(**kw):
    return PgCheckpointer(postgres_dsn="postgresql://x/y", redis=MagicMock(), **kw)


def _pg_store(**kw):
    return PgVectorStore(None, pool=MagicMock(), **kw)


def _graph():
    graph = StateGraph()
    graph.add_node("n", lambda state: state)
    graph.set_entry_point("n")
    graph.add_edge("n", END)
    return graph


@pytest.fixture
def production(monkeypatch):
    monkeypatch.setenv("MODE", "production")


def test_same_columns_pass(production):
    check_tenant_columns(
        _pg_cp(tenant_columns={"org_id": "uuid"}), _pg_store(tenant_columns={"org_id": "uuid"})
    )


def test_nothing_declared_passes(production):
    check_tenant_columns(_pg_cp(), _pg_store())
    check_tenant_columns(InMemoryCheckpointer(), MagicMock())
    check_tenant_columns(_pg_cp(tenant_columns=["org_id"]), None)  # no store: nothing to compare


@pytest.mark.parametrize(
    ("cp_cols", "store_cols"),
    [
        ({"org_id": "text"}, None),
        (None, {"org_id": "text"}),
        ({"org_id": "text"}, {"project_id": "text"}),
        ({"org_id": "text"}, {"org_id": "uuid"}),
        ({"org_id": "text"}, {"org_id": "text", "project_id": "text"}),
    ],
)
def test_mismatch_refused_in_production(production, cp_cols, store_cols):
    with pytest.raises(GraphError) as err:
        check_tenant_columns(_pg_cp(tenant_columns=cp_cols), _pg_store(tenant_columns=store_cols))
    assert err.value.error_code == "GRAPH_TENANT_001"


@pytest.mark.parametrize("mode", ["", "development", "dev"])
def test_mismatch_warns_outside_production(monkeypatch, caplog, mode):
    monkeypatch.setenv("MODE", mode)
    with caplog.at_level(logging.WARNING, logger="tenxgraph.graph"):
        check_tenant_columns(_pg_cp(tenant_columns=["org_id"]), _pg_store())
    assert "Declare the same tenant_columns on both" in caplog.text


def test_store_without_tenant_support_against_tenant_checkpointer(production):
    qdrant_like = SimpleNamespace()  # no tenant_column_names attribute
    with pytest.raises(GraphError):
        check_tenant_columns(_pg_cp(tenant_columns=["org_id"]), qdrant_like)


def test_compile_runs_the_check(production):
    with pytest.raises(GraphError, match="tenant columns"):
        _graph().compile(store=_pg_store(tenant_columns=["org_id"]))
    _graph().compile(store=_pg_store())  # InMemoryCheckpointer and no tenant columns: fine


def test_allow_mismatch_never_raises_and_logs_info(production, caplog):
    with caplog.at_level(logging.INFO, logger="tenxgraph.graph"):
        check_tenant_columns(_pg_cp(tenant_columns=["org_id"]), _pg_store(), allow_mismatch=True)
    assert "allowed on purpose" in caplog.text
    assert sum("allowed on purpose" in r.getMessage() for r in caplog.records) == 1


def test_compile_flag_allows_mismatch_in_production(production):
    _graph().compile(store=_pg_store(tenant_columns=["org_id"]), allow_tenant_mismatch=True)


def test_allow_mismatch_without_mismatch_is_noop(production, caplog):
    with caplog.at_level(logging.INFO, logger="tenxgraph.graph"):
        check_tenant_columns(_pg_cp(), _pg_store(), allow_mismatch=True)
        check_tenant_columns(
            _pg_cp(tenant_columns=["org_id"]),
            _pg_store(tenant_columns=["org_id"]),
            allow_mismatch=True,
        )
    assert "allowed on purpose" not in caplog.text
