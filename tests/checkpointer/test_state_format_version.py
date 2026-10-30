"""A state stored in a newer format is refused with a typed error, never misread (G-18)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from tenxgraph.core.exceptions import GraphError, StateFormatVersionError
from tenxgraph.core.state import AgentState
from tenxgraph.storage.checkpointer.base_checkpointer import (
    STATE_FORMAT_VERSION,
    STATE_META_KEY,
)
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer


def _row(fmt):
    data = AgentState().model_dump(mode="json")
    if fmt is not None:
        data[STATE_META_KEY] = {"format": fmt, "class": "AgentState"}
    return data


@pytest.fixture
def sqlite_cp():
    cp = SqliteCheckpointer(":memory:")
    cp.bind_state_type(AgentState)
    return cp


def _pg(monkeypatch):
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("tenxgraph.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)
    cp = PgCheckpointer(postgres_dsn="postgres://x", redis=AsyncMock())
    cp.bind_state_type(AgentState)
    return cp


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table,reader", [("af_states", "aget_state"), ("af_state_cache", "aget_state_cache")]
)
async def test_sqlite_refuses_newer_format(sqlite_cp, table, reader):
    cfg = {"thread_id": "t-new"}
    await sqlite_cp.aput_state(cfg, AgentState())
    await sqlite_cp.aput_state_cache(cfg, AgentState())
    conn = await sqlite_cp._get_conn()
    await conn.execute(
        f"UPDATE {table} SET state_data = ?", (json.dumps(_row(STATE_FORMAT_VERSION + 1)),)  # noqa: S608
    )
    await conn.commit()

    with pytest.raises(StateFormatVersionError) as exc:
        await getattr(sqlite_cp, reader)(cfg)
    err = exc.value
    assert isinstance(err, GraphError)
    assert err.error_code == "STATE_FORMAT_001"
    assert err.found == STATE_FORMAT_VERSION + 1
    assert err.supported == STATE_FORMAT_VERSION
    assert "t-new" in str(err)


@pytest.mark.asyncio
async def test_sqlite_current_old_and_missing_formats_still_load(sqlite_cp):
    cfg = {"thread_id": "t-ok"}
    await sqlite_cp.aput_state(cfg, AgentState())
    assert await sqlite_cp.aget_state(cfg) is not None
    conn = await sqlite_cp._get_conn()
    for fmt in (0, None):  # an older format, and a row with no header at all
        await conn.execute("UPDATE af_states SET state_data = ?", (json.dumps(_row(fmt)),))
        await conn.commit()
        assert await sqlite_cp.aget_state(cfg) is not None


def test_pg_payload_path_refuses_newer_format(monkeypatch):
    # Both the durable row and the Redis cache copy decode through this method.
    cp = _pg(monkeypatch)
    payload = _row(STATE_FORMAT_VERSION + 5)
    payload["_checkpoint_version"] = 3
    with pytest.raises(StateFormatVersionError):
        cp._deserialize_state_payload(payload)
    assert cp._deserialize_state_payload(_row(STATE_FORMAT_VERSION)) is not None


def test_pg_legacy_deserialize_also_refuses_newer_format(monkeypatch):
    cp = _pg(monkeypatch)
    with pytest.raises(StateFormatVersionError):
        cp._deserialize_state(_row(STATE_FORMAT_VERSION + 1), AgentState)
    with pytest.raises(StateFormatVersionError):
        cp._deserialize_state(json.dumps(_row(STATE_FORMAT_VERSION + 1)), AgentState)


def test_pg_payload_error_names_the_thread(monkeypatch):
    cp = _pg(monkeypatch)
    with pytest.raises(StateFormatVersionError, match="t-pg"):
        cp._deserialize_state_payload(_row(STATE_FORMAT_VERSION + 1), "t-pg")
