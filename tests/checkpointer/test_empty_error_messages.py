"""Errors with an empty str (a pool acquire TimeoutError) must still be named in logs."""

import logging
from unittest.mock import MagicMock

import pytest

from tenxgraph.core.graph.utils.utils import reload_state
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.utils.log_safety import describe_error


def test_describe_error_names_empty_errors():
    assert describe_error(TimeoutError()) == "TimeoutError()"
    assert describe_error(ValueError("boom")) == "boom"


@pytest.mark.asyncio
async def test_non_retryable_log_names_the_error(caplog):
    cp = PgCheckpointer(postgres_dsn="postgresql://x/y", redis=MagicMock())

    async def op():
        raise TimeoutError

    with caplog.at_level(logging.ERROR), pytest.raises(TimeoutError):
        await cp._retry_on_connection_error(op)
    assert any("TimeoutError" in r.getMessage() for r in caplog.records)


class _EmptyCheckpointer:
    async def aget_state_cache(self, config):
        return None

    async def aget_state(self, config):
        return None


@pytest.mark.asyncio
async def test_new_thread_reload_does_not_warn(caplog):
    old = object()
    with caplog.at_level(logging.DEBUG, logger="tenxgraph"):
        result = await reload_state({"thread_id": "t"}, old, checkpointer=_EmptyCheckpointer())
    assert result is old
    records = [r for r in caplog.records if "No existing state found" in r.getMessage()]
    assert records
    assert all(r.levelno == logging.DEBUG for r in records)
