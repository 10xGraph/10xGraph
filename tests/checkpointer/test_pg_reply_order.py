"""A converted reply sorts after the user message even when the provider's clock is behind."""

from __future__ import annotations

import os
import time
import uuid
from unittest.mock import patch

import pytest
import pytest_asyncio

from tenxgraph.core.state.message import Message
from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tests.adapters.test_openai_converter import MockModelResponse


POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://test:test@localhost:5433/test_agentflow")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/1")


@pytest_asyncio.fixture
async def checkpointer():
    asyncpg = pytest.importorskip("asyncpg")
    try:
        pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=1, timeout=5)
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {e}")
    schema = f"tg_{uuid.uuid4().hex[:10]}"
    await pool.execute(f'CREATE SCHEMA "{schema}"')
    cp = PgCheckpointer(
        postgres_dsn=POSTGRES_DSN,
        redis_url=REDIS_URL,
        schema=schema,
        redis_key_prefix=f"tgorder:{uuid.uuid4().hex[:8]}:",
    )
    await cp.asetup()
    yield cp
    await cp.arelease()
    await pool.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    await pool.close()


@pytest.mark.integration
@pytest.mark.real_schema_init
@pytest.mark.asyncio
@patch("tenxgraph.runtime.adapters.llm.openai_converter.HAS_OPENAI", True)
async def test_reply_stays_after_the_user_message(checkpointer):
    cfg = {"thread_id": "t1", "user_id": "u"}
    user = Message.text_message("question", role="user")
    user.timestamp = time.time()
    # The provider says the reply was created a second before the question.
    response = MockModelResponse(
        {
            "id": "same-id",
            "created": int(user.timestamp) - 1,
            "choices": [
                {"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}
            ],
        }
    )
    reply = await OpenAIConverter().convert_response(response)

    await checkpointer.aput_messages(cfg, [user, reply])

    listed = await checkpointer.alist_messages(cfg)
    assert [m.role for m in listed] == ["user", "assistant"]
