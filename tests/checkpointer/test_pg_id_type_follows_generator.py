"""The id column type follows the bound id generator; a passed ``id_type`` is deprecated."""

import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from injectq import InjectQ

from tenxgraph.core.state import Message
from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.utils.id_generator import (
    BaseIDGenerator,
    BigIntIDGenerator,
    IDType,
    IntIDGenerator,
    TimestampIDGenerator,
)


DSN = "postgresql://test:test@localhost/test"
REDIS = "redis://localhost:6379/0"


def _bind(generator) -> None:
    InjectQ.get_instance().bind_instance(BaseIDGenerator, generator, allow_concrete=True)


def _mock_cp(**kwargs) -> PgCheckpointer:
    with patch("asyncpg.create_pool"):
        return PgCheckpointer(postgres_dsn=DSN, redis_url=REDIS, **kwargs)


def _threads_sql(cp: PgCheckpointer) -> str:
    return next(sql for table, sql in cp._build_table_statements() if table == "threads")


def test_no_generator_means_string():
    assert _mock_cp().id_type == IDType.STRING


def test_generator_bound_after_construction_is_used():
    cp = _mock_cp()
    _bind(BigIntIDGenerator())
    assert cp.id_type == IDType.BIGINT
    assert "thread_id BIGSERIAL PRIMARY KEY" in _threads_sql(cp)
    _bind(IntIDGenerator())
    assert cp.id_type == IDType.INTEGER


def test_column_type_for_int_and_timestamp_generators():
    cp = _mock_cp()
    _bind(IntIDGenerator())
    assert cp.id_type == IDType.INTEGER
    assert "thread_id SERIAL PRIMARY KEY" in _threads_sql(cp)
    _bind(TimestampIDGenerator())
    assert cp.id_type == IDType.BIGINT
    assert "thread_id BIGSERIAL PRIMARY KEY" in _threads_sql(cp)


def test_id_type_is_fixed_once_the_schema_is_initialized():
    cp = _mock_cp()
    _bind(BigIntIDGenerator())
    cp._pinned_id_type = cp._generator_id_type()
    _bind(IntIDGenerator())
    assert cp.id_type == IDType.BIGINT


def test_passed_id_type_warns_and_wins():
    _bind(BigIntIDGenerator())
    with pytest.warns(DeprecationWarning, match="id_type is deprecated and removed in 2.0"):
        cp = _mock_cp(id_type="string")
    assert cp.id_type == IDType.STRING
    with pytest.warns(DeprecationWarning):
        assert _mock_cp(id_type="integer").id_type == IDType.INTEGER


def test_no_id_type_does_not_warn(recwarn):
    _mock_cp()
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


@pytest.mark.asyncio
async def test_existing_text_table_with_bigint_generator_is_refused():
    cp = _mock_cp()
    _bind(BigIntIDGenerator())
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value="character varying")
    with pytest.raises(ValueError, match=r"threads.*thread_id.*CHARACTER VARYING.*bigint ids"):
        await cp._check_existing_id_types(conn)


@pytest.mark.asyncio
async def test_missing_tables_pass_the_check():
    cp = _mock_cp()
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=None)
    await cp._check_existing_id_types(conn)


# ----------------------------------------------------------------------------- real Postgres

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://test:test@localhost:5433/test_agentflow")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/1")


@pytest_asyncio.fixture
async def pg_schema():
    asyncpg = pytest.importorskip("asyncpg")
    try:
        pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=2, timeout=5)
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {e}")
    name = f"tg_{uuid.uuid4().hex[:10]}"
    await pool.execute(f'CREATE SCHEMA "{name}"')
    yield pool, name
    await pool.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
    await pool.close()


def _real_cp(schema: str, **kwargs) -> PgCheckpointer:
    return PgCheckpointer(
        postgres_dsn=POSTGRES_DSN,
        redis_url=REDIS_URL,
        schema=schema,
        redis_key_prefix=f"tgidtype:{uuid.uuid4().hex[:8]}:",
        **kwargs,
    )


async def _column_type(pool, schema: str, table: str, column: str) -> str:
    return await pool.fetchval(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_schema = $1 AND table_name = $2 AND column_name = $3",
        schema,
        table,
        column,
    )


@pytest.mark.integration
@pytest.mark.real_schema_init
@pytest.mark.asyncio
async def test_real_no_generator_makes_text_columns(pg_schema):
    pool, schema = pg_schema
    cp = _real_cp(schema)
    try:
        await cp.asetup()
        assert await _column_type(pool, schema, "threads", "thread_id") == "character varying"
        assert await _column_type(pool, schema, "messages", "message_id") == "character varying"
        cfg = {"thread_id": "t1", "user_id": "u"}
        await cp.aput_messages(cfg, [Message.text_message("hi", message_id="m1")])
        assert [m.text() for m in await cp.alist_messages(cfg)] == ["hi"]
    finally:
        await cp.arelease()


@pytest.mark.integration
@pytest.mark.real_schema_init
@pytest.mark.asyncio
async def test_real_generator_bound_after_construction_makes_bigint_columns(pg_schema):
    pool, schema = pg_schema
    cp = _real_cp(schema)
    _bind(BigIntIDGenerator())  # after construction, before first use
    try:
        await cp.asetup()
        assert await _column_type(pool, schema, "threads", "thread_id") == IDType.BIGINT
        assert await _column_type(pool, schema, "messages", "message_id") == IDType.BIGINT
        cfg = {"thread_id": 7, "user_id": "u"}
        message = Message.text_message("hi")
        message.message_id = 9
        await cp.aput_messages(cfg, [message])
        assert [m.message_id for m in await cp.alist_messages(cfg)] == [9]
        with pytest.raises(ValueError, match="must be an integer"):
            await cp.aget_thread({"thread_id": "7", "user_id": "u"})
    finally:
        await cp.arelease()


@pytest.mark.integration
@pytest.mark.real_schema_init
@pytest.mark.asyncio
async def test_real_passed_id_type_warns_and_is_honoured(pg_schema):
    pool, schema = pg_schema
    _bind(BigIntIDGenerator())
    with pytest.warns(DeprecationWarning, match="id_type is deprecated"):
        cp = _real_cp(schema, id_type="string")
    try:
        await cp.asetup()
        assert await _column_type(pool, schema, "threads", "thread_id") == "character varying"
    finally:
        await cp.arelease()


@pytest.mark.integration
@pytest.mark.real_schema_init
@pytest.mark.asyncio
async def test_real_existing_text_tables_with_bigint_generator_fail_at_setup(pg_schema):
    _pool, schema = pg_schema
    first = _real_cp(schema)
    try:
        await first.asetup()
    finally:
        await first.arelease()
    _bind(BigIntIDGenerator())
    second = _real_cp(schema)
    try:
        with pytest.raises(ValueError) as err:
            await second.asetup()
    finally:
        await second.arelease()
    text = str(err.value)
    assert f'"{schema}"."threads"' in text
    assert "CHARACTER VARYING" in text
    assert "bigint" in text
    assert "migrate" in text
