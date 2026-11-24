"""
Comprehensive tests for PgCheckpointer implementation.

Tests cover:
- Schema creation and setup
- State persistence and caching
- Message storage and retrieval
- Thread management
- Error handling and retry logic
- Resource cleanup
- Async and sync variants
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.storage.checkpointer.pg_checkpointer import PgCheckpointer
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils.thread_info import ThreadInfo


class _Txn:
    """Minimal async context manager standing in for asyncpg's transaction()."""

    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class TestPgCheckpointer:
    """Test suite for PgCheckpointer."""

    @pytest.fixture
    def mock_pg_pool(self):
        """Mock PostgreSQL pool."""
        pool = MagicMock()
        connection = AsyncMock()
        # conn.transaction() must return an async context manager, not a coroutine.
        connection.transaction = MagicMock(return_value=_Txn())
        pool.is_closing.return_value = False

        # Create a mock async context manager for pool.acquire()
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=connection)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)

        # Make pool.acquire() return the async context manager
        pool.acquire.return_value = async_context_manager

        return pool

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis connection."""
        redis = AsyncMock()
        return redis

    @pytest.fixture
    def checkpointer(self, mock_pg_pool, mock_redis):
        """Create PgCheckpointer with mocked dependencies."""
        with patch("asyncpg.create_pool", return_value=mock_pg_pool):
            cp = PgCheckpointer(
                postgres_dsn="postgresql://test:test@localhost/test",
                redis_url="redis://localhost:6379/0",
                user_id_type="string",
                cache_ttl=3600,
            )
            # Replace with our mocks
            cp._pg_pool = mock_pg_pool
            cp.redis = mock_redis
            return cp

    @pytest.fixture
    def sample_state(self):
        """Create sample AgentState for testing."""
        state = AgentState()
        state.context = [
            Message.text_message("Hello", role="user", message_id="msg1"),
            Message.text_message("Hi there!", role="assistant", message_id="msg2"),
        ]
        return state

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "thread_id": "thread_123",
            "user_id": "user_456",
            "thread_name": "Test Thread",
            "meta": {"test": "data"},
        }


    def test_initialization_invalid(self):
        """Test invalid initialization scenarios."""
        # Missing PostgreSQL connection
        with pytest.raises(ValueError, match="Either postgres_dsn or pg_pool must be provided"):
            PgCheckpointer(redis_url="redis://localhost:6379/0")

        # Missing Redis connection
        with pytest.raises(
            ValueError, match="Either redis_url, redis_pool or redis instance must be provided"
        ):
            PgCheckpointer(postgres_dsn="postgresql://test:test@localhost/test")


    def test_schema_creation(self, checkpointer):
        """Test database schema creation."""
        sql_statements = [sql for _table, sql in checkpointer._build_table_statements()]

        assert len(sql_statements) >= 3  # the enum type and the tables

        # Check that key statements are present
        sql_text = " ".join(sql_statements)
        assert "CREATE TYPE message_role" in sql_text
        assert 'CREATE TABLE IF NOT EXISTS "public"."threads"' in sql_text
        assert 'CREATE TABLE IF NOT EXISTS "public"."states"' in sql_text
        assert 'CREATE TABLE IF NOT EXISTS "public"."messages"' in sql_text
        assert checkpointer._base_index_specs()

    @pytest.mark.real_schema_init
    @pytest.mark.asyncio
    async def test_setup_async(self, checkpointer, mock_pg_pool):
        """Test async setup method."""
        connection = mock_pg_pool.acquire.return_value.__aenter__.return_value
        connection.execute = AsyncMock()
        # No schema version recorded yet -> migrations run from scratch.
        connection.fetchrow = AsyncMock(return_value=None)

        await checkpointer.asetup()

        assert checkpointer._schema_initialized
        assert connection.execute.call_count >= 6  # Multiple SQL statements


    @pytest.mark.asyncio
    async def test_put_thread_returns_created_flag(self, checkpointer, sample_config, mock_pg_pool, mock_redis):
        """aput_thread should return True when inserted, False on update."""
        connection = mock_pg_pool.acquire.return_value.__aenter__.return_value
        connection.fetchrow = AsyncMock(return_value={"thread_id": "thread_123"})

        thread_info = ThreadInfo(thread_id="thread_123", thread_name="Test Thread")
        created = await checkpointer.aput_thread(sample_config, thread_info)
        assert created is True

        # Now simulate conflict -> should perform an owner-scoped update, return False.
        # The UPDATE uses RETURNING, so it must yield a row (the thread is ours).
        connection.fetchrow = AsyncMock(
            side_effect=[None, {"thread_id": "thread_123"}]
        )
        connection.execute = AsyncMock()

        thread_info2 = ThreadInfo(thread_id="thread_123", thread_name=None, metadata={"a": 1})
        created2 = await checkpointer.aput_thread(sample_config, thread_info2)
        assert created2 is False
        # Ensure we ran the meta-only update path, scoped by user_id, when thread_name is None
        update_sql = [str(call.args[0]) for call in connection.fetchrow.mock_calls if call.args]
        assert any("SET meta = $1, updated_at = NOW()" in sql for sql in update_sql)
        assert any("WHERE thread_id = $2 AND user_id = $3" in sql for sql in update_sql)

        # Test get_thread
        connection.fetchrow = AsyncMock()
        connection.fetchrow.return_value = {
            "thread_id": "thread_123",
            "thread_name": "Test Thread",
            "user_id": "user_456",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "meta": {"test": "data"},
        }

        thread = await checkpointer.aget_thread(sample_config)
        assert thread.thread_name == "Test Thread"

        # Test list_threads
        connection.fetch.return_value = [connection.fetchrow.return_value]
        threads = await checkpointer.alist_threads(sample_config)
        assert len(threads) == 1

        # Test clean_thread
        await checkpointer.aclean_thread(sample_config)
        connection.execute.assert_called()
        mock_redis.delete.assert_called()


    @pytest.mark.asyncio
    async def test_error_handling(self, checkpointer, sample_config):
        """Test error handling in various scenarios."""
        # Test missing required config
        with pytest.raises(ValueError):
            await checkpointer.aput_state({}, AgentState())

        with pytest.raises(ValueError):
            await checkpointer.aget_state({})

    @pytest.mark.asyncio
    async def test_retry_logic(self, checkpointer):
        """Test connection retry logic."""
        import asyncpg

        # Mock operation that fails twice then succeeds
        operation = AsyncMock()
        operation.side_effect = [
            asyncpg.PostgresConnectionError("Connection lost"),
            asyncpg.PostgresConnectionError("Still down"),
            "Success",
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await checkpointer._retry_on_connection_error(operation)
            assert result == "Success"
            assert operation.call_count == 3

        # Test max retries exceeded
        operation.reset_mock()
        operation.side_effect = [asyncpg.PostgresConnectionError("Connection lost")] * 5

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(asyncpg.PostgresConnectionError):
                await checkpointer._retry_on_connection_error(operation, max_retries=3)
            assert operation.call_count == 3

    @pytest.mark.asyncio
    async def test_resource_cleanup_errors(self, checkpointer, mock_pg_pool, mock_redis):
        """Test resource cleanup with errors."""
        mock_redis.aclose = AsyncMock(side_effect=Exception("Redis error"))
        mock_pg_pool.close = AsyncMock(side_effect=Exception("PG error"))

        # Should not raise - cleanup is best effort
        await checkpointer.arelease()


@pytest.mark.integration
class TestPgCheckpointerIntegration:
    """Integration tests requiring real PostgreSQL and Redis connections."""

    @pytest.fixture
    async def real_checkpointer(self):
        """Create a PgCheckpointer with real database connections."""
        checkpointer = PgCheckpointer(
            database_url="postgresql://postgres:password@localhost:5432/test_db",
            redis_url="redis://localhost:6379/1",  # Use different DB for testing
        )
        await checkpointer.asetup()
        try:
            yield checkpointer
        finally:
            await checkpointer.arelease()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
