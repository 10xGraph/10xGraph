"""
Tests for QdrantStore implementation.
"""

import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.storage.store.store_schema import MemoryType
from tenxgraph.storage.store.qdrant_store import QdrantStore
from tenxgraph.core.state.message import Message


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    async def aembed(self, text: str) -> list[float]:
        """Return a mock embedding vector."""
        # Create deterministic mock embeddings based on text
        hash_value = hash(text)
        return [float((hash_value + i) % 100) / 100.0 for i in range(self._dimension)]

    def embed(self, text: str) -> list[float]:
        """Synchronous version for compatibility."""
        import asyncio
        return asyncio.run(self.aembed(text))

    @property
    def dimension(self) -> int:
        return self._dimension


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    return MockEmbeddingService()


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = AsyncMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.create_collection.return_value = None
    client.upsert.return_value = None
    client.search.return_value = []
    client.retrieve.return_value = []
    client.delete.return_value = None
    return client


@pytest.fixture
def qdrant_store(mock_embedding_service, mock_qdrant_client):
    """Create a QdrantStore instance with mocked dependencies."""
    with patch("qdrant_client.AsyncQdrantClient") as mock_client_class:
        mock_client_class.return_value = mock_qdrant_client
        store = QdrantStore(
            embedding=mock_embedding_service,
            path="./test_data"
        )
        return store


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "user_id": "test_user",
        "thread_id": "test_thread",
        "collection": "test_collection"
    }


@pytest.fixture
def sample_message():
    """Sample message for testing."""
    return Message.text_message("Hello, this is a test message", role="user")


class TestEmbeddingService:
    """Test embedding service implementations."""

    def test_mock_embedding_service_properties(self, mock_embedding_service):
        """Test mock embedding service properties."""
        assert mock_embedding_service.dimension == 1536

    @pytest.mark.asyncio
    async def test_mock_embedding_service_embed(self, mock_embedding_service):
        """Test mock embedding generation."""
        text = "test text"
        embedding = await mock_embedding_service.aembed(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_mock_embedding_deterministic(self, mock_embedding_service):
        """Test that mock embeddings are deterministic."""
        text = "test text"
        embedding1 = await mock_embedding_service.aembed(text)
        embedding2 = await mock_embedding_service.aembed(text)
        
        assert embedding1 == embedding2


class TestQdrantStore:
    """Test QdrantStore functionality."""

    @pytest.mark.asyncio
    async def test_store_initialization(self, mock_embedding_service):
        """Test store initialization."""
        with patch("qdrant_client.AsyncQdrantClient") as mock_client:
            store = QdrantStore(
                embedding=mock_embedding_service,
                path="./test_data"
            )
            
            assert store.embedding == mock_embedding_service
            assert store.collection == "agentflow_memories"
            mock_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_asetup(self, qdrant_store, mock_qdrant_client):
        """Test async setup."""
        result = await qdrant_store.asetup()
        
        assert result is True
        mock_qdrant_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_collection_creation(self, qdrant_store, mock_qdrant_client):
        """Test collection creation when it doesn't exist."""
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])
        
        await qdrant_store._ensure_collection_exists("test_collection")
        
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_collection_creation_creates_payload_indexes(self, qdrant_store, mock_qdrant_client):
        """Test that payload indexes are created for all filterable fields on new collections."""
        from qdrant_client.http.models import PayloadSchemaType

        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        await qdrant_store._ensure_collection_exists("test_collection")

        expected_fields = ["user_id", "memory_type", "category", "memory_key"]
        assert mock_qdrant_client.create_payload_index.call_count == len(expected_fields)

        called_fields = [
            call.kwargs["field_name"]
            for call in mock_qdrant_client.create_payload_index.call_args_list
        ]
        assert called_fields == expected_fields

        for call in mock_qdrant_client.create_payload_index.call_args_list:
            assert call.kwargs["collection_name"] == "test_collection"
            assert call.kwargs["field_schema"] == PayloadSchemaType.KEYWORD

    @pytest.mark.asyncio
    async def test_payload_index_failure_is_best_effort(self, qdrant_store, mock_qdrant_client):
        """Test that individual payload index failures are caught and don't prevent caching."""
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])
        mock_qdrant_client.create_payload_index.side_effect = Exception("index creation failed")

        # Should NOT raise — individual index failures are best-effort.
        await qdrant_store._ensure_collection_exists("test_collection")

        # Collection should still be cached.
        assert "test_collection" in qdrant_store._collection_cache
        # All 4 fields should have been attempted.
        assert mock_qdrant_client.create_payload_index.call_count == 4

    @pytest.mark.asyncio
    async def test_existing_collection_ensures_payload_indexes(self, qdrant_store, mock_qdrant_client):
        """Test that payload indexes are ensured even when the collection already exists."""
        from qdrant_client.http.models import PayloadSchemaType

        existing = MagicMock()
        existing.name = "test_collection"
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[existing])

        await qdrant_store._ensure_collection_exists("test_collection")

        mock_qdrant_client.create_collection.assert_not_called()
        # Indexes should still be created (idempotent) for existing collections.
        expected_fields = ["user_id", "memory_type", "category", "memory_key"]
        assert mock_qdrant_client.create_payload_index.call_count == len(expected_fields)

        called_fields = [
            call.kwargs["field_name"]
            for call in mock_qdrant_client.create_payload_index.call_args_list
        ]
        assert called_fields == expected_fields

    @pytest.mark.asyncio
    async def test_collection_exists_cache(self, qdrant_store, mock_qdrant_client):
        """Test collection existence caching."""
        qdrant_store._collection_cache.add("test_collection")
        
        await qdrant_store._ensure_collection_exists("test_collection")
        
        # Should not call get_collections if cached
        mock_qdrant_client.get_collections.assert_not_called()

    @pytest.mark.asyncio
    async def test_astore_string_content(self, qdrant_store, mock_qdrant_client, sample_config):
        """Test storing string content."""
        content = "This is a test memory"
        
        memory_id = await qdrant_store.astore(
            config=sample_config,
            content=content,
            memory_type=MemoryType.EPISODIC,
            category="test"
        )
        
        assert isinstance(memory_id, str)
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_astore_message_content(self, qdrant_store, mock_qdrant_client, sample_config, sample_message):
        """Test storing Message content."""
        memory_id = await qdrant_store.astore(
            config=sample_config,
            content=sample_message,
            memory_type=MemoryType.EPISODIC,
            category="test"
        )
        
        assert isinstance(memory_id, str)
        mock_qdrant_client.upsert.assert_called_once()


    @pytest.mark.asyncio
    async def test_asearch(self, qdrant_store, mock_qdrant_client, sample_config):
        """Test search functionality."""
        # Mock search result
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.score = 0.95
        mock_point.payload = {
            "content": "test content",
            "user_id": "test_user",
            "thread_id": "test_thread",
            "memory_type": "episodic",
            "timestamp": "2023-01-01T00:00:00"
        }
        mock_qdrant_client.search.return_value = [mock_point]
        
        results = await qdrant_store.asearch(
            config=sample_config,
            query="test query",
            limit=10
        )
        
        assert len(results) == 1
        assert results[0].content == "test content"
        assert results[0].score == 0.95
        mock_qdrant_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_asearch_does_not_filter_by_thread_id(self, qdrant_store, mock_qdrant_client, sample_config):
        """LTM search is cross-thread; thread_id must NOT be passed to the filter."""
        mock_qdrant_client.search.return_value = []

        with patch.object(qdrant_store, "_build_qdrant_filter", wraps=qdrant_store._build_qdrant_filter) as mock_filter:
            await qdrant_store.asearch(config=sample_config, query="test")
            call_kwargs = mock_filter.call_args.kwargs
            assert "thread_id" not in call_kwargs or call_kwargs.get("thread_id") is None

    @pytest.mark.asyncio
    async def test_aget(self, qdrant_store, mock_qdrant_client, sample_config):
        """Test getting a specific memory."""
        # Mock retrieve result
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.payload = {
            "content": "test content",
            "user_id": "test_user",
            "thread_id": "test_thread",
            "memory_type": "episodic",
            "timestamp": "2023-01-01T00:00:00"
        }
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        result = await qdrant_store.aget(config=sample_config, memory_id="test_id")
        
        assert result is not None
        assert result.content == "test content"
        mock_qdrant_client.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_not_found(self, qdrant_store, mock_qdrant_client, sample_config):
        """Test getting a non-existent memory."""
        mock_qdrant_client.retrieve.return_value = []
        
        result = await qdrant_store.aget(config=sample_config, memory_id="nonexistent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_aupdate(self, qdrant_store, mock_qdrant_client, sample_config):
        """Test updating a memory."""
        # Mock existing memory
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.payload = {
            "content": "old content",
            "user_id": "test_user",
            "thread_id": "test_thread",
            "memory_type": "episodic",
            "timestamp": "2023-01-01T00:00:00"
        }
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        await qdrant_store.aupdate(
            config=sample_config,
            memory_id="test_id",
            content="new content"
        )
        
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_aupdate_not_found(self, qdrant_store, mock_qdrant_client, sample_config):
        """Test updating a non-existent memory."""
        mock_qdrant_client.retrieve.return_value = []
        
        with pytest.raises(ValueError, match="Memory test_id not found"):
            await qdrant_store.aupdate(
                config=sample_config,
                memory_id="test_id",
                content="new content"
            )

    @pytest.mark.asyncio
    async def test_adelete(self, qdrant_store, mock_qdrant_client, sample_config):
        """Test deleting a memory."""
        # Mock existing memory
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.payload = {
            "content": "test content",
            "user_id": "test_user",
            "thread_id": "test_thread",
            "memory_type": "episodic",
            "timestamp": "2023-01-01T00:00:00"
        }
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        await qdrant_store.adelete(config=sample_config, memory_id="test_id")
        
        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_adelete_not_found(self, qdrant_store, mock_qdrant_client, sample_config):
        """Test deleting a non-existent memory."""
        mock_qdrant_client.retrieve.return_value = []
        
        with pytest.raises(ValueError, match="Memory test_id not found"):
            await qdrant_store.adelete(config=sample_config, memory_id="test_id")

    @pytest.mark.asyncio
    async def test_aforget_memory(self, qdrant_store, mock_qdrant_client, sample_config):
        """Test forgetting all memories for a user/agent."""
        await qdrant_store.aforget_memory(config=sample_config)
        
        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_aforget_memory_no_filter(self, qdrant_store, mock_qdrant_client):
        """Test forgetting memories with no user/agent specified."""
        config = {}  # No user_id or agent_id
        
        await qdrant_store.aforget_memory(config=config)
        
        # Should not call delete if no filter criteria
        mock_qdrant_client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_arelease(self, qdrant_store, mock_qdrant_client):
        """Test resource cleanup."""
        mock_qdrant_client.close = AsyncMock()
        
        await qdrant_store.arelease()
        
        mock_qdrant_client.close.assert_called_once()

    def test_distance_metric_conversion(self, qdrant_store):
        """Test distance metric conversion."""
        from tenxgraph.storage.store.store_schema import DistanceMetric
        from qdrant_client.http.models import Distance
        
        assert qdrant_store._distance_metric_to_qdrant(DistanceMetric.COSINE) == Distance.COSINE
        assert qdrant_store._distance_metric_to_qdrant(DistanceMetric.EUCLIDEAN) == Distance.EUCLID
        assert qdrant_store._distance_metric_to_qdrant(DistanceMetric.DOT_PRODUCT) == Distance.DOT
        assert qdrant_store._distance_metric_to_qdrant(DistanceMetric.MANHATTAN) == Distance.MANHATTAN

    def test_extract_config_values(self, qdrant_store):
        """Test config value extraction."""
        config = {
            "user_id": "user123",
            "thread_id": "test_thread",
            "collection": "custom_collection"
        }

        user_id, thread_id, collection = qdrant_store._extract_config_values(config)

        assert user_id == "user123"
        assert thread_id == "test_thread"
        assert collection == "custom_collection"

    def test_extract_config_values_defaults(self, qdrant_store):
        """Test config value extraction with defaults."""
        config = {}

        user_id, thread_id, collection = qdrant_store._extract_config_values(config)

        assert user_id is None
        assert thread_id is None
        assert collection == qdrant_store.collection

    def test_prepare_content_string(self, qdrant_store):
        """Test content preparation with string."""
        text = "Hello world"
        result = qdrant_store._prepare_content(text)
        assert result == "Hello world"

    def test_prepare_content_message(self, qdrant_store, sample_message):
        """Test content preparation with Message."""
        result = qdrant_store._prepare_content(sample_message)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_point_to_search_result(self, qdrant_store):
        """Test conversion of Qdrant point to MemorySearchResult."""
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.score = 0.85
        mock_point.payload = {
            "content": "test content",
            "user_id": "user123",
            "thread_id": "test_thread",
            "memory_type": "episodic",
            "timestamp": "2023-01-01T00:00:00"
        }
        
        result = qdrant_store._point_to_search_result(mock_point)
        
        assert result.id == "test_id"
        assert result.content == "test content"
        assert result.score == 0.85
        assert result.user_id == "user123"
        assert result.thread_id == "test_thread"
        assert result.memory_type == MemoryType.EPISODIC

    def test_build_qdrant_filter(self, qdrant_store):
        """Test Qdrant filter building."""
        filter_obj = qdrant_store._build_qdrant_filter(
            user_id="user123",
            thread_id="test_thread",
            memory_type=MemoryType.EPISODIC,
            category="test",
            filters={"custom_field": "custom_value"}
        )
        
        assert filter_obj is not None
        assert len(filter_obj.must) == 5  # user_id, agent_id, memory_type, category, custom_field

    def test_build_qdrant_filter_empty(self, qdrant_store):
        """Test Qdrant filter building with no conditions."""
        filter_obj = qdrant_store._build_qdrant_filter()
        
        assert filter_obj is None


class TestConvenienceFunctions:
    """Test convenience factory functions."""

    def test_create_local_qdrant_store(self, mock_embedding_service):
        """Test local store creation."""
        with patch("qdrant_client.AsyncQdrantClient"):
            from tenxgraph.storage.store.qdrant_store import create_local_qdrant_store
            
            store = create_local_qdrant_store(
                path="./test_data",
                embedding=mock_embedding_service
            )
            
            assert isinstance(store, QdrantStore)

    def test_create_remote_qdrant_store(self, mock_embedding_service):
        """Test remote store creation."""
        with patch("qdrant_client.AsyncQdrantClient"):
            from tenxgraph.storage.store.qdrant_store import create_remote_qdrant_store
            
            store = create_remote_qdrant_store(
                host="localhost",
                port=6333,
                embedding=mock_embedding_service
            )
            
            assert isinstance(store, QdrantStore)

    def test_create_cloud_qdrant_store(self, mock_embedding_service):
        """Test cloud store creation."""
        with patch("qdrant_client.AsyncQdrantClient"):
            from tenxgraph.storage.store.qdrant_store import create_cloud_qdrant_store
            
            store = create_cloud_qdrant_store(
                url="https://test.qdrant.io",
                api_key="test-key",
                embedding=mock_embedding_service
            )
            
            assert isinstance(store, QdrantStore)


class TestSyncMethods:
    """Test synchronous wrapper methods."""

    def test_store_sync(self, qdrant_store, sample_config):
        """Test synchronous store method."""
        with patch.object(qdrant_store, 'astore') as mock_astore:
            mock_astore.return_value = "test_id"
            
            result = qdrant_store.store(
                config=sample_config,
                content="test content"
            )
            
            assert result == "test_id"
            mock_astore.assert_called_once()

    def test_search_sync(self, qdrant_store, sample_config):
        """Test synchronous search method."""
        with patch.object(qdrant_store, 'asearch') as mock_asearch:
            mock_asearch.return_value = []
            
            result = qdrant_store.search(
                config=sample_config,
                query="test query"
            )
            
            assert result == []
            mock_asearch.assert_called_once()

    def test_get_sync(self, qdrant_store, sample_config):
        """Test synchronous get method."""
        with patch.object(qdrant_store, 'aget') as mock_aget:
            mock_aget.return_value = None
            
            result = qdrant_store.get(
                config=sample_config,
                memory_id="test_id"
            )
            
            assert result is None
            mock_aget.assert_called_once()

    def test_update_sync(self, qdrant_store, sample_config):
        """Test synchronous update method."""
        with patch.object(qdrant_store, 'aupdate') as mock_aupdate:
            mock_aupdate.return_value = None
            
            qdrant_store.update(
                config=sample_config,
                memory_id="test_id",
                content="new content"
            )
            
            mock_aupdate.assert_called_once()

    def test_delete_sync(self, qdrant_store, sample_config):
        """Test synchronous delete method."""
        with patch.object(qdrant_store, 'adelete') as mock_adelete:
            mock_adelete.return_value = None
            
            qdrant_store.delete(
                config=sample_config,
                memory_id="test_id"
            )
            
            mock_adelete.assert_called_once()

    def test_forget_memory_sync(self, qdrant_store, sample_config):
        """Test synchronous forget_memory method."""
        with patch.object(qdrant_store, 'aforget_memory') as mock_aforget:
            mock_aforget.return_value = None
            
            qdrant_store.forget_memory(config=sample_config)
            
            mock_aforget.assert_called_once()


@pytest.mark.integration
class TestQdrantStoreIntegration:
    """Integration tests for QdrantStore (requires actual Qdrant instance)."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow with real embedding service."""
        # This test would require a real Qdrant instance and API keys
        # Skip if dependencies not available
        pytest.skip("Integration test requires real Qdrant instance")

    @pytest.mark.asyncio
    async def test_large_batch_store(self):
        """Test storing large batches of memories."""
        # This test would verify performance with large datasets
        pytest.skip("Integration test requires real Qdrant instance")

    @pytest.mark.asyncio
    async def test_complex_search_scenarios(self):
        """Test complex search scenarios with filters and thresholds."""
        # This test would verify advanced search functionality
        pytest.skip("Integration test requires real Qdrant instance")


if __name__ == "__main__":
    pytest.main([__file__])