# """
# Comprehensive tests for the unified BaseStore module.

# This test file validates the unified BaseStore class and its associated data structures
# using pytest and includes both sync and async test patterns.
# """

# import asyncio
# from datetime import datetime
# from typing import Any, Dict, List, Optional, Union
# from unittest.mock import Mock
# from uuid import uuid4

# import pytest

# from tenxgraph.store.base_store import (
#     DistanceMetric,
#     MemoryRecord,
#     MemorySearchResult,
#     MemoryType,
#     MessageMemoryStore,
#     BaseStore,
# )
# from tenxgraph.utils.message import Message, TextBlock, TokenUsages


# class MockBaseStore(BaseStore[Message]):
#     """Concrete implementation for testing the unified BaseStore class."""

#     def __init__(self, embedding_function=None):
#         """Initialize mock store with in-memory storage."""
#         super().__init__(embedding_function=embedding_function)
#         self.memories: Dict[str, MemorySearchResult] = {}
#         self.collections: Dict[str, Dict[str, Any]] = {}

#     def add(
#         self,
#         content: str,
#         user_id: str = None,
#         agent_id: str = None,
#         memory_type: str = "episodic",
#         category: str = "general",
#         metadata: Dict[str, Any] = None,
#         **kwargs
#     ) -> str:
#         """Add a new memory."""
#         memory_id = str(uuid4())
        
#         memory = MemorySearchResult(
#             id=memory_id,
#             content=content,
#             score=1.0,  # New memories have perfect score
#             memory_type=memory_type,
#             metadata=metadata or {},
#             user_id=user_id,
#             agent_id=agent_id,
#             created_at=datetime.now(),
#             updated_at=datetime.now()
#         )
        
#         self.memories[memory_id] = memory
#         return memory_id

#     async def aadd(
#         self,
#         content: str,
#         user_id: str = None,
#         agent_id: str = None,
#         memory_type: str = "episodic",
#         category: str = "general",
#         metadata: Dict[str, Any] = None,
#         **kwargs
#     ) -> str:
#         """Async version of add."""
#         return self.add(content, user_id, agent_id, memory_type, category, metadata, **kwargs)

#     def search(
#         self,
#         query: str,
#         user_id: str = None,
#         agent_id: str = None,
#         memory_type: str = None,
#         category: str = None,
#         limit: int = 10,
#         score_threshold: float = None,
#         filters: Dict[str, Any] = None,
#         **kwargs
#     ) -> List[MemorySearchResult]:
#         """Search memories."""
#         results = []
        
#         for memory in self.memories.values():
#             # Apply filters
#             if user_id and memory.user_id != user_id:
#                 continue
#             if agent_id and memory.agent_id != agent_id:
#                 continue
#             if memory_type and memory.memory_type != memory_type:
#                 continue
#             if filters:
#                 skip = False
#                 for key, value in filters.items():
#                     if memory.metadata.get(key) != value:
#                         skip = True
#                         break
#                 if skip:
#                     continue
            
#             # Simple text matching for search
#             if query and query.lower() not in memory.content.lower():
#                 continue
                
#             # Apply score threshold
#             if score_threshold and memory.score < score_threshold:
#                 continue
                
#             results.append(memory)
        
#         # Sort by score and limit results
#         results.sort(key=lambda x: x.score, reverse=True)
#         return results[:limit]

#     async def asearch(
#         self,
#         query: str,
#         user_id: str = None,
#         agent_id: str = None,
#         memory_type: str = None,
#         category: str = None,
#         limit: int = 10,
#         score_threshold: float = None,
#         filters: Dict[str, Any] = None,
#         **kwargs
#     ) -> List[MemorySearchResult]:
#         """Async version of search."""
#         return self.search(query, user_id, agent_id, memory_type, category, limit, score_threshold, filters, **kwargs)

#     def get(self, memory_id: str, **kwargs) -> Optional[MemorySearchResult]:
#         """Get a specific memory by ID."""
#         return self.memories.get(memory_id)

#     async def aget(self, memory_id: str, **kwargs) -> Optional[MemorySearchResult]:
#         """Async version of get."""
#         return self.get(memory_id, **kwargs)

#     def update(
#         self,
#         memory_id: str,
#         content: str = None,
#         metadata: Dict[str, Any] = None,
#         **kwargs
#     ) -> None:
#         """Update an existing memory."""
#         if memory_id not in self.memories:
#             raise ValueError(f"Memory {memory_id} not found")
        
#         memory = self.memories[memory_id]
#         if content:
#             memory.content = content
#         if metadata:
#             memory.metadata.update(metadata)
#         memory.updated_at = datetime.now()

#     async def aupdate(
#         self,
#         memory_id: str,
#         content: str = None,
#         metadata: Dict[str, Any] = None,
#         **kwargs
#     ) -> None:
#         """Async version of update."""
#         self.update(memory_id, content, metadata, **kwargs)

#     def delete(self, memory_id: str, **kwargs) -> None:
#         """Delete a memory by ID."""
#         if memory_id in self.memories:
#             del self.memories[memory_id]
#         else:
#             raise ValueError(f"Memory {memory_id} not found")

#     async def adelete(self, memory_id: str, **kwargs) -> None:
#         """Async version of delete."""
#         self.delete(memory_id, **kwargs)


# # Test Data Fixtures
# @pytest.fixture
# def sample_message():
#     """Create a sample message for testing."""
#     return Message(
#         role="user",
#         content=[TextBlock(text="Hello, this is a test message about machine learning.")],
#         timestamp=datetime.now(),
#         usages=TokenUsages(input_tokens=10, output_tokens=0, completion_tokens=0, prompt_tokens=10, total_tokens=10)
#     )


# @pytest.fixture
# def mock_embedding_function():
#     """Mock embedding function that returns predictable embeddings."""
#     def embedding_func(texts: List[str]) -> List[List[float]]:
#         # Return simple embeddings based on text length
#         return [[float(i) for i in range(len(text.split()))] for text in texts]
#     return embedding_func


# @pytest.fixture
# def mock_store():
#     """Create a mock store for testing."""
#     return MockBaseStore()


# @pytest.fixture
# def mock_store_with_embeddings(mock_embedding_function):
#     """Create a mock store with embedding function."""
#     return MockBaseStore(embedding_function=mock_embedding_function)


# # Test Classes
# class TestDistanceMetric:
#     """Test the DistanceMetric enum."""

#     def test_distance_metric_values(self):
#         """Test that distance metrics have expected values."""
#         assert DistanceMetric.COSINE.value == "cosine"
#         assert DistanceMetric.EUCLIDEAN.value == "euclidean"
#         assert DistanceMetric.DOT_PRODUCT.value == "dot_product"
#         assert DistanceMetric.MANHATTAN.value == "manhattan"


# class TestMemoryType:
#     """Test the MemoryType enum."""

#     def test_memory_type_values(self):
#         """Test that memory types have expected values."""
#         assert MemoryType.EPISODIC.value == "episodic"
#         assert MemoryType.SEMANTIC.value == "semantic"
#         assert MemoryType.PROCEDURAL.value == "procedural"
#         assert MemoryType.ENTITY.value == "entity"
#         assert MemoryType.RELATIONSHIP.value == "relationship"
#         assert MemoryType.CUSTOM.value == "custom"


# class TestMemorySearchResult:
#     """Test the MemorySearchResult dataclass."""

#     def test_memory_search_result_creation(self):
#         """Test creating a MemorySearchResult."""
#         result = MemorySearchResult(
#             id="test_id",
#             content="Test content",
#             score=0.95
#         )
        
#         assert result.id == "test_id"
#         assert result.content == "Test content"
#         assert result.score == 0.95
#         assert result.memory_type == "episodic"  # default
#         assert result.metadata == {}  # default
#         assert result.user_id is None
#         assert result.agent_id is None

#     def test_memory_search_result_to_dict(self):
#         """Test converting MemorySearchResult to dictionary."""
#         created_at = datetime.now()
#         result = MemorySearchResult(
#             id="test_id",
#             content="Test content",
#             score=0.95,
#             memory_type="semantic",
#             metadata={"key": "value"},
#             user_id="user_123",
#             agent_id="agent_456",
#             created_at=created_at
#         )
        
#         result_dict = result.to_dict()
        
#         assert result_dict["id"] == "test_id"
#         assert result_dict["content"] == "Test content"
#         assert result_dict["score"] == 0.95
#         assert result_dict["memory_type"] == "semantic"
#         assert result_dict["metadata"] == {"key": "value"}
#         assert result_dict["user_id"] == "user_123"
#         assert result_dict["agent_id"] == "agent_456"
#         assert result_dict["created_at"] == created_at.isoformat()


# class TestMemoryRecord:
#     """Test the MemoryRecord dataclass."""

#     def test_memory_record_creation(self):
#         """Test creating a MemoryRecord."""
#         record = MemoryRecord(content="Test memory content")
        
#         assert record.content == "Test memory content"
#         assert record.user_id is None
#         assert record.agent_id is None
#         assert record.memory_type == "episodic"
#         assert record.metadata == {}
#         assert record.category == "general"
#         assert record.id is not None  # auto-generated
#         assert record.created_at is not None
#         assert record.updated_at is not None

#     def test_memory_record_from_message(self, sample_message):
#         """Test creating MemoryRecord from Message."""
#         record = MemoryRecord.from_message(
#             sample_message,
#             user_id="user_123",
#             agent_id="agent_456",
#             vector=[0.1, 0.2, 0.3],
#             additional_metadata={"source": "test"}
#         )
        
#         assert record.content == "Hello, this is a test message about machine learning."
#         assert record.user_id == "user_123"
#         assert record.agent_id == "agent_456"
#         assert record.memory_type == "episodic"
#         assert record.vector == [0.1, 0.2, 0.3]
#         assert "role" in record.metadata
#         assert "source" in record.metadata
#         assert record.metadata["role"] == "user"
#         assert record.metadata["source"] == "test"

#     def test_memory_record_to_dict(self):
#         """Test converting MemoryRecord to dictionary."""
#         record = MemoryRecord(
#             content="Test content",
#             user_id="user_123",
#             memory_type="semantic",
#             metadata={"key": "value"}
#         )
        
#         record_dict = record.to_dict()
        
#         assert record_dict["content"] == "Test content"
#         assert record_dict["user_id"] == "user_123"
#         assert record_dict["memory_type"] == "semantic"
#         assert record_dict["metadata"] == {"key": "value"}
#         assert "id" in record_dict
#         assert "created_at" in record_dict
#         assert "updated_at" in record_dict


# class TestBaseStore:
#     """Test the BaseStore abstract base class and MockBaseStore implementation."""

#     def test_base_store_initialization(self, mock_embedding_function):
#         """Test BaseStore initialization."""
#         store = MockBaseStore(embedding_function=mock_embedding_function)
        
#         assert store.embedding_function == mock_embedding_function
#         assert store.embedding_dim == 768  # default
#         assert isinstance(store.memories, dict)

#     def test_add_memory(self, mock_store):
#         """Test adding a memory."""
#         memory_id = mock_store.add(
#             content="Test content",
#             user_id="user_123",
#             memory_type="semantic",
#             metadata={"source": "test"}
#         )
        
#         assert memory_id in mock_store.memories
#         memory = mock_store.memories[memory_id]
#         assert memory.content == "Test content"
#         assert memory.user_id == "user_123"
#         assert memory.memory_type == "semantic"
#         assert memory.metadata["source"] == "test"

#     @pytest.mark.asyncio
#     async def test_aadd_memory(self, mock_store):
#         """Test async adding a memory."""
#         memory_id = await mock_store.aadd(
#             content="Async test content",
#             user_id="user_456"
#         )
        
#         assert memory_id in mock_store.memories
#         memory = mock_store.memories[memory_id]
#         assert memory.content == "Async test content"
#         assert memory.user_id == "user_456"

#     def test_search_memories(self, mock_store):
#         """Test searching memories."""
#         # Add some test memories
#         id1 = mock_store.add("Content about machine learning", user_id="user_123")
#         id2 = mock_store.add("Content about deep learning", user_id="user_123")
#         id3 = mock_store.add("Content about cooking", user_id="user_456")
        
#         # Search with text query
#         results = mock_store.search("learning", user_id="user_123")
        
#         assert len(results) == 2
#         assert all(result.user_id == "user_123" for result in results)
#         assert all("learning" in result.content.lower() for result in results)

#     def test_search_with_filters(self, mock_store):
#         """Test searching with metadata filters."""
#         # Add memories with metadata
#         mock_store.add(
#             "Test content 1", 
#             user_id="user_123", 
#             metadata={"category": "work"}
#         )
#         mock_store.add(
#             "Test content 2", 
#             user_id="user_123", 
#             metadata={"category": "personal"}
#         )
        
#         # Search with filters
#         results = mock_store.search(
#             "", 
#             user_id="user_123", 
#             filters={"category": "work"}
#         )
        
#         assert len(results) == 1
#         assert results[0].metadata["category"] == "work"

#     @pytest.mark.asyncio
#     async def test_asearch_memories(self, mock_store):
#         """Test async searching memories."""
#         await mock_store.aadd("Async content about AI", user_id="user_789")
        
#         results = await mock_store.asearch("AI", user_id="user_789")
        
#         assert len(results) == 1
#         assert results[0].user_id == "user_789"

#     def test_get_memory(self, mock_store):
#         """Test getting a specific memory."""
#         memory_id = mock_store.add("Test content for get", user_id="user_123")
        
#         memory = mock_store.get(memory_id)
        
#         assert memory is not None
#         assert memory.id == memory_id
#         assert memory.content == "Test content for get"
        
#         # Test non-existent memory
#         non_existent = mock_store.get("non_existent_id")
#         assert non_existent is None

#     @pytest.mark.asyncio
#     async def test_aget_memory(self, mock_store):
#         """Test async getting a specific memory."""
#         memory_id = await mock_store.aadd("Async test content for get")
        
#         memory = await mock_store.aget(memory_id)
        
#         assert memory is not None
#         assert memory.id == memory_id

#     def test_update_memory(self, mock_store):
#         """Test updating a memory."""
#         memory_id = mock_store.add("Original content", user_id="user_123")
        
#         mock_store.update(
#             memory_id, 
#             content="Updated content",
#             metadata={"updated": True}
#         )
        
#         memory = mock_store.get(memory_id)
#         assert memory.content == "Updated content"
#         assert memory.metadata["updated"] is True

#     @pytest.mark.asyncio
#     async def test_aupdate_memory(self, mock_store):
#         """Test async updating a memory."""
#         memory_id = await mock_store.aadd("Original async content")
        
#         await mock_store.aupdate(memory_id, content="Updated async content")
        
#         memory = await mock_store.aget(memory_id)
#         assert memory.content == "Updated async content"

#     def test_delete_memory(self, mock_store):
#         """Test deleting a memory."""
#         memory_id = mock_store.add("Content to delete")
        
#         assert memory_id in mock_store.memories
        
#         mock_store.delete(memory_id)
        
#         assert memory_id not in mock_store.memories
        
#         # Test deleting non-existent memory
#         with pytest.raises(ValueError):
#             mock_store.delete("non_existent_id")

#     @pytest.mark.asyncio
#     async def test_adelete_memory(self, mock_store):
#         """Test async deleting a memory."""
#         memory_id = await mock_store.aadd("Async content to delete")
        
#         await mock_store.adelete(memory_id)
        
#         assert memory_id not in mock_store.memories

#     def test_get_all_memories(self, mock_store):
#         """Test getting all memories with filters."""
#         mock_store.add("Content 1", user_id="user_123", memory_type="episodic")
#         mock_store.add("Content 2", user_id="user_123", memory_type="semantic")
#         mock_store.add("Content 3", user_id="user_456", memory_type="episodic")
        
#         # Get all for user_123
#         results = mock_store.get_all(user_id="user_123")
#         assert len(results) == 2
        
#         # Get all episodic memories
#         results = mock_store.get_all(memory_type="episodic")
#         assert len(results) == 2
#         assert all(r.memory_type == "episodic" for r in results)

#     @pytest.mark.asyncio
#     async def test_aget_all_memories(self, mock_store):
#         """Test async getting all memories."""
#         await mock_store.aadd("Async content 1", user_id="user_123")
#         await mock_store.aadd("Async content 2", user_id="user_123")
        
#         results = await mock_store.aget_all(user_id="user_123")
#         assert len(results) == 2

#     def test_delete_all_memories(self, mock_store):
#         """Test deleting all memories with filters."""
#         mock_store.add("Content 1", user_id="user_123")
#         mock_store.add("Content 2", user_id="user_123")
#         mock_store.add("Content 3", user_id="user_456")
        
#         count = mock_store.delete_all(user_id="user_123")
        
#         assert count == 2
#         remaining = mock_store.get_all()
#         assert len(remaining) == 1
#         assert remaining[0].user_id == "user_456"

#     @pytest.mark.asyncio
#     async def test_adelete_all_memories(self, mock_store):
#         """Test async deleting all memories."""
#         await mock_store.aadd("Content 1", user_id="user_789")
#         await mock_store.aadd("Content 2", user_id="user_789")
        
#         count = await mock_store.adelete_all(user_id="user_789")
        
#         assert count == 2

#     def test_store_message(self, mock_store, sample_message):
#         """Test storing a TAF Message."""
#         memory_id = mock_store.store_message(
#             sample_message,
#             user_id="user_123",
#             agent_id="agent_456",
#             additional_metadata={"source": "test"}
#         )
        
#         assert memory_id in mock_store.memories
#         memory = mock_store.memories[memory_id]
#         assert memory.content == "Hello, this is a test message about machine learning."
#         assert memory.user_id == "user_123"
#         assert memory.agent_id == "agent_456"
#         assert memory.memory_type == "episodic"
#         assert memory.metadata["role"] == "user"
#         assert memory.metadata["source"] == "test"

#     @pytest.mark.asyncio
#     async def test_astore_message(self, mock_store, sample_message):
#         """Test async storing a TAF Message."""
#         memory_id = await mock_store.astore_message(
#             sample_message,
#             user_id="user_123"
#         )
        
#         memory = await mock_store.aget(memory_id)
#         assert memory.content == "Hello, this is a test message about machine learning."

#     def test_recall_similar_messages(self, mock_store, sample_message):
#         """Test recalling similar messages."""
#         # Store some messages
#         mock_store.store_message(sample_message, user_id="user_123")
        
#         other_message = Message(
#             role="assistant",
#             content=[TextBlock(text="I can help you with machine learning concepts.")],
#         )
#         mock_store.store_message(other_message, user_id="user_123")
        
#         unrelated_message = Message(
#             role="user",
#             content=[TextBlock(text="What's the weather like?")],
#         )
#         mock_store.store_message(unrelated_message, user_id="user_123")
        
#         # Recall similar messages
#         results = mock_store.recall_similar_messages(
#             "machine learning",
#             user_id="user_123"
#         )
        
#         assert len(results) == 2  # Two messages contain "machine learning"
#         assert all("learning" in r.content.lower() for r in results)

#     @pytest.mark.asyncio
#     async def test_arecall_similar_messages(self, mock_store, sample_message):
#         """Test async recalling similar messages."""
#         await mock_store.astore_message(sample_message, user_id="user_123")
        
#         results = await mock_store.arecall_similar_messages(
#             "machine",
#             user_id="user_123"
#         )
        
#         assert len(results) == 1

#     def test_embedding_generation(self, mock_store_with_embeddings):
#         """Test embedding generation."""
#         embedding = mock_store_with_embeddings._generate_embedding("test text")
        
#         assert isinstance(embedding, list)
#         assert len(embedding) == 2  # "test text" has 2 words
#         assert embedding == [0.0, 1.0]  # Based on mock function

#     @pytest.mark.asyncio
#     async def test_aembedding_generation(self, mock_store_with_embeddings):
#         """Test async embedding generation."""
#         embedding = await mock_store_with_embeddings._agenerate_embedding("test text")
        
#         assert isinstance(embedding, list)
#         assert len(embedding) == 2

#     def test_get_stats(self, mock_store):
#         """Test getting memory statistics."""
#         mock_store.add("Content 1", user_id="user_123", memory_type="episodic")
#         mock_store.add("Content 2", user_id="user_123", memory_type="semantic")
#         mock_store.add("Content 3", user_id="user_456", memory_type="episodic")
        
#         stats = mock_store.get_stats(user_id="user_123")
        
#         assert stats["total_memories"] == 2
#         assert stats["user_id"] == "user_123"
#         assert "episodic" in stats["memory_types"]
#         assert "semantic" in stats["memory_types"]

#     @pytest.mark.asyncio
#     async def test_aget_stats(self, mock_store):
#         """Test async getting memory statistics."""
#         await mock_store.aadd("Content 1", user_id="user_789")
        
#         stats = await mock_store.aget_stats(user_id="user_789")
        
#         assert stats["total_memories"] == 1
#         assert stats["user_id"] == "user_789"

#     def test_cleanup(self, mock_store):
#         """Test cleanup method."""
#         # Should not raise any errors
#         mock_store.cleanup()

#     @pytest.mark.asyncio
#     async def test_acleanup(self, mock_store):
#         """Test async cleanup method."""
#         # Should not raise any errors
#         await mock_store.acleanup()


# class TestTypeAliases:
#     """Test type aliases and backward compatibility."""

#     def test_message_memory_store_alias(self):
#         """Test MessageMemoryStore type alias."""
#         assert MessageMemoryStore == BaseStore[Message]

#     def test_base_store_has_expected_methods(self):
#         """Test that BaseStore defines expected methods."""
#         expected_methods = [
#             "add", "aadd",
#             "search", "asearch",
#             "get", "aget",
#             "update", "aupdate",
#             "delete", "adelete",
#             "get_all", "aget_all",
#             "delete_all", "adelete_all",
#             "store_message", "astore_message",
#             "recall_similar_messages", "arecall_similar_messages",
#             "get_stats", "aget_stats",
#             "cleanup", "acleanup",
#             # Legacy methods
#             "update_memory", "aupdate_memory",
#             "get_memory", "aget_memory",
#             "delete_memory", "adelete_memory",
#             "related_memory", "arelated_memory",
#             "release", "arelease"
#         ]

#         for method_name in expected_methods:
#             assert hasattr(BaseStore, method_name)
#             assert callable(getattr(BaseStore, method_name))