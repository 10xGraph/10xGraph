"""
Comprehensive tests for store schemas.

This module tests MemoryRecord, MemorySearchResult, and associated enums
including validation, serialization, and edge cases.
"""

from datetime import datetime, timezone
import pytest

from tenxgraph.storage.store.store_schema import (
    MemoryType,
    MemoryRecord,
    MemorySearchResult,
)
from tenxgraph.core.state import Message, TextBlock




class TestMemorySearchResult:
    """Test the MemorySearchResult model."""
    
    
    
    def test_score_validation(self):
        """Test that score must be >= 0.0."""
        # Valid scores
        MemorySearchResult(score=0.0)
        MemorySearchResult(score=1.0)
        MemorySearchResult(score=0.5)
        
        # Invalid scores should raise validation error
        with pytest.raises(ValueError):
            MemorySearchResult(score=-0.1)
        
        with pytest.raises(ValueError):
            MemorySearchResult(score=-1.0)
    
    def test_vector_validation_valid(self):
        """Test valid vector values."""
        # Valid vectors
        MemorySearchResult(vector=None)
        MemorySearchResult(vector=[])
        MemorySearchResult(vector=[1.0, 2.0, 3.0])
        MemorySearchResult(vector=[0, 1, 2])  # integers should be fine
        MemorySearchResult(vector=[0.1, -0.5, 2.7])  # negative values OK
    
    def test_vector_validation_invalid(self):
        """Test invalid vector values."""
        with pytest.raises(ValueError):
            MemorySearchResult(vector=["string", "values"])
        
        with pytest.raises(ValueError):
            MemorySearchResult(vector=[1.0, "mixed", 3.0])
        
        with pytest.raises(ValueError):
            MemorySearchResult(vector="not a list")
    
    def test_serialization(self):
        """Test that MemorySearchResult can be serialized."""
        result = MemorySearchResult(
            content="test",
            score=0.5,
            vector=[1.0, 2.0],
            metadata={"key": "value"}
        )
        
        # Test model_dump (Pydantic serialization)
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["content"] == "test"
        assert data["score"] == 0.5
        assert data["vector"] == [1.0, 2.0]
        assert data["metadata"] == {"key": "value"}
    
    def test_deserialization(self):
        """Test that MemorySearchResult can be deserialized."""
        data = {
            "id": "test-id",
            "content": "test content",
            "score": 0.8,
            "memory_type": "semantic",
            "metadata": {"test": True},
            "vector": [0.1, 0.2],
            "user_id": "user-1",
            "thread_id": "thread-1",
            "timestamp": "2023-01-01T12:00:00"
        }
        
        result = MemorySearchResult(**data)
        assert result.content == "test content"
        assert result.score == 0.8
        assert result.memory_type == MemoryType.SEMANTIC
        assert result.vector == [0.1, 0.2]


class TestMemoryRecord:
    """Test the MemoryRecord model."""
    
    
    
    
    def test_vector_validation_valid(self):
        """Test valid vector values for MemoryRecord."""
        # Valid vectors
        MemoryRecord(content="test", vector=None)
        MemoryRecord(content="test", vector=[])
        MemoryRecord(content="test", vector=[1.0, 2.0, 3.0])
        MemoryRecord(content="test", vector=[0, 1, 2])  # integers OK
    
    def test_vector_validation_invalid(self):
        """Test invalid vector values for MemoryRecord."""
        with pytest.raises(ValueError):
            MemoryRecord(content="test", vector=["invalid"])
        
        with pytest.raises(ValueError):
            MemoryRecord(content="test", vector=[1.0, None, 3.0])
    
    def test_from_message_basic(self):
        """Test creating MemoryRecord from Message."""
        # Create a test message
        message = Message(
            role="user",
            content=[TextBlock(text="Hello world")]
        )
        
        record = MemoryRecord.from_message(message)
        
        assert record.content == "Hello world"
        assert record.metadata["role"] == "user"
        assert record.metadata["message_id"] == str(message.message_id)
        assert record.user_id is None
        assert record.thread_id is None
        assert record.vector is None
    
    def test_from_message_with_parameters(self):
        """Test creating MemoryRecord from Message with additional parameters."""
        message = Message(
            role="assistant",
            content=[TextBlock(text="How can I help?")]
        )
        
        test_vector = [0.1, 0.2, 0.3]
        additional_metadata = {"source": "chat", "priority": "high"}
        
        record = MemoryRecord.from_message(
            message=message,
            user_id="user-123",
            thread_id="thread-456",
            vector=test_vector,
            additional_metadata=additional_metadata
        )
        
        assert record.content == "How can I help?"
        assert record.user_id == "user-123"
        assert record.thread_id == "thread-456"
        assert record.vector == test_vector
        assert record.metadata["role"] == "assistant"
        assert record.metadata["source"] == "chat"
        assert record.metadata["priority"] == "high"
    
    def test_from_message_empty_content(self):
        """Test creating MemoryRecord from Message with empty content."""
        message = Message(role="user", content=[])
        
        record = MemoryRecord.from_message(message)
        
        assert record.content == ""
        assert record.metadata["role"] == "user"
    
    def test_from_message_with_timestamp(self):
        """Test creating MemoryRecord from Message with timestamp."""
        test_time = datetime(2023, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        timestamp_float = test_time.timestamp()
        message = Message(
            role="user",
            content=[TextBlock(text="Test message")],
            timestamp=timestamp_float
        )

        record = MemoryRecord.from_message(message)

        assert record.metadata["timestamp"] == test_time.isoformat()
    
    def test_from_message_without_timestamp(self):
        """Test creating MemoryRecord from Message without timestamp."""
        message = Message(
            role="user",
            content=[TextBlock(text="Test message")],
            timestamp=None
        )
        
        record = MemoryRecord.from_message(message)
        
        assert record.metadata["timestamp"] is None
    
    def test_serialization(self):
        """Test that MemoryRecord can be serialized."""
        record = MemoryRecord(
            content="test memory",
            user_id="user-1",
            memory_type=MemoryType.SEMANTIC,
            vector=[1.0, 2.0, 3.0],
            metadata={"category": "facts"}
        )
        
        data = record.model_dump()
        assert isinstance(data, dict)
        assert data["content"] == "test memory"
        assert data["user_id"] == "user-1"
        # Enum serialization may keep enum object or convert to string
        assert data["memory_type"] == MemoryType.SEMANTIC or data["memory_type"] == "semantic"
        assert data["vector"] == [1.0, 2.0, 3.0]
        assert data["metadata"] == {"category": "facts"}
    
    def test_deserialization(self):
        """Test that MemoryRecord can be deserialized."""
        data = {
            "id": "memory-id",
            "content": "deserialized memory",
            "user_id": "user-2",
            "thread_id": "thread-2",
            "memory_type": "entity",
            "metadata": {"entity_type": "person"},
            "category": "entities",
            "vector": [0.5, 0.6, 0.7],
            "timestamp": "2023-01-01T15:30:00"
        }
        
        record = MemoryRecord(**data)
        assert record.content == "deserialized memory"
        assert record.user_id == "user-2"
        assert record.memory_type == MemoryType.ENTITY
        assert record.category == "entities"
        assert record.vector == [0.5, 0.6, 0.7]


class TestSchemaIntegration:
    """Integration tests for schema classes."""
    
    def test_memory_record_to_search_result_conversion(self):
        """Test converting MemoryRecord to MemorySearchResult."""
        record = MemoryRecord(
            id="test-id",
            content="test content",
            user_id="user-1",
            thread_id="thread-1",
            memory_type=MemoryType.SEMANTIC,
            metadata={"key": "value"},
            vector=[0.1, 0.2, 0.3]
        )
        
        # Manual conversion (since there's no built-in method)
        search_result = MemorySearchResult(
            id=record.id,
            content=record.content,
            score=0.85,  # Would come from search
            memory_type=record.memory_type,
            metadata=record.metadata,
            vector=record.vector,
            user_id=record.user_id,
            thread_id=record.thread_id,
            timestamp=record.timestamp
        )
        
        assert search_result.id == record.id
        assert search_result.content == record.content
        assert search_result.memory_type == record.memory_type
        assert search_result.vector == record.vector
        assert search_result.score == 0.85
    
    def test_enum_usage_in_models(self):
        """Test that enums work correctly in models."""
        # Test all memory types
        for memory_type in MemoryType:
            record = MemoryRecord(
                content="test",
                memory_type=memory_type
            )
            assert record.memory_type == memory_type
            
            search_result = MemorySearchResult(
                memory_type=memory_type
            )
            assert search_result.memory_type == memory_type
    
    def test_model_equality(self):
        """Test model equality comparison."""
        # Fix timestamp to make records truly equal
        fixed_time = datetime(2023, 1, 1, 12, 0, 0)
        
        record1 = MemoryRecord(
            id="same-id",
            content="same content",
            user_id="user-1",
            timestamp=fixed_time
        )
        
        record2 = MemoryRecord(
            id="same-id",
            content="same content",
            user_id="user-1",
            timestamp=fixed_time
        )
        
        # Pydantic models with same data should be equal
        assert record1 == record2
        
        # Different content should not be equal
        record3 = MemoryRecord(
            id="same-id",
            content="different content",
            user_id="user-1",
            timestamp=fixed_time
        )
        
        assert record1 != record3

