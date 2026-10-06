"""
Comprehensive tests for store schemas.

This module tests MemoryRecord, MemorySearchResult, and associated enums
including validation, serialization, and edge cases.
"""

from datetime import datetime
from unittest.mock import Mock
from uuid import UUID
import pytest

from tenxgraph.storage.store.store_schema import (
    DistanceMetric,
    MemoryType,
    RetrievalStrategy,
    MemoryRecord,
    MemorySearchResult,
)
from tenxgraph.core.state import Message, TextBlock


class TestEnums:
    """Test the enum classes used in store schemas."""
    
    def test_distance_metric_enum(self):
        """Test DistanceMetric enum values."""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.DOT_PRODUCT.value == "dot_product"
        assert DistanceMetric.MANHATTAN.value == "manhattan"
        
        # Test that all expected values are present
        expected_values = {"cosine", "euclidean", "dot_product", "manhattan"}
        actual_values = {metric.value for metric in DistanceMetric}
        assert actual_values == expected_values
    
    def test_memory_type_enum(self):
        """Test MemoryType enum values."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.ENTITY.value == "entity"
        assert MemoryType.RELATIONSHIP.value == "relationship"
        assert MemoryType.CUSTOM.value == "custom"
        assert MemoryType.DECLARATIVE.value == "declarative"
        
        # Test that all expected values are present
        expected_values = {
            "episodic", "semantic", "procedural", "entity", 
            "relationship", "custom", "declarative"
        }
        actual_values = {memory_type.value for memory_type in MemoryType}
        assert actual_values == expected_values
    
    def test_retrieval_strategy_enum(self):
        """Test RetrievalStrategy enum values."""
        assert RetrievalStrategy.SIMILARITY.value == "similarity"
        assert RetrievalStrategy.TEMPORAL.value == "temporal"
        assert RetrievalStrategy.RELEVANCE.value == "relevance"
        assert RetrievalStrategy.HYBRID.value == "hybrid"
        assert RetrievalStrategy.GRAPH_TRAVERSAL.value == "graph_traversal"
        
        # Test that all expected values are present
        expected_values = {
            "similarity", "temporal", "relevance", "hybrid", "graph_traversal"
        }
        actual_values = {strategy.value for strategy in RetrievalStrategy}
        assert actual_values == expected_values


class TestMemorySearchResult:
    """Test the MemorySearchResult model."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        result = MemorySearchResult()
        
        # Test that ID is generated
        assert isinstance(result.id, str)
        assert len(result.id) > 0
        # Test that it's a valid UUID format
        UUID(result.id)  # Should not raise exception
        
        # Test other defaults
        assert result.content == ""
        assert result.score == 0.0
        assert result.memory_type == MemoryType.EPISODIC
        assert result.metadata == {}
        assert result.vector is None
        assert result.user_id is None
        assert result.thread_id is None
        assert isinstance(result.timestamp, datetime)
    
    def test_explicit_values(self):
        """Test creating MemorySearchResult with explicit values."""
        test_time = datetime(2023, 1, 1, 12, 0, 0)
        test_vector = [0.1, 0.2, 0.3]
        test_metadata = {"key": "value", "score": 0.95}
        
        result = MemorySearchResult(
            id="custom-id",
            content="test content",
            score=0.85,
            memory_type=MemoryType.SEMANTIC,
            metadata=test_metadata,
            vector=test_vector,
            user_id="user-123",
            thread_id="thread-456",
            timestamp=test_time
        )
        
        assert result.id == "custom-id"
        assert result.content == "test content"
        assert result.score == 0.85
        assert result.memory_type == MemoryType.SEMANTIC
        assert result.metadata == test_metadata
        assert result.vector == test_vector
        assert result.user_id == "user-123"
        assert result.thread_id == "thread-456"
        assert result.timestamp == test_time
    
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
    
    def test_required_content(self):
        """Test that content is required."""
        # Should work with content
        record = MemoryRecord(content="test content")
        assert record.content == "test content"
        
        # Should fail without content
        with pytest.raises(ValueError):
            MemoryRecord()
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        record = MemoryRecord(content="test")
        
        # Test that ID is generated
        assert isinstance(record.id, str)
        assert len(record.id) > 0
        UUID(record.id)  # Should not raise exception
        
        # Test other defaults
        assert record.user_id is None
        assert record.thread_id is None
        assert record.memory_type == MemoryType.EPISODIC
        assert record.metadata == {}
        assert record.category == "general"
        assert record.vector is None
        assert isinstance(record.timestamp, datetime)
    
    def test_explicit_values(self):
        """Test creating MemoryRecord with explicit values."""
        test_time = datetime(2023, 1, 1, 12, 0, 0)
        test_vector = [0.4, 0.5, 0.6]
        test_metadata = {"source": "chat", "confidence": 0.9}
        
        record = MemoryRecord(
            id="record-id",
            content="important memory",
            user_id="user-789",
            thread_id="thread-101",
            memory_type=MemoryType.PROCEDURAL,
            metadata=test_metadata,
            category="skills",
            vector=test_vector,
            timestamp=test_time
        )
        
        assert record.id == "record-id"
        assert record.content == "important memory"
        assert record.user_id == "user-789"
        assert record.thread_id == "thread-101"
        assert record.memory_type == MemoryType.PROCEDURAL
        assert record.metadata == test_metadata
        assert record.category == "skills"
        assert record.vector == test_vector
        assert record.timestamp == test_time
    
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
        test_time = datetime(2023, 6, 15, 10, 30, 0)
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


class TestSchemaEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_content(self):
        """Test handling of very long content."""
        long_content = "a" * 100000  # 100k characters
        
        record = MemoryRecord(content=long_content)
        assert record.content == long_content
        
        search_result = MemorySearchResult(content=long_content)
        assert search_result.content == long_content
    
    def test_unicode_content(self):
        """Test handling of unicode content."""
        unicode_content = "Hello 世界 🌍 emoji and unicode"
        
        record = MemoryRecord(content=unicode_content)
        assert record.content == unicode_content
        
        search_result = MemorySearchResult(content=unicode_content)
        assert search_result.content == unicode_content
    
    def test_large_metadata(self):
        """Test handling of large metadata dictionaries."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        record = MemoryRecord(content="test", metadata=large_metadata)
        assert record.metadata == large_metadata
        
        search_result = MemorySearchResult(metadata=large_metadata)
        assert search_result.metadata == large_metadata
    
    def test_high_dimensional_vectors(self):
        """Test handling of high-dimensional vectors."""
        high_dim_vector = [float(i) for i in range(4096)]  # 4k dimensions
        
        record = MemoryRecord(content="test", vector=high_dim_vector)
        assert record.vector == high_dim_vector
        
        search_result = MemorySearchResult(vector=high_dim_vector)
        assert search_result.vector == high_dim_vector
    
    def test_extreme_scores(self):
        """Test handling of extreme score values."""
        # Very small positive score
        result1 = MemorySearchResult(score=1e-10)
        assert result1.score == 1e-10
        
        # Very large score
        result2 = MemorySearchResult(score=1e10)
        assert result2.score == 1e10
        
        # Zero score
        result3 = MemorySearchResult(score=0.0)
        assert result3.score == 0.0
    
    def test_nested_metadata(self):
        """Test handling of nested metadata structures."""
        nested_metadata = {
            "level1": {
                "level2": {
                    "level3": ["item1", "item2", "item3"],
                    "numbers": [1, 2, 3, 4, 5]
                },
                "simple": "value"
            },
            "list": [{"key": "value"}, {"key2": "value2"}]
        }
        
        record = MemoryRecord(content="test", metadata=nested_metadata)
        assert record.metadata == nested_metadata
        
        # Test serialization still works
        data = record.model_dump()
        assert data["metadata"] == nested_metadata
    
    def test_special_string_ids(self):
        """Test handling of special string IDs."""
        special_ids = [
            "id-with-dashes",
            "id_with_underscores",
            "id.with.dots",
            "id with spaces",
            "id/with/slashes",
            "123-numeric-start",
            "",  # empty string
        ]
        
        for special_id in special_ids:
            record = MemoryRecord(id=special_id, content="test")
            assert record.id == special_id
            
            search_result = MemorySearchResult(id=special_id)
            assert search_result.id == special_id