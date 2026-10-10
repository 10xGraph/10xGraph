"""Comprehensive tests for the exceptions module."""

import pytest

from tenxgraph.core.exceptions import (
    GraphError,
    GraphRecursionError,
    MetricsError,
    NodeError,
    SchemaVersionError,
    SerializationError,
    StorageError,
    TransientStorageError,
)


class TestGraphError:
    """Test the GraphError exception."""

    def test_graph_error_creation(self):
        """Test creating a GraphError with default error code."""
        error = GraphError("Test graph error")
        assert str(error) == "[GRAPH_000] Test graph error"  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101
        assert error.error_code == "GRAPH_000"  # noqa: S101
        assert error.message == "Test graph error"  # noqa: S101
        assert error.context == {}  # noqa: S101

    def test_graph_error_with_code_and_context(self):
        """Test creating a GraphError with custom error code and context."""
        context = {"node_count": 5, "edge_count": 3}
        error = GraphError(
            message="Invalid graph structure",
            error_code="GRAPH_001",
            context=context,
        )
        assert str(error) == "[GRAPH_001] Invalid graph structure"  # noqa: S101
        assert error.error_code == "GRAPH_001"  # noqa: S101
        assert error.context == context  # noqa: S101

    def test_graph_error_to_dict(self):
        """Test converting GraphError to dictionary."""
        context = {"node_count": 5}
        error = GraphError(
            message="Test error",
            error_code="GRAPH_001",
            context=context,
        )
        result = error.to_dict()
        assert result == {  # noqa: S101
            "error_type": "GraphError",
            "error_code": "GRAPH_001",
            "message": "Test error",
            "context": context,
        }

    def test_graph_error_repr(self):
        """Test GraphError repr method."""
        error = GraphError(
            message="Test",
            error_code="GRAPH_001",
            context={"key": "value"},
        )
        repr_str = repr(error)
        assert "GraphError" in repr_str  # noqa: S101
        assert "GRAPH_001" in repr_str  # noqa: S101
        assert "Test" in repr_str  # noqa: S101

    def test_graph_error_with_cause(self):
        """Test creating a GraphError with a cause."""
        cause = ValueError("Original error")
        try:
            raise GraphError("Graph failed") from cause
        except GraphError as error:
            assert str(error) == "[GRAPH_000] Graph failed"  # noqa: S101
            assert error.__cause__ == cause  # noqa: S101

    def test_graph_error_inheritance(self):
        """Test that GraphError inherits from Exception."""
        error = GraphError("Test error")
        assert isinstance(error, Exception)  # noqa: S101

    def test_graph_error_raise(self):
        """Test raising a GraphError."""
        with pytest.raises(GraphError) as exc_info:
            raise GraphError("Test graph error")

        assert str(exc_info.value) == "[GRAPH_000] Test graph error"  # noqa: S101


class TestNodeError:
    """Test the NodeError exception."""

    def test_node_error_creation(self):
        """Test creating a NodeError."""
        error = NodeError("Test node error")
        assert str(error) == "[NODE_000] Test node error"  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101
        assert error.error_code == "NODE_000"  # noqa: S101

    def test_node_error_with_context(self):
        """Test creating a NodeError with context."""
        context = {"node_name": "agent", "input_size": 100}
        error = NodeError(
            message="Node failed to execute",
            error_code="NODE_001",
            context=context,
        )
        assert error.context == context  # noqa: S101
        assert "NODE_001" in str(error)  # noqa: S101

    def test_node_error_with_node_name(self):
        """Test creating a NodeError with node context."""
        error = NodeError("Node 'agent' failed: Invalid input")
        assert "agent" in str(error)  # noqa: S101
        assert "failed" in str(error)  # noqa: S101

    def test_node_error_inheritance(self):
        """Test that NodeError inherits from GraphError."""
        error = NodeError("Test error")
        assert isinstance(error, GraphError)  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101

    def test_node_error_raise(self):
        """Test raising a NodeError."""
        with pytest.raises(NodeError) as exc_info:
            raise NodeError("Test node error")

        assert str(exc_info.value) == "[NODE_000] Test node error"  # noqa: S101


class TestGraphRecursionError:
    """Test the GraphRecursionError exception."""

    def test_recursion_error_creation(self):
        """Test creating a GraphRecursionError."""
        error = GraphRecursionError("Test recursion error")
        assert str(error) == "[RECURSION_000] Test recursion error"  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101
        assert error.error_code == "RECURSION_000"  # noqa: S101

    def test_recursion_error_with_context(self):
        """Test creating a GraphRecursionError with context."""
        context = {"max_steps": 50, "current_step": 51}
        error = GraphRecursionError(
            message="Recursion limit exceeded",
            error_code="RECURSION_001",
            context=context,
        )
        assert error.context == context  # noqa: S101
        assert "RECURSION_001" in str(error)  # noqa: S101

    def test_recursion_error_with_depth(self):
        """Test creating a GraphRecursionError with depth information."""
        error = GraphRecursionError("Maximum recursion depth exceeded: 100")
        assert "recursion" in str(error)  # noqa: S101
        assert "100" in str(error)  # noqa: S101

    def test_recursion_error_inheritance(self):
        """Test that GraphRecursionError inherits from GraphError."""
        error = GraphRecursionError("Test error")
        assert isinstance(error, GraphError)  # noqa: S101
        assert isinstance(error, Exception)  # noqa: S101

    def test_recursion_error_raise(self):
        """Test raising a GraphRecursionError."""
        with pytest.raises(GraphRecursionError) as exc_info:
            raise GraphRecursionError("Test recursion error")

        assert str(exc_info.value) == "[RECURSION_000] Test recursion error"  # noqa: S101


class TestStorageExceptions:
    """Test storage exception classes."""

    def test_storage_error_creation(self):
        """Test creating a StorageError."""
        error = StorageError("Test storage error")
        assert str(error) == "[STORAGE_000] Test storage error"  # noqa: S101
        assert error.error_code == "STORAGE_000"  # noqa: S101

    def test_storage_error_with_context(self):
        """Test creating a StorageError with context."""
        context = {"thread_id": "test-123"}
        error = StorageError(
            message="Failed to store",
            error_code="STORAGE_001",
            context=context,
        )
        assert error.context == context  # noqa: S101

    def test_transient_storage_error(self):
        """Test creating a TransientStorageError."""
        error = TransientStorageError("Connection timeout")
        assert str(error) == "[STORAGE_TRANSIENT_000] Connection timeout"  # noqa: S101
        assert isinstance(error, StorageError)  # noqa: S101

    def test_serialization_error(self):
        """Test creating a SerializationError."""
        error = SerializationError("Failed to serialize state")
        assert str(error) == "[STORAGE_SERIALIZATION_000] Failed to serialize state"  # noqa: S101
        assert isinstance(error, StorageError)  # noqa: S101

    def test_schema_version_error(self):
        """Test creating a SchemaVersionError."""
        error = SchemaVersionError("Schema mismatch")
        assert str(error) == "[STORAGE_SCHEMA_000] Schema mismatch"  # noqa: S101
        assert isinstance(error, StorageError)  # noqa: S101

    def test_metrics_error_creation(self):
        """Test creating a MetricsError."""
        error = MetricsError("Failed to emit metric")
        assert "METRICS_000" in str(error)  # noqa: S101
        assert error.error_code == "METRICS_000"  # noqa: S101


class TestExceptionHierarchy:
    """Test exception hierarchy and relationships."""

    def test_all_exceptions_are_exceptions(self):
        """Test that all custom exceptions inherit from Exception."""
        errors = [
            GraphError("test"),
            NodeError("test"),
            GraphRecursionError("test"),
            StorageError("test"),
            TransientStorageError("test"),
            SerializationError("test"),
            SchemaVersionError("test"),
            MetricsError("test"),
        ]

        for error in errors:
            assert isinstance(error, Exception)  # noqa: S101

    def test_node_error_is_graph_error(self):
        """Test that NodeError inherits from GraphError."""
        error = NodeError("test")
        assert isinstance(error, GraphError)  # noqa: S101

    def test_recursion_error_is_graph_error(self):
        """Test that GraphRecursionError inherits from GraphError."""
        error = GraphRecursionError("test")
        assert isinstance(error, GraphError)  # noqa: S101

    def test_storage_error_hierarchy(self):
        """Test that storage errors have correct hierarchy."""
        errors = [
            TransientStorageError("test"),
            SerializationError("test"),
            SchemaVersionError("test"),
        ]
        for error in errors:
            assert isinstance(error, StorageError)  # noqa: S101

    def test_exception_chaining(self):
        """Test exception chaining works correctly."""
        original_error = ValueError("Original problem")

        try:
            raise NodeError("Node failed") from original_error
        except NodeError as node_error:
            assert node_error.__cause__ == original_error  # noqa: S101
            assert isinstance(node_error, NodeError)  # noqa: S101

    def test_exception_context_preservation(self):
        """Test that exception context is preserved."""
        try:
            # First exception
            try:
                raise ValueError("First error")
            except ValueError:
                # Second exception that should preserve context
                raise GraphError("Graph error occurred")
        except GraphError as graph_error:
            assert graph_error.__context__ is not None  # noqa: S101
            assert isinstance(graph_error.__context__, ValueError)  # noqa: S101


def test_exceptions_module_imports():
    """Test that exceptions module imports work correctly."""
    assert GraphError is not None  # noqa: S101
    assert NodeError is not None  # noqa: S101
    assert GraphRecursionError is not None  # noqa: S101
    assert StorageError is not None  # noqa: S101
    assert TransientStorageError is not None  # noqa: S101
    assert SerializationError is not None  # noqa: S101
    assert SchemaVersionError is not None  # noqa: S101
    assert MetricsError is not None  # noqa: S101
