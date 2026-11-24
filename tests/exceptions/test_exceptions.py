"""Comprehensive tests for the exceptions module."""


from tenxgraph.core.exceptions import (
    MetricsError,
    GraphError,
    GraphRecursionError,
    NodeError,
    SchemaVersionError,
    SerializationError,
    StorageError,
    TransientStorageError,
)


class TestGraphError:
    """Test the GraphError exception."""


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


    def test_graph_error_with_cause(self):
        """Test creating a GraphError with a cause."""
        cause = ValueError("Original error")
        try:
            raise GraphError("Graph failed") from cause
        except GraphError as error:
            assert str(error) == "[GRAPH_000] Graph failed"  # noqa: S101
            assert error.__cause__ == cause  # noqa: S101




class TestNodeError:
    """Test the NodeError exception."""


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




class TestGraphRecursionError:
    """Test the GraphRecursionError exception."""


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




class TestStorageExceptions:
    """Test storage exception classes."""


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




def test_thread_not_found_error_does_not_log_at_error_level(caplog):
    """A missing thread is a normal 404 path, not an error worth a traceback in the logs."""
    import logging

    from tenxgraph.core.exceptions import ResourceNotFoundError, ThreadNotFoundError

    with caplog.at_level(logging.DEBUG, logger="tenxgraph.exceptions"):
        error = ThreadNotFoundError("t1")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert isinstance(error, ResourceNotFoundError)
    assert error.error_code == "STORAGE_NOT_FOUND_001"
    assert error.context == {"thread_id": "t1"}
    assert str(error) == "[STORAGE_NOT_FOUND_001] Thread not found"

    with caplog.at_level(logging.DEBUG, logger="tenxgraph.exceptions"):
        ResourceNotFoundError("other")
    assert [r for r in caplog.records if r.levelno == logging.ERROR]
