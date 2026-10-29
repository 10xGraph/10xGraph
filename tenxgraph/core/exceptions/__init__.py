"""
Custom exception classes for graph operations in agentflowntflow.

This package provides:
        - GraphError: Base exception for graph-related errors.
        - NodeError: Exception for node-specific errors.
        - GraphClosedError: A run was started on a graph that is closing or closed.
        - GraphRecursionError: Exception for recursion limit errors in graphs.
        - InterruptMismatchError: A resume names an interrupt the thread is not paused at.
        - ResumeValidationError: A resume value does not match the interrupt's response schema.
        - StorageError: Base exception for storage-related errors.
        - TransientStorageError: Exception for retryable storage errors.
        - SerializationError: Exception for serialization/deserialization errors.
        - SchemaVersionError: Exception for schema version mismatch errors.
        - MetricsError: Exception for metrics emission errors.
"""

from .closed_error import GraphClosedError
from .graph_error import GraphError
from .interrupt_error import InterruptMismatchError, ResumeValidationError
from .node_error import NodeError
from .recursion_error import GraphRecursionError
from .storage_exceptions import (
    MetricsError,
    ResourceNotFoundError,
    SchemaVersionError,
    SerializationError,
    StaleStateError,
    StorageError,
    TransientStorageError,
)
from .timeout_error import GraphStopRequested, NodeTimeoutError


__all__ = [
    "GraphClosedError",
    "GraphError",
    "GraphRecursionError",
    "GraphStopRequested",
    "InterruptMismatchError",
    "MetricsError",
    "NodeError",
    "NodeTimeoutError",
    "ResourceNotFoundError",
    "ResumeValidationError",
    "SchemaVersionError",
    "SerializationError",
    "StaleStateError",
    "StorageError",
    "TransientStorageError",
]
