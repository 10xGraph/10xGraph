from typing import Any

from ._structured import _StructuredError


class GraphError(_StructuredError, Exception):
    """
    Base exception for graph-related errors.

    This exception is raised when an error related to graph operations occurs.
    It provides structured error responses with error codes and contextual information.

    Attributes:
        message (str): Human-readable description of the error
        error_code (str): Unique error code for categorization
        context (dict): Additional contextual information about the error

    Example:
        >>> from tenxgraph.exceptions.graph_error import GraphError
        >>> raise GraphError(
        ...     message="Invalid graph structure",
        ...     error_code="GRAPH_001",
        ...     context={"node_count": 5, "edge_count": 3},
        ... )
    """

    _log_prefix = "GraphError"

    def __init__(
        self,
        message: str,
        error_code: str = "GRAPH_000",
        context: dict[str, Any] | None = None,
    ):
        """
        Initializes a GraphError with the given message, error code, and context.

        Args:
            message (str): Description of the error.
            error_code (str): Unique error code for categorization (default: "GRAPH_000")
            context (dict): Additional contextual information (default: None)
        """
        self._init_error(message, error_code, context)
        super().__init__(message)
