import logging
from typing import Any

from .graph_error import GraphError


logger = logging.getLogger("agentflow.exceptions")


class GraphRecursionError(GraphError):
    """
    Exception raised when graph execution exceeds the recursion limit.

    This exception is used to indicate that a graph operation has recursed too deeply.
    Inherits structured error handling from GraphError.

    Example:
        >>> from agentflow.exceptions.recursion_error import GraphRecursionError
        >>> raise GraphRecursionError(
        ...     message="Recursion limit exceeded in graph execution",
        ...     error_code="RECURSION_001",
        ...     context={"recursion_depth": 100, "max_depth": 50},
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: str = "RECURSION_000",
        context: dict[str, Any] | None = None,
    ):
        """
        Initializes a GraphRecursionError with the given message, error code, and context.

        Args:
            message (str): Description of the recursion error.
            error_code (str): Unique error code for categorization (default: "RECURSION_000")
            context (dict): Additional contextual information (default: None)
        """
        logger.error(
            "GraphRecursionError [%s]: %s | Context: %s",
            error_code,
            message,
            context or {},
            exc_info=True,
        )
        super().__init__(message, error_code, context)
