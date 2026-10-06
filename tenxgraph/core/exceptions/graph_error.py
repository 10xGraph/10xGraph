import logging
from typing import Any


logger = logging.getLogger("agentflow.exceptions")


class GraphError(Exception):
    """
    Base exception for graph-related errors.

    This exception is raised when an error related to graph operations occurs.
    It provides structured error responses with error codes and contextual information.

    Attributes:
        message (str): Human-readable description of the error
        error_code (str): Unique error code for categorization
        context (dict): Additional contextual information about the error

    Example:
        >>> from agentflow.exceptions.graph_error import GraphError
        >>> raise GraphError(
        ...     message="Invalid graph structure",
        ...     error_code="GRAPH_001",
        ...     context={"node_count": 5, "edge_count": 3},
        ... )
    """

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
        self.message = message
        self.error_code = error_code
        self.context = context or {}

        # Log the error with full context
        logger.error(
            "GraphError [%s]: %s | Context: %s",
            self.error_code,
            self.message,
            self.context,
            exc_info=True,
        )
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the error to a structured dictionary format.

        Returns:
            dict: Structured error response with error_code, message, and context
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:
        """Return a string representation of the error."""
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"context={self.context})"
        )
