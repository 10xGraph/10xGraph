import logging
from typing import Any

from .graph_error import GraphError


logger = logging.getLogger("agentflow.exceptions")


class NodeError(GraphError):
    """
    Exception raised when a node encounters an error.

    This exception is used for errors specific to nodes within a graph.
    Inherits structured error handling from GraphError.

    Example:
        >>> from agentflow.exceptions.node_error import NodeError
        >>> raise NodeError(
        ...     message="Node failed to execute",
        ...     error_code="NODE_001",
        ...     context={"node_name": "process_data", "input_size": 100},
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: str = "NODE_000",
        context: dict[str, Any] | None = None,
    ):
        """
        Initializes a NodeError with the given message, error code, and context.

        Args:
            message (str): Description of the node error.
            error_code (str): Unique error code for categorization (default: "NODE_000")
            context (dict): Additional contextual information (default: None)
        """
        logger.error(
            "NodeError [%s]: %s | Context: %s",
            error_code,
            message,
            context or {},
            exc_info=True,
        )
        super().__init__(message, error_code, context)
