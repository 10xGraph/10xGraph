import logging
from typing import Any

from .graph_error import GraphError


logger = logging.getLogger("tenxgraph.exceptions")


class InterruptMismatchError(GraphError, ValueError):
    """
    Raised when a resume names an interrupt that is not the one the thread is paused at.

    A client may send ``resume_interrupt_id`` with its resume value. If the id differs from
    the pending interrupt's id (a stale tab, a retried request after the first answer was
    consumed), the answer is rejected and nothing is recorded, so it cannot approve a
    prompt the human never saw. It is also a ``ValueError`` so existing handlers for a bad
    resume keep working.

    Example:
        >>> raise InterruptMismatchError(
        ...     message="Resume targets interrupt 'int_a' but the thread is paused at 'int_b'",
        ...     error_code="INTERRUPT_001",
        ...     context={"expected": "int_b", "received": "int_a"},
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INTERRUPT_001",
        context: dict[str, Any] | None = None,
    ):
        """
        Initialize an InterruptMismatchError.

        Args:
            message (str): Description of the mismatch.
            error_code (str): Unique error code (default: "INTERRUPT_001")
            context (dict): Additional contextual information (default: None)
        """
        super().__init__(message, error_code, context)


class ResumeValidationError(GraphError, ValueError):
    """
    Raised when a resume value does not satisfy the pending interrupt's ``response_schema``.

    Nothing is recorded when this is raised. It is also a ``ValueError`` so existing
    handlers for a bad resume keep working.

    Example:
        >>> raise ResumeValidationError(
        ...     message="Resume value does not match the response schema",
        ...     error_code="INTERRUPT_002",
        ...     context={"interrupt_id": "int_b", "errors": ["$.approved: expected boolean"]},
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INTERRUPT_002",
        context: dict[str, Any] | None = None,
    ):
        """
        Initialize a ResumeValidationError.

        Args:
            message (str): Description of the validation failure.
            error_code (str): Unique error code (default: "INTERRUPT_002")
            context (dict): Additional contextual information (default: None)
        """
        super().__init__(message, error_code, context)
