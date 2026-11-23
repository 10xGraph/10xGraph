"""Shared body of the structured exceptions (``GraphError``, ``StorageError``, ``MetricsError``)."""

from __future__ import annotations

import logging
from typing import Any

from ._redact import redact_context


logger = logging.getLogger("tenxgraph.exceptions")


class _StructuredError:
    """Mixin: message, error code and context, one log line, ``to_dict`` and string forms.

    Put it before ``Exception`` in the bases. ``_log_level`` and ``_log_exc_info`` set how
    the creation of the error is logged.
    """

    _log_prefix = "StructuredError"
    _log_level = logging.ERROR
    _log_exc_info = True

    message: str
    error_code: str
    context: dict[str, Any]

    def _init_error(self, message: str, error_code: str, context: dict[str, Any] | None) -> None:
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        logger.log(
            self._log_level,
            "%s [%s]: %s | Context: %s",
            self._log_prefix,
            self.error_code,
            self.message,
            redact_context(self.context),
            exc_info=self._log_exc_info,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the error to a structured dictionary format.

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
