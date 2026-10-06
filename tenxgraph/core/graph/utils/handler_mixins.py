"""Shared mixins for graph and node handler classes.

This module provides lightweight mixins that add common functionality to handler
classes without changing their core runtime behavior. The mixins follow the
composition pattern to keep responsibilities explicit and allow handlers to
inherit only the capabilities they need.

The mixins provide structured logging, configuration management, and other
cross-cutting concerns that are commonly needed across different handler types.
By using mixins, the core handler logic remains focused while gaining these
shared capabilities.

Typical usage:
    ```python
    class MyHandler(BaseLoggingMixin, InterruptConfigMixin):
        def __init__(self):
            self._set_interrupts(["node1"], ["node2"])
            self._log_start("Handler initialized")
    ```
"""

from __future__ import annotations

import logging
from typing import Any


class BaseLoggingMixin:
    """Provides structured logging helpers for handler classes.

    This mixin adds consistent logging capabilities to handler classes without
    requiring them to manage logger instances directly. It automatically creates
    loggers based on the module name and provides convenience methods for
    common logging operations.

    The mixin is designed to be lightweight and non-intrusive, adding only
    logging functionality without affecting the core behavior of the handler.

    Attributes:
        _logger: Cached logger instance for the handler class.

    Example:
        ```python
        class MyHandler(BaseLoggingMixin):
            def process(self):
                self._log_start("Processing started")
                try:
                    # Do work
                    self._log_debug("Work completed successfully")
                except Exception as e:
                    self._log_error("Processing failed: %s", e)
        ```
    """

    @property
    def _logger(self) -> logging.Logger:
        """Get or create a logger instance for this handler.

        Creates a logger using the handler's module name, providing consistent
        logging across different handler instances while maintaining proper
        logger hierarchy and configuration.

        Returns:
            Logger instance configured for this handler's module.
        """
        return logging.getLogger("agentflow.graph")

    def _log_start(self, msg: str, *args: Any) -> None:
        """Log an informational message for process start/initialization.

        Args:
            msg: Log message format string.
            *args: Arguments for message formatting.
        """
        self._logger.info(msg, *args)

    def _log_debug(self, msg: str, *args: Any) -> None:
        """Log a debug message for detailed execution information.

        Args:
            msg: Log message format string.
            *args: Arguments for message formatting.
        """
        self._logger.debug(msg, *args)

    def _log_error(self, msg: str, *args: Any) -> None:
        """Log an error message for exceptional conditions.

        Args:
            msg: Log message format string.
            *args: Arguments for message formatting.
        """
        self._logger.error(msg, *args)


class InterruptConfigMixin:
    """Manages interrupt configuration for graph-level execution handlers.

    This mixin provides functionality to store and manage interrupt points
    configuration for graph execution. Interrupts allow graph execution to be
    paused before or after specific nodes for debugging, human intervention,
    or checkpoint creation.

    The mixin maintains separate lists for "before" and "after" interrupts,
    allowing fine-grained control over when graph execution should pause.

    Attributes:
        interrupt_before: List of node names where execution should pause
            before node execution begins.
        interrupt_after: List of node names where execution should pause
            after node execution completes.

    Example:
        ```python
        class GraphHandler(InterruptConfigMixin):
            def __init__(self):
                self._set_interrupts(
                    interrupt_before=["approval_node"], interrupt_after=["data_processing"]
                )
        ```
    """

    interrupt_before: list[str] | None
    interrupt_after: list[str] | None

    def _set_interrupts(
        self,
        interrupt_before: list[str] | None,
        interrupt_after: list[str] | None,
    ) -> None:
        """Configure interrupt points for graph execution control.

        Sets up the interrupt configuration for this handler, defining which
        nodes should trigger execution pauses. This method normalizes None
        values to empty lists for consistent handling.

        Args:
            interrupt_before: List of node names where execution should be
                interrupted before the node begins execution. Pass None to
                disable before-interrupts.
            interrupt_after: List of node names where execution should be
                interrupted after the node completes execution. Pass None to
                disable after-interrupts.

        Note:
            This method should be called during handler initialization to
            establish the interrupt configuration before graph execution begins.
        """
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
