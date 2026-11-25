"""
Abstract base class for context management in TAF agent graphs.

This module provides BaseContextManager, which defines the interface for
trimming and managing message context in agent state objects.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TypeVar

from tenxgraph.utils.callable_utils import call_sync_or_async

from .agent_state import AgentState


S = TypeVar("S", bound=AgentState)

logger = logging.getLogger("tenxgraph.state")


class BaseContextManager[S](ABC):
    """
    Abstract base class for context management in AI interactions.

    Subclasses implement one method, `trim_context`, written as either `def` or
    `async def` (async is recommended, since the engine is async). Generic over
    AgentState or its subclasses.
    """

    @abstractmethod
    def trim_context(self, state: S) -> S | Awaitable[S]:
        """
        Trim the context in the given state.

        Write it as `async def trim_context(self, state)` (recommended) or as a plain
        `def`. The framework calls it through `call_sync_or_async`, so a sync version
        runs in a worker thread and an async version is awaited.

        Args:
            state: The state containing context to be trimmed.

        Returns:
            The state with trimmed context.
        """
        raise NotImplementedError("Subclasses must implement trim_context (sync or async)")

    async def atrim_context(self, state: S) -> S:
        """
        Deprecated: use `trim_context`. Will be removed in 2.0.

        Kept so 1.0 callers keep working. It warns and returns the result of `trim_context`.
        """
        warnings.warn(
            "atrim_context is deprecated and will be removed in 10xgraph 2.0. "
            "Implement and call trim_context (sync or async) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await call_sync_or_async(self.trim_context, state)
