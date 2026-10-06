"""
Agent state schema for TAF agent graphs.

This module provides the AgentState class, which tracks message context,
context summaries, and internal execution metadata for agent workflows.
Supports subclassing for custom application fields.
"""

import logging
from typing import Annotated

from pydantic import BaseModel, Field

from agentflow.utils.constants import START

from .execution_state import ExecutionState as ExecMeta
from .message import Message
from .reducers import add_messages


# Generic type variable for state subclassing

logger = logging.getLogger("agentflow.state")


class AgentState(BaseModel):
    """Common state schema that includes messages, context and internal execution metadata.

    This class can be subclassed to add application-specific fields while maintaining
    compatibility with the TAF framework. All internal execution metadata
    is preserved through subclassing.

    Notes:
    - `execution_meta` contains internal-only execution progress and interrupt info.
    - Users may subclass `AgentState` to add application fields; internal exec meta remains
      available to the runtime and will be persisted with the state.
    - When subclassing, add your fields but keep the core fields intact.

    Example:
        class MyCustomState(AgentState):
            user_data: dict = Field(default_factory=dict)
            custom_field: str = "default"
    """

    context: Annotated[list[Message], add_messages] = Field(default_factory=list)
    context_summary: str | None = None
    # Internal execution metadata (kept private-ish but accessible to runtime)
    execution_meta: ExecMeta = Field(default_factory=lambda: ExecMeta(current_node=START))

    # Convenience delegation methods for execution meta so callers can use the same API
    def set_interrupt(self, node: str, reason: str, status, data: dict | None = None) -> None:
        """
        Set an interrupt in the execution metadata.

        Args:
            node (str): Node where the interrupt occurred.
            reason (str): Reason for the interrupt.
            status: Execution status to set.
            data (dict | None): Optional additional interrupt data.
        """
        logger.debug("Setting interrupt at node '%s' with reason: %s", node, reason)
        self.execution_meta.set_interrupt(node, reason, status, data)

    def clear_interrupt(self) -> None:
        """
        Clear any interrupt in the execution metadata.
        """
        logger.debug("Clearing interrupt")
        self.execution_meta.clear_interrupt()

    def is_running(self) -> bool:
        """
        Check if the agent state is currently running.

        Returns:
            bool: True if running, False otherwise.
        """
        running = self.execution_meta.is_running()
        logger.debug("State is_running: %s", running)
        return running

    def is_interrupted(self) -> bool:
        """
        Check if the agent state is currently interrupted.

        Returns:
            bool: True if interrupted, False otherwise.
        """
        interrupted = self.execution_meta.is_interrupted()
        logger.debug("State is_interrupted: %s", interrupted)
        return interrupted

    def advance_step(self) -> None:
        """
        Advance the execution step in the metadata.
        """
        old_step = self.execution_meta.step
        self.execution_meta.advance_step()
        logger.debug("Advanced step from %d to %d", old_step, self.execution_meta.step)

    def set_current_node(self, node: str) -> None:
        """
        Set the current node in the execution metadata.

        Args:
            node (str): Node to set as current.
        """
        old_node = self.execution_meta.current_node
        self.execution_meta.set_current_node(node)
        logger.debug("Changed current node from '%s' to '%s'", old_node, node)

    def complete(self) -> None:
        """
        Mark the agent state as completed.
        """
        logger.info("Marking state as completed")
        self.execution_meta.complete()

    def error(self, error_msg: str) -> None:
        """
        Mark the agent state as errored.

        Args:
            error_msg (str): Error message to record.
        """
        logger.error("Setting state error: %s", error_msg)
        self.execution_meta.error(error_msg)

    def is_stopped_requested(self) -> bool:
        """
        Check if a stop has been requested for the agent state.

        Returns:
            bool: True if stop requested, False otherwise.
        """
        stopped = self.execution_meta.is_stopped_requested()
        logger.debug("State is_stopped_requested: %s", stopped)
        return stopped
