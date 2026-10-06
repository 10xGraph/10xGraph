"""
Execution state management for graph execution in TAF.

This module provides the ExecutionState class and related enums to track
progress, interruptions, and pause/resume functionality for agent graph execution.
"""

import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger("agentflow.state")


class ExecutionStatus(StrEnum):
    """Status of graph execution."""

    RUNNING = "running"
    INTERRUPTED_BEFORE = "interrupted_before"
    INTERRUPTED_AFTER = "interrupted_after"
    COMPLETED = "completed"
    ERROR = "error"


class StopRequestStatus(StrEnum):
    """Status of graph execution."""

    NONE = "none"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"


class ExecutionState(BaseModel):
    """
    Tracks the internal execution state of a graph.

    This class manages the execution progress, interrupt status, and internal
    data that should not be exposed to users.
    """

    # Core execution tracking
    current_node: str
    step: int = 0
    status: ExecutionStatus = ExecutionStatus.RUNNING

    # Interrupt management
    interrupted_node: str | None = None
    interrupt_reason: str | None = None
    interrupt_data: dict[str, Any] | None = None

    # Thread/session identification
    thread_id: str | None = None

    # Stop Current Execution Flag
    stop_current_execution: StopRequestStatus = StopRequestStatus.NONE

    # Internal execution data (hidden from user)
    internal_data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionState":
        """
        Create an ExecutionState instance from a dictionary.

        Args:
            data (dict[str, Any]): Dictionary containing execution state fields.

        Returns:
            ExecutionState: The deserialized execution state object.
        """
        return cls.model_validate(
            {
                "current_node": data["current_node"],
                "step": data.get("step", 0),
                "status": ExecutionStatus(data.get("status", "running")),
                "interrupted_node": data.get("interrupted_node"),
                "interrupt_reason": data.get("interrupt_reason"),
                "interrupt_data": data.get("interrupt_data"),
                "thread_id": data.get("thread_id"),
                "internal_data": data.get("_internal_data", {}),
            }
        )

    def set_interrupt(
        self, node: str, reason: str, status: ExecutionStatus, data: dict[str, Any] | None = None
    ) -> None:
        """
        Set the interrupt state for execution.

        Args:
            node (str): Node where the interrupt occurred.
            reason (str): Reason for the interrupt.
            status (ExecutionStatus): Status to set for the interrupt.
            data (dict[str, Any] | None): Optional additional interrupt data.
        """
        logger.debug(
            "Setting interrupt: node='%s', reason='%s', status='%s'",
            node,
            reason,
            status.value,
        )
        self.interrupted_node = node
        self.interrupt_reason = reason
        self.status = status
        self.interrupt_data = data

    def clear_interrupt(self) -> None:
        """
        Clear the interrupt state and resume execution.
        """
        logger.debug("Clearing interrupt, resuming execution")
        self.interrupted_node = None
        self.interrupt_reason = None
        self.interrupt_data = None
        self.status = ExecutionStatus.RUNNING

    def is_interrupted(self) -> bool:
        """
        Check if execution is currently interrupted.

        Returns:
            bool: True if interrupted, False otherwise.
        """
        interrupted = self.status in [
            ExecutionStatus.INTERRUPTED_BEFORE,
            ExecutionStatus.INTERRUPTED_AFTER,
        ]
        logger.debug("Execution is_interrupted: %s (status: %s)", interrupted, self.status.value)
        return interrupted

    def advance_step(self) -> None:
        """
        Advance to the next execution step.
        """
        old_step = self.step
        self.step += 1
        logger.debug("Advanced step from %d to %d", old_step, self.step)

    def set_current_node(self, node: str) -> None:
        """
        Update the current node in execution state.

        Args:
            node (str): Node to set as current.
        """
        old_node = self.current_node
        self.current_node = node
        logger.debug("Changed current node from '%s' to '%s'", old_node, node)

    def complete(self) -> None:
        """
        Mark execution as completed.
        """
        logger.info("Marking execution as completed")
        self.status = ExecutionStatus.COMPLETED

    def error(self, error_msg: str) -> None:
        """
        Mark execution as errored.

        Args:
            error_msg (str): Error message to record.
        """
        logger.error("Marking execution as errored: %s", error_msg)
        self.status = ExecutionStatus.ERROR
        self.internal_data["error"] = error_msg

    def is_running(self) -> bool:
        """
        Check if execution is currently running.

        Returns:
            bool: True if running, False otherwise.
        """
        running = self.status == ExecutionStatus.RUNNING
        logger.debug("Execution is_running: %s (status: %s)", running, self.status.value)
        return running

    def is_stopped_requested(self) -> bool:
        """
        Check if a stop has been requested for execution.

        Returns:
            bool: True if stop requested, False otherwise.
        """
        stopped = self.stop_current_execution == StopRequestStatus.STOP_REQUESTED
        logger.debug(
            "Execution is_stopped_requested: %s (stop_current_execution: %s)",
            stopped,
            self.stop_current_execution.value,
        )
        return stopped
