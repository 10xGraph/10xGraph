"""
Thread metadata and status tracking for agent graphs.

This module defines the ThreadInfo model, which tracks thread identity, user, metadata,
status, and timestamps for agent graph execution and orchestration.

Classes:
    ThreadInfo: Metadata and status for a thread in agent execution.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ThreadInfo(BaseModel):
    """
    Metadata and status for a thread in agent execution.

    Attributes:
        thread_id (int | str): Unique identifier for the thread.
        thread_name (str | None): Optional name for the thread.
        user_id (int | str | None): Optional user identifier associated with the thread.
        metadata (dict[str, Any] | None): Optional metadata for the thread.
        updated_at (datetime | None): Timestamp of last update.
        stop_requested (bool): Whether a stop has been requested for the thread.
        run_id (str | None): Optional run identifier for the thread execution.

    Example:
        >>> ThreadInfo(thread_id=1, thread_name="main", user_id=42)
    """

    thread_id: int | str
    thread_name: str | None = None
    user_id: int | str | None = None
    metadata: dict[str, Any] | None = None
    updated_at: datetime | None = Field(default_factory=datetime.now)
    run_id: str | None = None
