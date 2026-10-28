"""Shared board state. Every field here is sent to the browser as an AG-UI STATE_SNAPSHOT."""

from typing import Any

from pydantic import Field

from tenxgraph.core.state import AgentState


class BoardState(AgentState):
    """Task board. CopilotKit reads these fields with useAgent(...).state."""

    board_title: str = "My task board"
    # Each task: {"id": int, "title": str, "status": "todo" | "doing" | "done", "priority": str}
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    next_id: int = 1
