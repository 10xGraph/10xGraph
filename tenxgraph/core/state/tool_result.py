"""ToolResult: return type for tool functions that need to update state and return a message."""

from __future__ import annotations

from typing import Any


class ToolResult:
    """Return type for tool functions that need to update state AND return a message to the AI.

    When a tool needs to both mutate graph state fields *and* communicate something back
    to the language model, return a ``ToolResult`` instead of a plain string.

    Attributes:
        message: The text response to return to the AI (used as the tool result content).
            ``content`` is accepted as an alias.
        state: Optional dict mapping state field names to their new values.
            Only fields present in the dict are updated; other fields are left unchanged.
        is_error: If True, the result is marked as a failed tool call (status="failed").

    Example::

        class MyState(AgentState):
            jd_name: str = ""


        def update_context(state: MyState, jd_name: str) -> ToolResult:
            return ToolResult(
                message=f"JD name updated to '{jd_name}'",
                state={"jd_name": jd_name},
            )
    """

    def __init__(
        self,
        message: Any = None,
        state: dict[str, Any] | None = None,
        is_error: bool = False,
        *,
        content: Any = None,
    ) -> None:
        # Accept 'content' as an alias for 'message'
        if content is not None and message is None:
            message = content
        self.message = message
        self.state = state
        self.is_error = is_error

    def __repr__(self) -> str:
        return (
            f"ToolResult(message={self.message!r}, state={self.state!r}, "
            f"is_error={self.is_error!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolResult):
            return NotImplemented
        return (
            self.message == other.message
            and self.state == other.state
            and self.is_error == other.is_error
        )

    def __hash__(self) -> int:
        return hash(
            (self.message, frozenset(self.state.items()) if self.state else None, self.is_error)
        )
