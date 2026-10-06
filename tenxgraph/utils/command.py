"""
Command API for AgentGraph in TAF.

This module provides the Command class, which allows nodes to combine state updates
with control flow, similar to LangGraph's Command API. Nodes can update agent state
and direct graph execution to specific nodes or graphs.
"""

from typing import TYPE_CHECKING, TypeVar, Union

from agentflow.core.state.message import Message


if TYPE_CHECKING:
    # Import only for type checking to avoid circular imports at runtime
    from agentflow.core.state import AgentState
    from agentflow.runtime.adapters.llm.base_converter import BaseConverter


StateT = TypeVar("StateT", bound="AgentState")


class Command[StateT: AgentState]:
    """
    Command object that combines state updates with control flow.

    Allows nodes to update agent state and direct graph execution to specific nodes or graphs.
    Similar to LangGraph's Command API.
    """

    PARENT = "PARENT"

    def __init__(
        self,
        update: Union["StateT", None, Message, str, "BaseConverter"] = None,
        goto: str | None = None,
        graph: str | None = None,
        state: StateT | None = None,
    ):
        """
        Initialize a Command object.

        Args:
            update (StateT | None | Message | str | BaseConverter): State update to apply.
            goto (str | None): Next node to execute (node name or END).
            graph (str | None): Which graph to navigate to (None for current, PARENT for parent).
            state (StateT | None): Optional agent state to attach.
        """
        self.update = update
        self.goto = goto
        self.graph = graph
        self.state = state

    def __repr__(self) -> str:
        """
        Return a string representation of the Command object.

        Returns:
            str: String representation of the Command.
        """
        return (
            f"Command(update={self.update}, goto={self.goto}, \n"
            f" graph={self.graph}, state={self.state})"
        )
