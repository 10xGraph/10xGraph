"""
Message context management for agent state in TAF.

This module provides MessageContextManager, which trims and manages the message
history (context) for agent interactions, ensuring efficient context window usage.
"""

import logging
from typing import TypeVar

from agentflow.core.state import Message
from agentflow.core.state.reducers import remove_tool_messages

from .agent_state import AgentState
from .base_context import BaseContextManager


S = TypeVar("S", bound=AgentState)

logger = logging.getLogger("agentflow.state")


class MessageContextManager(BaseContextManager[S]):
    """
    Manages the context field for AI interactions.

    This class trims the context (message history) based on a maximum number of user messages,
    ensuring the first message (usually a system prompt) is always preserved.
    Optionally removes tool-related messages (AI messages with tool calls and tool result messages).
    Generic over AgentState or its subclasses.
    """

    def __init__(
        self,
        max_messages: int = 10,
        remove_tool_msgs: bool = False,
    ) -> None:
        """
        Initialize the MessageContextManager.

        Args:
            max_messages (int): Maximum number of
                user messages to keep in context. Default is 10.
            remove_tool_msgs (bool): Whether to remove tool messages from context.
                Default is False.
        """
        self.max_messages = max_messages
        self.remove_tool_msgs = remove_tool_msgs
        logger.debug("Initialized MessageContextManager with max_messages=%d", max_messages)

    def _trim(self, messages: list[Message]) -> list[Message] | None:
        """
        Trim messages keeping system messages and most recent user messages.

        Returns None if no trimming is needed, otherwise returns the trimmed list.
        """
        # check context is empty
        if not messages:
            logger.debug("No messages to trim; context is empty")
            return None

        # First, remove tool messages if requested
        if self.remove_tool_msgs:
            messages = remove_tool_messages(messages)
            logger.debug("Removed tool messages, %d messages remaining", len(messages))

        # Count user messages
        user_message_count = sum(1 for msg in messages if msg.role == "user")

        if user_message_count <= self.max_messages:
            # Check if we removed tool messages but no trimming needed
            if self.remove_tool_msgs:
                # Return the filtered messages even if count is within limits
                return messages
            # no trimming needed
            logger.debug(
                "No trimming needed; context is within limits (%d user messages)",
                user_message_count,
            )
            return None

        # Separate system messages (usually at the beginning)
        system_messages = [msg for msg in messages if msg.role == "system"]
        non_system_messages = [msg for msg in messages if msg.role != "system"]

        # Find the index of the oldest user message to keep
        user_count = 0
        start_index = len(non_system_messages)

        # Iterate from the end to find the position to start keeping messages
        for i in range(len(non_system_messages) - 1, -1, -1):
            msg = non_system_messages[i]
            if msg.role == "user":
                user_count += 1
                if user_count == self.max_messages:
                    start_index = i
                    break

        # Keep messages from start_index onwards
        final_non_system = non_system_messages[start_index:]

        # Combine system messages (at start) with trimmed conversation
        trimmed_messages = system_messages + final_non_system

        logger.debug(
            "Trimmed from %d to %d messages (%d user messages kept)",
            len(messages),
            len(trimmed_messages),
            self.max_messages,
        )

        return trimmed_messages

    def trim_context(self, state: S) -> S:
        """
        Trim the context in the given AgentState based on the maximum number of user messages.

        The first message (typically a system prompt) is always preserved. Only the most recent
        user messages up to `max_messages` are kept, along with the first message.

        If `remove_tool_msgs` is True, also removes:
        - AI messages that contain tool calls (intermediate tool-calling messages)
        - Tool result messages (role="tool")

        Args:
            state (AgentState): The agent state containing the context to trim.

        Returns:
            S: The updated agent state with trimmed context.
        """
        messages = state.context
        trimmed_messages = self._trim(messages)
        if trimmed_messages is not None:
            state.context = trimmed_messages
        return state

    async def atrim_context(self, state: S) -> S:
        """
        Asynchronous version of trim_context.

        If `remove_tool_msgs` is True, also removes:
        - AI messages that contain tool calls (intermediate tool-calling messages)
        - Tool result messages (role="tool")

        Args:
            state (AgentState): The agent state containing the context to trim.

        Returns:
            S: The updated agent state with trimmed context.
        """
        messages = state.context
        trimmed_messages = self._trim(messages)
        if trimmed_messages is not None:
            state.context = trimmed_messages
        return state
