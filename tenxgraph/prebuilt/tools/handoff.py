"""
Handoff tool creation utilities.

This module provides a factory function for creating agent handoff tools
that use a naming convention for transparent agent-to-agent transfers.
"""

import logging
from collections.abc import Callable

from agentflow.core.state.message import Message
from agentflow.core.state.message_block import TextBlock


logger = logging.getLogger("agentflow.prebuilt")


def create_handoff_tool(
    agent_name: str,
    description: str | None = None,
) -> Callable:
    """
    Create a handoff tool that transfers control to another agent.

    The tool uses a naming convention (transfer_to_<agent_name>) to signal
    handoff intent. The graph execution layer detects this pattern and
    navigates to the target agent WITHOUT executing the tool, keeping
    the conversation history clean.

    Args:
        agent_name: Name of the target agent/node to transfer to.
            Must match a node name in your graph.
        description: Optional description for the LLM. If not provided,
            a default description is generated.

    Returns:
        A callable tool function with the naming convention applied.

    Raises:
        ValueError: If agent_name is empty or invalid.
        TypeError: If agent_name is not a string.

    Example:
        >>> transfer_to_researcher = create_handoff_tool(
        ...     agent_name="researcher",
        ...     description="Transfer to research specialist for detailed investigation",
        ... )
        >>>
        >>> # Add to ToolNode
        >>> tools = ToolNode([transfer_to_researcher, other_tool])
        >>>
        >>> # When LLM calls this tool, the graph will automatically
        >>> # navigate to the "researcher" node without executing the tool.

    Note:
        The tool function should never actually execute in normal operation,
        as it's intercepted during graph execution. The implementation is
        provided as a fallback for edge cases.
    """
    # Validate agent_name
    if not agent_name:
        raise ValueError("agent_name cannot be empty")

    if not isinstance(agent_name, str):
        raise TypeError(f"agent_name must be str, got {type(agent_name)}")

    # Warn if agent_name contains underscores (might confuse pattern matching)
    if "_" in agent_name:
        logger.warning(
            "agent_name '%s' contains underscores which may complicate "
            "pattern matching. Consider using a simple name.",
            agent_name,
        )

    tool_name = f"transfer_to_{agent_name}"
    tool_description = description or f"Transfer control to {agent_name} agent"

    def handoff_tool() -> Message:
        """
        Handoff to the target agent.

        This function should never execute in normal operation as it's
        intercepted by the graph execution layer. If it does execute,
        it returns a simple message indicating the handoff.

        Returns:
            Message with handoff indication.
        """
        logger.info(
            "Handoff tool '%s' executed (should have been intercepted). "
            "Check that handoff detection is properly configured in "
            "invoke_node_handler and stream_node_handler.",
            tool_name,
        )

        return Message.tool_message(
            content=[TextBlock(text=f"Handoff to {agent_name}")],
        )

    # Set function metadata for introspection
    handoff_tool.__name__ = tool_name
    handoff_tool.__doc__ = tool_description
    handoff_tool.__handoff_tool__ = True  # type: ignore
    handoff_tool.__target_agent__ = agent_name  # type: ignore

    logger.debug(
        "Created handoff tool '%s' targeting agent '%s'",
        tool_name,
        agent_name,
    )

    return handoff_tool


def is_handoff_tool(tool_name: str) -> tuple[bool, str | None]:
    """
    Check if a tool name follows the handoff naming convention.

    Args:
        tool_name: Name of the tool to check.

    Returns:
        Tuple of (is_handoff, target_agent_name).
        If is_handoff is True, target_agent_name contains the extracted name.
        If is_handoff is False, target_agent_name is None.

    Example:
        >>> is_handoff_tool("transfer_to_researcher")
        (True, "researcher")
        >>> is_handoff_tool("calculate")
        (False, None)
    """
    prefix = "transfer_to_"

    if tool_name.startswith(prefix):
        target = tool_name[len(prefix) :]
        if target:  # Ensure there's actually a target name
            return True, target

    return False, None
