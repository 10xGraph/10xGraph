"""Streaming utility functions for TAF graph workflows.

This module provides helper functions for determining whether a result from a node
or tool execution should be treated as non-streaming (i.e., a complete result)
or processed incrementally as a stream. These utilities are used throughout the
graph execution engine to support both synchronous and streaming workflows.
"""

from agentflow.core.state import AgentState, Message


def check_non_streaming(result) -> bool:
    """Determine if a result should be treated as non-streaming.

    Checks whether the given result is a complete, non-streaming output (such as a list,
    dict, string, Message, or AgentState) or if it should be processed incrementally as a stream.

    Args:
        result: The result object returned from a node or tool execution. Can be any type.

    Returns:
        bool: True if the result is non-streaming and should be processed as a complete output;
        False if the result should be handled as a stream.

    Example:
        >>> check_non_streaming([Message.text_message("done")])
        True
        >>> check_non_streaming(Message.text_message("done"))
        True
        >>> check_non_streaming({"choices": [...]})
        True
        >>> check_non_streaming("some text")
        True
    """
    if isinstance(result, list | dict | str):
        return True

    if isinstance(result, Message):
        return True

    if isinstance(result, AgentState):
        return True

    if isinstance(result, dict) and "choices" in result:
        return True

    return bool(isinstance(result, Message))
