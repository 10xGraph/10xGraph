"""Constants and helpers for ToolNode package.

This module defines constants used throughout the ToolNode implementation,
particularly parameter names that are automatically injected by the TAF
framework during tool execution. These parameters are excluded from tool
schema generation since they are provided by the execution context.

The constants are separated into their own module to avoid circular imports
and maintain a clean public API.

Parameter names that are automatically injected during tool execution.

These parameters are provided by the TAF framework and should be excluded
from tool schema generation. They represent execution context and framework
services that are available to tool functions but not provided by the user.

Parameters:
    tool_call_id: Unique identifier for the current tool execution.
    state: Current AgentState instance for context-aware execution.
    config: Configuration dictionary with execution settings.
    generated_id: Framework-generated identifier for various purposes.
    context_manager: BaseContextManager instance for cross-node operations.
    publisher: BasePublisher instance for event publishing.
    checkpointer: BaseCheckpointer instance for state persistence.
    store: BaseStore instance for data storage operations.

Note:
    Tool functions can declare these parameters in their signatures to receive
    the corresponding services, but they should not be included in the tool
    schema since they're not user-provided arguments.
"""

from __future__ import annotations

import inspect


# Attribute set on skill tools (activate_skill, read_skill_resource) so the
# ToolNode reports their calls to callbacks as InvocationType.SKILL.
SKILL_TOOL_ATTR = "_agentflow_skill_tool"

INJECTABLE_PARAMS = {
    "tool_call_id",
    "state",
    "config",
    "emit",
    "generated_id",
    "context_manager",
    "publisher",
    "checkpointer",
    "store",
    "task_manager",
}


def has_injected_default(param: inspect.Parameter) -> bool:
    """Return True when a parameter default is an InjectQ sentinel.

    Tool signatures may use ``Inject[Service]`` defaults for DI-only parameters.
    Those defaults must never be exposed in tool schemas or user-facing arg lists.
    """
    if param.default is inspect._empty:
        return False

    try:
        return "Inject" in str(type(param.default))
    except Exception:
        return False


def is_injected_param(param_name: str, param: inspect.Parameter) -> bool:
    """Return True when a parameter is framework- or DI-injected."""
    return param_name in INJECTABLE_PARAMS or has_injected_default(param)
