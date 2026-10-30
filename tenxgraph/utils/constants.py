"""
Constants and enums for TAF agent graph execution and messaging.

This module defines special node names, message storage levels, execution states,
and response granularity options for agent workflows.
"""

from enum import Enum
from typing import Literal


# Special node names for graph execution flow
START: Literal["__start__"] = "__start__"
END: Literal["__end__"] = "__end__"

# Execution deadlines.
#
# There is no built-in limit on a node or a tool: how long they may run is the developer's
# call. Set the `node_timeout` / `tool_timeout` config keys (seconds) to bound them; a value
# set there applies on both `invoke` and `stream`. None (or 0) means no limit.
DEFAULT_NODE_TIMEOUT_SECONDS: float | None = None
DEFAULT_TOOL_TIMEOUT_SECONDS: float | None = None

# How many tool calls from one assistant message may run at the same time. Each
# parallel branch holds a deep copy of the state, so an unbounded fan-out
# multiplies memory and CPU by the call count. Override per run with the
# `max_parallel_tool_calls` config key, or process-wide with the env var below.
DEFAULT_MAX_PARALLEL_TOOL_CALLS: int = 10
MAX_PARALLEL_TOOL_CALLS_ENV: str = "TENXGRAPH_MAX_PARALLEL_TOOL_CALLS"

# Identity used when a run carries no user_id.
#
# This used to be "test-user-id", a placeholder that reads like a real account:
# with per-user isolation enabled, every un-authenticated run was silently filed
# under that one fake user. "anonymous" matches what the API layer already
# substitutes when no auth is configured, so both layers agree on who "nobody" is.
# It is deliberately not a valid-looking user id.
DEFAULT_ANONYMOUS_USER_ID: str = "anonymous"


class ExecutionState(str, Enum):
    """
    Graph execution states for agent workflows.

    Values:
        RUNNING: Execution is in progress.
        PAUSED: Execution is paused.
        COMPLETED: Execution completed successfully.
        ERROR: Execution encountered an error.
        INTERRUPTED: Execution was interrupted.
        ABORTED: Execution was aborted.
        IDLE: Execution is idle.
    """

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    ABORTED = "aborted"
    IDLE = "idle"


class ResponseGranularity(str, Enum):
    """
    Response granularity options for agent graph outputs.

    Values:
        FULL: State, latest messages.
        PARTIAL: Context, summary, latest messages.
        LOW: Only latest messages.
    """

    FULL = "full"
    PARTIAL = "partial"
    LOW = "low"
