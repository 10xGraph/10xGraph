"""Pause a graph from inside a node or tool, and resume it with a value.

Call :func:`interrupt` where the graph needs outside input: an approval, a correction, a choice.
The first time it runs, it stops the graph. The graph saves its state, reports the interrupt
(``StreamEvent.UPDATES`` with ``status="interrupted"`` when streaming, or
``state.pending_interrupt()`` after ``invoke``), and ends the run.

Resume by running the same thread again with a ``resume`` value::

    from tenxgraph.utils import interrupt


    async def refund(amount: float) -> str:
        decision = interrupt(
            {"amount": amount},
            message=f"Refund ${amount}?",
            response_schema={"type": "object", "properties": {"approved": {"type": "boolean"}}},
        )
        if not decision or not decision.get("approved"):
            return "Refund declined"
        return issue_refund(amount)


    await app.ainvoke({"messages": [...]}, config)  # pauses at interrupt()
    await app.ainvoke({"resume": {"approved": True}}, config)  # interrupt() returns the value

On resume the interrupted node (or tool) runs again from the start, and :func:`interrupt`
returns the resume value instead of stopping. Code before the call runs twice, so keep side
effects after it. Several calls in one node or tool are answered in order, one resume per call.
In a ``ToolNode`` running parallel tool calls under ``invoke``, calls that already finished are
not run again (they come from the tool-result ledger).

A client that cancels instead of answering resumes with ``None``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from tenxgraph.core.state import AgentState


# Input key that resumes a paused thread: ``app.invoke({"resume": value}, config)``.
RESUME_KEY = "resume"

# ``execution_meta.interrupt_reason`` for a pause raised by :func:`interrupt`.
INTERRUPT_REASON = "interrupt"

# ``execution_meta.interrupt_data`` key holding the pending :class:`Interrupt`.
INTERRUPT_DATA_KEY = "interrupt"

# ``execution_meta.internal_data`` key holding resume values for the node being retried.
RESUME_VALUES_KEY = "interrupt_resume"


class Interrupt(BaseModel):
    """A pause requested by :func:`interrupt`, waiting for a resume value."""

    id: str = Field(description="Unique id of this pause; clients echo it when resuming.")
    key: str = Field(description="Stable position of the call within its node or tool call.")
    node: str = Field(description="Node that was running when the graph paused.")
    value: Any = Field(default=None, description="Payload passed to interrupt().")
    message: str | None = Field(default=None, description="Human-readable prompt.")
    reason: str = Field(default="input_required", description="Why the graph paused.")
    response_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema the resume value should follow."
    )
    tool_call_id: str | None = Field(
        default=None, description="Tool call that paused, when interrupt() ran inside a tool."
    )


class GraphInterrupt(BaseException):  # - control flow, not an error
    """Raised by :func:`interrupt` to stop the graph; the graph catches it.

    It derives from ``BaseException`` so tool and node error handling (``except Exception``)
    passes it through untouched instead of reporting a failed tool.
    """

    def __init__(self, interrupt: Interrupt) -> None:
        super().__init__(interrupt.message or interrupt.reason)
        self.interrupt = interrupt


# Run-config key holding the running node's interrupt scope (internal; clients cannot set
# ``_`` keys). Plain data, so the config stays serializable for publishers.
SCOPE_CONFIG_KEY = "_interrupt_scope"


_scope: ContextVar[dict[str, Any] | None] = ContextVar("agentflow_interrupt_scope", default=None)
_tool_call_id: ContextVar[str | None] = ContextVar("agentflow_interrupt_tool_call", default=None)


def interrupt(
    value: Any = None,
    *,
    message: str | None = None,
    reason: str = "input_required",
    response_schema: dict[str, Any] | None = None,
) -> Any:
    """Pause the graph until it is resumed with a value, then return that value.

    Args:
        value: Data for whoever resumes the graph (what to approve, what to choose from).
        message: A human-readable prompt, shown by UIs such as CopilotKit.
        reason: A short machine-readable reason, e.g. ``"tool_approval"``.
        response_schema: JSON Schema describing the expected resume value.

    Returns:
        The resume value, on the run that resumes this interrupt (``None`` if the client
        cancelled).

    Raises:
        GraphInterrupt: On the first run, to stop the graph. Do not catch it.
        RuntimeError: When called outside a running graph node or tool.
    """
    scope = _scope.get()
    if scope is None:
        raise RuntimeError("interrupt() can only be called while a graph node or tool is running")

    tool_call_id = _tool_call_id.get()
    node = scope["node"]
    prefix = f"tool:{tool_call_id}" if tool_call_id else f"node:{node}"
    counters = scope["counters"]
    index = counters.get(prefix, 0)
    counters[prefix] = index + 1
    key = f"{prefix}:{index}"

    if key in scope["resume"]:
        return scope["resume"][key]

    raise GraphInterrupt(
        Interrupt(
            id=f"int_{uuid.uuid4().hex}",
            key=key,
            node=node,
            value=value,
            message=message,
            reason=reason,
            response_schema=response_schema,
            tool_call_id=tool_call_id,
        )
    )


def enter_node(config: dict[str, Any], node: str, resume: dict[str, Any] | None) -> None:
    """Open the interrupt scope for ``node`` in the run config (internal)."""
    config[SCOPE_CONFIG_KEY] = {"node": node, "resume": dict(resume or {}), "counters": {}}


def exit_node(config: dict[str, Any]) -> None:
    """Close the interrupt scope opened by :func:`enter_node` (internal)."""
    config.pop(SCOPE_CONFIG_KEY, None)


@contextmanager
def node_scope(config: dict[str, Any], node: str, resume: dict[str, Any] | None) -> Iterator[None]:
    """Keep ``node``'s interrupt scope in the run config while it runs (internal)."""
    enter_node(config, node, resume)
    try:
        yield
    finally:
        exit_node(config)


@contextmanager
def activate(config: dict[str, Any] | None, tool_call_id: str | None = None) -> Iterator[None]:
    """Let :func:`interrupt` see the running node's scope while user code runs (internal).

    Wraps the call into a node function or tool. ``tool_call_id`` keys a tool's calls to that
    tool call, so parallel tools resume independently.
    """
    scope = (config or {}).get(SCOPE_CONFIG_KEY)
    scope_token = _scope.set(scope)
    tool_token = _tool_call_id.set(tool_call_id or None)
    try:
        yield
    finally:
        _reset(_tool_call_id, tool_token)
        _reset(_scope, scope_token)


def _reset(var: ContextVar[Any], token: Any) -> None:
    try:
        var.reset(token)
    except ValueError:
        # The token belongs to another context (a generator resumed elsewhere); clear ours.
        var.set(None)


def pending_interrupt(state: AgentState) -> Interrupt | None:
    """The interrupt a paused thread is waiting on, or ``None``."""
    meta = state.execution_meta
    if meta.interrupt_reason != INTERRUPT_REASON or not state.is_interrupted():
        return None
    data = (meta.interrupt_data or {}).get(INTERRUPT_DATA_KEY)
    return Interrupt.model_validate(data) if data else None


def resume_values(state: AgentState) -> dict[str, Any]:
    """Resume values recorded for the node being retried (internal)."""
    return dict(state.execution_meta.internal_data.get(RESUME_VALUES_KEY) or {})


def record_resume(state: AgentState, value: Any) -> Interrupt:
    """Store ``value`` as the answer to the pending interrupt (internal).

    Raises:
        ValueError: The thread is not paused at an :func:`interrupt`.
    """
    pending = pending_interrupt(state)
    if pending is None:
        raise ValueError("Nothing to resume: this thread is not paused at an interrupt()")
    answers = state.execution_meta.internal_data.setdefault(RESUME_VALUES_KEY, {})
    answers[pending.key] = value
    return pending


def clear_resume_values(state: AgentState) -> None:
    """Forget resume values once the retried node has finished (internal)."""
    state.execution_meta.internal_data.pop(RESUME_VALUES_KEY, None)


def pause_at(state: AgentState, request: Interrupt) -> None:
    """Mark ``state`` as paused before ``request.node`` so resuming re-runs it (internal)."""
    from tenxgraph.core.state.execution_state import ExecutionStatus

    state.set_interrupt(
        request.node,
        INTERRUPT_REASON,
        ExecutionStatus.INTERRUPTED_BEFORE,
        {INTERRUPT_DATA_KEY: request.model_dump(mode="json")},
    )
