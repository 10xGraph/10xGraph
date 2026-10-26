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

A client that cancels instead of answering resumes with ``None``. A cancel is never checked
against ``response_schema``.

A resume can name the interrupt it answers with ``resume_interrupt_id`` (the ``id`` of the
pending :class:`Interrupt`). A different id raises ``InterruptMismatchError`` and records
nothing, so a retried or stale request cannot answer the next prompt. When the interrupt set a
``response_schema``, the value is validated against it first and a failure raises
``ResumeValidationError``. ``response_schema`` is a JSON Schema dict; see
:func:`validate_against_schema` for the supported subset.
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

# Optional input key naming the interrupt a resume answers (the pending ``Interrupt.id``).
RESUME_INTERRUPT_ID_KEY = "resume_interrupt_id"

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
        default=None,
        description="JSON Schema the resume value must follow (validated on resume).",
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


_JSON_TYPES: dict[str, Any] = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "null": lambda v: v is None,
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def validate_against_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Check ``value`` against a JSON Schema and return the problems found (empty if valid).

    A small validator for the common subset, so no JSON Schema library is needed. Supported
    keywords: ``type`` (a name or a list of names), ``enum``, ``properties``, ``required`` and
    ``items`` (a single schema). Any other keyword is ignored, so a schema using ``oneOf``,
    ``pattern``, ``minimum`` and the like is only checked for the keywords above.
    """
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        known = [n for n in names if n in _JSON_TYPES]
        if known and not any(_JSON_TYPES[n](value) for n in known):
            return [f"{path}: expected {' or '.join(names)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']!r}")
    if isinstance(value, dict):
        errors.extend(
            f"{path}.{name}: required" for name in schema.get("required", []) if name not in value
        )
        for name, sub in (schema.get("properties") or {}).items():
            if name in value and isinstance(sub, dict):
                errors.extend(validate_against_schema(value[name], sub, f"{path}.{name}"))
    items = schema.get("items")
    if isinstance(value, list) and isinstance(items, dict):
        for index, item in enumerate(value):
            errors.extend(validate_against_schema(item, items, f"{path}[{index}]"))
    return errors


def record_resume(state: AgentState, value: Any, interrupt_id: str | None = None) -> Interrupt:
    """Store ``value`` as the answer to the pending interrupt (internal).

    Args:
        state: The paused state.
        value: The resume value. ``None`` means the client cancelled and skips validation.
        interrupt_id: The interrupt the client says it is answering. Optional; when given it
            must equal the pending interrupt's id.

    Raises:
        ValueError: The thread is not paused at an :func:`interrupt`.
        InterruptMismatchError: ``interrupt_id`` is not the pending interrupt's id.
        ResumeValidationError: ``value`` does not satisfy the interrupt's ``response_schema``.

    Nothing is recorded when an error is raised.
    """
    from tenxgraph.core.exceptions import InterruptMismatchError, ResumeValidationError

    pending = pending_interrupt(state)
    if pending is None:
        raise ValueError("Nothing to resume: this thread is not paused at an interrupt()")
    if interrupt_id is not None and interrupt_id != pending.id:
        raise InterruptMismatchError(
            f"Resume answers interrupt '{interrupt_id}' but the thread is paused at "
            f"interrupt '{pending.id}'",
            context={"expected": pending.id, "received": interrupt_id},
        )
    if value is not None and pending.response_schema:
        errors = validate_against_schema(value, pending.response_schema)
        if errors:
            raise ResumeValidationError(
                "Resume value does not match the response schema of interrupt "
                f"'{pending.id}': {'; '.join(errors)}",
                context={"interrupt_id": pending.id, "errors": errors},
            )
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
