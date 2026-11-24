"""Core utility functions for graph execution and state management.

This module provides essential utilities for TAF graph execution, including
state management, message processing, response formatting, and execution flow control.
These functions handle the low-level operations that support graph workflow execution.

The utilities in this module are designed to work with TAF's dependency injection
system and provide consistent interfaces for common operations across different
execution contexts.

Key functionality areas:
- State loading, creation, and synchronization
- Message processing and deduplication
- Response formatting based on granularity levels
- Node execution result processing
- Interrupt handling and execution flow control
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import sys
from typing import Any, TypeVar

from injectq import Inject
from pydantic import TypeAdapter

from tenxgraph.core.authz import public_config
from tenxgraph.core.exceptions import GraphError
from tenxgraph.core.state import AgentState, ExecutionStatus, Message, ToolCallBlock
from tenxgraph.core.state.base_context import BaseContextManager
from tenxgraph.core.state.execution_state import ExecutionState as ExecMeta
from tenxgraph.core.state.execution_state import StopRequestStatus
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.events import ContentType, EventType
from tenxgraph.runtime.publisher.publish import publish_lifecycle
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.utils import (
    END,
    START,
    Command,
    ResponseGranularity,
    add_messages,
)
from tenxgraph.utils.background_task_manager import BackgroundTaskManager
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleContext
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.validators import has_unstorable_text, strip_unstorable


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("tenxgraph.graph")

# How long a cancelled run may take to save its own state before the cancel goes on. This bounds
# our own finalizer save, never a developer tool or node.
FINALIZE_TIMEOUT_SECONDS = 10.0

# ``interrupt_reason`` values of a run that was cut off before it finished (cancel, shutdown,
# crash recovery, stop). The thread is not waiting on anybody: a new user message starts a new
# turn (decision D1). Pauses a developer or client asked for (``interrupt()``,
# ``interrupt_before`` / ``interrupt_after``, client-side tools) are not in this set.
CANCELLED_REASON = "cancelled"
SHUTDOWN_REASON = "shutdown"
CRASHED_REASON = "crashed"
STOP_REASON = "stop_requested"
ABORTED_RUN_REASONS = frozenset({CANCELLED_REASON, SHUTDOWN_REASON, CRASHED_REASON, STOP_REASON})


# Cleanup that must outlive a cancelled waiter (cancel save, run-marker release). A strong
# reference keeps each task alive until it is done.
_detached_tasks: set[asyncio.Future] = set()


def _expire_detached(task: asyncio.Future, limit: float) -> None:
    if not task.done():
        logger.error("Cleanup task still running after %ss; cancelling it", limit)
        task.cancel()


def track_detached(task: asyncio.Future, lifetime: float | None = None) -> asyncio.Future:
    """Hold a strong reference to ``task`` and cancel it once it outlives ``lifetime``.

    The default lifetime is twice ``FINALIZE_TIMEOUT_SECONDS``, so a store that hangs cannot
    keep one task (and the state it holds) per abandoned run alive forever.
    """
    limit = 2 * FINALIZE_TIMEOUT_SECONDS if lifetime is None else lifetime
    timer = asyncio.get_running_loop().call_later(limit, _expire_detached, task, limit)

    def _done(done: asyncio.Future) -> None:
        _detached_tasks.discard(done)
        timer.cancel()

    _detached_tasks.add(task)
    task.add_done_callback(_done)
    return task


def detach_task(coro: Any, lifetime: float | None = None) -> asyncio.Task:
    """Start ``coro`` as a task that outlives a cancelled waiter (see :func:`track_detached`)."""
    task = asyncio.ensure_future(coro)
    track_detached(task, lifetime)
    return task


async def wait_detached(task: asyncio.Future, timeout: float | None = None) -> bool:
    """Wait for ``task`` up to ``timeout`` seconds, even while this coroutine is cancelled.

    A cancel that arrives during the wait does not end it: the wait goes on until ``task`` is
    done or the time is up, then that cancel is raised once. ``task`` is never cancelled here.
    Returns True when ``task`` is done. ``timeout`` defaults to ``FINALIZE_TIMEOUT_SECONDS``.

    A level-triggered anyio cancel scope (Starlette on a client disconnect) re-raises at every
    await, so a plain retry loop would spin; when anyio is loaded the wait runs in a shielded
    anyio scope instead. anyio is never imported here: core does not depend on it.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (FINALIZE_TIMEOUT_SECONDS if timeout is None else timeout)
    cancel: asyncio.CancelledError | None = None
    anyio = sys.modules.get("anyio")
    scope = anyio.CancelScope(shield=True) if anyio is not None else contextlib.nullcontext()
    with scope:
        while not task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait({task}, timeout=remaining)
            except asyncio.CancelledError as e:
                cancel = e
    if cancel is not None:
        raise cancel
    return task.done()


class LastSave:
    """The latest durable save a run started, so cancel cleanup can wait for it, not race it."""

    __slots__ = ("completed", "task")

    def __init__(self) -> None:
        self.task: asyncio.Future | None = None
        self.completed = False  # the saved state was COMPLETED

    def finished_completed(self) -> bool:
        """True when the run's COMPLETED state was saved without error."""
        task = self.task
        return (
            self.completed
            and task is not None
            and task.done()
            and not task.cancelled()
            and task.exception() is None
        )


def is_dead_run(state: AgentState, active_run_id: str | None, config: dict[str, Any]) -> bool:
    """True when ``state`` was left RUNNING by a run that is no longer alive.

    ``active_run_id`` is the run the thread's active-run marker names (None when the marker
    is gone or expired). A state saved before run ids existed (``run_id`` None) cannot be
    judged and is never treated as dead. A state written by the run that is loading it is the
    run's own.
    """
    meta = state.execution_meta
    if meta.status != ExecutionStatus.RUNNING or not meta.run_id:
        return False
    if meta.run_id == config.get("run_id"):
        return False
    return meta.run_id != active_run_id


def has_new_user_message(input_data: dict[str, Any]) -> bool:
    """True when the call brings at least one new message from the user."""
    return any(getattr(m, "role", None) == "user" for m in input_data.get("messages") or [])


async def parse_response(
    state: AgentState,
    messages: list[Message],
    response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    token_usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Parse and format execution response based on specified granularity level.

    Formats the final response from graph execution according to the requested
    granularity level, allowing clients to receive different levels of detail
    depending on their needs.

    Args:
        state: The final agent state after graph execution.
        messages: List of messages generated during execution.
        response_granularity: Level of detail to include in the response:
            - FULL: Returns complete state object and all messages
            - PARTIAL: Returns context, summary, and messages
            - LOW: Returns only the messages (default)

    Returns:
        Dictionary containing the formatted response with keys depending on
        granularity level. Always includes 'messages' key with execution results.

    Example:
        ```python
        # LOW granularity (default)
        response = await parse_response(state, messages)
        # Returns: {"messages": [Message(...), ...]}

        # FULL granularity
        response = await parse_response(state, messages, ResponseGranularity.FULL)
        # Returns: {"state": AgentState(...), "messages": [Message(...), ...]}
        ```
    """
    match response_granularity:
        case ResponseGranularity.FULL:
            # Return full state and messages
            return {"state": state, "messages": messages, "token_usage": token_usage}
        case ResponseGranularity.PARTIAL:
            # Return state and summary of messages
            return {
                "context": state.context,
                "summary": state.context_summary,
                "messages": messages,
                "token_usage": token_usage,
            }
        case ResponseGranularity.LOW:
            # Return all messages from state context
            return {"messages": messages, "token_usage": token_usage}

    return {"messages": messages, "token_usage": token_usage}


# Utility to update only provided fields in state
def _update_state_fields(state, partial: dict):
    """Update only the provided fields in the state object."""
    for k, v in partial.items():
        # Avoid updating special fields
        if k in ("context", "context_summary", "execution_meta"):
            continue
        if hasattr(state, k):
            setattr(state, k, v)


async def validate_message_content(
    message: list[Message],
    config: dict[str, Any] | None = None,
    callback_mgr: CallbackManager = Inject[CallbackManager],  # will be auto-injected
) -> bool:
    """Validate message content using registered validators in callback manager.

    Executes all validators registered in the callback manager.
    This allows for flexible, configurable validation strategies.

    Args:
        message: List of Message objects to validate.
        config: Execution config passed through to validators for event publishing.
        callback_manager: CallbackManager instance (required).

    Returns:
        True if all validators pass.

    Raises:
        ValidationError: If any validator fails.
    """
    callback_mgr = fresh(callback_mgr)
    # Require callback manager to be provided
    if not callback_mgr:
        logger.debug("No callback manager provided, skipping validation")
        return True

    try:
        await callback_mgr.execute_validators(message)
    except Exception as e:
        # Publish the rejection so the event stream captures it.
        if config:
            publish_lifecycle(
                config,
                "validation_rejected",
                {"error": str(e), "rejected_count": len(message)},
                event_type=EventType.ERROR,
                content_type=ContentType.ERROR,
            )
        raise

    logger.debug("All validators passed")
    return True


def clean_unstorable_messages(messages: list[Message], config: dict[str, Any]) -> int:
    """Clean NUL and lone surrogates out of ``messages`` in place; returns how many changed.

    Used on what the model and tools produced (and on a turn a crash left behind): the run
    must not die because of text it did not send. Client input is refused instead (see
    :func:`reject_nul_messages`). Costs one scan when nothing is wrong.
    """
    cleaned = 0
    for i, m in enumerate(messages):
        if has_unstorable_text(m):
            messages[i] = strip_unstorable(m)
            cleaned += 1
    if cleaned:
        logger.warning(
            "Cleaned NUL or lone surrogate characters out of %d message(s) on thread '%s'",
            cleaned,
            config.get("thread_id"),
        )
    return cleaned


def reject_nul_messages(messages: list[Any]) -> None:
    """Refuse messages holding NUL or a lone surrogate: Postgres can never store them.

    Saving one would fail on every later call of the thread (crash recovery retries the
    save), so it is refused before anything is written or run.
    """
    if any(has_unstorable_text(m) for m in messages or []):
        from tenxgraph.utils.validators import ValidationError

        raise ValidationError(
            "Messages cannot contain the NUL character (\\x00) or a lone surrogate.",
            "invalid_character",
        )


async def load_or_create_state[StateT: AgentState](  # noqa: PLR0912, PLR0915
    input_data: dict[str, Any],
    config: dict[str, Any],
    old_state: StateT,
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # will be auto-injected
) -> StateT:
    """Load existing state from checkpointer or create new state.

    Attempts to fetch a realtime-synced state first, then falls back to
    the persistent checkpointer. If no existing state is found, creates
    a new state from the `StateGraph`'s prototype state and merges any
    incoming messages. Supports partial state update via 'state' in input_data.
    """
    checkpointer = fresh(checkpointer)
    # Before the run is registered or any state is touched: a bad message must leave no trace.
    reject_nul_messages(input_data.get("messages", []))
    if has_unstorable_text(input_data.get("state") or {}):
        from tenxgraph.utils.validators import ValidationError

        raise ValidationError(
            "State cannot contain the NUL character (\\x00) or a lone surrogate.",
            "invalid_character",
        )
    logger.debug("Loading or creating state with thread_id=%s", config.get("thread_id", "default"))

    # Try to load existing state if checkpointer is available
    if checkpointer:
        # Register this run as the active one on the thread. A stop request belongs to the
        # run it targets: only a stale one (aimed at no run, or at a run that already ended)
        # is dropped here, so a new run cannot erase a stop meant for a run still going.
        # Read the marker first: it names the run that was alive before this one took over, and
        # crash recovery needs it. abegin_run overwrites it.
        previous_run = await checkpointer.aget_active_run(config)
        await checkpointer.abegin_run(config)

        logger.debug("Attempting to load existing state from checkpointer")
        # first check realtime-synced state
        existing_state: StateT | None = await checkpointer.aget_state_cache(config)
        if not existing_state:
            logger.debug("No synced state found, trying persistent checkpointer")
            # If no synced state, try to get from persistent checkpointer
            existing_state = await checkpointer.aget_state(config)

        if existing_state:
            logger.info(
                "Loaded existing state with %d context messages, current_node=%s, step=%d",
                len(existing_state.context) if existing_state.context else 0,
                existing_state.execution_meta.current_node,
                existing_state.execution_meta.step,
            )
            # A stop that interrupted an earlier run is persisted in the state and
            # is never cleared by set_interrupt, so a thread stopped mid-run would
            # carry STOP_REQUESTED into its next run and stop itself again at the
            # first node. Starting a run supersedes any prior stop request.
            existing_state.execution_meta.stop_current_execution = StopRequestStatus.NONE

            # A RUNNING state with no live run behind it is a run that died (SIGKILL, OOM, a
            # consumer that closed the stream early). Repair it from this state, which is the
            # cached one when a cache exists and so still holds the dead turn's user message.
            if is_dead_run(existing_state, previous_run, config):
                from .handler_utils import recover_crashed_run

                await recover_crashed_run(existing_state, config)

            # A run that was cut off (cancel, shutdown, crash, stop) and now gets a new user
            # message: start a new turn on the repaired context. Without new input the call
            # resumes at the saved node (see check_interrupted).
            meta = existing_state.execution_meta
            if (
                meta.is_interrupted()
                and meta.interrupt_reason in ABORTED_RUN_REASONS
                and has_new_user_message(input_data)
            ):
                logger.info(
                    "New message after a run that ended '%s': starting a new turn",
                    meta.interrupt_reason,
                )
                meta.current_node = START
                meta.step = 0
                meta.clear_interrupt()
                meta.reset_turn_data()

            # Reset execution if graph previously completed (current_node == END)
            # This allows multi-turn conversations to continue from a completed state
            if existing_state.execution_meta.current_node == END:
                logger.info(
                    "Resetting completed state for multi-turn continuation (was at END, step=%d)",
                    existing_state.execution_meta.step,
                )
                existing_state.execution_meta.current_node = START
                existing_state.execution_meta.step = 0
                existing_state.execution_meta.status = ExecutionStatus.RUNNING
                existing_state.execution_meta.clear_interrupt()
                existing_state.execution_meta.stop_current_execution = StopRequestStatus.NONE
                # Loop limits (max_rounds, max_iterations, max_attempts) are per turn.
                existing_state.execution_meta.reset_turn_data()
            # Merge new messages with existing context
            new_messages = input_data.get("messages", [])
            if new_messages:
                logger.debug("Merging %d new messages with existing context", len(new_messages))
                # Validate message content before adding
                await validate_message_content(new_messages, config=config)
                pre_merge_len = len(existing_state.context)
                existing_state.context = add_messages(existing_state.context, new_messages)

                # Fallback: when all input messages were filtered out by ID deduplication
                # (e.g. frontend sends full history with placeholder id="0") and the last
                # input message is a user message while the context ends with an assistant
                # message, forcibly add the last user message so the conversation continues.
                if len(existing_state.context) == pre_merge_len:
                    last_input = new_messages[-1]
                    last_ctx = existing_state.context[-1] if existing_state.context else None
                    if last_input.role == "user" and (last_ctx is None or last_ctx.role != "user"):
                        from tenxgraph.core.state.message import generate_id

                        new_msg = last_input.model_copy(update={"message_id": generate_id(None)})
                        existing_state.context = [*existing_state.context, new_msg]
                        logger.info(
                            "Added last user message with new ID to continue conversation "
                            "(fallback for duplicate message_id in input)"
                        )

            # Merge partial state fields if provided
            partial_state = input_data.get("state", {})
            if partial_state and isinstance(partial_state, dict):
                logger.debug("Merging partial state with %d fields", len(partial_state))
                _update_state_fields(existing_state, partial_state)
            # Update current node if available
            if "current_node" in partial_state and partial_state["current_node"] is not None:
                existing_state.set_current_node(partial_state["current_node"])
            existing_state.execution_meta.run_id = config.get("run_id")
            return existing_state
    else:
        logger.debug("No checkpointer available, will create new state")

    # Create new state by deep copying the graph's prototype state
    logger.info("Creating new state from graph prototype")
    state = copy.deepcopy(old_state)

    # Ensure core AgentState fields are properly initialized
    if hasattr(state, "context") and not isinstance(state.context, list):
        state.context = []
        logger.debug("Initialized empty context list")
    if hasattr(state, "context_summary") and state.context_summary is None:
        state.context_summary = None
        logger.debug("Initialized context_summary as None")
    if hasattr(state, "execution_meta"):
        # Create a fresh execution metadata
        state.execution_meta = ExecMeta(current_node=START)
        logger.debug("Created fresh execution metadata starting at %s", START)

    # Set thread_id in execution metadata
    thread_id = config.get("thread_id", "default")
    state.execution_meta.thread_id = thread_id
    state.execution_meta.run_id = config.get("run_id")
    logger.debug("Set thread_id to %s", thread_id)

    # Merge new messages with context
    new_messages = input_data.get("messages", [])
    if new_messages:
        logger.debug("Adding %d new messages to fresh state", len(new_messages))
        # Validate message content before adding
        await validate_message_content(new_messages, config=config)
        state.context = add_messages(state.context, new_messages)
    # Merge partial state fields if provided
    partial_state = input_data.get("state", {})
    if partial_state and isinstance(partial_state, dict):
        logger.debug("Merging partial state with %d fields", len(partial_state))
        _update_state_fields(state, partial_state)

    logger.info(
        "Created new state with %d context messages", len(state.context) if state.context else 0
    )
    if "current_node" in partial_state and partial_state["current_node"] is not None:
        state.set_current_node(partial_state["current_node"])
    return state  # type: ignore[return-value]


async def reload_state[StateT: AgentState](
    config: dict[str, Any],
    old_state: StateT,
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # will be auto-injected
) -> StateT:
    """Load existing state from checkpointer or create new state.

    Attempts to fetch a realtime-synced state first, then falls back to
    the persistent checkpointer. If no existing state is found, creates
    a new state from the `StateGraph`'s prototype state and merges any
    incoming messages. Supports partial state update via 'state' in input_data.
    """
    checkpointer = fresh(checkpointer)
    logger.debug("Loading or creating state with thread_id=%s", config.get("thread_id", "default"))

    if not checkpointer:
        return old_state

    # first check realtime-synced state
    existing_state: AgentState | None = await checkpointer.aget_state_cache(config)
    if not existing_state:
        logger.debug("No synced state found, trying persistent checkpointer")
        # If no synced state, try to get from persistent checkpointer
        existing_state = await checkpointer.aget_state(config)

    if not existing_state:
        logger.debug("No existing state found to reload, returning old state")
        return old_state

    logger.info(
        "Loaded existing state with %d context messages, current_node=%s, step=%d",
        len(existing_state.context) if existing_state.context else 0,
        existing_state.execution_meta.current_node,
        existing_state.execution_meta.step,
    )
    return existing_state


_field_adapters: dict[tuple[type, str], TypeAdapter] = {}


def _validated_field(state: AgentState, name: str, value: Any) -> Any:
    """``value`` checked against the type of the state field ``name``; ValueError if unknown."""
    cls = type(state)
    field = cls.model_fields.get(name)
    if field is None:
        known = ", ".join(sorted(cls.model_fields))
        raise ValueError(f"Unknown state field '{name}' in a node's dict result. Fields: {known}.")
    adapter = _field_adapters.get((cls, name))
    if adapter is None:
        adapter = _field_adapters[(cls, name)] = TypeAdapter(field.annotation)
    return adapter.validate_python(value)


async def process_node_result[StateT: AgentState](  # noqa: PLR0912, PLR0915
    result: Any,
    state: StateT,
    messages: list[Message],
    seen_message_ids: set[str] | None = None,
) -> tuple[StateT, list[Message], str | None]:
    """
    Applies what a node returned to the agent state and the node's message list.

    Accepted return values (anything else raises ``ValueError``, which fails the node with
    ``NODE_001``):

    - ``None``: no change.
    - ``Message``, ``str``, or a ``ModelResponseConverter``: appended as a message (a ``str``
      becomes an assistant message).
    - ``list`` of those: each item appended.
    - ``AgentState``: replaces the state; its new messages are recorded.
    - ``dict``: a partial state update. Each key is a state field set to the value, validated
      against the field's type; the key ``"messages"`` (a ``Message`` or a list of them) appends
      messages. An unknown field name raises ``ValueError``.
    - ``Command``: ``update`` may be any of the above (``None`` or empty means no change);
      ``goto`` is returned as the next node.

    Messages are deduplicated by ``message_id``.

    Args:
        result (Any): The output of the node.
        state (StateT): The current agent state.
        messages (list[Message]): The list of messages accumulated so far.
        seen_message_ids (set[str] | None): Ids of the messages in ``state.context`` before the
            node ran. A node that appends to ``state.context`` and returns the same state
            object leaves nothing to compare against, so without this snapshot its new
            messages would never be reported (or streamed). Defaults to the ids in ``state``.

    Returns:
        tuple[StateT, list[Message], str | None]:
            - The updated agent state.
            - The updated list of messages (with new, unique messages added).
            - The identifier of the next node to execute, if specified; otherwise, None.
    """
    next_node = None
    existing_ids = {msg.message_id for msg in messages}
    new_messages = []

    def add_unique_message(msg: Message) -> None:
        """Add message only if it doesn't already exist."""
        if msg.message_id not in existing_ids:
            new_messages.append(msg)
            existing_ids.add(msg.message_id)

    async def create_and_add_message(content: Any) -> Message:
        """Create message from content and add if unique."""
        if isinstance(content, Message):
            msg = content
        elif isinstance(content, ModelResponseConverter):
            msg = await content.invoke()
        elif isinstance(content, str):
            msg = Message.text_message(
                content,
                role="assistant",
            )

        else:
            err = f"""
            Unsupported node return type: {type(content)}.
            Supported: None, Message, str, ModelResponseConverter, AgentState, Command,
            a list of messages, or a dict of state field updates.
            """
            raise ValueError(err)

        add_unique_message(msg)
        return msg

    def handle_state_message(old_state: StateT, new_state: StateT) -> None:
        """Handle state messages by updating the context."""
        if seen_message_ids is not None:
            old_ids = seen_message_ids
        else:
            old_ids = {msg.message_id for msg in old_state.context or []}

        if not new_state.context:
            return
        # now save all the new messages
        for msg in new_state.context:
            if msg.message_id in old_ids:
                continue
            # otherwise save it
            add_unique_message(msg)

    async def apply_dict(update: dict[str, Any]) -> None:
        """A partial state update: set the fields, append ``messages``."""
        update = dict(update)
        new = update.pop("messages", None)
        checked = {name: _validated_field(state, name, value) for name, value in update.items()}
        items = new if isinstance(new, list) else [] if new is None else [new]
        if not all(isinstance(item, Message) for item in items):
            raise ValueError("The 'messages' key of a dict result must be a Message or a list.")
        for name, value in checked.items():
            setattr(state, name, value)
        for item in items:
            add_unique_message(item)

    # Process different result types
    if result is None:
        pass

    elif isinstance(result, Command):
        # Handle state updates
        if result.update:
            if isinstance(result.update, dict):
                await apply_dict(result.update)
            elif isinstance(result.update, AgentState):
                handle_state_message(state, result.update)  # type: ignore[assignment]
                state = result.update  # type: ignore[assignment]
            elif isinstance(result.update, list):
                for item in result.update:
                    await create_and_add_message(item)
            else:
                await create_and_add_message(result.update)

        # Handle navigation
        next_node = result.goto

    elif isinstance(result, AgentState):
        handle_state_message(state, result)  # type: ignore[assignment]
        state = result  # type: ignore[assignment]

    elif isinstance(result, Message):
        add_unique_message(result)

    elif isinstance(result, dict):
        await apply_dict(result)

    elif isinstance(result, list):
        # Handle list of items (convert each to message)
        for item in result:
            await create_and_add_message(item)
    else:
        # A str, a ModelResponseConverter, or an unsupported type (raises)
        await create_and_add_message(result)

    # Add new messages to the main list and state context
    if new_messages:
        messages.extend(new_messages)
        state.context = add_messages(state.context, new_messages)

    return state, messages, next_node


def get_next_node(
    current_node: str,
    state: AgentState,
    edges: list,
) -> str:
    """Get the next node to execute based on edges."""
    # Find outgoing edges from current node
    outgoing_edges = [e for e in edges if e.from_node == current_node]

    if not outgoing_edges:
        logger.debug("No outgoing edges from node '%s', ending execution", current_node)
        return END

    # Handle conditional edges
    for edge in outgoing_edges:
        if edge.condition:
            # A condition that raises used to be swallowed with `continue`, so the
            # graph quietly fell through to the first static edge -- or to END --
            # and the run took a path nobody chose. Silent misrouting is worse than
            # a failed run: the caller gets a confident, wrong answer. Routing is a
            # correctness decision, so a broken condition fails the run instead.
            try:
                condition_result = edge.condition(state)
            except Exception as e:
                logger.exception(
                    "Condition for edge %s -> %s raised; cannot route",
                    edge.from_node,
                    edge.to_node,
                )
                raise GraphError(
                    message=(f"Conditional edge from '{edge.from_node}' raised while routing: {e}"),
                    error_code="GRAPH_ROUTING_001",
                    context={
                        "from_node": edge.from_node,
                        "to_node": edge.to_node,
                        "error_type": type(e).__name__,
                    },
                ) from e

            if hasattr(edge, "condition_result") and edge.condition_result is not None:
                # Mapped conditional edge
                if condition_result == edge.condition_result:
                    return edge.to_node
            elif isinstance(condition_result, str):
                return condition_result
            elif condition_result:
                return edge.to_node

    # Return first static edge if no conditions matched
    static_edges = [e for e in outgoing_edges if not e.condition]
    if static_edges:
        return static_edges[0].to_node

    logger.debug("No valid edges found from node '%s', ending execution", current_node)
    return END


async def call_realtime_sync(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # will be auto-injected
) -> None:
    """Call the realtime state sync hook if provided."""
    checkpointer = fresh(checkpointer)
    if checkpointer:
        logger.debug("Calling realtime state sync hook")
        # await call_sync_or_async(checkpointer.a, config, state)
        await checkpointer.aput_state_cache(config, state)


async def sync_data(
    state: AgentState,
    config: dict[str, Any],
    messages: list[Message],
    trim: bool = False,
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # will be auto-injected
    context_manager: BaseContextManager = Inject[BaseContextManager],  # will be auto-injected
    callback_mgr: CallbackManager = Inject[CallbackManager],  # will be auto-injected
    last_save: LastSave | None = None,
    task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
) -> bool:
    """Sync the current state and messages to the checkpointer.

    ``last_save`` records the save this call starts, for the run's cancel cleanup. Once a
    COMPLETED state is saved, this run's tool ledger rows are cleared in the background (see
    :func:`_schedule_ledger_clear`).
    """
    checkpointer = fresh(checkpointer)
    context_manager = fresh(context_manager)
    callback_mgr = fresh(callback_mgr)
    task_manager = fresh(task_manager)
    is_context_trimmed = False

    # Model and tool output must be storable, or the save fails on every later call.
    clean_unstorable_messages(state.context, config)
    clean_unstorable_messages(messages, config)

    # Ids of the run's tool-calling messages, taken before trimming drops any of them.
    origin_ids = _tool_origin_ids([*state.context, *messages])
    new_state = copy.deepcopy(state)
    # if context manager is available then utilize it
    if context_manager and trim:
        new_state = await context_manager.atrim_context(state)
        is_context_trimmed = True

    # Fire on_checkpoint hook before persisting
    if callback_mgr and callback_mgr._lifecycle_hooks:
        lifecycle_context = GraphLifecycleContext(config=public_config(config))
        new_state, messages = await callback_mgr.fire_on_checkpoint(
            lifecycle_context,
            state=new_state,
            messages=messages,
            is_context_trimmed=is_context_trimmed,
        )
    publish_lifecycle(config, "checkpoint", {}, {"trimmed": is_context_trimmed})

    if checkpointer:
        logger.debug("Persisting state and %d messages to checkpointer", len(messages))

        # Durable store is the source of truth: write state + messages atomically
        # FIRST (audit M7), so a failure surfaces before the cache is updated and
        # the cache can never hold state that was never persisted (audit B5).
        # Both writes run as one shielded unit: a cancel that arrives during the save cannot
        # leave the durable store new and the cache old.
        async def _persist() -> None:
            await checkpointer.aput_checkpoint(config, new_state, messages)
            # Then refresh the realtime cache with the SAME trimmed state we just
            # persisted, so cache and durable store never diverge in shape (audit M6).
            await checkpointer.aput_state_cache(config, new_state)

        save = asyncio.ensure_future(_persist())
        if last_save is not None:
            last_save.task = save
            last_save.completed = new_state.execution_meta.status == ExecutionStatus.COMPLETED
        try:
            await asyncio.shield(save)
        except asyncio.CancelledError:
            if not save.done():
                # Cancelled mid-save: the save goes on. Wait for it (bounded) so the cleanup
                # that follows never runs beside it, then let the cancel go on.
                track_detached(save)
                if not await wait_detached(save):
                    logger.warning(
                        "Save still running %ss after cancel; it goes on in the background",
                        FINALIZE_TIMEOUT_SECONDS,
                    )
            if save.done() and not save.cancelled() and save.exception() is not None:
                logger.warning("Save failed after cancel (%r)", save.exception())
            raise

        if new_state.execution_meta.status == ExecutionStatus.COMPLETED:
            _schedule_ledger_clear(checkpointer, config, task_manager, origin_ids)

    return is_context_trimmed


def _tool_origin_ids(messages: list[Message]) -> list[str]:
    """``message_id`` of each message that issued tool calls, as the ledger keys spell it.

    ToolNode keys a call on the message holding it in ``tools_calls``; a ``ToolCallBlock`` in
    the content counts too, so no recorded call is missed.
    """
    ids: dict[str, None] = {}
    for msg in messages:
        if msg.tools_calls or any(isinstance(b, ToolCallBlock) for b in msg.content or []):
            ids[str(msg.message_id)] = None
    return list(ids)


async def _clear_ledger(
    checkpointer: BaseCheckpointer, config: dict[str, Any], origin_ids: list[str]
) -> None:
    try:
        deleted = await checkpointer.aclear_tool_results(config, origin_ids)
        logger.debug("Cleared %s tool ledger rows of thread '%s'", deleted, config.get("thread_id"))
    except Exception as e:
        logger.warning(
            "Could not clear the tool ledger of thread '%s' (%r); aprune_tool_results "
            "removes the rows later",
            config.get("thread_id"),
            e,
        )


def _schedule_ledger_clear(
    checkpointer: BaseCheckpointer,
    config: dict[str, Any],
    task_manager: BackgroundTaskManager | None,
    origin_ids: list[str],
) -> None:
    """Delete the run's tool ledger rows after it completed, without holding the caller.

    A completed run has no node left to replay, so its records are spent. Only rows keyed on
    ``origin_ids`` (the run's tool-calling messages) go: a newer run on the thread may record
    rows before this task runs, and those must stay. Rows of a message trimmed out of the
    state before this save stay too, until pruned. Runs that paused, stopped, failed or were
    cancelled never save COMPLETED and keep them. The task runs on the graph's own task
    manager, which ``aclose()`` drains before it releases the checkpointer.
    """
    if not origin_ids or not getattr(checkpointer, "record_tool_result", True):
        return
    if task_manager is None:
        return
    # A copy: the caller goes on using (and popping keys from) its config.
    task = task_manager.create_task(
        _clear_ledger(checkpointer, dict(config), origin_ids), name="clear_tool_ledger"
    )
    if task is None:
        logger.debug(
            "Tool ledger clear of thread '%s' dropped by backpressure; the rows stay until pruned",
            config.get("thread_id"),
        )


def calculate_token_usage(messages: list[Message]) -> dict[str, int]:
    """Calculate total token usage from all messages in the state.

    Aggregates token usage across all messages in the state's context,
    including input tokens (prompt_tokens), output tokens (completion_tokens),
    and reasoning tokens.

    Args:
        messages: The list of messages containing token usage information.

    Returns:
        Dictionary containing:
            - total_input_tokens: Total prompt/input tokens used
            - total_output_tokens: Total completion/output tokens used
            - total_reasoning_tokens: Total reasoning tokens used
            - total_tokens: Sum of input and output tokens

    Example:
        ```python
        usage = calculate_token_usage(state)
        # Returns: {
        #     "total_input_tokens": 1500,
        #     "total_output_tokens": 800,
        #     "total_reasoning_tokens": 200,
        #     "total_tokens": 2300
        # }
        ```
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0

    if not messages:
        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

    for message in messages:
        if message.usages:
            total_input_tokens += message.usages.prompt_tokens
            total_output_tokens += message.usages.completion_tokens
            total_reasoning_tokens += message.usages.reasoning_tokens

    # Note: total_tokens is input + output only (reasoning tracked separately)
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_tokens": total_tokens,
    }
