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
from collections.abc import Callable
from typing import Any, TypeVar

from injectq import Inject

from tenxgraph.core.exceptions import GraphError
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.core.state.base_context import BaseContextManager
from tenxgraph.core.state.execution_state import ExecutionState as ExecMeta
from tenxgraph.core.state.execution_state import StopRequestStatus
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.utils import (
    END,
    START,
    Command,
    ResponseGranularity,
    add_messages,
)
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleContext
from tenxgraph.utils.injection import fresh


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

    # Execute validation
    await callback_mgr.execute_validators(message, config=config)

    logger.debug("All validators passed")
    return True


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

            # Normalize legacy node names (backward compatibility)
            # Some older runs may have persisted 'start'/'end' instead of '__start__'/'__end__'
            if existing_state.execution_meta.current_node == "start":
                existing_state.execution_meta.current_node = START
                logger.debug("Normalized legacy current_node 'start' to '%s'", START)
            elif existing_state.execution_meta.current_node == "end":
                existing_state.execution_meta.current_node = END
                logger.debug("Normalized legacy current_node 'end' to '%s'", END)
            elif existing_state.execution_meta.current_node == "__start__":
                existing_state.execution_meta.current_node = START
                logger.debug("Normalized legacy current_node '__start__' to '%s'", START)
            elif existing_state.execution_meta.current_node == "__end__":
                existing_state.execution_meta.current_node = END
                logger.debug("Normalized legacy current_node '__end__' to '%s'", END)

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
        # Normalize legacy values if provided in partial state
        next_node = partial_state["current_node"]
        if next_node == "__start__":
            next_node = START
        elif next_node == "__end__":
            next_node = END
        state.set_current_node(next_node)
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
        logger.warning("No existing state found to reload, returning old state")
        return old_state

    logger.info(
        "Loaded existing state with %d context messages, current_node=%s, step=%d",
        len(existing_state.context) if existing_state.context else 0,
        existing_state.execution_meta.current_node,
        existing_state.execution_meta.step,
    )
    # Normalize legacy node names (backward compatibility)
    # Some older runs may have persisted 'start'/'end' instead of '__start__'/'__end__'
    if existing_state.execution_meta.current_node == "start":
        existing_state.execution_meta.current_node = START
        logger.debug("Normalized legacy current_node 'start' to '%s'", START)
    elif existing_state.execution_meta.current_node == "end":
        existing_state.execution_meta.current_node = END
        logger.debug("Normalized legacy current_node 'end' to '%s'", END)
    elif existing_state.execution_meta.current_node == "__start__":
        existing_state.execution_meta.current_node = START
        logger.debug("Normalized legacy current_node '__start__' to '%s'", START)
    elif existing_state.execution_meta.current_node == "__end__":
        existing_state.execution_meta.current_node = END
        logger.debug("Normalized legacy current_node '__end__' to '%s'", END)
    return existing_state


async def process_node_result[StateT: AgentState](  # noqa: PLR0915
    result: Any,
    state: StateT,
    messages: list[Message],
    seen_message_ids: set[str] | None = None,
) -> tuple[StateT, list[Message], str | None]:
    """
    Processes the result from a node execution, updating the agent state, message list,
    and determining the next node.

    Supports:
    - Handling results of type Command, AgentState, Message, list, str, dict,
            or other types.
        - Deduplicating messages by message_id.
        - Updating the agent state and its context with new messages.
        - Extracting navigation information (next node) from Command results.

    Args:
        result (Any): The output from a node execution. Can be a Command, AgentState, Message,
            list, str, dict, ModelResponse, or other types.
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
            Unsupported content type for message: {type(content)}.
            Supported types are: AgentState, Message, ModelResponseConverter, Command, str,
            dict (OpenAI style/Native Message).
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

    # Process different result types
    if isinstance(result, Command):
        # Handle state updates
        if result.update:
            if isinstance(result.update, AgentState):
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

    elif isinstance(result, list):
        # Handle list of items (convert each to message)
        for item in result:
            await create_and_add_message(item)
    else:
        # Handle single items (str, dict, model_dump-capable, or other)
        await create_and_add_message(result)

    # Add new messages to the main list and state context
    if new_messages:
        messages.extend(new_messages)
        state.context = add_messages(state.context, new_messages)

    return state, messages, next_node


async def check_and_handle_interrupt(
    interrupt_before: list[str],
    interrupt_after: list[str],
    current_node: str,
    interrupt_type: str,
    state: AgentState,
    config: dict[str, Any],
    _sync_data: Callable,
) -> bool:
    """Check for interrupts and save state if needed. Returns True if interrupted."""
    interrupt_nodes = interrupt_before if interrupt_type == "before" else interrupt_after

    if current_node in interrupt_nodes:
        status = (
            ExecutionStatus.INTERRUPTED_BEFORE
            if interrupt_type == "before"
            else ExecutionStatus.INTERRUPTED_AFTER
        )
        state.set_interrupt(
            current_node,
            f"interrupt_{interrupt_type}: {current_node}",
            status,
        )
        # Save state and interrupt
        await _sync_data(state, config, [])
        logger.debug("Node '%s' interrupted", current_node)
        return True

    logger.debug(
        "No interrupts found for node '%s', continuing execution",
        current_node,
    )
    return False


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
) -> bool:
    """Sync the current state and messages to the checkpointer.

    ``last_save`` records the save this call starts, for the run's cancel cleanup.
    """
    checkpointer = fresh(checkpointer)
    context_manager = fresh(context_manager)
    callback_mgr = fresh(callback_mgr)
    is_context_trimmed = False

    new_state = copy.deepcopy(state)
    # if context manager is available then utilize it
    if context_manager and trim:
        new_state = await context_manager.atrim_context(state)
        is_context_trimmed = True

    # Fire on_checkpoint hook before persisting
    if callback_mgr and callback_mgr._lifecycle_hooks:
        lifecycle_context = GraphLifecycleContext(config=config)
        new_state, messages = await callback_mgr.fire_on_checkpoint(
            lifecycle_context,
            state=new_state,
            messages=messages,
            is_context_trimmed=is_context_trimmed,
        )

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

    return is_context_trimmed


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
