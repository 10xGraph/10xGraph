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
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from injectq import Inject

from tenxgraph.core.authz import public_config
from tenxgraph.core.exceptions import StoredNodeMissingError
from tenxgraph.core.state import AgentState, ExecutionStatus
from tenxgraph.core.state.execution_state import StopRequestStatus
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import RemoteToolCallBlock, ToolResultBlock
from tenxgraph.core.state.reducers import add_messages
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import listening, publish_event, publish_lifecycle
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.utils import (
    END,
    START,
    metrics,
)
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleContext
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import (
    INTERRUPT_REASON,
    RESUME_INTERRUPT_ID_KEY,
    RESUME_KEY,
    Interrupt,
    clear_resume_values,
    pause_at,
    pending_interrupt,
    record_resume,
)
from tenxgraph.utils.temp_perf import done as _perf_done  # TEMP-PERF
from tenxgraph.utils.temp_perf import now as _perf_now  # TEMP-PERF

from .run_registry import current_run
from .tool_dispatch import tool_call_name
from .utils import (
    CANCELLED_REASON,
    CRASHED_REASON,
    FINALIZE_TIMEOUT_SECONDS,
    STOP_REASON,
    LastSave,
    _detached_tasks,  # noqa: F401  (re-exported: cleanup tasks still running)
    clean_unstorable_messages,
    detach_task,
    get_next_node,
    sync_data,
    wait_detached,
)


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("tenxgraph.graph")


async def fire_interrupt(
    callback_mgr: CallbackManager | None,
    config: dict[str, Any],
    node: str,
    interrupt_type: str,
    state: AgentState,
) -> None:
    """Run the ``on_interrupt`` hooks, then publish the run-end event of the pause.

    ``run.stopped`` for a stop request, else ``run.interrupted`` (``before``, ``after``,
    ``interrupt``, ``remote_tool``), with the paused state and, for ``interrupt()``, the
    request (id and value).
    """
    if callback_mgr and callback_mgr._lifecycle_hooks:
        await callback_mgr.fire_on_interrupt(
            GraphLifecycleContext(config=public_config(config)),
            interrupted_node=node,
            interrupt_type=interrupt_type,
            state=state,
        )
    if not listening():
        return
    stopped = interrupt_type == "stop"
    data: dict[str, Any] = {
        "interrupted_node": node,
        "interrupt_type": interrupt_type,
        "state": state,
    }
    request = pending_interrupt(state) if interrupt_type == "interrupt" else None
    if request is not None:
        data["interrupt"] = request.model_dump(mode="json")
    if stopped:
        data["reason"] = STOP_REASON
    publish_event(
        EventModel.default(
            config,
            data=data,
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.INTERRUPTED,
            content_type=[ContentType.STATE],
            # A stop is about the run; only an interrupt belongs to a node.
            node_name="" if stopped else node,
            extra={
                "event_name": "run.stopped" if stopped else "run.interrupted",
                "interrupt_type": interrupt_type,
            },
        )
    )


async def check_interrupted[StateT: AgentState](
    state: StateT,
    input_data: dict[str, Any],
    config: dict[str, Any],
    callback_mgr: CallbackManager = Inject[CallbackManager],
) -> tuple[StateT, dict[str, Any]]:
    callback_mgr = fresh(callback_mgr)
    from tenxgraph.core.exceptions import InterruptPendingError, NothingToResumeError

    request = pending_interrupt(state)
    if request is not None:
        # Paused by interrupt(): the retried node needs the answer, and a plain new message
        # cannot stand in for it.
        if RESUME_KEY not in input_data:
            raise InterruptPendingError(
                f"Thread is paused at interrupt '{request.id}' in node '{request.node}'. "
                f"Resume it with {{'{RESUME_KEY}': value}}."
            )
        record_resume(state, input_data[RESUME_KEY], input_data.get(RESUME_INTERRUPT_ID_KEY))
    elif RESUME_KEY in input_data:
        raise NothingToResumeError()

    if state.is_interrupted():
        logger.info(
            "Resuming from interrupted state at node '%s'", state.execution_meta.current_node
        )

        # Fire on_resume hook before clearing the interrupt
        resumed_node = state.execution_meta.interrupted_node or ""
        if callback_mgr and callback_mgr._lifecycle_hooks:
            context = GraphLifecycleContext(config=public_config(config))
            modified = await callback_mgr.fire_on_resume(
                context,
                resumed_node=resumed_node,
                state=state,
                resume_data=input_data,
            )
            if modified is not None and modified is not state:
                state = modified  # type: ignore[assignment]
        # A span event on the run's trace; event consumers see the resume in ``run.started``.
        publish_lifecycle(
            config,
            "resume",
            {"resumed_node": resumed_node},
            {"resumed_node": resumed_node},
            tag="observability",
        )

        # Paused for a client-side tool: the resumed run continues after that node.
        if state.execution_meta.interrupt_reason == REMOTE_TOOL_REASON:
            config[RESUME_AFTER_NODE_KEY] = state.execution_meta.interrupted_node

        # Save the interrupted node info before clearing so we don't re-interrupt
        config["_skip_interrupt_at"] = {
            "node": state.execution_meta.interrupted_node,
            "status": state.execution_meta.status,
        }
        # This is a resume case - clear interrupt and merge input data
        if input_data:
            config["resume_data"] = input_data
            logger.debug("Added resume data with %d keys", len(input_data))
        state.clear_interrupt()
    elif not input_data.get("messages") and not state.context:
        # This is a fresh execution - validate input data
        error_msg = "Input data must contain 'messages' for new execution."
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif state.execution_meta.status == ExecutionStatus.COMPLETED:
        # Previous execution completed - reset to entry point for new execution
        logger.info(
            "Previous execution completed. Resetting to entry point for new execution "
            "with %d messages",
            len(input_data.get("messages", [])),
        )
        # Reset execution metadata for fresh start
        state.execution_meta.current_node = START
        state.execution_meta.step = 0
        state.execution_meta.status = ExecutionStatus.RUNNING
        state.execution_meta.interrupted_node = None
        state.execution_meta.interrupt_reason = None
        state.execution_meta.interrupt_data = None
        # A new turn: loop limits (max_rounds, max_iterations, max_attempts) start over.
        state.execution_meta.reset_turn_data()
    else:
        # Fresh execution, state is already at START
        logger.info(
            "Starting fresh execution with %d messages", len(input_data.get("messages", []))
        )
        # A new message after a run that errored is a new turn too, so a failed turn does
        # not use up the next one's loop budget. A stopped run is a pause (handled above).
        if state.execution_meta.status == ExecutionStatus.ERROR and input_data.get("messages"):
            state.execution_meta.reset_turn_data()

    return state, config


async def check_and_handle_interrupt[StateT: AgentState](
    current_node: str,
    interrupt_type: str,
    state: StateT,
    config: dict[str, Any],
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    callback_mgr: CallbackManager = Inject[CallbackManager],
    messages: list[Message] | None = None,
    last_save: LastSave | None = None,
) -> bool:
    """Check for interrupts and save state if needed. Returns True if interrupted.

    ``messages`` are the run's messages not yet saved; the pause saves them with the state.
    ``last_save`` records the save, so a cancel cleanup waits for it instead of racing it.
    """
    callback_mgr = fresh(callback_mgr)
    interrupt_nodes: list[str] = (
        interrupt_before if interrupt_type == "before" else interrupt_after
    ) or []

    # Check if we just resumed from an interrupt at this node with this type
    skip_info = config.get("_skip_interrupt_at", {})
    if skip_info.get("node") == current_node:
        expected_status = (
            ExecutionStatus.INTERRUPTED_BEFORE
            if interrupt_type == "before"
            else ExecutionStatus.INTERRUPTED_AFTER
        )
        if skip_info.get("status") == expected_status:
            logger.debug(
                "Skipping %s interrupt check for node '%s' - just resumed from it",
                interrupt_type,
                current_node,
            )
            # Clear the flag after using it once
            config.pop("_skip_interrupt_at", None)
            return False

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

        await fire_interrupt(callback_mgr, config, current_node, interrupt_type, state)

        # Save state and interrupt
        await sync_data(
            state=state,
            config=config,
            messages=messages or [],
            trim=True,
            last_save=last_save,
        )
        logger.debug("Node '%s' interrupted", current_node)
        return True

    logger.debug(
        "No interrupts found for node '%s', continuing execution",
        current_node,
    )
    return False


# ``execution_meta.interrupt_reason`` while a tool call waits for the client to run it.
REMOTE_TOOL_REASON = "remote_tool_call"

# Run-config key telling the loop to continue after the node that paused for a client tool.
RESUME_AFTER_NODE_KEY = "_resume_after_node"


def resolve_stored_node(
    stored: str,
    nodes: dict[str, Any],
    node_aliases: dict[str, str] | None,
    thread_id: Any,
) -> str:
    """Map the node a thread was stored at onto a node the deployed graph has.

    A node the graph still has is returned unchanged. A missing one is followed through
    ``node_aliases`` (set at ``compile()``); with no usable alias the run stops with
    ``StoredNodeMissingError`` instead of a bare ``KeyError`` deep in the loop.
    """
    if stored in nodes:
        return stored
    seen = {stored}
    name = stored
    while name in (node_aliases or {}):
        name = (node_aliases or {})[name]
        if name in nodes:
            logger.info("Thread '%s': stored node '%s' resumes at '%s'", thread_id, stored, name)
            return name
        if name in seen:
            break
        seen.add(name)
    raise StoredNodeMissingError(stored, thread_id, available_nodes=list(nodes))


def is_remote_tool_marker(item: Any) -> bool:
    """Whether ``item`` is the placeholder a ToolNode returns for a client-executed call."""
    return isinstance(item, Message) and any(
        isinstance(block, RemoteToolCallBlock) for block in item.content or []
    )


def node_after(
    current_node: str,
    explicit_next: str | None,
    state: AgentState,
    edges: list,
) -> str:
    """The node that follows ``current_node``: the one a Command chose, else the graph's."""
    return explicit_next or get_next_node(current_node, state, edges)


def skip_answered_tool_node(
    node: Any,
    current_node: str,
    state: AgentState,
    next_node: Callable[[str], str],
) -> str:
    """Move past a ToolNode whose tool calls are all answered already.

    After a cancel or crash was repaired inside a ToolNode, every call of the saved tool
    message has a result (from the ledger, or a synthetic one). The node has nothing left to
    run, so a call with no new input resumes at the node after it.
    """
    from tenxgraph.core.graph.tool_node import ToolNode

    if not isinstance(getattr(node, "func", None), ToolNode):
        return current_node
    last = state.context[-1] if state.context else None
    if last is None or (last.role == "assistant" and last.tools_calls):
        return current_node
    following = next_node(current_node)
    logger.info(
        "Tool node '%s' has no unanswered tool calls; resuming at '%s'", current_node, following
    )
    return following


async def interrupt_graph[StateT: AgentState](
    current_node: str,
    state: StateT,
    config: dict[str, Any],
    callback_mgr: CallbackManager = Inject[CallbackManager],
    messages: list[Message] | None = None,
    last_save: LastSave | None = None,
) -> bool:
    """Pause after ``current_node`` until the client sends the results of its tool calls.

    ``current_node`` stays put: which node comes next can depend on those results, so the
    resumed run works it out once they are in the context (see ``RESUME_AFTER_NODE_KEY``).
    ``last_save`` records the save (see :func:`check_and_handle_interrupt`).
    """
    callback_mgr = fresh(callback_mgr)
    status = ExecutionStatus.INTERRUPTED_AFTER
    state.set_interrupt(
        current_node,
        REMOTE_TOOL_REASON,
        status,
    )

    await fire_interrupt(callback_mgr, config, current_node, "remote_tool", state)

    # Save state and interrupt
    await sync_data(
        state=state,
        config=config,
        messages=messages or [],
        trim=False,
        last_save=last_save,
    )
    logger.debug("Node '%s' interrupted", current_node)
    return True


async def pause_for_interrupt[StateT: AgentState](
    current_node: str,
    state: StateT,
    config: dict[str, Any],
    request: Interrupt,
    callback_mgr: CallbackManager = Inject[CallbackManager],
    messages: list[Message] | None = None,
    last_save: LastSave | None = None,
) -> None:
    """Save ``state`` paused before ``current_node`` because it called ``interrupt()``.

    Resuming re-runs the node, and ``interrupt()`` then returns the resume value.
    ``last_save`` records the save (see :func:`check_and_handle_interrupt`).
    """
    callback_mgr = fresh(callback_mgr)
    pause_at(state, request)

    await fire_interrupt(callback_mgr, config, current_node, "interrupt", state)

    await sync_data(
        state=state,
        config=config,
        messages=messages or [],
        trim=False,
        last_save=last_save,
    )
    logger.info("Node '%s' paused at interrupt '%s'", current_node, request.id)


async def check_stop_requested[StateT: AgentState](
    state: StateT,
    current_node: str,
    messages: list[Message],
    config: dict[str, Any],
    callback_mgr: CallbackManager = Inject[CallbackManager],
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],
    resume_after: Callable[[], str] | None = None,
    last_save: LastSave | None = None,
    remote: bool = True,
) -> bool:
    """Check if a stop has been requested externally.

    ``resume_after`` is passed when ``current_node`` has just finished: it returns the node
    that follows it. The stop is then saved at that node, so a call with no new input
    continues there and does not run the finished node again. If nothing follows (the graph
    is done) there is nothing left to stop: the request is dropped and the run completes.

    ``last_save`` records the stop save (see :func:`check_and_handle_interrupt`). The stop is
    reported once, by ``run.stopped`` (GRAPH_EXECUTION INTERRUPTED, ``node_name`` = the
    stopped node).

    ``remote=False`` checks only the in-process flags and skips the checkpointer read. The
    engine passes it after a node: the remote flag is read once per step, before the node.
    """
    callback_mgr = fresh(callback_mgr)
    checkpointer = fresh(checkpointer)
    # Cheapest first: the live state and the run's in-process flag (set by astop and aclose)
    # cost nothing. A stop from another process is one small cache key the run loop never
    # writes, so its own state-cache writes cannot swallow it. The full state is not
    # reloaded here: it is read only when a run starts or resumes.
    stop_requested = state.is_stopped_requested()
    run = current_run.get()
    if run is not None and run.stop_requested:
        stop_requested = True
    if not stop_requested and remote and checkpointer:
        try:
            _tp0 = _perf_now()  # TEMP-PERF
            stop_requested = await checkpointer.ais_stop_requested(config)
            _perf_done("core_stop_check", _tp0)  # TEMP-PERF
        except Exception as e:  # an unreachable cache must not fail the run
            logger.warning("Stop check failed (%r); relying on the in-process stop flag", e)

    # Check if a stop was requested externally (e.g., frontend)
    if stop_requested:
        resume_at = resume_after() if resume_after is not None else None
        if resume_at == END:
            if checkpointer:
                await checkpointer.aclear_stop_request(config)
            return False
        logger.info(
            "Stop requested for thread '%s' at node '%s'",
            config.get("thread_id"),
            current_node,
        )
        # Consume the request so it cannot leak into the next run on this thread.
        if checkpointer:
            await checkpointer.aclear_stop_request(config)

        # The request is consumed. Saving it as STOPPED (not STOP_REQUESTED) keeps the next
        # run on this thread from reading it back from the cache and stopping at once.
        state.execution_meta.stop_current_execution = StopRequestStatus.STOPPED
        if resume_at is not None:
            # The node finished: its interrupt() answers are spent and the run moves on.
            clear_resume_values(state)
            state.set_current_node(resume_at)
            state.advance_step()
        state.set_interrupt(
            current_node,
            STOP_REASON,
            ExecutionStatus.INTERRUPTED_AFTER,
            data={"source": "stop", "info": "requested via is_stopped_requested"},
        )

        await fire_interrupt(callback_mgr, config, current_node, "stop", state)

        await sync_data(
            state=state, config=config, messages=messages, trim=True, last_save=last_save
        )
        return True
    return False


# Text of the result given to a tool call that never finished.
UNFINISHED_TOOL_CALL_TEXT = "tool call did not finish: the run was interrupted"


async def _recorded_tool_message(
    checkpointer: BaseCheckpointer | None,
    config: dict[str, Any],
    tool_call_id: str,
) -> Message | None:
    """The tool message recorded in the ledger for ``tool_call_id``, if one exists."""
    if not checkpointer:
        return None
    try:
        recorded = await checkpointer.aget_tool_result(config, tool_call_id)
        if isinstance(recorded, dict) and recorded.get("__kind__") == "message":
            return Message.model_validate(recorded["value"])
    except Exception as e:  # a ledger read must never block the repair
        logger.warning("Could not read tool ledger for call %s (%s)", tool_call_id, e)
    return None


async def close_unanswered_tool_calls(
    state: AgentState,
    config: dict[str, Any],
    messages: list[Message],
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],
) -> list[Message]:
    """Give every unanswered tool call of the last tool-calling assistant message a result.

    A call gets the result recorded in the tool ledger if there is one, else an error result
    saying it did not finish. Providers reject a history that holds a tool call with no
    answer. Calls waiting on the client (a remote tool pause) or on ``interrupt()`` are a
    legitimate pause and are left alone. The closing messages are added to ``state.context``
    and to ``messages``, and returned.
    """
    checkpointer = fresh(checkpointer)
    meta = state.execution_meta
    if meta.status != ExecutionStatus.RUNNING and meta.interrupt_reason in (
        REMOTE_TOOL_REASON,
        INTERRUPT_REASON,
    ):
        return []

    last_idx = next(
        (i for i in range(len(state.context) - 1, -1, -1) if state.context[i].tools_calls),
        None,
    )
    if last_idx is None or state.context[last_idx].role != "assistant":
        return []

    answered = {
        block.call_id
        for msg in state.context[last_idx + 1 :]
        for block in msg.content or []
        if isinstance(block, ToolResultBlock)
    }
    closing: list[Message] = []
    for call in state.context[last_idx].tools_calls or []:
        call_id = call.get("id")
        if not call_id or call_id in answered:
            continue
        message = await _recorded_tool_message(checkpointer, config, call_id)
        if message is None:
            message = Message.tool_message(
                content=[
                    ToolResultBlock(
                        call_id=call_id,
                        name=tool_call_name(call) or None,
                        output=UNFINISHED_TOOL_CALL_TEXT,
                        is_error=True,
                        status="failed",
                    )
                ]
            )
        closing.append(message)

    if closing:
        state.context = add_messages(state.context, closing)
        messages.extend(closing)
        logger.info("Closed %d unanswered tool call(s)", len(closing))
    return closing


async def finalize_cancelled_run(
    state: AgentState,
    config: dict[str, Any],
    messages: list[Message],
    current_node: str,
    reason: str = CANCELLED_REASON,
    last_save: LastSave | None = None,
) -> None:
    """Save a cancelled run as INTERRUPTED so the thread stays usable.

    Closes unanswered tool calls, marks the state INTERRUPTED with ``reason``, syncs it to the
    durable store and then publishes ``run.stopped``, in one detached task that finishes
    even if this coroutine keeps being cancelled (a level-triggered anyio cancel scope). This
    coroutine waits for that task up to ``FINALIZE_TIMEOUT_SECONDS``, also while cancelled, so
    what runs after it (the run-marker release, the caller's thread lock) sees the save done. If
    the save fails or is still running the state cache is left as it is: it holds the newest
    state, including the user's message. This never raises; the caller re-raises the cancel.

    ``last_save`` is the run's own latest save. One still running is waited for first, so two
    saves of the run never race. When it saved the COMPLETED state, the run is over and stays
    completed: nothing is marked INTERRUPTED.

    The saved position stays at ``current_node``. What the next call does with it is decided on
    load (decision D1): a new user message starts a new turn, a call with no new input resumes
    at this node.
    """
    # A hard cancel by ``aclose()`` is recorded as a shutdown, not a plain cancel.
    run = current_run.get()
    if run is not None and run.cancel_reason and reason == CANCELLED_REASON:
        reason = run.cancel_reason
    thread_id = config.get("thread_id")
    prior = last_save.task if last_save is not None else None

    async def _finalize() -> None:
        if prior is not None and not prior.done():
            await asyncio.wait({prior})
        if last_save is not None and last_save.finished_completed():
            logger.info("Run on thread '%s' was cancelled after its final save", thread_id)
            return
        try:
            await close_unanswered_tool_calls(state, config, messages)
            state.set_interrupt(
                current_node,
                reason,
                ExecutionStatus.INTERRUPTED_AFTER,
                data={"source": "cancel", "info": "run was cancelled"},
            )
            await sync_data(state=state, config=config, messages=messages, trim=True)
        finally:
            # After the save, so the event carries the saved state (tool calls closed,
            # interrupt set).
            _publish_cancel_event(state, config, current_node, reason)

    # All the work runs in one task of its own. A cancel scope that stays cancelled (anyio,
    # as used by Starlette on a client disconnect) re-raises CancelledError at every await of
    # this coroutine, so nothing that must finish can be awaited here directly.
    task = detach_task(_finalize())

    def _log_failure(done: asyncio.Future) -> None:
        if not done.cancelled() and done.exception() is not None:
            logger.error(
                "Could not save cancelled run for thread '%s' (%r); the state cache is kept",
                thread_id,
                done.exception(),
            )

    task.add_done_callback(_log_failure)
    try:
        finished = await wait_detached(task)
    except asyncio.CancelledError:
        # Cancelled again while waiting; the wait is over and the caller re-raises its cancel.
        finished = task.done()
    if not finished:
        logger.error(
            "Saving cancelled run for thread '%s' did not finish within %ss; the save is still "
            "running in the background and the state cache is kept",
            thread_id,
            FINALIZE_TIMEOUT_SECONDS,
        )


def _publish_cancel_event(
    state: AgentState, config: dict[str, Any], current_node: str, reason: str
) -> None:
    """Publish ``run.stopped`` for a cancelled run, with its reason and saved state."""
    try:
        publish_event(
            EventModel.default(
                config,
                data={"state": state, "reason": reason, "interrupted_node": current_node},
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.INTERRUPTED,
                content_type=[ContentType.STATE],
                extra={"event_name": "run.stopped", "interrupt_type": "cancel"},
            )
        )
    except Exception as e:  # publishing must never mask the cancel
        logger.warning("Could not publish cancel event (%s)", e)


async def _unsaved_turn_messages(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: BaseCheckpointer | None,
) -> list[Message]:
    """Messages of the latest turn (from the last user message on) the durable store lacks.

    A crashed run may have died before its messages reached the durable store: the state cache
    is written before every node, the durable store only at step boundaries.
    """
    start = next(
        (i for i in range(len(state.context) - 1, -1, -1) if state.context[i].role == "user"),
        None,
    )
    if start is None:
        return []
    if not checkpointer:
        return []
    unsaved: list[Message] = []
    for msg in state.context[start:]:
        try:
            await checkpointer.aget_message(config, msg.message_id)
        except Exception:  # not stored (or unreadable): store it; stores upsert by id
            unsaved.append(msg)
    return unsaved


async def recover_crashed_run(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],
) -> None:
    """Repair a state a dead run left RUNNING, from that state itself (decision D2).

    The state is the cached one when a cache exists, so it still holds the crashed turn's user
    message. Unanswered tool calls are closed, the state is marked INTERRUPTED with reason
    ``crashed`` and saved durably (store and cache), together with the turn's messages the store
    does not have yet. The caller decides what the next call does with it (decision D1).
    Never raises: if the save fails the repaired state is still returned to the caller, whose
    own saves persist it. A turn holding NUL or a lone surrogate (from a model or tool reply, or
    stored by an older version) can never be saved, so it is cleaned first.
    """
    checkpointer = fresh(checkpointer)
    meta = state.execution_meta
    clean_unstorable_messages(state.context, config)
    dead_run = meta.run_id
    logger.warning(
        "Run '%s' on thread '%s' ended without finishing; repairing the thread",
        dead_run,
        config.get("thread_id"),
    )
    try:
        unsaved = await _unsaved_turn_messages(state, config, checkpointer)
        closing: list[Message] = []
        await close_unanswered_tool_calls(state, config, closing)
        state.set_interrupt(
            meta.current_node,
            CRASHED_REASON,
            ExecutionStatus.INTERRUPTED_AFTER,
            data={"source": "recovery", "info": "the previous run ended without finishing"},
        )
        await sync_data(state=state, config=config, messages=[*unsaved, *closing], trim=False)
    except Exception as e:
        logger.error("Could not save the repaired thread '%s' (%r)", config.get("thread_id"), e)
    try:
        metrics.counter("tenxgraph.run.recovered").inc()
        # Published before the new run's ``run.started`` (the repair happens while the
        # thread loads); tracing adds it to that run's span once it starts.
        publish_event(
            EventModel.default(
                config,
                data={"dead_run_id": dead_run},
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.INTERRUPTED,
                content_type=[ContentType.STATE],
                extra={"event_name": "run.recovered"},
            )
        )
    except Exception as e:  # publishing must never block the repair
        logger.warning("Could not publish recovery event (%s)", e)
