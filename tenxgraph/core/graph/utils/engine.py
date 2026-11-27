"""One graph loop for invoke and stream.

``GraphEngine.run`` is an async generator that runs a whole graph run: the outer
GRAPH_EXECUTION events, the step loop and every exit (done, pause, stop, error, cancel).
With ``streaming=True`` it yields ``StreamChunk``s as the run goes; with ``streaming=False``
it builds no chunk and an agent node makes a non-streaming LLM call. Either way the last
item is a ``RunResult``. ``ainvoke`` drains it, ``astream`` filters the chunks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from injectq import Inject, inject

from tenxgraph.core.authz import public_config
from tenxgraph.core.exceptions import GraphStopRequested
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.reducers import add_messages
from tenxgraph.core.state.stream_chunks import StreamChunk, StreamEvent
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.storage.checkpointer import BaseCheckpointer
from tenxgraph.utils import END, ResponseGranularity
from tenxgraph.utils.callbacks import CallbackManager, GraphLifecycleContext
from tenxgraph.utils.injection import fresh
from tenxgraph.utils.interrupt import GraphInterrupt, node_scope, resume_values
from tenxgraph.utils.logging import log_context_scope
from tenxgraph.utils.temp_perf import done as _perf_done  # TEMP-PERF
from tenxgraph.utils.temp_perf import now as _perf_now  # TEMP-PERF

from .error_text import client_error_text
from .guards import execute_with_guards, node_guards, node_metrics, stream_with_guards
from .handler_mixins import GraphRunMixin, mirror_last_message
from .handler_utils import (
    check_stop_requested,
    finalize_cancelled_run,
    interrupt_graph,
    node_after,
)
from .node_runner import NodeResult
from .utils import LastSave, calculate_token_usage


logger = logging.getLogger("tenxgraph.graph")


@dataclass(slots=True)
class RunResult:
    """The end of a run: final state, the run's messages (client-tool placeholders last)."""

    state: Any
    messages: list[Message] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    time_taken: float = 0.0


async def _last(steps: AsyncIterator[Any]) -> Any:
    """Drain ``steps`` and return its last item (a node's ``NodeResult`` under invoke)."""
    item = None
    async for item in steps:  # noqa: B007
        pass
    return item


class GraphEngine(GraphRunMixin):
    """Runs a compiled graph; one loop for invoke and stream."""

    @inject
    def __init__(
        self,
        nodes: dict[str, Any],
        edges: list[Any],
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        get_node_factory: Callable[[str], Any] | None = None,
        node_aliases: dict[str, str] | None = None,
        callback_mgr: CallbackManager = Inject[CallbackManager],  # type: ignore[assignment]
    ):
        self.nodes = nodes
        self.edges = edges
        # Reads the live dict, so override_node after compile is seen.
        self._get_node = get_node_factory if get_node_factory else lambda x: nodes[x]
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        self.callback_mgr = fresh(callback_mgr)
        self.node_aliases = dict(node_aliases or {})

    async def run(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: AgentState,
        *,
        streaming: bool,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> AsyncIterator[StreamChunk | RunResult]:
        """Run the graph: chunks while it runs (streaming only), the ``RunResult`` last.

        ``response_granularity`` shapes only the caller's response; events do not use it.
        """
        config = config or {}
        # Every log line of this run carries its ids; the earlier ids come back at the end.
        with log_context_scope(config):
            async with aclosing(self._run(input_data, config, default_state, streaming)) as items:
                async for item in items:
                    yield item

    async def _run(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        default_state: AgentState,
        streaming: bool,
    ) -> AsyncGenerator[StreamChunk | RunResult]:
        """The run events: ``run.started``, then ``run.completed`` or ``run.failed``.

        A paused or stopped run ends with ``run.interrupted`` or ``run.stopped`` from where it
        paused (``fire_interrupt``) and a cancelled one with ``run.stopped``
        (``finalize_cancelled_run``), so no ``run.completed`` follows them.
        """
        _tp0 = _perf_now()  # TEMP-PERF
        logger.info("Starting graph run, streaming=%s", streaming)
        start_time = time.time()
        input_data = input_data or {}
        state = await self._load_state(input_data, config, default_state)

        # The state is for observability (the run's input on the trace); event publishers
        # get it only on the run-end events.
        event = EventModel.default(
            config,
            data={"state": state},
            event=Event.GRAPH_EXECUTION,
            content_type=[ContentType.STATE],
            extra={
                "event_name": "run.started",
                "current_node": state.execution_meta.current_node,
                "step": state.execution_meta.step,
                "is_interrupted": state.is_interrupted(),
            },
        )
        event.event_type = EventType.START
        publish_event(event)

        result = RunResult(state)
        try:
            # Apply a resume, then fire on_graph_start before execution begins. Inside the
            # try: a refused resume still ends the run with run.failed.
            state, config = await self._start_run(state, input_data, config)
            result.state = state
            _perf_done("core_engine_setup", _tp0)  # TEMP-PERF
            async with aclosing(self._loop(result, input_data, config, streaming)) as chunks:
                async for chunk in chunks:
                    yield chunk
            result.token_usage = calculate_token_usage(result.messages)
            result.time_taken = time.time() - start_time
            logger.info("Graph run finished in %.2f seconds", result.time_taken)

            state = result.state
            if not state.is_interrupted():
                _set_run_end(event, EventType.END, "run.completed", result)
                event.metadata.update({"time_taken": result.time_taken, **result.token_usage})
                event.data["messages"] = result.messages
                mirror_last_message(event, result.messages)
                event.content_type = [ContentType.STATE, ContentType.MESSAGE]
                publish_event(event)
            yield result
        except Exception as e:
            logger.exception("Graph execution failed: %s", e)
            _set_run_end(event, EventType.ERROR, "run.failed", result)
            event.metadata["time_taken"] = time.time() - start_time
            event.data["error"] = str(e)
            event.content_type = [ContentType.ERROR, ContentType.STATE]
            publish_event(event)
            raise

    async def _run_node(
        self,
        name: str,
        config: dict[str, Any],
        state: AgentState,
        streaming: bool,
        checkpointer: BaseCheckpointer = Inject[BaseCheckpointer],  # type: ignore[assignment]
        callback_mgr: CallbackManager = Inject[CallbackManager],  # type: ignore[assignment]
    ) -> AsyncGenerator[Any]:
        """Run one node under its deadline, cancellable by a stop (in-process or remote).

        Under stream the node's generator is driven by a task core owns
        (``stream_with_guards``); under invoke it is drained in one task
        (``execute_with_guards``), with no queue between items.
        """
        guards = node_guards(name, config, fresh(checkpointer))
        steps = self._get_node(name).runner.run(
            config, state, fresh(callback_mgr), streaming=streaming
        )
        with node_metrics(name):
            if streaming:
                async with aclosing(stream_with_guards(steps, **guards)) as items:
                    async for item in items:
                        yield item
            else:
                yield await execute_with_guards(_last(steps), **guards)

    async def _loop(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        run: RunResult,
        input_data: dict[str, Any],
        config: dict[str, Any],
        streaming: bool,
    ) -> AsyncGenerator[StreamChunk]:
        """The step loop. Keeps ``run.state`` and ``run.messages`` current at every exit.

        Chunks are built only under ``if streaming``, so invoke builds none.
        """
        state = run.state
        messages = run.messages
        seen: set[Any] = set()
        persisted_upto = 0  # how many of ``messages`` a per-step save has written
        last_save = LastSave()  # the latest durable save, for the cancel cleanup
        max_steps = config.get("recursion_limit", 25)

        # The run's input is what this call brought; a resume with no input brings nothing.
        for message in input_data.get("messages") or []:
            if message.message_id not in seen:
                seen.add(message.message_id)
                messages.append(message)
                if streaming:
                    meta = {"status": "invoking_graph", "reason": "initial human message"}
                    yield _chunk(config, StreamEvent.MESSAGE, message=message, metadata=meta)

        current_node, step = self._start_node(state, config)
        lifecycle_context = GraphLifecycleContext(config=public_config(config))

        def note(
            status: str, kind: StreamEvent = StreamEvent.UPDATES, **fields: Any
        ) -> StreamChunk:
            """A chunk about the current step; ``state`` is attached when given."""
            chunk_state = fields.pop("state", None)
            data = {"status": status, "node": current_node, "step": step, "max_steps": max_steps}
            return _chunk(config, kind, state=chunk_state, data={**data, **fields})

        def state_chunk() -> StreamChunk:
            meta = {"node": current_node, "step": step}
            return _chunk(config, StreamEvent.STATE, state=state, metadata=meta)

        def merge(result: NodeResult) -> list[Message]:
            """Add the node's new messages (deduped) to the run and the context."""
            new = [m for m in result.messages if not m.delta and m.message_id not in seen]
            seen.update(m.message_id for m in new)
            messages.extend(new)
            state.context = add_messages(state.context, new)
            return new

        # The node's result from the moment it arrives until ``_advance`` moved past the node:
        # a cancel in that window must not save the run at the node that already finished.
        done: NodeResult | None = None
        stopped = "Graph execution stopped by request"
        if streaming:
            yield note("invoking_graph")
        try:
            while current_node != END and step < max_steps:
                # The one remote stop read per step. It runs after the previous step's save, so
                # a stop that arrived while that node ran or while it was saved is seen here.
                if await check_stop_requested(
                    state, current_node, messages, config, last_save=last_save
                ):
                    if streaming:
                        yield note("invoking_node", reason=stopped)
                    return

                await self._begin_step(state, config, step, current_node)
                if await self._interrupt_before(
                    current_node, state, config, messages[persisted_upto:], last_save
                ):
                    if streaming:
                        reason = "Graph execution interrupted before node execution"
                        yield note("invoking_node", reason=reason, pause="before")
                    return

                has_hooks, old_state, node_event = self._start_node_event(
                    state, config, step, current_node
                )
                if streaming:
                    yield note("node_invoked")

                result = NodeResult(state)  # replaced by the runner's last item
                try:
                    with node_scope(config, current_node, resume_values(state)):
                        async with aclosing(
                            self._run_node(current_node, config, state, streaming)
                        ) as items:
                            async for item in items:
                                if isinstance(item, NodeResult):
                                    result = done = item
                                    state = run.state = item.state
                                else:
                                    yield item
                except GraphInterrupt as paused:
                    await self._pause_at_interrupt(
                        paused,
                        current_node,
                        state,
                        config,
                        messages[persisted_upto:],
                        last_save,
                    )
                    if streaming:
                        yield note(
                            "interrupted",
                            pause="interrupt",
                            reason="Graph execution paused by interrupt()",
                            interrupt=paused.interrupt.model_dump(mode="json"),
                        )
                    return
                except GraphStopRequested:
                    await self._stop_mid_node(current_node, state, config, messages, last_save)
                    if streaming:
                        yield note("invoking_node", reason=stopped)
                    return

                state = run.state = result.state
                next_node = result.next_node
                if streaming:
                    yield state_chunk()
                # Merge this node's output only, so the next node sees it.
                new = merge(result)
                self._end_node_event(node_event, next_node, current_node, new)
                if streaming and messages:
                    yield state_chunk()

                state = run.state = await self._after_node(
                    has_hooks, lifecycle_context, current_node, old_state, state, config, step
                )

                if result.remote_calls:
                    # The client runs these calls; the run resumes after this node once their
                    # results arrive. The placeholders go back to the caller, not the context.
                    # A cancel from here on keeps the position at this node.
                    done = None
                    await interrupt_graph(
                        current_node,
                        state,
                        config,
                        messages=messages[persisted_upto:],
                        last_save=last_save,
                    )
                    messages.extend(result.remote_calls)
                    if streaming:
                        reason = "Remote tool call - graph interrupted"
                        yield note("node_invoked", reason=reason, pause="remote_tool")
                    return

                # The node finished, so a stop now is saved at the node that follows it. Only the
                # in-process flags: the remote one is read at the top of the next step.
                if await check_stop_requested(
                    state,
                    current_node,
                    messages,
                    config,
                    resume_after=partial(node_after, current_node, next_node, state, self.edges),
                    last_save=last_save,
                    remote=False,
                ):
                    if streaming:
                        yield note("node_invoked", reason=stopped)
                    return

                paused_after, next_node = await self._interrupt_after(
                    current_node,
                    next_node,
                    state,
                    config,
                    messages[persisted_upto:],
                    last_save,
                )
                if paused_after and streaming:
                    reason = "Graph execution interrupted after node execution"
                    yield note("node_invoked", reason=reason, pause="after")

                current_node, step, persisted_upto = await self._advance(
                    current_node,
                    next_node,
                    state,
                    config,
                    step,
                    messages,
                    persisted_upto,
                    last_save,
                )
                done = None
                if streaming:
                    yield note("node_invoked", state=state)
                if paused_after:
                    return

                if step >= max_steps and current_node != END:
                    error = await self._recursion_limit_hit(
                        state, config, step, max_steps, current_node
                    )
                    if streaming:
                        reason = self.RECURSION_MESSAGE
                        yield note("graph_invoked", StreamEvent.ERROR, state=state, reason=reason)
                    raise error

            _tp3 = _perf_now()  # TEMP-PERF
            state, trimmed = await self._complete_run(
                state, config, lifecycle_context, messages, step, current_node, last_save
            )
            _perf_done("core_complete_run", _tp3)  # TEMP-PERF
            run.state = state
            if streaming:
                yield note(
                    "graph_invoked",
                    state=state,
                    is_context_trimmed=trimmed,
                    reason="Graph execution completed successfully",
                )

        except (asyncio.CancelledError, GeneratorExit):
            # Cancelled mid-run (client gone, shutdown, timeout around the run) or the stream
            # closed by its consumer (GeneratorExit at a yield): save the thread as
            # INTERRUPTED so the next run can use it, then let it go on. Nothing is yielded
            # here, so awaiting during aclose() is allowed.
            resume_at: str | None = None
            if current_node == END:
                resume_at = END  # every node ran; only the completed save was left
            elif done is not None:
                # The node finished: save its output and, as a stop does, the node after it,
                # unless ``_advance`` already counted the step (it then names that node).
                merge(done)
                if state.execution_meta.step == step:
                    resume_at = node_after(current_node, done.next_node, state, self.edges)
            await finalize_cancelled_run(
                state, config, messages, current_node, last_save=last_save, resume_at=resume_at
            )
            raise
        except Exception as e:
            state = run.state = await self._fail_run(
                e,
                state,
                config,
                lifecycle_context,
                messages,
                step,
                current_node,
                last_save,
            )
            if streaming:
                reason = client_error_text(e)
                yield note("invoked_graph", StreamEvent.ERROR, state=state, reason=reason)
            raise


def _set_run_end(event: EventModel, event_type: EventType, name: str, result: RunResult) -> None:
    """Turn the run's event into a run-end event carrying the final state."""
    state = result.state
    event.event_type = event_type
    event.metadata["event_name"] = name
    event.metadata["step"] = state.execution_meta.step
    event.metadata["current_node"] = state.execution_meta.current_node
    event.metadata["is_interrupted"] = state.is_interrupted()
    event.metadata["total_messages"] = len(state.context) if state.context else 0
    event.data["state"] = state


def _chunk(config: dict[str, Any], event: StreamEvent, **fields: Any) -> StreamChunk:
    """A stream chunk of this run (only ever built when streaming)."""
    return StreamChunk(
        event=event, thread_id=config.get("thread_id"), run_id=config.get("run_id"), **fields
    )
