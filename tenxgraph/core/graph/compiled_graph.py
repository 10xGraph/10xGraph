from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import logging
from collections.abc import AsyncIterator, Callable, Generator
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

from injectq import InjectQ

from tenxgraph.core.exceptions.graph_error import GraphError
from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.graph.tool_node.base import ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.core.state.execution_state import StopRequestStatus
from tenxgraph.core.state.stream_chunks import StreamChunk
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.storage.checkpointer import base_checkpointer
from tenxgraph.storage.checkpointer.base_checkpointer import BaseCheckpointer
from tenxgraph.storage.media.url_policy import MediaFetchPolicy
from tenxgraph.storage.store.base_store import BaseStore
from tenxgraph.utils import (
    CallbackManager,
    ResponseGranularity,
    resource_registry,
)
from tenxgraph.utils.background_task_manager import BackgroundTaskManager, current_task_manager
from tenxgraph.utils.callable_utils import ensure_no_running_loop, resolve_awaitable_id
from tenxgraph.utils.constants import DEFAULT_ANONYMOUS_USER_ID

from .node import Node
from .remote_tool import RemoteToolConfig
from .utils.invoke_handler import InvokeHandler
from .utils.stream_handler import StreamHandler


if TYPE_CHECKING:
    from types import TracebackType

    from .state_graph import StateGraph


StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("tenxgraph.graph")


def _new_id() -> Any:
    """A fresh id from the graph's id generator (sync or async), else a uuid4 string."""
    return resolve_awaitable_id(InjectQ.get_instance().try_get("generated_id")) or str(uuid4())


def _reset_task_manager(token: contextvars.Token) -> None:
    """Undo the per-run task manager binding.

    An async generator can be closed from a different context than the one that started
    it, where reset() raises. The binding then simply lapses with that context.
    """
    try:
        current_task_manager.reset(token)
    except ValueError:
        logger.debug("Task manager binding was set in another context; leaving it")


class CompiledGraph[StateT: AgentState]:
    """A fully compiled and executable graph ready for workflow execution.

    CompiledGraph represents the final executable form of a StateGraph after compilation.
    It encapsulates all the execution logic, handlers, and services needed to run
    agent workflows. The graph supports both synchronous and asynchronous execution
    with comprehensive state management, checkpointing, event publishing, and
    streaming capabilities.

    This class is generic over state types to support custom AgentState subclasses,
    ensuring type safety throughout the execution process.

    Key Features:
    - Synchronous and asynchronous execution methods
    - Real-time streaming with incremental results
    - State persistence and checkpointing
    - Interrupt and resume capabilities
    - Event publishing for monitoring and debugging
    - Background task management
    - Graceful error handling and recovery

    Attributes:
        _state: The initial/template state for graph executions.
        _invoke_handler: Handler for non-streaming graph execution.
        _stream_handler: Handler for streaming graph execution.
        _checkpointer: Optional state persistence backend.
        _publisher: Optional event publishing backend.
        _store: Optional data storage backend.
        _state_graph: Reference to the source StateGraph.
        _interrupt_before: Nodes where execution should pause before execution.
        _interrupt_after: Nodes where execution should pause after execution.
        _task_manager: Manager for background async tasks.

    Example:
        ```python
        # After building and compiling a StateGraph
        compiled = graph.compile()

        # Synchronous execution
        result = compiled.invoke({"messages": [Message.text_message("Hello")]})

        # Asynchronous execution with streaming
        async for chunk in compiled.astream({"messages": [message]}):
            print(f"Streamed: {chunk.content}")

        # Graceful cleanup
        await compiled.aclose()
        ```

    Note:
        CompiledGraph instances should be properly closed using aclose() to
        release resources like database connections, background tasks, and
        event publishers.
    """

    def __init__(  # noqa: PLR0913
        self,
        state: StateT,
        checkpointer: BaseCheckpointer[StateT] | None,
        publisher: BasePublisher | None,
        store: BaseStore | None,
        state_graph: StateGraph[StateT],
        interrupt_before: list[str],
        interrupt_after: list[str],
        task_manager: BackgroundTaskManager,
        shutdown_timeout: float = 30.0,
        media_policy: MediaFetchPolicy | None = None,
        owns_resources: bool = True,
    ):
        logger.info(
            f"Initializing CompiledGraph with nodes: {list(state_graph.nodes.keys())}",
        )

        # Save initial state
        self._state = state
        self._shutdown_timeout = shutdown_timeout
        self._media_policy: MediaFetchPolicy = media_policy or MediaFetchPolicy()

        # create handlers
        self._invoke_handler: InvokeHandler[StateT] = InvokeHandler[StateT](
            nodes=state_graph.nodes,
            edges=state_graph.edges,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
        )
        self._stream_handler: StreamHandler[StateT] = StreamHandler[StateT](
            nodes=state_graph.nodes,
            edges=state_graph.edges,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
        )

        self._checkpointer: BaseCheckpointer[StateT] | None = checkpointer
        self._publisher: BasePublisher | None = publisher
        self._store: BaseStore | None = store
        self._state_graph: StateGraph[StateT] = state_graph
        self._interrupt_before: list[str] = interrupt_before
        self._interrupt_after: list[str] = interrupt_after
        # generate task manager
        self._task_manager = task_manager
        # Guards aclose() against being run more than once (e.g. an explicit
        # aclose() inside an ``async with`` block followed by __aexit__).
        self._closed = False
        # Resource ownership: aclose() closes a shared resource only when the last live
        # graph holding it closes; a graph with owns_resources=False never closes it.
        self._owns_resources = owns_resources
        if owns_resources:
            for resource in (checkpointer, publisher, store):
                resource_registry.register(resource, self)

    def _skip_close(self, resource: object, name: str) -> dict[str, Any] | None:
        """Return a skip result when this graph must not close ``resource``, else None."""
        if not self._owns_resources:
            return {"status": "skipped", "reason": "owns_resources=False"}
        if not resource_registry.release(resource, self):
            logger.info("%s is still used by another live graph; not closing it", name)
            return {"status": "skipped", "reason": "shared with another live graph"}
        return None

    @property
    def media_policy(self) -> MediaFetchPolicy:
        """The policy for downloading external media URLs (never ``None``).

        Set with ``StateGraph.compile(media_policy=...)``; ``MediaFetchPolicy()`` otherwise.
        Call ``media_policy.check_url(url)`` to check a URL without touching the network.
        """
        return self._media_policy

    async def __aenter__(self) -> CompiledGraph[StateT]:
        """Enter an async context; returns this graph unchanged.

        Enables ``async with compiled_graph as graph: ...``, which guarantees
        :meth:`aclose` runs on exit even if the body raises.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the async context, releasing all resources via :meth:`aclose`."""
        await self.aclose()

    def _prepare_config(
        self,
        config: dict[str, Any] | None,
        is_stream: bool = False,
    ) -> dict[str, Any]:
        cfg = dict(config or {})

        if "thread_id" not in cfg:
            cfg["thread_id"] = _new_id()
            # A caller who omits thread_id gets a fresh random one, which means the
            # run cannot be resumed and `stop()` can never target it -- the caller
            # has no way to name it. That is fine for a one-shot script but is
            # almost never what a server wants, so say so rather than fail silently.
            logger.warning(
                "No thread_id in config; generated '%s'. This run cannot be resumed "
                "or stopped, because the caller does not know its thread_id. Pass an "
                "explicit thread_id for any run you may want to resume or stop.",
                cfg["thread_id"],
            )

        if "is_stream" not in cfg:
            cfg["is_stream"] = is_stream
        if "user_id" not in cfg:
            # Was "test-user-id" -- a placeholder that looks like a real user. With
            # per-user isolation enabled that silently filed every un-authenticated
            # run under one fake account. "anonymous" matches what the API layer
            # uses when no auth is configured, so the two agree.
            cfg["user_id"] = DEFAULT_ANONYMOUS_USER_ID
            logger.debug(
                "No user_id in config; using '%s'. Enable auth if you need per-user "
                "isolation -- every anonymous run shares one identity.",
                DEFAULT_ANONYMOUS_USER_ID,
            )
        if "run_id" not in cfg:
            cfg["run_id"] = _new_id()

        if "timestamp" not in cfg:
            cfg["timestamp"] = datetime.datetime.now().isoformat()

        return cfg

    def invoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
        """Execute the graph synchronously and return the final results.

        Runs the complete graph workflow from start to finish, handling state
        management, node execution, and result formatting. This method automatically
        detects whether to start a fresh execution or resume from an interrupted state.

        The execution is synchronous but internally uses async operations, making it
        suitable for use in non-async contexts while still benefiting from async
        capabilities for I/O operations.

        Args:
            input_data: Input dictionary for graph execution. For new executions,
                should contain 'messages' key with list of initial messages.
                For resumed executions, can contain additional data to merge.
            config: Optional configuration dictionary containing execution settings:
                - user_id: Identifier for the user/session
                - thread_id: Unique identifier for this execution thread
                - run_id: Unique identifier for this specific run
                - recursion_limit: Maximum steps before stopping (default: 25)
            response_granularity: Level of detail in the response:
                - LOW: Returns only messages (default)
                - PARTIAL: Returns context, summary, and messages
                - FULL: Returns complete state and messages

        Returns:
            Dictionary containing execution results formatted according to the
            specified granularity level. Always includes execution messages
            and may include additional state information.

        Raises:
            ValueError: If input_data is invalid for new execution.
            GraphRecursionError: If execution exceeds recursion limit.
            Various exceptions: Depending on node execution failures.

        Example:
            ```python
            # Basic execution
            result = compiled.invoke({"messages": [Message.text_message("Process this data")]})
            print(result["messages"])  # Final execution messages

            # With configuration and full details
            result = compiled.invoke(
                input_data={"messages": [message]},
                config={"user_id": "user123", "thread_id": "session456", "recursion_limit": 50},
                response_granularity=ResponseGranularity.FULL,
            )
            print(result["state"])  # Complete final state
            ```

        Note:
            This method uses asyncio.run() internally and raises RuntimeError when called
            from within an async context. Use ainvoke() instead for async execution.

            Publisher events are best-effort background work. Before returning, this
            method waits up to ``shutdown_timeout`` for pending events to be delivered.
            Events are shed (with a warning in the log) when more than 1000 are pending,
            for example when the publisher's sink is slow or down.
        """
        ensure_no_running_loop("invoke", "ainvoke")
        logger.info(
            "Starting synchronous graph execution with %d input keys, granularity=%s",
            len(input_data) if input_data else 0,
            response_granularity,
        )
        logger.debug("Input data keys: %s", list(input_data.keys()) if input_data else [])
        # Async Will Handle Event Publish

        try:
            result = asyncio.run(self._invoke_and_drain(input_data, config, response_granularity))
            logger.info("Synchronous graph execution completed successfully")
            return result
        except Exception as e:
            logger.exception("Synchronous graph execution failed: %s", e)
            raise

    async def _invoke_and_drain(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None,
        response_granularity: ResponseGranularity,
    ) -> dict[str, Any]:
        """Run ainvoke, then wait for the background work it started (sync invoke only).

        ``asyncio.run`` cancels whatever is still pending when it returns, which would
        drop publisher events, memory writes and thread naming.
        """
        try:
            return await self.ainvoke(input_data, config, response_granularity)
        finally:
            await self._drain_background_tasks()

    async def _drain_background_tasks(self) -> None:
        """Wait up to the shutdown timeout for tracked background tasks to finish."""
        try:
            await self._task_manager.wait_for_all(
                timeout=self._shutdown_timeout, return_exceptions=True
            )
        except Exception as e:
            logger.warning("Draining background tasks failed: %s", e)

    async def ainvoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> dict[str, Any]:
        """Execute the graph asynchronously.

        Auto-detects whether to start fresh execution or resume from interrupted state
        based on the AgentState's execution metadata.

        Args:
            input_data: Input dict with 'messages' key (for new execution) or
                       additional data for resuming
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Returns:
            Response dict based on granularity
        """
        self._guard_not_realtime()
        cfg = self._prepare_config(config, is_stream=False)
        heartbeat = self._start_heartbeat(cfg)
        manager_token = current_task_manager.set(self._task_manager)

        try:
            return await self._invoke_handler.invoke(
                input_data,
                cfg,
                self._state,
                response_granularity,
            )
        finally:
            _reset_task_manager(manager_token)
            await self._end_run(cfg, heartbeat)

    def _start_heartbeat(self, cfg: dict[str, Any]) -> asyncio.Task | None:
        """Keep this run's active-run marker alive while the run lasts.

        The marker is short-lived on purpose: a run that dies without cleanup (SIGKILL, OOM)
        leaves one that expires within a minute, and crash recovery reads that as a dead run.
        """
        if not self._checkpointer:
            return None
        checkpointer = self._checkpointer

        async def beat() -> None:
            while True:
                await asyncio.sleep(base_checkpointer.ACTIVE_RUN_HEARTBEAT_SECONDS)
                try:
                    await checkpointer.arefresh_run(cfg)
                except Exception as e:  # a missed beat is retried on the next tick
                    logger.warning("Run heartbeat failed for %s (%s)", cfg.get("thread_id"), e)

        return asyncio.ensure_future(beat())

    async def _end_run(self, cfg: dict[str, Any], heartbeat: asyncio.Task | None = None) -> None:
        """Tell the checkpointer this run is over so its stop scope is released."""
        if heartbeat is not None:
            heartbeat.cancel()
            # Wait for it to unwind. asyncio.wait never raises the task's own cancel, and an
            # outer cancel of this coroutine still propagates.
            await asyncio.wait({heartbeat})
        if not self._checkpointer:
            return
        try:
            await self._checkpointer.aend_run(cfg)
        except Exception:
            logger.warning("Failed to release run scope for thread_id: %s", cfg.get("thread_id"))

    def stop(self, config: dict[str, Any]) -> dict[str, Any]:
        """Request the current graph execution to stop (sync helper).

        This sets a stop flag in the checkpointer's thread store keyed by thread_id.
        Handlers periodically check this flag and interrupt execution.
        Returns a small status dict.
        """
        ensure_no_running_loop("stop", "astop")
        return asyncio.run(self.astop(config))

    async def astop(self, config: dict[str, Any]) -> dict[str, Any]:
        """Request the current graph execution to stop (async).

        Contract:
        - Requires a valid thread_id in config
        - If no active thread or no checkpointer, returns not-running
        - If state exists and is running, set stop_requested flag in thread info
        """
        cfg = self._prepare_config(config, is_stream=bool((config or {}).get("is_stream", False)))
        if not self._checkpointer:
            return {"ok": False, "reason": "no-checkpointer"}

        # Load state to see if this thread is running
        # Lets load from the cache, incase if not available lets load from the db
        # In prebuilt implement like pgsql, there its called internally, no need this call
        # Still lets do that incase user use their own
        state = await self._checkpointer.aget_state_cache(
            cfg
        ) or await self._checkpointer.aget_state(cfg)
        if not state:
            return {"ok": False, "running": False, "reason": "no-state"}

        running = state.is_running() and not state.is_interrupted()
        # Set stop flag regardless; handlers will act if running
        if running:
            # Record the stop in its own cache entry FIRST. The running graph
            # rewrites the state cache after every node with its own flag-less
            # copy of the state (at the same durable version, so a version-guarded
            # cache accepts it), which would silently swallow a flag written only
            # into the state blob. Nothing in the run loop writes this key, so a
            # stop recorded here always survives to the next stop check.
            await self._checkpointer.arequest_stop(cfg)

            # Also mirror it into the cached state. Redundant for checkpointers
            # that back the stop key, but it keeps stop working for a custom
            # checkpointer that has not implemented the generic cache (whose
            # arequest_stop inherits the no-op default).
            state.execution_meta.stop_current_execution = StopRequestStatus.STOP_REQUESTED
            await self._checkpointer.aput_state_cache(cfg, state)
            logger.info("Set stop_current_execution flag for thread_id: %s", cfg.get("thread_id"))
            return {"ok": True, "running": running}

        logger.info(
            "No running execution to stop for thread_id: %s (running=%s, interrupted=%s)",
            cfg.get("thread_id"),
            running,
            state.is_interrupted(),
        )
        return {"ok": True, "running": running, "reason": "not-running"}

    def override_node(
        self,
        name: str,
        func: Callable | ToolNode | BaseAgent,
    ) -> CompiledGraph[StateT]:
        """Override a node in an already-compiled graph.

        Useful for testing pre-built production graphs by swapping
        nodes with test doubles after compilation.

        Args:
            name: Name of the existing node to override
            func: New function, ToolNode, or Agent to use

        Returns:
            CompiledGraph: The graph instance for method chaining.

        Raises:
            KeyError: If the node doesn't exist

        Example:
            ```python
            # Production factory
            def create_production_workflow():
                graph = StateGraph()
                graph.add_node("MAIN", production_agent)
                # ... complex setup
                return graph.compile()


            # Test
            compiled = create_production_workflow()
            compiled.override_node("MAIN", test_agent)  # Override after compile
            result = await compiled.ainvoke(...)
            ```
        """
        if name not in self._state_graph.nodes:
            raise KeyError(f"Node '{name}' does not exist")

        # Create new Node and update the graph's dict
        new_node = Node(name, func)
        self._state_graph.nodes[name] = new_node

        # Re-register in container so handlers pick up the change
        # The handlers read from container's "get_node" factory
        self._state_graph._container.bind_factory(
            "get_node",
            lambda x: self._state_graph.nodes[x],
        )

        logger.debug("Overrode node '%s' in compiled graph", name)
        return self

    def stream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> Generator[StreamChunk]:
        """Execute the graph synchronously with streaming support.

        Yields Message objects containing incremental responses.
        If nodes return streaming responses, yields them directly.
        If nodes return complete responses, simulates streaming by chunking.

        Args:
            input_data: Input dict
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Yields:
            Message objects with incremental content
        """

        ensure_no_running_loop("stream", "astream")

        # For sync streaming, we'll use asyncio.run to handle the async implementation
        async def _async_stream():
            async for chunk in self.astream(input_data, config, response_granularity):
                yield chunk

        # Convert async generator to sync iteration with a dedicated event loop
        gen = _async_stream()
        loop = asyncio.new_event_loop()
        logger.info("Synchronous streaming started")

        try:
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            # Attempt to close the async generator cleanly
            with contextlib.suppress(Exception):
                loop.run_until_complete(gen.aclose())  # type: ignore[attr-defined]
            # Same as sync invoke: loop.close() would drop the pending background work.
            with contextlib.suppress(Exception):
                loop.run_until_complete(self._drain_background_tasks())
            loop.close()
        logger.info("Synchronous streaming completed")

    async def astream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        response_granularity: ResponseGranularity = ResponseGranularity.LOW,
    ) -> AsyncIterator[StreamChunk]:
        """Execute the graph asynchronously with streaming support.

        Yields Message objects containing incremental responses.
        If nodes return streaming responses, yields them directly.
        If nodes return complete responses, simulates streaming by chunking.

        Args:
            input_data: Input dict
            config: Configuration dictionary
            response_granularity: Response parsing granularity

        Yields:
            Message objects with incremental content
        """
        self._guard_not_realtime()
        cfg = self._prepare_config(config, is_stream=True)
        heartbeat = self._start_heartbeat(cfg)
        manager_token = current_task_manager.set(self._task_manager)

        try:
            async for chunk in self._stream_handler.stream(
                input_data,
                cfg,
                self._state,
                response_granularity,
            ):
                yield chunk
        finally:
            _reset_task_manager(manager_token)
            await self._end_run(cfg, heartbeat)

    def attach_remote_tools(
        self,
        tools: list[dict | RemoteToolConfig],
        node_name: str | None = None,
    ) -> None:
        """Attach remote tools to a ToolNode in the graph.

        Remote tools are executed by the client, not the server. The graph only
        advertises their schemas to the model and emits a ``RemoteToolCallBlock``
        when one is called.

        Args:
            tools: Either a list of :class:`RemoteToolConfig` values (or their
                flat dictionary form), or OpenAI function-calling schemas when
                ``node_name`` is supplied.
            node_name: ToolNode name for OpenAI-schema input. Omit this when
                each tool carries its own ``node``/``node_name`` value.

        Raises:
            GraphError: If the specified node is not a ToolNode.

        Example:
            >>> tool_configs = [
            ...     {
            ...         "type": "function",
            ...         "function": {
            ...             "name": "read_clipboard",
            ...             "description": "Read the current clipboard text.",
            ...             "parameters": {"type": "object", "properties": {}, "required": []},
            ...         },
            ...     }
            ... ]
            >>> graph.attach_remote_tools(tool_configs, "tool_node")
        """
        if node_name is None:
            configs = [
                tool
                if isinstance(tool, RemoteToolConfig)
                else RemoteToolConfig.model_validate(tool)
                for tool in tools
            ]
            seen: set[str] = set()
            grouped: dict[str, list[dict[str, Any]]] = {}
            for config in configs:
                if config.name in seen:
                    raise ValueError(f"Duplicate remote tool name '{config.name}'")
                seen.add(config.name)
                grouped.setdefault(config.node_name, []).append(config.to_tool_schema())

            for configured_node, schemas in grouped.items():
                self.attach_remote_tools(schemas, configured_node)
            return

        normalized_tools: list[dict] = []
        for tool in tools:
            if isinstance(tool, RemoteToolConfig):
                if tool.node_name != node_name:
                    raise ValueError(
                        f"Remote tool '{tool.name}' targets node '{tool.node_name}', "
                        f"not '{node_name}'"
                    )
                normalized_tools.append(tool.to_tool_schema())
            else:
                normalized_tools.append(tool)

        logger.debug(
            "Attaching remote tools to node '%s': %s",
            node_name,
            normalized_tools,
        )
        node: Node | None = self._state_graph.nodes.get(node_name)
        if not node:
            raise GraphError(
                message=f"Node '{node_name}' not found in graph",
                error_code="GRAPH_004",
                context={"node_name": node_name},
            )

        if not isinstance(node.func, ToolNode):
            error_msg = f"Node '{node_name}' is not a ToolNode"
            logger.error(error_msg)
            raise GraphError(
                message=error_msg,
                error_code="GRAPH_005",
                context={"node_name": node_name},
            )

        tool_node: ToolNode = node.func
        tool_node.set_remote_tool(normalized_tools)
        logger.info(
            "Attached %d remote tools to ToolNode '%s'",
            len(normalized_tools),
            node_name,
        )

    # ------------------------------------------------------------------ #
    # Realtime runtime (audio-to-audio). A separate runtime from the
    # super-step invoke/stream loop: the live agent owns the turn loop.
    # ------------------------------------------------------------------ #
    def _find_live_nodes(self) -> list[tuple[str, Node]]:
        from tenxgraph.core.realtime.live_agent import LiveAgent

        return [
            (name, node)
            for name, node in self._state_graph.nodes.items()
            if isinstance(node.func, LiveAgent)
        ]

    def is_realtime(self) -> bool:
        """Whether this graph is a realtime (live) graph.

        True when the graph contains at least one ``LiveAgent`` node, meaning it must be
        driven via ``arealtime()`` / ``realtime()`` rather than invoke/stream.
        """
        return bool(self._find_live_nodes())

    def _guard_not_realtime(self) -> None:
        """Forcing rule: a graph containing a LiveAgent must use arealtime()."""
        if self._find_live_nodes():
            raise RuntimeError(
                "This graph contains a LiveAgent; use .arealtime() / .realtime() instead of "
                "invoke/ainvoke/stream/astream."
            )

    async def arealtime(
        self,
        input_queue: Any,
        config: dict[str, Any] | None = None,
        state: AgentState | None = None,
    ) -> AsyncIterator[Any]:
        """Run the graph's realtime (audio) session, yielding normalized RealtimeEvents.

        Forcing rule: the graph must contain exactly one LiveAgent (the root controller);
        ordinary turn-based graphs must use invoke/stream.
        """
        live = self._find_live_nodes()
        if not live:
            raise RuntimeError(
                "arealtime() requires a graph rooted at a LiveAgent (e.g. AudioAgent); "
                "this graph has none. Use invoke/stream for turn-based graphs."
            )
        if len(live) > 1:
            raise RuntimeError(
                "Only one LiveAgent is allowed per realtime run in v1 "
                f"(found {len(live)}: {[name for name, _ in live]})."
            )

        name, node = live[0]
        agent = node.func
        agent._node_name = name
        cfg = self._prepare_config(config, is_stream=True)
        callback_manager = InjectQ.get_instance().try_get(CallbackManager)
        context_manager = self._state_graph._context_manager
        run_state = state if state is not None else (self._state or AgentState())

        async for event in agent.arun(
            input_queue,
            cfg,
            run_state,
            checkpointer=self._checkpointer,
            callback_manager=callback_manager,
            context_manager=context_manager,
        ):
            yield event

    def realtime(
        self,
        input_queue: Any,
        config: dict[str, Any] | None = None,
        state: AgentState | None = None,
    ) -> Generator[Any]:
        """Synchronous wrapper over :meth:`arealtime` for non-async consumers.

        Must be called from a thread with no running event loop; from inside an async
        context (FastAPI handler, Jupyter), use :meth:`arealtime` directly.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no running loop: safe to drive a private one below
        else:
            raise RuntimeError(
                "realtime() (sync) cannot be called from a running event loop; "
                "await arealtime() instead."
            )

        agen = self.arealtime(input_queue, config, state)
        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(agen.aclose())
            loop.close()

    async def aclose(self) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
        """
        Close the graph and release all resources gracefully.

        This method ensures proper cleanup of all components with timeout handling.
        It follows a structured shutdown sequence:
        1. Background task manager (running tasks)
        2. Checkpointer (state persistence)
        3. Publisher (event publishing)
        4. Store (data storage)

        Returns:
            Dictionary with detailed shutdown statistics for each component.

        Calling this more than once is a no-op; the second call returns
        ``{"status": "already_closed"}``. Prefer the async-context-manager form,
        which calls this automatically on exit:

        Example:
            ```python
            async def main():
                async with await build_and_compile_graph() as graph:
                    await graph.ainvoke(input_data)
                # graph.aclose() has run here, even if ainvoke raised


            # Or manage the lifecycle manually:
            async def main_manual():
                graph = await build_and_compile_graph()
                try:
                    await graph.ainvoke(input_data)
                finally:
                    stats = await graph.aclose()
                    print(f"Shutdown completed: {stats}")
            ```
        """
        from tenxgraph.utils.shutdown import shutdown_with_timeout

        if self._closed:
            logger.debug("CompiledGraph.aclose() called again; already closed")
            return {"status": "already_closed"}
        self._closed = True

        logger.info("Initiating graceful shutdown of CompiledGraph")
        stats: dict[str, Any] = {}
        start_time = asyncio.get_event_loop().time()

        # 1. Shutdown background task manager (handles all pending async writes)
        try:
            logger.debug("Shutting down background task manager...")
            shutdown_stats = await self._task_manager.shutdown(timeout=self._shutdown_timeout)
            logger.info("Background task manager shutdown completed")
            stats["background_tasks"] = shutdown_stats
        except Exception as e:
            stats["background_tasks"] = {"status": "error", "error": str(e)}
            logger.exception("Error shutting down background tasks: %s", e)

        # 2. Close checkpointer (may have pending writes)
        if self._checkpointer and (skip := self._skip_close(self._checkpointer, "Checkpointer")):
            stats["checkpointer"] = skip
        elif self._checkpointer:
            try:
                logger.debug("Releasing checkpointer...")
                result = await shutdown_with_timeout(
                    self._checkpointer.arelease(),
                    timeout=self._shutdown_timeout / 3,  # Give 1/3 of total timeout
                    task_name="checkpointer",
                )
                logger.info("Checkpointer closed successfully")
                stats["checkpointer"] = result
            except Exception as e:
                stats["checkpointer"] = {"status": "error", "error": str(e)}
                logger.exception("Error closing checkpointer: %s", e)
        else:
            stats["checkpointer"] = {"status": "skipped", "reason": "no checkpointer"}

        # 3. Close publisher (event delivery)
        if self._publisher and (skip := self._skip_close(self._publisher, "Publisher")):
            stats["publisher"] = skip
        elif self._publisher:
            try:
                logger.debug("Closing publisher...")
                result = await shutdown_with_timeout(
                    self._publisher.close(),
                    timeout=self._shutdown_timeout / 3,
                    task_name="publisher",
                )
                logger.info("Publisher closed successfully")
                stats["publisher"] = result
            except Exception as e:
                stats["publisher"] = {"status": "error", "error": str(e)}
                logger.exception("Error closing publisher: %s", e)
        else:
            stats["publisher"] = {"status": "skipped", "reason": "no publisher"}

        # 4. Close store (data persistence)
        if self._store and (skip := self._skip_close(self._store, "Store")):
            stats["store"] = skip
        elif self._store:
            try:
                logger.debug("Releasing store...")
                result = await shutdown_with_timeout(
                    self._store.arelease(),
                    timeout=self._shutdown_timeout / 3,
                    task_name="store",
                )
                logger.info("Store closed successfully")
                stats["store"] = result
            except Exception as e:
                stats["store"] = {"status": "error", "error": str(e)}
                logger.exception("Error closing store: %s", e)
        else:
            stats["store"] = {"status": "skipped", "reason": "no store"}

        total_duration = asyncio.get_event_loop().time() - start_time
        stats["total_duration"] = total_duration
        logger.info(f"CompiledGraph shutdown completed in {total_duration:.2f}s. Stats: {stats}")
        return stats

    def generate_graph(self) -> dict[str, Any]:
        """Generate the graph representation.

        Returns:
            A dictionary representing the graph structure.
        """
        graph = {
            "info": {},
            "nodes": [],
            "edges": [],
        }
        # Populate the graph with nodes and edges
        for node_name in self._state_graph.nodes:
            graph["nodes"].append(
                {
                    "id": str(uuid4()),
                    "name": node_name,
                }
            )

        for edge in self._state_graph.edges:
            graph["edges"].append(
                {
                    "id": str(uuid4()),
                    "source": edge.from_node,
                    "target": edge.to_node,
                }
            )

        # Add few more extra info
        graph["info"] = {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "checkpointer": self._checkpointer is not None,
            "checkpointer_type": type(self._checkpointer).__name__ if self._checkpointer else None,
            "publisher": self._publisher is not None,
            "store": self._store is not None,
            "interrupt_before": self._interrupt_before,
            "interrupt_after": self._interrupt_after,
            "context_type": self._state_graph._context_manager.__class__.__name__,
            "id_generator": self._state_graph._id_generator.__class__.__name__,
            "id_type": self._state_graph._id_generator.id_type.value,
            "state_type": self._state.__class__.__name__,
            "state_fields": list(self._state.model_dump().keys()),
        }
        return graph
