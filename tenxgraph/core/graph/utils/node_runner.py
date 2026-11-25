"""One runner for a graph node, under invoke and stream alike.

``NodeRunner.run`` runs one step of a node: a function node, an agent node, or a ToolNode
(through ``ToolRunner``). It yields ``StreamChunk``s (only when streaming) and last a
``NodeResult``. Under invoke no chunk is built and an agent node makes a non-streaming LLM
call (``converter.invoke()``); under stream it reads ``converter.stream()``.
"""

import inspect
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from tenxgraph.core.authz import public_config
from tenxgraph.core.exceptions import NodeError
from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.stream_chunks import StreamChunk, StreamEvent
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.utils import CallbackContext, CallbackManager, InvocationType, call_sync_or_async
from tenxgraph.utils.command import Command
from tenxgraph.utils.constants import END, START
from tenxgraph.utils.interrupt import activate as activate_interrupts
from tenxgraph.utils.log_safety import error_details

from .handler_utils import is_remote_tool_marker
from .tool_runner import ToolRunner, ToolRunResult
from .utils import process_node_result


logger = logging.getLogger("tenxgraph.graph")


@dataclass(slots=True)
class NodeResult:
    """What one node step produced.

    ``state`` is the state after the step (a ``Command(state=S)`` makes it ``S``),
    ``messages`` the new messages in order, ``next_node`` the node a ``Command`` or a
    handoff chose, and ``remote_calls`` the placeholders of tool calls the client runs.
    """

    state: AgentState
    messages: list[Message] = field(default_factory=list)
    next_node: str | None = None
    remote_calls: list[Message] = field(default_factory=list)


class NodeRunner:
    """Runs one step of the graph node ``name`` (``func``: function, agent or ToolNode)."""

    # Signatures of node functions, inspected once.
    _signature_cache: dict[Callable, inspect.Signature] = {}

    def __init__(self, name: str, func: Any):
        self.name = name
        self.func = func

    async def run(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager,
        *,
        streaming: bool,
    ) -> AsyncIterator[StreamChunk | NodeResult]:
        """Run the node; chunks while it runs (streaming only), the ``NodeResult`` last.

        Raises ``NodeError`` (``NODE_001``) when the node fails and no ``on_error``
        callback recovered it.
        """
        if self.name in (START, END):
            # Internal pseudo-nodes: no user callbacks and no node events.
            yield NodeResult(state)
            return
        logger.info("Executing node '%s'", self.name)
        if isinstance(self.func, ToolNode):
            steps = self._run_tools(config, state, streaming)
        elif isinstance(self.func, BaseAgent):
            steps = self._run_agent(config, state, callback_mgr, streaming)
        else:
            steps = self._run_function(config, state, callback_mgr, streaming)
        try:
            async for item in steps:
                yield item
        except Exception as e:
            logger.exception("Node '%s' execution failed: %s", self.name, e)
            raise NodeError(
                message=f"Error in node '{self.name}': {e!s}",
                error_code="NODE_001",
                context={"node_name": self.name, "error_type": type(e).__name__},
            ) from e
        logger.info("Node '%s' execution completed successfully", self.name)

    async def _run_tools(
        self, config: dict[str, Any], state: AgentState, streaming: bool
    ) -> AsyncIterator[StreamChunk | NodeResult]:
        """A ToolNode step; client-tool placeholders are split off the messages here."""
        result = ToolRunResult()
        async for item in ToolRunner(self.name, self.func).run(state, config, streaming=streaming):
            if isinstance(item, ToolRunResult):
                result = item
            else:
                yield item
        messages = [m for m in result.messages if not is_remote_tool_marker(m)]
        remote = [m for m in result.messages if is_remote_tool_marker(m)]
        yield NodeResult(state, messages, result.next_node, remote)

    async def _run_function(  # noqa: PLR0912
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager,
        streaming: bool,
    ) -> AsyncIterator[StreamChunk | NodeResult]:
        """A function node: callbacks, the call, then its return value turned into messages."""
        function_name = getattr(self.func, "__name__", str(self.func))
        context = self._callback_context(config, function_name)
        input_data = self._prepare_input_data(state, config)
        event = self._error_event(config, state, function_name)
        try:
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            if streaming:
                yield self._update_chunk("invoking_node", function_name, config)

            # Snapshot the context: a node may append to it and return the same state object.
            seen_ids: set[Any] = {msg.message_id for msg in state.context or []}
            # interrupt() in the node reads the scope the graph loop opened for this node.
            with activate_interrupts(config):
                result = await call_sync_or_async(self.func, **input_data)
            result = await callback_mgr.execute_after_invoke(context, input_data, result)
            if streaming:
                yield self._update_chunk("node_invoked", function_name, config)

            messages: list[Message] = []
            next_node = None
            is_command = isinstance(result, Command)
            if is_command:
                next_node = result.goto
                if result.state is not None:
                    # Command(state=S): S is the new state; its new messages are recorded.
                    state = result.state
                    messages = [m for m in state.context or [] if m.message_id not in seen_ids]
                    seen_ids.update(m.message_id for m in messages)
                result = result.update

            if streaming and isinstance(result, ModelResponseConverter):
                stream = result.stream(
                    config, node_name=self.name, meta={"function_name": function_name}
                )
                async for item in stream:
                    if isinstance(item, Message) and not item.delta:
                        messages.append(item)
                    yield self._message_chunk(item, function_name, config)
                self._stamp_node_name(messages)
            else:
                # Command(update=None) and a bare None add nothing; any other value must be a
                # supported one (process_node_result raises for the rest: NODE_001).
                if not (is_command and not result):
                    state, messages, _ = await process_node_result(
                        result, state, messages, seen_ids
                    )
                self._stamp_node_name(messages)
                if streaming:
                    for message in messages:
                        yield self._message_chunk(message, function_name, config)

            yield NodeResult(state, messages, next_node)
        except Exception as e:
            recovery = await self._recover(callback_mgr, context, input_data, event, e, config)
            if streaming:
                yield self._message_chunk(recovery, function_name, config)
            yield NodeResult(state, [recovery])

    async def _run_agent(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager,
        streaming: bool,
    ) -> AsyncIterator[StreamChunk | NodeResult]:
        """An agent node: one LLM call; the graph loop adds the reply to the state."""
        agent_name = type(self.func).__name__
        context = self._callback_context(config, agent_name)
        input_data: dict[str, Any] = {"state": state, "config": config}
        event = self._error_event(config, state, agent_name)
        try:
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            converter = await self.func.execute(state, config)
            messages: list[Message] = []
            if streaming:
                stream = converter.stream(
                    config, node_name=self.name, meta={"function_name": self.name}
                )
                async for item in stream:
                    if isinstance(item, Message) and not item.delta:
                        messages.append(item)
                    yield self._message_chunk(item, self.name, config)
            else:
                messages.append(await converter.invoke())
            self._stamp_node_name(messages)

            result: Any = {"state": state, "messages": messages, "next_node": None}
            result = await callback_mgr.execute_after_invoke(context, input_data, result)
            next_node = None
            if isinstance(result, dict):
                state = result.get("state", state)
                messages = result.get("messages", messages)
                next_node = result.get("next_node")

            yield NodeResult(state, messages, next_node)
        except Exception as e:
            recovery = await self._recover(callback_mgr, context, input_data, event, e, config)
            if streaming:
                yield self._message_chunk(recovery, self.name, config)
            yield NodeResult(state, [recovery])

    async def _recover(
        self,
        callback_mgr: CallbackManager,
        context: CallbackContext,
        input_data: dict[str, Any],
        event: EventModel,
        error: Exception,
        config: dict[str, Any],
    ) -> Message:
        """The ``on_error`` recovery Message, or ``error`` re-raised when there is none.

        A failure with no recovery publishes the node ERROR (observability only: it ends the
        node span with the error; event consumers see ``run.failed``).
        """
        kind = "Agent" if isinstance(self.func, BaseAgent) else "Function"
        logger.warning(
            "Node '%s' execution failed, executing error callbacks: %s", self.name, error
        )
        recovery = await callback_mgr.execute_on_error(context, input_data, error)
        if not isinstance(recovery, Message):
            logger.error("Node '%s' could not recover from error", self.name)
            event.event_type = EventType.ERROR
            event.metadata["status"] = event.content = f"{kind} execution failed: {error}"
            event.data["error"] = str(error)
            event.data.update(error_details(error, config))
            event.content_type = [ContentType.ERROR, ContentType.STATE]
            publish_event(event)
            raise error
        logger.info("Node '%s' recovered from error using callback result", self.name)
        return recovery

    def _stamp_node_name(self, messages: list[Message]) -> None:
        """Record the producing node on assistant replies; existing values are kept."""
        for message in messages:
            if message.role == "assistant":
                message.metadata.setdefault("node_name", self.name)

    def _callback_context(self, config: dict[str, Any], function_name: str) -> CallbackContext:
        """The callback context of a function or agent node call."""
        return CallbackContext(
            invocation_type=InvocationType.AI,
            node_name=self.name,
            function_name=function_name,
            metadata={"config": public_config(config)},
        )

    def _error_event(
        self, config: dict[str, Any], state: AgentState, function_name: str
    ) -> EventModel:
        """The node ERROR event, published by ``_recover`` only when the node fails.

        The node's START and END come from the run loop (``GraphRunMixin``).
        """
        return EventModel.default(
            config,
            data={"state": state},
            event=Event.NODE_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=self.name,
            extra={"node": self.name, "function_name": function_name},
            tag="observability",
        )

    def _update_chunk(self, status: str, function_name: str, config: dict[str, Any]) -> StreamChunk:
        return StreamChunk(
            event=StreamEvent.UPDATES,
            data={"status": status, "tool_name": function_name, "node": self.name},
            thread_id=config.get("thread_id"),
            run_id=config.get("run_id"),
        )

    def _message_chunk(
        self, message: Any, function_name: str, config: dict[str, Any]
    ) -> StreamChunk:
        # One chunk per item: the stream is queued before it is read, so a shared chunk
        # would show every reader the last item.
        return StreamChunk(
            event=StreamEvent.MESSAGE,
            message=message,
            thread_id=config.get("thread_id"),
            run_id=config.get("run_id"),
            metadata={"node": self.name, "function_name": function_name},
        )

    def _prepare_input_data(self, state: AgentState, config: dict[str, Any]) -> dict:
        """The keyword arguments of the node function: ``state`` and ``config`` by name.

        A function with exactly one required parameter that is neither ``state`` nor
        ``config`` gets the state in it (``lambda s: s``). Any other required parameter
        raises ``TypeError``.
        """
        sig = self._signature_cache.get(self.func)
        if sig is None:
            sig = self._signature_cache[self.func] = inspect.signature(self.func)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        required = [p for p in params if p.default is inspect.Parameter.empty]
        if len(required) == 1 and required[0].name not in ("state", "config"):
            return {required[0].name: state}

        defaults = {"state": state, "config": config}
        input_data = {}
        for param in params:
            if param.name in defaults:
                input_data[param.name] = defaults[param.name]
            elif param.default is inspect.Parameter.empty:
                raise TypeError(
                    f"Missing required parameter '{param.name}' for function '{self.func}'"
                )
        return input_data
