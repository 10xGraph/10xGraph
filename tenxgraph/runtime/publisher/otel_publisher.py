"""OtelPublisher — OTEL backend for 10xGraph.

Reads EventModel events from the publisher stream and reconstructs OTEL spans.

``publish_event`` hands events to this publisher on the caller's path (``inline``), not in
a background task: a span then exists as soon as its START is published, before the LLM
request or MCP call it covers starts, and events are processed in the order they happen, so
an END never runs before its START. Span work is in-memory and the background task would run
on the same event loop thread anyway. Execution code makes these spans current around
library calls (see ``trace_context``), so library spans nest under them.

Parent-child relationships between 10xGraph spans are tracked explicitly via SpanRegistry
rather than through the current context. Span timing is accurate because publish_event stamps
each event with the moment it is published (events are reused between phases) and that time
is passed as start_time / end_time to the OTEL SDK.

Requires: pip install '10xgraph[otel]'
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from tenxgraph.utils.constants import START

from . import _genai
from .base_publisher import BasePublisher
from .events import Event, EventModel, EventType
from .identity import UserIdMode, transform_user_id, validate_user_id_mode
from .otel_attributes import (
    GEN_AI_COMPLETION,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROMPT,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    GRAPH_MODEL,
    GRAPH_RUN_ID,
    GRAPH_RUN_STATUS,
    GRAPH_THREAD_ID,
    GRAPH_TOTAL_STEPS,
    GRAPH_USER_ID,
    NODE_NAME,
    NODE_STEP,
    PROVIDER_NAME_MAP,
    SPAN_UNFINISHED,
    TOOL_DENIED,
    TOOL_NAME,
    TOOL_REMOTE,
    TOOL_TYPE,
    USAGE_INPUT_TOKENS,
    USAGE_OUTPUT_TOKENS,
)


if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


logger = logging.getLogger("tenxgraph.otel")

# Span names follow the OTEL GenAI conventions, ``{gen_ai.operation.name} {target}``: the run
# root is ``invoke_agent 10xgraph``, an LLM call ``chat {model}``, a tool call
# ``execute_tool {tool}``, and a node span carries the node's name.
GRAPH_SPAN_NAME = "invoke_agent 10xgraph"

# The span a backend hook is called for, and when: once after a span starts, once before it ends.
SpanRole = Literal["graph", "node", "llm", "tool"]
SpanPhase = Literal["start", "end"]


class ObservabilityLevel(StrEnum):
    """Controls how much data OtelPublisher emits as OTEL span attributes/events.

    SPANS    — timing and structure only. No I/O data on spans.
    STANDARD — adds token counts, model name, request params when available; with
               ``PhoenixPublisher`` also the graph state on node spans (``input.value`` =
               state in, ``output.value`` = state after, messages produced, next node).
    FULL     — adds model input messages, output messages, tool I/O, system prompt.
               The default. Sends message content and graph state to the trace backend;
               use ``STANDARD`` or ``SPANS`` to keep content out.
    """

    SPANS = "spans"
    STANDARD = "standard"
    FULL = "full"


def _ns(ts: float) -> int:
    """Convert a UNIX float timestamp (seconds) to nanoseconds for the OTEL SDK."""
    return int(ts * 1e9)


def _to_str(value: Any) -> str:
    """Serialize a value to a compact JSON string for span event attributes."""
    import json

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _provider_system_name(raw_provider: str) -> str:
    """Map an agentflow provider string to the OTEL gen_ai.system value."""
    return PROVIDER_NAME_MAP.get(raw_provider.lower(), raw_provider) if raw_provider else ""


class SpanRegistry:
    """In-memory store that maps run IDs to open OTEL spans.

    Not thread-safe on its own; access is serialized by the single asyncio
    event loop that runs the background publish tasks.
    """

    def __init__(self) -> None:
        self._graph: dict[str, Span] = {}
        self._children: dict[tuple[str, str], Span] = {}
        # run.recovered arrives before the run's graph span starts; held until it does.
        self._recovered: dict[str, str] = {}

    def set_graph(self, run_id: str, span: Span) -> None:
        self._graph[run_id] = span

    def get_graph(self, run_id: str) -> Span | None:
        return self._graph.get(run_id)

    def pop_graph(self, run_id: str) -> Span | None:
        return self._graph.pop(run_id, None)

    def set_child(self, run_id: str, key: str, span: Span) -> None:
        self._children[(run_id, key)] = span

    def get_child(self, run_id: str, key: str) -> Span | None:
        return self._children.get((run_id, key))

    def pop_child(self, run_id: str, key: str) -> Span | None:
        return self._children.pop((run_id, key), None)

    def pop_children(self, run_id: str) -> list[Span]:
        """Remove and return every open child span of a run."""
        keys = [k for k in self._children if k[0] == run_id]
        return [self._children.pop(k) for k in keys]


def llm_span_key(node_name: str) -> str:
    """Registry key of the LLM span of a node (one LLM call is open per node at a time)."""
    return f"llm:{node_name}"


def tool_span_key(tool_call_id: str) -> str:
    """Registry key of a tool span: the call id, so parallel calls of one tool do not collide."""
    return f"tool:{tool_call_id}"


class OtelPublisher(BasePublisher):
    """Publisher backend that maps EventModel events to OTEL spans.

    Writes the OTEL GenAI conventions (``gen_ai.*``) and ``tenxgraph.*`` attributes only.
    The keys a trace backend reads (OpenInference for Phoenix, ``langsmith.*``,
    ``logfire.*``) come from ``span_attributes``, which ``PhoenixPublisher``,
    ``LangsmithPublisher`` and ``LogfirePublisher`` override.

    Usage (minimal — FULL level, global TracerProvider):
        from tenxgraph.core.graph import StateGraph
        from tenxgraph.runtime.publisher import OtelPublisher

        graph = StateGraph(publisher=OtelPublisher())
        compiled = graph.compile(...)

    Usage (FULL level with explicit provider):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry import trace

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)

        graph = StateGraph(publisher=OtelPublisher(level=ObservabilityLevel.FULL))
        compiled = graph.compile(...)
    """

    kind = "observability"
    inline = True

    def __init__(
        self,
        tracer: Tracer | None = None,
        level: ObservabilityLevel = ObservabilityLevel.FULL,
        user_id_mode: UserIdMode = "raw",
    ) -> None:
        """Create the publisher.

        Args:
            tracer: Optional explicit Tracer. Uses the global TracerProvider if None.
            level: How much data to emit.
            user_id_mode: How ``user_id`` appears on spans: ``"raw"`` (default, unchanged),
                ``"hash"`` (16 hex chars of SHA-256, or HMAC-SHA256 when the
                ``TENXGRAPH_TRACE_USER_ID_KEY`` env var is set; plain SHA-256 can be
                guessed for short ids) or ``"mask"`` (e.g. ``"al***42"``). See
                ``tenxgraph.runtime.publisher.identity``.

        Raises:
            ValueError: If ``user_id_mode`` is not one of the allowed values.
        """
        super().__init__({})
        self._user_id_mode: UserIdMode = validate_user_id_mode(user_id_mode)
        self._tracer_arg = tracer
        self._tracer: Tracer | None = None
        self._registry = SpanRegistry()
        self._level = level

    @property
    def level(self) -> ObservabilityLevel:
        """How much data this publisher puts on spans."""
        return self._level

    @property
    def user_id_mode(self) -> UserIdMode:
        """How ``user_id`` appears on spans (``raw``, ``hash`` or ``mask``)."""
        return self._user_id_mode

    def span_attributes(
        self,
        role: SpanRole,
        phase: SpanPhase,
        name: str,
        event: EventModel,
        attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Backend-specific attributes for one span. The base adds none.

        Called once after a span starts (``phase="start"``) and once before it ends
        (``phase="end"``), for ``role`` ``graph``, ``node``, ``llm`` or ``tool``. ``name`` is
        the span name and ``event`` the START or END event. Return ``attrs`` with the
        backend's keys added; they are set on the span. A subclass for a trace backend
        (``PhoenixPublisher``, ``LangsmithPublisher``, ``LogfirePublisher``) overrides this,
        and must honour ``self.level`` for content and ``self.user_id_mode`` for user ids.
        Not called for an error end, a remote tool's end, or spans ended as unfinished.
        """
        return attrs

    def _set_backend_attrs(
        self, span: Span, role: SpanRole, phase: SpanPhase, name: str, event: EventModel
    ) -> None:
        _set_attrs(span, self.span_attributes(role, phase, name, event, {}))

    def _get_tracer(self) -> Tracer:
        if self._tracer is None:
            from opentelemetry import trace

            self._tracer = self._tracer_arg or trace.get_tracer("10xgraph")
        return self._tracer

    async def publish(self, event: EventModel) -> None:
        self.publish_inline(event)

    def publish_inline(self, event: EventModel) -> None:
        """Turn one event into span work now. Never raises: a failure is logged."""
        if self._is_closed:
            return
        try:
            self._dispatch(event)
        except Exception:
            logger.exception(
                "OtelPublisher failed to process event %s/%s", event.event, event.event_type
            )

    def open_span(self, run_id: str, key: str) -> Span | None:
        """The open span of a run under a registry key (node name, ``llm_span_key``,
        ``tool_span_key``), or None."""
        return self._registry.get_child(run_id, key)

    def _dispatch(self, event: EventModel) -> None:
        ev = event.event
        et = event.event_type
        if ev == Event.GRAPH_EXECUTION:
            if et == EventType.START:
                self._on_graph_start(event)
            elif et == EventType.END:
                self._on_graph_end(event)
            elif et == EventType.ERROR:
                self._on_graph_error(event)
            elif et == EventType.INTERRUPTED:
                self._on_graph_interrupted(event)
            elif et == EventType.UPDATE:
                self._on_graph_update(event)
        elif ev == Event.NODE_EXECUTION:
            if et == EventType.START:
                self._on_node_start(event)
            elif et == EventType.END:
                self._on_node_end(event)
            elif et == EventType.ERROR:
                self._on_node_error(event)
        elif ev == Event.LLM_CALL:
            if et == EventType.START:
                self._on_llm_start(event)
            elif et == EventType.END:
                self._on_llm_end(event)
            elif et == EventType.ERROR:
                self._on_llm_error(event)
        elif ev == Event.TOOL_EXECUTION:
            if et == EventType.START:
                self._on_tool_start(event)
            elif et == EventType.END:
                self._on_tool_end(event)
            elif et == EventType.ERROR:
                self._on_tool_error(event)

    # ── Graph span handlers ───────────────────────────────────────────────────

    def _start_graph_span(self, event: EventModel) -> Span:
        span = self._get_tracer().start_span(GRAPH_SPAN_NAME, start_time=_ns(event.timestamp))
        span.set_attribute(GEN_AI_SYSTEM, "10xgraph")
        span.set_attribute(GEN_AI_OPERATION, "invoke_agent")
        _set_common_attrs(span, event, self._user_id_mode)
        self._set_backend_attrs(span, "graph", "start", GRAPH_SPAN_NAME, event)
        return span

    def _on_graph_start(self, event: EventModel) -> None:
        span = self._start_graph_span(event)
        dead_run = self._registry._recovered.pop(event.run_id, None)
        if dead_run is not None:
            span.add_event("graph.recovered", attributes={"dead_run_id": dead_run})
        self._registry.set_graph(event.run_id, span)

    def _on_graph_end(self, event: EventModel) -> None:
        """``run.completed``: run totals on the graph span, then end it."""
        from opentelemetry.trace import StatusCode

        span = self._registry.pop_graph(event.run_id)
        if span is None:
            return
        self._end_unfinished(event)
        total_steps = event.data.get("total_steps") or event.metadata.get("total_steps")
        if total_steps is not None:
            span.set_attribute(GRAPH_TOTAL_STEPS, total_steps)
        if self._level != ObservabilityLevel.SPANS:
            inp = event.metadata.get("total_input_tokens") or 0
            out = event.metadata.get("total_output_tokens") or 0
            if inp:
                span.set_attribute(USAGE_INPUT_TOKENS, int(inp))
            if out:
                span.set_attribute(USAGE_OUTPUT_TOKENS, int(out))
        if self._level == ObservabilityLevel.FULL:
            self._add_run_content(span, event)
        self._set_backend_attrs(span, "graph", "end", GRAPH_SPAN_NAME, event)
        span.set_attribute(GRAPH_RUN_STATUS, "completed")
        span.set_status(StatusCode.OK)
        span.end(end_time=_ns(event.timestamp))

    def _add_run_content(self, span: Any, event: EventModel) -> None:
        """The run's messages and system prompt as span events (FULL level)."""
        messages = event.data.get("messages") or []
        if messages:
            span.add_event(
                "gen_ai.content.completion", attributes={GEN_AI_COMPLETION: _to_str(messages)}
            )
        # The system prompt lives in state.context as role="system" messages.
        context = (event.data.get("state") or {}).get("context") or []
        system_msgs = [m for m in context if isinstance(m, dict) and m.get("role") == "system"]
        if system_msgs:
            span.add_event(
                "gen_ai.content.system", attributes={GEN_AI_PROMPT: _to_str(system_msgs)}
            )

    def _on_graph_error(self, event: EventModel) -> None:
        """``run.failed`` ends the graph span with the error. ``input.rejected`` comes before
        any graph span (the input is checked while the thread loads), so it is a span of its
        own, started and ended at once."""
        from opentelemetry.trace import StatusCode

        error_msg = str(event.data.get("error", ""))
        if event.metadata.get("event_name") == "input.rejected":
            rejected = self._start_graph_span(event)
            rejected.set_attribute(GRAPH_RUN_STATUS, "input_rejected")
            rejected.set_status(StatusCode.ERROR, "input_rejected")
            rejected.add_event("graph.validation_rejected", attributes={"error": error_msg})
            rejected.end(end_time=_ns(event.timestamp))
            return

        span = self._registry.pop_graph(event.run_id)
        if span is None:
            return
        self._end_unfinished(event)
        span.set_attribute(GRAPH_RUN_STATUS, "failed")
        span.set_status(StatusCode.ERROR, error_msg)
        span.add_event("graph.error", attributes={"error": error_msg})
        span.end(end_time=_ns(event.timestamp))

    def _end_unfinished(self, event: EventModel, *, cancelled: bool = False) -> None:
        """End child spans of a finished run that never got an END (interrupt, skipped END).

        With ``cancelled`` (``run.stopped``) they end as ERROR "cancelled", the same status a
        cancelled tool span gets, so a trace shows the cut.
        """
        from opentelemetry.trace import StatusCode

        for child in self._registry.pop_children(event.run_id):
            child.set_attribute(SPAN_UNFINISHED, True)
            if cancelled:
                child.set_status(StatusCode.ERROR, "cancelled")
            child.end(end_time=_ns(event.timestamp))

    def _on_graph_interrupted(self, event: EventModel) -> None:
        """``run.interrupted`` and ``run.stopped`` end the graph span (the run is over until
        a resume, which is a new run). ``run.recovered`` is a span event on the new run."""
        from opentelemetry.trace import StatusCode

        name = event.metadata.get("event_name")
        if name == "run.recovered":
            dead_run = str(event.data.get("dead_run_id") or "")
            span = self._registry.get_graph(event.run_id)
            if span is None:
                self._registry._recovered[event.run_id] = dead_run
            else:
                span.add_event("graph.recovered", attributes={"dead_run_id": dead_run})
            return
        span = self._registry.pop_graph(event.run_id)
        if span is None:
            return
        self._end_unfinished(event, cancelled=name == "run.stopped")
        status = "stopped" if name == "run.stopped" else "interrupted"
        attrs: dict[str, Any] = {"interrupted": True}
        for key in ("interrupted_node", "interrupt_type", "reason"):
            value = event.data.get(key) or event.metadata.get(key)
            if value:
                attrs[key] = str(value)
        span.add_event(f"graph.{status}", attributes=attrs)
        span.set_attribute(GRAPH_RUN_STATUS, status)
        span.set_status(StatusCode.OK)
        span.end(end_time=_ns(event.timestamp))

    def _on_graph_update(self, event: EventModel) -> None:
        span = self._registry.get_graph(event.run_id)
        if span is None:
            return
        if event.metadata.get("lifecycle") == "resume":
            resumed_node = event.data.get("resumed_node", "")
            span.add_event("graph.resumed", attributes={"resumed_node": resumed_node})

    # ── Node span handlers ────────────────────────────────────────────────────

    def _on_node_start(self, event: EventModel) -> None:
        from opentelemetry import trace

        if event.node_name == START:
            # The start node does no work; its span would be empty in every trace.
            return
        parent = self._registry.get_graph(event.run_id)
        ctx = trace.set_span_in_context(parent) if parent else None
        span = self._get_tracer().start_span(
            event.node_name,
            context=ctx,
            start_time=_ns(event.timestamp),
        )
        span.set_attribute(GEN_AI_OPERATION, "node")
        span.set_attribute(NODE_NAME, event.node_name)
        step = event.data.get("step")
        if step is not None:
            span.set_attribute(NODE_STEP, step)

        if self._level == ObservabilityLevel.FULL:
            input_messages = event.data.get("input_messages") or event.data.get("messages")
            if input_messages:
                span.add_event(
                    "gen_ai.content.prompt",
                    attributes={GEN_AI_PROMPT: _to_str(input_messages)},
                )
            system_prompt = event.data.get("system_prompt")
            if system_prompt:
                span.add_event(
                    "gen_ai.content.system",
                    attributes={"gen_ai.system_prompt": _to_str(system_prompt)},
                )

        self._set_backend_attrs(span, "node", "start", event.node_name, event)
        self._registry.set_child(event.run_id, event.node_name, span)

    def _on_node_end(self, event: EventModel) -> None:
        from opentelemetry.trace import StatusCode

        span = self._registry.pop_child(event.run_id, event.node_name)
        if span is None:
            return

        if self._level != ObservabilityLevel.SPANS:
            messages = event.data.get("messages") or []
            last_msg = messages[-1] if messages else None
            if isinstance(last_msg, dict):
                usages = last_msg.get("usages") or {}
                inp = usages.get("prompt_tokens") or 0
                out = usages.get("completion_tokens") or 0
                if inp:
                    span.set_attribute(USAGE_INPUT_TOKENS, int(inp))
                if out:
                    span.set_attribute(USAGE_OUTPUT_TOKENS, int(out))
            model = event.metadata.get("model")
            if model:
                span.set_attribute(GRAPH_MODEL, str(model))

        if self._level == ObservabilityLevel.FULL:
            output_messages = event.data.get("messages")
            if output_messages:
                span.add_event(
                    "gen_ai.content.completion",
                    attributes={GEN_AI_COMPLETION: _to_str(output_messages)},
                )

        self._set_backend_attrs(span, "node", "end", event.node_name, event)
        span.set_status(StatusCode.OK)
        span.end(end_time=_ns(event.timestamp))

    def _on_node_error(self, event: EventModel) -> None:
        self._end_with_error(event, event.node_name, "node.error")

    def _end_with_error(self, event: EventModel, key: str, name: str) -> None:
        """End the open child span ``key`` of the run as failed, with the ``<name>`` event."""
        span = self._registry.pop_child(event.run_id, key)
        if span is None:
            return
        _record_error(span, event, name)
        span.end(end_time=_ns(event.timestamp))

    # ── LLM span handlers ────────────────────────────────────────────────────

    def _on_llm_start(self, event: EventModel) -> None:
        from opentelemetry import trace

        parent = self._registry.get_child(event.run_id, event.node_name)
        if parent is None:
            parent = self._registry.get_graph(event.run_id)
        ctx = trace.set_span_in_context(parent) if parent else None

        model = event.data.get("model", "")
        provider = event.data.get("provider", "")
        system_name = _provider_system_name(provider)

        name = f"chat {model}" if model else "chat"
        span = self._get_tracer().start_span(name, context=ctx, start_time=_ns(event.timestamp))
        # Standard gen_ai attributes
        if system_name:
            span.set_attribute(GEN_AI_SYSTEM, system_name)
            span.set_attribute(GEN_AI_PROVIDER_NAME, system_name)
        span.set_attribute(GEN_AI_OPERATION, "chat")
        if model:
            span.set_attribute(GEN_AI_REQUEST_MODEL, model)
            span.set_attribute(GRAPH_MODEL, model)  # keep agentflow-namespaced copy

        if self._level != ObservabilityLevel.SPANS:
            # Request parameters at STANDARD level
            params = event.data.get("request_params") or {}
            _set_request_params(span, params)

            tool_count = event.data.get("tool_count", 0)
            if tool_count:
                span.set_attribute("gen_ai.request.tool_count", int(tool_count))

        if self._level == ObservabilityLevel.FULL:
            input_messages = event.data.get("input_messages")
            if input_messages:
                # Emit both the legacy event and the modern attribute form
                span.add_event(
                    "gen_ai.content.prompt",
                    attributes={GEN_AI_PROMPT: _to_str(input_messages)},
                )
                semconv_in = _genai.input_messages(input_messages, event.data.get("system_prompt"))
                span.set_attribute(GEN_AI_INPUT_MESSAGES, _to_str(semconv_in))
            system_prompt = event.data.get("system_prompt")
            if system_prompt:
                span.add_event(
                    "gen_ai.content.system",
                    attributes={"gen_ai.system_prompt": _to_str(system_prompt)},
                )

        self._set_backend_attrs(span, "llm", "start", name, event)
        # key: "llm:{node_name}" — allows multiple LLM calls within one node
        self._registry.set_child(event.run_id, llm_span_key(event.node_name), span)

    def _on_llm_end(self, event: EventModel) -> None:
        from opentelemetry.trace import StatusCode

        span = self._registry.pop_child(event.run_id, llm_span_key(event.node_name))
        if span is None:
            return

        if self._level != ObservabilityLevel.SPANS:
            inp = event.data.get("input_tokens") or 0
            out = event.data.get("output_tokens") or 0
            if inp:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, int(inp))
            if out:
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, int(out))

            cache_read = event.data.get("cache_read_tokens") or 0
            if cache_read:
                span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, int(cache_read))

            cache_creation = event.data.get("cache_creation_tokens") or 0
            if cache_creation:
                span.set_attribute(GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, int(cache_creation))

            reasoning = event.data.get("reasoning_tokens") or 0
            if reasoning:
                span.set_attribute(GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, int(reasoning))

            finish_reason = event.data.get("finish_reason") or ""
            if finish_reason:
                # semconv expects an array; wrap single reason for consistency
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])

            response_id = event.data.get("response_id") or ""
            if response_id:
                span.set_attribute(GEN_AI_RESPONSE_ID, response_id)

            response_model = event.data.get("response_model") or ""
            if response_model:
                span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)

        if self._level == ObservabilityLevel.FULL:
            output_text = event.data.get("output_response")
            output_messages = _genai.output_messages(event.data)
            if output_text:
                span.add_event(
                    "gen_ai.content.completion",
                    attributes={GEN_AI_COMPLETION: _to_str(output_text)},
                )
            if output_messages:
                span.set_attribute(GEN_AI_OUTPUT_MESSAGES, _to_str(output_messages))

        self._set_backend_attrs(span, "llm", "end", _span_name(span), event)
        span.set_status(StatusCode.OK)
        span.end(end_time=_ns(event.timestamp))
        # The run's model on the graph span: the last model that answered.
        graph = self._registry.get_graph(event.run_id)
        model = event.data.get("response_model") or event.data.get("model")
        if graph is not None and model and self._level != ObservabilityLevel.SPANS:
            graph.set_attribute(GRAPH_MODEL, str(model))

    def _on_llm_error(self, event: EventModel) -> None:
        self._end_with_error(event, llm_span_key(event.node_name), "llm.error")

    # ── Tool span handlers ────────────────────────────────────────────────────

    def _on_tool_start(self, event: EventModel) -> None:
        from opentelemetry import trace

        # Parent is the span of the graph node running the tool, else the graph span.
        parent = self._registry.get_child(event.run_id, event.metadata.get("graph_node", ""))
        if parent is None:
            parent = self._registry.get_graph(event.run_id)
        ctx = trace.set_span_in_context(parent) if parent else None

        tool_name = _tool_name(event)
        is_mcp = event.metadata.get("is_mcp", False)
        tool_type = "mcp" if is_mcp else "local"

        name = f"execute_tool {tool_name}" if tool_name else "execute_tool"
        span = self._get_tracer().start_span(name, context=ctx, start_time=_ns(event.timestamp))
        span.set_attribute(GEN_AI_OPERATION, "execute_tool")
        span.set_attribute(GEN_AI_TOOL_NAME, tool_name)
        # The model generates the arguments of a predefined function, local or MCP alike.
        span.set_attribute(GEN_AI_TOOL_TYPE, "function")
        tool_call_id = event.data.get("tool_call_id")
        if tool_call_id:
            span.set_attribute(GEN_AI_TOOL_CALL_ID, str(tool_call_id))
        span.set_attribute(TOOL_NAME, tool_name)
        span.set_attribute(TOOL_TYPE, tool_type)

        if self._level == ObservabilityLevel.FULL:
            tool_input = event.data.get("args")
            if tool_input is None:
                tool_input = event.data.get("tool_input")
            if tool_input is not None:
                span.add_event(
                    "tool.input",
                    attributes={"tool.input": _to_str(tool_input)},
                )

        self._set_backend_attrs(span, "tool", "start", name, event)
        if event.metadata.get("is_remote"):
            # The client runs this tool and no END ever arrives, so the span ends now.
            span.set_attribute(TOOL_REMOTE, True)
            span.end(end_time=_ns(event.timestamp))
            return
        self._registry.set_child(event.run_id, _tool_key(event), span)

    def _on_tool_end(self, event: EventModel) -> None:
        from opentelemetry.trace import StatusCode

        span = self._registry.pop_child(event.run_id, _tool_key(event))
        if span is None:
            return

        if self._level == ObservabilityLevel.FULL:
            tool_output = event.data.get("message") or event.data.get("tool_output")
            if tool_output:
                span.add_event(
                    "tool.output",
                    attributes={"tool.output": _to_str(tool_output)},
                )

        self._set_backend_attrs(span, "tool", "end", _span_name(span), event)
        if event.metadata.get("denied"):
            span.set_attribute(TOOL_DENIED, True)
        if event.data.get("error"):
            _record_error(span, event, "tool.error")
        else:
            span.set_status(StatusCode.OK)
        span.end(end_time=_ns(event.timestamp))

    def _on_tool_error(self, event: EventModel) -> None:
        self._end_with_error(event, _tool_key(event), "tool.error")

    # ── BasePublisher interface ───────────────────────────────────────────────

    async def close(self) -> None:
        self._is_closed = True
        for span in list(self._registry._graph.values()):
            with contextlib.suppress(Exception):
                span.end()
        self._registry._graph.clear()
        self._registry._children.clear()
        self._registry._recovered.clear()

    def sync_close(self) -> None:
        try:
            asyncio.run(self.close())
        except RuntimeError:
            self._is_closed = True


# ── Helpers ───────────────────────────────────────────────────────────────────


def _set_common_attrs(span: Span, event: EventModel, user_id_mode: UserIdMode = "raw") -> None:
    span.set_attribute(GRAPH_THREAD_ID, str(event.thread_id))
    span.set_attribute(GRAPH_RUN_ID, str(event.run_id))
    if event.user_id is not None:
        span.set_attribute(GRAPH_USER_ID, transform_user_id(event.user_id, user_id_mode))


def _tool_name(event: EventModel) -> str:
    return (
        event.data.get("function_name")
        or event.data.get("tool_name")
        or event.metadata.get("tool_name", "")
    )


def _tool_key(event: EventModel) -> str:
    return tool_span_key(event.data.get("tool_call_id") or _tool_name(event))


def _record_error(span: Span, event: EventModel, name: str) -> None:
    """Error status, the ``<name>`` event, and the OTEL ``exception`` event when the event
    carries the exception (``error_type``, ``error_message``, ``error_traceback``)."""
    from opentelemetry.trace import StatusCode

    error_msg = str(event.data.get("error") or "")
    span.set_status(StatusCode.ERROR, error_msg)
    span.add_event(name, attributes={"error": error_msg})
    error_type = event.data.get("error_type")
    if not error_type:
        return
    attrs = {
        "exception.type": str(error_type),
        "exception.message": str(event.data.get("error_message") or error_msg),
    }
    if event.data.get("error_traceback"):
        attrs["exception.stacktrace"] = str(event.data["error_traceback"])
    span.add_event("exception", attributes=attrs)


def _span_name(span: Span) -> str:
    """The name of an SDK span; ``""`` for a span type that has none."""
    name = getattr(span, "name", "")
    return name if isinstance(name, str) else ""


def _set_attrs(span: Span, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        span.set_attribute(key, value)


def _set_request_params(span: Span, params: dict[str, Any]) -> None:
    """Write LLM request parameters onto a span at STANDARD+ level."""
    _PARAM_ATTR_MAP: dict[str, tuple[str, type]] = {
        "temperature": (GEN_AI_REQUEST_TEMPERATURE, float),
        "max_tokens": (GEN_AI_REQUEST_MAX_TOKENS, int),
        "top_p": (GEN_AI_REQUEST_TOP_P, float),
        "top_k": (GEN_AI_REQUEST_TOP_K, int),
        "frequency_penalty": (GEN_AI_REQUEST_FREQUENCY_PENALTY, float),
        "presence_penalty": (GEN_AI_REQUEST_PRESENCE_PENALTY, float),
        "seed": (GEN_AI_REQUEST_SEED, int),
        "stop": (GEN_AI_REQUEST_STOP_SEQUENCES, str),
    }
    for key, (attr_name, cast_type) in _PARAM_ATTR_MAP.items():
        val = params.get(key)
        if val is None:
            continue
        try:
            if cast_type is str:
                # stop can be str or list[str]; normalise to a single string
                span.set_attribute(attr_name, _to_str(val))
            else:
                span.set_attribute(attr_name, cast_type(val))
        except (TypeError, ValueError):
            pass
