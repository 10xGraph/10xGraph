"""OtelPublisher — fire-and-forget OTEL backend for Agentflow.

Reads EventModel events from the publisher stream and reconstructs OTEL spans.
No synchronous work happens on the execution path; all span operations occur
inside the background task that BackgroundTaskManager already creates for
every publish_event call.

Parent-child relationships are tracked explicitly via SpanRegistry rather than
relying on contextvars (which do not transfer across asyncio.create_task boundaries).
Span timing is accurate because EventModel.timestamp is captured at execution time
and passed as start_time / end_time to the OTEL SDK.

Requires: pip install '10xscale-agentflow[otel]'
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .base_publisher import BasePublisher
from .events import Event, EventModel, EventType
from .otel_attributes import (
    GEN_AI_COMPLETION,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROMPT,
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
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    GRAPH_MODEL,
    GRAPH_RUN_ID,
    GRAPH_THREAD_ID,
    GRAPH_TOTAL_STEPS,
    GRAPH_USER_ID,
    LIFECYCLE,
    NODE_NAME,
    NODE_STEP,
    PROVIDER_NAME_MAP,
    SESSION_ID,
    TOOL_NAME,
    TOOL_TYPE,
)


if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

    from agentflow.core.graph.state_graph import StateGraph

logger = logging.getLogger("agentflow.otel")


class ObservabilityLevel(StrEnum):
    """Controls how much data OtelPublisher emits as OTEL span attributes/events.

    SPANS    — timing and structure only. No I/O data on spans.
    STANDARD — adds token counts, model name, request params when available. (default)
    FULL     — adds model input messages, output messages, tool I/O, system prompt.
               May contain PII — use only in controlled environments.
    """

    SPANS = "spans"
    STANDARD = "standard"
    FULL = "full"


def _guard() -> None:
    """Raise a helpful ImportError if opentelemetry-api is not installed."""
    try:
        import opentelemetry  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "OpenTelemetry is required for tracing. "
            "Install with: pip install '10xscale-agentflow[otel]'"
        ) from exc


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


class OtelPublisher(BasePublisher):
    """Publisher backend that maps EventModel events to OTEL spans.

    Usage (minimal — STANDARD level, global TracerProvider):
        from agentflow.runtime.publisher.otel_publisher import setup_tracing

        setup_tracing(graph)           # call before graph.compile()
        compiled = graph.compile(...)

    Usage (FULL level with explicit provider):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry import trace

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)

        setup_tracing(graph, level=ObservabilityLevel.FULL)
        compiled = graph.compile(...)
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        level: ObservabilityLevel = ObservabilityLevel.STANDARD,
    ) -> None:
        super().__init__({})
        self._tracer_arg = tracer
        self._tracer: Tracer | None = None
        self._registry = SpanRegistry()
        self._level = level

    def _get_tracer(self) -> Tracer:
        if self._tracer is None:
            from opentelemetry import trace

            self._tracer = self._tracer_arg or trace.get_tracer("agentflow")
        return self._tracer

    async def publish(self, event: EventModel) -> None:
        if self._is_closed:
            return
        try:
            self._dispatch(event)
        except Exception:
            logger.exception(
                "OtelPublisher failed to process event %s/%s", event.event, event.event_type
            )

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
            elif et == EventType.UPDATE:
                self._on_node_update(event)
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

    def _on_graph_start(self, event: EventModel) -> None:
        from opentelemetry.trace import StatusCode

        lifecycle = event.metadata.get("lifecycle", "")
        if lifecycle == "validation_rejected":
            span = self._get_tracer().start_span(
                "agentflow.graph",
                start_time=_ns(event.timestamp),
            )
            span.set_attribute(GEN_AI_SYSTEM, "agentflow")
            span.set_attribute(GEN_AI_OPERATION, "graph")
            _set_common_attrs(span, event)
            span.set_status(StatusCode.ERROR, "validation_rejected")
            span.add_event("graph.validation_rejected", attributes=_extra_attrs(event))
            span.end(end_time=_ns(event.timestamp))
            return

        span = self._get_tracer().start_span(
            "agentflow.graph",
            start_time=_ns(event.timestamp),
        )
        span.set_attribute(GEN_AI_SYSTEM, "agentflow")
        span.set_attribute(GEN_AI_OPERATION, "graph")
        _set_common_attrs(span, event)
        # Expose thread_id as session.id so Langfuse groups multi-turn conversations
        span.set_attribute(SESSION_ID, str(event.thread_id))
        self._registry.set_graph(event.run_id, span)

    def _on_graph_end(self, event: EventModel) -> None:
        span = self._registry.pop_graph(event.run_id)
        if span is None:
            return
        total_steps = event.data.get("total_steps") or event.metadata.get("total_steps")
        if total_steps is not None:
            span.set_attribute(GRAPH_TOTAL_STEPS, total_steps)
        span.end(end_time=_ns(event.timestamp))

    def _on_graph_error(self, event: EventModel) -> None:
        from opentelemetry.trace import StatusCode

        lifecycle = event.metadata.get("lifecycle", "")
        if lifecycle == "validation_rejected":
            return

        span = self._registry.pop_graph(event.run_id)
        if span is None:
            return
        error_msg = event.data.get("error", "")
        span.set_status(StatusCode.ERROR, error_msg)
        span.add_event("graph.error", attributes={"error": error_msg})
        span.end(end_time=_ns(event.timestamp))

    def _on_graph_interrupted(self, event: EventModel) -> None:
        span = self._registry.get_graph(event.run_id)
        if span is None:
            return
        attrs: dict[str, Any] = {"interrupted": True}
        interrupted_node = event.data.get("interrupted_node") or event.metadata.get(
            "interrupted_node", ""
        )
        if interrupted_node:
            attrs["interrupted_node"] = interrupted_node
        span.add_event("graph.interrupted", attributes=attrs)

    def _on_graph_update(self, event: EventModel) -> None:
        span = self._registry.get_graph(event.run_id)
        if span is None:
            return
        lifecycle = event.metadata.get("lifecycle", "")
        if lifecycle == "resume":
            resumed_node = event.data.get("resumed_node", "")
            span.add_event("graph.resumed", attributes={"resumed_node": resumed_node})
        elif lifecycle == "checkpoint":
            trimmed = event.metadata.get("trimmed", False)
            span.add_event("graph.checkpoint", attributes={"trimmed": trimmed})
        elif lifecycle == "graph_start":
            span.add_event("graph.lifecycle.start")
        elif lifecycle == "graph_end":
            total_steps = event.data.get("total_steps", 0)
            span.add_event("graph.lifecycle.end", attributes={"total_steps": total_steps})
        elif lifecycle == "graph_error":
            error_msg = event.data.get("error", "")
            span.add_event("graph.lifecycle.error", attributes={"error": error_msg})
        elif lifecycle == "graph_interrupt":
            interrupt_type = event.metadata.get("interrupt_type", "")
            span.add_event(
                "graph.lifecycle.interrupt", attributes={"interrupt_type": interrupt_type}
            )
        elif not lifecycle:
            # Post-node-execution state update: messages + state are present.
            self._enrich_from_node_output(span, event)

    def _enrich_from_node_output(self, span: Any, event: EventModel) -> None:
        """Extract token counts, model, system prompt, and output from a state-update event."""
        messages = event.data.get("messages") or []
        if not messages:
            return

        # Token counts come from the last assistant message's usages field.
        last_msg = messages[-1] if messages else None
        if last_msg and self._level != ObservabilityLevel.SPANS:
            usages = last_msg.get("usages") if isinstance(last_msg, dict) else None
            if usages:
                inp = usages.get("prompt_tokens") or 0
                out = usages.get("completion_tokens") or 0
                if inp:
                    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, int(inp))
                if out:
                    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, int(out))
            model = last_msg.get("model") if isinstance(last_msg, dict) else None
            if model:
                span.set_attribute(GRAPH_MODEL, str(model))

        if self._level == ObservabilityLevel.FULL:
            # Output messages from this node turn.
            span.add_event(
                "gen_ai.content.completion",
                attributes={GEN_AI_COMPLETION: _to_str(messages)},
            )
            # System prompt lives in state.context as role="system" messages.
            context = (event.data.get("state") or {}).get("context") or []
            system_msgs = [m for m in context if isinstance(m, dict) and m.get("role") == "system"]
            if system_msgs:
                span.add_event(
                    "gen_ai.content.system",
                    attributes={GEN_AI_PROMPT: _to_str(system_msgs)},
                )

    # ── Node span handlers ────────────────────────────────────────────────────

    def _on_node_start(self, event: EventModel) -> None:
        from opentelemetry import trace

        parent = self._registry.get_graph(event.run_id)
        ctx = trace.set_span_in_context(parent) if parent else None
        span = self._get_tracer().start_span(
            "agentflow.node",
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

        self._registry.set_child(event.run_id, event.node_name, span)

    def _on_node_end(self, event: EventModel) -> None:
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
                    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, int(inp))
                if out:
                    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, int(out))
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

        span.end(end_time=_ns(event.timestamp))

    def _on_node_error(self, event: EventModel) -> None:
        from opentelemetry.trace import StatusCode

        span = self._registry.pop_child(event.run_id, event.node_name)
        if span is None:
            return
        error_msg = event.data.get("error", "")
        span.set_status(StatusCode.ERROR, error_msg)
        span.add_event("node.error", attributes={"error": error_msg})
        span.end(end_time=_ns(event.timestamp))

    def _on_node_update(self, event: EventModel) -> None:
        span = self._registry.get_child(event.run_id, event.node_name)
        if span is None:
            return
        lifecycle = event.metadata.get("lifecycle", "")
        if lifecycle == "state_update":
            step = event.data.get("step", 0)
            span.add_event("node.state_update", attributes={"step": step})

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

        span = self._get_tracer().start_span(
            "agentflow.llm",
            context=ctx,
            start_time=_ns(event.timestamp),
        )

        # Standard gen_ai attributes
        if system_name:
            span.set_attribute(GEN_AI_SYSTEM, system_name)
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
                span.set_attribute(GEN_AI_INPUT_MESSAGES, _to_str(input_messages))
            system_prompt = event.data.get("system_prompt")
            if system_prompt:
                span.add_event(
                    "gen_ai.content.system",
                    attributes={"gen_ai.system_prompt": _to_str(system_prompt)},
                )

        # key: "llm:{node_name}" — allows multiple LLM calls within one node
        self._registry.set_child(event.run_id, f"llm:{event.node_name}", span)

    def _on_llm_end(self, event: EventModel) -> None:
        span = self._registry.pop_child(event.run_id, f"llm:{event.node_name}")
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
            if output_text:
                span.add_event(
                    "gen_ai.content.completion",
                    attributes={GEN_AI_COMPLETION: _to_str(output_text)},
                )
                span.set_attribute(GEN_AI_OUTPUT_MESSAGES, _to_str(output_text))

        span.end(end_time=_ns(event.timestamp))

    def _on_llm_error(self, event: EventModel) -> None:
        from opentelemetry.trace import StatusCode

        span = self._registry.pop_child(event.run_id, f"llm:{event.node_name}")
        if span is None:
            return
        error_msg = event.data.get("error", "")
        span.set_status(StatusCode.ERROR, error_msg)
        span.add_event("llm.error", attributes={"error": error_msg})
        span.end(end_time=_ns(event.timestamp))

    # ── Tool span handlers ────────────────────────────────────────────────────

    def _on_tool_start(self, event: EventModel) -> None:
        from opentelemetry import trace

        # Tools are invoked by the model, so prefer the LLM span as parent.
        # Fall back to node span, then graph span.
        parent = self._registry.get_child(event.run_id, f"llm:{event.node_name}")
        if parent is None:
            parent = self._registry.get_child(event.run_id, event.node_name)
        if parent is None:
            parent = self._registry.get_graph(event.run_id)
        ctx = trace.set_span_in_context(parent) if parent else None

        tool_name = (
            event.data.get("function_name")
            or event.data.get("tool_name")
            or event.metadata.get("tool_name", "")
        )
        is_mcp = event.metadata.get("is_mcp", False)
        tool_type = "mcp" if is_mcp else "local"
        span_key = f"tool:{tool_name}"

        span = self._get_tracer().start_span(
            "agentflow.tool",
            context=ctx,
            start_time=_ns(event.timestamp),
        )
        span.set_attribute(GEN_AI_OPERATION, "tool_call")
        span.set_attribute(TOOL_NAME, tool_name)
        span.set_attribute(TOOL_TYPE, tool_type)

        if self._level == ObservabilityLevel.FULL:
            tool_input = event.data.get("args") or event.data.get("tool_input")
            if tool_input:
                span.add_event(
                    "tool.input",
                    attributes={"tool.input": _to_str(tool_input)},
                )

        self._registry.set_child(event.run_id, span_key, span)

    def _on_tool_end(self, event: EventModel) -> None:
        tool_name = (
            event.data.get("function_name")
            or event.data.get("tool_name")
            or event.metadata.get("tool_name", "")
        )
        span_key = f"tool:{tool_name}"
        span = self._registry.pop_child(event.run_id, span_key)
        if span is None:
            return

        if self._level == ObservabilityLevel.FULL:
            tool_output = event.data.get("message") or event.data.get("tool_output")
            if tool_output:
                span.add_event(
                    "tool.output",
                    attributes={"tool.output": _to_str(tool_output)},
                )

        span.end(end_time=_ns(event.timestamp))

    def _on_tool_error(self, event: EventModel) -> None:
        from opentelemetry.trace import StatusCode

        tool_name = (
            event.data.get("function_name")
            or event.data.get("tool_name")
            or event.metadata.get("tool_name", "")
        )
        span_key = f"tool:{tool_name}"
        span = self._registry.pop_child(event.run_id, span_key)
        if span is None:
            return
        error_msg = event.data.get("error", "")
        span.set_status(StatusCode.ERROR, error_msg)
        span.add_event("tool.error", attributes={"error": error_msg})
        span.end(end_time=_ns(event.timestamp))

    # ── BasePublisher interface ───────────────────────────────────────────────

    async def close(self) -> None:
        self._is_closed = True
        for span in list(self._registry._graph.values()):
            with contextlib.suppress(Exception):
                span.end()
        self._registry._graph.clear()
        self._registry._children.clear()

    def sync_close(self) -> None:
        try:
            asyncio.run(self.close())
        except RuntimeError:
            self._is_closed = True


# ── Helpers ───────────────────────────────────────────────────────────────────


def _set_common_attrs(span: Span, event: EventModel) -> None:
    span.set_attribute(GRAPH_THREAD_ID, str(event.thread_id))
    span.set_attribute(GRAPH_RUN_ID, str(event.run_id))
    if event.user_id is not None:
        span.set_attribute(GRAPH_USER_ID, str(event.user_id))


def _extra_attrs(event: EventModel) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    lifecycle = event.metadata.get("lifecycle")
    if lifecycle:
        attrs[LIFECYCLE] = lifecycle
    return attrs


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


def setup_tracing(
    graph: StateGraph,
    tracer: Tracer | None = None,
    level: ObservabilityLevel = ObservabilityLevel.STANDARD,
) -> OtelPublisher:
    """Register OtelPublisher as the publisher backend on the graph.

    Must be called before graph.compile() so the publisher is bound in the
    DI container when the graph is compiled.

    Args:
        graph: The StateGraph instance to instrument.
        tracer: Optional explicit Tracer. Uses the global TracerProvider if None.
        level: How much data to emit. STANDARD by default. Use FULL for complete
               prompt/response visibility (may contain PII).

    Returns:
        The OtelPublisher instance.

    Raises:
        ImportError: If opentelemetry-api is not installed.
    """
    _guard()
    publisher = OtelPublisher(tracer=tracer, level=level)
    graph._publisher = publisher
    return publisher
