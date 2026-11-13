"""OpenInference attributes, span timing, status and level gating on real spans."""

import asyncio
import json
import os

import pytest


pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("openai")

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from tenxgraph.core.graph import Agent, StateGraph
from tenxgraph.core.state import Message
from tenxgraph.runtime.publisher import _openinference as oi
from tenxgraph.runtime.publisher.otel_publisher import ObservabilityLevel, setup_tracing
from tenxgraph.utils.constants import END


SLEEP = 0.15


def test_content_text_variants():
    assert oi.content_text("hi") == "hi"
    assert oi.content_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert oi.content_text(None) == ""


def test_last_message_text_picks_role():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "q1"}]},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert oi.last_message_text(msgs, "user") == "q2"
    assert oi.last_message_text(msgs, "assistant") == "a1"
    assert oi.last_message_text(None, "user") == ""


def test_llm_request_messages_adds_system_once():
    out = oi.llm_request_messages([{"role": "user", "content": "x"}], "be brief")
    assert out[0] == {"role": "system", "content": "be brief"}
    again = oi.llm_request_messages([{"role": "system", "content": "s"}], "be brief")
    assert len(again) == 1


def test_llm_standard_and_output_attrs():
    attrs = oi.llm_standard_attrs({"input_tokens": 3, "output_tokens": 4}, "m", "openai")
    assert attrs[oi.TOKENS_TOTAL] == 7 and attrs[oi.LLM_MODEL_NAME] == "m"
    out = oi.llm_output_attrs(
        {"output_response": "", "output_tool_calls": [{"name": "f", "arguments": "{}"}]}
    )
    key = "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
    assert out[key] == "f" and out[oi.OUTPUT_MIME] == oi.JSON_MIME
    assert oi.llm_output_attrs({}) == {}


def test_output_tool_call_id():
    out = oi.llm_output_attrs(
        {"output_tool_calls": [{"id": "call_1", "name": "f", "arguments": "{}"}]}
    )
    assert out["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == "call_1"


def test_input_attrs_keep_tool_calls_and_call_id():
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]
    attrs = oi.llm_input_attrs(msgs)
    p = "llm.input_messages.1.message.tool_calls.0.tool_call"
    assert attrs[f"{p}.id"] == "c1"
    assert attrs[f"{p}.function.name"] == "f"
    assert attrs[f"{p}.function.arguments"] == '{"a":1}'
    assert attrs["llm.input_messages.2.message.tool_call_id"] == "c1"
    assert "tool_calls" in json.loads(attrs[oi.INPUT_VALUE])[1]


def test_tool_attrs():
    assert oi.tool_input_attrs({"a": 1})[oi.INPUT_VALUE] == '{"a":1}'
    msg = {"content": [{"type": "tool_result", "output": "42"}]}
    assert oi.tool_output_attrs({"message": msg})[oi.OUTPUT_VALUE] == "42"
    assert oi.tool_output_attrs({}) == {}


class _FakeAgent(Agent):
    async def _call_llm(self, messages, tools=None, stream=False, **kwargs):
        await asyncio.sleep(SLEEP)
        return ChatCompletion(
            id="c1",
            object="chat.completion",
            created=1,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content="the answer"),
                )
            ],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12),
        )


async def _spans(level):
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-not-used")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    graph = StateGraph()
    graph.add_node(
        "MAIN",
        _FakeAgent(
            model="openai/gpt-4o-mini",
            api_style="chat",
            system_prompt=[{"role": "system", "content": "be brief"}],
        ),
    )
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)
    setup_tracing(graph, tracer=provider.get_tracer("t"), level=level)
    app = graph.compile()
    await app.ainvoke(
        {"messages": [Message.text_message("what?")]},
        config={"thread_id": "t1", "user_id": "u1"},
    )
    await app.aclose()
    return {s.name: s for s in exporter.get_finished_spans()}


@pytest.mark.asyncio
async def test_full_level_timing_status_kinds_and_content():
    spans = await _spans(ObservabilityLevel.FULL)
    graph, llm = spans["invoke_agent 10xgraph"], spans["chat gpt-4o-mini"]
    assert (llm.end_time - llm.start_time) / 1e9 >= SLEEP
    for child in (s for n, s in spans.items() if n != "invoke_agent 10xgraph"):
        assert graph.start_time <= child.start_time and graph.end_time >= child.end_time
    for s in spans.values():
        assert s.status.status_code == StatusCode.OK
        assert s.attributes[oi.USER_ID] == "u1"
    assert graph.attributes[oi.SPAN_KIND] == "CHAIN"
    assert llm.attributes[oi.SPAN_KIND] == "LLM"
    assert graph.attributes[oi.INPUT_VALUE] == "what?"
    assert graph.attributes[oi.OUTPUT_VALUE] == "the answer"
    a = llm.attributes
    assert a["llm.input_messages.0.message.role"] == "system"
    assert a["llm.input_messages.1.message.content"] == "what?"
    assert a[oi.OUTPUT_VALUE] == "the answer"
    assert a["llm.output_messages.0.message.role"] == "assistant"
    assert a[oi.TOKENS_TOTAL] == 12 and a[oi.LLM_MODEL_NAME] == "gpt-4o-mini"
    assert json.loads(a["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "the answer"}],
            "finish_reason": "stop",
        }
    ]
    assert json.loads(a["gen_ai.input.messages"])[-1] == {
        "role": "user",
        "parts": [{"type": "text", "content": "what?"}],
    }
    assert a["langsmith.span.kind"] == "llm"
    assert graph.attributes["langsmith.span.kind"] == "chain"
    assert a["langsmith.metadata.thread_id"] == "t1"
    for s in spans.values():
        if s is not llm:
            assert not any(k.startswith("gen_ai.usage.") for k in s.attributes)


@pytest.mark.asyncio
async def test_spans_level_has_kind_and_user_but_no_content():
    spans = await _spans(ObservabilityLevel.SPANS)
    llm, graph = spans["chat gpt-4o-mini"], spans["invoke_agent 10xgraph"]
    assert llm.attributes[oi.SPAN_KIND] == "LLM" and llm.attributes[oi.USER_ID] == "u1"
    for s in (llm, graph):
        assert oi.INPUT_VALUE not in s.attributes and oi.OUTPUT_VALUE not in s.attributes
    assert oi.TOKENS_TOTAL not in llm.attributes


@pytest.mark.asyncio
async def test_standard_level_has_tokens_but_no_content():
    llm = (await _spans(ObservabilityLevel.STANDARD))["chat gpt-4o-mini"]
    assert llm.attributes[oi.TOKENS_TOTAL] == 12
    assert oi.INPUT_VALUE not in llm.attributes
    assert "llm.input_messages.0.message.role" not in llm.attributes
