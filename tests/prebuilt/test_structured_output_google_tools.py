"""StructuredOutputAgent on Google with tools: the two-step GENERATE + FORMAT graph.

Google cannot combine function calling with a JSON schema in one request, so the agent
splits the work. The LLM is mocked at ``Agent._call_google`` with Google-shaped mock
responses, so nodes, converters, routing and the checkpointer all run for both
``ainvoke`` and ``astream``. Nothing here calls a real provider.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pydantic
import pytest

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent.structured_output import (
    FORMAT_REQUEST_NOTE,
    StructuredOutputAgent,
)
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tests.adapters.test_google_genai_converter import (
    MockCandidate,
    MockContent,
    MockFunctionCall,
    MockGenerateContentResponse,
    MockPart,
)


MODES = ["ainvoke", "astream"]
FULL = ResponseGranularity.FULL
SYSTEM = [{"role": "system", "content": "You are a weather bot. Set needs_location when no city."}]


class WeatherReport(pydantic.BaseModel):
    city: str
    summary: str
    needs_location: bool = False


GOOD = json.dumps({"city": "Paris", "summary": "sunny", "needs_location": False})
BAD = json.dumps({"city": "Paris"})


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"sunny in {city}"


def _response(spec) -> MockGenerateContentResponse:
    """spec: str (text) or ("tool", name, args)."""
    if isinstance(spec, tuple):
        part = MockPart(function_call=MockFunctionCall(spec[1], spec[2]))
    else:
        part = MockPart(text=spec)
    return MockGenerateContentResponse(candidates=[MockCandidate(content=MockContent([part]))])


class Script:
    """Scripted Gemini: replies per node kind (schema set = FORMAT/REPAIR) and a request log."""

    def __init__(self, generate, fmt):
        self.replies = {"GENERATE": list(generate), "FORMAT": list(fmt)}
        self.requests: list[dict] = []

    async def __call__(self, agent, messages, tools=None, stream=False, **kwargs):
        kind = "FORMAT" if getattr(agent, "output_schema", None) is not None else "GENERATE"
        self.requests.append(
            {
                "kind": kind,
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "schema": getattr(agent, "output_schema", None),
            }
        )
        resp = _response(self.replies[kind].pop(0))
        if not stream:
            return resp

        async def gen():
            yield resp

        return gen()

    def of(self, kind):
        return [r for r in self.requests if r["kind"] == kind]


def _make(model="gemini-2.5-flash", provider="google", tools=(get_weather,), **kw):
    return StructuredOutputAgent(
        model=model,
        provider=provider,
        output_schema=WeatherReport,
        tools=list(tools) if tools else None,
        system_prompt=SYSTEM,
        **kw,
    )


async def run(script, mode, *, text="HI", thread="t", **kw):
    cp = InMemoryCheckpointer()
    with (
        patch.object(Agent, "_create_client", return_value=MagicMock()),
        patch.object(Agent, "_call_google", lambda self, *a, **k: script(self, *a, **k)),
    ):
        app = _make(**kw).compile(checkpointer=cp)
        config = {"thread_id": f"{thread}-{mode}", "user_id": "u"}
        data = {"messages": [Message.text_message(text)]}
        if mode == "ainvoke":
            await app.ainvoke(data, config, response_granularity=FULL)
        else:
            _ = [c async for c in app.astream(data, config, response_granularity=FULL)]
        return await cp.aget_state(config)


def _graph(**kw):
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        agent = _make(**kw)
        agent.compile()
    return agent._graph


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_google_with_tools_builds_generate_format_and_tool():
    graph = _graph()
    assert set(graph.nodes) >= {"GENERATE", "FORMAT", "TOOL", "REPAIR"}
    assert graph.nodes["GENERATE"].func.output_schema is None
    assert graph.nodes["GENERATE"].func.tool_node is not None
    # FORMAT is a wrapper around a schema agent with no tools
    assert callable(graph.nodes["FORMAT"].func)


def test_google_without_tools_keeps_single_generate_node():
    graph = _graph(tools=None)
    assert "FORMAT" not in graph.nodes
    assert "TOOL" not in graph.nodes
    assert graph.nodes["GENERATE"].func.output_schema is WeatherReport


def test_openai_with_tools_keeps_single_generate_node():
    graph = _graph(model="gpt-4o-mini", provider="openai")
    assert "FORMAT" not in graph.nodes
    assert "TOOL" in graph.nodes
    assert graph.nodes["GENERATE"].func.output_schema is WeatherReport


def test_provider_resolved_from_model_string():
    graph = _graph(model="gemini/gemini-2.5-flash", provider=None)
    assert "FORMAT" in graph.nodes


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_tool_loop_then_format_ends_with_valid_reply(mode):
    script = Script(
        generate=[("tool", "get_weather", {"city": "Paris"}), "It is sunny in Paris."],
        fmt=[GOOD],
    )
    state = await run(script, mode)

    assert [r["kind"] for r in script.requests] == ["GENERATE", "GENERATE", "FORMAT"]
    # GENERATE never carries a schema, FORMAT never carries tools
    assert all(r["schema"] is None for r in script.of("GENERATE"))
    assert all(r["tools"] for r in script.of("GENERATE"))
    assert script.of("FORMAT")[0]["schema"] is WeatherReport
    assert not script.of("FORMAT")[0]["tools"]

    last = state.context[-1]
    assert last.role == "assistant"
    assert WeatherReport.model_validate_json(last.text()).city == "Paris"
    assert last.metadata.get("structured_output_valid") is not False
    # GENERATE's plain-text answer stays in the saved context
    assert any(m.text() == "It is sunny in Paris." for m in state.context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_format_request_ends_with_user_turn_and_note_not_saved(mode):
    script = Script(generate=["Which city?"], fmt=[GOOD])
    state = await run(script, mode)

    request = script.of("FORMAT")[0]["messages"]
    assert request[-1]["role"] == "user"
    assert request[-1]["content"] == FORMAT_REQUEST_NOTE
    assert request[-2]["role"] == "assistant"
    # the user's system prompt reaches FORMAT too
    assert "needs_location" in str(request[0]["content"])
    assert FORMAT_REQUEST_NOTE not in " ".join(m.text() for m in state.context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_format_request_has_no_tool_calls_or_results(mode):
    script = Script(
        generate=[("tool", "get_weather", {"city": "Paris"}), "It is sunny in Paris."],
        fmt=[GOOD],
    )
    state = await run(script, mode)

    request = script.of("FORMAT")[0]["messages"]
    assert all(m["role"] != "tool" for m in request)
    assert not any(m.get("tool_calls") for m in request)
    assert "sunny in Paris." in str([m["content"] for m in request])
    # saved state still holds the tool turns
    assert any(m.role == "tool" for m in state.context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_invalid_format_reply_repairs_back_into_format(mode):
    script = Script(generate=["Paris is sunny."], fmt=[BAD, GOOD])
    state = await run(script, mode)

    # GENERATE is not run again after REPAIR
    assert [r["kind"] for r in script.requests] == ["GENERATE", "FORMAT", "FORMAT"]
    second = script.of("FORMAT")[1]["messages"]
    assert second[-1]["role"] == "user"
    assert "did not conform" in second[-1]["content"]
    assert FORMAT_REQUEST_NOTE not in [m["content"] for m in second]
    last = state.context[-1]
    assert WeatherReport.model_validate_json(last.text()).summary == "sunny"
    assert last.metadata.get("structured_output_valid") is not False


@pytest.mark.asyncio
async def test_out_of_attempts_ends_with_invalid_flag():
    script = Script(generate=["Paris is sunny."], fmt=[BAD, BAD, BAD])
    state = await run(script, "ainvoke", max_attempts=2)

    assert [r["kind"] for r in script.requests] == ["GENERATE", "FORMAT", "FORMAT", "FORMAT"]
    assert state.context[-1].metadata["structured_output_valid"] is False
