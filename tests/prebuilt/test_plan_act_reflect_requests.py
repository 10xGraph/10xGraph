"""PlanActReflect request shape, [DONE] handling, empty replies and node labels.

The LLM is mocked at ``Agent._call_llm`` with real OpenAI response types, so the whole
graph (nodes, converters, routing, checkpointer) runs for both ``ainvoke`` and ``astream``.
"""

from __future__ import annotations

import itertools
import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent.plan_act_reflect import (
    PLAN_CONTINUE_NOTE,
    REFLECT_REQUEST_NOTE,
    PlanActReflectAgent,
)
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import END, ResponseGranularity


_ids = itertools.count()
MODES = ["ainvoke", "astream"]
FULL = ResponseGranularity.FULL


def _completion(spec) -> ChatCompletion:
    """spec: str (text, "" for empty) or ("tool", name, args)."""
    if isinstance(spec, tuple):
        _, name, args = spec
        message = ChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(name=name, arguments=json.dumps(args)),
                )
            ],
        )
        finish = "tool_calls"
    else:
        message = ChatCompletionMessage(role="assistant", content=spec or None)
        finish = "stop"
    return ChatCompletion(
        id=f"cmpl-{next(_ids)}",
        created=1,
        model="gpt-4o",
        object="chat.completion",
        choices=[Choice(index=0, finish_reason=finish, message=message)],
    )


async def _chunks(spec):
    if isinstance(spec, tuple):
        _, name, args = spec
        delta = ChoiceDelta(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="call_1",
                    type="function",
                    function=ChoiceDeltaToolCallFunction(name=name, arguments=json.dumps(args)),
                )
            ]
        )
        finish = "tool_calls"
    else:
        delta = ChoiceDelta(content=spec or None)
        finish = "stop"
    yield ChatCompletionChunk(
        id=f"chunk-{next(_ids)}",
        created=1,
        model="gpt-4o",
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish)],
    )


class Script:
    """Scripted LLM: one reply list per node kind, and a log of every request."""

    def __init__(self, plan, reflect):
        self.replies = {"PLAN": list(plan), "REFLECT": list(reflect)}
        self.requests: list[tuple[str, list[dict]]] = []

    async def __call__(self, agent, messages, tools=None, stream=False, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        kind = "REFLECT" if "critical evaluator" in system else "PLAN"
        self.requests.append((kind, [dict(m) for m in messages]))
        spec = self.replies[kind].pop(0)
        return _chunks(spec) if stream else _completion(spec)

    def of(self, kind):
        return [msgs for k, msgs in self.requests if k == kind]


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"sunny in {city}"


async def run(script, mode, *, tools=None, max_iterations=3, text="HI", thread="t", cp_holder=None):
    cp = InMemoryCheckpointer()
    with (
        patch.object(Agent, "_create_client", return_value=MagicMock()),
        patch.object(Agent, "_call_llm", lambda self, *a, **k: script(self, *a, **k)),
    ):
        agent = PlanActReflectAgent(
            model="gpt-4o",
            provider="openai",
            api_style="chat",
            tools=tools,
            max_iterations=max_iterations,
        )
        app = agent.compile(checkpointer=cp)
        config = {"thread_id": f"{thread}-{mode}", "user_id": "u"}
        if cp_holder is not None:
            cp_holder.update(cp=cp, config=config)
        data = {"messages": [Message.text_message(text)]}
        if mode == "ainvoke":
            await app.ainvoke(data, config, response_granularity=FULL)
        else:
            _ = [c async for c in app.astream(data, config, response_granularity=FULL)]
        return await cp.aget_state(config)


def texts(state):
    return [(m.role, m.text()) for m in state.context]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_hi_scenario_ends_after_two_llm_calls_and_drops_token(mode):
    script = Script(
        plan=["I can help with the weather. Which city?"],
        reflect=["I can help with the weather. Which city? [DONE]"],
    )
    state = await run(script, mode)

    assert len(script.requests) == 2
    reflect_msg = [m for m in state.context if m.metadata.get("node_name") == "REFLECT"]
    assert len(reflect_msg) == 1
    assert "[done]" not in reflect_msg[0].text().lower()
    assert reflect_msg[0].text() == "I can help with the weather. Which city?"
    assert not any("[done]" in m.text().lower() for m in state.context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_reflector_request_ends_with_user_turn_and_note_is_not_saved(mode):
    script = Script(plan=["plan answer"], reflect=["fine [DONE]"])
    state = await run(script, mode)

    reflect_request = script.of("REFLECT")[0]
    assert reflect_request[-1]["role"] == "user"
    assert reflect_request[-1]["content"] == REFLECT_REQUEST_NOTE
    assert reflect_request[-2]["role"] == "assistant"
    saved = " ".join(m.text() for m in state.context)
    assert REFLECT_REQUEST_NOTE not in saved
    assert PLAN_CONTINUE_NOTE not in saved


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_later_planner_request_ends_with_user_turn_first_does_not(mode):
    script = Script(
        plan=["first try", "second try"],
        reflect=["missing the second step", "all good [DONE]"],
    )
    state = await run(script, mode)

    first, second = script.of("PLAN")
    assert first[-1]["role"] == "user" and first[-1]["content"] == "HI"
    assert PLAN_CONTINUE_NOTE not in [m.get("content") for m in first]
    assert second[-1]["role"] == "user"
    assert second[-1]["content"] == PLAN_CONTINUE_NOTE
    assert second[-2]["role"] == "assistant"
    assert PLAN_CONTINUE_NOTE not in " ".join(m.text() for m in state.context)
    assert len(script.requests) == 4


def test_reflect_route_ends_turn_on_empty_reply_with_warning(caplog):
    from tenxgraph.core.state import AgentState
    from tenxgraph.prebuilt.agent.plan_act_reflect import _make_reflect_route

    state = AgentState()
    state.context = [
        Message.text_message("HI", role="user"),
        Message.text_message("", role="assistant"),
    ]
    with caplog.at_level(logging.WARNING):
        assert _make_reflect_route(3)(state) == END
    assert "empty reply" in " ".join(r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_empty_reply_is_retried_once_and_only_the_real_reply_is_saved(mode, caplog):
    script = Script(plan=["", "a useful planner reply"], reflect=["fine [DONE]"])
    with caplog.at_level(logging.WARNING):
        state = await run(script, mode)

    assert [k for k, _ in script.requests] == ["PLAN", "PLAN", "REFLECT"]
    assert texts(state) == [
        ("user", "HI"),
        ("assistant", "a useful planner reply"),
        ("assistant", "fine"),
    ]
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Node 'PLAN'" in messages and "gpt-4o" in messages
    assert "ends with an assistant turn" in messages


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_two_empty_replies_fail_the_node_and_save_nothing_empty(mode):
    script = Script(plan=["", ""], reflect=[])
    cp_holder = {}
    with pytest.raises(Exception, match="returned an empty reply in node 'PLAN'"):
        await run(script, mode, cp_holder=cp_holder)

    assert len(script.requests) == 2
    state = await cp_holder["cp"].aget_state(cp_holder["config"])
    assert not state or not any(m.role == "assistant" for m in state.context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_tool_call_only_reply_is_not_empty(mode):
    script = Script(plan=[("tool", "get_weather", {"city": "Paris"})], reflect=["sunny [DONE]"])
    await run(script, mode, tools=[get_weather])
    assert [k for k, _ in script.requests] == ["PLAN", "REFLECT"]


def test_request_builder_skips_saved_empty_assistant_replies():
    from tenxgraph.core.state import AgentState
    from tenxgraph.core.state.message_block import ReasoningBlock, TextBlock
    from tenxgraph.utils.converter import convert_messages

    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    with_calls = Message.text_message("", role="assistant")
    with_calls.tools_calls = [call]
    signed = Message(
        role="assistant",
        content=[ReasoningBlock(summary="t", signature="sig"), TextBlock(text="")],
    )
    state = AgentState()
    state.context = [
        Message.text_message("HI", role="user"),
        Message.text_message("", role="assistant"),
        Message(role="assistant", content=[]),
        Message(role="assistant", content=[ReasoningBlock(summary="only thinking")]),
        Message.text_message("real", role="assistant"),
        with_calls,
        signed,
    ]
    out = convert_messages(system_prompts=[], state=state)
    assert [m["role"] for m in out] == ["user", "assistant", "assistant", "assistant"]
    assert out[1]["content"] == "real"
    assert out[2].get("tool_calls")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_tool_calls_still_go_plan_act_reflect(mode):
    script = Script(
        plan=[("tool", "get_weather", {"city": "Paris"}), "unused"],
        reflect=["It is sunny in Paris. [DONE]"],
    )
    state = await run(script, mode, tools=[get_weather])

    kinds = [k for k, _ in script.requests]
    assert kinds == ["PLAN", "REFLECT"]
    roles = [m.role for m in state.context]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_msg = state.context[2]
    assert tool_msg.metadata["node_name"] == "ACT"
    # the planner was offered the tool
    assert state.context[1].tools_calls
    # the reflector never saw the tool messages
    assert not any(m["role"] == "tool" for m in script.of("REFLECT")[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_assistant_messages_carry_node_name(mode):
    script = Script(plan=["p"], reflect=["r [DONE]"])
    state = await run(script, mode)
    names = [m.metadata.get("node_name") for m in state.context if m.role == "assistant"]
    assert names == ["PLAN", "REFLECT"]
