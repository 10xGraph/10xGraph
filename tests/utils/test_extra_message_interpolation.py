"""``{field}`` placeholders in extra_messages and system prompts are filled from the state."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import (
    ImageBlock,
    MediaRef,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END
from tenxgraph.utils.converter import (
    _fill_placeholders,
    _interpolate_message,
    _interpolate_system_prompts,
    convert_messages,
)


class CityState(AgentState):
    city: str = ""
    user: dict = {}


# ---------------------------------------------------------------- helper


def test_fills_present_keys():
    assert _fill_placeholders("in {city}, {n}", {"city": "Paris", "n": 3}) == "in Paris, 3"


def test_missing_key_stays_and_others_are_filled(caplog):
    with caplog.at_level(logging.WARNING, logger="tenxgraph.utils"):
        out = _fill_placeholders("{a} {missing} {b}", {"a": 1, "b": 2})
    assert out == "1 {missing} 2"
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "missing" in warnings[0]


def test_warning_does_not_log_text_or_values(caplog):
    with caplog.at_level(logging.WARNING, logger="tenxgraph.utils"):
        _fill_placeholders("secret prose {missing}", {"a": "hidden-value"})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "secret prose" not in text
    assert "hidden-value" not in text


def test_bad_format_spec_stays_as_written():
    assert _fill_placeholders("{x:zz} ok", {"x": 5}) == "{x:zz} ok"


def test_conversion_and_spec_are_applied():
    assert _fill_placeholders("{x!r} {y:>4}", {"x": "a", "y": "b"}) == "'a'    b"


def test_spec_with_placeholder_is_filled():
    assert _fill_placeholders("{x:>{w}}", {"x": "a", "w": 3}) == "  a"


def test_double_braces_are_literal():
    assert _fill_placeholders("{{x}} {x}", {"x": 1}) == "{x} 1"


def test_unbalanced_brace_returns_original():
    assert _fill_placeholders("oops { {x}", {"x": 1}) == "oops { {x}"


def test_json_like_text_is_unchanged(caplog):
    text = 'reply as {"a": 1}'
    with caplog.at_level(logging.WARNING, logger="tenxgraph.utils"):
        assert _fill_placeholders(text, {"a": 1}) == text
    assert not caplog.records


def test_positional_fields_stay():
    assert _fill_placeholders("{} {0} {x}", {"x": 1}) == "{} {0} 1"


def test_index_and_attribute_access():
    assert _fill_placeholders("{a[b]} {a[zz]}", {"a": {"b": "ok"}}) == "ok {a[zz]}"
    assert _fill_placeholders("{o.real}", {"o": 3}) == "3"


# ---------------------------------------------------------------- system prompts


def test_one_missing_key_does_not_block_other_system_keys():
    state = CityState(city="Paris")
    prompts = [
        {"role": "system", "content": "in {city} {nope}"},
        {"role": "system", "content": [{"type": "text", "text": "{city}"}]},
    ]
    out = _interpolate_system_prompts(prompts, state)
    assert out[0]["content"] == "in Paris {nope}"
    assert out[1] is prompts[1]


# ---------------------------------------------------------------- convert_messages


def _roles(res):
    return [m["role"] for m in res]


def test_string_content_extra_message_filled_and_last():
    state = CityState(city="Paris", context=[Message.text_message("hi")])
    extra = Message.text_message("now in {city}", role="user")
    res = convert_messages([{"role": "system", "content": "sys"}], state, [extra])
    assert _roles(res) == ["system", "user", "user"]
    assert res[-1]["content"] == "now in Paris"


def test_text_block_content_filled():
    state = CityState(city="Paris")
    extra = Message(
        role="user",
        content=[TextBlock(text="a {city}"), TextBlock(text="b {city}")],
    )
    res = convert_messages([{"role": "system", "content": "s"}], state, [extra])
    texts = (
        [p["text"] for p in res[-1]["content"]]
        if isinstance(res[-1]["content"], list)
        else [res[-1]["content"]]
    )
    assert "Paris" in "".join(texts)
    assert "{city}" not in "".join(texts)


def test_original_message_is_not_mutated():
    state = CityState(city="Paris")
    extra = Message.text_message("now in {city}", role="user")
    convert_messages([{"role": "system", "content": "s"}], state, [extra])
    convert_messages([{"role": "system", "content": "s"}], CityState(city="Rome"), [extra])
    assert extra.text() == "now in {city}"
    assert extra.content[0].text == "now in {city}"


def test_media_and_tool_blocks_are_untouched():
    values = {"city": "Paris"}
    image = ImageBlock(media=MediaRef(kind="url", url="https://x/{city}.png"))
    msg = Message(role="user", content=[TextBlock(text="{city}"), image])
    out = _interpolate_message(msg, values)
    assert out.content[0].text == "Paris"
    assert out.content[1] is image

    call = Message(
        role="assistant",
        content=[ToolCallBlock(id="c1", name="f", args={"q": "{city}"})],
    )
    result = Message.tool_message([ToolResultBlock(call_id="c1", output="{city}")])
    assert _interpolate_message(call, values) is call
    assert _interpolate_message(result, values) is result


def test_state_none_leaves_text_unchanged():
    extra = Message.text_message("now in {city}", role="user")
    res = convert_messages([{"role": "system", "content": "s {city}"}], None, [extra])
    assert res[-1]["content"] == "now in {city}"
    assert res[0]["content"] == "s {city}"


# ---------------------------------------------------------------- Agent run


class _CapturingAgent(Agent):
    seen: list = []

    async def _call_llm(self, messages, tools=None, stream=False, **kwargs):
        self.seen.append(messages)
        return ChatCompletion.model_validate(
            {
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            }
        )


@pytest.mark.asyncio
async def test_agent_run_sends_filled_extra_message_and_does_not_save_it():
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        agent = _CapturingAgent(
            model="gpt-4o",
            provider="openai",
            api_style="chat",
            reasoning_config=None,
            system_prompt=[{"role": "system", "content": "sys {city} {nope}"}],
            extra_messages=[Message.text_message("now in {city}", role="user")],
        )
    agent.seen = []
    graph = StateGraph(CityState())
    graph.add_node("MAIN", agent)
    graph.add_edge("MAIN", END)
    graph.set_entry_point("MAIN")
    cp = InMemoryCheckpointer()
    app = graph.compile(checkpointer=cp)
    config = {"thread_id": "t-extra"}

    await app.ainvoke(
        {"messages": [Message.text_message("hi")], "state": {"city": "Paris"}}, config
    )

    sent = agent.seen[0]
    assert sent[0]["content"] == "sys Paris {nope}"
    assert sent[-1] == {"role": "user", "content": "now in Paris"}
    state = await cp.aget_state(config)
    assert [m.text() for m in state.context] == ["hi", "ok"]


# ---------------------------------------------------------------- provider adapters

TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}
PLAIN = [
    {"role": "user", "content": "hi"},
    {"role": "user", "content": "now in Paris"},
]
WITH_TOOL = [
    {"role": "user", "content": "weather?"},
    {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    {"role": "user", "content": "now in Paris"},
]


def _agent(provider: str, **kwargs) -> Agent:
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        return Agent(model="x-model", provider=provider, reasoning_config=None, **kwargs)


@pytest.mark.parametrize("messages", [PLAIN, WITH_TOOL], ids=["plain", "tool"])
def test_google_trailing_user_message_is_last(messages):
    agent = _agent("google")
    system, contents = agent._convert_to_google_format(messages)
    assert system is None
    assert contents[-1].role == "user"
    assert contents[-1].parts[-1].text == "now in Paris"


@pytest.mark.parametrize("messages", [PLAIN, WITH_TOOL], ids=["plain", "tool"])
@pytest.mark.asyncio
async def test_openai_chat_trailing_user_message_is_last(messages):
    agent = _agent("openai", api_style="chat")
    agent.client.chat.completions.create = _async_mock()
    await agent._call_openai(messages, tools=None, stream=False)
    sent = agent.client.chat.completions.create.call_args[1]["messages"]
    assert sent[-1] == {"role": "user", "content": "now in Paris"}


@pytest.mark.parametrize("messages", [PLAIN, WITH_TOOL], ids=["plain", "tool"])
@pytest.mark.asyncio
async def test_openai_responses_trailing_user_message_is_last(messages):
    agent = _agent("openai", api_style="responses")
    agent.client.responses.create = _async_mock()
    await agent._call_openai_responses(messages, tools=None, stream=False)
    items = agent.client.responses.create.call_args[1]["input"]
    assert items[-1]["role"] == "user"
    assert "now in Paris" in str(items[-1]["content"])


@pytest.mark.parametrize("messages", [PLAIN, WITH_TOOL], ids=["plain", "tool"])
def test_anthropic_trailing_user_message_is_last(messages):
    agent = _agent("anthropic")
    sent, _ = agent._build_anthropic_request(messages)
    assert sent[-1]["role"] == "user"
    assert "now in Paris" in str(sent[-1]["content"])


def _async_mock():
    return AsyncMock(return_value=MagicMock())
