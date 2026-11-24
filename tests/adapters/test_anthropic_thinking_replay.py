"""Signed thinking and redacted_thinking blocks survive a replay to Anthropic (J-3).

Round trip: SDK response or stream -> stored Message -> checkpoint JSON -> converted request
dict -> Anthropic request. Other providers must never receive the blocks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tenxgraph.core.graph.agent_internal.anthropic_request import merge_tool_results
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ReasoningBlock, TextBlock
from tenxgraph.runtime.adapters.llm.anthropic_converter import AnthropicConverter
from tenxgraph.runtime.adapters.llm.base_converter import BaseConverter
from tenxgraph.utils.converter import (
    THINKING_BLOCKS_KEY,
    _convert_dict,
    strip_provider_blocks,
)


def _usage():
    return SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def _response(content):
    return SimpleNamespace(
        id="msg_1",
        role="assistant",
        model="claude-opus-5",
        content=content,
        stop_reason="tool_use",
        stop_details=None,
        usage=_usage(),
    )


THINK = SimpleNamespace(type="thinking", thinking="let me see", signature="SIG==")
REDACTED = SimpleNamespace(type="redacted_thinking", data="OPAQUE")
TEXT = SimpleNamespace(type="text", text="calling")
TOOL = SimpleNamespace(type="tool_use", id="tu_1", name="lookup", input={"q": "x"})


class _Stream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self._events:
            yield e


def _start(index, **block):
    return SimpleNamespace(
        type="content_block_start", index=index, content_block=SimpleNamespace(**block)
    )


def _delta(index, dtype, **fields):
    return SimpleNamespace(
        type="content_block_delta", index=index, delta=SimpleNamespace(type=dtype, **fields)
    )


async def _stored_from_response() -> Message:
    return await AnthropicConverter().convert_response(_response([THINK, REDACTED, TEXT, TOOL]))


def _expected_thinking():
    return [
        {"type": "thinking", "thinking": "let me see", "signature": "SIG=="},
        {"type": "redacted_thinking", "data": "OPAQUE"},
    ]


@pytest.mark.asyncio
async def test_response_keeps_signature_and_redacted_data():
    msg = await _stored_from_response()
    reasoning = [b for b in msg.content if isinstance(b, ReasoningBlock)]
    assert reasoning[0].signature == "SIG==" and reasoning[0].summary == "let me see"
    assert reasoning[1].redacted_data == "OPAQUE"


@pytest.mark.asyncio
async def test_stream_keeps_signature_redacted_data_and_order():
    events = [
        _start(0, type="thinking", thinking="", signature=""),
        _delta(0, "thinking_delta", thinking="let me "),
        _delta(0, "thinking_delta", thinking="see"),
        _delta(0, "signature_delta", signature="SIG=="),
        _start(1, type="redacted_thinking", data="OPAQUE"),
        _delta(2, "text_delta", text="calling"),
        _start(3, type="tool_use", id="tu_1", name="lookup"),
        _delta(3, "input_json_delta", partial_json='{"q":"x"}'),
        SimpleNamespace(type="content_block_stop", index=3),
    ]
    out = [
        m
        async for m in AnthropicConverter().convert_streaming_response(
            {"thread_id": "t"}, "agent", _Stream(events)
        )
    ]
    final = out[-1]
    kinds = [type(b).__name__ for b in final.content]
    assert kinds[:2] == ["ReasoningBlock", "ReasoningBlock"]
    assert kinds.index("TextBlock") > 1 and kinds.index("ToolCallBlock") > 1
    assert final.content[0].summary == "let me see"
    assert final.content[0].signature == "SIG=="
    assert final.content[1].redacted_data == "OPAQUE"
    assert _convert_dict(final)[THINKING_BLOCKS_KEY] == _expected_thinking()


@pytest.mark.asyncio
async def test_round_trip_through_checkpoint_json_to_request():
    msg = await _stored_from_response()
    # What a checkpointer stores and loads.
    restored = Message.model_validate(msg.model_dump(mode="json"))

    request_dicts = [
        {"role": "user", "content": "hi"},
        _convert_dict(restored),
        {"role": "tool", "tool_call_id": "tu_1", "content": "result"},
    ]
    built = merge_tool_results(request_dicts)

    assistant = built[1]
    assert THINKING_BLOCKS_KEY not in assistant
    assert assistant["content"][:2] == _expected_thinking()
    types = [b["type"] for b in assistant["content"]]
    assert types == ["thinking", "redacted_thinking", "text", "tool_use"]
    assert built[2]["role"] == "user" and built[2]["content"][0]["type"] == "tool_result"


def test_plain_assistant_turn_with_thinking_gets_blocks_first():
    msg = Message(
        role="assistant",
        content=[
            ReasoningBlock(summary="hm", signature="S"),
            ReasoningBlock(summary="", redacted_data="R"),
            TextBlock(text="done"),
        ],
    )
    built = merge_tool_results([_convert_dict(msg)])
    assert built[0]["content"] == [
        {"type": "thinking", "thinking": "hm", "signature": "S"},
        {"type": "redacted_thinking", "data": "R"},
        {"type": "text", "text": "done"},
    ]


def test_unsigned_reasoning_is_not_replayed():
    msg = Message(role="assistant", content=[ReasoningBlock(summary="from another provider")])
    assert THINKING_BLOCKS_KEY not in _convert_dict(msg)


def test_other_providers_do_not_receive_the_blocks():
    msg = Message(role="assistant", content=[ReasoningBlock(summary="hm", signature="S")])
    converted = _convert_dict(msg)
    assert THINKING_BLOCKS_KEY in converted
    stripped = strip_provider_blocks([converted, {"role": "user", "content": "x"}])
    assert all(THINKING_BLOCKS_KEY not in m for m in stripped)
    # The input is not mutated.
    assert THINKING_BLOCKS_KEY in converted


@pytest.mark.asyncio
async def test_call_llm_strips_blocks_for_non_anthropic_provider():
    from tenxgraph.core.graph import Agent

    seen: dict = {}

    async def fake_call(self, messages, tools, stream, **kwargs):
        seen["messages"] = messages
        return "ok"

    agent = Agent.__new__(Agent)
    agent.provider = "openai"
    agent.api_style = "chat"
    agent.output_type = "text"
    agent.model = "gpt-x"
    agent.reasoning_config = None
    agent.base_url = None
    agent._set_api_style = lambda style: None
    agent._call_openai = fake_call.__get__(agent)
    msgs = [{"role": "assistant", "content": "a", THINKING_BLOCKS_KEY: [{"type": "thinking"}]}]
    assert await agent._call_llm(msgs) == "ok"
    assert THINKING_BLOCKS_KEY not in seen["messages"][0]


def _signed_reply(node: str | None) -> Message:
    return Message(
        role="assistant",
        content=[ReasoningBlock(summary="hm", signature="S"), TextBlock(text="draft")],
        tools_calls=[
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
        metadata={"node_name": node} if node else {},
    )


def _history_for(node: str | None, reply: Message) -> dict:
    from tenxgraph.utils.converter import convert_messages

    state = AgentState(context=[reply])
    return convert_messages(system_prompts=[], state=state, node=node)[0]


def test_other_nodes_thinking_is_dropped_but_text_and_tool_calls_stay():
    out = _history_for("reviewer", _signed_reply("writer"))
    assert THINKING_BLOCKS_KEY not in out
    assert out["content"] == "draft"
    assert out["tool_calls"][0]["id"] == "c1"


def test_own_nodes_thinking_is_replayed():
    out = _history_for("writer", _signed_reply("writer"))
    assert out[THINKING_BLOCKS_KEY] == [{"type": "thinking", "thinking": "hm", "signature": "S"}]


def test_unstamped_reply_keeps_thinking_for_any_node():
    assert THINKING_BLOCKS_KEY in _history_for("reviewer", _signed_reply(None))


def test_no_node_keeps_thinking():
    assert THINKING_BLOCKS_KEY in _history_for(None, _signed_reply("writer"))


class _StubConverter(BaseConverter):
    async def convert_response(self, response):
        return Message.text_message("hi", role="assistant")

    async def convert_streaming_response(self, config, node_name, response, meta=None):
        yield Message.text_message("hi", role="assistant")


@pytest.mark.asyncio
async def test_invoke_agent_node_stamps_node_name_on_reply():
    from unittest.mock import AsyncMock, MagicMock, patch

    from tenxgraph.core.graph.utils.node_runner import NodeRunner
    from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
    from tenxgraph.utils.callbacks import CallbackManager

    agent = MagicMock()
    agent.execute = AsyncMock(return_value=ModelResponseConverter(object(), _StubConverter()))
    with patch("tenxgraph.core.graph.utils.node_runner.publish_event"):
        steps = NodeRunner("writer", agent)._run_agent(
            {"thread_id": "t"}, AgentState(), CallbackManager(), streaming=False
        )
        result = [item async for item in steps][-1]
    assert result.messages[0].metadata["node_name"] == "writer"


@pytest.mark.asyncio
async def test_stream_reply_carries_node_name_from_the_converter():
    """The stream path stamps node_name in the provider converter (anthropic_converter.py:316)."""
    events = [_delta(0, "text_delta", text="hi")]
    out = [
        m
        async for m in AnthropicConverter().convert_streaming_response(
            {"thread_id": "t"}, "reviewer", _Stream(events)
        )
    ]
    assert out[-1].metadata["node_name"] == "reviewer"
