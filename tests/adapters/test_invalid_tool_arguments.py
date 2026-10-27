"""Model tool-call arguments that are cut off, empty or not an object (B7).

Every converter must survive them, keep the call provider-shaped, and never hand
the tool guessed arguments.
"""

from types import SimpleNamespace

import pytest

from tenxgraph.core.graph.agent_internal.anthropic_request import convert_assistant_tool_calls
from tenxgraph.core.state.message_block import ToolCallBlock
from tenxgraph.runtime.adapters.llm.anthropic_converter import AnthropicConverter
from tenxgraph.runtime.adapters.llm.openai_converter import OpenAIConverter
from tenxgraph.runtime.adapters.llm.openai_responses_converter import OpenAIResponsesConverter
from tenxgraph.runtime.adapters.llm.tool_arguments import parse_tool_arguments


INVALID = ['{"a": ', "[]", '"str"', "42", "null", "{", '{"a": 1} trailing']
VALID_OBJ = '{"a": 1}'


class TestParser:
    @pytest.mark.parametrize("raw", [None, "", "   ", "\n"])
    def test_empty_means_no_arguments(self, raw):
        assert parse_tool_arguments(raw) == ({}, None)

    def test_dict_passes_through(self):
        d = {"a": 1}
        assert parse_tool_arguments(d) == (d, None)

    def test_json_object(self):
        assert parse_tool_arguments(VALID_OBJ) == ({"a": 1}, None)

    @pytest.mark.parametrize("raw", INVALID + [5, ["x"]])
    def test_invalid_gives_error(self, raw):
        args, error = parse_tool_arguments(raw)
        assert args == {}
        assert error

    def test_warning_truncates_raw_arguments(self, caplog):
        raw = '{"a": "' + "x" * 5000
        with caplog.at_level("WARNING", logger="tenxgraph.llm"):
            parse_tool_arguments(raw, tool_name="t")
        assert caplog.records
        assert len(caplog.text) < 1500


def _openai_response(arguments):
    function = SimpleNamespace(name="calc", arguments=arguments)
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=function,
        model_dump=lambda: {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calc", "arguments": arguments},
        },
    )
    message = SimpleNamespace(
        role="assistant",
        content=None,
        audio=None,
        images=None,
        reasoning_content=None,
        tool_calls=[tool_call],
    )
    choice = SimpleNamespace(message=message, finish_reason="length")
    usage = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        completion_tokens_details=None,
        prompt_tokens_details=None,
    )
    return SimpleNamespace(id="r1", model="gpt-x", created=1, usage=usage, choices=[choice])


def _expected(raw):
    return ({"a": 1} if raw == VALID_OBJ else {}), raw


class TestOpenAIChat:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", INVALID + ["", VALID_OBJ])
    async def test_non_stream(self, raw):
        msg = await OpenAIConverter().convert_response(_openai_response(raw))
        block = next(b for b in msg.content if isinstance(b, ToolCallBlock))
        args, kept = _expected(raw)
        assert block.args == args
        # raw string kept so the ToolNode can reject the call
        assert msg.tools_calls[0]["function"]["arguments"] == kept

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", INVALID + ["", VALID_OBJ])
    async def test_stream_final_blocks(self, raw):
        tc = {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": raw}}
        blocks = OpenAIConverter._build_stream_blocks("", "", [tc])
        assert blocks[0].args == _expected(raw)[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", INVALID + [VALID_OBJ])
    async def test_stream_delta_keeps_call(self, raw):
        tool_call = SimpleNamespace(
            id="tc-1", type="function", function=SimpleNamespace(name="calc", arguments=raw)
        )
        delta = SimpleNamespace(tool_calls=[tool_call])
        blocks: list = []
        calls: list = []
        OpenAIConverter()._process_delta_tool_calls(delta, calls, set(), blocks)
        assert calls[0]["function"]["arguments"] == raw
        assert blocks[0].args == _expected(raw)[0]


class TestOpenAIResponses:
    @pytest.mark.parametrize("raw", INVALID + ["", VALID_OBJ])
    def test_item_extraction(self, raw):
        item = SimpleNamespace(type="function_call", name="calc", arguments=raw, call_id="c1")
        block, tool_call = OpenAIResponsesConverter._extract_tool_call_from_item(item)
        assert block.args == _expected(raw)[0]
        assert tool_call["function"]["arguments"] == (raw or "{}")

    @pytest.mark.parametrize("raw", INVALID + ["", VALID_OBJ])
    def test_parse_json_args(self, raw):
        assert OpenAIResponsesConverter._parse_json_args(raw) == _expected(raw)[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", INVALID + [VALID_OBJ])
    async def test_stream(self, raw):
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="function_call", id="fc1", name="calc", call_id="c1", arguments=""
                ),
            ),
            SimpleNamespace(type="response.function_call_arguments.delta", delta=raw),
            SimpleNamespace(type="response.function_call_arguments.done", arguments=raw),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call", id="fc1", name="calc", call_id="c1", arguments=raw
                ),
            ),
        ]

        async def aiter():
            for e in events:
                yield e

        out = [
            m
            async for m in OpenAIResponsesConverter().convert_streaming_response(
                config={}, node_name="n", response=aiter()
            )
        ]
        final = out[-1]
        assert final.delta is False
        assert all(
            b.args == _expected(raw)[0] for b in final.content if isinstance(b, ToolCallBlock)
        )


async def _anthropic_stream(raw_fragments):
    class FakeStream:
        def __init__(self, events):
            self.events = events

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def __aiter__(self):
            async def gen():
                for e in self.events:
                    yield e

            return gen()

    events = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="calc"),
        )
    ]
    for frag in raw_fragments:
        events.append(
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json=frag),
            )
        )
    events.append(SimpleNamespace(type="content_block_stop", index=0))
    return [
        m
        async for m in AnthropicConverter().convert_streaming_response(
            {"thread_id": "t"}, "agent", FakeStream(events)
        )
    ]


class TestAnthropic:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", INVALID + [VALID_OBJ])
    async def test_stream_keeps_raw_for_invalid(self, raw):
        final = (await _anthropic_stream([raw]))[-1]
        block = next(b for b in final.content if isinstance(b, ToolCallBlock))
        assert block.args == _expected(raw)[0]
        assert final.tools_calls[0]["function"]["arguments"] == raw

    @pytest.mark.asyncio
    async def test_stream_no_input_means_empty_object(self):
        final = (await _anthropic_stream([]))[-1]
        assert final.tools_calls[0]["function"]["arguments"] == "{}"

    @pytest.mark.parametrize("raw", INVALID + ["", VALID_OBJ])
    def test_history_input_is_always_an_object(self, raw):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "calc", "arguments": raw}}
            ],
        }
        block = convert_assistant_tool_calls(msg)["content"][-1]
        assert block["type"] == "tool_use"
        assert block["input"] == _expected(raw)[0]


class TestGoogle:
    @pytest.mark.parametrize("raw", INVALID + ["", VALID_OBJ])
    def test_history_args_are_a_mapping(self, raw):
        pytest.importorskip("google.genai")
        from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin

        part = AgentGoogleMixin()._create_function_call_part(
            {"id": "c1", "function": {"name": "calc", "arguments": raw}}
        )
        assert part.function_call.args == _expected(raw)[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", INVALID + [VALID_OBJ])
    async def test_stream_final_blocks(self, raw):
        pytest.importorskip("google.genai")
        from tenxgraph.runtime.adapters.llm import google_genai_converter as mod

        # The final-block loop parses ``tools_calls`` strings with the shared parser.
        from tenxgraph.runtime.adapters.llm.tool_arguments import parse_tool_arguments

        assert mod.parse_tool_arguments is parse_tool_arguments
        assert parse_tool_arguments(raw)[0] == _expected(raw)[0]


def test_openai_history_keeps_arguments_a_string():
    """OpenAI chat history needs ``arguments`` as a string; the raw text is replayed as is."""
    raw = '{"a": '
    msg = OpenAIConverter._serialize_tool_call(
        SimpleNamespace(id="c1", type="function", function=SimpleNamespace(name="x", arguments=raw))
    )
    assert isinstance(msg["function"]["arguments"], str)
