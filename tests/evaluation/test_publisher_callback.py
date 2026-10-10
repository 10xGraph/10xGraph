from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message import TokenUsages
from tenxgraph.qa.evaluation.collectors.publisher_callback import PublisherCallback
from tenxgraph.runtime.adapters.llm.base_converter import BaseConverter
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.runtime.publisher.events import ContentType, Event, EventType
from tenxgraph.utils.callbacks import CallbackContext, InvocationType


class _DummyPublisher(BasePublisher):
    def __init__(self):
        super().__init__(config={})
        self.publish_mock = AsyncMock()

    async def publish(self, event):
        return await self.publish_mock(event)

    async def close(self):
        return None

    def sync_close(self):
        return None


class _DummyConverter(BaseConverter):
    async def convert_response(self, response):
        return response

    async def convert_streaming_response(
        self, config: dict, node_name: str, response, meta=None
    ) -> AsyncGenerator:
        if False:
            yield response


@pytest.mark.asyncio
async def test_callback_publishes_tool_event():
    publisher = _DummyPublisher()
    callback = PublisherCallback(publisher, config={"thread_id": "th", "run_id": "rn"})

    ctx = CallbackContext(
        invocation_type=InvocationType.TOOL,
        node_name="tool_node",
        function_name="get_weather",
        metadata={"tool_call_id": "tc-1"},
    )

    output = await callback(ctx, {"city": "NYC"}, "sunny")
    assert output == "sunny"
    assert publisher.publish_mock.await_count == 1

    event = publisher.publish_mock.await_args.args[0]
    assert event.event == Event.TOOL_EXECUTION
    assert event.event_type == EventType.END
    assert event.node_name == "get_weather"
    assert event.content_type == [ContentType.TOOL_RESULT]
    assert event.data["args"] == {"city": "NYC"}
    assert event.data["result"] == "sunny"
    assert event.data["tool_call_id"] == "tc-1"


@pytest.mark.asyncio
async def test_callback_publishes_ai_event_with_tool_calls_and_usage():
    publisher = _DummyPublisher()
    callback = PublisherCallback(publisher, config={"thread_id": "thread", "run_id": "run"})

    state = AgentState()
    state.context = [Message.text_message("hello", role="user")]

    node_message = Message.text_message("I will call a tool", role="assistant")
    node_message.tools_calls = [{"name": "search"}]
    node_message.usages = TokenUsages(
        completion_tokens=2,
        prompt_tokens=3,
        total_tokens=5,
        cache_read_input_tokens=1,
        cache_creation_input_tokens=4,
    )

    converter = ModelResponseConverter(response=node_message, converter=_DummyConverter())

    ctx = CallbackContext(invocation_type=InvocationType.AI, node_name="MAIN")
    result = await callback(ctx, {"state": state}, converter)

    assert result is converter
    assert publisher.publish_mock.await_count == 1

    event = publisher.publish_mock.await_args.args[0]
    assert event.event == Event.NODE_EXECUTION
    assert event.event_type == EventType.END
    assert event.node_name == "MAIN"
    assert event.content_type == [ContentType.MESSAGE]
    assert event.data["response_text"] == "I will call a tool"
    assert event.data["has_tool_calls"] is True
    assert event.data["tool_call_names"] == ["search"]
    assert event.data["is_final"] is False
    assert event.data["token_usage"] == {
        "input_tokens": 3,
        "output_tokens": 2,
        "cache_read_tokens": 1,
        "cache_creation_tokens": 4,
    }


@pytest.mark.asyncio
async def test_extract_node_message_from_dict_and_fallback_none():
    publisher = _DummyPublisher()
    callback = PublisherCallback(publisher)

    msg = Message.text_message("done", role="assistant")
    extracted = await callback._extract_node_message({"messages": [msg]})
    assert extracted is msg

    extracted_none = await callback._extract_node_message("not-supported")
    assert extracted_none is None


def test_build_event_returns_none_for_unhandled_invocation_type():
    publisher = _DummyPublisher()
    callback = PublisherCallback(publisher)

    ctx = CallbackContext(invocation_type=InvocationType.INPUT_VALIDATION, node_name="X")
    event = callback._build_event(ctx, {}, {})
    assert event is None
