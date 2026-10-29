"""``has_reasoning`` is derived from ReasoningBlocks, and old saved messages still load."""

from tenxgraph.core.state import AgentState, Message, TextBlock
from tenxgraph.core.state.message_block import ReasoningBlock
from tenxgraph.storage.store.store_schema import MemoryRecord


def _has_reasoning(*blocks):
    message = Message(role="assistant", content=list(blocks))
    return MemoryRecord.from_message(message).metadata["has_reasoning"]


def test_text_only_has_no_reasoning():
    assert _has_reasoning(TextBlock(text="hi")) is False


def test_reasoning_summary_counts():
    assert _has_reasoning(ReasoningBlock(summary="thinking"), TextBlock(text="hi")) is True


def test_redacted_reasoning_counts():
    assert _has_reasoning(ReasoningBlock(summary="", redacted_data="OPAQUE")) is True


def test_details_only_reasoning_counts():
    assert _has_reasoning(ReasoningBlock(summary="", details=["step"])) is True


def test_empty_reasoning_block_does_not_count():
    assert _has_reasoning(ReasoningBlock(summary="")) is False


def test_message_has_no_reasoning_field():
    assert "reasoning" not in Message.model_fields


def test_old_message_dict_with_reasoning_string_still_loads():
    old = {
        "message_id": "m1",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "reasoning": "old string reasoning",
    }
    message = Message.model_validate(old)
    assert message.text() == "hi"
    assert not hasattr(message, "reasoning")


def test_old_state_with_reasoning_string_still_loads():
    old = {
        "context": [
            {
                "message_id": "m1",
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "reasoning": "old string reasoning",
            }
        ]
    }
    state = AgentState.model_validate(old)
    assert state.context[0].text() == "hi"
