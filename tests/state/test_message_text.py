from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import TextBlock


def test_text_extracts_from_mixed_block_shapes():
    message = Message.model_construct(
        role="assistant",
        content=[
            TextBlock(text="Hello "),
            {"type": "tool_result", "call_id": "call-1", "output": {"value": 42}},
            {"type": "text", "text": " world"},
        ],
    )

    assert message.text() == 'Hello {"value": 42} world'


def test_text_handles_string_and_non_list_content():
    string_message = Message.model_construct(role="user", content="plain text")
    other_message = Message.model_construct(role="user", content={"raw": True})

    assert string_message.text() == "plain text"
    assert other_message.text() == "{'raw': True}"
