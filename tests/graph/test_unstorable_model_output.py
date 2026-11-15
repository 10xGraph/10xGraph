"""NUL and lone surrogates in model and tool output are cleaned at the source."""

import pytest

from tenxgraph.core.graph import StateGraph, ToolNode
from tenxgraph.core.graph.utils.utils import reject_nul_messages
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.adapters.llm.base_converter import BaseConverter
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END
from tenxgraph.utils.validators import ValidationError, has_unstorable_text


BAD = ["a\x00b", "x\ud800y", "x\udfffy"]


class FakeConverter(BaseConverter):
    def __init__(self, bad):
        self.bad = bad

    async def convert_response(self, response):
        return Message.text_message(f"hi {self.bad}", role="assistant")

    async def convert_streaming_response(self, config, node_name, response, meta=None):
        for part in ("hi ", self.bad):
            delta = Message.text_message(part, role="assistant")
            delta.delta = True
            yield delta
        yield Message.text_message(f"hi {self.bad}", role="assistant")


def clean(text):
    return not has_unstorable_text(text)


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.asyncio
async def test_invoke_reply_is_clean(bad):
    msg = await ModelResponseConverter("raw", FakeConverter(bad)).invoke()
    assert clean(msg.text())
    assert msg.text().startswith("hi ")
    msg.model_dump_json()


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.asyncio
async def test_every_stream_chunk_is_clean_and_serializable(bad):
    conv = ModelResponseConverter("raw", FakeConverter(bad))
    chunks = [m async for m in conv.stream({"thread_id": "t"}, "MAIN")]
    assert len(chunks) == 3
    for chunk in chunks:
        assert clean(chunk.text())
        chunk.model_dump_json()


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.asyncio
async def test_saved_state_is_clean(bad):
    cp = InMemoryCheckpointer()
    g = StateGraph()
    g.add_node("a", lambda s: ModelResponseConverter("raw", FakeConverter(bad)))
    g.set_entry_point("a")
    g.add_edge("a", END)
    app = g.compile(checkpointer=cp)
    config = {"thread_id": "t-src"}
    async for chunk in app.astream({"messages": [Message.text_message("ok")]}, config):
        chunk.model_dump_json()
    saved = await cp.aget_state(config)
    assert all(not has_unstorable_text(m) for m in saved.context)
    assert saved.context[-1].text().startswith("hi ")


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.asyncio
async def test_tool_result_is_clean(bad):
    def tool(x: str) -> str:
        return f"out {bad}"

    node = ToolNode([tool])
    res = await node.invoke("tool", {"x": "1"}, "c1", {"thread_id": "t"}, AgentState())
    assert not has_unstorable_text(res)
    res.model_dump_json()
    streamed = [
        m async for m in node.stream("tool", {"x": "1"}, "c2", {"thread_id": "t"}, AgentState())
    ]
    assert all(not has_unstorable_text(m) for m in streamed)


@pytest.mark.parametrize("bad", BAD)
def test_client_input_is_still_refused(bad):
    with pytest.raises(ValidationError):
        reject_nul_messages([Message.text_message(bad)])
