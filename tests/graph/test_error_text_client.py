"""Raw exception text from providers and node code is not sent to the client (G-64)."""

import pytest

from tenxgraph.core.exceptions import NodeError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.graph.utils.error_text import client_error_text
from tenxgraph.core.state import Message
from tenxgraph.utils.constants import END


def _graph(exc: Exception):
    def boom(state):
        raise exc

    g = StateGraph()
    g.add_node("MAIN", boom)
    g.add_edge("MAIN", END)
    g.set_entry_point("MAIN")
    return g.compile()


def test_framework_errors_keep_their_message():
    assert "node failed" in client_error_text(NodeError("node failed", "NODE_001"))


def test_foreign_errors_only_show_the_type():
    text = client_error_text(RuntimeError("key sk-1234 at https://internal/db"))
    assert "sk-1234" not in text
    assert "RuntimeError" in text


@pytest.mark.asyncio
async def test_stream_error_chunk_hides_exception_text():
    compiled = _graph(RuntimeError("secret provider detail"))
    chunks = []
    with pytest.raises(NodeError):
        async for chunk in compiled.astream({"messages": [Message.text_message("hi")]}):
            chunks.append(chunk)
    errors = [c for c in chunks if getattr(c.event, "value", c.event) == "error"]
    assert errors
    assert "secret provider detail" not in str(errors[-1].data)
    assert "secret provider detail" not in str(errors[-1].state.execution_meta)


def test_wrapper_around_foreign_error_hides_the_wrapped_text():
    try:
        try:
            raise RuntimeError("secret provider detail")
        except RuntimeError as inner:
            raise NodeError("Error in node 'X': secret provider detail", "NODE_001") from inner
    except NodeError as wrapped:
        text = client_error_text(wrapped)
    assert "secret" not in text
    assert "RuntimeError" in text
