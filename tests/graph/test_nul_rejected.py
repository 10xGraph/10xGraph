import asyncio, pytest
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END
from tenxgraph.utils.validators import ValidationError

def build():
    g = StateGraph()
    g.add_node("a", lambda s: [Message.text_message("hi", role="assistant")])
    g.set_entry_point("a"); g.add_edge("a", END)
    return g.compile(checkpointer=InMemoryCheckpointer())

def test_nul_rejected():
    app = build(); cfg = {"thread_id": "t"}
    with pytest.raises(ValidationError):
        app.invoke({"messages": [Message.text_message("a\x00b")]}, cfg)
    out = app.invoke({"messages": [Message.text_message("ok")]}, cfg)
    assert out["messages"]
