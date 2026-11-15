"""Text Postgres cannot store (NUL, lone surrogate) is refused at input, on load, in validators."""

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.graph.utils.utils import reject_nul_messages
from tenxgraph.core.state import AgentState, ExecutionStatus, Message
from tenxgraph.core.state.execution_state import ExecutionState
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END
from tenxgraph.utils.validators import PromptInjectionValidator, ValidationError


BAD = ["a\x00b", "x\ud800y", "x\udfffy"]


def build():
    g = StateGraph()
    g.add_node("a", lambda s: [Message.text_message("hi", role="assistant")])
    g.set_entry_point("a")
    g.add_edge("a", END)
    return g.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.parametrize("bad", BAD)
def test_unstorable_text_rejected_at_input(bad):
    app = build()
    cfg = {"thread_id": "t"}
    with pytest.raises(ValidationError) as exc:
        app.invoke({"messages": [Message.text_message(bad)]}, cfg)
    assert exc.value.violation_type == "invalid_character"
    out = app.invoke({"messages": [Message.text_message("ok")]}, cfg)
    assert out["messages"]


def test_nul_rejected():
    test_unstorable_text_rejected_at_input("a\x00b")


def test_real_emoji_is_accepted():
    app = build()
    out = app.invoke({"messages": [Message.text_message("hi \U0001f600")]}, {"thread_id": "e"})
    assert out["messages"]


def test_surrogate_in_metadata_is_refused():
    msg = Message.text_message("fine")
    msg.metadata = {"note": ["x\ud800y"]}
    with pytest.raises(ValidationError):
        reject_nul_messages([msg])


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.asyncio
async def test_poisoned_cached_state_is_refused_on_load(bad):
    cp = InMemoryCheckpointer()
    config = {"thread_id": "t-poison", "user_id": "u1"}
    state = AgentState(context=[Message.text_message(bad)])
    state.execution_meta = ExecutionState(
        current_node="a", step=1, status=ExecutionStatus.RUNNING, run_id="dead-run"
    )
    await cp.aput_state_cache(config, state)
    g = StateGraph()
    g.add_node("a", lambda s: [Message.text_message("hi", role="assistant")])
    g.set_entry_point("a")
    g.add_edge("a", END)
    app = g.compile(checkpointer=cp)
    with pytest.raises(ValidationError):
        await app.ainvoke({"messages": [Message.text_message("ok")]}, config)


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.asyncio
async def test_validator_refuses_unstorable_text(bad):
    v = PromptInjectionValidator(strict_mode=True)
    with pytest.raises(ValidationError) as exc:
        await v.validate([Message.text_message(bad, role="user")])
    assert exc.value.violation_type == "invalid_character"


@pytest.mark.asyncio
async def test_validator_accepts_real_emoji():
    v = PromptInjectionValidator(strict_mode=True)
    assert await v.validate([Message.text_message("hello \U0001f600", role="user")]) is True


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.asyncio
async def test_unstorable_text_in_caller_state_is_refused(bad):
    app = build()
    cfg = {"thread_id": "t-state"}
    for state in ({"note": bad}, {"nested": {"k": ["ok", bad]}}, {bad: "v"}):
        with pytest.raises(ValidationError) as exc:
            await app.ainvoke({"messages": [Message.text_message("hi")], "state": state}, cfg)
        assert exc.value.violation_type == "invalid_character"
        assert "State" in str(exc.value)
    out = await app.ainvoke({"messages": [Message.text_message("ok")]}, cfg)
    assert out["messages"]


@pytest.mark.asyncio
async def test_real_emoji_in_caller_state_is_accepted():
    app = build()
    out = await app.ainvoke(
        {"messages": [Message.text_message("hi")], "state": {"note": "ok \U0001f600"}},
        {"thread_id": "t-state-ok"},
    )
    assert out["messages"]
