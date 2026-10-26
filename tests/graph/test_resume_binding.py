"""A resume is tied to the interrupt it answers and checked against ``response_schema``."""

import pytest

from tenxgraph.core.exceptions import InterruptMismatchError, ResumeValidationError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END
from tenxgraph.utils.interrupt import (
    RESUME_VALUES_KEY,
    interrupt,
    pending_interrupt,
    validate_against_schema,
)


FULL = ResponseGranularity.FULL
BOOL_SCHEMA = {
    "type": "object",
    "properties": {"approved": {"type": "boolean"}},
    "required": ["approved"],
}


def _app():
    async def ask(state: AgentState):
        first = interrupt("step 1", message="Approve step 1?", response_schema=BOOL_SCHEMA)
        second = interrupt("wire $1M", message="Wire $1M?", response_schema=BOOL_SCHEMA)
        return Message.text_message(f"first={first} second={second}", role="assistant")

    graph = StateGraph()
    graph.add_node("ASK", ask)
    graph.add_edge("ASK", END)
    graph.set_entry_point("ASK")
    return graph.compile(checkpointer=InMemoryCheckpointer())


async def _pause(app, thread: str):
    result = await app.ainvoke(
        {"messages": [Message.text_message("go")]}, {"thread_id": thread}, response_granularity=FULL
    )
    return pending_interrupt(result["state"])


async def _recorded(app, thread: str) -> dict:
    state = await app._checkpointer.aget_state({"thread_id": thread, "user_id": "anonymous"})
    return dict(state.execution_meta.internal_data.get(RESUME_VALUES_KEY) or {})


@pytest.mark.asyncio
async def test_matching_id_is_accepted():
    app = _app()
    first = await _pause(app, "m1")
    result = await app.ainvoke(
        {"resume": {"approved": True}, "resume_interrupt_id": first.id},
        {"thread_id": "m1"},
        response_granularity=FULL,
    )
    second = pending_interrupt(result["state"])
    assert second is not None
    assert second.id != first.id


@pytest.mark.asyncio
async def test_wrong_id_is_rejected_and_nothing_recorded():
    app = _app()
    first = await _pause(app, "m2")
    with pytest.raises(InterruptMismatchError) as exc:
        await app.ainvoke(
            {"resume": {"approved": True}, "resume_interrupt_id": "int_other"},
            {"thread_id": "m2"},
        )
    assert exc.value.error_code == "INTERRUPT_001"
    assert isinstance(exc.value, ValueError)
    assert await _recorded(app, "m2") == {}
    # Still paused at the same interrupt.
    again = await app.ainvoke(
        {"resume": {"approved": True}, "resume_interrupt_id": first.id},
        {"thread_id": "m2"},
        response_granularity=FULL,
    )
    assert pending_interrupt(again["state"]) is not None


@pytest.mark.asyncio
async def test_resume_without_id_still_works():
    app = _app()
    await _pause(app, "m3")
    result = await app.ainvoke(
        {"resume": {"approved": True}}, {"thread_id": "m3"}, response_granularity=FULL
    )
    assert pending_interrupt(result["state"]) is not None


@pytest.mark.asyncio
async def test_schema_valid_value_is_accepted():
    app = _app()
    await _pause(app, "s1")
    result = await app.ainvoke(
        {"resume": {"approved": False}}, {"thread_id": "s1"}, response_granularity=FULL
    )
    assert pending_interrupt(result["state"]) is not None


@pytest.mark.asyncio
async def test_schema_invalid_value_is_rejected_and_nothing_recorded():
    app = _app()
    await _pause(app, "s2")
    with pytest.raises(ResumeValidationError) as exc:
        await app.ainvoke({"resume": {"approved": "no"}}, {"thread_id": "s2"})
    assert exc.value.error_code == "INTERRUPT_002"
    assert isinstance(exc.value, ValueError)
    assert "approved" in str(exc.value)
    assert await _recorded(app, "s2") == {}
    with pytest.raises(ResumeValidationError):
        await app.ainvoke({"resume": {"x": "no"}}, {"thread_id": "s2"})


@pytest.mark.asyncio
async def test_cancel_with_none_skips_validation():
    app = _app()
    await _pause(app, "c1")
    result = await app.ainvoke({"resume": None}, {"thread_id": "c1"}, response_granularity=FULL)
    # The node re-ran with the cancel answer and paused at the second prompt.
    assert pending_interrupt(result["state"]) is not None


@pytest.mark.asyncio
async def test_replayed_resume_does_not_approve_the_next_prompt():
    app = _app()
    first = await _pause(app, "r1")
    body = {"resume": {"approved": True}, "resume_interrupt_id": first.id}
    result = await app.ainvoke(body, {"thread_id": "r1"}, response_granularity=FULL)
    second = pending_interrupt(result["state"])
    assert second is not None

    # Same request again (double click, stale tab): the id belongs to the answered prompt.
    with pytest.raises(InterruptMismatchError):
        await app.ainvoke(body, {"thread_id": "r1"})
    # Only the first answer is on record; the replay did not answer the second prompt.
    assert list(await _recorded(app, "r1")) == [first.key]
    still = await app.ainvoke(
        {"resume": {"approved": True}, "resume_interrupt_id": second.id},
        {"thread_id": "r1"},
        response_granularity=FULL,
    )
    assert pending_interrupt(still["state"]) is None


def test_validate_against_schema_subset():
    schema = {
        "type": "object",
        "required": ["n", "tags"],
        "properties": {
            "n": {"type": ["integer", "null"]},
            "tags": {"type": "array", "items": {"enum": ["a", "b"]}},
        },
    }
    assert validate_against_schema({"n": 1, "tags": ["a"]}, schema) == []
    assert validate_against_schema({"n": None, "tags": []}, schema) == []
    assert "$.n: required" in validate_against_schema({"tags": []}, schema)
    errors = validate_against_schema({"n": True, "tags": ["c"]}, schema)
    assert any(e.startswith("$.n") for e in errors)
    assert any(e.startswith("$.tags[0]") for e in errors)
    assert validate_against_schema("x", {"type": "boolean"}) == ["$: expected boolean, got str"]
    assert validate_against_schema(1.5, {"type": "integer"})
    assert validate_against_schema(True, {"type": "number"})
