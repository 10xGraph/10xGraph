"""Checkpointers restore state into the graph's own state class.

Rows hold plain JSON data only. The class to rebuild into is bound by
``StateGraph.compile()``, never read from the row, so moving or renaming a state
class (or the framework package) cannot break stored threads, and stored data can
never choose which module gets imported.
"""

import logging
from datetime import datetime
from enum import Enum

import pytest
from pydantic import BaseModel

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.checkpointer.base_checkpointer import STATE_META_KEY
from tenxgraph.utils.constants import END


class Level(Enum):
    JUNIOR = "junior"
    SENIOR = "senior"


class JobDescription(BaseModel):
    title: str
    skills: list[str]


class CandidateState(AgentState):
    jd: JobDescription | None = None
    current_location: str = ""
    current_ctc: float = 0.0
    level: Level = Level.JUNIOR
    applied_at: datetime | None = None


class OtherState(AgentState):
    note: str = ""


def _candidate() -> CandidateState:
    return CandidateState(
        context=[Message.text_message("hi", role="user")],
        jd=JobDescription(title="Backend", skills=["python"]),
        current_location="Dhaka",
        current_ctc=12.5,
        level=Level.SENIOR,
        applied_at=datetime(2026, 10, 4, 9, 30),
    )


def _bound(state_type: type[AgentState]) -> InMemoryCheckpointer:
    cp = InMemoryCheckpointer()
    cp.bind_state_type(state_type)
    return cp


def test_unbound_checkpointer_defaults_to_agent_state():
    assert InMemoryCheckpointer().state_type is AgentState


def test_bind_rejects_non_agent_state_class():
    with pytest.raises(TypeError):
        InMemoryCheckpointer().bind_state_type(JobDescription)  # type: ignore[arg-type]


def test_rebinding_same_class_is_allowed():
    cp = _bound(CandidateState)
    cp.bind_state_type(CandidateState)
    assert cp.state_type is CandidateState


def test_rebinding_different_class_raises():
    cp = _bound(CandidateState)
    with pytest.raises(ValueError, match="CandidateState"):
        cp.bind_state_type(OtherState)


def test_explicit_type_mismatch_raises():
    cp = InMemoryCheckpointer[CandidateState]()
    with pytest.raises(ValueError, match="CandidateState"):
        cp.bind_state_type(OtherState)


def test_explicit_base_type_accepts_subclass():
    cp = InMemoryCheckpointer[AgentState]()
    cp.bind_state_type(CandidateState)
    assert cp.state_type is CandidateState


def test_explicit_matching_type_is_accepted():
    cp = InMemoryCheckpointer[CandidateState]()
    cp.bind_state_type(CandidateState)
    assert cp.state_type is CandidateState


def test_encode_stores_data_and_meta_but_no_class_path():
    data = _bound(CandidateState)._encode_state(_candidate())
    assert data[STATE_META_KEY] == {"format": 1, "class": "CandidateState"}
    assert "__class_path__" not in data
    assert data["current_ctc"] == 12.5


def test_roundtrip_keeps_custom_fields_and_types():
    cp = _bound(CandidateState)
    original = _candidate()
    restored = cp._decode_state(cp._encode_state(original))
    assert type(restored) is CandidateState
    assert restored == original
    assert isinstance(restored.jd, JobDescription)
    assert restored.level is Level.SENIOR


def test_legacy_class_path_row_loads_without_importing_it():
    # Rows written before this change carry a module path. It must be ignored,
    # not imported: this one points at a module that does not exist.
    row = _candidate().model_dump(mode="json")
    row["__class_path__"] = "tenxgraph.gone.module.CandidateState"
    restored = _bound(CandidateState)._decode_state(row)
    assert type(restored) is CandidateState
    assert restored.current_location == "Dhaka"


def test_class_name_mismatch_logs_warning(caplog):
    row = _bound(OtherState)._encode_state(OtherState(note="x"))
    with caplog.at_level(logging.WARNING):
        _bound(CandidateState)._decode_state(row)
    assert "OtherState" in caplog.text
    assert "CandidateState" in caplog.text


def test_decode_does_not_mutate_input_row():
    cp = _bound(CandidateState)
    row = cp._encode_state(_candidate())
    cp._decode_state(row)
    assert STATE_META_KEY in row


def test_compile_binds_graph_state_class():
    async def node(state: CandidateState):
        return state

    graph = StateGraph(CandidateState())
    graph.add_node("MAIN", node)
    graph.add_edge("MAIN", END)
    graph.set_entry_point("MAIN")
    cp = InMemoryCheckpointer()
    graph.compile(checkpointer=cp)
    assert cp.state_type is CandidateState
