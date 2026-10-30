"""Loop limits of the prebuilt agents (max_rounds, max_iterations, max_attempts) are per turn."""

import pytest
from pydantic import BaseModel

from tenxgraph.core.exceptions import NodeError
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.execution_state import ExecutionState, register_per_turn_key
from tenxgraph.prebuilt.agent.plan_act_reflect import (
    _INCREMENT_ITERATIONS_NODE,
    _ITERATIONS_KEY,
    _make_increment_node,
    _make_reflect_route,
)
from tenxgraph.prebuilt.agent.structured_output import (
    _ATTEMPTS_KEY,
    _make_repair_fn,
    _make_route_fn,
)
from tenxgraph.prebuilt.agent.supervisor_team import (
    _ROUNDS_KEY,
    _ROUTE_HISTORY_KEY,
    _make_increment_rounds_node,
    _make_supervisor_node,
)
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END
from tenxgraph.utils.interrupt import interrupt


FULL = ResponseGranularity.FULL


def _user(text: str = "hi") -> dict:
    return {"messages": [Message.text_message(text)]}


async def _turn(app, thread: str, stream: bool, data: dict | None = None) -> None:
    data = data or _user()
    cfg = {"thread_id": thread}
    if stream:
        async for _ in app.astream(data, cfg):
            pass
    else:
        await app.ainvoke(data, cfg)


class _FakeConverter:
    def __init__(self, name: str):
        self.name = name

    async def invoke(self) -> Message:
        return Message.text_message(self.name, role="assistant")


class _FakeSupervisor:
    """Never says FINISH: alternates two workers (naming the one that just ran ends the turn)."""

    def __init__(self):
        self.calls = 0

    async def execute(self, state, config):
        self.calls += 1
        return _FakeConverter("A" if self.calls % 2 else "B")


def _supervisor_graph(runs: list, max_rounds: int = 2):
    async def worker(state: AgentState):
        runs.append(1)
        return []

    g = StateGraph()
    g.add_node("SUPERVISOR", _make_supervisor_node(_FakeSupervisor(), ["A", "B"], max_rounds))
    g.add_node("PRE", _make_increment_rounds_node())
    g.add_node("A", worker)
    g.add_node("B", worker)
    g.add_edge("PRE", "SUPERVISOR")
    g.add_edge("A", "PRE")
    g.add_edge("B", "PRE")
    g.set_entry_point("SUPERVISOR")
    return g.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_supervisor_max_rounds_is_per_turn(stream):
    runs: list = []
    app = _supervisor_graph(runs)
    for turn in range(1, 4):
        await _turn(app, "t", stream)
        assert len(runs) == 2 * turn


def _par_graph(plays: list, max_iterations: int = 2):
    async def plan(state: AgentState):
        plays.append(1)
        return Message.text_message("again", role="assistant")

    g = StateGraph()
    g.add_node("PLAN", plan)
    g.add_node(_INCREMENT_ITERATIONS_NODE, _make_increment_node(_ITERATIONS_KEY))
    g.add_conditional_edges(
        "PLAN",
        _make_reflect_route(max_iterations),
        {_INCREMENT_ITERATIONS_NODE: _INCREMENT_ITERATIONS_NODE, END: END},
    )
    g.add_edge(_INCREMENT_ITERATIONS_NODE, "PLAN")
    g.set_entry_point("PLAN")
    return g.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_plan_act_reflect_iterations_are_per_turn(stream):
    plays: list = []
    app = _par_graph(plays)
    for turn in range(1, 4):
        await _turn(app, "t", stream)
        assert len(plays) == 3 * turn  # initial pass + 2 iterations


class _Out(BaseModel):
    x: int


def _soa_graph(gens: list, max_attempts: int = 2):
    async def generate(state: AgentState):
        gens.append(1)
        return Message.text_message("not json", role="assistant")

    g = StateGraph()
    g.add_node("GENERATE", generate)
    g.add_node("REPAIR", _make_repair_fn(_Out))
    g.add_conditional_edges(
        "GENERATE",
        _make_route_fn(_Out, max_attempts, has_tools=False),
        {"REPAIR": "REPAIR", END: END},
    )
    g.add_edge("REPAIR", "GENERATE")
    g.set_entry_point("GENERATE")
    return g.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_structured_output_attempts_are_per_turn(stream):
    gens: list = []
    app = _soa_graph(gens)
    for turn in range(1, 4):
        await _turn(app, "t", stream)
        assert len(gens) == 3 * turn  # initial + 2 repair attempts


def _pause_graph():
    async def count(state: AgentState):
        data = state.execution_meta.internal_data
        data[_ROUNDS_KEY] = data.get(_ROUNDS_KEY, 0) + 1
        return []

    async def ask(state: AgentState):
        interrupt({"q": 1})
        return Message.text_message("done", role="assistant")

    g = StateGraph()
    g.add_node("COUNT", count)
    g.add_node("ASK", ask)
    g.add_edge("COUNT", "ASK")
    g.add_edge("ASK", END)
    g.set_entry_point("COUNT")
    return g.compile(checkpointer=InMemoryCheckpointer())


async def _rounds(app, thread: str) -> int | None:
    state = await app._checkpointer.aget_state({"thread_id": thread})
    return state.execution_meta.internal_data.get(_ROUNDS_KEY)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_resume_of_paused_thread_keeps_counters(stream):
    app = _pause_graph()
    await _turn(app, "t", stream)  # turn 1: counts 1, pauses
    assert await _rounds(app, "t") == 1
    await _turn(app, "t", stream, {"resume": "ok"})  # resume: not a new turn
    assert await _rounds(app, "t") == 1
    await _turn(app, "t", stream)  # new turn: budget starts over
    assert await _rounds(app, "t") == 1
    await _turn(app, "t", stream, {"resume": "ok"})
    assert await _rounds(app, "t") == 1


def _failing_graph(fail: list):
    async def count(state: AgentState):
        data = state.execution_meta.internal_data
        data[_ROUNDS_KEY] = data.get(_ROUNDS_KEY, 0) + 1
        return []

    async def work(state: AgentState):
        if fail:
            fail.pop()
            raise RuntimeError("tool failed")
        return Message.text_message("done", role="assistant")

    g = StateGraph()
    g.add_node("COUNT", count)
    g.add_node("WORK", work)
    g.add_edge("COUNT", "WORK")
    g.add_edge("WORK", END)
    g.set_entry_point("COUNT")
    return g.compile(checkpointer=InMemoryCheckpointer())


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_new_message_after_error_resets_counters(stream):
    app = _failing_graph(fail=[1])
    with pytest.raises(NodeError):
        await _turn(app, "t", stream)  # turn 1: counts 1, then errors
    state = await app._checkpointer.aget_state({"thread_id": "t"})
    assert state.execution_meta.status == "error"
    assert state.execution_meta.internal_data.get(_ROUNDS_KEY) == 1

    # The next message continues at the failed node, as a new turn: the failed turn's
    # count is cleared instead of eating into this turn's budget.
    await _turn(app, "t", stream)
    state = await app._checkpointer.aget_state({"thread_id": "t"})
    assert state.execution_meta.status == "completed"
    assert state.execution_meta.internal_data.get(_ROUNDS_KEY) is None


def test_reset_turn_data_clears_only_registered_keys():
    key = register_per_turn_key("test_per_turn_counter")
    meta = ExecutionState(current_node="x")
    meta.internal_data.update({key: 3, _ROUNDS_KEY: 2, _ROUTE_HISTORY_KEY: ["W"], "keep": 1})
    meta.internal_data.update({_ITERATIONS_KEY: 1, _ATTEMPTS_KEY: 1})
    meta.reset_turn_data()
    assert meta.internal_data == {"keep": 1}
