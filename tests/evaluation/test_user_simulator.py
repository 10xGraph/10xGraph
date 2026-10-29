"""Tests for UserSimulator and BatchSimulator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any
import asyncio
import json
import uuid

from tenxgraph.qa.evaluation.simulators.user_simulator import (
    UserSimulator,
    BatchSimulator,
    ConversationScenario,
    SimulationResult
)
from tenxgraph.qa.evaluation.config.eval_config import UserSimulatorConfig
from tenxgraph.qa.evaluation.criteria.base import BaseCriterion, CriterionResult
from tenxgraph.core.graph.compiled_graph import CompiledGraph
from tenxgraph.core.state import Message

class MockCriterion(BaseCriterion):
    """Mock criterion for evaluation."""
    def __init__(self, name="mock_criterion"):
        super().__init__()
        self.name = name

    async def evaluate(self, execution_result: Any, eval_case: Any) -> CriterionResult:
        return CriterionResult(
            criterion=self.name,
            score=0.9,
            passed=True,
            details={"notes": "good response"}
        )

@pytest.mark.asyncio
async def test_user_simulator_init():
    # 1. Init with defaults
    sim = UserSimulator()
    assert sim.model == "gemini/gemini-2.5-flash"
    assert sim.temperature == 0.7
    assert sim.max_turns == 10
    assert sim.api_style == "responses"

    # 2. Init with config
    config = UserSimulatorConfig(
        model="gpt-4o",
        temperature=0.5,
        max_invocations=5
    )
    # Mocking config object to have api_style
    config.api_style = "chat"
    sim_config = UserSimulator(config=config)
    assert sim_config.model == "gpt-4o"
    assert sim_config.temperature == 0.5
    assert sim_config.max_turns == 5
    assert sim_config.api_style == "chat"

@pytest.mark.asyncio
async def test_user_simulator_run_success():
    # Prepare scenario
    scenario = ConversationScenario(
        scenario_id="test_scen",
        description="A test scenario",
        starting_prompt="Hello agent",
        goals=["Say hello back"],
        max_turns=3
    )

    # Mock CompiledGraph
    mock_graph = MagicMock(spec=CompiledGraph)
    msg = Message.text_message("Hello back!", role="assistant")
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

    # Mock call_llm response for goal check
    mock_result_json = json.dumps({"achieved": True, "reasoning": "Agent said hello back"})
    
    criterion = MockCriterion("test_criterion")
    sim = UserSimulator(criteria=[criterion])

    with patch("tenxgraph.qa.evaluation.simulators.user_simulator.call_llm") as mock_call:
        mock_call.return_value = (mock_result_json, 10, 20, 0)
        
        result = await sim.run(mock_graph, scenario)
        
        assert result.completed is True
        assert "Say hello back" in result.goals_achieved
        assert result.turns == 1
        assert result.criterion_scores["test_criterion"] == 0.9
        assert result.duration_seconds > 0

@pytest.mark.asyncio
async def test_user_simulator_run_no_starting_prompt():
    scenario = ConversationScenario(
        scenario_id="test_scen_no_start",
        description="Test no start",
        starting_prompt="",
        goals=["Finish"],
        max_turns=2
    )

    mock_graph = MagicMock(spec=CompiledGraph)
    msg = Message.text_message("Got it", role="assistant")
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

    sim = UserSimulator()

    with patch("tenxgraph.qa.evaluation.simulators.user_simulator.call_llm") as mock_call:
        mock_call.side_effect = [
            ("Start prompt", 5, 5, 0),
            (json.dumps({"achieved": True, "reasoning": "Done"}), 10, 10, 0)
        ]
        
        result = await sim.run(mock_graph, scenario)
        assert result.completed is True
        assert result.conversation[0]["content"] == "Start prompt"

@pytest.mark.asyncio
async def test_user_simulator_run_max_turns():
    scenario = ConversationScenario(
        scenario_id="test_max_turns",
        description="Test max turns limit",
        starting_prompt="Hello",
        goals=["Goal never met"],
        max_turns=2
    )

    mock_graph = MagicMock(spec=CompiledGraph)
    msg = Message.text_message("Agent reply", role="assistant")
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

    sim = UserSimulator()

    with patch("tenxgraph.qa.evaluation.simulators.user_simulator.call_llm") as mock_call:
        mock_call.side_effect = [
            (json.dumps({"achieved": False, "reasoning": "Not yet"}), 5, 5, 0),
            ("User follow-up", 5, 5, 0),
            (json.dumps({"achieved": False, "reasoning": "Still not yet"}), 5, 5, 0),
            ("User second follow-up", 5, 5, 0),
        ]
        
        result = await sim.run(mock_graph, scenario)
        assert result.completed is False
        assert result.turns == 2
        assert len(result.conversation) == 4

@pytest.mark.asyncio
async def test_user_simulator_graph_exception():
    scenario = ConversationScenario(
        scenario_id="test_graph_err",
        starting_prompt="Hello",
        goals=["Done"]
    )
    mock_graph = MagicMock(spec=CompiledGraph)
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Graph crash"))

    sim = UserSimulator()
    result = await sim.run(mock_graph, scenario)
    assert result.completed is False
    assert "Graph crash" in result.error

@pytest.mark.asyncio
async def test_user_simulator_check_goals_fallback():
    sim = UserSimulator()
    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "the magic word is banana"}
    ]
    
    with patch("tenxgraph.qa.evaluation.simulators.user_simulator.call_llm", side_effect=Exception("LLM down")):
        achieved, usage = await sim._check_goals(
            all_goals=["banana", "apple"],
            achieved=[],
            conversation=conversation
        )
        assert "banana" in achieved
        assert "apple" not in achieved

@pytest.mark.asyncio
async def test_user_simulator_general_exception():
    scenario = ConversationScenario(scenario_id="crash")
    sim = UserSimulator()
    mock_graph = MagicMock(spec=CompiledGraph)
    mock_graph.ainvoke = AsyncMock(return_value={"messages": []})

    # Cause crash inside try block by raising TypeError in _generate_initial_message
    with patch.object(sim, "_generate_initial_message", side_effect=TypeError("Crashed inside")):
        result = await sim.run(mock_graph, scenario)
        assert result.completed is False
        assert "Crashed inside" in result.error

def test_extract_response_fallback_paths():
    sim = UserSimulator()
    # 1. empty result
    assert sim._extract_response({}) == ""
    # 2. messages is empty
    assert sim._extract_response({"messages": []}) == ""
    # 3. assistant content block list
    msg1 = type("Message", (), {
        "role": "assistant",
        "content": [
            type("Block", (), {"text": "Hello"}),
            type("Block", (), {})
        ]
    })
    assert sim._extract_response({"messages": [msg1]}) == "Hello"

    # 4. plain dict format
    msg2 = {"role": "assistant", "content": "Dict text"}
    assert sim._extract_response({"messages": [msg2]}) == "Dict text"

@pytest.mark.asyncio
async def test_batch_simulator():
    sim = UserSimulator()
    batch = BatchSimulator(simulator=sim, max_concurrency=2)
    
    mock_graph = MagicMock(spec=CompiledGraph)
    msg = Message.text_message("Done", role="assistant")
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [msg]})

    scenarios = [
        ConversationScenario(scenario_id="s1", starting_prompt="p1", goals=["g1"]),
        ConversationScenario(scenario_id="s2", starting_prompt="p2", goals=["g2"])
    ]

    with patch("tenxgraph.qa.evaluation.simulators.user_simulator.call_llm") as mock_call:
        mock_call.return_value = (json.dumps({"achieved": True, "reasoning": "ok"}), 1, 1, 0)
        
        results = await batch.run_batch(mock_graph, scenarios)
        assert len(results) == 2
        assert results[0].scenario_id == "s1"
        assert results[1].scenario_id == "s2"
        
        summary = batch.summary(results)
        assert summary["total_scenarios"] == 2
        assert summary["completed"] == 2
        assert summary["completion_rate"] == 1.0
        assert summary["errors"] == 0


# ---------------------------------------------------------------------------
# Progress events, concurrency and history
# ---------------------------------------------------------------------------


def _llm_patch(goal_answers):
    """call_llm stub: goal checks answer by goal text, anything else is a user message."""

    async def fake(model, prompt, **kwargs):
        if "evaluating whether a conversation goal" in prompt:
            for goal, achieved in goal_answers.items():
                if f'GOAL: "{goal}"' in prompt:
                    return json.dumps({"achieved": achieved}), 1, 1, 0
        return "next user message", 1, 1, 0

    return patch("tenxgraph.qa.evaluation.simulators.user_simulator.call_llm", side_effect=fake)


@pytest.mark.asyncio
async def test_on_event_order_and_payloads():
    scenario = ConversationScenario(
        scenario_id="ev", description="d", starting_prompt="Hi", goals=["a", "b"], max_turns=3
    )
    graph = MagicMock(spec=CompiledGraph)
    graph.ainvoke = AsyncMock(
        return_value={"messages": [Message.text_message("reply", role="assistant")]}
    )
    events = []
    sim = UserSimulator(criteria=[MockCriterion("c1"), MockCriterion("c2")])
    with _llm_patch({"a": True, "b": False}):
        await sim.run(graph, scenario, on_event=lambda e, d: events.append((e, d)))

    names = [e for e, _ in events]
    turn_block = ["turn_started", "agent_replied", "goals_checked"]
    assert names == [*turn_block, *turn_block, *turn_block, "evaluating"]
    assert events[0][1] == {"turn": 1, "max_turns": 3, "user_message": "Hi"}
    assert events[1][1] == {"turn": 1, "response": "reply"}
    assert events[2][1] == {"turn": 1, "newly_achieved": ["a"], "achieved": 1, "total": 2}
    assert events[3][1]["user_message"] == "next user message"
    assert events[-1][1] == {"criteria": ["c1", "c2"]}


@pytest.mark.asyncio
async def test_on_event_agent_error():
    scenario = ConversationScenario(scenario_id="e", starting_prompt="Hi", goals=["a"])
    graph = MagicMock(spec=CompiledGraph)
    graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    events = []
    sim = UserSimulator()
    result = await sim.run(graph, scenario, on_event=lambda e, d: events.append((e, d)))
    assert result.error == "Agent error: boom"
    assert [e for e, _ in events] == ["turn_started", "agent_error"]
    assert events[1][1] == {"turn": 1, "error": "boom"}


@pytest.mark.asyncio
async def test_raising_callback_does_not_break_run():
    scenario = ConversationScenario(
        scenario_id="r", starting_prompt="Hi", goals=["a"], max_turns=2
    )
    graph = MagicMock(spec=CompiledGraph)
    graph.ainvoke = AsyncMock(
        return_value={"messages": [Message.text_message("ok", role="assistant")]}
    )
    calls = []

    def bad(event, data):
        calls.append(event)
        raise ValueError("callback bug")

    sim = UserSimulator(criteria=[MockCriterion("c")])
    with _llm_patch({"a": True}):
        result = await sim.run(graph, scenario, on_event=bad)
    assert result.completed is True
    assert result.criterion_scores == {"c": 0.9}
    assert calls == ["turn_started", "agent_replied", "goals_checked", "evaluating"]


@pytest.mark.asyncio
async def test_check_goals_concurrent_keeps_order_and_fallback():
    sim = UserSimulator()
    conversation = [{"role": "user", "content": "weather"}, {"role": "assistant", "content": "sunny"}]

    async def fake(model, prompt, **kwargs):
        if 'GOAL: "first"' in prompt:
            await asyncio.sleep(0.05)  # finishes last, must still be listed first
            return json.dumps({"achieved": True}), 3, 4, 0
        if 'GOAL: "second"' in prompt:
            return json.dumps({"achieved": False}), 3, 4, 0
        if 'GOAL: "third"' in prompt:
            return json.dumps({"achieved": True}), 3, 4, 0
        raise RuntimeError("llm down")  # "weather sunny" falls back to keyword match

    goals = ["first", "second", "third", "weather sunny"]
    with patch("tenxgraph.qa.evaluation.simulators.user_simulator.call_llm", side_effect=fake):
        achieved, usage = await sim._check_goals(goals, [], conversation)
    assert achieved == ["first", "third", "weather sunny"]
    assert usage.input_tokens == 9  # the failed call adds no usage


@pytest.mark.asyncio
async def test_criteria_concurrent_order_and_error_handling():
    class Slow(MockCriterion):
        async def evaluate(self, execution_result, eval_case):
            await asyncio.sleep(0.05)
            return await super().evaluate(execution_result, eval_case)

    class Broken(MockCriterion):
        async def evaluate(self, execution_result, eval_case):
            raise RuntimeError("bad criterion")

    sim = UserSimulator(criteria=[Slow("one"), Broken("two"), MockCriterion("three")])
    scenario = ConversationScenario(scenario_id="s", description="d", goals=["g"])
    scores, details, results = await sim._evaluate_simulation(
        scenario, [{"role": "user", "content": "hi"}]
    )
    assert list(scores) == ["one", "two", "three"]
    assert scores == {"one": 0.9, "two": 0.0, "three": 0.9}
    assert details["two"] == {"error": "bad criterion"}
    assert [r.criterion for r in results] == ["one", "three"]


def _recording_graph(checkpointer):
    """Graph stub that keeps state like a checkpointer would, or not."""
    from tenxgraph.core.state.reducers import add_messages

    graph = MagicMock(spec=CompiledGraph)
    graph._checkpointer = checkpointer
    state: list = []
    seen_user_texts: list[list[str]] = []

    async def ainvoke(input_data, config=None):
        incoming = input_data["messages"]
        if checkpointer is not None:
            state[:] = add_messages(state, incoming)
            context = list(state)
        else:
            context = list(incoming)
        seen_user_texts.append([m.text() for m in context if m.role == "user"])
        reply = Message.text_message(f"answer {len(seen_user_texts)}", role="assistant")
        if checkpointer is not None:
            state.append(reply)
        return {"messages": [reply]}

    graph.ainvoke = ainvoke
    return graph, seen_user_texts


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpointer", [object(), None])
async def test_agent_sees_each_user_message_once(checkpointer):
    scenario = ConversationScenario(
        scenario_id="h", starting_prompt="m1", goals=["never"], max_turns=3
    )
    graph, seen = _recording_graph(checkpointer)
    sim = UserSimulator()
    with _llm_patch({"never": False}):
        await sim.run(graph, scenario)
    assert seen == [
        ["m1"],
        ["m1", "next user message"],
        ["m1", "next user message", "next user message"],
    ]
