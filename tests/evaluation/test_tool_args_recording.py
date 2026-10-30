"""Trajectory args hold only what the model sent, never injected or reserved parameters."""

import pytest
from injectq import Inject

from tenxgraph.core.graph.tool_node import ToolNode
from tenxgraph.core.state import AgentState
from tenxgraph.core.state.stream_emitter import StreamEmitter
from tenxgraph.qa.evaluation import EvalCase, ToolCall
from tenxgraph.qa.evaluation.collectors.trajectory_collector import (
    TrajectoryCollector,
    make_trajectory_callback,
)
from tenxgraph.qa.evaluation.config import MatchType
from tenxgraph.qa.evaluation.config.criterion_config import CriterionConfig
from tenxgraph.qa.evaluation.criteria.trajectory import TrajectoryMatchCriterion
from tenxgraph.qa.evaluation.execution.result import ExecutionResult
from tenxgraph.runtime.publisher.events import Event, EventModel, EventType


class Service:
    pass


def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: AgentState | None = None,
    config: dict | None = None,
    emit: StreamEmitter | None = None,
    svc: Service = Inject[Service],
) -> str:
    """Get the weather."""
    return f"The weather in {location} is sunny"


@pytest.mark.asyncio
async def test_real_tool_call_records_only_model_args():
    collector = TrajectoryCollector()
    _, mgr = make_trajectory_callback(collector, config={"thread_id": "t"})
    node = ToolNode([get_weather])

    await node.invoke(
        "get_weather",
        {"location": "London"},
        "call-1",
        {"thread_id": "t", "run_id": "r"},
        AgentState(),
        mgr,
    )

    assert [tc.args for tc in collector.tool_calls] == [{"location": "London"}]
    assert collector.trajectory[-1].args == {"location": "London"}

    execution = ExecutionResult(
        actual_response="ok", tool_calls=list(collector.tool_calls), trajectory=[]
    )
    case = EvalCase.single_turn(
        eval_id="c",
        user_query="q",
        expected_tools=[ToolCall(name="get_weather", args={"location": "London"})],
    )
    criterion = TrajectoryMatchCriterion(
        config=CriterionConfig.trajectory(match_type=MatchType.EXACT, check_args=True)
    )
    result = await criterion.evaluate(execution, case)
    assert result.passed


@pytest.mark.asyncio
async def test_collector_filters_every_injectable_name_from_event_args():
    collector = TrajectoryCollector()
    args = {
        "location": "Paris",
        "emit": None,
        "state": "s",
        "config": {},
        "tool_call_id": "x",
        "generated_id": 1,
        "context_manager": None,
        "publisher": None,
        "checkpointer": None,
        "store": None,
        "task_manager": None,
    }
    await collector.publish(
        EventModel(
            event=Event.TOOL_EXECUTION,
            event_type=EventType.END,
            data={"function_name": "get_weather", "args": args, "tool_call_id": "c1"},
            node_name="get_weather",
        )
    )
    assert collector.tool_calls[0].args == {"location": "Paris"}
