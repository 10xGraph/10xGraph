"""Multi-agent handoffs never leave an unanswered tool call or a trailing assistant turn.

Scripted Gemini end to end: ``Agent._call_google`` is patched to record each request and
return Google-shaped mock responses, so nodes, converters, routing and the checkpointer
all run, on ``ainvoke`` and ``astream``. Nothing here calls a real provider.

Every captured request is checked with ``assert_valid_provider_request`` and the saved
state and messages table with ``assert_valid_saved_history``. Covered: SwarmAgent (the
maintainer's TRIAGE / WEATHER / PLANNER shape), SupervisorTeamAgent, and a hand-built graph
whose ToolNode runs ``create_handoff_tool``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.graph import Agent, StateGraph, ToolNode
from tenxgraph.core.graph.utils.tool_dispatch import handoff_result_text
from tenxgraph.core.state import Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.prebuilt.agent import SupervisorTeamAgent, SwarmAgent
from tenxgraph.prebuilt.agent.supervisor_team import WorkerConfig
from tenxgraph.prebuilt.agent.swarm import SwarmMemberConfig
from tenxgraph.prebuilt.tools.handoff import create_handoff_tool
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils import ResponseGranularity
from tenxgraph.utils.constants import END
from tests.adapters.test_google_genai_converter import (
    MockCandidate,
    MockContent,
    MockFunctionCall,
    MockGenerateContentResponse,
    MockPart,
)
from tests.helpers.provider_request_rules import (
    assert_valid_provider_request,
    assert_valid_saved_history,
)


MODES = ["ainvoke", "astream"]
FULL = ResponseGranularity.FULL
WEATHER_RUNS: list[str] = []


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    WEATHER_RUNS.append(city)
    return f"sunny in {city}"


def _response(spec) -> MockGenerateContentResponse:
    """spec: str (text), or a list of ("tool", name, args) calls in one reply."""
    if isinstance(spec, list):
        parts = [MockPart(function_call=MockFunctionCall(name, args)) for _, name, args in spec]
    else:
        parts = [MockPart(text=spec)]
    return MockGenerateContentResponse(candidates=[MockCandidate(content=MockContent(parts))])


def call(name: str, **args) -> list:
    return [("tool", name, args)]


class Script:
    """Scripted Gemini: replies per agent name, and a log of every request."""

    def __init__(self, replies: dict[str, list]):
        self.replies = {name: list(items) for name, items in replies.items()}
        self.names: dict[int, str] = {}
        self.requests: list[dict] = []

    def name(self, agent: Agent, name: str) -> Agent:
        self.names[id(agent)] = name
        return agent

    async def __call__(self, agent, messages, tools=None, stream=False, **kwargs):
        name = self.names.get(id(agent), "SUPERVISOR")
        self.requests.append({"node": name, "messages": [dict(m) for m in messages]})
        resp = _response(self.replies[name].pop(0))
        if not stream:
            return resp

        async def gen():
            yield resp

        return gen()

    def nodes(self) -> list[str]:
        return [r["node"] for r in self.requests]

    def of(self, node: str) -> list[list[dict]]:
        return [r["messages"] for r in self.requests if r["node"] == node]


async def run(script: Script, build, mode: str, thread: str):
    """Build the app inside the patches, run one user turn, return (state, messages)."""
    WEATHER_RUNS.clear()
    cp = InMemoryCheckpointer()
    with (
        patch.object(Agent, "_create_client", return_value=MagicMock()),
        patch.object(Agent, "_call_google", lambda self, *a, **k: script(self, *a, **k)),
    ):
        app = build(script).compile(checkpointer=cp)
        config = {"thread_id": f"{thread}-{mode}", "user_id": "u"}
        data = {"messages": [Message.text_message("Plan a day in Paris for me.")]}
        if mode == "ainvoke":
            await app.ainvoke(data, config, response_granularity=FULL)
        else:
            _ = [c async for c in app.astream(data, config, response_granularity=FULL)]
        state = await cp.aget_state(config)
        saved = await cp.alist_messages(config)
    return state, saved


def check_all(script: Script, state, saved) -> None:
    for request in script.requests:
        assert_valid_provider_request(request["messages"])
    assert_valid_saved_history(state.context)
    assert_valid_saved_history(saved)
    ids = [m.message_id for m in saved]
    assert len(ids) == len(set(ids)), "a message was saved twice"


def tool_outputs(context: list[Message]) -> list[str]:
    return [
        str(b.output)
        for m in context
        if m.role == "tool"
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]


# ---------------------------------------------------------------------------
# SwarmAgent: the maintainer's TRIAGE / WEATHER / PLANNER shape
# ---------------------------------------------------------------------------


def build_swarm(script: Script) -> SwarmAgent:
    def agent(name: str, **kw) -> Agent:
        return script.name(
            Agent(
                model="gemini-2.5-flash",
                provider="google",
                system_prompt=[{"role": "system", "content": f"You are {name}."}],
                **kw,
            ),
            name,
        )

    return SwarmAgent(
        members={
            "TRIAGE": SwarmMemberConfig(
                agent=agent("TRIAGE"),
                can_handoff_to=["WEATHER", "PLANNER"],
                description="Front desk.",
            ),
            "WEATHER": SwarmMemberConfig(
                agent=agent("WEATHER", tool_node=ToolNode([get_weather])),
                can_handoff_to=["PLANNER"],
                description="Weather lookups.",
            ),
            "PLANNER": SwarmMemberConfig(
                agent=agent("PLANNER"),
                can_handoff_to=["WEATHER"],
                description="Trip advice.",
            ),
        },
        entry="TRIAGE",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_swarm_triage_hands_off_to_planner(mode):
    """The playground failure: TRIAGE (no tools) calls transfer_to_planner."""
    script = Script({"TRIAGE": [call("transfer_to_planner")], "PLANNER": ["Walk the Seine."]})
    state, saved = await run(script, build_swarm, mode, "swarm-1")

    assert script.nodes() == ["TRIAGE", "PLANNER"]
    check_all(script, state, saved)
    planner_request = script.of("PLANNER")[0]
    assert planner_request[-1]["role"] == "tool"
    assert tool_outputs(state.context) == [handoff_result_text("PLANNER")]
    assert state.context[-1].text() == "Walk the Seine."


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_swarm_weather_tool_then_handoff_to_planner(mode):
    """TRIAGE -> WEATHER (tool call, tool result) -> PLANNER."""
    script = Script(
        {
            "TRIAGE": [call("transfer_to_weather")],
            "WEATHER": [call("get_weather", city="Paris"), call("transfer_to_planner")],
            "PLANNER": ["Sunny: wear a hat."],
        }
    )
    state, saved = await run(script, build_swarm, mode, "swarm-2")

    assert script.nodes() == ["TRIAGE", "WEATHER", "WEATHER", "PLANNER"]
    check_all(script, state, saved)
    assert WEATHER_RUNS == ["Paris"]
    assert tool_outputs(state.context) == [
        handoff_result_text("WEATHER"),
        "sunny in Paris",
        handoff_result_text("PLANNER"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_swarm_member_without_tools_hands_back(mode):
    """PLANNER (no own tools) hands off to WEATHER, which answers and ends."""
    script = Script(
        {
            "TRIAGE": [call("transfer_to_planner")],
            "PLANNER": [call("transfer_to_weather")],
            "WEATHER": [call("get_weather", city="Paris"), "It is sunny in Paris."],
        }
    )
    state, saved = await run(script, build_swarm, mode, "swarm-3")

    assert script.nodes() == ["TRIAGE", "PLANNER", "WEATHER", "WEATHER"]
    check_all(script, state, saved)
    assert tool_outputs(state.context).count(handoff_result_text("WEATHER")) == 1
    assert state.context[-1].text() == "It is sunny in Paris."


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_swarm_handoff_mixed_with_regular_tool_call(mode):
    """A handoff next to a regular call: the handoff wins, the regular call is not run."""
    script = Script(
        {
            "TRIAGE": [call("transfer_to_weather")],
            "WEATHER": [
                [
                    ("tool", "get_weather", {"city": "Paris"}),
                    ("tool", "transfer_to_planner", {}),
                ]
            ],
            "PLANNER": ["I need the weather first."],
        }
    )
    state, saved = await run(script, build_swarm, mode, "swarm-4")

    assert script.nodes() == ["TRIAGE", "WEATHER", "PLANNER"]
    check_all(script, state, saved)
    assert WEATHER_RUNS == []
    assert tool_outputs(state.context) == [
        handoff_result_text("WEATHER"),
        "Not run: control was transferred to PLANNER.",
        handoff_result_text("PLANNER"),
    ]


def test_handoff_result_tells_the_target_it_is_in_charge():
    text = handoff_result_text("WEATHER")
    assert text.startswith("Transferred to WEATHER.")
    assert "You are now WEATHER" in text
    assert "answer the user's latest request yourself" in text
    assert "call your tools" in text
    assert "\u2014" not in text


def test_swarm_graph_shows_the_handoff_nodes():
    script = Script({})
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        swarm = build_swarm(script)
        swarm.compile()
    graph = swarm._graph
    assert {"TRIAGE_HANDOFF", "WEATHER_HANDOFF", "PLANNER_HANDOFF", "WEATHER_TOOL"} <= set(
        graph.nodes
    )
    edges = {(e.from_node, e.to_node) for e in graph.edges}
    assert ("TRIAGE", "TRIAGE_HANDOFF") in edges
    assert ("TRIAGE_HANDOFF", "PLANNER") in edges
    assert ("TRIAGE_HANDOFF", "WEATHER") in edges
    assert ("WEATHER_HANDOFF", "PLANNER") in edges
    assert ("WEATHER_HANDOFF", "WEATHER") not in edges


# ---------------------------------------------------------------------------
# SupervisorTeamAgent
# ---------------------------------------------------------------------------


def build_team(script: Script) -> SupervisorTeamAgent:
    def worker(name: str, **kw) -> Agent:
        return script.name(
            Agent(
                model="gemini-2.5-flash",
                provider="google",
                system_prompt=[{"role": "system", "content": f"You are {name}."}],
                **kw,
            ),
            name,
        )

    return SupervisorTeamAgent(
        supervisor_model="gemini-2.5-flash",
        provider="google",
        workers={
            "RESEARCHER": WorkerConfig(
                agent=worker("RESEARCHER", tool_node=ToolNode([get_weather])),
                description="Looks up the weather.",
            ),
            "WRITER": WorkerConfig(agent=worker("WRITER"), description="Writes the plan."),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_supervisor_team_requests_are_valid(mode):
    script = Script(
        {
            "SUPERVISOR": ["RESEARCHER", "WRITER", "FINISH"],
            "RESEARCHER": [call("get_weather", city="Paris"), "Paris is sunny."],
            "WRITER": ["Plan: a sunny walk."],
        }
    )
    state, saved = await run(script, build_team, mode, "team-1")

    assert script.nodes() == [
        "SUPERVISOR",
        "RESEARCHER",
        "RESEARCHER",
        "SUPERVISOR",
        "WRITER",
        "SUPERVISOR",
    ]
    check_all(script, state, saved)
    assert WEATHER_RUNS == ["Paris"]
    # The transient request notes are never saved.
    assert [m.role for m in state.context] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "assistant",
    ]
    assert state.context[-1].text() == "Plan: a sunny walk."


# ---------------------------------------------------------------------------
# Hand-built graph: a ToolNode runs create_handoff_tool
# ---------------------------------------------------------------------------


def lookup(topic: str) -> str:
    """Look a topic up."""
    WEATHER_RUNS.append(topic)
    return f"found {topic}"


def build_hand_built(script: Script) -> StateGraph:
    main = script.name(
        Agent(
            model="gemini-2.5-flash",
            provider="google",
            system_prompt=[{"role": "system", "content": "You route."}],
            tool_node=ToolNode([create_handoff_tool("worker"), lookup]),
        ),
        "MAIN",
    )
    worker = script.name(
        Agent(
            model="gemini-2.5-flash",
            provider="google",
            system_prompt=[{"role": "system", "content": "You work."}],
        ),
        "worker",
    )

    def route(state) -> str:
        last = state.context[-1]
        return "TOOL" if last.role == "assistant" and last.tools_calls else END

    graph = StateGraph()
    graph.add_node("MAIN", main)
    graph.add_node("TOOL", main.get_tool_node())
    graph.add_node("worker", worker)
    graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.add_edge("worker", END)
    graph.set_entry_point("MAIN")
    return graph


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "calls",
    [
        [("tool", "transfer_to_worker", {})],
        [("tool", "lookup", {"topic": "x"}), ("tool", "transfer_to_worker", {})],
    ],
    ids=["handoff", "mixed"],
)
async def test_hand_built_handoff_tool_requests_are_valid(mode, calls):
    script = Script({"MAIN": [calls], "worker": ["Done."]})
    state, saved = await run(script, build_hand_built, mode, f"hand-{len(calls)}")

    assert script.nodes() == ["MAIN", "worker"]
    check_all(script, state, saved)
    assert WEATHER_RUNS == []
    outputs = tool_outputs(state.context)
    assert outputs[-1] == handoff_result_text("worker")
    assert outputs.count(handoff_result_text("worker")) == 1
    assert len(outputs) == len(calls)
