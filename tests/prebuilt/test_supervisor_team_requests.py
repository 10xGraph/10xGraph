"""SupervisorTeamAgent: what the supervisor's own call sees, and when the turn ends.

A live Gemini run re-dispatched WEATHER and PLANNER after both had answered. The
supervisor's request held the workers' replies as unlabelled assistant turns, followed by
its past decisions replayed as bare ``WEATHER`` / ``PLANNER`` assistant turns, so it could
not tell who said what or that the user was answered. These tests capture every provider
request with a scripted Gemini (no real API) and check the supervisor's view, the
back-to-back repeat rule and the RESPOND reply for a greeting.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.graph import Agent, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.message_block import ToolResultBlock
from tenxgraph.prebuilt.agent import SupervisorTeamAgent
from tenxgraph.prebuilt.agent.supervisor_team import (
    _RESPOND_NODE,
    _ROUTE_HISTORY_KEY,
    SUPERVISOR_REQUEST_NOTE,
    WorkerConfig,
    _make_supervisor_node,
    _supervisor_view,
)
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


def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    return f"The weather in {location} is sunny"


def _response(spec) -> MockGenerateContentResponse:
    if isinstance(spec, tuple):
        part = MockPart(function_call=MockFunctionCall(spec[1], spec[2]))
    else:
        part = MockPart(text=spec)
    return MockGenerateContentResponse(candidates=[MockCandidate(content=MockContent([part]))])


class Script:
    """Scripted Gemini that tells the agents apart by their system prompt."""

    def __init__(self, replies: dict[str, list]):
        self.replies = {name: list(items) for name, items in replies.items()}
        self.requests: list[tuple[str, list[dict]]] = []

    @staticmethod
    def who(messages: list[dict]) -> str:
        system = " ".join(str(m.get("content")) for m in messages if m.get("role") == "system")
        if "supervisor agent" in system:
            return "SUPERVISOR"
        if "No worker is needed" in system:
            return _RESPOND_NODE
        return "WEATHER" if "weather specialist" in system else "PLANNER"

    async def __call__(self, agent, messages, tools=None, stream=False, **kwargs):
        name = self.who(messages)
        self.requests.append((name, [dict(m) for m in messages]))
        resp = _response(self.replies[name].pop(0))
        if not stream:
            return resp

        async def gen():
            yield resp

        return gen()

    def of(self, name: str) -> list[list[dict]]:
        return [msgs for who, msgs in self.requests if who == name]

    def order(self) -> list[str]:
        return [who for who, _ in self.requests]


def build_team() -> SupervisorTeamAgent:
    """The maintainer's playground team: WEATHER (with a tool) and PLANNER."""
    weather = Agent(
        model="gemini-2.5-flash",
        provider="google",
        tool_node=ToolNode([get_weather]),
        system_prompt=[{"role": "system", "content": "You are a weather specialist."}],
    )
    planner = Agent(
        model="gemini-2.5-flash",
        provider="google",
        system_prompt=[{"role": "system", "content": "You are a trip planner."}],
    )
    return SupervisorTeamAgent(
        supervisor_model="gemini-2.5-flash",
        provider="google",
        workers={
            "WEATHER": WorkerConfig(agent=weather, description="Looks up the weather."),
            "PLANNER": WorkerConfig(agent=planner, description="Gives trip advice."),
        },
        max_rounds=6,
    )


async def run(script: Script, mode: str, text: str):
    cp = InMemoryCheckpointer()
    with (
        patch.object(Agent, "_create_client", return_value=MagicMock()),
        patch.object(Agent, "_call_google", lambda self, *a, **k: script(self, *a, **k)),
    ):
        app = build_team().compile(checkpointer=cp)
        config = {"thread_id": f"t-{mode}", "user_id": "u"}
        data = {"messages": [Message.text_message(text)]}
        if mode == "ainvoke":
            await app.ainvoke(data, config, response_granularity=FULL)
        else:
            _ = [c async for c in app.astream(data, config, response_granularity=FULL)]
        return await cp.aget_state(config), await cp.alist_messages(config)


def check_requests(script: Script, state, saved) -> None:
    for _, messages in script.requests:
        assert_valid_provider_request(messages)
    assert_valid_saved_history(state.context)
    assert_valid_saved_history(saved)


def last_text(messages: list[dict]) -> str:
    return str(messages[-1].get("content"))


# ---------------------------------------------------------------------------
# End to end: the playground scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_weather_then_advice_runs_each_worker_once(mode):
    script = Script(
        {
            "SUPERVISOR": ["WEATHER", "PLANNER", "FINISH"],
            "WEATHER": [("tool", "get_weather", {"location": "Khulna"}), "Khulna is sunny."],
            "PLANNER": ["Wear light clothing and sunscreen."],
        }
    )
    state, saved = await run(script, mode, "I'm going to Khulna tomorrow, what should I wear?")

    assert script.order() == [
        "SUPERVISOR",
        "WEATHER",
        "WEATHER",
        "SUPERVISOR",
        "PLANNER",
        "SUPERVISOR",
    ]
    check_requests(script, state, saved)

    # The last supervisor request names each worker's reply, in order, and the workers
    # already run; no bare routing tokens and no unlabelled worker turns.
    final = script.of("SUPERVISOR")[-1]
    note = last_text(final)
    assert final[-1]["role"] == "user"
    assert note.index("WEATHER replied: Khulna is sunny.") < note.index(
        "PLANNER replied: Wear light clothing and sunscreen."
    )
    assert "Workers already run for it, in order: WEATHER, PLANNER." in note
    assert note.endswith(SUPERVISOR_REQUEST_NOTE)
    roles = [m["role"] for m in final if m["role"] != "system"]
    assert roles == ["user", "user"]  # the user's message, then the labelled summary

    assert state.context[-1].text() == "Wear light clothing and sunscreen."


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_repeating_the_worker_that_just_ran_finishes(mode):
    script = Script(
        {
            "SUPERVISOR": ["PLANNER", "PLANNER"],
            "PLANNER": ["Wear light clothing."],
        }
    )
    state, saved = await run(script, mode, "What should I wear in Khulna?")

    assert script.order() == ["SUPERVISOR", "PLANNER", "SUPERVISOR"]
    check_requests(script, state, saved)
    assert state.context[-1].text() == "Wear light clothing."


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_greeting_gets_a_reply_from_respond(mode):
    script = Script(
        {
            "SUPERVISOR": ["FINISH"],
            _RESPOND_NODE: ["Hello! I can look up the weather or help plan a trip."],
        }
    )
    state, saved = await run(script, mode, "HI")

    assert script.order() == ["SUPERVISOR", _RESPOND_NODE]
    check_requests(script, state, saved)
    reply = state.context[-1]
    assert reply.role == "assistant"
    assert reply.text() == "Hello! I can look up the weather or help plan a trip."
    assert reply.metadata.get("node_name") == _RESPOND_NODE
    # The RESPOND agent is told what the team can do.
    respond_system = script.of(_RESPOND_NODE)[0][0]
    assert respond_system["role"] == "system"
    assert "WEATHER" in str(respond_system["content"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_finish_after_a_worker_does_not_respond_again(mode):
    script = Script(
        {
            "SUPERVISOR": ["PLANNER", "FINISH"],
            "PLANNER": ["Wear light clothing."],
        }
    )
    state, saved = await run(script, mode, "What should I wear in Khulna?")

    assert _RESPOND_NODE not in script.order()
    check_requests(script, state, saved)


# ---------------------------------------------------------------------------
# Units: the supervisor's view and decisions
# ---------------------------------------------------------------------------


def _reply(text: str, node: str) -> Message:
    msg = Message.text_message(text, role="assistant")
    msg.metadata["node_name"] = node
    return msg


def test_view_is_unchanged_before_any_worker_ran():
    context = [Message.text_message("hi")]
    assert _supervisor_view(context, []) == context


def test_view_labels_replies_in_order_and_keeps_earlier_turns_as_text():
    earlier_call = Message.text_message("", role="assistant")
    earlier_call.tools_calls = [{"id": "c1", "function": {"name": "get_weather"}}]
    context = [
        Message.text_message("first question"),
        earlier_call,
        Message.tool_message([ToolResultBlock(call_id="c1", output="sunny")]),
        _reply("first answer", "WEATHER"),
        Message.text_message("second question"),
        _reply("It is sunny.", "WEATHER"),
        _reply("Wear a hat.", "PLANNER"),
    ]
    view = _supervisor_view(context, ["WEATHER", "PLANNER"])

    assert [m.role for m in view] == ["user", "assistant", "user", "user"]
    assert view[1].text() == "first answer"
    assert not view[1].tools_calls
    note = view[-1].text()
    assert "WEATHER replied: It is sunny.\n\nPLANNER replied: Wear a hat." in note
    assert "in order: WEATHER, PLANNER." in note
    assert "first answer" not in note


def test_view_when_a_worker_ran_but_said_nothing():
    context = [Message.text_message("q")]
    note = _supervisor_view(context, ["WEATHER"])[-1].text()
    assert "No worker has replied yet." in note
    assert "in order: WEATHER." in note


class _FakeConverter:
    def __init__(self, text: str):
        self.text = text

    async def invoke(self) -> Message:
        return Message.text_message(self.text, role="assistant")


class _FakeSupervisor:
    def __init__(self, text: str):
        self.text = text
        self.seen: list[list[Message]] = []

    async def execute(self, state, config):
        self.seen.append(list(state.context))
        return _FakeConverter(self.text)


@pytest.mark.asyncio
async def test_node_view_is_transient():
    sup = _FakeSupervisor("FINISH")
    node = _make_supervisor_node(sup, ["WEATHER"], max_rounds=10)
    state = AgentState()
    state.execution_meta.internal_data[_ROUTE_HISTORY_KEY] = ["WEATHER"]
    state.context = [Message.text_message("q"), _reply("sunny", "WEATHER")]
    before = [m.message_id for m in state.context]

    await node(state, {})

    assert "WEATHER replied: sunny" in sup.seen[0][-1].text()
    assert [m.message_id for m in state.context] == before


@pytest.mark.asyncio
async def test_node_repeat_of_last_worker_is_end():
    node = _make_supervisor_node(_FakeSupervisor("WEATHER"), ["WEATHER", "PLANNER"], 10)
    state = AgentState()
    state.execution_meta.internal_data[_ROUTE_HISTORY_KEY] = ["WEATHER"]
    cmd = await node(state, {})
    assert cmd.goto == END
    assert state.execution_meta.internal_data[_ROUTE_HISTORY_KEY] == ["WEATHER"]


@pytest.mark.asyncio
async def test_node_can_return_to_an_earlier_worker():
    node = _make_supervisor_node(_FakeSupervisor("WEATHER"), ["WEATHER", "PLANNER"], 10)
    state = AgentState()
    state.execution_meta.internal_data[_ROUTE_HISTORY_KEY] = ["WEATHER", "PLANNER"]
    cmd = await node(state, {})
    assert cmd.goto == "WEATHER"


@pytest.mark.asyncio
async def test_node_finish_with_no_worker_goes_to_respond_node():
    node = _make_supervisor_node(_FakeSupervisor("FINISH"), ["WEATHER"], 10, respond_node="R")
    cmd = await node(AgentState(), {})
    assert cmd.goto == "R"


@pytest.mark.asyncio
async def test_node_finish_after_a_worker_ends():
    node = _make_supervisor_node(_FakeSupervisor("FINISH"), ["WEATHER"], 10, respond_node="R")
    state = AgentState()
    state.execution_meta.internal_data[_ROUTE_HISTORY_KEY] = ["WEATHER"]
    cmd = await node(state, {})
    assert cmd.goto == END


def test_respond_is_a_reserved_worker_name():
    with pytest.raises(ValueError, match="reserved"):
        SupervisorTeamAgent(
            supervisor_model="gemini-2.5-flash",
            workers={_RESPOND_NODE: WorkerConfig(agent=MagicMock())},
        )
