"""``model_registry`` reaches the inner Agents of the prebuilt agents; graph info lists names."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.graph.agent import Agent
from tenxgraph.core.state import AgentState
from tenxgraph.prebuilt.agent import (
    PlanActReflectAgent,
    ReactAgent,
    StructuredOutputAgent,
    SupervisorTeamAgent,
)
from tenxgraph.prebuilt.agent.supervisor_team import WorkerConfig
from tenxgraph.utils.constants import END


pytestmark = pytest.mark.asyncio

EXEC = "tenxgraph.core.graph.agent_internal.execution"

REGISTRY = {
    "openai": {"model": "gpt-5-mini", "api_key": "sk-secret-openai"},
    "vllm": {
        "provider": "openai",
        "model": "deepseek-r1",
        "base_url": "http://gpu1.internal:8000/v1",
        "api_key": "vllm-secret",
    },
}
SECRETS = ("sk-secret-openai", "vllm-secret", "gpt-5-mini", "deepseek-r1", "gpu1.internal")


class Out(BaseModel):
    answer: str


@pytest.fixture(autouse=True)
def dummy_clients():
    with patch.object(Agent, "_create_client", lambda self, *a, **k: MagicMock()):
        yield


def _agents(compiled) -> dict[str, Agent]:
    """Agent of every node of a compiled graph (unwrapping prebuilt node wrappers)."""
    found = {}
    for name, node in compiled._state_graph.nodes.items():
        agent = getattr(node.func, "agent", node.func)
        if isinstance(agent, Agent):
            found[name] = agent
    return found


def _info_models(compiled) -> dict[str, Any]:
    info = compiled.generate_graph()["info"]
    # No key, URL or model string may appear anywhere in the graph description.
    dumped = json.dumps(compiled.generate_graph())
    for secret in SECRETS:
        assert secret not in dumped
    return info["models"]


EXPECTED = {"default": "openai", "names": ["openai", "vllm"], "config_key": "model"}


class TestGraphInfo:
    def test_no_registry_is_empty(self):
        compiled = ReactAgent(model="gpt-5-mini").compile()
        assert _info_models(compiled) == {}

    def test_plain_agent_node_listed(self):
        graph = StateGraph()
        graph.add_node("MAIN", Agent(model="openai", model_registry=REGISTRY))
        graph.add_node("OTHER", Agent(model="gpt-5-mini"))
        graph.add_node("fn", lambda state: [])
        graph.set_entry_point("MAIN")
        graph.add_edge("MAIN", "OTHER")
        graph.add_edge("OTHER", "fn")
        graph.add_edge("fn", END)
        compiled = graph.compile()
        assert _info_models(compiled) == {"MAIN": EXPECTED}

    def test_custom_config_key(self):
        compiled = ReactAgent(
            model="vllm", model_registry=REGISTRY, model_config_key="llm"
        ).compile()
        assert _info_models(compiled) == {
            "MAIN": {"default": "vllm", "names": ["openai", "vllm"], "config_key": "llm"}
        }

    def test_react(self):
        compiled = ReactAgent(model="openai", model_registry=REGISTRY).compile()
        assert _info_models(compiled) == {"MAIN": EXPECTED}

    def test_plan_act_reflect(self):
        compiled = PlanActReflectAgent(model="openai", model_registry=REGISTRY).compile()
        assert _info_models(compiled) == {"PLAN": EXPECTED, "REFLECT": EXPECTED}

    def test_structured_output(self):
        compiled = StructuredOutputAgent(
            model="openai", output_schema=Out, model_registry=REGISTRY
        ).compile()
        assert _info_models(compiled) == {"GENERATE": EXPECTED}

    def test_structured_output_repair_agent(self):
        compiled = StructuredOutputAgent(
            model="openai",
            output_schema=Out,
            repair_system_prompt=[{"role": "system", "content": "fix"}],
            model_registry=REGISTRY,
        ).compile()
        assert _info_models(compiled) == {"GENERATE": EXPECTED, "REPAIR": EXPECTED}

    def test_supervisor_team(self):
        worker = Agent(model="gpt-5-mini")
        compiled = SupervisorTeamAgent(
            supervisor_model="openai",
            workers={"w": WorkerConfig(description="d", agent=worker)},
            model_registry=REGISTRY,
        ).compile()
        assert _info_models(compiled) == {"SUPERVISOR": EXPECTED, "RESPOND": EXPECTED}


class TestPrebuiltPassthrough:
    def test_react(self):
        agent = ReactAgent(model="openai", model_registry=REGISTRY, model_config_key="m")
        inner = agent._agent
        assert list(inner._model_entries) == ["openai", "vllm"]
        assert (inner.default_model_name, inner.model_config_key) == ("openai", "m")

    def test_plan_act_reflect_both_agents(self):
        compiled = PlanActReflectAgent(model="openai", model_registry=REGISTRY).compile()
        agents = _agents(compiled)
        assert set(agents) == {"PLAN", "REFLECT"}
        for inner in agents.values():
            assert list(inner._model_entries) == ["openai", "vllm"]

    def test_plan_reflect_model_overrides_are_names(self):
        compiled = PlanActReflectAgent(
            model="openai", plan_model="vllm", model_registry=REGISTRY
        ).compile()
        agents = _agents(compiled)
        assert agents["PLAN"].default_model_name == "vllm"
        assert agents["REFLECT"].default_model_name == "openai"

    def test_structured_output_all_agents(self):
        compiled = StructuredOutputAgent(
            model="openai", output_schema=Out, model_registry=REGISTRY
        ).compile()
        assert list(_agents(compiled)["GENERATE"]._model_entries) == ["openai", "vllm"]
        helper = StructuredOutputAgent(model="openai", output_schema=Out, model_registry=REGISTRY)
        assert list(helper._build_format_agent()._model_entries) == ["openai", "vllm"]
        assert list(helper._build_repair_agent()._model_entries) == ["openai", "vllm"]

    def test_supervisor_team_both_agents(self):
        team = SupervisorTeamAgent(
            supervisor_model="openai",
            workers={"w": WorkerConfig(description="d", agent=Agent(model="gpt-5-mini"))},
            model_registry=REGISTRY,
        )
        for inner in (team._build_supervisor_agent(), team._build_respond_agent()):
            assert list(inner._model_entries) == ["openai", "vllm"]

    async def test_react_config_picks_entry(self):
        seen: list[str] = []

        async def fake_call(self, messages, tools=None, stream=False, **kwargs):
            seen.append(self.model)
            return SimpleNamespace(model=self.model)

        agent = ReactAgent(model="openai", model_registry=REGISTRY, retry_config=None)
        with (
            patch.object(Agent, "_call_llm", fake_call),
            patch(f"{EXEC}.convert_messages", return_value=[{"role": "user", "content": "hi"}]),
            patch("tenxgraph.runtime.publisher.publish.publish_event"),
            patch(f"{EXEC}.ModelResponseConverter"),
        ):
            await agent._agent.execute(AgentState(), {"_node_name": "MAIN", "model": "vllm"})
            await agent._agent.execute(AgentState(), {"_node_name": "MAIN"})
        assert seen == ["deepseek-r1", "gpt-5-mini"]
