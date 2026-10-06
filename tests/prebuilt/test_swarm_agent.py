"""Comprehensive unit tests for SwarmAgent."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from tenxgraph.core.graph import CompiledGraph, ToolNode
from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END
from tenxgraph.utils.callbacks import CallbackManager
from tenxgraph.prebuilt.agent.swarm import (
    SwarmAgent,
    SwarmMemberConfig,
    _make_member_route,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class FakeManagedAgent(BaseAgent):
    """Minimal Agent stub — never calls an LLM."""

    def __init__(self, model: str = "fake", tool_node: ToolNode | None = None, **kwargs):
        super().__init__(model=model, tool_node=tool_node, **kwargs)

    async def execute(self, state: AgentState, config: dict) -> AgentState:
        return state

    async def _call_llm(self, messages: list[dict], tools: list | None = None, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_agent(**kwargs) -> FakeManagedAgent:
    return FakeManagedAgent(**kwargs)


def _msg(text: str, role: str = "assistant", tool_calls: list | None = None) -> Message:
    m = Message.text_message(text, role=role)  # type: ignore[arg-type]
    if tool_calls:
        m.tools_calls = tool_calls
    return m


def _tc(name: str) -> dict:
    """Minimal tool-call dict matching the Message.tools_calls schema."""
    return {"name": name}


def _state_with(*messages: Message) -> AgentState:
    state = AgentState()
    state.context = list(messages)
    return state


def _two_members() -> dict[str, SwarmMemberConfig]:
    return {
        "TRIAGE": SwarmMemberConfig(
            agent=_fake_agent(),
            description="Triage agent",
        ),
        "WRITER": SwarmMemberConfig(
            agent=_fake_agent(),
            description="Writer agent",
        ),
    }


def _three_members() -> dict[str, SwarmMemberConfig]:
    return {
        "TRIAGE": SwarmMemberConfig(
            agent=_fake_agent(),
            description="Triage agent",
            can_handoff_to=["RESEARCHER", "WRITER"],
        ),
        "RESEARCHER": SwarmMemberConfig(
            agent=_fake_agent(),
            description="Research agent",
            can_handoff_to=["WRITER"],
        ),
        "WRITER": SwarmMemberConfig(
            agent=_fake_agent(),
            description="Writer agent",
        ),
    }


# ===========================================================================
# Tests for _make_member_route
# ===========================================================================


class TestMakeMemberRoute:
    def test_routes_to_target_on_handoff_tool_call(self):
        fn = _make_member_route("TRIAGE", ["RESEARCHER", "WRITER"], None)
        state = _state_with(_msg("", tool_calls=[_tc("transfer_to_researcher")]))
        assert fn(state) == "RESEARCHER"

    def test_routes_to_end_with_no_tool_calls(self):
        fn = _make_member_route("TRIAGE", ["RESEARCHER"], None)
        state = _state_with(_msg("direct answer"))
        assert fn(state) == END

    def test_routes_to_end_on_empty_context(self):
        fn = _make_member_route("WRITER", ["TRIAGE"], None)
        state = AgentState()
        assert fn(state) == END

    def test_routes_to_end_when_tool_not_a_handoff(self):
        fn = _make_member_route("TRIAGE", ["WRITER"], None)
        state = _state_with(_msg("", tool_calls=[_tc("web_search")]))
        assert fn(state) == END

    def test_routes_to_end_when_target_not_in_allowed(self):
        """Handoff tool exists but target is not in allowed_targets for this member."""
        fn = _make_member_route("WRITER", ["TRIAGE"], None)  # only TRIAGE allowed
        state = _state_with(_msg("", tool_calls=[_tc("transfer_to_researcher")]))
        assert fn(state) == END

    def test_routes_to_end_on_non_assistant_message(self):
        fn = _make_member_route("TRIAGE", ["WRITER"], None)
        state = _state_with(_msg("", role="user", tool_calls=[_tc("transfer_to_writer")]))
        assert fn(state) == END

    def test_first_matching_handoff_wins(self):
        fn = _make_member_route("TRIAGE", ["RESEARCHER", "WRITER"], None)
        state = _state_with(
            _msg(
                "",
                tool_calls=[_tc("transfer_to_researcher"), _tc("transfer_to_writer")],
            )
        )
        assert fn(state) == "RESEARCHER"

    def test_empty_allowed_targets_always_ends(self):
        fn = _make_member_route("WRITER", [], None)
        state = _state_with(_msg("", tool_calls=[_tc("transfer_to_triage")]))
        assert fn(state) == END

    def test_case_insensitive_target_matching(self):
        """Target in allowed list is UPPER; handoff tool uses lower — should match."""
        fn = _make_member_route("TRIAGE", ["WRITER"], None)
        state = _state_with(_msg("", tool_calls=[_tc("transfer_to_writer")]))
        assert fn(state) == "WRITER"


# ===========================================================================
# Tests for SwarmMemberConfig
# ===========================================================================


class TestSwarmMemberConfig:
    def test_minimal_config(self):
        agent = _fake_agent()
        cfg = SwarmMemberConfig(agent=agent)
        assert cfg.agent is agent
        assert cfg.can_handoff_to is None
        assert cfg.description == ""

    def test_full_config(self):
        agent = _fake_agent()
        cfg = SwarmMemberConfig(
            agent=agent,
            can_handoff_to=["B", "C"],
            description="Main agent",
        )
        assert cfg.agent is agent
        assert cfg.can_handoff_to == ["B", "C"]
        assert cfg.description == "Main agent"

    def test_accepts_any_base_agent_subclass(self):
        """SwarmMemberConfig accepts any BaseAgent subclass."""
        agent = _fake_agent()
        cfg = SwarmMemberConfig(agent=agent)
        assert isinstance(cfg.agent, BaseAgent)


# ===========================================================================
# Tests for SwarmAgent.__init__ validation
# ===========================================================================


class TestSwarmAgentInit:
    def test_basic_init(self):
        agent = SwarmAgent(members=_two_members(), entry="TRIAGE")
        assert agent is not None

    def test_empty_members_raises(self):
        with pytest.raises(ValueError, match="at least one member"):
            SwarmAgent(members={}, entry="TRIAGE")

    def test_invalid_entry_raises(self):
        with pytest.raises(ValueError, match="not in members"):
            SwarmAgent(members=_two_members(), entry="MISSING")

    def test_stores_members_and_entry(self):
        agent = SwarmAgent(members=_two_members(), entry="TRIAGE")
        assert agent._entry == "TRIAGE"
        assert set(agent._members.keys()) == {"TRIAGE", "WRITER"}

    def test_single_member_allowed(self):
        """A single-member swarm is technically valid (routes to END immediately)."""
        agent = SwarmAgent(
            members={"SOLO": SwarmMemberConfig(agent=_fake_agent())},
            entry="SOLO",
        )
        assert agent is not None

    def test_no_agent_pass_through_params(self):
        """SwarmAgent.__init__ no longer accepts per-agent kwargs."""
        import inspect
        sig = inspect.signature(SwarmAgent.__init__)
        param_names = set(sig.parameters)
        # These should NOT be in the signature any more
        for removed in ("skills", "memory", "retry_config", "fallback_models",
                         "multimodal_config", "client", "reasoning_config"):
            assert removed not in param_names, f"{removed!r} should not be in SwarmAgent.__init__"


# ===========================================================================
# Tests for SwarmAgent._resolve_targets
# ===========================================================================


class TestResolveTargets:
    def test_explicit_can_handoff_to(self):
        members = _three_members()
        agent = SwarmAgent(members=members, entry="TRIAGE")
        assert agent._resolve_targets("TRIAGE") == ["RESEARCHER", "WRITER"]

    def test_none_means_all_others(self):
        members = _two_members()
        # TRIAGE has can_handoff_to=None
        agent = SwarmAgent(members=members, entry="TRIAGE")
        targets = agent._resolve_targets("TRIAGE")
        assert targets == ["WRITER"]

    def test_terminal_member_no_explicit_targets(self):
        members = _three_members()
        agent = SwarmAgent(members=members, entry="TRIAGE")
        # WRITER has can_handoff_to=None so gets all others
        targets = agent._resolve_targets("WRITER")
        assert set(targets) == {"TRIAGE", "RESEARCHER"}


# ===========================================================================
# Tests for SwarmAgent._build_handoff_tools
# ===========================================================================


class TestBuildHandoffTools:
    def test_handoff_tools_created_for_each_target(self):
        agent = SwarmAgent(members=_three_members(), entry="TRIAGE")
        tools = agent._build_handoff_tools("TRIAGE")
        # TRIAGE can_handoff_to = ["RESEARCHER", "WRITER"]
        assert len(tools) == 2
        names = {t.__name__ for t in tools}
        assert "transfer_to_researcher" in names
        assert "transfer_to_writer" in names

    def test_no_handoff_tools_for_none_targets(self):
        """WRITER has no explicit can_handoff_to (None = all others = TRIAGE, RESEARCHER)."""
        agent = SwarmAgent(members=_three_members(), entry="TRIAGE")
        tools = agent._build_handoff_tools("WRITER")
        names = {t.__name__ for t in tools}
        assert "transfer_to_triage" in names
        assert "transfer_to_researcher" in names

    def test_description_used_in_tool_docstring(self):
        members = {
            "A": SwarmMemberConfig(agent=_fake_agent(), can_handoff_to=["B"]),
            "B": SwarmMemberConfig(agent=_fake_agent(), description="Specialist B"),
        }
        agent = SwarmAgent(members=members, entry="A")
        tools = agent._build_handoff_tools("A")
        assert len(tools) == 1
        assert "Specialist B" in tools[0].__doc__


# ===========================================================================
# Tests for _inject_handoff_tools
# ===========================================================================


class TestInjectHandoffTools:
    def test_creates_tool_node_when_agent_has_none(self):
        """Agent with no tool_node gets one created from handoff tools."""
        members = {
            "A": SwarmMemberConfig(agent=_fake_agent(), can_handoff_to=["B"]),
            "B": SwarmMemberConfig(agent=_fake_agent()),
        }
        swarm = SwarmAgent(members=members, entry="A")
        swarm._inject_handoff_tools("A")
        assert members["A"].agent.tool_node is not None
        assert isinstance(members["A"].agent.tool_node, ToolNode)

    def test_adds_to_existing_tool_node(self):
        """Agent with an existing ToolNode gets handoff tools added to it."""
        def domain_tool(x: str) -> str:
            return x

        existing_tn = ToolNode([domain_tool])
        member_agent = _fake_agent(tool_node=existing_tn)

        members = {
            "A": SwarmMemberConfig(agent=member_agent, can_handoff_to=["B"]),
            "B": SwarmMemberConfig(agent=_fake_agent()),
        }
        swarm = SwarmAgent(members=members, entry="A")
        swarm._inject_handoff_tools("A")

        # Original tool node is reused and now contains both tools
        assert members["A"].agent.tool_node is existing_tn
        assert "domain_tool" in existing_tn._funcs
        assert "transfer_to_b" in existing_tn._funcs

    def test_no_injection_when_no_targets(self):
        """Single-member swarm — no handoff tools injected."""
        agent = _fake_agent()
        members = {"SOLO": SwarmMemberConfig(agent=agent)}
        swarm = SwarmAgent(members=members, entry="SOLO")
        swarm._inject_handoff_tools("SOLO")
        # tool_node stays None — no handoffs needed
        assert agent.tool_node is None


# ===========================================================================
# Tests for SwarmAgent.compile — graph topology
# ===========================================================================


class TestSwarmAgentCompile:
    def test_compile_returns_compiled_graph(self):
        agent = SwarmAgent(members=_two_members(), entry="TRIAGE")
        compiled = agent.compile()
        assert isinstance(compiled, CompiledGraph)

    def test_graph_has_all_member_nodes(self):
        agent = SwarmAgent(members=_three_members(), entry="TRIAGE")
        agent.compile()
        assert "TRIAGE" in agent._graph.nodes
        assert "RESEARCHER" in agent._graph.nodes
        assert "WRITER" in agent._graph.nodes

    def test_all_member_nodes_are_agents(self):
        members = _two_members()
        swarm = SwarmAgent(members=members, entry="TRIAGE")
        swarm._configure_graph()
        for name, cfg in members.items():
            node = swarm._graph.nodes[name]
            assert node.func is cfg.agent, f"{name} node func should be the member agent"

    def test_member_with_domain_tools_keeps_tool_node(self):
        def dummy(x: str) -> str:
            return x

        existing_tn = ToolNode([dummy])
        members = {
            "A": SwarmMemberConfig(
                agent=_fake_agent(tool_node=existing_tn),
                can_handoff_to=["B"],
            ),
            "B": SwarmMemberConfig(agent=_fake_agent()),
        }
        swarm = SwarmAgent(members=members, entry="A")
        swarm._configure_graph()

        node_a: FakeManagedAgent = swarm._graph.nodes["A"].func  # type: ignore
        assert node_a.tool_node is existing_tn

    def test_compile_with_checkpointer(self):
        checkpointer = Mock()
        agent = SwarmAgent(members=_two_members(), entry="TRIAGE")
        compiled = agent.compile(checkpointer=checkpointer)
        assert isinstance(compiled, CompiledGraph)

    def test_compile_with_callback_manager(self):
        agent = SwarmAgent(members=_two_members(), entry="TRIAGE")
        compiled = agent.compile(callback_manager=CallbackManager())
        assert isinstance(compiled, CompiledGraph)

    def test_compile_with_interrupt_options(self):
        agent = SwarmAgent(members=_two_members(), entry="TRIAGE")
        compiled = agent.compile(
            interrupt_before=["TRIAGE"],
            interrupt_after=["WRITER"],
        )
        assert isinstance(compiled, CompiledGraph)

    def test_compile_forwards_media_store_and_shutdown_timeout(self):
        media_store = Mock()
        compiled_graph = Mock(spec=CompiledGraph)

        agent = SwarmAgent(members=_two_members(), entry="TRIAGE")

        with patch(
            "tenxgraph.prebuilt.agent.swarm.StateGraph.compile",
            autospec=True,
            return_value=compiled_graph,
        ) as compile_mock:
            result = agent.compile(media_store=media_store, shutdown_timeout=45.0)

        assert result is compiled_graph
        assert compile_mock.call_args.kwargs["media_store"] is media_store
        assert compile_mock.call_args.kwargs["shutdown_timeout"] == 45.0

    def test_compile_multiple_times_resets_graph(self):
        agent = SwarmAgent(members=_two_members(), entry="TRIAGE")
        compiled1 = agent.compile()
        compiled2 = agent.compile()
        assert isinstance(compiled1, CompiledGraph)
        assert isinstance(compiled2, CompiledGraph)

    def test_single_member_compiles(self):
        agent = SwarmAgent(
            members={"SOLO": SwarmMemberConfig(agent=_fake_agent())},
            entry="SOLO",
        )
        compiled = agent.compile()
        assert isinstance(compiled, CompiledGraph)


# ===========================================================================
# Tests for member agent system prompt passthrough
# ===========================================================================


class TestSystemPromptPassthrough:
    def test_member_system_prompt_set_on_agent(self):
        """The system_prompt is set on the Agent at creation time by the user."""
        custom_prompt = [{"role": "system", "content": "You are a triage agent."}]
        triage_agent = _fake_agent(system_prompt=custom_prompt)
        members = {
            "TRIAGE": SwarmMemberConfig(
                agent=triage_agent,
                can_handoff_to=["WRITER"],
            ),
            "WRITER": SwarmMemberConfig(agent=_fake_agent()),
        }
        swarm = SwarmAgent(members=members, entry="TRIAGE")
        swarm._configure_graph()

        triage_node: FakeManagedAgent = swarm._graph.nodes["TRIAGE"].func  # type: ignore
        assert triage_node.system_prompt == custom_prompt

    def test_member_with_no_prompt(self):
        members = {
            "A": SwarmMemberConfig(agent=_fake_agent(), can_handoff_to=["B"]),
            "B": SwarmMemberConfig(agent=_fake_agent()),
        }
        swarm = SwarmAgent(members=members, entry="A")
        swarm._configure_graph()

        node_a: FakeManagedAgent = swarm._graph.nodes["A"].func  # type: ignore
        # BaseAgent normalises None → [] so check for falsy
        assert not node_a.system_prompt

    def test_each_member_independently_configured(self):
        """Each member has its own independently set system_prompt."""
        prompt_a = [{"role": "system", "content": "Agent A"}]
        prompt_b = [{"role": "system", "content": "Agent B"}]
        members = {
            "A": SwarmMemberConfig(agent=_fake_agent(system_prompt=prompt_a), can_handoff_to=["B"]),
            "B": SwarmMemberConfig(agent=_fake_agent(system_prompt=prompt_b)),
        }
        swarm = SwarmAgent(members=members, entry="A")
        swarm._configure_graph()

        node_a: FakeManagedAgent = swarm._graph.nodes["A"].func  # type: ignore
        node_b: FakeManagedAgent = swarm._graph.nodes["B"].func  # type: ignore
        assert node_a.system_prompt == prompt_a
        assert node_b.system_prompt == prompt_b


# ===========================================================================
# Integration-style tests — routing function behaviour
# ===========================================================================


class TestSwarmIntegration:
    def test_triage_routes_to_researcher_on_handoff(self):
        fn = _make_member_route("TRIAGE", ["RESEARCHER", "WRITER"], None)
        state = _state_with(_msg("", tool_calls=[_tc("transfer_to_researcher")]))
        assert fn(state) == "RESEARCHER"

    def test_researcher_routes_to_writer_on_handoff(self):
        fn = _make_member_route("RESEARCHER", ["WRITER"], None)
        state = _state_with(_msg("", tool_calls=[_tc("transfer_to_writer")]))
        assert fn(state) == "WRITER"

    def test_writer_with_no_targets_routes_to_end(self):
        fn = _make_member_route("WRITER", [], None)
        state = _state_with(_msg("Final report complete."))
        assert fn(state) == END

    def test_researcher_cannot_route_back_to_triage(self):
        """Researcher's allowed_targets is [WRITER], so TRIAGE is not reachable."""
        fn = _make_member_route("RESEARCHER", ["WRITER"], None)
        state = _state_with(_msg("", tool_calls=[_tc("transfer_to_triage")]))
        assert fn(state) == END

    def test_route_to_end_when_no_handoff_despite_tools(self):
        """Regular tool call (non-handoff) routes to END."""
        fn = _make_member_route("TRIAGE", ["WRITER"], None)
        state = _state_with(_msg("", tool_calls=[_tc("web_search")]))
        assert fn(state) == END

    def test_handoff_tools_are_callable(self):
        swarm = SwarmAgent(members=_three_members(), entry="TRIAGE")
        tools = swarm._build_handoff_tools("TRIAGE")
        for t in tools:
            assert callable(t)

    def test_handoff_tools_injected_on_compile(self):
        """After compile(), each member agent has handoff tools in its tool_node."""
        members = _two_members()
        swarm = SwarmAgent(members=members, entry="TRIAGE")
        swarm.compile()

        # TRIAGE has can_handoff_to=None → all others = WRITER
        triage_tn = members["TRIAGE"].agent.tool_node
        assert triage_tn is not None
        assert "transfer_to_writer" in triage_tn._funcs

        # WRITER has can_handoff_to=None → all others = TRIAGE
        writer_tn = members["WRITER"].agent.tool_node
        assert writer_tn is not None
        assert "transfer_to_triage" in writer_tn._funcs
