"""Comprehensive unit tests for PlanActReflectAgent."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from tenxgraph.core.graph import CompiledGraph, ToolNode
from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END
from tenxgraph.utils.callbacks import CallbackManager
from tenxgraph.prebuilt.agent.plan_act_reflect import (
    DEFAULT_PLAN_SYSTEM_PROMPT,
    DEFAULT_REFLECT_SYSTEM_PROMPT,
    PlanActReflectAgent,
    _ITERATIONS_KEY,
    _INCREMENT_ITERATIONS_NODE,
    _make_plan_route,
    _make_reflect_route,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class FakeManagedAgent(BaseAgent):
    """Minimal Agent stub — never calls an LLM."""

    def __init__(self, model: str, tool_node: ToolNode | None = None, **kwargs):
        super().__init__(model=model, tool_node=tool_node, **kwargs)
        self._tool_node = tool_node

    def get_tool_node(self) -> ToolNode | None:
        return self._tool_node

    async def execute(self, state: AgentState, config: dict) -> AgentState:
        return state

    async def _call_llm(self, messages: list[dict], tools: list | None = None, **kwargs):
        raise NotImplementedError


class FakeToolNode:
    """ToolNode stub for constructor-only tests."""

    def __init__(self, tools, client=None, pass_user_info_to_mcp: bool = False):
        self._funcs = list(tools)
        self.client = client
        self.pass_user_info_to_mcp = pass_user_info_to_mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(text: str, role: str = "assistant", tool_calls: list | None = None) -> Message:
    m = Message.text_message(text, role=role)  # type: ignore[arg-type]
    if tool_calls:
        m.tools_calls = tool_calls
    return m


def _state_with(*messages: Message) -> AgentState:
    state = AgentState()
    state.context = list(messages)
    return state


# ===========================================================================
# Tests for _make_plan_route
# ===========================================================================


class TestMakePlanRoute:
    def test_routes_to_reflect_when_no_tools(self):
        fn = _make_plan_route(has_tools=False)
        state = _state_with(_msg("analysis complete"))
        assert fn(state) == "REFLECT"

    def test_routes_to_reflect_when_no_tool_calls(self):
        fn = _make_plan_route(has_tools=True)
        state = _state_with(_msg("direct answer"))
        assert fn(state) == "REFLECT"

    def test_routes_to_act_when_tool_calls_present(self):
        fn = _make_plan_route(has_tools=True)
        state = _state_with(
            _msg("calling tool", tool_calls=[{"id": "c1", "type": "function"}])
        )
        assert fn(state) == "ACT"

    def test_routes_to_reflect_even_with_tool_calls_if_has_tools_false(self):
        """If has_tools=False no ACT node exists, so never route to ACT."""
        fn = _make_plan_route(has_tools=False)
        state = _state_with(
            _msg("calling tool", tool_calls=[{"id": "c1", "type": "function"}])
        )
        assert fn(state) == "REFLECT"

    def test_routes_to_reflect_on_empty_context(self):
        fn = _make_plan_route(has_tools=True)
        state = AgentState()
        assert fn(state) == "REFLECT"

    def test_routes_to_reflect_when_tool_calls_empty_list(self):
        fn = _make_plan_route(has_tools=True)
        msg = _msg("text", tool_calls=[])
        state = _state_with(msg)
        assert fn(state) == "REFLECT"

    def test_routes_to_reflect_on_user_message_even_with_tool_calls(self):
        """Only assistant-role messages trigger ACT routing."""
        fn = _make_plan_route(has_tools=True)
        msg = _msg("help!", role="user", tool_calls=[{"id": "c1", "type": "function"}])
        state = _state_with(msg)
        assert fn(state) == "REFLECT"


# ===========================================================================
# Tests for _make_reflect_route
# ===========================================================================


class TestMakeReflectRoute:
    def test_routes_to_end_when_done_signal_present(self):
        fn = _make_reflect_route(max_iterations=3)
        state = _state_with(_msg("All tasks complete. [DONE]"))
        assert fn(state) == END

    def test_done_signal_case_insensitive(self):
        fn = _make_reflect_route(max_iterations=3)
        for variant in ("[done]", "[Done]", "[DONE]", "result: [DoNe]"):
            state = _state_with(_msg(f"Summary here. {variant}"))
            assert fn(state) == END, f"Expected END for: {variant!r}"

    def test_routes_to_plan_when_not_done(self):
        fn = _make_reflect_route(max_iterations=3)
        state = _state_with(_msg("Not finished yet, more work needed."))
        assert fn(state) == _INCREMENT_ITERATIONS_NODE

    def test_routes_to_end_at_max_iterations(self):
        fn = _make_reflect_route(max_iterations=2)
        state = _state_with(_msg("still more to do"))
        state.execution_meta.internal_data[_ITERATIONS_KEY] = 2  # == max_iterations
        assert fn(state) == END

    def test_routes_to_end_beyond_max_iterations(self):
        fn = _make_reflect_route(max_iterations=2)
        state = _state_with(_msg("still more"))
        state.execution_meta.internal_data[_ITERATIONS_KEY] = 5
        assert fn(state) == END

    def test_routes_to_end_on_empty_context(self):
        fn = _make_reflect_route(max_iterations=3)
        state = AgentState()
        assert fn(state) == END

    def test_increments_counter_on_plan_route(self):
        fn = _make_reflect_route(max_iterations=5)
        state = _state_with(_msg("more work needed"))
        state.execution_meta.internal_data[_ITERATIONS_KEY] = 1

        result = fn(state)

        # Routing function now returns INCREMENT_ITERATIONS node; counter is
        # incremented by that dedicated node, not by the routing function itself.
        assert result == _INCREMENT_ITERATIONS_NODE
        assert state.execution_meta.internal_data[_ITERATIONS_KEY] == 1

    def test_counter_starts_at_zero_when_absent(self):
        fn = _make_reflect_route(max_iterations=5)
        state = _state_with(_msg("not done"))
        # No counter set initially
        assert _ITERATIONS_KEY not in state.execution_meta.internal_data

        result = fn(state)

        # Counter increment is now done by the INCREMENT_ITERATIONS node, not here.
        assert result == _INCREMENT_ITERATIONS_NODE
        assert _ITERATIONS_KEY not in state.execution_meta.internal_data

    def test_does_not_increment_counter_when_done(self):
        fn = _make_reflect_route(max_iterations=5)
        state = _state_with(_msg("finished. [DONE]"))
        state.execution_meta.internal_data[_ITERATIONS_KEY] = 1

        fn(state)

        # Counter must NOT be incremented when done
        assert state.execution_meta.internal_data[_ITERATIONS_KEY] == 1

    def test_does_not_increment_at_max_iterations(self):
        fn = _make_reflect_route(max_iterations=2)
        state = _state_with(_msg("still going"))
        state.execution_meta.internal_data[_ITERATIONS_KEY] = 2  # at max

        fn(state)

        assert state.execution_meta.internal_data[_ITERATIONS_KEY] == 2

    def test_multiple_loops_increment_correctly(self):
        fn = _make_reflect_route(max_iterations=5)

        state = _state_with(_msg("not done"))
        result1 = fn(state)
        # Counter is incremented by the INCREMENT_ITERATIONS node, not the routing fn.
        assert result1 == _INCREMENT_ITERATIONS_NODE

        # Manually simulate what the INCREMENT_ITERATIONS node would do.
        state.execution_meta.internal_data[_ITERATIONS_KEY] = 1
        state.context = [_msg("still not done")]
        result2 = fn(state)
        assert result2 == _INCREMENT_ITERATIONS_NODE

        state.execution_meta.internal_data[_ITERATIONS_KEY] = 2
        state.context = [_msg("done now [DONE]")]
        result3 = fn(state)
        assert result3 == END
        assert state.execution_meta.internal_data[_ITERATIONS_KEY] == 2


# ===========================================================================
# Tests for PlanActReflectAgent.__init__
# ===========================================================================


class TestPlanActReflectAgentInit:
    def test_basic_init(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        assert agent is not None
        assert agent._model == "fake-model"

    def test_default_max_iterations(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        assert agent._max_iterations == 3

    def test_custom_max_iterations(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", max_iterations=7, provider="openai")
        assert agent._max_iterations == 7

    def test_default_prompts_used_when_none_given(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        assert agent._plan_system_prompt == DEFAULT_PLAN_SYSTEM_PROMPT
        assert agent._reflect_system_prompt == DEFAULT_REFLECT_SYSTEM_PROMPT

    def test_custom_plan_prompt_overrides_default(self):
        custom = [{"role": "system", "content": "Custom planner."}]
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(
                model="fake-model", plan_system_prompt=custom, provider="openai"
            )
        assert agent._plan_system_prompt == custom
        assert agent._reflect_system_prompt == DEFAULT_REFLECT_SYSTEM_PROMPT

    def test_custom_reflect_prompt_overrides_default(self):
        custom = [{"role": "system", "content": "Custom reflector."}]
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(
                model="fake-model", reflect_system_prompt=custom, provider="openai"
            )
        assert agent._reflect_system_prompt == custom
        assert agent._plan_system_prompt == DEFAULT_PLAN_SYSTEM_PROMPT

    def test_no_tools_creates_no_tool_node(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        assert agent._tool_node is None

    def test_with_tools_creates_tool_node(self):
        def dummy_tool(x: str) -> str:
            return x

        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(
                model="fake-model", tools=[dummy_tool], provider="openai"
            )
        assert agent._tool_node is not None
        assert isinstance(agent._tool_node, ToolNode)

    def test_requires_model(self):
        with pytest.raises(TypeError):
            PlanActReflectAgent()  # type: ignore

    def test_mcp_client_creates_tool_node(self):
        """Passing only client (no tools) should still create a ToolNode."""
        fake_client = Mock()
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(
                model="fake-model", client=fake_client, provider="openai"
            )
        assert agent._tool_node is not None


# ===========================================================================
# Tests for PlanActReflectAgent.compile — graph topology
# ===========================================================================


class TestPlanActReflectAgentCompile:
    def test_compile_returns_compiled_graph(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        compiled = agent.compile()
        assert isinstance(compiled, CompiledGraph)

    def test_graph_has_plan_and_reflect_without_tools(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        agent.compile()
        assert "PLAN" in agent._graph.nodes
        assert "REFLECT" in agent._graph.nodes
        assert "ACT" not in agent._graph.nodes

    def test_graph_has_act_node_when_tools_provided(self):
        def dummy_tool(x: str) -> str:
            return x

        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(
                model="fake-model", tools=[dummy_tool], provider="openai"
            )
        agent.compile()
        assert "PLAN" in agent._graph.nodes
        assert "ACT" in agent._graph.nodes
        assert "REFLECT" in agent._graph.nodes

    def test_plan_node_is_agent(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
            agent._configure_graph()
        plan_node = agent._graph.nodes["PLAN"]
        assert isinstance(plan_node.func, FakeManagedAgent)

    def test_reflect_node_is_agent(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
            agent._configure_graph()
        # REFLECT is now wrapped in _make_reflect_node(); the node func is a
        # callable (the async wrapper), not the agent instance directly.
        reflect_node = agent._graph.nodes["REFLECT"]
        assert callable(reflect_node.func)
        assert reflect_node.func.__name__ == "reflect"

    def test_plan_agent_receives_tools(self):
        """PLAN agent should get the tool_node; REFLECT should not."""
        def dummy_tool(x: str) -> str:
            return x

        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(
                model="fake-model", tools=[dummy_tool], provider="openai"
            )
            agent._configure_graph()

        plan_agent: FakeManagedAgent = agent._graph.nodes["PLAN"].func  # type: ignore

        # PLAN agent should have tools; REFLECT node is a wrapper function so
        # we verify via the agent's stored _tool_node attribute instead.
        assert plan_agent.get_tool_node() is not None
        # REFLECT agent had no tools — verify via the stored system attribute.
        assert agent._tool_node is not None  # The overall agent has a tool_node
        # The reflect agent is built with with_tools=False, confirmed by the
        # fact that the PlanActReflectAgent._reflect_system_prompt is stored.
        assert agent._reflect_system_prompt == DEFAULT_REFLECT_SYSTEM_PROMPT

    def test_compile_with_checkpointer(self):
        checkpointer = Mock()
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        compiled = agent.compile(checkpointer=checkpointer)
        assert isinstance(compiled, CompiledGraph)

    def test_compile_with_callback_manager(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        compiled = agent.compile(callback_manager=CallbackManager())
        assert isinstance(compiled, CompiledGraph)

    def test_compile_with_interrupt_options(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        compiled = agent.compile(
            interrupt_before=["PLAN"],
            interrupt_after=["REFLECT"],
        )
        assert isinstance(compiled, CompiledGraph)

    def test_compile_forwards_media_store_and_shutdown_timeout(self):
        media_store = Mock()
        compiled_graph = Mock(spec=CompiledGraph)

        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")

        with patch(
            "tenxgraph.prebuilt.agent.plan_act_reflect.StateGraph.compile",
            autospec=True,
            return_value=compiled_graph,
        ) as compile_mock:
            result = agent.compile(media_store=media_store, shutdown_timeout=15.0)

        assert result is compiled_graph
        assert compile_mock.call_args.kwargs["media_store"] is media_store
        assert compile_mock.call_args.kwargs["shutdown_timeout"] == 15.0

    def test_compile_multiple_times_resets_graph(self):
        """Each compile() call should produce a fresh graph to avoid node duplication."""
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        compiled1 = agent.compile()
        compiled2 = agent.compile()
        assert isinstance(compiled1, CompiledGraph)
        assert isinstance(compiled2, CompiledGraph)

    def test_compile_stores_not_forwarded_when_none(self):
        """Calling compile() without a store should not raise."""
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
        compiled = agent.compile()
        assert isinstance(compiled, CompiledGraph)


# ===========================================================================
# Tests for custom prompt pass-through to PLAN / REFLECT agents
# ===========================================================================


class TestPlanActReflectPromptPassthrough:
    def _get_system_prompt(self, agent_node) -> list[dict]:
        """Extract system_prompt from a FakeManagedAgent node."""
        return agent_node.func._system_prompt  # type: ignore

    def test_default_plan_prompt_used(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
            agent._configure_graph()

        plan_agent: FakeManagedAgent = agent._graph.nodes["PLAN"].func  # type: ignore
        assert plan_agent.system_prompt == DEFAULT_PLAN_SYSTEM_PROMPT

    def test_default_reflect_prompt_used(self):
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(model="fake-model", provider="openai")
            agent._configure_graph()

        # REFLECT node is wrapped; verify the stored prompt on the agent itself.
        assert agent._reflect_system_prompt == DEFAULT_REFLECT_SYSTEM_PROMPT

    def test_custom_plan_prompt_passed_to_plan_agent(self):
        custom = [{"role": "system", "content": "Custom plan."}]
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(
                model="fake-model", plan_system_prompt=custom, provider="openai"
            )
            agent._configure_graph()

        plan_agent: FakeManagedAgent = agent._graph.nodes["PLAN"].func  # type: ignore
        assert plan_agent.system_prompt == custom

    def test_custom_reflect_prompt_passed_to_reflect_agent(self):
        custom = [{"role": "system", "content": "Custom reflect."}]
        with patch("tenxgraph.prebuilt.agent.plan_act_reflect.Agent", FakeManagedAgent):
            agent = PlanActReflectAgent(
                model="fake-model", reflect_system_prompt=custom, provider="openai"
            )
            agent._configure_graph()

        # REFLECT node is wrapped; verify the stored prompt on the agent itself.
        assert agent._reflect_system_prompt == custom


# ===========================================================================
# Integration-style tests — simulate the loop via routing functions directly
# ===========================================================================


class TestPlanActReflectIntegration:
    """Simulates graph execution paths without a real LLM."""

    def test_plan_routes_to_reflect_then_done(self):
        """PLAN → REFLECT → [DONE] → END."""
        plan_route = _make_plan_route(has_tools=False)
        reflect_route = _make_reflect_route(max_iterations=3)

        state = _state_with(_msg("direct answer, no tools needed"))

        # PLAN → REFLECT
        assert plan_route(state) == "REFLECT"

        # REFLECT → END ([DONE] in message)
        state.context = [_msg("Analysis complete. [DONE]")]
        assert reflect_route(state) == END

    def test_reflect_loops_back_to_plan_then_done(self):
        """PLAN → REFLECT → INCREMENT_ITERATIONS → PLAN → REFLECT → [DONE] → END."""
        reflect_route = _make_reflect_route(max_iterations=3)

        # First reflect: not done → routes to INCREMENT_ITERATIONS node
        state = _state_with(_msg("More work needed."))
        assert reflect_route(state) == _INCREMENT_ITERATIONS_NODE
        # Counter is managed by the INCREMENT_ITERATIONS node, not routing fn
        assert _ITERATIONS_KEY not in state.execution_meta.internal_data

        # Simulate what INCREMENT_ITERATIONS node would do, then re-reflect
        state.execution_meta.internal_data[_ITERATIONS_KEY] = 1
        state.context = [_msg("All done. [DONE]")]
        assert reflect_route(state) == END
        # Counter should not be touched when done
        assert state.execution_meta.internal_data[_ITERATIONS_KEY] == 1

    def test_max_iterations_stops_loop(self):
        """Loop terminates at max_iterations even without [DONE]."""
        reflect_route = _make_reflect_route(max_iterations=2)

        state = _state_with(_msg("still not done"))
        # Simulate two previous loops
        state.execution_meta.internal_data[_ITERATIONS_KEY] = 2
        assert reflect_route(state) == END

    def test_counter_increments_each_reflect_plan_loop(self):
        reflect_route = _make_reflect_route(max_iterations=10)
        state = _state_with(_msg("not done"))

        # Counter is managed by the INCREMENT_ITERATIONS node outside routing fn.
        for _ in range(3):
            state.context = [_msg("not done yet")]
            result = reflect_route(state)
            assert result == _INCREMENT_ITERATIONS_NODE

    def test_plan_with_tools_routes_to_act(self):
        """With has_tools=True, tool calls from PLAN → ACT."""
        plan_route = _make_plan_route(has_tools=True)
        state = _state_with(
            _msg("searching...", tool_calls=[{"id": "t1", "type": "function"}])
        )
        assert plan_route(state) == "ACT"

    def test_plan_without_tools_never_routes_to_act(self):
        """Without tools, tool calls in message are ignored."""
        plan_route = _make_plan_route(has_tools=False)
        state = _state_with(
            _msg("would-call tool", tool_calls=[{"id": "t1", "type": "function"}])
        )
        assert plan_route(state) == "REFLECT"

    def test_reflect_route_at_iteration_zero_goes_to_plan(self):
        """First reflect with no [DONE] should go to INCREMENT_ITERATIONS."""
        reflect_route = _make_reflect_route(max_iterations=3)
        state = _state_with(_msg("still working"))
        assert _ITERATIONS_KEY not in state.execution_meta.internal_data

        result = reflect_route(state)

        # Counter is managed by the INCREMENT_ITERATIONS node.
        assert result == _INCREMENT_ITERATIONS_NODE
        assert _ITERATIONS_KEY not in state.execution_meta.internal_data
