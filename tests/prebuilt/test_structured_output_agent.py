"""Comprehensive unit tests for StructuredOutputAgent."""

from __future__ import annotations

import json
from typing import TypedDict
from unittest.mock import Mock, patch

import pydantic
import pytest

from tenxgraph.core.graph import CompiledGraph, ToolNode
from tenxgraph.core.graph.base_agent import BaseAgent
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils import END
from tenxgraph.utils.callbacks import CallbackManager
from tenxgraph.prebuilt.agent.structured_output import (
    StructuredOutputAgent,
    _build_type_adapter,
    _make_repair_fn,
    _make_route_fn,
    _schema_json,
    _validate_message,
)


# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------


class MovieReview(pydantic.BaseModel):
    title: str
    rating: float
    summary: str


class PersonInfo(TypedDict):
    name: str
    age: int


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
        self.tools = list(tools)
        self.client = client
        self.pass_user_info_to_mcp = pass_user_info_to_mcp


# ---------------------------------------------------------------------------
# Helper: build a Message with given text and optional tool calls
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
# Tests for _build_type_adapter / _schema_json
# ===========================================================================


class TestBuildTypeAdapter:
    def test_pydantic_model(self):
        adapter = _build_type_adapter(MovieReview)
        assert adapter is not None
        # Should validate correctly
        adapter.validate_python({"title": "Inception", "rating": 9.0, "summary": "Great"})

    def test_typeddict(self):
        adapter = _build_type_adapter(PersonInfo)
        adapter.validate_python({"name": "Alice", "age": 30})

    def test_pydantic_model_rejects_invalid(self):
        adapter = _build_type_adapter(MovieReview)
        with pytest.raises(pydantic.ValidationError):
            adapter.validate_python({"title": "X"})  # missing rating, summary

    def test_schema_json_returns_string(self):
        result = _schema_json(MovieReview)
        assert isinstance(result, str)
        assert "title" in result


# ===========================================================================
# Tests for _validate_message
# ===========================================================================


class TestValidateMessage:
    def _adapter(self, schema=MovieReview):
        return _build_type_adapter(schema)

    def test_valid_json_text(self):
        payload = {"title": "Inception", "rating": 9.0, "summary": "A dream heist."}
        msg = _msg(json.dumps(payload))
        ok, err = _validate_message(msg, self._adapter())
        assert ok is True
        assert err == ""

    def test_invalid_json_text(self):
        msg = _msg("not json at all")
        ok, err = _validate_message(msg, self._adapter())
        assert ok is False
        assert "JSON" in err or err  # some error message present

    def test_missing_required_fields(self):
        payload = {"title": "Inception"}  # missing rating, summary
        msg = _msg(json.dumps(payload))
        ok, err = _validate_message(msg, self._adapter())
        assert ok is False

    def test_empty_content(self):
        msg = _msg("")
        ok, err = _validate_message(msg, self._adapter())
        assert ok is False
        assert "empty" in err.lower()

    def test_markdown_code_fence_stripped(self):
        payload = {"title": "Inception", "rating": 9.0, "summary": "A dream heist."}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        msg = _msg(fenced)
        ok, err = _validate_message(msg, self._adapter())
        assert ok is True, f"Expected valid but got: {err}"

    def test_parsed_content_preferred(self):
        """If parsed_content is already a valid Pydantic model, text is ignored."""
        msg = _msg("this text is irrelevant")
        msg.parsed_content = MovieReview(title="X", rating=8.5, summary="Y")
        ok, err = _validate_message(msg, self._adapter())
        assert ok is True

    def test_parsed_content_dict_also_validated(self):
        msg = _msg("irrelevant")
        msg.parsed_content = {"title": "X", "rating": 8.5, "summary": "Y"}
        ok, err = _validate_message(msg, self._adapter())
        assert ok is True

    def test_typeddict_valid(self):
        adapter = self._adapter(PersonInfo)
        msg = _msg(json.dumps({"name": "Alice", "age": 30}))
        ok, err = _validate_message(msg, adapter)
        assert ok is True

    def test_typeddict_invalid(self):
        adapter = self._adapter(PersonInfo)
        msg = _msg(json.dumps({"name": "Alice"}))  # missing age
        ok, err = _validate_message(msg, adapter)
        assert ok is False


# ===========================================================================
# Tests for _make_route_fn
# ===========================================================================


class TestMakeRouteFn:
    def _route(self, schema=MovieReview, max_attempts=2, has_tools=False):
        return _make_route_fn(schema, max_attempts, has_tools=has_tools)

    def _valid_payload(self):
        return {"title": "Inception", "rating": 9.0, "summary": "A dream heist."}

    def test_routes_to_end_on_valid_output(self):
        state = _state_with(_msg(json.dumps(self._valid_payload())))
        fn = self._route()
        assert fn(state) == END

    def test_routes_to_repair_on_invalid_output(self):
        state = _state_with(_msg("bad output"))
        fn = self._route()
        assert fn(state) == "REPAIR"

    def test_routes_to_end_when_max_attempts_reached(self):
        state = _state_with(_msg("still bad"))
        state.execution_meta.internal_data["soa_attempts"] = 2  # == max_attempts
        fn = self._route(max_attempts=2)
        assert fn(state) == END

    def test_routes_to_tool_when_tool_calls_present(self):
        state = _state_with(
            _msg("calling tool", tool_calls=[{"id": "c1", "type": "function"}])
        )
        fn = self._route(has_tools=True)
        assert fn(state) == "TOOL"

    def test_no_tool_route_when_has_tools_false(self):
        """Tool calls should be ignored if has_tools=False (no TOOL node in graph)."""
        state = _state_with(
            _msg("calling tool", tool_calls=[{"id": "c1", "type": "function"}])
        )
        fn = self._route(has_tools=False)
        # Should fall through to validation, not route to TOOL
        result = fn(state)
        assert result in ("REPAIR", END)
        assert result != "TOOL"

    def test_empty_context_returns_end(self):
        state = AgentState()
        fn = self._route()
        assert fn(state) == END

    def test_routes_repair_on_first_attempt(self):
        state = _state_with(_msg("not valid json"))
        state.execution_meta.internal_data["soa_attempts"] = 0
        fn = self._route(max_attempts=3)
        assert fn(state) == "REPAIR"

    def test_routes_repair_until_max_then_end(self):
        fn = self._route(max_attempts=2)

        for attempt in range(2):
            state = _state_with(_msg("bad"))
            state.execution_meta.internal_data["soa_attempts"] = attempt
            assert fn(state) == "REPAIR", f"Expected REPAIR at attempt {attempt}"

        # At max_attempts, must return END
        state = _state_with(_msg("bad"))
        state.execution_meta.internal_data["soa_attempts"] = 2
        assert fn(state) == END


# ===========================================================================
# Tests for _make_repair_fn
# ===========================================================================


class TestMakeRepairFn:
    @pytest.mark.asyncio
    async def test_increments_attempt_counter(self):
        fn = _make_repair_fn(MovieReview)
        state = _state_with(_msg("bad"))
        state.execution_meta.internal_data["soa_attempts"] = 0

        result = await fn(state, {})

        assert state.execution_meta.internal_data["soa_attempts"] == 1
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_injected_message_is_user_role(self):
        fn = _make_repair_fn(MovieReview)
        state = _state_with(_msg("bad"))
        messages = await fn(state, {})
        assert messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_injected_message_contains_schema_hint(self):
        fn = _make_repair_fn(MovieReview)
        state = _state_with(_msg("bad"))
        messages = await fn(state, {})
        text = messages[0].text()
        assert "title" in text or "MovieReview" in text or "schema" in text.lower()

    @pytest.mark.asyncio
    async def test_handles_empty_context_gracefully(self):
        fn = _make_repair_fn(MovieReview)
        state = AgentState()
        messages = await fn(state, {})
        assert isinstance(messages, list)
        assert len(messages) == 1


# ===========================================================================
# Tests for StructuredOutputAgent construction
# ===========================================================================


class TestStructuredOutputAgentInit:
    def test_basic_init(self):
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )
        assert agent is not None
        assert agent._output_schema is MovieReview
        assert agent._max_attempts == 2  # default

    def test_custom_max_attempts(self):
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                max_attempts=5,
                provider="openai",
            )
        assert agent._max_attempts == 5

    def test_no_tools_creates_no_tool_node(self):
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )
        assert agent._tool_node is None

    def test_with_tools_creates_tool_node(self):
        def dummy_tool(x: str) -> str:
            return x

        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                tools=[dummy_tool],
                provider="openai",
            )
        assert agent._tool_node is not None
        assert isinstance(agent._tool_node, ToolNode)

    def test_requires_model(self):
        with pytest.raises(TypeError):
            StructuredOutputAgent(output_schema=MovieReview)  # type: ignore

    def test_requires_output_schema(self):
        with pytest.raises(TypeError):
            StructuredOutputAgent(model="fake-model")  # type: ignore


# ===========================================================================
# Tests for StructuredOutputAgent.compile — graph topology
# ===========================================================================


class TestStructuredOutputAgentCompile:
    def test_compile_returns_compiled_graph(self):
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )
        compiled = agent.compile()
        assert isinstance(compiled, CompiledGraph)

    def test_graph_has_generate_and_repair_nodes_without_tools(self):
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )
        agent.compile()
        assert "GENERATE" in agent._graph.nodes
        assert "REPAIR" in agent._graph.nodes
        assert "TOOL" not in agent._graph.nodes

    def test_graph_has_tool_node_when_tools_provided(self):
        def dummy_tool(x: str) -> str:
            return x

        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                tools=[dummy_tool],
                provider="openai",
            )
        agent.compile()
        assert "GENERATE" in agent._graph.nodes
        assert "REPAIR" in agent._graph.nodes
        assert "TOOL" in agent._graph.nodes

    def test_compile_with_checkpointer(self):
        checkpointer = Mock()
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )
        compiled = agent.compile(checkpointer=checkpointer)
        assert isinstance(compiled, CompiledGraph)

    def test_compile_with_callback_manager(self):
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )
        compiled = agent.compile(callback_manager=CallbackManager())
        assert isinstance(compiled, CompiledGraph)

    def test_compile_with_interrupt_options(self):
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )
        compiled = agent.compile(
            interrupt_before=["GENERATE"],
            interrupt_after=["REPAIR"],
        )
        assert isinstance(compiled, CompiledGraph)

    def test_compile_forwards_media_store_and_shutdown_timeout(self):
        media_store = Mock()
        compiled_graph = Mock(spec=CompiledGraph)

        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )

        with patch(
            "tenxgraph.prebuilt.agent.structured_output.StateGraph.compile",
            autospec=True,
            return_value=compiled_graph,
        ) as compile_mock:
            result = agent.compile(media_store=media_store, shutdown_timeout=15.0)

        assert result is compiled_graph
        assert compile_mock.call_args.kwargs["media_store"] is media_store
        assert compile_mock.call_args.kwargs["shutdown_timeout"] == 15.0

    def test_compile_multiple_times_resets_graph(self):
        """Each compile() call should produce a fresh graph to avoid node duplication."""
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )
        compiled1 = agent.compile()
        compiled2 = agent.compile()
        assert isinstance(compiled1, CompiledGraph)
        assert isinstance(compiled2, CompiledGraph)


# ===========================================================================
# Tests for repair_system_prompt — selects Agent vs function repair node
# ===========================================================================


class TestRepairNodeSelection:
    def test_default_repair_is_function_node(self):
        """Without repair_system_prompt, REPAIR should be a coroutine function node."""
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                provider="openai",
            )
        agent._configure_graph()
        repair_node = agent._graph.nodes["REPAIR"]
        # The repair node should be callable (function node), not a FakeManagedAgent
        assert callable(repair_node.func)
        assert not isinstance(repair_node.func, FakeManagedAgent)

    def test_repair_system_prompt_uses_agent_node(self):
        """With repair_system_prompt, REPAIR should be an Agent instance."""
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=MovieReview,
                repair_system_prompt=[{"role": "system", "content": "Fix the JSON."}],
                provider="openai",
            )
            # Call _configure_graph inside the patch so _build_repair_agent uses the fake
            agent._configure_graph()

        repair_node = agent._graph.nodes["REPAIR"]
        assert isinstance(repair_node.func, FakeManagedAgent)


# ===========================================================================
# Tests for TypedDict schema support
# ===========================================================================


class TestTypedDictSchema:
    def test_compile_with_typeddict_schema(self):
        with patch("tenxgraph.prebuilt.agent.structured_output.Agent", FakeManagedAgent):
            agent = StructuredOutputAgent(
                model="fake-model",
                output_schema=PersonInfo,
                provider="openai",
            )
        compiled = agent.compile()
        assert isinstance(compiled, CompiledGraph)

    def test_route_fn_validates_typeddict(self):
        fn = _make_route_fn(PersonInfo, max_attempts=2, has_tools=False)
        state = _state_with(_msg(json.dumps({"name": "Alice", "age": 30})))
        assert fn(state) == END

    def test_route_fn_rejects_invalid_typeddict(self):
        fn = _make_route_fn(PersonInfo, max_attempts=2, has_tools=False)
        state = _state_with(_msg(json.dumps({"name": "Alice"})))  # missing age
        assert fn(state) == "REPAIR"


# ===========================================================================
# Integration-style test — simulated multi-turn run via graph nodes
# ===========================================================================


class TestStructuredOutputAgentIntegration:
    """Simulates the graph execution path without a real LLM."""

    @pytest.mark.asyncio
    async def test_repair_increments_counter_each_call(self):
        """Calling _repair multiple times should increment soa_attempts correctly."""
        repair_fn = _make_repair_fn(MovieReview)
        state = _state_with(_msg("bad"))

        await repair_fn(state, {})
        assert state.execution_meta.internal_data["soa_attempts"] == 1

        await repair_fn(state, {})
        assert state.execution_meta.internal_data["soa_attempts"] == 2

    @pytest.mark.asyncio
    async def test_route_gives_end_after_repair_with_valid_output(self):
        """Simulates: GENERATE (invalid) → REPAIR → GENERATE (valid) → END."""
        route_fn = _make_route_fn(MovieReview, max_attempts=2, has_tools=False)
        repair_fn = _make_repair_fn(MovieReview)

        # First GENERATE produces invalid output
        state = _state_with(_msg("bad output"))
        assert route_fn(state) == "REPAIR"

        # REPAIR runs
        messages = await repair_fn(state, {})
        state.context.extend(messages)

        # Second GENERATE produces valid output
        valid_payload = {"title": "Inception", "rating": 9.0, "summary": "Great film."}
        state.context.append(_msg(json.dumps(valid_payload)))

        assert route_fn(state) == END

    @pytest.mark.asyncio
    async def test_route_gives_end_after_exhausting_attempts(self):
        """Simulates: max repairs reached without valid output → END."""
        route_fn = _make_route_fn(MovieReview, max_attempts=1, has_tools=False)
        repair_fn = _make_repair_fn(MovieReview)

        state = _state_with(_msg("bad"))
        assert route_fn(state) == "REPAIR"

        await repair_fn(state, {})
        state.context.append(_msg("still bad"))

        # attempts == max_attempts now → should return END
        assert route_fn(state) == END
