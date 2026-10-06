"""
Tests for the evaluation module data models.

Tests for EvalSet, EvalCase, EvalConfig, and related models.
"""

import json
import tempfile
from pathlib import Path

import pytest

from tenxgraph.qa.evaluation import (
    CriterionConfig,
    CriteriaConfig,
    EvalCase,
    EvalConfig,
    EvalSet,
    Invocation,
    MatchType,
    MessageContent,
    Rubric,
    SessionInput,
    StepType,
    ToolCall,
    TrajectoryStep,
)


class TestToolCall:
    """Tests for ToolCall model."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tc = ToolCall(name="get_weather", args={"location": "NYC"})
        assert tc.name == "get_weather"
        assert tc.args == {"location": "NYC"}
        assert tc.call_id is None
        assert tc.result is None

    def test_tool_call_with_all_fields(self):
        """Test tool call with all fields."""
        tc = ToolCall(
            name="search",
            args={"query": "test"},
            call_id="call_123",
            result={"results": ["a", "b"]},
        )
        assert tc.name == "search"
        assert tc.call_id == "call_123"
        assert tc.result == {"results": ["a", "b"]}

    def test_tool_call_matches_same(self):
        """Test matching identical tool calls."""
        tc1 = ToolCall(name="get_weather", args={"location": "NYC"})
        tc2 = ToolCall(name="get_weather", args={"location": "NYC"})
        assert tc1.matches(tc2)
        assert tc1 == tc2

    def test_tool_call_matches_different_name(self):
        """Test matching tool calls with different names."""
        tc1 = ToolCall(name="get_weather", args={"location": "NYC"})
        tc2 = ToolCall(name="get_time", args={"location": "NYC"})
        assert not tc1.matches(tc2)
        assert tc1 != tc2

    def test_tool_call_matches_different_args(self):
        """Test matching tool calls with different args."""
        tc1 = ToolCall(name="get_weather", args={"location": "NYC"})
        tc2 = ToolCall(name="get_weather", args={"location": "LA"})
        assert not tc1.matches(tc2, check_args=True)
        assert tc1.matches(tc2, check_args=False)

    def test_tool_call_matches_with_call_id(self):
        """Test matching tool calls checking call_id."""
        tc1 = ToolCall(name="get_weather", args={}, call_id="call_1")
        tc2 = ToolCall(name="get_weather", args={}, call_id="call_2")
        assert tc1.matches(tc2, check_call_id=False)
        assert not tc1.matches(tc2, check_call_id=True)


class TestTrajectoryStep:
    """Tests for TrajectoryStep model."""

    def test_trajectory_step_node_factory(self):
        """Test creating node step via factory method."""
        step = TrajectoryStep.node("agent_node", timestamp=1234567890.0)
        assert step.step_type == StepType.NODE
        assert step.name == "agent_node"
        assert step.timestamp == 1234567890.0
        assert step.args == {}

    def test_trajectory_step_tool_factory(self):
        """Test creating tool step via factory method."""
        step = TrajectoryStep.tool(
            "get_weather",
            args={"location": "NYC"},
            timestamp=1234567890.0,
            tool_call_id="call_123",
        )
        assert step.step_type == StepType.TOOL
        assert step.name == "get_weather"
        assert step.args == {"location": "NYC"}
        assert step.metadata["tool_call_id"] == "call_123"

    def test_trajectory_step_direct_creation(self):
        """Test creating step directly."""
        step = TrajectoryStep(
            step_type=StepType.MESSAGE,
            name="assistant_response",
            metadata={"content": "Hello!"},
        )
        assert step.step_type == StepType.MESSAGE
        assert step.name == "assistant_response"


class TestMessageContent:
    """Tests for MessageContent model."""

    def test_message_content_user_factory(self):
        """Test creating user message."""
        msg = MessageContent.user("What's the weather?")
        assert msg.role == "user"
        assert msg.content == "What's the weather?"

    def test_message_content_assistant_factory(self):
        """Test creating assistant message."""
        msg = MessageContent.assistant("The weather is sunny.")
        assert msg.role == "assistant"
        assert msg.content == "The weather is sunny."

    def test_message_content_get_text_string(self):
        """Test extracting text from string content."""
        msg = MessageContent(role="user", content="Hello world")
        assert msg.get_text() == "Hello world"

    def test_message_content_get_text_blocks(self):
        """Test extracting text from content blocks."""
        msg = MessageContent(
            role="user",
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        )
        assert msg.get_text() == "Hello World"


class TestInvocation:
    """Tests for Invocation model."""

    def test_invocation_simple_factory(self):
        """Test creating invocation via simple factory."""
        inv = Invocation.simple(
            user_query="What's the weather?",
            expected_response="It's sunny.",
            expected_tools=[ToolCall(name="get_weather", args={"location": "NYC"})],
        )
        assert inv.user_content.get_text() == "What's the weather?"
        assert inv.expected_final_response is not None
        assert inv.expected_final_response.get_text() == "It's sunny."
        assert len(inv.expected_tool_trajectory) == 1
        assert inv.expected_tool_trajectory[0].name == "get_weather"

    def test_invocation_has_uuid(self):
        """Test invocation generates UUID by default."""
        inv = Invocation(user_content=MessageContent.user("Hello"))
        assert inv.invocation_id is not None
        assert len(inv.invocation_id) > 0


class TestSessionInput:
    """Tests for SessionInput model."""

    def test_session_input_defaults(self):
        """Test session input default values."""
        session = SessionInput()
        assert session.app_name == ""
        assert session.user_id == "test_user"
        assert session.state == {}
        assert session.config == {}

    def test_session_input_with_values(self):
        """Test session input with custom values."""
        session = SessionInput(
            app_name="weather_agent",
            user_id="user_123",
            state={"location": "NYC"},
            config={"temperature": 0.7},
        )
        assert session.app_name == "weather_agent"
        assert session.user_id == "user_123"
        assert session.state["location"] == "NYC"


class TestEvalCase:
    """Tests for EvalCase model."""

    def test_eval_case_single_turn_factory(self):
        """Test creating single-turn eval case."""
        case = EvalCase.single_turn(
            eval_id="test_001",
            user_query="What's the weather in NYC?",
            expected_response="It's sunny in NYC.",
            expected_tools=[ToolCall(name="get_weather", args={"location": "NYC"})],
            name="Weather Query Test",
        )
        assert case.eval_id == "test_001"
        assert case.name == "Weather Query Test"
        assert len(case.conversation) == 1
        assert case.conversation[0].user_content.get_text() == "What's the weather in NYC?"

    def test_eval_case_multi_turn(self):
        """Test creating multi-turn eval case."""
        case = EvalCase(
            eval_id="test_002",
            name="Multi-turn Test",
            conversation=[
                Invocation.simple("Hello"),
                Invocation.simple("What's the weather?", "It's sunny."),
            ],
            tags=["weather", "multi-turn"],
        )
        assert len(case.conversation) == 2
        assert "weather" in case.tags
        assert "multi-turn" in case.tags


class TestEvalSet:
    """Tests for EvalSet model."""

    def test_eval_set_creation(self):
        """Test creating eval set."""
        eval_set = EvalSet(
            eval_set_id="weather_tests",
            name="Weather Agent Tests",
            description="Tests for weather agent functionality",
        )
        assert eval_set.eval_set_id == "weather_tests"
        assert eval_set.name == "Weather Agent Tests"
        assert len(eval_set) == 0

    def test_eval_set_add_case(self):
        """Test adding cases to eval set."""
        eval_set = EvalSet(eval_set_id="tests")
        case = EvalCase.single_turn("test_001", "Hello", "Hi there!")
        eval_set.add_case(case)
        assert len(eval_set) == 1
        assert eval_set.get_case("test_001") == case

    def test_eval_set_iteration(self):
        """Test iterating over eval set."""
        eval_set = EvalSet(eval_set_id="tests")
        for i in range(3):
            eval_set.add_case(EvalCase.single_turn(f"test_{i}", f"Query {i}"))
        
        ids = [case.eval_id for case in eval_set]
        assert ids == ["test_0", "test_1", "test_2"]

    def test_eval_set_filter_by_tags(self):
        """Test filtering cases by tags."""
        eval_set = EvalSet(eval_set_id="tests")
        eval_set.add_case(EvalCase(eval_id="1", tags=["weather", "basic"]))
        eval_set.add_case(EvalCase(eval_id="2", tags=["weather", "advanced"]))
        eval_set.add_case(EvalCase(eval_id="3", tags=["search"]))
        
        weather_cases = eval_set.filter_by_tags(["weather"])
        assert len(weather_cases) == 2
        
        advanced_weather = eval_set.filter_by_tags(["weather", "advanced"])
        assert len(advanced_weather) == 1
        assert advanced_weather[0].eval_id == "2"

    def test_eval_set_serialization(self):
        """Test saving and loading eval set to/from file."""
        eval_set = EvalSet(
            eval_set_id="test_set",
            name="Test Set",
            eval_cases=[
                EvalCase.single_turn("test_001", "Hello", "Hi!"),
            ],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_set.json"
            eval_set.to_file(str(path))
            
            # Verify file was created
            assert path.exists()
            
            # Load back
            loaded = EvalSet.from_file(str(path))
            assert loaded.eval_set_id == eval_set.eval_set_id
            assert loaded.name == eval_set.name
            assert len(loaded) == 1
            assert loaded.get_case("test_001") is not None


class TestCriterionConfig:
    """Tests for CriterionConfig model."""

    def test_criterion_config_defaults(self):
        """Test default criterion config."""
        config = CriterionConfig()
        assert config.threshold == 0.8
        assert config.match_type == MatchType.EXACT
        assert config.enabled is True

    def test_criterion_config_trajectory_factory(self):
        """Test trajectory config factory."""
        config = CriterionConfig.trajectory(
            threshold=0.9,
            match_type=MatchType.IN_ORDER,
            check_args=False,
        )
        assert config.threshold == 0.9
        assert config.match_type == MatchType.IN_ORDER
        assert config.check_args is False

    def test_criterion_config_llm_judge_factory(self):
        """Test LLM judge config factory."""
        config = CriterionConfig.llm_judge(
            threshold=0.85,
            judge_model="gpt-4",
            num_samples=5,
        )
        assert config.threshold == 0.85
        assert config.judge_model == "gpt-4"
        assert config.num_samples == 5

    def test_criterion_config_rubric_factory(self):
        """Test rubric config factory."""
        rubrics = [
            Rubric(rubric_id="concise", content="Response is concise."),
            Rubric(rubric_id="helpful", content="Response is helpful."),
        ]
        config = CriterionConfig.rubric_based(rubrics=rubrics, threshold=0.75)
        assert config.threshold == 0.75
        assert len(config.rubrics) == 2


class TestEvalConfig:
    """Tests for EvalConfig model."""

    def test_eval_config_default(self):
        """Test default eval config."""
        config = EvalConfig.default()
        assert config.criteria.trajectory is not None
        assert config.criteria.response_match is not None
        assert config.criteria.trajectory.threshold == 1.0
        assert config.criteria.response_match.threshold == 0.8

    def test_eval_config_strict(self):
        """Test strict eval config."""
        config = EvalConfig.strict()
        assert config.criteria.trajectory.threshold == 1.0
        assert config.criteria.response_match.threshold == 0.9
        assert config.criteria.llm_judge is not None

    def test_eval_config_relaxed(self):
        """Test relaxed eval config."""
        config = EvalConfig.relaxed()
        assert config.criteria.trajectory.threshold == 0.8
        assert config.criteria.trajectory.match_type == MatchType.IN_ORDER
        assert config.criteria.response_match.threshold == 0.6

    def test_eval_config_enable_criterion(self):
        """Test enabling a criterion."""
        config = EvalConfig()
        config.enable_criterion(
            "llm_judge",
            CriterionConfig(threshold=0.7),
        )
        assert config.criteria.llm_judge is not None
        assert config.criteria.llm_judge.threshold == 0.7

    def test_eval_config_disable_criterion(self):
        """Test disabling a criterion."""
        config = EvalConfig.default()
        config.disable_criterion("trajectory")
        assert config.criteria.trajectory.enabled is False

    def test_eval_config_with_rubrics(self):
        """Test adding rubrics to config."""
        config = EvalConfig.default()
        rubrics = [Rubric(rubric_id="test", content="Test rubric")]
        new_config = config.with_rubrics(rubrics)
        
        assert new_config.criteria.rubric_based is not None
        assert len(new_config.criteria.rubric_based.rubrics) == 1
        # Original should be unchanged
        assert config.criteria.rubric_based is None

    def test_eval_config_serialization(self):
        """Test saving and loading eval config."""
        config = EvalConfig.default()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.to_file(str(path))
            
            loaded = EvalConfig.from_file(str(path))
            assert loaded.criteria.trajectory is not None
            assert loaded.criteria.trajectory.threshold == 1.0

    def test_eval_config_serialization(self):
        """Test saving and loading eval config."""
        config = EvalConfig.default()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.to_file(str(path))
            
            loaded = EvalConfig.from_file(str(path))
            assert loaded.criteria.trajectory is not None
            assert loaded.criteria.trajectory.threshold == 1.0


class TestRubric:
    """Tests for Rubric model."""

    def test_rubric_creation(self):
        """Test creating a rubric."""
        rubric = Rubric(
            rubric_id="conciseness",
            content="The response is concise and to the point.",
            weight=1.5,
        )
        assert rubric.rubric_id == "conciseness"
        assert rubric.weight == 1.5

    def test_rubric_create_factory(self):
        """Test rubric create factory."""
        rubric = Rubric.create(
            rubric_id="helpful",
            content="Response is helpful.",
            weight=2.0,
        )
        assert rubric.rubric_id == "helpful"
        assert rubric.weight == 2.0


# ── TokenUsage ────────────────────────────────────────────────────────────────


class TestTokenUsage:
    """Unit tests for the TokenUsage dataclass."""

    def _make(self, inp: int = 0, out: int = 0, cr: int = 0, cc: int = 0):
        from tenxgraph.qa.evaluation.token_usage import TokenUsage

        return TokenUsage(
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cr,
            cache_creation_tokens=cc,
        )

    def test_total_tokens_is_input_plus_output(self):
        tok = self._make(inp=100, out=50)
        assert tok.total_tokens == 150

    def test_total_tokens_excludes_cache_fields(self):
        tok = self._make(inp=100, out=50, cr=200, cc=300)
        assert tok.total_tokens == 150

    def test_add_sums_all_fields(self):
        a = self._make(inp=100, out=40, cr=10, cc=5)
        b = self._make(inp=200, out=80, cr=20, cc=15)
        result = a + b
        assert result.input_tokens == 300
        assert result.output_tokens == 120
        assert result.cache_read_tokens == 30
        assert result.cache_creation_tokens == 20
        assert result.total_tokens == 420

    def test_add_returns_new_instance(self):
        a = self._make(inp=10, out=5)
        b = self._make(inp=20, out=10)
        result = a + b
        assert result is not a
        assert result is not b

    def test_radd_with_zero_supports_sum(self):
        from tenxgraph.qa.evaluation.token_usage import TokenUsage

        tokens = [
            self._make(inp=100, out=30),
            self._make(inp=200, out=60),
            self._make(inp=50, out=15),
        ]
        total = sum(tokens, TokenUsage())
        assert total.input_tokens == 350
        assert total.output_tokens == 105
        assert total.total_tokens == 455

    def test_radd_with_non_zero_returns_not_implemented(self):
        tok = self._make(inp=10, out=5)
        result = tok.__radd__(42)
        assert result is NotImplemented

    def test_to_dict_includes_all_fields_and_total(self):
        tok = self._make(inp=312, out=84, cr=100, cc=0)
        d = tok.to_dict()
        assert d == {
            "input_tokens": 312,
            "output_tokens": 84,
            "cache_read_tokens": 100,
            "cache_creation_tokens": 0,
            "total_tokens": 396,
        }

    def test_default_instance_is_all_zeros(self):
        from tenxgraph.qa.evaluation.token_usage import TokenUsage

        tok = TokenUsage()
        assert tok.input_tokens == 0
        assert tok.output_tokens == 0
        assert tok.total_tokens == 0


# ── EvalSummary token aggregation ─────────────────────────────────────────────


class TestEvalSummaryTokenAggregation:
    """Verify token fields are correctly aggregated in EvalSummary."""

    def test_total_token_usage_sums_all_cases(self):
        from tenxgraph.qa.evaluation.eval_result import EvalCaseResult, EvalSummary
        from tenxgraph.qa.evaluation.token_usage import TokenUsage

        r1 = EvalCaseResult.success(
            eval_id="c1",
            actual_response="",
            actual_tool_calls=[],
            actual_trajectory=[],
            criterion_results=[],
            duration_seconds=1.0,
            token_usage=TokenUsage(input_tokens=100, output_tokens=40),
        )
        r2 = EvalCaseResult.success(
            eval_id="c2",
            actual_response="",
            actual_tool_calls=[],
            actual_trajectory=[],
            criterion_results=[],
            duration_seconds=1.0,
            token_usage=TokenUsage(input_tokens=200, output_tokens=60),
        )
        summary = EvalSummary.from_results([r1, r2])
        assert summary.total_token_usage.input_tokens == 300
        assert summary.total_token_usage.output_tokens == 100
        assert summary.total_token_usage.total_tokens == 400
        assert summary.avg_tokens_per_case == 200.0

    def test_per_case_token_usage_keyed_by_eval_id(self):
        from tenxgraph.qa.evaluation.eval_result import EvalCaseResult, EvalSummary
        from tenxgraph.qa.evaluation.token_usage import TokenUsage

        r = EvalCaseResult.success(
            eval_id="my_case",
            actual_response="",
            actual_tool_calls=[],
            actual_trajectory=[],
            criterion_results=[],
            duration_seconds=0.5,
            token_usage=TokenUsage(input_tokens=50, output_tokens=25),
        )
        summary = EvalSummary.from_results([r])
        assert "my_case" in summary.per_case_token_usage
        assert summary.per_case_token_usage["my_case"].total_tokens == 75

    def test_model_dump_includes_total_tokens_field(self):
        from tenxgraph.qa.evaluation.eval_result import EvalCaseResult, EvalSummary
        from tenxgraph.qa.evaluation.token_usage import TokenUsage

        r = EvalCaseResult.success(
            eval_id="x",
            actual_response="",
            actual_tool_calls=[],
            actual_trajectory=[],
            criterion_results=[],
            duration_seconds=0.1,
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        summary = EvalSummary.from_results([r])
        d = summary.model_dump()
        assert "total_tokens" in d["total_token_usage"]
        assert d["total_token_usage"]["total_tokens"] == 15
        assert "total_tokens" in list(d["per_case_token_usage"].values())[0]

