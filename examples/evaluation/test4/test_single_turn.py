"""
test4 — QuickEval, EvalSetBuilder & Reporter tests (compact).

5 focused tests covering:
  1. QuickEval.check()
  2. QuickEval.batch() + tool_usage()
  3. QuickEval.conversation_flow()
  4. EvalSetBuilder + from_builder + presets
  5. Reporters (JSON + HTML)

Run:
    pytest examples/evaluation/test4/ -v -s
"""

import tempfile
from pathlib import Path

import pytest

pytest.skip(
    "Disabled in examples/evaluation so root pytest runs are unaffected.",
    allow_module_level=True,
)

from tenxgraph.qa.evaluation import (
    AgentEvaluator,
    CriterionConfig,
    EvalConfig,
    EvalPresets,
    EvalReport,
    EvalSetBuilder,
    QuickEval,
    ReporterConfig,
    assert_eval_passed,
)
from tenxgraph.qa.evaluation.dataset import EvalCase, ToolCall
from tenxgraph.qa.evaluation.dataset.eval_set import EvalSet

from .samples import BATCH_EVAL_SET


# ═══════════════════════════════════════════════════════════════════════
# 1. QuickEval.check()
# ═══════════════════════════════════════════════════════════════════════


class TestQuickEvalCheck:
    @pytest.mark.asyncio
    async def test_check(self, compiled_graph, collector):
        """QuickEval.check() with expected tool and response."""
        report = await QuickEval.check(
            graph=compiled_graph,
            collector=collector,
            query="What is the weather in London?",
            expected_response_contains="sunny",
            expected_tools=["get_weather"],
            threshold=0.5,
            print_results=True,
        )
        assert isinstance(report, EvalReport)
        assert_eval_passed(report)


# ═══════════════════════════════════════════════════════════════════════
# 2. QuickEval.batch() + tool_usage()
# ═══════════════════════════════════════════════════════════════════════


class TestQuickEvalBatchAndTools:
    @pytest.mark.asyncio
    async def test_batch_and_tool_usage(self, compiled_graph, collector):
        """batch() runs multiple pairs; tool_usage() verifies strict matching."""
        # -- batch --
        report_batch = await QuickEval.batch(
            graph=compiled_graph,
            collector=collector,
            test_pairs=[
                ("Weather in NYC?", "The weather in NYC is sunny"),
                ("Capital of Spain?", "The capital of Spain is Madrid"),
            ],
            threshold=0.3,
            print_results=True,
        )
        assert isinstance(report_batch, EvalReport)
        assert report_batch.summary.total_cases == 2
        assert_eval_passed(report_batch)

        # -- tool usage --
        report_tools = await QuickEval.tool_usage(
            graph=compiled_graph,
            collector=collector,
            test_cases=[
                ("Weather in Paris?", "sunny", ["get_weather"]),
                ("Forecast for Tokyo?", "forecast", ["get_forecast"]),
            ],
            strict=True,
            print_results=True,
        )
        assert isinstance(report_tools, EvalReport)
        assert report_tools.summary.total_cases == 2
        assert_eval_passed(report_tools)


# ═══════════════════════════════════════════════════════════════════════
# 3. QuickEval.conversation_flow()
# ═══════════════════════════════════════════════════════════════════════


class TestQuickEvalConversation:
    @pytest.mark.asyncio
    async def test_conversation_flow(self, compiled_graph, collector):
        """Multi-turn conversation evaluation via QuickEval."""
        report = await QuickEval.conversation_flow(
            graph=compiled_graph,
            collector=collector,
            conversation=[
                ("What's the weather in London?", "The weather in London is sunny"),
                ("And in Paris?", "The weather in Paris is sunny"),
            ],
            threshold=0.3,
            print_results=True,
        )
        assert isinstance(report, EvalReport)
        assert_eval_passed(report)


# ═══════════════════════════════════════════════════════════════════════
# 4. EvalSetBuilder + from_builder + presets
# ═══════════════════════════════════════════════════════════════════════


class TestBuilderAndPresets:
    @pytest.mark.asyncio
    async def test_builder_preset_and_from_builder(self, compiled_graph, collector):
        """Build EvalSet, run with preset, then use from_builder shorthand."""
        # -- builder + preset --
        eval_set = EvalSet(
            eval_set_id="preset_test",
            name="Preset Quick Check",
            eval_cases=[
                EvalCase.single_turn(
                    eval_id="preset_1",
                    user_query="Weather in Rome?",
                    expected_response="The weather in Rome is sunny",
                    expected_tools=[ToolCall(name="get_weather")],
                ),
            ],
        )
        report_preset = await QuickEval.preset(
            graph=compiled_graph,
            collector=collector,
            preset=EvalPresets.quick_check(),
            eval_set=eval_set,
            print_results=True,
        )
        assert isinstance(report_preset, EvalReport)
        assert_eval_passed(report_preset)

        # -- from_builder --
        builder = EvalSetBuilder("quick_builder")
        builder.add_case(
            query="Weather in Amsterdam?",
            expected="The weather in Amsterdam is sunny",
            expected_tools=["get_weather"],
        )
        builder.add_case(query="Capital of Brazil?", expected="The capital of Brazil is Brasilia")
        report_fb = await QuickEval.from_builder(
            graph=compiled_graph,
            collector=collector,
            builder=builder,
            print_results=True,
        )
        assert isinstance(report_fb, EvalReport)
        assert report_fb.summary.total_cases == 2
        assert_eval_passed(report_fb)


# ═══════════════════════════════════════════════════════════════════════
# 5. Reporters (JSON + HTML)
# ═══════════════════════════════════════════════════════════════════════


class TestReporters:
    @pytest.mark.asyncio
    async def test_json_and_html_reporter(self, compiled_graph, collector):
        """Reporter generates both JSON and HTML output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(
                criteria={"response_match_score": CriterionConfig.response_match(threshold=0.3)},
                reporter=ReporterConfig(
                    enabled=True,
                    output_dir=tmpdir,
                    console=False,
                    json_report=True,
                    html=True,
                    junit_xml=False,
                    timestamp_files=False,
                ),
            )
            evaluator = AgentEvaluator(compiled_graph, collector, config=config)
            await evaluator.evaluate(BATCH_EVAL_SET)

            files = list(Path(tmpdir).iterdir())
            extensions = {f.suffix for f in files}
            assert ".json" in extensions
            assert ".html" in extensions
