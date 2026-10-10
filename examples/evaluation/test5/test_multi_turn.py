"""
test5 — AgentEvaluator API & Config deep-dive tests (compact).

5 focused tests covering:
  1. evaluate_case() — single-case result structure + inspection
  2. evaluate() — batch with report structure
  3. EvalConfig manipulation — rubrics + enable/disable
  4. Presets comparison + EvaluationRunner
  5. Auto-reporter — JSON + HTML file generation

Run:
    pytest examples/evaluation/test5/ -v -s
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
    EvaluationRunner,
    ReporterConfig,
    Rubric,
)
from tenxgraph.qa.evaluation.dataset.eval_set import EvalSet

from .samples import NO_TOOL_CASE, SMALL_EVAL_SET, WEATHER_CASE


# ═══════════════════════════════════════════════════════════════════════
# 1. evaluate_case() — result structure + inspection
# ═══════════════════════════════════════════════════════════════════════


class TestEvaluateCase:
    @pytest.mark.asyncio
    async def test_result_structure_and_fields(self, compiled_graph, collector):
        """evaluate_case() returns correct id, scores, duration, response, tools."""
        config = EvalConfig(
            criteria={
                "tool_name_match_score": CriterionConfig.tool_name_match(threshold=1.0),
                "response_match_score": CriterionConfig.response_match(threshold=0.3),
            },
            reporter={"enabled": True},
        )
        evaluator = AgentEvaluator(compiled_graph, collector, config=config)
        result = await evaluator.evaluate_case(WEATHER_CASE)

        # structure
        assert result.eval_id == "deep_weather"
        assert isinstance(result.passed, bool)
        assert len(result.criterion_results) == 2
        for cr in result.criterion_results:
            assert cr.criterion
            assert 0.0 <= cr.score <= 1.0
            assert isinstance(cr.passed, bool)
        assert result.duration_seconds > 0

        # inspection fields
        assert result.actual_response is not None
        assert len(result.actual_response) > 0
        assert result.actual_tool_calls is not None
        assert len(result.actual_tool_calls) >= 1


# ═══════════════════════════════════════════════════════════════════════
# 2. evaluate() — batch report
# ═══════════════════════════════════════════════════════════════════════


class TestEvaluateBatch:
    @pytest.mark.asyncio
    async def test_report_structure(self, compiled_graph, collector):
        """evaluate() returns EvalReport with all cases and correct eval_set_id."""
        config = EvalConfig(
            criteria={"response_match_score": CriterionConfig.response_match(threshold=0.3)},
            reporter={"enabled": True},
        )
        evaluator = AgentEvaluator(compiled_graph, collector, config=config)
        report = await evaluator.evaluate(SMALL_EVAL_SET)

        assert isinstance(report, EvalReport)
        assert report.summary.total_cases == 3
        assert len(report.results) == 3
        assert report.eval_set_id == "deep_eval_set"


# ═══════════════════════════════════════════════════════════════════════
# 3. EvalConfig — rubrics + enable/disable
# ═══════════════════════════════════════════════════════════════════════


class TestConfigAndRubrics:
    @pytest.mark.asyncio
    async def test_rubrics_and_disable(self, compiled_graph, collector):
        """with_rubrics() adds rubric criterion; disable removes one."""
        # -- rubrics --
        base_cfg = EvalConfig(
            criteria={"response_match_score": CriterionConfig.response_match(threshold=0.3)},
            reporter={"enabled": True},
        )
        rubrics = [
            Rubric.create("helpfulness", "Is the response helpful?"),
            Rubric.create("relevance", "Does it address the question?"),
        ]
        cfg = base_cfg.with_rubrics(rubrics)
        assert "rubric_based" in cfg.criteria
        assert "response_match_score" in cfg.criteria

        evaluator = AgentEvaluator(compiled_graph, collector, config=cfg)
        result = await evaluator.evaluate_case(WEATHER_CASE)
        assert len(result.criterion_results) >= 2

        # -- disable criterion --
        config2 = EvalConfig(
            criteria={
                "tool_name_match_score": CriterionConfig.tool_name_match(threshold=1.0),
                "response_match_score": CriterionConfig.response_match(threshold=0.3),
            },
            reporter={"enabled": True},
        )
        config2.disable_criterion("response_match_score")
        evaluator2 = AgentEvaluator(compiled_graph, collector, config=config2)
        result2 = await evaluator2.evaluate_case(WEATHER_CASE)
        assert len(result2.criterion_results) == 1


# ═══════════════════════════════════════════════════════════════════════
# 4. Presets + EvaluationRunner
# ═══════════════════════════════════════════════════════════════════════


class TestPresetsAndRunner:
    @pytest.mark.asyncio
    async def test_presets_compare_and_runner(self, compiled_graph, collector):
        """Comprehensive has more criteria; runner aggregates multiple sets."""
        # -- presets comparison --
        quick_cfg = EvalPresets.quick_check()
        quick_cfg.reporter.enabled = True
        comp_cfg = EvalPresets.comprehensive(threshold=0.3, use_llm_judge=False)
        comp_cfg.reporter.enabled = True

        quick_eval = AgentEvaluator(compiled_graph, collector, config=quick_cfg)
        comp_eval = AgentEvaluator(compiled_graph, collector, config=comp_cfg)

        quick_result = await quick_eval.evaluate_case(WEATHER_CASE)
        comp_result = await comp_eval.evaluate_case(WEATHER_CASE)
        assert len(comp_result.criterion_results) > len(quick_result.criterion_results)

        # -- runner --
        set_a = EvalSet(eval_set_id="runner_a", name="Set A", eval_cases=[WEATHER_CASE])
        set_b = EvalSet(eval_set_id="runner_b", name="Set B", eval_cases=[NO_TOOL_CASE])

        runner_cfg = EvalConfig(
            criteria={"response_match_score": CriterionConfig.response_match(threshold=0.3)},
            reporter={"enabled": True},
        )
        runner = EvaluationRunner(default_config=runner_cfg)
        results = await runner.run(
            [
                (compiled_graph, collector, set_a),
                (compiled_graph, collector, set_b),
            ]
        )
        assert len(results) == 2
        assert runner.summary["total_evaluations"] == 2
        assert runner.summary["total_cases"] == 2


# ═══════════════════════════════════════════════════════════════════════
# 5. Auto-reporter — JSON + HTML
# ═══════════════════════════════════════════════════════════════════════


class TestAutoReporter:
    @pytest.mark.asyncio
    async def test_reporters_generate_files(self, compiled_graph, collector):
        """Reporter generates JSON + HTML files when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(
                criteria={"response_match_score": CriterionConfig.response_match(threshold=0.3)},
                reporter=ReporterConfig(
                    enabled=True,
                    output_dir=tmpdir,
                    console=False,
                    json_report=True,
                    html=True,
                    junit_xml=True,
                    timestamp_files=False,
                ),
            )
            evaluator = AgentEvaluator(compiled_graph, collector, config=config)
            await evaluator.evaluate(SMALL_EVAL_SET)

            files = list(Path(tmpdir).iterdir())
            extensions = {f.suffix for f in files}
            assert ".json" in extensions
            assert ".html" in extensions
