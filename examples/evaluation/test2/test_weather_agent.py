"""
test2 — All criteria single-turn evaluation tests (compact).

Verifies every criterion type works. Non-LLM criteria are batched into
one test, LLM criteria into another, to minimize agent invocations.

Run:
    pytest examples/evaluation/test2/ -v -s
"""

import pytest

pytest.skip(
    "Disabled in examples/evaluation so root pytest runs are unaffected.",
    allow_module_level=True,
)

from tenxgraph.qa.evaluation import (
    AgentEvaluator,
    CriterionConfig,
    EvalConfig,
    MatchType,
    Rubric,
)

from .samples import (
    CAPITAL_QUESTION,
    WEATHER_NYC,
    WEATHER_NYC_NODE_ORDER,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. All non-LLM criteria in one pass (tool match + trajectory + ROUGE
#    + keywords + node order) — single agent call
# ═══════════════════════════════════════════════════════════════════════


class TestNonLLMCriteria:
    @pytest.mark.asyncio
    async def test_all_non_llm_criteria(self, compiled_graph, collector):
        """tool_name_match + trajectory + rouge + keywords on one weather case."""
        config = EvalConfig(
            criteria={
                "tool_name_match_score": CriterionConfig.tool_name_match(threshold=1.0),
                "tool_trajectory_avg_score": CriterionConfig.trajectory(
                    threshold=1.0,
                    match_type=MatchType.EXACT,
                ),
                "rouge_match": CriterionConfig.rouge_match(threshold=0.4),
                "contains_keywords": CriterionConfig.contains_keywords(
                    keywords=["New York", "sunny"], threshold=0.5
                ),
            },
            reporter={"enabled": True},
        )
        evaluator = AgentEvaluator(compiled_graph, collector, config=config)
        result = await evaluator.evaluate_case(WEATHER_NYC)
        assert result.passed, f"Failed: {[c.criterion for c in result.failed_criteria]}"

    @pytest.mark.asyncio
    async def test_no_tool_case(self, compiled_graph, collector):
        """General knowledge: no tool called, keywords present."""
        config = EvalConfig(
            criteria={
                "tool_name_match_score": CriterionConfig.tool_name_match(threshold=1.0),
                "contains_keywords": CriterionConfig.contains_keywords(
                    keywords=["weather"], threshold=1.0
                ),
            },
            reporter={"enabled": True},
        )
        evaluator = AgentEvaluator(compiled_graph, collector, config=config)
        result = await evaluator.evaluate_case(CAPITAL_QUESTION)
        assert result.passed, f"Failed: {[c.criterion for c in result.failed_criteria]}"

    @pytest.mark.asyncio
    async def test_node_order(self, compiled_graph, collector):
        """Node order matches expected __start__ → MAIN → MAIN sequence."""
        config = EvalConfig(
            criteria={
                "node_order_score": CriterionConfig.node_order(
                    threshold=1.0,
                    match_type=MatchType.EXACT,
                ),
            },
            reporter={"enabled": True},
        )
        evaluator = AgentEvaluator(compiled_graph, collector, config=config)
        result = await evaluator.evaluate_case(WEATHER_NYC_NODE_ORDER)
        assert result.passed, f"Failed: {[c.criterion for c in result.failed_criteria]}"


# ═══════════════════════════════════════════════════════════════════════
# 2. All LLM criteria in one pass — single agent call, all judges run
# ═══════════════════════════════════════════════════════════════════════


class TestLLMCriteria:
    @pytest.mark.asyncio
    async def test_all_llm_criteria(self, compiled_graph, collector):
        """LLM judge + factual accuracy + hallucination + safety + rubric."""
        config = EvalConfig(
            criteria={
                "llm_judge": CriterionConfig.llm_judge(threshold=0.6, num_samples=1),
                "factual_accuracy_v1": CriterionConfig.factual_accuracy(
                    threshold=0.6, num_samples=1
                ),
                "hallucinations_v1": CriterionConfig.hallucination(threshold=0.6, num_samples=1),
                "safety_v1": CriterionConfig.safety(threshold=0.6, num_samples=1),
                "rubric_based": CriterionConfig.rubric_based(
                    rubrics=[
                        Rubric.create(
                            "accuracy", "Does the response contain factually correct information?"
                        ),
                        Rubric.create(
                            "relevance", "Is the response directly relevant to the user query?"
                        ),
                    ],
                    threshold=0.3,
                ),
            },
            reporter={"enabled": True},
        )
        evaluator = AgentEvaluator(compiled_graph, collector, config=config)
        result = await evaluator.evaluate_case(WEATHER_NYC)
        assert result.passed, f"Failed: {[c.criterion for c in result.failed_criteria]}"
        for cr in result.criterion_results:
            print(f"  {cr.criterion}: {cr.score:.3f} ({'PASS' if cr.passed else 'FAIL'})")


# ═══════════════════════════════════════════════════════════════════════
# 3. Full combined pipeline — all criteria types together
# ═══════════════════════════════════════════════════════════════════════


class TestCombinedPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, compiled_graph, collector):
        """All non-LLM + LLM criteria in one evaluation pass."""
        config = EvalConfig(
            criteria={
                "tool_name_match_score": CriterionConfig.tool_name_match(threshold=1.0),
                "tool_trajectory_avg_score": CriterionConfig.trajectory(
                    threshold=1.0,
                    match_type=MatchType.IN_ORDER,
                ),
                "rouge_match": CriterionConfig.rouge_match(threshold=0.4),
                "llm_judge": CriterionConfig.llm_judge(threshold=0.6, num_samples=1),
                "safety_v1": CriterionConfig.safety(threshold=0.6, num_samples=1),
            },
            reporter={"enabled": True},
        )
        evaluator = AgentEvaluator(compiled_graph, collector, config=config)
        result = await evaluator.evaluate_case(WEATHER_NYC)
        assert result.passed, f"Failed: {[c.criterion for c in result.failed_criteria]}"
