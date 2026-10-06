"""LLM-as-Judge criterion for semantic response matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.criteria.llm_base import TemplatedLLMCriterion
from agentflow.qa.evaluation.eval_result import CriterionResult


if TYPE_CHECKING:
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase
    from agentflow.qa.evaluation.execution.result import ExecutionResult


SEMANTIC_MATCH_PROMPT = """
You are an expert evaluator. Your task is to determine if the ACTUAL response is
semantically equivalent to the EXPECTED response.

QUESTION: {question}

EXPECTED RESPONSE: {expected}

ACTUAL RESPONSE: {actual}

Evaluate if the ACTUAL response conveys the same meaning and information as the
EXPECTED response. Minor differences in wording are acceptable as long as the core
meaning is preserved.

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "reasoning": "<brief explanation of your evaluation>"
}}

Score guidelines:
- 1.0: Perfect semantic match, all key information preserved
- 0.8-0.9: Minor differences but core meaning intact
- 0.5-0.7: Partial match, some key information missing or different
- 0.2-0.4: Significant differences but some overlap
- 0.0-0.1: Completely different or contradictory
"""


class LLMJudgeCriterion(TemplatedLLMCriterion):
    """Use an LLM to judge semantic equivalence between actual and expected responses."""

    name = "final_response_match_v2"
    description = "LLM-based semantic response matching"

    def _get_skip_result(
        self, actual: ExecutionResult, expected: EvalCase
    ) -> CriterionResult | None:
        if not self._extract_last_expected_response(expected):
            return self._result(1.0, {"note": "No expected response to compare"})
        return super()._get_skip_result(actual, expected)

    def _build_prompt(self, actual: ExecutionResult, expected: EvalCase) -> str:
        return SEMANTIC_MATCH_PROMPT.format(
            question=self._extract_question(expected),
            expected=self._extract_last_expected_response(expected),
            actual=actual.actual_response,
        )

    def _build_details(
        self,
        scores: list[float],
        reasonings: list[str],
        aggregated_extras: dict[str, Any],
        final_score: float,
    ) -> dict[str, Any]:
        return {
            "samples": len(scores),
            "individual_scores": scores,
            "reasonings": reasonings,
            "judge_model": self.config.judge_model,
            "reason": reasonings[-1] if reasonings else "",
        }
