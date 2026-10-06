"""Rubric-based evaluation criterion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.criteria.llm_base import TemplatedLLMCriterion
from agentflow.qa.evaluation.eval_result import CriterionResult


if TYPE_CHECKING:
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase
    from agentflow.qa.evaluation.execution.result import ExecutionResult


RUBRIC_PROMPT = """
You are an expert evaluator. Your task is to evaluate the quality of an AI
assistant's response.

QUESTION: {question}

RESPONSE: {response}

{rubric_section}

Evaluate the response and provide a score.

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "reasoning": "<brief explanation of your evaluation>"
}}
"""


class RubricBasedCriterion(TemplatedLLMCriterion):
    """Evaluate response quality against custom rubrics."""

    name = "rubric_based_final_response_quality_v1"
    description = "LLM-based rubric evaluation"

    def _get_skip_result(
        self, actual: ExecutionResult, expected: EvalCase
    ) -> CriterionResult | None:
        if not self.config.rubrics:
            return self._result(1.0, {"note": "No rubrics configured"})
        return super()._get_skip_result(actual, expected)

    def _build_prompt(self, actual: ExecutionResult, expected: EvalCase) -> str:
        rubric_lines = ["Evaluate the response against these criteria:"]
        for rubric in self.config.rubrics:
            rubric_lines.append(f"- {rubric.rubric_id}: {rubric.content}")
        return RUBRIC_PROMPT.format(
            question=self._extract_question(expected),
            response=actual.actual_response,
            rubric_section="\n".join(rubric_lines),
        )

    def _build_details(
        self,
        scores: list[float],
        reasonings: list[str],
        aggregated_extras: dict[str, Any],
        final_score: float,
    ) -> dict[str, Any]:
        return {
            "rubrics": [r.rubric_id for r in self.config.rubrics],
            "scores": scores,
            "reasonings": reasonings,
        }
