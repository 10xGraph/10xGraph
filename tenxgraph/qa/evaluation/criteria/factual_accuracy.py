"""Factual accuracy evaluation criterion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.criteria.llm_base import TemplatedLLMCriterion


if TYPE_CHECKING:
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase
    from agentflow.qa.evaluation.execution.result import ExecutionResult


FACTUAL_ACCURACY_PROMPT = """
You are an expert fact-checker. Your task is to evaluate the factual accuracy
of the RESPONSE.

QUESTION:
{question}

RESPONSE:
{response}

REFERENCE FACTS (if available):
{reference}

Evaluate the response for factual accuracy:
1. Are all stated facts correct?
2. Are numbers, dates, and names accurate?
3. Are any claims verifiably false?

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "is_accurate": <true/false>,
    "errors": [<list of factual errors found>],
    "reasoning": "<explanation>"
}}
"""


class FactualAccuracyCriterion(TemplatedLLMCriterion):
    """Evaluate factual accuracy of responses."""

    name = "factual_accuracy_v1"
    description = "LLM-based factual accuracy evaluation"

    def _build_prompt(self, actual: ExecutionResult, expected: EvalCase) -> str:
        reference = self._extract_reference(expected)
        return FACTUAL_ACCURACY_PROMPT.format(
            question=self._extract_question(expected),
            response=actual.actual_response,
            reference=reference or "Not provided",
        )

    def _collect_extras(self, result_dict: dict[str, Any]) -> dict[str, Any]:
        return {"errors": result_dict.get("errors", [])}

    def _aggregate_extras(self, per_sample: list[dict[str, Any]]) -> dict[str, Any]:
        all_errors: list[str] = []
        for s in per_sample:
            all_errors.extend(s.get("errors", []))
        return {"errors": list(set(all_errors))}

    def _build_details(
        self,
        scores: list[float],
        reasonings: list[str],
        aggregated_extras: dict[str, Any],
        final_score: float,
    ) -> dict[str, Any]:
        return {
            "is_accurate": final_score >= self.threshold,
            "samples": len(scores),
            "reasonings": reasonings,
            **aggregated_extras,
        }

    def _extract_reference(self, expected: EvalCase) -> str:
        text = self._extract_last_expected_response(expected)
        return text or expected.metadata.get("reference_facts", "")
