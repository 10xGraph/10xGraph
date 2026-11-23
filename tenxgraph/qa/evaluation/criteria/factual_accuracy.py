"""Factual accuracy evaluation criterion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tenxgraph.qa.evaluation.criteria.llm_base import TemplatedLLMCriterion


if TYPE_CHECKING:
    from tenxgraph.qa.evaluation.dataset.eval_set import EvalCase
    from tenxgraph.qa.evaluation.execution.result import ExecutionResult


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

    _verdict_key = "is_accurate"
    _list_key = "errors"

    def _build_prompt(self, actual: ExecutionResult, expected: EvalCase) -> str:
        reference = self._extract_reference(expected)
        return FACTUAL_ACCURACY_PROMPT.format(
            question=self._extract_question(expected),
            response=actual.actual_response,
            reference=reference or "Not provided",
        )

    def _extract_reference(self, expected: EvalCase) -> str:
        text = self._extract_last_expected_response(expected)
        return text or expected.metadata.get("reference_facts", "")
