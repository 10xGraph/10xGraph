"""Hallucination/groundedness detection criterion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.criteria.llm_base import TemplatedLLMCriterion


if TYPE_CHECKING:
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase
    from agentflow.qa.evaluation.execution.result import ExecutionResult


HALLUCINATION_PROMPT = """
You are an expert fact-checker. Your task is to evaluate whether the RESPONSE is
grounded in the provided CONTEXT and does not contain hallucinations.

CONTEXT (Information available to the agent):
{context}

USER QUESTION:
{question}

AGENT RESPONSE:
{response}

Evaluate the response for hallucinations:
1. Check if all claims in the response are supported by the context
2. Check if any facts are fabricated or contradicted by the context
3. Check if any numbers, dates, or specific details are incorrect

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "is_grounded": <true/false>,
    "hallucinations": [<list of specific hallucinations found, empty if none>],
    "reasoning": "<explanation>"
}}

Score: 1.0 = fully grounded, 0.0 = mostly hallucinated.
"""


class HallucinationCriterion(TemplatedLLMCriterion):
    """Evaluate response groundedness and detect hallucinations."""

    name = "hallucinations_v1"
    description = "LLM-based hallucination/groundedness detection"

    def _build_prompt(self, actual: ExecutionResult, expected: EvalCase) -> str:
        context = self._extract_context(actual, expected)
        return HALLUCINATION_PROMPT.format(
            context=context or "No context available",
            question=self._extract_question(expected),
            response=actual.actual_response,
        )

    def _collect_extras(self, result_dict: dict[str, Any]) -> dict[str, Any]:
        return {"hallucinations": result_dict.get("hallucinations", [])}

    def _aggregate_extras(self, per_sample: list[dict[str, Any]]) -> dict[str, Any]:
        all_items: list[str] = []
        for s in per_sample:
            all_items.extend(s.get("hallucinations", []))
        return {"hallucinations": list(set(all_items))}

    def _build_details(
        self,
        scores: list[float],
        reasonings: list[str],
        aggregated_extras: dict[str, Any],
        final_score: float,
    ) -> dict[str, Any]:
        return {
            "is_grounded": final_score >= self.threshold,
            "samples": len(scores),
            "reasonings": reasonings,
            **aggregated_extras,
        }

    def _extract_context(self, actual: ExecutionResult, expected: EvalCase) -> str:
        parts: list[str] = []
        for tc in actual.tool_calls:
            if tc.result:
                parts.append(f"Tool {tc.name} returned: {tc.result}")
        if expected.metadata.get("context"):
            parts.append(f"Reference context: {expected.metadata['context']}")
        return "\n\n".join(parts)
