"""
Base class for all LLM-as-judge criteria.

All five LLM criteria (llm_judge, hallucination, factual_accuracy, safety, rubric)
follow the same pattern:

  1. Optional early-return guard (no response, no rubrics, …)
  2. Build a prompt from the execution result and expected case
  3. Call the LLM num_samples times, collect scores + extras per sample
  4. Average scores, aggregate extras
  5. Return CriterionResult with structured details

TemplatedLLMCriterion encodes that skeleton once.  Subclasses only need to
implement _build_prompt() and optionally override the hooks below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.criteria.base import BaseCriterion
from agentflow.qa.evaluation.criteria.llm_utils import LLMCallerMixin
from agentflow.qa.evaluation.eval_result import CriterionResult
from agentflow.qa.evaluation.token_usage import TokenUsage


if TYPE_CHECKING:
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase
    from agentflow.qa.evaluation.execution.result import ExecutionResult

logger = logging.getLogger("agentflow.evaluation")


class TemplatedLLMCriterion(LLMCallerMixin, BaseCriterion):
    """Base for LLM criteria that follow the sample-and-average pattern.

    Subclasses must implement:
        _build_prompt(actual, expected) -> str

    Subclasses may override any of the hooks below to customise behaviour
    without rewriting the entire evaluate() loop.
    """

    # ------------------------------------------------------------------
    # Hooks — override in subclasses as needed
    # ------------------------------------------------------------------

    def _get_skip_result(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult | None:
        """Return a ready-made result to skip evaluation, or None to proceed.

        Default: skip (score=1.0) when actual_response is empty.
        """
        if not actual.actual_response:
            return self._result(1.0, {"note": "No response to evaluate"})
        return None

    def _build_prompt(self, actual: ExecutionResult, expected: EvalCase) -> str:
        raise NotImplementedError(f"{type(self).__name__} must implement _build_prompt()")

    def _collect_extras(self, result_dict: dict[str, Any]) -> dict[str, Any]:
        """Extract criterion-specific fields from one LLM response dict.

        Default: nothing extra.
        """
        return {}

    def _aggregate_extras(self, per_sample: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge per-sample extras into a single dict for the details block.

        Default: nothing extra.
        """
        return {}

    def _build_details(
        self,
        scores: list[float],
        reasonings: list[str],
        aggregated_extras: dict[str, Any],
        final_score: float,
    ) -> dict[str, Any]:
        """Build the details dict for the CriterionResult.

        Default: samples count + reasonings + aggregated extras.
        """
        return {"samples": len(scores), "reasonings": reasonings, **aggregated_extras}

    # ------------------------------------------------------------------
    # Shared sampling loop
    # ------------------------------------------------------------------

    async def _run_samples(
        self, prompt: str
    ) -> tuple[list[float], list[str], list[dict[str, Any]], TokenUsage]:
        """Call the judge LLM num_samples times.

        Returns:
            (scores, reasonings, per_sample_extras, total_token_usage)
        """
        scores: list[float] = []
        reasonings: list[str] = []
        extras: list[dict[str, Any]] = []
        total_usage = TokenUsage()

        for _ in range(self.config.num_samples):
            try:
                result, usage = await self._call_llm_json(prompt)
                scores.append(float(result.get("score", 0.0)))
                reasonings.append(result.get("reasoning", ""))
                extras.append(self._collect_extras(result))
                total_usage = total_usage + usage
            except Exception as e:
                logger.warning("%s sample failed: %s", self.name, e)

        return scores, reasonings, extras, total_usage

    # ------------------------------------------------------------------
    # Template evaluate() — shared across all subclasses
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        try:
            skip = self._get_skip_result(actual, expected)
            if skip is not None:
                return skip

            prompt = self._build_prompt(actual, expected)
            scores, reasonings, extras, token_usage = await self._run_samples(prompt)

            if not scores:
                return self._failure("All LLM samples failed")

            final_score = sum(scores) / len(scores)
            agg = self._aggregate_extras(extras)
            details = self._build_details(scores, reasonings, agg, final_score)

            return self._result(final_score, details, token_usage)

        except Exception as e:
            logger.error("%s evaluation failed: %s", self.name, e)
            return self._failure(str(e))
