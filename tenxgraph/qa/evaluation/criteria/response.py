"""
Response matching criteria.

ResponseMatchCriterion: LLM semantic matching (default, recommended).
RougeMatchCriterion: Fast ROUGE-1 F1 score (opt-in for cheap/fast checks).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agentflow.qa.evaluation.criteria.base import SyncCriterion
from agentflow.qa.evaluation.criteria.llm_judge import LLMJudgeCriterion


if TYPE_CHECKING:
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase
    from agentflow.qa.evaluation.eval_result import CriterionResult
    from agentflow.qa.evaluation.execution.result import ExecutionResult


class ResponseMatchCriterion(LLMJudgeCriterion):
    """LLM-based semantic response matching (recommended default).

    Uses gemini-2.5-flash to judge whether the actual response is semantically
    equivalent to the expected response. Handles paraphrasing, short answers,
    and variations in phrasing that ROUGE-1 fails on.

    For fast/cheap checks, use RougeMatchCriterion instead.
    """

    name = "response_match_score"
    description = "LLM semantic response matching"


class RougeMatchCriterion(SyncCriterion):
    """Compare response similarity using ROUGE-1 F1 score.

    Opt-in for fast, cheap checks where LLM calls are not desired.
    Fails for short expected outputs (e.g. single-word answers) and
    semantically equivalent but differently-worded responses.
    Use ResponseMatchCriterion (LLM-based) as the default instead.
    """

    name = "rouge_match"
    description = "Evaluates response text similarity using ROUGE-1"

    def evaluate_sync(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        expected_response = ""
        for invocation in expected.conversation:
            if invocation.expected_final_response:
                expected_response = invocation.expected_final_response.get_text()

        actual_response = actual.actual_response

        if not expected_response:
            score = 1.0 if not actual_response else 0.5
            return self._result(score, {"note": "No expected response defined"})

        precision, recall, f1 = self._rouge1_score(actual_response, expected_response)

        return self._result(
            f1,
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "actual_response": actual_response[:500],
                "expected_response": expected_response[:500],
            },
        )

    def _rouge1_score(self, actual: str, expected: str) -> tuple[float, float, float]:
        actual_tokens = set(self._tokenize(actual))
        expected_tokens = set(self._tokenize(expected))

        if not expected_tokens:
            return (1.0, 1.0, 1.0) if not actual_tokens else (0.0, 0.0, 0.0)
        if not actual_tokens:
            return (0.0, 0.0, 0.0)

        overlap = len(actual_tokens & expected_tokens)
        precision = overlap / len(actual_tokens)
        recall = overlap / len(expected_tokens)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        return (precision, recall, f1)

    def _tokenize(self, text: str) -> list[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text.split() if t]


class ExactMatchCriterion(SyncCriterion):
    """Require exact string match between actual and expected response."""

    name = "exact_match"
    description = "Evaluates exact string match of response"

    def evaluate_sync(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        expected_response = ""
        for invocation in expected.conversation:
            if invocation.expected_final_response:
                expected_response = invocation.expected_final_response.get_text().strip()

        actual_response = actual.actual_response.strip()
        is_match = actual_response == expected_response

        return self._result(
            1.0 if is_match else 0.0,
            {
                "match": is_match,
                "actual_length": len(actual_response),
                "expected_length": len(expected_response),
            },
        )


class ContainsKeywordsCriterion(SyncCriterion):
    """Check if response contains required keywords."""

    name = "contains_keywords"
    description = "Evaluates presence of required keywords in response"

    def __init__(self, keywords: list[str] | None = None, config=None):
        super().__init__(config)
        self.keywords = (config.keywords if config and config.keywords else None) or keywords or []

    def evaluate_sync(
        self,
        actual: ExecutionResult,
        expected: EvalCase,
    ) -> CriterionResult:
        if not self.keywords:
            return self._result(1.0, {"note": "No keywords specified"})

        actual_response = actual.actual_response.lower()
        found = [kw for kw in self.keywords if kw.lower() in actual_response]
        missing = [kw for kw in self.keywords if kw.lower() not in actual_response]
        score = len(found) / len(self.keywords)

        return self._result(score, {"found": found, "missing": missing, "keywords": self.keywords})
