"""Conversation goals criterion (UserSimulator only).

Evaluates whether goals were achieved across a full multi-turn
conversation transcript produced by UserSimulator.

This criterion expects actual_response to contain the full conversation
transcript (all turns), as set by UserSimulator._evaluate_simulation().
It will not work correctly with the normal AgentEvaluator flow where
actual_response is only the agent's final response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.criteria.llm_base import TemplatedLLMCriterion
from agentflow.qa.evaluation.eval_result import CriterionResult
from agentflow.qa.evaluation.token_usage import TokenUsage


if TYPE_CHECKING:
    from agentflow.qa.evaluation.dataset.eval_set import EvalCase
    from agentflow.qa.evaluation.execution.result import ExecutionResult


CONVERSATION_GOALS_PROMPT = (
    "You are evaluating whether an AI assistant achieved specific goals "
    "across a multi-turn conversation.\n\n"
    "CONVERSATION TRANSCRIPT:\n{conversation}\n\n"
    "GOALS TO ACHIEVE:\n{goals}\n\n"
    "For each goal, determine if it was achieved at any point during the conversation -\n"
    "not just in the final message.\n\n"
    "Respond with a JSON object:\n"
    "{{\n"
    '    "score": <float 0.0 to 1.0, where 1.0 = all goals achieved, 0.0 = none achieved>,\n'
    '    "achieved_goals": ["<exact goal text>", ...],\n'
    '    "unachieved_goals": ["<exact goal text>", ...],\n'
    '    "reasoning": "<brief explanation covering each goal>"\n'
    "}}\n\n"
    "Score = number_of_achieved_goals / total_goals."
)


class SimulationGoalsCriterion(TemplatedLLMCriterion):
    """Evaluate whether goals were achieved in a UserSimulator conversation.

    UserSimulator-only — expects actual_response to contain the full
    multi-turn conversation transcript, not just the final agent reply.

    Score: achieved_goals / total_goals  (0.0 - 1.0)
    """

    name = "simulation_goals"
    description = "LLM-based goal achievement evaluation for UserSimulator conversations"

    def _get_skip_result(
        self, actual: ExecutionResult, expected: EvalCase
    ) -> CriterionResult | None:
        if not self._extract_last_expected_response(expected):
            return self._result(1.0, {"note": "No goals defined — skipping evaluation"})
        return None  # no "empty response" guard — transcript may be long

    def _build_prompt(self, actual: ExecutionResult, expected: EvalCase) -> str:
        return CONVERSATION_GOALS_PROMPT.format(
            conversation=actual.actual_response,
            goals=self._extract_last_expected_response(expected),
        )

    async def _run_samples(
        self, prompt: str
    ) -> tuple[list[float], list[str], list[dict[str, Any]], TokenUsage]:
        """Single call — goals evaluation doesn't need majority voting."""
        result, usage = await self._call_llm_json(prompt)
        score = float(result.get("score", 0.0))
        extras = {
            "achieved_goals": result.get("achieved_goals", []),
            "unachieved_goals": result.get("unachieved_goals", []),
        }
        return [score], [result.get("reasoning", "")], [extras], usage

    def _build_details(
        self,
        scores: list[float],
        reasonings: list[str],
        aggregated_extras: dict[str, Any],
        final_score: float,
    ) -> dict[str, Any]:
        return {
            "achieved_goals": aggregated_extras.get("achieved_goals", []),
            "unachieved_goals": aggregated_extras.get("unachieved_goals", []),
            "reasoning": reasonings[0] if reasonings else "",
            "reason": reasonings[0] if reasonings else "",
            "judge_model": self.config.judge_model,
        }

    def _aggregate_extras(self, per_sample: list[dict[str, Any]]) -> dict[str, Any]:
        return per_sample[0] if per_sample else {}
