"""
AI-powered user simulation for dynamic conversation testing.
This module provides the UserSimulator class which uses an LLM to
simulate realistic user behavior during agent evaluation.

"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from tenxgraph.core.llm.caller import call_llm
from tenxgraph.qa.evaluation.token_usage import TokenUsage


if TYPE_CHECKING:
    from tenxgraph.core.graph.compiled_graph import CompiledGraph
    from tenxgraph.qa.evaluation.config.eval_config import UserSimulatorConfig
    from tenxgraph.qa.evaluation.criteria.base import BaseCriterion

logger = logging.getLogger("tenxgraph.evaluation")

USER_SIMULATOR_PROMPT = """You are simulating a user interacting with an AI assistant.

SCENARIO:
{scenario}

CONVERSATION PLAN:
{conversation_plan}

CONVERSATION SO FAR:
{conversation_history}

Based on the scenario and plan, generate the user's next message.
The message should:
1. Follow the conversation plan naturally
2. Be realistic and natural
3. Test the agent's capabilities appropriately

Respond with only the user's message (no JSON, no explanation).
"""

USER_RESPONSE_PROMPT = """You are simulating a user responding to an AI assistant.

SCENARIO:
{scenario}

LAST ASSISTANT MESSAGE:
{assistant_message}

REMAINING CONVERSATION GOALS:
{goals}

Generate a natural user response that:
1. Reacts appropriately to the assistant's message
2. Advances toward the conversation goals
3. Feels realistic and natural

Respond with only the user's message (no JSON, no explanation).
"""

GOAL_CHECK_PROMPT = """You are evaluating whether a conversation goal has been achieved.

CONVERSATION:
{conversation}

GOAL: "{goal}"

Has this goal been achieved based on the conversation above?
Respond with JSON only:
{{"achieved": true or false, "reasoning": "<one sentence>"}}
"""


class ConversationScenario(BaseModel):
    """Defines a conversation scenario for user simulation.

    Attributes:
        scenario_id: Unique identifier for the scenario.
        description: Description of the overall scenario.
        starting_prompt: Initial user message to start the conversation.
        conversation_plan: High-level plan of conversation flow.
        goals: List of goals to achieve during conversation.
        max_turns: Maximum number of conversation turns.
    """

    scenario_id: str = ""
    description: str = ""
    starting_prompt: str = ""
    conversation_plan: str = ""
    goals: list[str] = Field(default_factory=list)
    max_turns: int = 10
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    """Result of a user simulation session.

    Attributes:
        scenario_id: ID of the scenario that was run.
        turns: Number of conversation turns.
        conversation: Full conversation history.
        goals_achieved: List of goals that were achieved.
        completed: Whether the simulation completed successfully.
        error: Error message if simulation failed.
        criterion_scores: Scores from each evaluation criterion (name -> score 0.0-1.0).
        criterion_details: Detailed output from each criterion (name -> details dict).
        simulator_token_usage: Tokens consumed by simulator LLM calls (user-turn generation).
        criterion_results: Full CriterionResult objects with per-criterion token usage.
        duration_seconds: Wall-clock time of the whole simulation, criteria included.
    """

    scenario_id: str = ""
    turns: int = 0
    conversation: list[dict[str, str]] = Field(default_factory=list)
    goals_achieved: list[str] = Field(default_factory=list)
    completed: bool = False
    error: str | None = None
    criterion_scores: dict[str, float] = Field(default_factory=dict)
    criterion_details: dict[str, Any] = Field(default_factory=dict)
    simulator_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    criterion_results: list[Any] = Field(default_factory=list)  # list[CriterionResult]
    duration_seconds: float = 0.0


class UserSimulator:
    """AI-powered user simulation for testing agents.

    Uses an LLM to generate realistic user messages for testing
    agents with dynamic conversations rather than fixed prompts.
    Optionally runs evaluation criteria (e.g. LLMJudgeCriterion) after
    each simulation to score response quality.

    Attributes:
        model: The LLM model to use for user simulation.
        temperature: Temperature for user message generation.
        max_turns: Maximum conversation turns per scenario.
        criteria: Optional list of BaseCriterion to evaluate the simulation result.

    Example:
        ```python
        from tenxgraph.qa.evaluation import (
            UserSimulator,
            ConversationScenario,
            SimulationGoalsCriterion,
            CriterionConfig,
        )

        judge = SimulationGoalsCriterion(config=CriterionConfig(threshold=0.7))
        simulator = UserSimulator(model="gemini/gemini-2.5-flash", criteria=[judge])

        scenario = ConversationScenario(
            scenario_id="weather_lookup",
            description="User wants to know the weather for travel planning",
            starting_prompt="I'm planning a trip next week",
            conversation_plan="1. Ask about weather\\n2. Ask about packing\\n3. Confirm plans",
            goals=["Get weather info", "Get packing advice"],
            max_turns=6,
        )

        result = await simulator.run(graph, scenario)
        print(f"Completed: {result.completed}")
        print(f"Goals achieved: {result.goals_achieved}")
        print(f"LLM judge score: {result.criterion_scores}")
        ```
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        temperature: float = 0.7,
        max_turns: int = 10,
        config: UserSimulatorConfig | None = None,
        criteria: list[BaseCriterion] | None = None,
        api_style: str = "responses",
    ):
        """Initialize the user simulator.

        Args:
            model: LLM model to use for user simulation.
            temperature: Temperature for message generation.
            max_turns: Default maximum turns per scenario.
            config: Optional configuration override.
            criteria: Optional list of BaseCriterion to run after simulation.
            api_style: OpenAI API style — ``"responses"`` (default) or
                ``"chat"`` for legacy/third-party models.
        """
        if config:
            self.model = config.model
            self.temperature = config.temperature
            self.max_turns = config.max_invocations
            self.api_style: str = getattr(config, "api_style", "responses")
        else:
            self.model = model
            self.temperature = temperature
            self.max_turns = max_turns
            self.api_style = api_style

        self.criteria: list[BaseCriterion] = criteria or []

    async def run(
        self,
        graph: CompiledGraph,
        scenario: ConversationScenario,
        config: dict[str, Any] | None = None,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> SimulationResult:
        """Run a user simulation against an agent and time it.

        See ``_run`` for the arguments; ``duration_seconds`` on the result is the
        wall-clock time of the whole run.
        """
        started = time.perf_counter()
        result = await self._run(graph, scenario, config, on_event=on_event)
        result.duration_seconds = time.perf_counter() - started
        return result

    async def _run(
        self,
        graph: CompiledGraph,
        scenario: ConversationScenario,
        config: dict[str, Any] | None = None,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> SimulationResult:
        """Run a user simulation against an agent.

        Args:
            graph: The compiled agent graph to test.
            scenario: The conversation scenario to simulate.
            config: Optional config to pass to graph execution.
            on_event: Optional progress callback ``(event, data)``. Events:
                ``turn_started``, ``agent_replied``, ``goals_checked``, ``evaluating``
                and ``agent_error``. A callback that raises is ignored.

        Returns:
            SimulationResult with conversation history, outcomes, and criterion scores.
        """
        from tenxgraph.core.state import Message

        conversation: list[dict[str, str]] = []
        goals_achieved: list[str] = []
        sim_tokens = TokenUsage()
        max_turns = scenario.max_turns or self.max_turns

        # Each simulation gets its own thread_id so checkpointer state
        # doesn't bleed between concurrent or sequential runs.
        run_config = dict(config or {})
        run_config.setdefault("thread_id", f"{scenario.scenario_id}-{uuid.uuid4().hex[:8]}")

        try:
            # Start with the initial prompt
            user_message = scenario.starting_prompt
            if not user_message:
                user_message, usage = await self._generate_initial_message(scenario)
                sim_tokens = sim_tokens + usage

            for turn in range(max_turns):
                # Add user message to conversation
                conversation.append({"role": "user", "content": user_message})
                self._emit(
                    on_event,
                    "turn_started",
                    {"turn": turn + 1, "max_turns": max_turns, "user_message": user_message},
                )

                # With a checkpointer the thread already holds the earlier turns (user and
                # agent messages), so only the new user message is sent. Resending the whole
                # history would add every earlier message again under a fresh message_id.
                # Without a checkpointer nothing is kept between calls, so send it all.
                if getattr(graph, "_checkpointer", None) is not None:
                    to_send = [conversation[-1]]
                else:
                    to_send = conversation
                input_data = {
                    "messages": [
                        Message.text_message(msg["content"], role=msg["role"]) for msg in to_send
                    ]
                }

                try:
                    result = await graph.ainvoke(input_data, config=run_config)
                except Exception as e:
                    logger.error("Agent execution failed: %s", e)
                    self._emit(on_event, "agent_error", {"turn": turn + 1, "error": str(e)})
                    (
                        criterion_scores,
                        criterion_details,
                        criterion_results,
                    ) = await self._evaluate_simulation(scenario, conversation, on_event)
                    return SimulationResult(
                        scenario_id=scenario.scenario_id,
                        turns=turn + 1,
                        conversation=conversation,
                        goals_achieved=goals_achieved,
                        completed=False,
                        error=f"Agent error: {e}",
                        criterion_scores=criterion_scores,
                        criterion_details=criterion_details,
                        simulator_token_usage=sim_tokens,
                        criterion_results=criterion_results,
                    )

                # Extract agent response
                agent_response = self._extract_response(result)
                conversation.append({"role": "assistant", "content": agent_response})
                self._emit(
                    on_event, "agent_replied", {"turn": turn + 1, "response": agent_response}
                )

                # Check for goal completion using LLM
                achieved, check_usage = await self._check_goals(
                    scenario.goals,
                    goals_achieved,
                    conversation,
                )
                sim_tokens = sim_tokens + check_usage
                goals_achieved.extend(achieved)
                self._emit(
                    on_event,
                    "goals_checked",
                    {
                        "turn": turn + 1,
                        "newly_achieved": list(achieved),
                        "achieved": len(goals_achieved),
                        "total": len(scenario.goals),
                    },
                )

                # Check if we're done
                if len(goals_achieved) >= len(scenario.goals):
                    (
                        criterion_scores,
                        criterion_details,
                        criterion_results,
                    ) = await self._evaluate_simulation(scenario, conversation, on_event)
                    return SimulationResult(
                        scenario_id=scenario.scenario_id,
                        turns=turn + 1,
                        conversation=conversation,
                        goals_achieved=goals_achieved,
                        completed=True,
                        criterion_scores=criterion_scores,
                        criterion_details=criterion_details,
                        simulator_token_usage=sim_tokens,
                        criterion_results=criterion_results,
                    )

                # Generate next user message
                remaining_goals = [g for g in scenario.goals if g not in goals_achieved]
                user_message, resp_usage = await self._generate_response(
                    scenario=scenario,
                    conversation=conversation,
                    remaining_goals=remaining_goals,
                )
                sim_tokens = sim_tokens + resp_usage

            # Max turns reached
            (
                criterion_scores,
                criterion_details,
                criterion_results,
            ) = await self._evaluate_simulation(scenario, conversation, on_event)
            return SimulationResult(
                scenario_id=scenario.scenario_id,
                turns=max_turns,
                conversation=conversation,
                goals_achieved=goals_achieved,
                completed=len(goals_achieved) >= len(scenario.goals),
                criterion_scores=criterion_scores,
                criterion_details=criterion_details,
                simulator_token_usage=sim_tokens,
                criterion_results=criterion_results,
            )

        except Exception as e:
            logger.error("Simulation failed: %s", e)
            # Attempt criterion evaluation even on failure
            criterion_scores: dict[str, float] = {}
            criterion_details: dict[str, Any] = {}
            criterion_results: list[Any] = []
            try:
                (
                    criterion_scores,
                    criterion_details,
                    criterion_results,
                ) = await self._evaluate_simulation(scenario, conversation, on_event)
            except Exception as eval_err:
                logger.warning("Criterion evaluation also failed: %s", eval_err)
            return SimulationResult(
                scenario_id=scenario.scenario_id,
                turns=len(conversation) // 2,
                conversation=conversation,
                goals_achieved=goals_achieved,
                completed=False,
                error=str(e),
                criterion_scores=criterion_scores,
                criterion_details=criterion_details,
                simulator_token_usage=sim_tokens,
                criterion_results=criterion_results,
            )

    @staticmethod
    def _emit(
        on_event: Callable[[str, dict[str, Any]], None] | None,
        event: str,
        data: dict[str, Any],
    ) -> None:
        """Call the progress callback; a failing callback never breaks the run."""
        if on_event is None:
            return
        try:
            on_event(event, data)
        except Exception as exc:
            logger.debug("on_event callback failed for %s: %s", event, exc)

    async def _generate_initial_message(
        self,
        scenario: ConversationScenario,
    ) -> tuple[str, TokenUsage]:
        """Generate the initial user message."""
        prompt = USER_SIMULATOR_PROMPT.format(
            scenario=scenario.description,
            conversation_plan=scenario.conversation_plan,
            conversation_history="(This is the start of the conversation)",
        )

        return await self._call_llm(prompt)

    async def _generate_response(
        self,
        scenario: ConversationScenario,
        conversation: list[dict[str, str]],
        remaining_goals: list[str],
    ) -> tuple[str, TokenUsage]:
        """Generate the next user message."""
        # Get last assistant message
        last_assistant = ""
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break

        prompt = USER_RESPONSE_PROMPT.format(
            scenario=scenario.description,
            assistant_message=last_assistant,
            goals="\n".join(f"- {g}" for g in remaining_goals),
        )

        return await self._call_llm(prompt)

    async def _check_goals(
        self,
        all_goals: list[str],
        achieved: list[str],
        conversation: list[dict[str, str]],
    ) -> tuple[list[str], TokenUsage]:
        """Check which goals have been newly achieved using LLM-based evaluation.

        Falls back to keyword matching if the LLM call fails.
        """
        remaining = [g for g in all_goals if g not in achieved]
        newly_achieved = []
        total_usage = TokenUsage()

        conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)

        async def _check_one(goal: str) -> tuple[bool, TokenUsage]:
            prompt = GOAL_CHECK_PROMPT.format(conversation=conv_text, goal=goal)
            try:
                raw, usage = await self._call_llm(prompt)
                # Strip markdown code fences that some LLMs wrap JSON in
                clean = raw.strip()
                if clean.startswith("```"):
                    lines = clean.splitlines()
                    _min_fenced_lines = 2
                    clean = "\n".join(lines[1:-1]) if len(lines) > _min_fenced_lines else clean
                data = json.loads(clean)
                return bool(data.get("achieved")), usage
            except Exception:
                # Fallback to keyword matching if LLM call or JSON parse fails
                _min_word_len = 3
                words = [w for w in goal.lower().split() if len(w) > _min_word_len]
                text = " ".join(m["content"].lower() for m in conversation)
                return bool(words and all(w in text for w in words)), TokenUsage()

        outcomes = await asyncio.gather(*[_check_one(g) for g in remaining])
        for goal, (hit, usage) in zip(remaining, outcomes, strict=True):
            total_usage = total_usage + usage
            if hit:
                newly_achieved.append(goal)

        return newly_achieved, total_usage

    async def _evaluate_simulation(
        self,
        scenario: ConversationScenario,
        conversation: list[dict[str, str]],
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, float], dict[str, Any], list[Any]]:
        """Run configured criteria against the completed simulation.

        Builds a minimal ExecutionResult and EvalCase from the simulation
        data, then calls each criterion's evaluate() method.
        The full conversation transcript is passed as actual_response so
        the LLM judge evaluates goal achievement across all turns.

        Args:
            scenario: The scenario that was simulated.
            conversation: Full conversation history.

        Returns:
            Tuple of (criterion_scores, criterion_details, criterion_results).
        """
        if not self.criteria or not conversation:
            return {}, {}, []

        from tenxgraph.qa.evaluation.dataset.eval_set import (
            EvalCase,
            Invocation,
            MessageContent,
        )
        from tenxgraph.qa.evaluation.execution.result import ExecutionResult

        # Build ExecutionResult — pass the full conversation so the LLM judge
        # evaluates goal achievement across all turns, not just the last message.
        execution = ExecutionResult()
        execution.actual_response = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in conversation
        )

        # Build EvalCase — use scenario goals as the expected success description
        goals_text = "; ".join(scenario.goals) if scenario.goals else ""
        case = EvalCase(
            eval_id=scenario.scenario_id,
            name=scenario.description,
            conversation=[
                Invocation(
                    user_content=MessageContent(
                        role="user",
                        content=scenario.starting_prompt or scenario.description,
                    ),
                    expected_final_response=MessageContent(
                        role="assistant",
                        content=goals_text,
                    )
                    if goals_text
                    else None,
                )
            ],
        )

        scores: dict[str, float] = {}
        details: dict[str, Any] = {}
        results: list[Any] = []

        self._emit(on_event, "evaluating", {"criteria": [c.name for c in self.criteria]})

        async def _run_one(criterion: BaseCriterion) -> Any:
            try:
                return await criterion.evaluate(execution, case)
            except Exception as e:
                logger.warning("Criterion %s failed: %s", criterion.name, e)
                return e

        outcomes = await asyncio.gather(*[_run_one(c) for c in self.criteria])
        for criterion, outcome in zip(self.criteria, outcomes, strict=True):
            if isinstance(outcome, Exception):
                scores[criterion.name] = 0.0
                details[criterion.name] = {"error": str(outcome)}
            else:
                scores[criterion.name] = outcome.score
                details[criterion.name] = outcome.details or {}
                results.append(outcome)

        return scores, details, results

    async def _call_llm(self, prompt: str) -> tuple[str, TokenUsage]:
        """Call the LLM for user simulation. Returns (text, TokenUsage)."""
        try:
            text, inp, out, cache = await call_llm(
                self.model,
                prompt,
                temperature=self.temperature,
                api_style=self.api_style,  # type: ignore[arg-type]
            )
        except Exception as exc:
            logger.warning("LLM call failed (%s): %s", type(exc).__name__, exc)
            return "I have a follow-up question.", TokenUsage()

        if not text:
            return "I have a follow-up question.", TokenUsage()

        return text, TokenUsage(input_tokens=inp, output_tokens=out, cache_read_tokens=cache)

    def _extract_response(self, result: dict[str, Any]) -> str:
        """Extract text response from graph result.

        Handles agentflow Message objects (content = list[ContentBlock])
        and plain dicts with a string content field.
        """
        if not result:
            return ""

        messages = result.get("messages", [])
        if not messages:
            return ""

        for msg in reversed(messages):
            # agentflow Message object: role attr + content is list[ContentBlock]
            if hasattr(msg, "role") and msg.role == "assistant":
                if hasattr(msg, "content") and isinstance(msg.content, list):
                    texts = [
                        block.text
                        for block in msg.content
                        if hasattr(block, "text") and isinstance(block.text, str) and block.text
                    ]
                    if texts:
                        return " ".join(texts)
                continue

            # Plain dict format
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    return content

        return ""


class BatchSimulator:
    """Run multiple simulation scenarios concurrently.

    Scenarios are executed in parallel using asyncio.gather, matching the
    industry pattern used by DeepEval and Promptfoo. Each scenario gets its
    own isolated thread_id so checkpointer state never bleeds between runs.
    """

    def __init__(
        self,
        simulator: UserSimulator | None = None,
        max_concurrency: int = 5,
        **kwargs,
    ):
        """Initialize batch simulator.

        Args:
            simulator: Optional pre-configured UserSimulator.
            max_concurrency: Maximum number of scenarios to run in parallel.
            **kwargs: Arguments to pass to UserSimulator if not provided.
        """
        self.simulator = simulator or UserSimulator(**kwargs)
        self.max_concurrency = max_concurrency

    async def run_batch(
        self,
        graph: CompiledGraph,
        scenarios: list[ConversationScenario],
        config: dict[str, Any] | None = None,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> list[SimulationResult]:
        """Run multiple scenarios concurrently.

        Args:
            graph: The compiled agent graph to test.
            scenarios: List of scenarios to run.
            config: Optional base config passed to each graph execution.
            on_event: Optional progress callback shared by all scenarios (see
                ``UserSimulator.run``). Events are not tagged with a scenario id.

        Returns:
            List of simulation results in the same order as scenarios.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _run_one(scenario: ConversationScenario) -> SimulationResult:
            async with semaphore:
                logger.info("Running scenario: %s", scenario.scenario_id)
                result = await self.simulator.run(graph, scenario, config, on_event=on_event)
                status = "completed" if result.completed else "incomplete"
                logger.info(
                    "Scenario %s %s after %d turns (%d/%d goals)",
                    scenario.scenario_id,
                    status,
                    result.turns,
                    len(result.goals_achieved),
                    len(scenario.goals),
                )
                return result

        return list(await asyncio.gather(*[_run_one(s) for s in scenarios]))

    def summary(self, results: list[SimulationResult]) -> dict[str, Any]:
        """Generate summary statistics for batch results."""
        total = len(results)
        completed = sum(1 for r in results if r.completed)
        total_goals = sum(len(r.goals_achieved) for r in results)
        total_turns = sum(r.turns for r in results)
        errors = sum(1 for r in results if r.error)

        return {
            "total_scenarios": total,
            "completed": completed,
            "completion_rate": completed / total if total > 0 else 0.0,
            "total_goals_achieved": total_goals,
            "average_turns": total_turns / total if total > 0 else 0,
            "errors": errors,
        }
