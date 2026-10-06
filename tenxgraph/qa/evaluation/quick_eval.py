"""QuickEval - Simplified evaluation interface for Agentflow agents.

Reduces evaluation setup from ~50 lines to ~5 lines with presets and builders.
"""

import logging
from typing import TYPE_CHECKING

from agentflow.qa.evaluation.collectors.trajectory_collector import TrajectoryCollector
from agentflow.qa.evaluation.config.eval_config import EvalConfig
from agentflow.qa.evaluation.config.presets import EvalPresets
from agentflow.qa.evaluation.dataset.builder import EvalSetBuilder
from agentflow.qa.evaluation.dataset.eval_set import EvalCase, EvalSet, ToolCall
from agentflow.qa.evaluation.eval_result import EvalReport
from agentflow.qa.evaluation.evaluator import AgentEvaluator
from agentflow.qa.evaluation.reporters.console import print_report


if TYPE_CHECKING:
    from agentflow.core.graph.compiled_graph import CompiledGraph

logger = logging.getLogger("agentflow.evaluation")


class QuickEval:
    """Simplified evaluation interface with presets and one-liners.

    Provides easy methods for common evaluation patterns:
    - Quick single-test checks
    - Preset configurations
    - Batch evaluations
    - Auto-generated eval sets

    Example:
        ```python
        # Quick single check
        report = await QuickEval.check(
            graph=compiled_graph,
            collector=collector,
            query="Hello",
            expected_response_contains="Hi",
        )

        # Use preset
        report = await QuickEval.preset(
            graph=compiled_graph,
            collector=collector,
            preset=EvalPresets.tool_usage(),
            eval_set="tests/tool_tests.json",
        )

        # Batch tests
        reports = await QuickEval.batch(
            graph,
            collector,
            [
                ("Hello", "Hi"),
                ("Weather?", "Sunny"),
            ],
        )
        ```
    """

    @classmethod
    async def check(
        cls,
        graph: "CompiledGraph",
        collector: TrajectoryCollector,
        query: str,
        expected_response_contains: str | None = None,
        expected_response_equals: str | None = None,
        expected_tools: list[str] | None = None,
        threshold: float = 0.7,
        verbose: bool = True,
        print_results: bool = True,
    ) -> EvalReport:
        """Quick single-test check.

        Args:
            graph: Compiled agent graph to test
            query: User query to test
            expected_response_contains: Text that should be in response
            expected_response_equals: Exact expected response
            expected_tools: List of tool names that should be called
            threshold: Pass threshold
            verbose: Whether to log progress
            print_results: Whether to print report to console

        Returns:
            EvalReport with results
        """
        # Build eval case
        expected = expected_response_equals or expected_response_contains or "response"

        tool_calls = None
        if expected_tools:
            tool_calls = [ToolCall(name=name, args={}) for name in expected_tools]

        case = EvalCase.single_turn(
            eval_id="quick_check",
            user_query=query,
            expected_response=expected,
            expected_tools=tool_calls,
        )

        eval_set = EvalSet(
            eval_set_id="quick_check",
            name="Quick Check",
            eval_cases=[case],
        )

        # Choose config based on what's being tested
        if expected_tools:
            config = EvalPresets.custom(
                response_threshold=threshold,
                tool_threshold=1.0,
            )
        else:
            config = EvalPresets.quick_check()

        # Run evaluation
        evaluator = AgentEvaluator(graph, collector, config=config)
        report = await evaluator.evaluate(eval_set, verbose=verbose)

        if print_results:
            print_report(report)

        return report

    @classmethod
    async def preset(
        cls,
        graph: "CompiledGraph",
        collector: TrajectoryCollector,
        preset: EvalConfig,
        eval_set: EvalSet | str,
        verbose: bool = True,
        print_results: bool = True,
    ) -> EvalReport:
        """Evaluate using a preset configuration.

        Args:
            graph: Compiled agent graph
            preset: EvalConfig from EvalPresets
            eval_set: EvalSet object or path to JSON file
            verbose: Whether to log progress
            print_results: Whether to print report

        Returns:
            EvalReport
        """
        evaluator = AgentEvaluator(graph, collector, config=preset)
        report = await evaluator.evaluate(eval_set, verbose=verbose)

        if print_results:
            print_report(report)

        return report

    @classmethod
    async def batch(
        cls,
        graph: "CompiledGraph",
        collector: TrajectoryCollector,
        test_pairs: list[tuple[str, str]],
        threshold: float = 0.7,
        verbose: bool = True,
        print_results: bool = True,
    ) -> EvalReport:
        """Run batch evaluation from query-response pairs.

        Args:
            graph: Compiled agent graph
            test_pairs: List of (query, expected_response) tuples
            threshold: Pass threshold
            verbose: Whether to log progress
            print_results: Whether to print report

        Returns:
            EvalReport
        """
        # Build eval set from pairs
        eval_set = EvalSetBuilder.quick(*test_pairs)

        config = EvalPresets.quick_check()
        config.criteria.rouge_match.threshold = threshold

        evaluator = AgentEvaluator(graph, collector, config=config)
        report = await evaluator.evaluate(eval_set, verbose=verbose)

        if print_results:
            print_report(report)

        return report

    @classmethod
    async def tool_usage(
        cls,
        graph: "CompiledGraph",
        collector: TrajectoryCollector,
        test_cases: list[tuple[str, str, list[str]]],
        strict: bool = True,
        verbose: bool = True,
        print_results: bool = True,
    ) -> EvalReport:
        """Evaluate tool usage specifically.

        Args:
            graph: Compiled agent graph
            test_cases: List of (query, expected_response, expected_tools) tuples
            strict: Whether to require exact tool matches
            verbose: Whether to log progress
            print_results: Whether to print report

        Returns:
            EvalReport
        """
        builder = EvalSetBuilder("tool_usage_tests")

        for i, (query, expected, tools) in enumerate(test_cases):
            builder.add_case(
                query=query,
                expected=expected,
                expected_tools=tools,
                case_id=f"tool_test_{i + 1}",
            )

        eval_set = builder.build()
        config = EvalPresets.tool_usage(strict=strict)

        evaluator = AgentEvaluator(graph, collector, config=config)
        report = await evaluator.evaluate(eval_set, verbose=verbose)

        if print_results:
            print_report(report)

        return report

    @classmethod
    async def conversation_flow(
        cls,
        graph: "CompiledGraph",
        collector: TrajectoryCollector,
        conversation: list[tuple[str, str]],
        threshold: float = 0.8,
        verbose: bool = True,
        print_results: bool = True,
    ) -> EvalReport:
        """Evaluate multi-turn conversation flow.

        Args:
            graph: Compiled agent graph
            conversation: List of (user_query, expected_response) tuples
            threshold: Pass threshold
            verbose: Whether to log progress
            print_results: Whether to print report

        Returns:
            EvalReport
        """
        case = EvalCase.multi_turn(
            eval_id="conversation_flow",
            conversation=conversation,
        )

        eval_set = EvalSet(
            eval_set_id="conversation_test",
            name="Conversation Flow Test",
            eval_cases=[case],
        )

        config = EvalPresets.conversation_flow(threshold=threshold)

        evaluator = AgentEvaluator(graph, collector, config=config)
        report = await evaluator.evaluate(eval_set, verbose=verbose)

        if print_results:
            print_report(report)

        return report

    @classmethod
    async def from_builder(
        cls,
        graph: "CompiledGraph",
        collector: TrajectoryCollector,
        builder: EvalSetBuilder,
        config: EvalConfig | None = None,
        verbose: bool = True,
        print_results: bool = True,
    ) -> EvalReport:
        """Evaluate using an EvalSetBuilder.

        Args:
            graph: Compiled agent graph
            builder: EvalSetBuilder with test cases
            config: Optional custom config (uses default if not provided)
            verbose: Whether to log progress
            print_results: Whether to print report

        Returns:
            EvalReport
        """
        eval_set = builder.build()
        config = config or EvalPresets.quick_check()

        evaluator = AgentEvaluator(graph, collector, config=config)
        report = await evaluator.evaluate(eval_set, verbose=verbose)

        if print_results:
            print_report(report)

        return report

    @classmethod
    def run_sync(
        cls,
        graph: "CompiledGraph",
        collector: TrajectoryCollector,
        eval_set: EvalSet | str,
        config: EvalConfig | None = None,
        verbose: bool = True,
        print_results: bool = True,
    ) -> EvalReport:
        """Synchronous wrapper for evaluation.

        Args:
            graph: Compiled agent graph
            eval_set: EvalSet object or path to JSON
            config: Optional config
            verbose: Whether to log progress
            print_results: Whether to print report

        Returns:
            EvalReport
        """
        import asyncio

        config = config or EvalPresets.quick_check()
        evaluator = AgentEvaluator(graph, collector, config=config)

        report = asyncio.run(evaluator.evaluate(eval_set, verbose=verbose))

        if print_results:
            print_report(report)

        return report
