"""Tests for QuickEval simplified evaluation interface.

Mocks AgentEvaluator.evaluate() to avoid running actual agent graphs.
"""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from tenxgraph.qa.evaluation.config.presets import EvalPresets
from tenxgraph.qa.evaluation.dataset.builder import EvalSetBuilder
from tenxgraph.qa.evaluation.dataset.eval_set import EvalSet
from tenxgraph.qa.evaluation.eval_result import EvalReport, EvalSummary
from tenxgraph.qa.evaluation.quick_eval import QuickEval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_report(name: str = "mock_set") -> EvalReport:
    return EvalReport(
        eval_set_id="mock-id",
        eval_set_name=name,
        results=[],
        summary=EvalSummary(total_cases=1, passed_cases=1, pass_rate=1.0),
    )


@pytest.fixture
def mock_graph():
    return MagicMock()


@pytest.fixture
def mock_collector():
    return MagicMock()


@pytest.fixture
def mock_evaluator_class():
    """Patch AgentEvaluator so it always returns _mock_report."""
    report = _mock_report()

    mock_evaluator_instance = MagicMock()
    mock_evaluator_instance.evaluate = AsyncMock(return_value=report)

    with patch(
        "tenxgraph.qa.evaluation.quick_eval.AgentEvaluator",
        return_value=mock_evaluator_instance,
    ) as mock_cls:
        mock_cls.return_value = mock_evaluator_instance
        yield mock_evaluator_instance, report


# ---------------------------------------------------------------------------
# QuickEval.check
# ---------------------------------------------------------------------------

class TestQuickEvalCheck:
    @pytest.mark.asyncio
    async def test_check_returns_report(self, mock_graph, mock_collector, mock_evaluator_class):
        mock_ev, expected_report = mock_evaluator_class
        report = await QuickEval.check(
            graph=mock_graph,
            collector=mock_collector,
            query="What is 2+2?",
            expected_response_contains="4",
            print_results=False,
        )
        assert report is expected_report
        mock_ev.evaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_with_expected_response_equals(
        self, mock_graph, mock_collector, mock_evaluator_class
    ):
        mock_ev, _ = mock_evaluator_class
        report = await QuickEval.check(
            graph=mock_graph,
            collector=mock_collector,
            query="Hello",
            expected_response_equals="Hi",
            print_results=False,
        )
        assert report is not None

    @pytest.mark.asyncio
    async def test_check_with_expected_tools(
        self, mock_graph, mock_collector, mock_evaluator_class
    ):
        mock_ev, _ = mock_evaluator_class
        report = await QuickEval.check(
            graph=mock_graph,
            collector=mock_collector,
            query="Weather in Paris?",
            expected_tools=["get_weather"],
            print_results=False,
        )
        assert report is not None

    @pytest.mark.asyncio
    async def test_check_no_expected(self, mock_graph, mock_collector, mock_evaluator_class):
        """check() with no expected_response_* uses fallback 'response'."""
        mock_ev, _ = mock_evaluator_class
        report = await QuickEval.check(
            graph=mock_graph,
            collector=mock_collector,
            query="Hello",
            print_results=False,
        )
        assert report is not None

    @pytest.mark.asyncio
    async def test_check_print_results_calls_print(
        self, mock_graph, mock_collector, mock_evaluator_class
    ):
        with patch("tenxgraph.qa.evaluation.quick_eval.print_report") as mock_print:
            await QuickEval.check(
                graph=mock_graph,
                collector=mock_collector,
                query="test",
                print_results=True,
            )
        mock_print.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_verbose_false(self, mock_graph, mock_collector, mock_evaluator_class):
        mock_ev, _ = mock_evaluator_class
        await QuickEval.check(
            graph=mock_graph,
            collector=mock_collector,
            query="test",
            verbose=False,
            print_results=False,
        )
        mock_ev.evaluate.assert_called_once_with(
            ANY, verbose=False
        )


# ---------------------------------------------------------------------------
# QuickEval.preset
# ---------------------------------------------------------------------------

class TestQuickEvalPreset:
    @pytest.mark.asyncio
    async def test_preset_with_eval_set(
        self, mock_graph, mock_collector, mock_evaluator_class
    ):
        mock_ev, expected_report = mock_evaluator_class
        eval_set = EvalSetBuilder.quick(("q", "r"))
        report = await QuickEval.preset(
            graph=mock_graph,
            collector=mock_collector,
            preset=EvalPresets.quick_check(),
            eval_set=eval_set,
            print_results=False,
        )
        assert report is expected_report

    @pytest.mark.asyncio
    async def test_preset_print_results(self, mock_graph, mock_collector, mock_evaluator_class):
        eval_set = EvalSetBuilder.quick(("q", "r"))
        with patch("tenxgraph.qa.evaluation.quick_eval.print_report") as mock_print:
            await QuickEval.preset(
                graph=mock_graph,
                collector=mock_collector,
                preset=EvalPresets.quick_check(),
                eval_set=eval_set,
                print_results=True,
            )
        mock_print.assert_called_once()


# ---------------------------------------------------------------------------
# QuickEval.batch
# ---------------------------------------------------------------------------

class TestQuickEvalBatch:
    @pytest.mark.asyncio
    async def test_batch_returns_report(self, mock_graph, mock_collector, mock_evaluator_class):
        mock_ev, expected_report = mock_evaluator_class
        report = await QuickEval.batch(
            graph=mock_graph,
            collector=mock_collector,
            test_pairs=[("Hello", "Hi"), ("Bye", "Goodbye")],
            print_results=False,
        )
        assert report is expected_report

    @pytest.mark.asyncio
    async def test_batch_custom_threshold(
        self, mock_graph, mock_collector, mock_evaluator_class
    ):
        mock_ev, _ = mock_evaluator_class
        await QuickEval.batch(
            graph=mock_graph,
            collector=mock_collector,
            test_pairs=[("q", "r")],
            threshold=0.9,
            print_results=False,
        )
        mock_ev.evaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_print_results(self, mock_graph, mock_collector, mock_evaluator_class):
        with patch("tenxgraph.qa.evaluation.quick_eval.print_report") as mock_print:
            await QuickEval.batch(
                graph=mock_graph,
                collector=mock_collector,
                test_pairs=[("q", "r")],
                print_results=True,
            )
        mock_print.assert_called_once()


# ---------------------------------------------------------------------------
# QuickEval.tool_usage
# ---------------------------------------------------------------------------

class TestQuickEvalToolUsage:
    @pytest.mark.asyncio
    async def test_tool_usage_returns_report(
        self, mock_graph, mock_collector, mock_evaluator_class
    ):
        mock_ev, expected_report = mock_evaluator_class
        report = await QuickEval.tool_usage(
            graph=mock_graph,
            collector=mock_collector,
            test_cases=[("query", "response", ["tool_a"])],
            print_results=False,
        )
        assert report is expected_report

    @pytest.mark.asyncio
    async def test_tool_usage_strict_false(
        self, mock_graph, mock_collector, mock_evaluator_class
    ):
        await QuickEval.tool_usage(
            graph=mock_graph,
            collector=mock_collector,
            test_cases=[("q", "r", ["t"])],
            strict=False,
            print_results=False,
        )


# ---------------------------------------------------------------------------
# QuickEval.conversation_flow
# ---------------------------------------------------------------------------

class TestQuickEvalConversationFlow:
    @pytest.mark.asyncio
    async def test_conversation_flow(self, mock_graph, mock_collector, mock_evaluator_class):
        mock_ev, expected_report = mock_evaluator_class
        conversation = [("Hi", "Hello"), ("How are you?", "Fine")]
        report = await QuickEval.conversation_flow(
            graph=mock_graph,
            collector=mock_collector,
            conversation=conversation,
            print_results=False,
        )
        assert report is expected_report


# ---------------------------------------------------------------------------
# QuickEval.from_builder
# ---------------------------------------------------------------------------

class TestQuickEvalFromBuilder:
    @pytest.mark.asyncio
    async def test_from_builder_default_config(
        self, mock_graph, mock_collector, mock_evaluator_class
    ):
        mock_ev, expected_report = mock_evaluator_class
        builder = EvalSetBuilder("test_set")
        builder.add_case("q", "r")
        report = await QuickEval.from_builder(
            graph=mock_graph,
            collector=mock_collector,
            builder=builder,
            print_results=False,
        )
        assert report is expected_report

    @pytest.mark.asyncio
    async def test_from_builder_custom_config(
        self, mock_graph, mock_collector, mock_evaluator_class
    ):
        builder = EvalSetBuilder("test_set")
        builder.add_case("q", "r")
        report = await QuickEval.from_builder(
            graph=mock_graph,
            collector=mock_collector,
            builder=builder,
            config=EvalPresets.comprehensive(),
            print_results=False,
        )
        assert report is not None


# ---------------------------------------------------------------------------
# QuickEval.run_sync
# ---------------------------------------------------------------------------

class TestQuickEvalRunSync:
    def test_run_sync(self, mock_graph, mock_collector):
        """run_sync wraps evaluate() in asyncio.run()."""
        mock_report = _mock_report("sync_test")

        mock_evaluator_instance = MagicMock()
        mock_evaluator_instance.evaluate = AsyncMock(return_value=mock_report)

        eval_set = EvalSetBuilder.quick(("q", "r"))

        with patch(
            "tenxgraph.qa.evaluation.quick_eval.AgentEvaluator",
            return_value=mock_evaluator_instance,
        ):
            report = QuickEval.run_sync(
                graph=mock_graph,
                collector=mock_collector,
                eval_set=eval_set,
                print_results=False,
            )

        assert report is mock_report

    def test_run_sync_print_results(self, mock_graph, mock_collector):
        mock_report = _mock_report()
        mock_evaluator_instance = MagicMock()
        mock_evaluator_instance.evaluate = AsyncMock(return_value=mock_report)
        eval_set = EvalSetBuilder.quick(("q", "r"))

        with (
            patch(
                "tenxgraph.qa.evaluation.quick_eval.AgentEvaluator",
                return_value=mock_evaluator_instance,
            ),
            patch("tenxgraph.qa.evaluation.quick_eval.print_report") as mock_print,
        ):
            QuickEval.run_sync(
                graph=mock_graph,
                collector=mock_collector,
                eval_set=eval_set,
                print_results=True,
            )
        mock_print.assert_called_once()
