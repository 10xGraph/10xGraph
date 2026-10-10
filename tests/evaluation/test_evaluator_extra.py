from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import json

import pytest

from tenxgraph.qa.evaluation.collectors.trajectory_collector import TrajectoryCollector
from tenxgraph.qa.evaluation.config.eval_config import EvalConfig
from tenxgraph.qa.evaluation.dataset.eval_set import EvalCase, EvalSet
from tenxgraph.qa.evaluation.eval_result import CriterionResult, EvalCaseResult
from tenxgraph.qa.evaluation.evaluator import AgentEvaluator, EvaluationRunner


@pytest.mark.asyncio
async def test_evaluate_parallel_handles_exception_items():
    evaluator = AgentEvaluator(MagicMock(), TrajectoryCollector(), config=EvalConfig.default())

    cases = [
        EvalCase.single_turn("c1", "hello", "hi"),
        EvalCase.single_turn("c2", "hello", "hi"),
    ]

    async def fake_run(case, collector_override=None):
        if case.eval_id == "c2":
            raise RuntimeError("boom")
        return MagicMock(passed=True, eval_id=case.eval_id, name=case.name, duration_seconds=0.1)

    evaluator._evaluate_case = fake_run
    results = await evaluator._evaluate_parallel(cases, max_concurrency=2, verbose=False)

    assert len(results) == 2
    assert any(r.eval_id == "c2" and r.error for r in results)


@pytest.mark.asyncio
async def test_evaluate_routes_parallel_and_runs_reporters_with_output_dir():
    evaluator = AgentEvaluator(MagicMock(), TrajectoryCollector(), config=EvalConfig.default())
    evaluator.config.reporter.enabled = True
    cases = [EvalCase.single_turn("c1", "u", "a"), EvalCase.single_turn("c2", "u", "a")]
    eval_set = EvalSet(eval_set_id="s1", eval_cases=cases)

    with patch.object(evaluator, "_evaluate_parallel", new_callable=AsyncMock) as p, patch.object(
        evaluator, "_run_reporters"
    ) as run_rep:
        p.return_value = [
            EvalCaseResult.success(eval_id="c1", criterion_results=[]),
            EvalCaseResult.success(eval_id="c2", criterion_results=[]),
        ]
        report = await evaluator.evaluate(eval_set, parallel=True, max_concurrency=2, output_dir="out")

    assert report.eval_set_id == "s1"
    run_rep.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_loads_eval_set_from_file_path(tmp_path):
    eval_set = EvalSet(eval_set_id="s-file", eval_cases=[EvalCase.single_turn("c1", "u", "a")])
    path = tmp_path / "s.evalset.json"
    eval_set.to_file(str(path))

    evaluator = AgentEvaluator(MagicMock(), TrajectoryCollector(), config=EvalConfig.default())
    with patch.object(evaluator, "_evaluate_sequential", new_callable=AsyncMock) as seq:
        seq.return_value = [EvalCaseResult.success(eval_id="c1", criterion_results=[])]
        report = await evaluator.evaluate(str(path), parallel=False)

    assert report.eval_set_id == "s-file"


def test_run_reporters_handles_pipeline_exception():
    evaluator = AgentEvaluator(MagicMock(), TrajectoryCollector(), config=EvalConfig.default())
    report = MagicMock()

    with patch("tenxgraph.qa.evaluation.reporters.manager.ReporterManager", side_effect=RuntimeError("x")):
        evaluator._run_reporters(report)


def test_load_graph_prefers_compile_if_available():
    class _Graph:
        def compile(self):
            return "compiled"

    module = SimpleNamespace(graph=_Graph())
    with patch("importlib.import_module", return_value=module):
        loaded = AgentEvaluator._load_graph("x.module")
    assert loaded == "compiled"


def test_load_graph_raises_when_no_known_attribute():
    module = SimpleNamespace(other=1)
    with patch("importlib.import_module", return_value=module):
        with pytest.raises(AttributeError):
            AgentEvaluator._load_graph("x.module")


def test_load_eval_set_raises_when_missing_file():
    evaluator = AgentEvaluator(MagicMock(), TrajectoryCollector(), config=EvalConfig.default())
    with pytest.raises(FileNotFoundError):
        evaluator._load_eval_set("/tmp/definitely-missing.evalset.json")


@pytest.mark.asyncio
async def test_evaluate_file_uses_loaded_graph_and_config(tmp_path):
    eval_set = EvalSet(
        eval_set_id="s1",
        eval_cases=[EvalCase.single_turn("c1", "hello", "hi")],
    )
    eval_path = tmp_path / "set.json"
    eval_set.to_file(str(eval_path))

    fake_compiled = MagicMock()
    fake_compiled.compile = MagicMock(return_value=fake_compiled)

    with patch.object(AgentEvaluator, "_load_graph", return_value=fake_compiled), patch.object(
        AgentEvaluator, "evaluate", new_callable=AsyncMock
    ) as mock_eval:
        mock_eval.return_value = MagicMock(eval_set_id="s1")
        out = await AgentEvaluator.evaluate_file("x.module", str(eval_path))

    assert out.eval_set_id == "s1"


@pytest.mark.asyncio
async def test_runner_run_reporters_is_best_effort():
    runner = EvaluationRunner()
    runner.results = {"s1": MagicMock(), "s2": MagicMock()}
    cfg = EvalConfig.default()

    with patch("tenxgraph.qa.evaluation.reporters.manager.ReporterManager") as mgr_cls:
        mgr = MagicMock()
        mgr.run_all.side_effect = RuntimeError("boom")
        mgr_cls.return_value = mgr
        runner._run_reporters(cfg)


@pytest.mark.asyncio
async def test_run_conversation_turns_returns_failure_on_graph_exception():
    collector = TrajectoryCollector()
    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph crash"))
    evaluator = AgentEvaluator(MagicMock(), collector, config=EvalConfig.default())
    case = EvalCase.single_turn("c-error", "u", "a")

    out = await evaluator._run_conversation_turns(case, collector, graph, {"thread_id": "t"}, 0.0)
    assert hasattr(out, "passed")
    assert out.passed is False


def test_runner_summary_with_results():
    runner = EvaluationRunner()
    r1 = MagicMock()
    r1.summary.total_cases = 3
    r1.summary.passed_cases = 2
    r2 = MagicMock()
    r2.summary.total_cases = 1
    r2.summary.passed_cases = 1
    runner.results = {"a": r1, "b": r2}

    summary = runner.summary
    assert summary["total_evaluations"] == 2
    assert summary["total_cases"] == 4
    assert summary["passed_cases"] == 3


@pytest.mark.asyncio
async def test_run_conversation_turns_collects_multi_turn_data():
    collector = TrajectoryCollector()
    graph = MagicMock()
    graph.ainvoke = AsyncMock()
    evaluator = AgentEvaluator(MagicMock(), collector, config=EvalConfig.default())

    case = EvalCase(
        eval_id="c1",
        conversation=[
            EvalCase.single_turn("tmp", "u1", "a1").conversation[0],
            EvalCase.single_turn("tmp2", "u2", "a2").conversation[0],
        ],
    )

    async def _invoke_side_effect(payload, config=None):
        user_msg = payload["messages"][-1].text()
        collector.final_response = f"resp:{user_msg}"
        collector.tool_calls = []
        collector.trajectory = []
        collector.node_visits = ["MAIN"]
        collector.node_responses = []
        collector.start_time = 1.0
        collector.end_time = 2.0

    graph.ainvoke.side_effect = _invoke_side_effect

    out = await evaluator._run_conversation_turns(case, collector, graph, {"thread_id": "t"}, 0.0)
    assert isinstance(out, tuple)
    execution, turn_results = out
    assert len(turn_results) == 2
    assert execution.actual_response.startswith("resp:")


@pytest.mark.asyncio
async def test_evaluate_criteria_wraps_exceptions_as_failure():
    evaluator = AgentEvaluator(MagicMock(), TrajectoryCollector(), config=EvalConfig.default())

    class _Good:
        name = "good"

        async def evaluate(self, execution, case):
            return CriterionResult.success("good", 1.0, 0.5)

    class _Bad:
        name = "bad"

        async def evaluate(self, execution, case):
            raise RuntimeError("criterion failed")

    evaluator.criteria = [_Good(), _Bad()]
    case = EvalCase.single_turn("c1", "u", "a")
    results = await evaluator._evaluate_criteria(MagicMock(), case)
    assert len(results) == 2
    assert any(r.criterion == "bad" and r.error for r in results)


def test_load_config_reads_eval_config_json(tmp_path):
    config_path = tmp_path / "eval_config.json"
    config_path.write_text(json.dumps(EvalConfig.default().model_dump()), encoding="utf-8")
    loaded = AgentEvaluator._load_config(str(config_path))
    assert isinstance(loaded, EvalConfig)


def test_evaluate_sync_runs_async_evaluate(monkeypatch):
    fake_report = MagicMock(eval_set_id="s1")

    monkeypatch.setattr(AgentEvaluator, "evaluate", AsyncMock(return_value=fake_report))
    out = AgentEvaluator.evaluate_sync(MagicMock(), TrajectoryCollector(), "dummy")
    assert out is fake_report
