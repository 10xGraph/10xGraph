from unittest.mock import MagicMock, patch

from tenxgraph.qa.evaluation.config.eval_config import ReporterConfig
from tenxgraph.qa.evaluation.eval_result import CriterionResult, EvalCaseResult, EvalReport
from tenxgraph.qa.evaluation.reporters.manager import ReporterManager, ReporterOutput


def _sample_report() -> EvalReport:
    case = EvalCaseResult.success(
        eval_id="case-1",
        criterion_results=[
            CriterionResult.success(
                criterion="response_match_score",
                score=1.0,
                threshold=0.8,
            )
        ],
        actual_response="ok",
    )
    return EvalReport.create(eval_set_id="set:1/alpha", results=[case], eval_set_name="Sample")


def test_reporter_output_helpers():
    output = ReporterOutput(json_path="a.json", html_path=None, junit_path="j.xml")
    assert output.has_errors is False
    assert output.generated_files == ["a.json", "j.xml"]


def test_run_all_returns_empty_when_disabled():
    manager = ReporterManager(ReporterConfig(enabled=False))
    output = manager.run_all(_sample_report())
    assert output.generated_files == []
    assert output.console_output is False


def test_run_all_executes_enabled_reporters_and_generates_paths(tmp_path):
    config = ReporterConfig(
        enabled=True,
        output_dir=str(tmp_path),
        console=True,
        json_report=True,
        html=True,
        junit_xml=True,
        timestamp_files=False,
    )
    manager = ReporterManager(config)
    report = _sample_report()

    with patch("tenxgraph.qa.evaluation.reporters.console.ConsoleReporter") as console_cls, patch(
        "tenxgraph.qa.evaluation.reporters.json.JSONReporter"
    ) as json_cls, patch("tenxgraph.qa.evaluation.reporters.html.HTMLReporter") as html_cls, patch(
        "tenxgraph.qa.evaluation.reporters.json.JUnitXMLReporter"
    ) as junit_cls:
        console_inst = MagicMock()
        json_inst = MagicMock()
        html_inst = MagicMock()
        junit_inst = MagicMock()

        console_cls.return_value = console_inst
        json_cls.return_value = json_inst
        html_cls.return_value = html_inst
        junit_cls.return_value = junit_inst

        out = manager.run_all(report)

    assert out.console_output is True
    assert out.errors == []
    assert out.json_path is not None and out.json_path.endswith("set_1_alpha.json")
    assert out.html_path is not None and out.html_path.endswith("set_1_alpha.html")
    assert out.junit_path is not None and out.junit_path.endswith("set_1_alpha_junit.xml")

    console_inst.report.assert_called_once()
    json_inst.save.assert_called_once()
    html_inst.save.assert_called_once()
    junit_inst.save.assert_called_once()


def test_run_all_collects_reporter_errors_and_continues(tmp_path):
    config = ReporterConfig(
        enabled=True,
        output_dir=str(tmp_path),
        console=True,
        json_report=True,
        html=False,
        junit_xml=False,
        timestamp_files=False,
    )
    manager = ReporterManager(config)

    with patch("tenxgraph.qa.evaluation.reporters.console.ConsoleReporter") as console_cls, patch(
        "tenxgraph.qa.evaluation.reporters.json.JSONReporter"
    ) as json_cls:
        console_inst = MagicMock()
        console_inst.report.side_effect = RuntimeError("console boom")
        json_inst = MagicMock()
        json_inst.save.side_effect = RuntimeError("json boom")
        console_cls.return_value = console_inst
        json_cls.return_value = json_inst

        out = manager.run_all(_sample_report())

    assert out.console_output is False
    assert out.json_path is None
    assert out.has_errors is True
    names = [name for name, _ in out.errors]
    assert "ConsoleReporter" in names
    assert "JSONReporter" in names


def test_build_stem_sanitizes_eval_set_id_and_respects_timestamp_flag():
    config = ReporterConfig(enabled=True, timestamp_files=False)
    manager = ReporterManager(config)
    stem = manager._build_stem(_sample_report())
    assert stem == "set_1_alpha"


def test_run_all_records_directory_creation_error(monkeypatch, tmp_path):
    config = ReporterConfig(
        enabled=True,
        output_dir=str(tmp_path / "bad_dir"),
        console=True,
        json_report=False,
        html=False,
        junit_xml=False,
    )
    manager = ReporterManager(config)

    def _boom(*args, **kwargs):
        raise OSError("mkdir failed")

    monkeypatch.setattr("pathlib.Path.mkdir", _boom)

    with patch("tenxgraph.qa.evaluation.reporters.console.ConsoleReporter") as console_cls:
        console_inst = MagicMock()
        console_cls.return_value = console_inst
        out = manager.run_all(_sample_report())

    assert out.has_errors is True
    assert any(name == "directory" for name, _ in out.errors)
    assert out.console_output is True
