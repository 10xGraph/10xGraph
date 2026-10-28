"""Report an evaluation: Console, JSON, HTML and JUnit XML reporters.

The report here is built by hand, so this script needs no API key and no LLM.
With a real run you get the same `EvalReport` back from `AgentEvaluator.evaluate()`.
Run: python examples/evaluation/reporters.py   (writes to eval_reports/example/)
"""

from pathlib import Path

from tenxgraph.qa.evaluation import (
    ConsoleReporter,
    CriterionResult,
    EvalCaseResult,
    EvalReport,
    HTMLReporter,
    JSONReporter,
    JUnitXMLReporter,
    ReporterConfig,
    ReporterManager,
    ToolCall,
)


def sample_report() -> EvalReport:
    passing = EvalCaseResult.success(
        eval_id="nyc_weather",
        name="NYC weather",
        criterion_results=[
            CriterionResult.success("tool_name_match_score", 1.0, threshold=1.0),
            CriterionResult.success("rouge_match", 0.92, threshold=0.5),
        ],
        actual_tool_calls=[ToolCall(name="get_weather", args={"location": "New York City"})],
        actual_response="The weather in New York City is sunny.",
        duration_seconds=2.4,
    )
    failing = EvalCaseResult.success(
        eval_id="capital",
        name="Capital, no tool",
        criterion_results=[
            CriterionResult.success("tool_name_match_score", 0.0, threshold=1.0),
        ],
        actual_response="Berlin is the capital of Germany.",
        duration_seconds=1.1,
    )
    return EvalReport.create(
        eval_set_id="weather_demo",
        eval_set_name="Weather demo",
        results=[passing, failing],
    )


def main() -> None:
    report = sample_report()
    out = Path("eval_reports/example")

    # Console: colour is on by default; turn it off for CI logs.
    ConsoleReporter(use_color=False, verbose=True, include_actual_response=True).report(report)

    # Files: each reporter has save(report, path); to_json()/to_html()/to_xml() return strings.
    JSONReporter(indent=2).save(report, str(out / "report.json"))
    HTMLReporter().save(report, str(out / "report.html"))
    JUnitXMLReporter(suite_name="weather-agent").save(report, str(out / "junit.xml"))

    # The evaluator does the same for you when EvalConfig.reporter is enabled. This is
    # the manager it uses, driven by a ReporterConfig.
    output = ReporterManager(
        ReporterConfig(console=False, junit_xml=True, output_dir=str(out / "managed"))
    ).run_all(report)
    print("files written by ReporterManager:", output.generated_files)

    print(f"pass rate {report.summary.pass_rate:.0%}; files in {out}/")


if __name__ == "__main__":
    main()
