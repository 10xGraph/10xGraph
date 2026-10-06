"""
JSON reporter for evaluation results.

Exports evaluation reports to JSON files with optional formatting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentflow.qa.evaluation.reporters.base import BaseReporter


if TYPE_CHECKING:
    from agentflow.qa.evaluation.eval_result import EvalReport


class JSONReporter(BaseReporter):
    """Export evaluation results to JSON files.

    Provides options for formatting, filtering, and customizing
    the JSON output structure.

    Attributes:
        indent: JSON indentation level (None for compact).
        include_details: Whether to include full criterion details.
        include_trajectory: Whether to include trajectory data.
        include_node_responses: Whether to include node response data.
        include_actual_response: Whether to include raw agent response.
        include_tool_call_details: Whether to include tool call args/results.

    Example:
        ```python
        reporter = JSONReporter(indent=2)
        reporter.save(report, "results/eval_report.json")

        # Or get as string
        json_str = reporter.to_json(report)
        ```
    """

    def __init__(
        self,
        indent: int | None = 2,
        include_details: bool = True,
        include_trajectory: bool = True,
        include_node_responses: bool = True,
        include_actual_response: bool = True,
        include_tool_call_details: bool = True,
    ):
        """Initialize the JSON reporter.

        Args:
            indent: JSON indentation (None for minified output).
            include_details: Include criterion details in output.
            include_trajectory: Include trajectory data in output.
            include_node_responses: Include node-level response data.
            include_actual_response: Include agent response text.
            include_tool_call_details: Include tool call args/results.
        """
        self.indent = indent
        self.include_details = include_details
        self.include_trajectory = include_trajectory
        self.include_node_responses = include_node_responses
        self.include_actual_response = include_actual_response
        self.include_tool_call_details = include_tool_call_details

    # --- BaseReporter interface ---

    def generate(self, report: EvalReport, output_dir: str | None = None) -> str | None:
        """Generate a JSON report.

        If *output_dir* is provided the file is written there; otherwise
        the JSON string is returned.
        """
        if output_dir is None:
            return self.to_json(report)
        path = str(Path(output_dir) / "report.json")
        self.save(report, path)
        return path

    # --- Existing public API ---

    def to_dict(self, report: EvalReport) -> dict[str, Any]:
        """Convert report to dictionary.

        Args:
            report: The evaluation report.

        Returns:
            Dictionary representation of the report.
        """
        data = report.model_dump()

        for result in data.get("results", []):
            # Criterion details
            if not self.include_details:
                for cr in result.get("criterion_results", []):
                    cr.pop("details", None)

            # Trajectory
            if not self.include_trajectory:
                result.pop("actual_trajectory", None)

            # Tool calls
            if not self.include_tool_call_details:
                result.pop("actual_tool_calls", None)

            # Node responses
            if not self.include_node_responses:
                result.pop("node_responses", None)
                result.pop("node_details", None)

            # Actual response text
            if not self.include_actual_response:
                result.pop("actual_response", None)

        return data

    def to_json(self, report: EvalReport) -> str:
        """Convert report to JSON string.

        Args:
            report: The evaluation report.

        Returns:
            JSON string representation.
        """
        data = self.to_dict(report)
        return json.dumps(data, indent=self.indent, default=str)

    def save(self, report: EvalReport, path: str) -> None:
        """Save report to a JSON file.

        Args:
            report: The evaluation report.
            path: Output file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.to_json(report))

    @classmethod
    def quick_save(
        cls,
        report: EvalReport,
        path: str,
        indent: int = 2,
    ) -> None:
        """Convenience method to quickly save a report.

        Args:
            report: The evaluation report.
            path: Output file path.
            indent: JSON indentation.
        """
        reporter = cls(indent=indent)
        reporter.save(report, path)


class JUnitXMLReporter(BaseReporter):
    """Export evaluation results to JUnit XML format.

    Useful for CI/CD integration with tools that understand
    JUnit test result format.

    Example:
        ```python
        reporter = JUnitXMLReporter()
        reporter.save(report, "results/junit.xml")
        ```
    """

    def __init__(self, suite_name: str = "agent-evaluation"):
        """Initialize the JUnit reporter.

        Args:
            suite_name: Name of the test suite.
        """
        self.suite_name = suite_name

    # --- BaseReporter interface ---

    def generate(self, report: EvalReport, output_dir: str | None = None) -> str | None:
        """Generate a JUnit XML report.

        If *output_dir* is provided the file is written there; otherwise
        the XML string is returned.
        """
        if output_dir is None:
            return self.to_xml(report)
        path = str(Path(output_dir) / "junit.xml")
        self.save(report, path)
        return path

    def to_xml(self, report: EvalReport) -> str:  # noqa: PLR0912, PLR0915
        """Convert report to JUnit XML format.

        Args:
            report: The evaluation report.

        Returns:
            XML string in JUnit format.
        """
        import json
        import xml.etree.ElementTree as ET

        from agentflow.qa.evaluation.reporters._utils import format_timestamp

        # Create root element
        testsuite = ET.Element("testsuite")
        testsuite.set("name", self.suite_name)
        testsuite.set("tests", str(report.summary.total_cases))
        testsuite.set("failures", str(report.summary.failed_cases))
        testsuite.set("errors", str(report.summary.error_cases))
        testsuite.set("time", f"{report.summary.total_duration_seconds:.3f}")
        testsuite.set(
            "timestamp",
            format_timestamp(report.timestamp, fmt="%Y-%m-%dT%H:%M:%S"),
        )

        # Add testsuite-level properties for report metadata
        properties = ET.SubElement(testsuite, "properties")
        prop_items = {
            "eval_set_id": report.eval_set_id,
            "eval_set_name": report.eval_set_name,
            "pass_rate": f"{report.summary.pass_rate:.4f}",
            "avg_duration": f"{report.summary.avg_duration_seconds:.3f}",
        }
        if report.config_used:
            prop_items["config_used"] = json.dumps(report.config_used, default=str)
        if report.metadata:
            prop_items["metadata"] = json.dumps(report.metadata, default=str)

        for pk, pv in prop_items.items():
            prop = ET.SubElement(properties, "property")
            prop.set("name", pk)
            prop.set("value", str(pv))

        # Add per-criterion stats as properties
        for crit_name, crit_stats in report.summary.criterion_stats.items():
            prop = ET.SubElement(properties, "property")
            prop.set("name", f"criterion.{crit_name}")
            prop.set("value", json.dumps(crit_stats, default=str))

        # Add test cases
        for result in report.results:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", result.name or result.eval_id)
            testcase.set("classname", report.eval_set_id)
            testcase.set("time", f"{result.duration_seconds:.3f}")

            if result.is_error:
                error = ET.SubElement(testcase, "error")
                error.set("message", result.error or "Unknown error")
                error.text = result.error
            elif not result.passed:
                # Add failure for each failed criterion with full details
                for cr in result.failed_criteria:
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("type", cr.criterion)
                    msg = f"Score {cr.score:.2f} below threshold {cr.threshold}"
                    if cr.reason:
                        msg += f" | {cr.reason}"
                    failure.set("message", msg)
                    # Include full details in failure body
                    failure_body_parts = []
                    if cr.error:
                        failure_body_parts.append(f"Error: {cr.error}")
                    if cr.details:
                        failure_body_parts.append(
                            f"Details: {json.dumps(cr.details, default=str, indent=2)}"
                        )
                    failure.text = "\n".join(failure_body_parts) if failure_body_parts else None

            # Add system-out with full case data for every test case
            system_out = ET.SubElement(testcase, "system-out")
            out_parts = []

            if result.actual_response:
                out_parts.append(f"=== Agent Response ===\n{result.actual_response}")

            if result.actual_tool_calls:
                tc_data = [
                    tc.model_dump() if hasattr(tc, "model_dump") else tc
                    for tc in result.actual_tool_calls
                ]
                tc_json = json.dumps(tc_data, default=str, indent=2)
                out_parts.append(f"=== Tool Calls ({len(tc_data)}) ===\n{tc_json}")

            if result.actual_trajectory:
                traj_data = [
                    s.model_dump() if hasattr(s, "model_dump") else s
                    for s in result.actual_trajectory
                ]
                traj_json = json.dumps(traj_data, default=str, indent=2)
                out_parts.append(f"=== Trajectory ({len(traj_data)} steps) ===\n{traj_json}")

            if getattr(result, "node_visits", None):
                out_parts.append(f"=== Node Visits ===\n{' -> '.join(result.node_visits)}")

            if getattr(result, "node_responses", None):
                nr_json = json.dumps(result.node_responses, default=str, indent=2)
                out_parts.append(
                    f"=== Node Responses ({len(result.node_responses)}) ===" f"\n{nr_json}"
                )

            if getattr(result, "messages", None):
                msg_json = json.dumps(result.messages, default=str, indent=2)
                out_parts.append(f"=== Messages ({len(result.messages)}) ===\n{msg_json}")

            if getattr(result, "metadata", None):
                out_parts.append(
                    f"=== Metadata ===\n{json.dumps(result.metadata, default=str, indent=2)}"
                )

            if getattr(result, "turn_results", None):
                tr_json = json.dumps(result.turn_results, default=str, indent=2)
                out_parts.append(
                    f"=== Turn Results ({len(result.turn_results)}) ===" f"\n{tr_json}"
                )

            # Include ALL criteria results (passed + failed)
            cr_parts = []
            for cr in result.criterion_results:
                cr_status = "PASS" if cr.passed else "FAIL"
                cr_parts.append(
                    f"  [{cr_status}] {cr.criterion}: {cr.score:.2f} (threshold: {cr.threshold})"
                )
                if cr.reason:
                    cr_parts.append(f"    Reason: {cr.reason}")
                if cr.details:
                    for dk, dv in cr.details.items():
                        if dk == "reason":
                            continue
                        cr_parts.append(f"    {dk}: {dv}")
                if cr.error:
                    cr_parts.append(f"    Error: {cr.error}")
            if cr_parts:
                out_parts.append("=== Criteria Results ===\n" + "\n".join(cr_parts))

            system_out.text = "\n\n".join(out_parts) if out_parts else "No execution data"

        # Convert to string
        return ET.tostring(testsuite, encoding="unicode", xml_declaration=True)

    def save(self, report: EvalReport, path: str) -> None:
        """Save report to JUnit XML file.

        Args:
            report: The evaluation report.
            path: Output file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.to_xml(report))
