"""
HTML reporter for evaluation results.

Generates a self-contained HTML scorecard for an evaluation run: a verdict
against the pass-rate gate, headline numbers with deltas against the previous
report in the same folder, a cases-by-criteria matrix, and a detail drawer per
case. The file opens offline from ``file://`` and makes no network requests.

Implementation is split across:
    _html_payload.py  - JSON payload, gate, average score, baseline lookup
    _html_template.py - page structure (``build_html``), CSP
    _html_css.py      - stylesheet (light/dark, phone, print)
    _html_js.py       - rendering and interaction (reads the embedded payload)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tenxgraph.qa.evaluation.reporters._html_css import CSS_CONTENT
from tenxgraph.qa.evaluation.reporters._html_js import JS_CONTENT
from tenxgraph.qa.evaluation.reporters._html_payload import (
    build_payload,
    dumps_for_script,
    find_baseline,
)
from tenxgraph.qa.evaluation.reporters._html_template import build_html
from tenxgraph.qa.evaluation.reporters.base import BaseReporter
from tenxgraph.qa.evaluation.reporters.json import JSONReporter


if TYPE_CHECKING:
    from tenxgraph.qa.evaluation.eval_result import EvalReport


class HTMLReporter(BaseReporter):
    """Generate the HTML scorecard report for evaluation results.

    The previous run is found automatically: the most recent JSON report of the
    same ``eval_set_id`` in the output folder whose timestamp is earlier than the
    current report's. ``save`` and ``generate`` look in the folder they write to;
    ``to_html`` accepts ``baseline_dir`` for the same lookup.

    Example::

        reporter = HTMLReporter()
        reporter.save(report, "results/report.html")
    """

    def __init__(
        self,
        include_details: bool = True,
        include_actual_response: bool = True,
        include_tool_call_details: bool = True,
        include_node_responses: bool = True,
        include_trajectory: bool = True,
    ) -> None:
        self.include_details = include_details
        self.include_actual_response = include_actual_response
        self.include_tool_call_details = include_tool_call_details
        self.include_node_responses = include_node_responses
        self.include_trajectory = include_trajectory

    # -- BaseReporter interface ----------------------------------------------

    def generate(self, report: EvalReport, output_dir: str | None = None) -> str | None:
        """Generate an HTML report.

        If *output_dir* is provided the file is written there; otherwise
        the HTML string is returned.
        """
        if output_dir is None:
            return self.to_html(report)
        path = str(Path(output_dir) / "report.html")
        self.save(report, path)
        return path

    # -- Public API ----------------------------------------------------------

    def to_html(
        self,
        report: EvalReport,
        baseline_dir: str | None = None,
        exclude: list[str] | None = None,
    ) -> str:
        """Return the complete HTML report as a string.

        Args:
            report: The evaluation report.
            baseline_dir: Folder to search for the previous JSON report. Without
                it the page shows its "no previous run" state.
            exclude: JSON files to ignore as baseline (the current run's file).
        """
        baseline = find_baseline(report, baseline_dir, exclude) if baseline_dir else None
        current = JSONReporter(
            include_details=self.include_details,
            include_trajectory=self.include_trajectory,
            include_node_responses=self.include_node_responses,
            include_actual_response=self.include_actual_response,
            include_tool_call_details=self.include_tool_call_details,
        ).to_dict(report)
        payload = build_payload(current, report, baseline)
        title = f"{report.eval_set_name or report.eval_set_id} | 10xGraph eval report"
        return build_html(
            title=title,
            css=CSS_CONTENT,
            js=JS_CONTENT,
            data_json=dumps_for_script(payload),
        )

    def save(self, report: EvalReport, path: str) -> None:
        """Write the report to *path*, creating parent directories as needed.

        The previous run is looked up in the same folder; the sibling ``.json``
        file with the same name (written by the current run) is never used.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        html = self.to_html(
            report,
            baseline_dir=str(file_path.parent),
            exclude=[str(file_path.with_suffix(".json"))],
        )
        file_path.write_bytes(html.encode("utf-8"))

    @classmethod
    def quick_save(cls, report: EvalReport, path: str) -> None:
        """Convenience class-method to save a report in one call."""
        cls().save(report, path)
