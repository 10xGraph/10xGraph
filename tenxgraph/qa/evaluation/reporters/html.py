"""
HTML reporter for evaluation results.

Generates interactive, self-contained HTML reports for evaluation results.
The report includes summary stat cards, SVG bar charts (criterion breakdown
and per-case scores), dark mode, and expandable case detail panes with
criterion score bars.

Implementation is split across:
    _html_css.py      — stylesheet (light/dark themes, score bars)
    _html_js.py       — interactivity + SVG chart renderer (no CDN)
    _html_template.py — page structure (``build_html``)
    _html_render.py   — case / criterion HTML rendering
"""

from __future__ import annotations

import html as _html_lib
from pathlib import Path
from typing import TYPE_CHECKING

from agentflow.qa.evaluation.reporters._html_css import CSS_CONTENT
from agentflow.qa.evaluation.reporters._html_js import JS_CONTENT
from agentflow.qa.evaluation.reporters._html_render import render_case
from agentflow.qa.evaluation.reporters._html_template import build_html
from agentflow.qa.evaluation.reporters._utils import format_timestamp
from agentflow.qa.evaluation.reporters.base import BaseReporter


if TYPE_CHECKING:
    from agentflow.qa.evaluation.eval_result import EvalCaseResult, EvalReport


class HTMLReporter(BaseReporter):
    """Generate interactive HTML reports for evaluation results.

    Creates a self-contained HTML file (dark mode, SVG charts, criterion score
    bars, case search/filter) with no external CDN dependencies.

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

    # ── BaseReporter interface ───────────────────────────────────────────────

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

    # ── Public API ───────────────────────────────────────────────────────────

    def to_html(self, report: EvalReport) -> str:
        """Return the complete HTML report as a string."""
        title = report.eval_set_name or report.eval_set_id
        case_items = "\n".join(self._render_case(r) for r in report.results)

        tok = report.summary.total_token_usage
        if tok.total_tokens:
            cache_part = (
                f"&nbsp;&nbsp;cache_read: <strong>{tok.cache_read_tokens:,}</strong>"
                if tok.cache_read_tokens
                else ""
            )
            token_summary = (
                f"&#x1F4B0; Tokens &mdash; "
                f"input: <strong>{tok.input_tokens:,}</strong>&nbsp;&nbsp;"
                f"output: <strong>{tok.output_tokens:,}</strong>"
                f"{cache_part}&nbsp;&nbsp;"
                f"total: <strong>{tok.total_tokens:,}</strong>&nbsp;&nbsp;"
                f"avg/case: <strong>{report.summary.avg_tokens_per_case:.0f}</strong>"
            )
        else:
            token_summary = ""  # nosec B105

        return build_html(
            title=_html_lib.escape(title),
            timestamp=format_timestamp(report.timestamp),
            total_cases=report.summary.total_cases,
            passed_cases=report.summary.passed_cases,
            failed_cases=report.summary.failed_cases,
            error_cases=report.summary.error_cases,
            pass_rate_pct=f"{report.summary.pass_rate * 100:.0f}",
            duration=f"{report.summary.total_duration_seconds:.2f}",
            case_items=case_items,
            css=CSS_CONTENT,
            js=JS_CONTENT,
            token_summary=token_summary,
        )

    def _render_case(self, result: EvalCaseResult) -> str:
        return render_case(
            result,
            include_details=self.include_details,
            include_actual_response=self.include_actual_response,
            include_tool_call_details=self.include_tool_call_details,
            include_node_responses=self.include_node_responses,
            include_trajectory=self.include_trajectory,
        )

    def save(self, report: EvalReport, path: str) -> None:
        """Write the report to *path*, creating parent directories as needed."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.to_html(report), encoding="utf-8")

    @classmethod
    def quick_save(cls, report: EvalReport, path: str) -> None:
        """Convenience class-method to save a report in one call."""
        cls().save(report, path)
