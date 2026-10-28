"""Parse the DeepSeek output into a `ResearchReport` and save it as markdown."""

from __future__ import annotations

import re
from pathlib import Path

from models import ResearchReport


def parse_report(text: str) -> ResearchReport:
    """Validate the model's JSON. Raises pydantic.ValidationError if it is malformed."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return ResearchReport.model_validate_json(cleaned)


def to_markdown(report: ResearchReport) -> str:
    lines = [f"# {report.title}", "", report.summary, "", "## Key findings", ""]
    for i, f in enumerate(report.key_findings, 1):
        lines += [
            f"{i}. **{f.claim}**",
            f"   - Evidence: {f.evidence}",
            f"   - Source: {f.source_url}",
        ]
    if report.open_questions:
        lines += ["", "## Open questions", ""] + [f"- {q}" for q in report.open_questions]
    if report.sources:
        lines += ["", "## Sources", ""] + [f"- {s}" for s in report.sources]
    return "\n".join(lines) + "\n"


def save_markdown(report: ResearchReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(report), encoding="utf-8")
    return path
