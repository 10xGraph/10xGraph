"""The structured report the research assistant produces."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Finding(BaseModel):
    claim: str = Field(description="One specific, checkable statement.")
    evidence: str = Field(description="What in the sources supports the claim.")
    source_url: str = Field(description="URL of the page that supports the claim.")


class ResearchReport(BaseModel):
    title: str
    summary: str = Field(description="Three to five sentences answering the question.")
    key_findings: list[Finding]
    open_questions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list, description="All URLs used.")
