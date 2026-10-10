"""
Reporter and simulator configuration models for agent evaluation.

This module defines:
    - UserSimulatorConfig: Configuration for AI-powered user simulation.
    - ReporterConfig: Configuration for evaluation result reporters.
"""

from __future__ import annotations

from pydantic import BaseModel


class UserSimulatorConfig(BaseModel):
    """Configuration for AI-powered user simulation.

    Attributes:
        model: Model to use for generating user prompts.
        max_invocations: Maximum number of conversation turns.
        temperature: Temperature for generation.
        thinking_enabled: Whether to enable thinking/reasoning.
        thinking_budget: Token budget for thinking (if enabled).
        api_style: OpenAI API style — ``"responses"`` (default) or ``"chat"``
            for models that only support the legacy Chat Completions endpoint.
    """

    model: str = "gemini-2.5-flash"
    max_invocations: int = 10
    temperature: float = 0.7
    thinking_enabled: bool = False
    thinking_budget: int = 10240
    api_style: str = "responses"


class ReporterConfig(BaseModel):
    """Configuration for evaluation result reporters.

    Controls which reporters are enabled and where output files
    are written after an evaluation completes.

    Attributes:
        enabled: Master switch — when False, no reporters run automatically.
        output_dir: Directory for generated report files.
        console: Enable console (stdout) reporting.
        json_report: Enable JSON file reporting.
        html: Enable HTML file reporting.
        junit_xml: Enable JUnit XML file reporting.
        verbose: Verbose console output (show all cases, not just failures).
        include_details: Include full criterion details in file reports.
        include_trajectory: Include trajectory data in JSON reports.
        include_node_responses: Include per-node intermediate data in reports.
        include_actual_response: Include agent final response in reports.
        include_tool_call_details: Include tool arguments and results in reports.
        timestamp_files: Append timestamp to generated filenames.
    """

    enabled: bool = True
    output_dir: str = "eval_reports"
    console: bool = True
    json_report: bool = True
    html: bool = True
    junit_xml: bool = False
    verbose: bool = True
    include_details: bool = True
    include_trajectory: bool = True
    include_node_responses: bool = True
    include_actual_response: bool = True
    include_tool_call_details: bool = True
    timestamp_files: bool = True
