"""
Evaluation result reporters.

This module provides various output formats for evaluation results:
    - ConsoleReporter: Pretty-print results to console
    - JSONReporter: Export results to JSON file
    - HTMLReporter: Generate HTML report
    - JUnitXMLReporter: Export results to JUnit XML
    - ReporterManager: Orchestrates all enabled reporters
    - BaseReporter: Abstract base class for custom reporters
"""

from tenxgraph.qa.evaluation.reporters.base import BaseReporter
from tenxgraph.qa.evaluation.reporters.console import (
    Colors,
    ConsoleReporter,
    print_report,
)
from tenxgraph.qa.evaluation.reporters.html import (
    HTMLReporter,
)
from tenxgraph.qa.evaluation.reporters.json import (
    JSONReporter,
    JUnitXMLReporter,
)
from tenxgraph.qa.evaluation.reporters.manager import (
    ReporterManager,
    ReporterOutput,
)


__all__ = [
    # Base
    "BaseReporter",
    "Colors",
    # Console
    "ConsoleReporter",
    # HTML
    "HTMLReporter",
    # JSON
    "JSONReporter",
    "JUnitXMLReporter",
    # Manager
    "ReporterManager",
    "ReporterOutput",
    "print_report",
]
