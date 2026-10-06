"""
Base reporter protocol for evaluation results.

Defines the abstract interface that all reporters must implement,
enabling polymorphic invocation via ReporterManager.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agentflow.qa.evaluation.eval_result import EvalReport


class BaseReporter(abc.ABC):
    """Abstract base class for evaluation reporters.

    All reporters (Console, JSON, HTML, JUnit XML) implement this
    interface so that ``ReporterManager`` can invoke them uniformly.

    Subclasses must implement :meth:`generate` which produces the
    report output (prints to console, writes files, etc.).

    Example:
        ```python
        class MyReporter(BaseReporter):
            def generate(self, report, output_dir=None): ...
        ```
    """

    @abc.abstractmethod
    def generate(self, report: EvalReport, output_dir: str | None = None) -> str | None:
        """Generate a report from evaluation results.

        Args:
            report: The evaluation report to render.
            output_dir: Optional directory for file-based reporters.
                        Console reporters may ignore this.

        Returns:
            Path to the generated file, or None for console-only reporters.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable reporter name for logging."""
        return self.__class__.__name__
