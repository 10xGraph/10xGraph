"""
Shared types and constants for evaluation configuration.

This module defines base constants and enumerations used across
evaluation configuration models.
"""

from __future__ import annotations

from enum import StrEnum


# Default judge model used across all LLM-based evaluation criteria.
# Users can override per-criterion or per-preset by passing judge_model="gpt-4o" etc.
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"


class MatchType(StrEnum):
    """Match type for trajectory comparison.

    Values:
        EXACT: Require perfect match - same tools, args, and order.
        IN_ORDER: Expected tools must appear in order, extras allowed.
        ANY_ORDER: Expected tools must appear in any order, extras allowed.
    """

    EXACT = "EXACT"
    IN_ORDER = "IN_ORDER"
    ANY_ORDER = "ANY_ORDER"
