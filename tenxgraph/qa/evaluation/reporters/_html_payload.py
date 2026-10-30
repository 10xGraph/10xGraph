"""Data payload for the HTML evaluation report.

The page is rendered in the browser from one JSON document embedded in the
file. This module builds that document: the current report, the gate, the
average score and a trimmed copy of the previous report found next to the
output file.

Everything in a report is untrusted text (model output, user input, tool
results). ``dumps_for_script`` makes the JSON safe to embed in a script element.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from tenxgraph.qa.evaluation.reporters._utils import safe_stem


if TYPE_CHECKING:
    from collections.abc import Sequence

    from tenxgraph.qa.evaluation.eval_result import EvalReport


logger = logging.getLogger("tenxgraph.qa.evaluation.reporters.html")

PAYLOAD_VERSION = 1
# Baseline search limits: skip very large files and look at the newest files only.
_MAX_BASELINE_BYTES = 64 * 1024 * 1024
_MAX_BASELINE_FILES = 200
_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "total_tokens",
)


# ── Safe embedding ───────────────────────────────────────────────────────────


def json_safe(obj: Any) -> Any:
    """Return ``obj`` with non-finite floats replaced by ``None`` and keys as strings."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    return obj


def dumps_for_script(payload: Any) -> str:
    """Serialise ``payload`` for a ``<script type="application/json">`` element.

    ``<``, ``>`` and ``&`` are written as unicode escapes, so ``</script>``,
    ``<!--`` and entities in any string cannot end or alter the element. With
    ``ensure_ascii`` every non-ASCII character, including the line separators U+2028
    and U+2029, is written as a ``\\uXXXX`` escape.
    """
    text = json.dumps(
        json_safe(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


# ── Gate and averages ────────────────────────────────────────────────────────


def mean_score(results: list[Any]) -> float | None:
    """Mean of every criterion score across ``results`` (dicts or models); ``None`` if none."""
    scores: list[float] = []
    for r in results:
        crs = r.get("criterion_results", []) if isinstance(r, dict) else r.criterion_results
        for c in crs or []:
            score = c.get("score") if isinstance(c, dict) else c.score
            val = _num(score, None)
            if val is not None:
                scores.append(val)
    return sum(scores) / len(scores) if scores else None


def resolve_gate(report: EvalReport) -> dict[str, Any]:
    """Return ``{"threshold": float, "source": "threshold" | "default"}``.

    Read from ``report.metadata["pass_rate_threshold"]`` (set by ``10xgraph eval``).
    Without it the gate is 1.0 and the page says every case must pass.
    """
    meta = report.metadata if isinstance(report.metadata, dict) else {}
    threshold = _num(meta.get("pass_rate_threshold"), None)
    if threshold is None or not 0.0 <= threshold <= 1.0:
        return {"threshold": 1.0, "source": "default"}
    if threshold == 1.0 and meta.get("pass_rate_threshold_source") != "threshold":
        return {"threshold": 1.0, "source": "default"}
    return {"threshold": threshold, "source": "threshold"}


# ── Baseline (previous report in the same folder) ────────────────────────────


def find_baseline(
    report: EvalReport,
    directory: str | Path,
    exclude: Sequence[str | Path] | None = None,
) -> dict[str, Any] | None:
    """Find and trim the most recent earlier JSON report of the same eval set.

    Candidates are ``*.json`` files in ``directory`` (not recursive) named after
    the eval set id, whose ``eval_set_id`` matches and whose ``timestamp`` is
    strictly earlier than ``report.timestamp``. Files in ``exclude`` (the file
    written by the current run) are skipped. Any unreadable or malformed file is
    skipped; if nothing usable is found the result is ``None``.
    """
    try:
        folder = Path(directory)
        if not folder.is_dir():
            return None
        skipped = {_resolve(p) for p in exclude or []}
        prefix = safe_stem(report.eval_set_id)
        files = [
            p
            for p in folder.glob("*.json")
            if p.stem.startswith(prefix) and p.is_file() and _resolve(p) not in skipped
        ]
        files.sort(key=_mtime, reverse=True)
        found: list[tuple[float, str, dict[str, Any]]] = []
        for path in files[:_MAX_BASELINE_FILES]:
            data = _load_report(path)
            if data is None or data.get("eval_set_id") != report.eval_set_id:
                continue
            ts = _num(data.get("timestamp"), None)
            if ts is None or ts >= report.timestamp:
                continue
            found.append((ts, path.name, data))
        found.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for ts, name, data in found:
            try:
                return trim_baseline(data, name, ts)
            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                logger.debug("Skipping baseline candidate %s: %s", name, exc)
    except Exception as exc:  # a baseline is optional, never fail the report
        logger.debug("Baseline search failed in %s: %s", directory, exc)
    return None


def trim_baseline(data: dict[str, Any], filename: str, timestamp: float) -> dict[str, Any]:
    """Keep only the fields the page compares, coerced to plain strings and numbers."""
    summary = data["summary"]
    results_in = data["results"]
    if not isinstance(summary, dict) or not isinstance(results_in, list):
        raise ValueError("summary or results has the wrong type")

    results: list[dict[str, Any]] = []
    for r in results_in:
        if not isinstance(r, dict):
            continue
        crits = []
        for c in r.get("criterion_results") or []:
            if not isinstance(c, dict):
                continue
            crits.append(
                {
                    "criterion": _s(c.get("criterion")),
                    "score": _num(c.get("score"), 0.0),
                    "passed": c.get("passed") is True,
                    "threshold": _num(c.get("threshold"), 0.0),
                }
            )
        err = r.get("error")
        results.append(
            {
                "eval_id": _s(r.get("eval_id")),
                "name": _s(r.get("name")),
                "passed": r.get("passed") is True,
                "error": _s(err) if err else None,
                "duration_seconds": _num(r.get("duration_seconds"), 0.0),
                "token_usage": _tokens(r.get("token_usage")),
                "criterion_results": crits,
            }
        )

    stats: dict[str, dict[str, float]] = {}
    for name, st in (summary.get("criterion_stats") or {}).items():
        if isinstance(st, dict):
            stats[_s(name)] = {
                "total": _num(st.get("total"), 0.0),
                "passed": _num(st.get("passed"), 0.0),
                "failed": _num(st.get("failed"), 0.0),
                "avg_score": _num(st.get("avg_score"), 0.0),
                "pass_rate": _num(st.get("pass_rate"), 0.0),
            }

    return {
        "timestamp": timestamp,
        "filename": filename,
        "avg_score": mean_score(results),
        "summary": {
            "total_cases": _num(summary.get("total_cases"), 0.0),
            "passed_cases": _num(summary.get("passed_cases"), 0.0),
            "failed_cases": _num(summary.get("failed_cases"), 0.0),
            "error_cases": _num(summary.get("error_cases"), 0.0),
            "pass_rate": _num(summary.get("pass_rate"), 0.0),
            "avg_duration_seconds": _num(summary.get("avg_duration_seconds"), 0.0),
            "total_duration_seconds": _num(summary.get("total_duration_seconds"), 0.0),
            "criterion_stats": stats,
            "total_token_usage": _tokens(summary.get("total_token_usage")),
        },
        "results": results,
    }


# ── Payload ──────────────────────────────────────────────────────────────────


def build_payload(
    current: dict[str, Any],
    report: EvalReport,
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the document the page reads.

    Top-level keys: ``version``, ``cur`` (the report as the JSON reporter writes
    it), ``prev`` (trimmed baseline or ``None``), ``gate`` and ``avg_score``.
    """
    return {
        "version": PAYLOAD_VERSION,
        "cur": current,
        "prev": baseline,
        "gate": resolve_gate(report),
        "avg_score": mean_score(report.results),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


@overload
def _num(value: Any, default: float) -> float: ...


@overload
def _num(value: Any, default: None) -> float | None: ...


def _num(value: Any, default: float | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value) if math.isfinite(value) else default


def _s(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _tokens(value: Any) -> dict[str, float]:
    src = value if isinstance(value, dict) else {}
    return {k: _num(src.get(k), 0.0) for k in _TOKEN_KEYS}


def _resolve(path: str | Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _load_report(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > _MAX_BASELINE_BYTES:
            return None
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.debug("Ignoring unreadable report %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return None
    return data
