"""Tests for the HTML scorecard reporter (payload, gate, baseline, escaping)."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tenxgraph.qa.evaluation.config.reporter_config import ReporterConfig
from tenxgraph.qa.evaluation.eval_result import CriterionResult, EvalCaseResult, EvalReport
from tenxgraph.qa.evaluation.reporters import _html_payload as payload_mod
from tenxgraph.qa.evaluation.reporters._html_js import JS_CONTENT
from tenxgraph.qa.evaluation.reporters.html import HTMLReporter
from tenxgraph.qa.evaluation.reporters.json import JSONReporter
from tenxgraph.qa.evaluation.reporters.manager import ReporterManager


_DATA_RE = re.compile(r'<script type="application/json" id="eval-data">(.*?)</script>', re.DOTALL)


def _case(eval_id: str, scores: list[float], threshold: float = 0.7, **kw) -> EvalCaseResult:
    return EvalCaseResult.success(
        eval_id=eval_id,
        name=kw.pop("name", eval_id.title()),
        criterion_results=[
            CriterionResult.success(criterion=f"crit_{i}", score=s, threshold=threshold)
            for i, s in enumerate(scores)
        ],
        **kw,
    )


def _report(
    cases: list[EvalCaseResult] | None = None,
    *,
    set_id: str = "set_1",
    ts: float = 2000.0,
    metadata: dict | None = None,
) -> EvalReport:
    report = EvalReport.create(
        eval_set_id=set_id,
        results=cases if cases is not None else [_case("a", [1.0, 0.5]), _case("b", [0.9])],
        eval_set_name="My Set",
    )
    report.timestamp = ts
    if metadata:
        report.metadata.update(metadata)
    return report


def _save_json(report: EvalReport, path: Path) -> None:
    JSONReporter().save(report, str(path))


def _payload(html: str) -> dict:
    match = _DATA_RE.search(html)
    assert match, "payload element missing"
    return json.loads(match.group(1))


# ── File shape ───────────────────────────────────────────────────────────────


def test_file_written_and_self_contained(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "r.html"
    HTMLReporter().save(_report(), str(out))
    html = out.read_text(encoding="utf-8")

    assert html.startswith("<!doctype html>")
    assert "<title>My Set | 10xGraph eval report</title>" in html
    assert 'aria-label="10xGraph"' in html  # inline logo
    # No remote resource loads; the only URL allowed is the SVG namespace.
    assert not re.search(r"""(?:src|href|action)\s*=\s*["']?(?:https?:)?//""", html)
    assert not re.search(r"url\(\s*[\"']?(?:https?:)?//", html)
    assert "http://" not in html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "https://" not in html
    assert "prototype" not in html.lower()


def test_csp_hash_matches_page_script(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    HTMLReporter().save(_report(), str(out))
    html = out.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 1
    digest = base64.b64encode(hashlib.sha256(scripts[0].encode("utf-8")).digest()).decode()
    assert f"script-src &#x27;sha256-{digest}&#x27;" in html
    assert "default-src &#x27;none&#x27;" in html


def test_to_html_without_directory_has_no_baseline() -> None:
    html = HTMLReporter().to_html(_report())
    assert _payload(html)["prev"] is None
    assert "most recent earlier report" in html


def test_generate_returns_string_or_path(tmp_path: Path) -> None:
    reporter = HTMLReporter()
    assert reporter.generate(_report()).startswith("<!doctype html>")
    path = reporter.generate(_report(), str(tmp_path))
    assert Path(path).name == "report.html" and Path(path).exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_script_parses(tmp_path: Path) -> None:
    js = tmp_path / "page.js"
    js.write_text(JS_CONTENT, encoding="utf-8")
    node = shutil.which("node")
    result = subprocess.run([node, "--check", str(js)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


# ── Payload ──────────────────────────────────────────────────────────────────


def test_payload_parses_back_with_expected_keys() -> None:
    report = _report()
    data = _payload(HTMLReporter().to_html(report))
    assert set(data) == {"version", "cur", "prev", "gate", "avg_score"}
    assert data["cur"]["eval_set_id"] == "set_1"
    assert [r["eval_id"] for r in data["cur"]["results"]] == ["a", "b"]
    assert data["cur"]["summary"]["total_cases"] == 2


def test_include_flags_trim_payload() -> None:
    case = _case("a", [1.0], actual_response="secret text")
    html = HTMLReporter(include_actual_response=False).to_html(_report([case]))
    assert "actual_response" not in _payload(html)["cur"]["results"][0]
    assert "secret text" not in html


def test_non_finite_scores_keep_payload_valid() -> None:
    case = EvalCaseResult(
        eval_id="n",
        passed=False,
        criterion_results=[
            CriterionResult(criterion="c", score=float("nan"), passed=False, threshold=0.5)
        ],
    )
    data = _payload(HTMLReporter().to_html(_report([case])))
    assert data["cur"]["results"][0]["criterion_results"][0]["score"] is None
    assert data["avg_score"] is None


def test_average_score_is_mean_of_all_criterion_scores() -> None:
    report = _report([_case("a", [1.0, 0.5]), _case("b", [0.9])])
    data = _payload(HTMLReporter().to_html(report))
    assert data["avg_score"] == pytest.approx((1.0 + 0.5 + 0.9) / 3)


def test_average_score_none_without_criteria() -> None:
    data = _payload(HTMLReporter().to_html(_report([])))
    assert data["avg_score"] is None


# ── Gate ─────────────────────────────────────────────────────────────────────


def test_gate_from_metadata() -> None:
    report = _report(
        metadata={"pass_rate_threshold": 0.62, "pass_rate_threshold_source": "threshold"}
    )
    assert _payload(HTMLReporter().to_html(report))["gate"] == {
        "threshold": 0.62,
        "source": "threshold",
    }


def test_gate_defaults_to_all_cases_must_pass() -> None:
    assert _payload(HTMLReporter().to_html(_report()))["gate"] == {
        "threshold": 1.0,
        "source": "default",
    }


def test_gate_default_source_from_cli() -> None:
    report = _report(metadata={"pass_rate_threshold": 1.0, "pass_rate_threshold_source": "default"})
    assert payload_mod.resolve_gate(report) == {"threshold": 1.0, "source": "default"}


def test_explicit_full_gate_keeps_threshold_source() -> None:
    report = _report(
        metadata={"pass_rate_threshold": 1.0, "pass_rate_threshold_source": "threshold"}
    )
    assert payload_mod.resolve_gate(report) == {"threshold": 1.0, "source": "threshold"}


@pytest.mark.parametrize("bad", ["0.5", None, True, 1.5, -0.1, float("nan")])
def test_invalid_gate_falls_back_to_default(bad) -> None:
    report = _report(metadata={"pass_rate_threshold": bad})
    assert payload_mod.resolve_gate(report) == {"threshold": 1.0, "source": "default"}


# ── Baseline ─────────────────────────────────────────────────────────────────


def test_baseline_picks_latest_earlier_report(tmp_path: Path) -> None:
    _save_json(_report([_case("a", [0.2])], ts=1000.0), tmp_path / "set_1_old.json")
    _save_json(_report([_case("a", [0.6])], ts=1500.0), tmp_path / "set_1_mid.json")
    _save_json(_report(ts=2500.0), tmp_path / "set_1_newer.json")  # later than current

    html_path = tmp_path / "set_1_cur.html"
    HTMLReporter().save(_report(ts=2000.0), str(html_path))
    prev = _payload(html_path.read_text(encoding="utf-8"))["prev"]

    assert prev["filename"] == "set_1_mid.json"
    assert prev["timestamp"] == 1500.0
    assert prev["results"][0]["criterion_results"][0]["score"] == 0.6
    assert set(prev) == {"timestamp", "filename", "avg_score", "summary", "results"}
    assert prev["avg_score"] == pytest.approx(0.6)


def test_baseline_excludes_current_run_file_and_equal_timestamp(tmp_path: Path) -> None:
    current = _report(ts=2000.0)
    _save_json(current, tmp_path / "set_1_cur.json")  # written by the current run
    html_path = tmp_path / "set_1_cur.html"
    HTMLReporter().save(current, str(html_path))
    assert _payload(html_path.read_text(encoding="utf-8"))["prev"] is None


def test_baseline_exclude_argument_skips_named_file(tmp_path: Path) -> None:
    _save_json(_report(ts=1000.0), tmp_path / "set_1_a.json")
    _save_json(_report(ts=1500.0), tmp_path / "set_1_b.json")
    got = payload_mod.find_baseline(_report(), tmp_path, exclude=[tmp_path / "set_1_b.json"])
    assert got["filename"] == "set_1_a.json"


def test_baseline_ignores_other_eval_set(tmp_path: Path) -> None:
    _save_json(_report(set_id="other", ts=1900.0), tmp_path / "set_1_other.json")
    _save_json(_report(ts=1000.0), tmp_path / "set_1_ok.json")
    got = payload_mod.find_baseline(_report(), tmp_path)
    assert got["filename"] == "set_1_ok.json"


def test_baseline_ignores_corrupt_and_malformed_files(tmp_path: Path) -> None:
    (tmp_path / "set_1_bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "set_1_list.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "set_1_nosummary.json").write_text(
        json.dumps({"eval_set_id": "set_1", "timestamp": 1900.0, "results": [{}]}),
        encoding="utf-8",
    )
    (tmp_path / "set_1_binary.json").write_bytes(b"\xff\xfe\x00")
    _save_json(_report(ts=1000.0), tmp_path / "set_1_good.json")
    got = payload_mod.find_baseline(_report(), tmp_path)
    assert got["filename"] == "set_1_good.json"


def test_baseline_none_for_missing_or_empty_dir(tmp_path: Path) -> None:
    assert payload_mod.find_baseline(_report(), tmp_path) is None
    assert payload_mod.find_baseline(_report(), tmp_path / "nope") is None


def test_baseline_values_are_coerced(tmp_path: Path) -> None:
    hostile = {
        "eval_set_id": "set_1",
        "timestamp": 1000.0,
        "summary": {"pass_rate": "high", "criterion_stats": {"c": "x", "d": {"avg_score": "z"}}},
        "results": [
            {
                "eval_id": ["<b>"],
                "name": {"a": 1},
                "passed": "yes",
                "duration_seconds": "fast",
                "criterion_results": [{"criterion": 1, "score": "bad", "threshold": None}, 5],
            },
            "junk",
        ],
    }
    (tmp_path / "set_1_hostile.json").write_text(json.dumps(hostile), encoding="utf-8")
    prev = payload_mod.find_baseline(_report(), tmp_path)
    case = prev["results"][0]
    assert isinstance(case["eval_id"], str) and isinstance(case["name"], str)
    assert case["passed"] is False and case["duration_seconds"] == 0.0
    assert case["criterion_results"] == [
        {"criterion": "1", "score": 0.0, "passed": False, "threshold": 0.0}
    ]
    assert prev["summary"]["pass_rate"] == 0.0
    assert set(prev["summary"]["criterion_stats"]) == {"d"}


def test_baseline_keeps_zero_durations_for_not_comparable_handling(tmp_path: Path) -> None:
    _save_json(
        _report([_case("a", [1.0], duration_seconds=0.0)], ts=1000.0), tmp_path / "set_1_z.json"
    )
    prev = payload_mod.find_baseline(_report(), tmp_path)
    assert prev["results"][0]["duration_seconds"] == 0.0


def test_manager_run_links_second_report_to_first(tmp_path: Path) -> None:
    config = ReporterConfig(
        output_dir=str(tmp_path), html=True, json_report=True, console=False, timestamp_files=True
    )
    first = ReporterManager(config).run_all(_report(ts=1000.0))
    second = ReporterManager(config).run_all(_report(ts=2000.0))
    assert first.html_path and second.html_path
    assert _payload(Path(first.html_path).read_text(encoding="utf-8"))["prev"] is None
    prev = _payload(Path(second.html_path).read_text(encoding="utf-8"))["prev"]
    assert prev["timestamp"] == 1000.0
    assert prev["filename"] == Path(first.json_path).name


# ── Escaping ─────────────────────────────────────────────────────────────────

_SCRIPT_BREAK = "</script><script>alert(1)</script>"
_IMG = "<img src=x onerror=alert(1)>"
_LS, _PS = chr(0x2028), chr(0x2029)


def test_hostile_strings_are_embedded_inert(tmp_path: Path) -> None:
    case = _case(
        "x</script><b>",
        [0.5],
        name="<!-- name -->" + _IMG,
        actual_response=f"{_SCRIPT_BREAK}\n{_IMG}\n&amp; {_LS} {_PS} <!--",
        messages=[{"role": "user", "content": _SCRIPT_BREAK}],
        metadata={"note": _IMG},
    )
    out = tmp_path / "r.html"
    HTMLReporter().save(_report([case]), str(out))
    html = out.read_text(encoding="utf-8")

    # Nothing hostile survives as markup anywhere in the file.
    assert _SCRIPT_BREAK not in html
    assert _IMG not in html
    assert "<script>alert" not in html
    assert "onerror=alert(1)>" not in html
    # Only our two script elements exist and the data element is not broken early.
    assert len(re.findall(r"<script\b", html)) == 2
    assert len(re.findall(r"</script>", html)) == 2
    data_text = _DATA_RE.search(html).group(1)
    for ch in ("<", ">", "&", _LS, _PS):
        assert ch not in data_text
    # The data still round-trips exactly.
    result = _payload(html)["cur"]["results"][0]
    assert result["eval_id"] == "x</script><b>"
    assert result["actual_response"].startswith(_SCRIPT_BREAK)
    assert _IMG in result["actual_response"]
    assert result["messages"][0]["content"] == _SCRIPT_BREAK


def test_hostile_title_is_escaped() -> None:
    report = _report()
    report.eval_set_name = "</title><script>alert(1)</script>"
    html = HTMLReporter().to_html(report)
    assert "<title></title>" not in html
    assert "<script>alert(1)" not in html
    assert "&lt;/title&gt;" in html


def test_page_script_escapes_quotes_and_keeps_judge_fallback_note() -> None:
    # One escaper covers quotes too; the simulation judge fallback warning is kept.
    assert "\"'\":'&#39;'" in JS_CONTENT
    assert "no llm provider" in JS_CONTENT
