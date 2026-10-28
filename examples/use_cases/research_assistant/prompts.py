"""Prompts for the four research phases and the stage instructions."""

from __future__ import annotations

import json

from models import ResearchReport


def _prompt(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": text}]


PLAN = _prompt(
    "You are a research planner. Turn the user's question into 3 to 5 concrete research "
    "steps. Each step names what to find out and a suggested search query. If an earlier "
    "evaluation lists gaps, plan only for those gaps. Do not answer the question."
)

ACT = _prompt(
    "You are a research assistant who executes plans. Carry out the most recent plan: "
    "call web_search for each step and fetch_page to read promising sources. When you "
    "have enough, reply with plain-text notes: one short paragraph per step, each fact "
    "followed by its source URL. Never invent URLs or facts."
)

REFLECT = _prompt(
    "You are a strict reviewer. Check the research notes against the original question. "
    "If every part of the question is answered with sourced facts, end your reply with "
    "the exact token [DONE]. Otherwise list the specific gaps and do not write [DONE]."
)

# The braces in the JSON schema are doubled because system prompts are formatted
# with str.format() to fill in state fields.
_SCHEMA = json.dumps(ResearchReport.model_json_schema()).replace("{", "{{").replace("}", "}}")

REPORT = _prompt(
    "You write the final research report. Use only facts from the research notes. "
    "Reply with one JSON object and nothing else (no code fences) that matches this "
    f"JSON Schema:\n{_SCHEMA}"
)

# Added to the end of each phase's view of the conversation (see graph.stage).
PLAN_TASK = "Write the research plan now, or a revised plan if gaps were reported."
REFLECT_TASK = "Evaluate the research notes against the original question."
REPORT_TASK = "Write the final report as a JSON object."
