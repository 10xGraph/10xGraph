"""Shared fixtures for the evaluation suite.

One compiled weather agent (OpenAI gpt-5-mini) is shared by every test file.
Live tests are marked with the providers they call and are skipped when the
matching API key is missing, so `pytest examples/evaluation` is safe to run
without keys: only the offline tests execute.
"""

import os

import pytest
from dotenv import load_dotenv
from weather_agent import build_eval_app


load_dotenv()

_KEYS = {
    "needs_openai": ("OPENAI_API_KEY",),
    "needs_anthropic": ("ANTHROPIC_API_KEY",),
    "needs_gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, env_names in _KEYS.items():
            if marker in item.keywords and not any(os.getenv(n) for n in env_names):
                item.add_marker(pytest.mark.skip(reason=f"{' or '.join(env_names)} is not set"))


@pytest.fixture(scope="session")
def eval_app():
    """`(compiled_graph, collector)`, built once per session."""
    return build_eval_app()


@pytest.fixture(scope="session")
def app(eval_app):
    return eval_app[0]


@pytest.fixture(scope="session")
def collector(eval_app):
    return eval_app[1]


def assert_judged(case_result) -> None:
    """Fail when an LLM criterion silently fell back to its placeholder score.

    If the judge model cannot be called, the criterion returns 0.5 with the reasoning
    "No LLM provider available" and does not raise. Checking for it keeps a broken
    judge setup from looking like a mediocre score.
    """
    for cr in case_result.criterion_results:
        reasons = cr.details.get("reasonings", [])
        assert "No LLM provider available" not in reasons, (
            f"{cr.criterion}: the judge model could not be called"
        )
