"""
Shared LLM calling utilities for evaluation criteria.

Provides LLMCallerMixin with _call_llm_score() used by
LLMJudgeCriterion, RubricBasedCriterion, and others.

Delegates to tenxgraph.core.llm.caller.call_llm so that all single-turn
provider dispatch lives in one place.
"""

from __future__ import annotations

import json
import logging
import os

from tenxgraph.core.llm.caller import call_llm
from tenxgraph.core.llm.client_factory import detect_provider
from tenxgraph.qa.evaluation.token_usage import TokenUsage


logger = logging.getLogger("tenxgraph.evaluation")


def _resolve_use_vertex_ai(config: object) -> bool:
    """Decide whether the judge LLM should go through Vertex AI.

    Honours an explicit ``use_vertex_ai`` on the criterion config when present,
    otherwise falls back to the ``GOOGLE_GENAI_USE_VERTEXAI`` environment
    variable — the same signal the agent runtime uses. Without this, the judge
    always defaulted to the Gemini Developer API even when the project is only
    reachable via Vertex, producing 401/403 auth failures.
    """
    explicit = getattr(config, "use_vertex_ai", None)
    if explicit is not None:
        return bool(explicit)
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").strip().lower() == "true"


def _parse_model_provider(model: str, use_vertex_ai: bool = False) -> tuple[str, str]:
    """Return (provider, model_name_without_prefix)."""
    provider = detect_provider(model, use_vertex_ai=use_vertex_ai)
    model_name = model.split("/", 1)[-1] if "/" in model else model
    return provider, model_name


class LLMCallerMixin:
    """Mixin providing shared LLM calling logic for LLM-based criteria.

    Reads ``self.config.judge_model`` to determine provider and model.
    """

    async def _call_google_json(self, prompt: str) -> tuple[dict, TokenUsage]:
        """Call Google LLM and return (parsed JSON dict, token usage).

        Raises the underlying exception on failure so the criterion reports a real error.
        """
        judge_model: str = self.config.judge_model  # type: ignore[attr-defined]
        use_vertex_ai: bool = _resolve_use_vertex_ai(self.config)
        text, inp, out, cache = await call_llm(
            judge_model,
            prompt,
            temperature=0.3,
            json_mode=True,
            use_vertex_ai=use_vertex_ai,
        )
        if not text:
            raise RuntimeError(f"Judge model {judge_model} returned an empty response")
        data = json.loads(text)
        usage = TokenUsage(input_tokens=inp, output_tokens=out, cache_read_tokens=cache)
        return data, usage

    async def _call_openai_json(self, prompt: str) -> tuple[dict, TokenUsage]:
        """Call OpenAI LLM and return (parsed JSON dict, token usage).

        Raises the underlying exception on failure so the criterion reports a real error.
        """
        judge_model: str = self.config.judge_model  # type: ignore[attr-defined]
        api_style: str = getattr(self.config, "api_style", "responses")  # type: ignore[attr-defined]
        text, inp, out, cache = await call_llm(
            judge_model,
            prompt,
            temperature=0.3,
            json_mode=True,
            api_style=api_style,  # type: ignore[arg-type]
        )
        if not text:
            raise RuntimeError(f"Judge model {judge_model} returned an empty response")
        data = json.loads(text)
        usage = TokenUsage(input_tokens=inp, output_tokens=out, cache_read_tokens=cache)
        return data, usage

    async def _call_llm_json(self, prompt: str) -> tuple[dict, TokenUsage]:
        """Call the judge LLM and return (parsed JSON dict, token usage)."""
        judge_model: str = self.config.judge_model  # type: ignore[attr-defined]
        use_vertex_ai: bool = _resolve_use_vertex_ai(self.config)
        provider, _ = _parse_model_provider(judge_model, use_vertex_ai=use_vertex_ai)

        # LLM failures propagate so the criterion is reported as errored, never scored.
        if provider == "google":
            return await self._call_google_json(prompt)
        if provider == "openai":
            return await self._call_openai_json(prompt)

        raise RuntimeError(f"No LLM provider available for model: {judge_model}")

    async def _call_llm_score(self, prompt: str) -> tuple[float, str, TokenUsage]:
        """Call the judge LLM and return (score, reasoning, token_usage)."""
        result, usage = await self._call_llm_json(prompt)
        return float(result.get("score", 0.0)), result.get("reasoning", ""), usage
