"""Google-backed search tools for AgentFlow agents."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from tenxgraph.core.llm.client_factory import get_default_llm_timeout
from tenxgraph.utils.decorators import tool


_DEFAULT_MODEL = "gemini-2.5-flash"
_DEFAULT_MAX_CHARS = 20_000


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_plain(value.model_dump())
    if hasattr(value, "to_json_dict"):
        return _to_plain(value.to_json_dict())
    return str(value)


def _response_payload(response: Any, max_chars: int) -> dict[str, Any]:
    text = getattr(response, "text", "") or ""
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    grounding_metadata = None
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        grounding_metadata = _to_plain(getattr(candidates[0], "grounding_metadata", None))

    return {
        "content": text,
        "grounding_metadata": grounding_metadata,
        "truncated": truncated,
    }


def _google_web_search_sync(query: str, model: str, max_chars: int) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {
            "error": (
                "google-genai is required for google_web_search. "
                "Install with: pip install 10xgraph[google-genai]"
            )
        }

    client = genai.Client(
        http_options=types.HttpOptions(timeout=int(get_default_llm_timeout() * 1000))
    )
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return _response_payload(response, max_chars)


def _vertex_ai_search_sync(
    query: str,
    datastore: str,
    model: str,
    max_chars: int,
) -> dict[str, Any]:
    if not datastore:
        return {"error": "datastore is required"}
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {
            "error": (
                "google-genai is required for vertex_ai_search. "
                "Install with: pip install 10xgraph[google-genai]"
            )
        }

    client = genai.Client(
        http_options=types.HttpOptions(
            api_version="v1", timeout=int(get_default_llm_timeout() * 1000)
        )
    )
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    retrieval=types.Retrieval(
                        vertex_ai_search=types.VertexAISearch(datastore=datastore),
                    )
                )
            ],
        ),
    )
    return _response_payload(response, max_chars)


@tool(
    name="google_web_search",
    description=(
        "Search the public web with Gemini Google Search grounding and return the grounded "
        "answer plus grounding metadata."
    ),
    tags=["web", "search", "google"],
    capabilities=["network_access"],
)
async def google_web_search(
    query: str,
    model: str = _DEFAULT_MODEL,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Search the public web with Gemini Google Search grounding."""
    if not query:
        return json.dumps({"error": "query is required"})
    safe_max_chars = max(1, min(int(max_chars), _DEFAULT_MAX_CHARS))
    result = await asyncio.to_thread(_google_web_search_sync, query, model, safe_max_chars)
    return json.dumps(result)


@tool(
    name="vertex_ai_search",
    description=(
        "Search a configured Vertex AI Search datastore with Gemini grounding. "
        "The datastore must be a full Vertex AI Search datastore resource path."
    ),
    tags=["search", "google", "vertex_ai"],
    capabilities=["network_access"],
)
async def vertex_ai_search(
    query: str,
    datastore: str,
    model: str = _DEFAULT_MODEL,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Search a Vertex AI Search datastore with Gemini grounding."""
    if not query:
        return json.dumps({"error": "query is required"})
    safe_max_chars = max(1, min(int(max_chars), _DEFAULT_MAX_CHARS))
    result = await asyncio.to_thread(
        _vertex_ai_search_sync,
        query,
        datastore,
        model,
        safe_max_chars,
    )
    return json.dumps(result)


def create_google_web_search(model: str = _DEFAULT_MODEL):
    """Create a ``google_web_search`` tool with the model fixed, so the LLM cannot pick it."""

    @tool(
        name="google_web_search",
        description=(
            "Search the public web with Gemini Google Search grounding and return the "
            "grounded answer plus grounding metadata."
        ),
        tags=["web", "search", "google"],
        capabilities=["network_access"],
    )
    async def scoped_google_web_search(query: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
        """Search the public web with Gemini Google Search grounding."""
        return await google_web_search(query, model=model, max_chars=max_chars)

    return scoped_google_web_search


def create_vertex_ai_search(datastore: str, model: str = _DEFAULT_MODEL):
    """Create a ``vertex_ai_search`` tool locked to one datastore and model.

    Prefer this over the module-level ``vertex_ai_search``, whose ``datastore`` and ``model``
    arguments are chosen by the LLM.
    """
    if not datastore:
        raise ValueError("datastore is required")

    @tool(
        name="vertex_ai_search",
        description="Search a configured Vertex AI Search datastore with Gemini grounding.",
        tags=["search", "google", "vertex_ai"],
        capabilities=["network_access"],
    )
    async def scoped_vertex_ai_search(query: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
        """Search the configured Vertex AI Search datastore with Gemini grounding."""
        return await vertex_ai_search(query, datastore=datastore, model=model, max_chars=max_chars)

    return scoped_vertex_ai_search
