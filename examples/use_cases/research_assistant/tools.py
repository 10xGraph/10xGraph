"""Research tools for the ACT phase.

Both wrap prebuilt tools from `tenxgraph.prebuilt.tools` and only shrink the
output, so a few fetched pages do not fill the context window.

`google_web_search` calls Gemini with Google Search grounding through its own
google-genai client. That is independent of the model running the ACT phase, so
it works with an OpenAI model. It needs GEMINI_API_KEY or GOOGLE_API_KEY.
To use another search service, replace `web_search` with any async function
that returns text, for example one that calls the Tavily or Brave API.
"""

from __future__ import annotations

from tenxgraph.prebuilt.tools import fetch_url, google_web_search


async def web_search(query: str) -> str:
    """Search the web and return a grounded summary with sources as JSON."""
    return await google_web_search(query, max_chars=5000)


async def fetch_page(url: str) -> str:
    """Download a public web page and return its text (truncated) as JSON."""
    return await fetch_url(url, max_chars=5000)


RESEARCH_TOOLS = [web_search, fetch_page]
