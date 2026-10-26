"""Network fetch tools for AgentFlow agents."""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import json
import socket  # noqa: F401  (tests patch fetch.socket)
from html.parser import HTMLParser

from tenxgraph.utils import safe_http
from tenxgraph.utils.decorators import tool


_DEFAULT_MAX_CHARS = 20_000
_USER_AGENT = "10xgraph-prebuilt-tools/1.0"
_HTTP_ERROR_STATUS = 400


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            text = data.strip()
            if text:
                self.parts.append(text)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return safe_http.is_public_ip(ip)


def _resolve_public_ip(hostname: str | None, port: int) -> str | None:
    """Resolve ``hostname`` once and return an address to connect to (``None`` if refused)."""
    return safe_http.resolve_public_ip(hostname, port)


def _html_to_text(html: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(html)
    return parser.text()


def _fetch_sync(url: str, timeout: float, max_chars: int) -> dict[str, object]:
    timeout = max(1.0, min(float(timeout), 30.0))

    # Redirects are followed in safe_http, so every hop is checked
    try:
        result = safe_http.fetch(
            url,
            timeout=timeout,
            max_bytes=max_chars + 1,
            user_agent=_USER_AGENT,
            resolve=lambda host, port: _resolve_public_ip(host, port),
        )
    except safe_http.SafeHTTPError as exc:
        return {"error": str(exc)}
    except (OSError, http.client.HTTPException) as exc:
        return {"error": f"URL error: {exc}"}

    url = result.url
    status_code, content_type, raw = result.status, result.content_type, result.body

    if status_code >= _HTTP_ERROR_STATUS:
        return {"error": f"HTTP error: {status_code}", "status_code": status_code}

    truncated = len(raw) > max_chars
    body = raw[:max_chars].decode("utf-8", errors="replace")
    if "html" in content_type.lower():
        body = _html_to_text(body)
        if len(body) > max_chars:
            body = body[:max_chars]
            truncated = True

    return {
        "url": url,
        "status_code": status_code,
        "content_type": content_type,
        "content": body,
        "truncated": truncated,
    }


@tool(
    name="fetch_url",
    description=(
        "Fetch a public http/https URL and return text content. Blocks private/local hosts, "
        "including through redirects, applies a timeout, and truncates long responses."
    ),
    tags=["web", "fetch", "network"],
    capabilities=["network_access"],
)
async def fetch_url(url: str, timeout: float = 10.0, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Fetch a public URL and return normalized text content."""
    safe_max_chars = max(1, min(int(max_chars), _DEFAULT_MAX_CHARS))
    result = await asyncio.to_thread(_fetch_sync, url, timeout, safe_max_chars)
    return json.dumps(result)
