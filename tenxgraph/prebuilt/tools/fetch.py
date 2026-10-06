"""Network fetch tools for AgentFlow agents."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from html.parser import HTMLParser
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from agentflow.utils.decorators import tool


_DEFAULT_MAX_CHARS = 20_000
_USER_AGENT = "agentflow-prebuilt-tools/1.0"


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


def _is_public_hostname(hostname: str | None) -> bool:
    if not hostname:
        return False
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for addr in addresses:
        ip = ipaddress.ip_address(addr[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def _html_to_text(html: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(html)
    return parser.text()


def _fetch_sync(url: str, timeout: float, max_chars: int) -> dict[str, object]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return {"error": "only http and https URLs are supported"}
    if not _is_public_hostname(parsed.hostname):
        return {"error": "URL host is not public or could not be resolved"}

    req = request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    try:
        with request.urlopen(  # noqa: S310  # nosec B310
            req, timeout=max(1.0, min(float(timeout), 30.0))
        ) as response:
            raw = response.read(max_chars + 1)
            status_code = response.getcode()
            final_url = response.geturl()
            content_type = response.headers.get("content-type", "")
    except HTTPError as exc:
        return {"error": f"HTTP error: {exc.code}", "status_code": exc.code}
    except URLError as exc:
        return {"error": f"URL error: {exc.reason}"}

    truncated = len(raw) > max_chars
    body = raw[:max_chars].decode("utf-8", errors="replace")
    if "html" in content_type.lower():
        body = _html_to_text(body)
        if len(body) > max_chars:
            body = body[:max_chars]
            truncated = True

    return {
        "url": final_url,
        "status_code": status_code,
        "content_type": content_type,
        "content": body,
        "truncated": truncated,
    }


@tool(
    name="fetch_url",
    description=(
        "Fetch a public http/https URL and return text content. Blocks private/local hosts, "
        "applies a timeout, and truncates long responses."
    ),
    tags=["web", "fetch", "network"],
    capabilities=["network_access"],
)
async def fetch_url(url: str, timeout: float = 10.0, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Fetch a public URL and return normalized text content."""
    safe_max_chars = max(1, min(int(max_chars), _DEFAULT_MAX_CHARS))
    result = await asyncio.to_thread(_fetch_sync, url, timeout, safe_max_chars)
    return json.dumps(result)
