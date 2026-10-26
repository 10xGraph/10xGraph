"""Network fetch tools for AgentFlow agents."""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import json
import socket
import ssl
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from tenxgraph.utils.decorators import tool


_DEFAULT_MAX_CHARS = 20_000
_USER_AGENT = "10xgraph-prebuilt-tools/1.0"
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_HTTP_ERROR_STATUS = 400
_DEFAULT_PORTS = {"http": 80, "https": 443}
# IPv6 forms that embed an IPv4 address, which ``is_global`` does not unwrap
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")


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
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64 or ip in _IPV4_COMPATIBLE:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.is_global and not ip.is_multicast


def _resolve_public_ip(hostname: str | None, port: int) -> str | None:
    """Resolve ``hostname`` once and return an address to connect to.

    Returns ``None`` when the name does not resolve or any of its addresses is not public,
    so a name that mixes public and internal records is refused.
    """
    if not hostname:
        return None
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return None

    ips = [ipaddress.ip_address(str(addr[4][0]).split("%", 1)[0]) for addr in addresses]
    if not ips or not all(_is_public_ip(ip) for ip in ips):
        return None
    return str(ips[0])


def _open_connection(
    scheme: str, host: str, port: int, ip: str, timeout: float
) -> http.client.HTTPConnection:
    conn: http.client.HTTPConnection
    if scheme == "https":
        conn = http.client.HTTPSConnection(
            host, port, timeout=timeout, context=ssl.create_default_context()
        )
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)

    # Connect to the address that was checked, so DNS cannot point the connection somewhere
    # else between the check and the request. The Host header, SNI and the certificate check
    # still use ``host``.
    def create_connection(address: tuple[str, int], *args: object) -> socket.socket:
        return socket.create_connection((ip, address[1]), *args)  # type: ignore[arg-type]

    conn._create_connection = create_connection  # type: ignore[attr-defined]
    return conn


def _html_to_text(html: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(html)
    return parser.text()


def _get(url: str, timeout: float, max_bytes: int) -> tuple[int, str, str, bytes] | str:
    """GET ``url`` without following redirects.

    Returns ``(status, location, content_type, body)``, or an error message when the URL or
    its host is refused.
    """
    parts = urlsplit(url)
    if parts.scheme not in _DEFAULT_PORTS:
        return "only http and https URLs are supported"
    try:
        port = parts.port or _DEFAULT_PORTS[parts.scheme]
    except ValueError:
        return "URL has an invalid port"
    hostname = parts.hostname
    ip = _resolve_public_ip(hostname, port)
    if hostname is None or ip is None:
        return "URL host is not public or could not be resolved"

    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"

    conn = _open_connection(parts.scheme, hostname, port, ip, timeout)
    try:
        conn.request("GET", target, headers={"User-Agent": _USER_AGENT})
        response = conn.getresponse()
        location = response.getheader("location") or ""
        content_type = response.getheader("content-type") or ""
        body = b"" if response.status in _REDIRECT_STATUSES else response.read(max_bytes)
        return response.status, location, content_type, body
    finally:
        conn.close()


def _fetch_sync(url: str, timeout: float, max_chars: int) -> dict[str, object]:
    timeout = max(1.0, min(float(timeout), 30.0))

    # Redirects are followed here, not by the HTTP client, so every hop is checked
    for _ in range(_MAX_REDIRECTS + 1):
        try:
            result = _get(url, timeout, max_chars + 1)
        except (OSError, http.client.HTTPException) as exc:
            return {"error": f"URL error: {exc}"}
        if isinstance(result, str):
            return {"error": result}

        status_code, location, content_type, raw = result
        if status_code in _REDIRECT_STATUSES and location:
            url = urljoin(url, location)
            continue
        break
    else:
        return {"error": f"too many redirects (more than {_MAX_REDIRECTS})"}

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
