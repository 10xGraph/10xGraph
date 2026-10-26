"""Small HTTP client that refuses to talk to internal addresses.

Used wherever the server downloads a URL it did not choose itself (the ``fetch_url`` tool and
client-supplied media links). The rules are fixed:

- only ``http`` and ``https`` URLs
- the host name is resolved once and every address must be public (cloud metadata addresses
  are never allowed, even when private addresses are opted in for a host)
- the connection is pinned to the checked address, so DNS cannot change it afterwards
- redirects are followed here, one hop at a time, and every hop is checked again (max 5)
- no proxy environment variables are read
- the body is capped, and an optional overall deadline applies

The functions are blocking. Call them with ``asyncio.to_thread``.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit


MAX_REDIRECTS = 5
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
DEFAULT_PORTS = {"http": 80, "https": 443}
_READ_CHUNK = 64 * 1024

# IPv6 forms that embed an IPv4 address, which ``is_global`` does not unwrap
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")
# Cloud metadata services: never reachable, not even for a host the developer allowed.
# 169.254.0.0/16 covers AWS, GCP, Azure and the ECS task endpoint.
_METADATA_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.100.100.200/32"),
    ipaddress.ip_network("fd00:ec2::/32"),
)

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class SafeHTTPError(Exception):
    """The request was refused or could not be completed safely."""


@dataclass(frozen=True)
class Response:
    """Final response of :func:`fetch`."""

    url: str
    status: int
    content_type: str
    body: bytes


def _unwrap(ip: IPAddress) -> IPAddress:
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip in _NAT64 or ip in _IPV4_COMPATIBLE:
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip


def is_public_ip(ip: IPAddress) -> bool:
    ip = _unwrap(ip)
    return ip.is_global and not ip.is_multicast


def is_metadata_ip(ip: IPAddress) -> bool:
    ip = _unwrap(ip)
    return any(ip in net for net in _METADATA_NETWORKS if net.version == ip.version)


def resolve_public_ip(
    hostname: str | None, port: int, *, allow_private: bool = False
) -> str | None:
    """Resolve ``hostname`` once and return an address to connect to.

    Returns ``None`` when the name does not resolve or any of its addresses is not public,
    so a name that mixes public and internal records is refused. With ``allow_private`` any
    address is accepted except cloud metadata addresses.
    """
    if not hostname:
        return None
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return None

    ips = [ipaddress.ip_address(str(addr[4][0]).split("%", 1)[0]) for addr in addresses]
    if not ips or any(is_metadata_ip(ip) for ip in ips):
        return None
    if not allow_private and not all(is_public_ip(ip) for ip in ips):
        return None
    return str(ips[0])


def open_connection(
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


def _read_body(
    response: http.client.HTTPResponse, max_bytes: int, truncate: bool, deadline: float | None
) -> bytes:
    if truncate and deadline is None:
        return response.read(max_bytes)

    limit = max_bytes if truncate else max_bytes + 1
    chunks: list[bytes] = []
    total = 0
    while total < limit:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("download timed out")
        chunk = response.read(min(_READ_CHUNK, limit - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if not truncate and total > max_bytes:
        raise SafeHTTPError(f"response is larger than {max_bytes} bytes")
    return b"".join(chunks)


def _get(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    truncate: bool,
    deadline: float | None,
    user_agent: str,
    resolve: Callable[[str | None, int], str | None],
    check_content_type: Callable[[str], None] | None,
) -> tuple[int, str, str, bytes]:
    """GET ``url`` without following redirects: ``(status, location, content_type, body)``."""
    parts = urlsplit(url)
    if parts.scheme not in DEFAULT_PORTS:
        raise SafeHTTPError("only http and https URLs are supported")
    try:
        port = parts.port or DEFAULT_PORTS[parts.scheme]
    except ValueError:
        raise SafeHTTPError("URL has an invalid port") from None
    hostname = parts.hostname
    ip = resolve(hostname, port)
    if hostname is None or ip is None:
        raise SafeHTTPError("URL host is not public or could not be resolved")

    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"

    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("download timed out")
        timeout = min(timeout, remaining)

    conn = open_connection(parts.scheme, hostname, port, ip, timeout)
    try:
        conn.request("GET", target, headers={"User-Agent": user_agent})
        response = conn.getresponse()
        location = response.getheader("location") or ""
        content_type = response.getheader("content-type") or ""
        if response.status in REDIRECT_STATUSES:
            return response.status, location, content_type, b""
        if response.status < http.client.BAD_REQUEST and check_content_type is not None:
            check_content_type(content_type)
        if not truncate:
            declared = response.getheader("content-length") or ""
            if declared.isdigit() and int(declared) > max_bytes:
                raise SafeHTTPError(f"response is larger than {max_bytes} bytes")
        return (
            response.status,
            location,
            content_type,
            _read_body(response, max_bytes, truncate, deadline),
        )
    finally:
        conn.close()


def fetch(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    user_agent: str,
    truncate: bool = True,
    deadline: float | None = None,
    resolve: Callable[[str | None, int], str | None] = resolve_public_ip,
    check_url: Callable[[str], None] | None = None,
    check_content_type: Callable[[str], None] | None = None,
) -> Response:
    """GET ``url``, following redirects by hand so that every hop is checked.

    Args:
        timeout: Socket timeout per request.
        max_bytes: Body cap. With ``truncate`` the body is cut at the cap, otherwise a larger
            response raises :class:`SafeHTTPError`.
        deadline: Optional ``time.monotonic()`` value after which the download is abandoned.
        resolve: Returns the address to connect to for ``(host, port)``, or ``None`` to refuse.
        check_url: Called with every URL (first request and each redirect target) before it
            is requested; it may raise to refuse the hop.
        check_content_type: Called with the Content-Type of the final response before its body
            is read; it may raise to refuse it.

    Raises:
        SafeHTTPError: A URL, host, redirect count or size was refused.
        OSError, http.client.HTTPException: The request itself failed.
    """
    for _ in range(MAX_REDIRECTS + 1):
        if check_url is not None:
            check_url(url)
        status, location, content_type, body = _get(
            url,
            timeout=timeout,
            max_bytes=max_bytes,
            truncate=truncate,
            deadline=deadline,
            user_agent=user_agent,
            resolve=resolve,
            check_content_type=check_content_type,
        )
        if status in REDIRECT_STATUSES and location:
            url = urljoin(url, location)
            continue
        return Response(url=url, status=status, content_type=content_type, body=body)
    raise SafeHTTPError(f"too many redirects (more than {MAX_REDIRECTS})")
