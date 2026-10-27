"""SSRF protection in the prebuilt ``fetch_url`` tool (security review 01-F2, F16)."""

from __future__ import annotations

import ipaddress
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tenxgraph.prebuilt.tools import fetch as fetch_module
from tenxgraph.prebuilt.tools import fetch_url


# A name the tests treat as public. It resolves to the loopback test server.
PUBLIC_HOST = "public.test"


class _Handler(BaseHTTPRequestHandler):
    hosts: list[str] = []
    paths: list[str] = []

    def log_message(self, *args: object) -> None:
        pass

    def _send(self, status: int, body: bytes = b"", **headers: str) -> None:
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        type(self).hosts.append(self.headers["Host"])
        type(self).paths.append(self.path)
        port = self.server.server_address[1]
        if self.path == "/":
            self._send(
                200,
                b"<html><body><h1>Hello</h1><script>skip()</script><p>10xGraph</p></body></html>",
                Content_Type="text/html; charset=utf-8",
            )
        elif self.path == "/to-internal":
            self._send(302, Location=f"http://127.0.0.1:{port}/secret")
        elif self.path == "/to-metadata":
            self._send(301, Location="http://169.254.169.254/latest/meta-data/")
        elif self.path == "/to-file":
            self._send(302, Location="file:///etc/passwd")
        elif self.path == "/to-same-host":
            self._send(302, Location="/")
        elif self.path == "/loop":
            self._send(302, Location="/loop")
        elif self.path == "/secret":
            self._send(200, b"INTERNAL-SECRET", Content_Type="text/plain")
        else:
            self._send(404)


@pytest.fixture
def server(monkeypatch) -> Iterator[int]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.hosts = []
    _Handler.paths = []
    thread = threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True)
    thread.start()

    real_resolve = fetch_module._resolve_public_ip

    def resolve(hostname, port):
        if hostname == PUBLIC_HOST:
            return "127.0.0.1"
        return real_resolve(hostname, port)

    monkeypatch.setattr(fetch_module, "_resolve_public_ip", resolve)
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


async def _fetch(url: str) -> dict:
    return json.loads(await fetch_url(url))


@pytest.mark.asyncio
async def test_fetch_url_returns_normalized_html(server) -> None:
    result = await _fetch(f"http://{PUBLIC_HOST}:{server}/")

    assert result["status_code"] == 200
    assert "Hello" in result["content"]
    assert "10xGraph" in result["content"]
    assert "skip" not in result["content"]


@pytest.mark.asyncio
async def test_fetch_url_connects_to_the_checked_address(server) -> None:
    # public.test does not exist in DNS, so the request only works if the connection goes
    # to the address returned by the check instead of resolving the name again
    result = await _fetch(f"http://{PUBLIC_HOST}:{server}/")

    assert result["status_code"] == 200
    assert _Handler.hosts == [f"{PUBLIC_HOST}:{server}"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/to-internal", "/to-metadata"])
async def test_fetch_url_blocks_redirect_to_internal_address(server, path) -> None:
    result = await _fetch(f"http://{PUBLIC_HOST}:{server}{path}")

    assert result == {"error": "URL host is not public or could not be resolved"}
    assert _Handler.paths == [path]


@pytest.mark.asyncio
async def test_fetch_url_blocks_redirect_to_other_scheme(server) -> None:
    result = await _fetch(f"http://{PUBLIC_HOST}:{server}/to-file")

    assert result == {"error": "only http and https URLs are supported"}


@pytest.mark.asyncio
async def test_fetch_url_follows_redirect_to_a_public_host(server) -> None:
    result = await _fetch(f"http://{PUBLIC_HOST}:{server}/to-same-host")

    assert result["status_code"] == 200
    assert result["url"] == f"http://{PUBLIC_HOST}:{server}/"
    assert "Hello" in result["content"]


@pytest.mark.asyncio
async def test_fetch_url_stops_after_too_many_redirects(server) -> None:
    result = await _fetch(f"http://{PUBLIC_HOST}:{server}/loop")

    assert result == {"error": "too many redirects (more than 5)"}
    assert len(_Handler.hosts) == 6


@pytest.mark.asyncio
async def test_fetch_url_reports_http_errors(server) -> None:
    result = await _fetch(f"http://{PUBLIC_HOST}:{server}/missing")

    assert result == {"error": "HTTP error: 404", "status_code": 404}


@pytest.mark.asyncio
async def test_fetch_url_blocks_direct_loopback(server) -> None:
    result = await _fetch(f"http://127.0.0.1:{server}/secret")

    assert result == {"error": "URL host is not public or could not be resolved"}
    assert _Handler.hosts == []


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",  # shared address space, used to pass
        "100.100.100.200",  # Alibaba Cloud metadata, used to pass
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",  # NAT64 form of 127.0.0.1
        "::127.0.0.1",  # IPv4-compatible form of 127.0.0.1
        "2002:7f00:1::",  # 6to4 form of 127.0.0.1
    ],
)
def test_is_public_ip_rejects_internal_addresses(address) -> None:
    assert fetch_module._is_public_ip(ipaddress.ip_address(address)) is False


@pytest.mark.parametrize(
    "address", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808"]
)
def test_is_public_ip_accepts_public_addresses(address) -> None:
    assert fetch_module._is_public_ip(ipaddress.ip_address(address)) is True


def test_resolve_public_ip_refuses_names_with_any_internal_record(monkeypatch) -> None:
    records = [
        (None, None, None, "", ("8.8.8.8", 80)),
        (None, None, None, "", ("10.0.0.1", 80)),
    ]
    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", lambda *args, **kwargs: records)

    assert fetch_module._resolve_public_ip("mixed.test", 80) is None


def test_resolve_public_ip_returns_the_checked_address(monkeypatch) -> None:
    records = [(None, None, None, "", ("8.8.8.8", 80))]
    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", lambda *args, **kwargs: records)

    assert fetch_module._resolve_public_ip("dns.test", 80) == "8.8.8.8"
