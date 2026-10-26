"""MediaFetchPolicy and fetch_media_url (security review Q1, B1)."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tenxgraph.storage.media import MediaFetchPolicy, MediaURLError, fetch_media_url
from tenxgraph.utils import safe_http


PUBLIC_HOST = "public.test"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def _send(self, status: int, body: bytes = b"", **headers: str) -> None:
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # the client may give up (size cap or timeout)
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def do_GET(self) -> None:
        port = self.server.server_address[1]
        if self.path == "/img.png":
            self._send(200, b"PNGDATA", Content_Type="Image/PNG; charset=binary")
        elif self.path == "/page":
            self._send(200, b"<html></html>", Content_Type="text/html")
        elif self.path == "/big.png":
            self._send(200, b"x" * 100, Content_Type="image/png")
        elif self.path == "/no-type":
            self._send(200, b"x")
        elif self.path == "/slow.png":
            time.sleep(1.5)
            self._send(200, b"x", Content_Type="image/png")
        elif self.path == "/missing":
            self._send(404, Content_Type="image/png")
        elif self.path == "/to-internal":
            self._send(302, Location=f"http://127.0.0.1:{port}/img.png")
        elif self.path == "/to-metadata":
            self._send(302, Location="http://169.254.169.254/latest/meta-data/")
        elif self.path == "/to-file":
            self._send(302, Location="file:///etc/passwd")
        elif self.path == "/to-image":
            self._send(302, Location="/img.png")
        elif self.path == "/loop":
            self._send(302, Location="/loop")
        else:
            self._send(404)


@pytest.fixture
def server(monkeypatch) -> Iterator[int]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True)
    thread.start()

    real_resolve = safe_http.resolve_public_ip

    def resolve(hostname, port, *, allow_private=False):
        if hostname == PUBLIC_HOST:
            return "127.0.0.1"
        return real_resolve(hostname, port, allow_private=allow_private)

    monkeypatch.setattr(safe_http, "resolve_public_ip", resolve)
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# check_url (no network)
# ---------------------------------------------------------------------------


class TestCheckUrl:
    def test_public_https_and_http_pass(self) -> None:
        policy = MediaFetchPolicy()
        policy.check_url("https://example.com/a.png")
        policy.check_url("http://example.com/a.png")
        policy.check_url("https://8.8.8.8/a.png")

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/a.png",
            "data:image/png;base64,AAAA",
            "javascript:alert(1)",
            "//example.com/a.png",
            "example.com/a.png",
            "",
            "https:///a.png",
            "https://user:pass@example.com/a.png",
        ],
    )
    def test_bad_urls_rejected(self, url: str) -> None:
        with pytest.raises(MediaURLError):
            MediaFetchPolicy().check_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/a",
            "http://127.0.0.1:6379/",
            "http://10.0.0.5/a",
            "http://192.168.1.1/a",
            "http://[::1]/a",
            "http://[fd00::1]/a",
            "http://[::ffff:127.0.0.1]/a",
            "http://169.254.169.254/latest/meta-data/",
            "http://[fd00:ec2::254]/latest/meta-data/",
            "http://100.100.100.200/",
        ],
    )
    def test_non_public_ip_literals_rejected(self, url: str) -> None:
        with pytest.raises(MediaURLError):
            MediaFetchPolicy().check_url(url)

    def test_scheme_subset(self) -> None:
        policy = MediaFetchPolicy(allowed_schemes=("https",))
        policy.check_url("https://example.com/a.png")
        with pytest.raises(MediaURLError):
            policy.check_url("http://example.com/a.png")

    def test_only_http_schemes_can_be_allowed(self) -> None:
        with pytest.raises(ValueError):
            MediaFetchPolicy(allowed_schemes=("https", "file"))
        with pytest.raises(ValueError):
            MediaFetchPolicy(allowed_schemes=("ftp",))

    def test_disabled(self) -> None:
        with pytest.raises(MediaURLError, match="disabled"):
            MediaFetchPolicy(enabled=False).check_url("https://example.com/a.png")

    def test_host_allowlist_exact_and_wildcard(self) -> None:
        policy = MediaFetchPolicy(allowed_hosts=("cdn.example.com", "*.images.test"))
        policy.check_url("https://cdn.example.com/a.png")
        policy.check_url("https://CDN.Example.com./a.png")
        policy.check_url("https://a.images.test/a.png")
        policy.check_url("https://a.b.images.test/a.png")
        for url in (
            "https://example.com/a.png",
            "https://evil-cdn.example.com/a.png",
            "https://images.test/a.png",
            "https://notimages.test/a.png",
            "https://cdn.example.com.evil.test/a.png",
        ):
            with pytest.raises(MediaURLError):
                policy.check_url(url)

    def test_private_opt_in_only_for_listed_hosts(self) -> None:
        policy = MediaFetchPolicy(
            allowed_hosts=("10.1.2.3", "media.internal"), allow_private_for_allowed_hosts=True
        )
        policy.check_url("http://10.1.2.3/a.png")
        policy.check_url("http://media.internal/a.png")
        with pytest.raises(MediaURLError):
            policy.check_url("http://10.1.2.4/a.png")

    def test_private_opt_in_needs_a_host_list(self) -> None:
        policy = MediaFetchPolicy(allow_private_for_allowed_hosts=True)
        with pytest.raises(MediaURLError):
            policy.check_url("http://10.1.2.3/a.png")

    def test_metadata_never_allowed_even_with_opt_in(self) -> None:
        policy = MediaFetchPolicy(
            allowed_hosts=("169.254.169.254", "fd00:ec2::254"),
            allow_private_for_allowed_hosts=True,
        )
        with pytest.raises(MediaURLError):
            policy.check_url("http://169.254.169.254/latest/meta-data/")
        with pytest.raises(MediaURLError):
            policy.check_url("http://[fd00:ec2::254]/latest/meta-data/")

    def test_gs_needs_an_allowed_bucket(self) -> None:
        with pytest.raises(MediaURLError, match="gs:// bucket not allowed: other"):
            MediaFetchPolicy(allowed_gcs_buckets=("mine",)).check_url("gs://other/a.png")
        with pytest.raises(MediaURLError, match="gs:// bucket not allowed: mine"):
            MediaFetchPolicy().check_url("gs://mine/a.png")
        MediaFetchPolicy(allowed_gcs_buckets=("mine",)).check_url("gs://mine/dir/a.png")

    def test_gs_bucket_match_is_exact(self) -> None:
        policy = MediaFetchPolicy(allowed_gcs_buckets=("mine",))
        for url in ("gs://mine2/a", "gs://my/a", "gs://Mine/a", "gs://mine.evil/a"):
            with pytest.raises(MediaURLError):
                policy.check_url(url)

    def test_disabled_does_not_change_gs_rule(self) -> None:
        MediaFetchPolicy(enabled=False, allowed_gcs_buckets=("mine",)).check_url("gs://mine/a")


# ---------------------------------------------------------------------------
# fetch_media_url
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_bytes_and_mime_without_parameters(server) -> None:
    data, mime = await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/img.png")
    assert data == b"PNGDATA"
    assert mime == "image/png"


@pytest.mark.asyncio
async def test_fetch_follows_a_redirect_to_a_public_host(server) -> None:
    data, _ = await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/to-image")
    assert data == b"PNGDATA"


@pytest.mark.asyncio
async def test_content_type_not_allowed(server) -> None:
    with pytest.raises(MediaURLError, match="content type not allowed: text/html"):
        await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/page")


@pytest.mark.asyncio
async def test_missing_content_type_rejected(server) -> None:
    with pytest.raises(MediaURLError, match="content type"):
        await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/no-type")


@pytest.mark.asyncio
async def test_content_type_wildcard_and_exact(server) -> None:
    policy = MediaFetchPolicy(allowed_content_types=("text/html",))
    data, mime = await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/page", policy)
    assert (data, mime) == (b"<html></html>", "text/html")
    with pytest.raises(MediaURLError):
        await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/img.png", policy)


@pytest.mark.asyncio
async def test_size_cap(server) -> None:
    policy = MediaFetchPolicy(max_bytes=10)
    with pytest.raises(MediaURLError, match="larger than 10 bytes"):
        await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/big.png", policy)
    policy = MediaFetchPolicy(max_bytes=100)
    data, _ = await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/big.png", policy)
    assert len(data) == 100


@pytest.mark.asyncio
async def test_timeout(server) -> None:
    policy = MediaFetchPolicy(timeout=0.3)
    with pytest.raises(MediaURLError, match="download failed"):
        await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/slow.png", policy)


@pytest.mark.asyncio
async def test_http_error_status(server) -> None:
    with pytest.raises(MediaURLError, match="HTTP 404"):
        await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/missing")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/to-internal", "/to-metadata", "/to-file"])
async def test_redirect_to_a_forbidden_target_is_blocked(server, path) -> None:
    with pytest.raises(MediaURLError):
        await fetch_media_url(f"http://{PUBLIC_HOST}:{server}{path}")


@pytest.mark.asyncio
async def test_too_many_redirects(server) -> None:
    with pytest.raises(MediaURLError, match="too many redirects"):
        await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/loop")


@pytest.mark.asyncio
async def test_direct_loopback_and_bad_schemes_never_connect(server) -> None:
    for url in (
        f"http://127.0.0.1:{server}/img.png",
        "file:///etc/passwd",
        "ftp://example.com/a.png",
        "gs://bucket/a.png",
    ):
        with pytest.raises(MediaURLError):
            await fetch_media_url(url)


@pytest.mark.asyncio
async def test_gs_is_not_downloaded_even_when_the_bucket_is_allowed() -> None:
    policy = MediaFetchPolicy(allowed_gcs_buckets=("mine",))
    with pytest.raises(MediaURLError):
        await fetch_media_url("gs://mine/a.png", policy)


@pytest.mark.asyncio
async def test_a_name_resolving_to_an_internal_address_is_refused(monkeypatch) -> None:
    import socket

    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
    monkeypatch.setattr(safe_http.socket, "getaddrinfo", lambda *a, **k: records)
    with pytest.raises(MediaURLError, match="not public"):
        await fetch_media_url("http://rebind.test/img.png")


@pytest.mark.asyncio
async def test_disabled_policy_never_downloads(server) -> None:
    with pytest.raises(MediaURLError, match="disabled"):
        await fetch_media_url(
            f"http://{PUBLIC_HOST}:{server}/img.png", MediaFetchPolicy(enabled=False)
        )


@pytest.mark.asyncio
async def test_private_opt_in_reaches_a_listed_private_host(server) -> None:
    policy = MediaFetchPolicy(allowed_hosts=("127.0.0.1",), allow_private_for_allowed_hosts=True)
    data, mime = await fetch_media_url(f"http://127.0.0.1:{server}/img.png", policy)
    assert (data, mime) == (b"PNGDATA", "image/png")


@pytest.mark.asyncio
async def test_private_opt_in_does_not_reach_unlisted_redirect_targets(server) -> None:
    policy = MediaFetchPolicy(allowed_hosts=(PUBLIC_HOST,), allow_private_for_allowed_hosts=True)
    # /to-internal redirects to 127.0.0.1, which is not in allowed_hosts
    with pytest.raises(MediaURLError):
        await fetch_media_url(f"http://{PUBLIC_HOST}:{server}/to-internal", policy)


def test_metadata_resolution_refused_even_with_allow_private(monkeypatch) -> None:
    import socket

    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))]
    monkeypatch.setattr(safe_http.socket, "getaddrinfo", lambda *a, **k: records)
    assert safe_http.resolve_public_ip("meta.test", 80, allow_private=True) is None


# ---------------------------------------------------------------------------
# Call sites use the policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolvers_use_the_policy() -> None:
    from tenxgraph.storage.media.media_resolver import MediaResolver
    from tenxgraph.storage.media.resolver import MediaRefResolver

    policy = MediaFetchPolicy(enabled=False)
    for resolver in (MediaRefResolver(media_policy=policy), MediaResolver(media_policy=policy)):
        with pytest.raises(MediaURLError):
            await resolver._fetch_external_url("https://example.com/a.png")
    # file:// and internal addresses fail under the default policy too
    for url in ("file:///etc/passwd", "http://127.0.0.1/a.png"):
        with pytest.raises(MediaURLError):
            await MediaRefResolver()._fetch_external_url(url)
        with pytest.raises(MediaURLError):
            await MediaResolver()._fetch_external_url(url)


@pytest.mark.asyncio
async def test_resolver_falls_back_to_the_bound_policy(server) -> None:
    from tenxgraph.core.graph import StateGraph
    from tenxgraph.storage.media.resolver import MediaRefResolver

    graph = StateGraph()
    graph.add_node("a", lambda state: state)
    graph.set_entry_point("a")
    graph.compile(media_policy=MediaFetchPolicy(enabled=False))
    with pytest.raises(MediaURLError, match="disabled"):
        await MediaRefResolver()._fetch_external_url("https://example.com/a.png")


class TestGooglePrefetch:
    @pytest.fixture
    def mixin(self):
        from tenxgraph.core.graph.agent_internal.google import AgentGoogleMixin

        class _Mixin(AgentGoogleMixin):
            pass

        return _Mixin()

    @pytest.mark.asyncio
    async def test_file_and_internal_urls_are_not_read(self, mixin, tmp_path) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("TOP-SECRET")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": secret.as_uri()}},
                    {"type": "image_url", "image_url": {"url": "http://127.0.0.1:9/x.png"}},
                    {"type": "document", "document": {"url": "file:///etc/passwd"}},
                    {"type": "video", "video": {"url": "http://169.254.169.254/latest/"}},
                ],
            }
        ]
        fetched = await mixin._prefetch_google_media(messages)
        assert set(fetched.values()) == {None}
        _, contents = mixin._convert_to_google_format(messages, fetched)
        texts = [p.text for p in contents[0].parts]
        assert texts == [
            "[Failed to load image]",
            "[Failed to load image]",
            "[Failed to load media]",
            "[Failed to load media]",
        ]
        assert all(p.inline_data is None for p in contents[0].parts)

    def test_sync_conversion_never_downloads(self, mixin) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
                ],
            }
        ]
        _, contents = mixin._convert_to_google_format(messages)
        assert contents[0].parts[0].text == "[Failed to load image]"

    @pytest.mark.asyncio
    async def test_prefetch_downloads_off_the_loop_with_the_bound_policy(
        self, mixin, server
    ) -> None:
        url = f"http://{PUBLIC_HOST}:{server}/img.png"
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
        fetched = await mixin._prefetch_google_media(messages)
        assert fetched[url] == (b"PNGDATA", "image/png")
        _, contents = mixin._convert_to_google_format(messages, fetched)
        assert contents[0].parts[0].inline_data.mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_document_content_type_is_checked_by_the_policy(self, mixin, server) -> None:
        url = f"http://{PUBLIC_HOST}:{server}/page"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "document", "document": {"url": url, "mime_type": "application/pdf"}}
                ],
            }
        ]
        fetched = await mixin._prefetch_google_media(messages)
        assert fetched[url] is None

    @pytest.mark.asyncio
    async def test_gs_and_data_urls_pass_through(self, mixin) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "gs://bucket/a.png"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
        assert await mixin._prefetch_google_media(messages) == {}

    @pytest.mark.asyncio
    async def test_failure_log_has_no_query_string(self, mixin, caplog) -> None:
        url = "http://127.0.0.1/a.png?token=SECRET"
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
        with caplog.at_level("WARNING", logger="tenxgraph.agent"):
            await mixin._prefetch_google_media(messages)
        assert "SECRET" not in caplog.text
        assert "127.0.0.1/a.png" in caplog.text


# ---------------------------------------------------------------------------
# compile(media_policy=...)
# ---------------------------------------------------------------------------


def _graph():
    from tenxgraph.core.graph import StateGraph

    graph = StateGraph()
    graph.add_node("a", lambda state: state)
    graph.set_entry_point("a")
    return graph


def test_compile_exposes_the_policy() -> None:
    policy = MediaFetchPolicy(allowed_hosts=("cdn.example.com",), max_bytes=1024)
    compiled = _graph().compile(media_policy=policy)
    assert compiled.media_policy is policy
    with pytest.raises(MediaURLError):
        compiled.media_policy.check_url("https://other.example.com/a.png")


def test_compile_without_policy_exposes_the_default() -> None:
    compiled = _graph().compile()
    assert compiled.media_policy == MediaFetchPolicy()
    assert getattr(compiled, "media_policy", None) is not None


def test_compile_binds_the_policy_in_the_container() -> None:
    from tenxgraph.storage.media.url_policy import resolve_media_policy

    policy = MediaFetchPolicy(max_bytes=123)
    _graph().compile(media_policy=policy)
    assert resolve_media_policy() is policy
    _graph().compile()
    assert resolve_media_policy() == MediaFetchPolicy()


def test_media_policy_property_is_read_only() -> None:
    compiled = _graph().compile()
    with pytest.raises(AttributeError):
        compiled.media_policy = MediaFetchPolicy()  # type: ignore[misc]


def test_prebuilt_agents_pass_the_policy_through() -> None:
    import inspect

    from tenxgraph.prebuilt.agent import (
        RAGAgent,
        ReactAgent,
        SupervisorTeamAgent,
        SwarmAgent,
    )

    for cls in (ReactAgent, RAGAgent, SwarmAgent, SupervisorTeamAgent):
        assert "media_policy" in inspect.signature(cls.compile).parameters
