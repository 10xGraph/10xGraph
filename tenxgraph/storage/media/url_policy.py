"""Policy and safe download for media URLs that a client sends.

OpenAI and Anthropic fetch a media URL themselves. On some paths (Google GenAI, the Google
File API, inline bytes) this server downloads it instead, so the URL is untrusted input.
``MediaFetchPolicy`` says what may be downloaded; ``fetch_media_url`` does the download.

Rules that cannot be turned off: only http and https are ever downloaded, cloud metadata
addresses are never reached, every redirect is checked again (max 5), the connection is
pinned to the checked address, and a size cap and a timeout always apply.

Set a policy in code with ``StateGraph.compile(media_policy=...)``. The default policy
applies when none is set.
"""

from __future__ import annotations

import asyncio
import fnmatch
import http.client
import ipaddress
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from tenxgraph.utils import safe_http


logger = logging.getLogger("tenxgraph.media.url_policy")

_USER_AGENT = "10xgraph-media/1.0"
_HARD_SCHEMES = frozenset({"http", "https"})


class MediaURLError(ValueError):
    """A media URL is refused by the policy or could not be downloaded."""


def redact_url(url: str) -> str:
    """Return ``url`` without its query string, fragment and credentials, for logs."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if not parts.scheme or not host:
            return "<invalid url>"
        if ":" in host:
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
        return f"{parts.scheme}://{host}{parts.path}"
    except ValueError:
        return "<invalid url>"


def _host_matches(host: str, patterns: tuple[str, ...]) -> bool:
    host = host.lower().rstrip(".")
    for raw in patterns:
        pattern = raw.lower().rstrip(".")
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]) and len(host) > len(pattern) - 1:
                return True
        elif host == pattern:
            return True
    return False


@dataclass(frozen=True)
class MediaFetchPolicy:
    """What the server may download when a message carries an external media URL.

    Attributes:
        enabled: ``False`` means the server never downloads external URLs. Passing a URL on
            to a provider that fetches it itself (OpenAI, Anthropic) is not affected.
        allowed_schemes: Subset of ``("https", "http")``. No other scheme can be allowed.
        allowed_hosts: Empty allows any public host. Entries are exact host names or
            ``"*.example.com"`` (subdomains only).
        allow_private_for_allowed_hosts: On-prem opt-in. Hosts listed in ``allowed_hosts``
            may resolve to private addresses. Cloud metadata addresses stay blocked.
        allowed_content_types: Content-Type values accepted, ``"image/*"`` style wildcards
            allowed.
        max_bytes: Largest download.
        timeout: Seconds for the whole download, redirects included.
        allowed_gcs_buckets: ``gs://`` buckets a client may reference. The server never
            downloads ``gs://`` itself (Vertex AI reads it with the server's credentials), so
            the list is enforced by :meth:`check_url` only. Empty rejects every ``gs://`` URL.
    """

    enabled: bool = True
    allowed_schemes: tuple[str, ...] = ("https", "http")
    allowed_hosts: tuple[str, ...] = ()
    allow_private_for_allowed_hosts: bool = False
    allowed_content_types: tuple[str, ...] = ("image/*", "application/pdf")
    max_bytes: int = 20 * 1024 * 1024
    timeout: float = 10.0
    allowed_gcs_buckets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        bad = [s for s in self.allowed_schemes if str(s).lower() not in _HARD_SCHEMES]
        if bad:
            raise ValueError(f"allowed_schemes may only contain http and https, got: {bad}")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def _host_allowed(self, host: str) -> bool:
        return not self.allowed_hosts or _host_matches(host, self.allowed_hosts)

    def allows_private(self, host: str) -> bool:
        """True when ``host`` is listed in ``allowed_hosts`` and the private opt-in is on."""
        return (
            self.allow_private_for_allowed_hosts
            and bool(self.allowed_hosts)
            and _host_matches(host, self.allowed_hosts)
        )

    def check_url(self, url: str) -> None:
        """Check ``url`` without touching the network.

        Looks at the enabled flag, scheme, host allowlist and, for an IP literal, whether the
        address may be reached. ``gs://`` URLs are checked against ``allowed_gcs_buckets``.
        Names are resolved, and their addresses checked, at download time.

        Raises:
            MediaURLError: The URL is refused.
        """
        if not isinstance(url, str) or not url:
            raise MediaURLError("media URL is empty")
        try:
            parts = urlsplit(url)
            host = parts.hostname
            parts.port  # noqa: B018  (raises ValueError on a bad port)
        except ValueError as exc:
            raise MediaURLError("media URL is not valid") from exc

        scheme = parts.scheme.lower()
        if scheme == "gs":
            bucket = parts.netloc
            if bucket not in self.allowed_gcs_buckets:
                raise MediaURLError(f"gs:// bucket not allowed: {bucket}")
            return

        if not self.enabled:
            raise MediaURLError("downloading external media URLs is disabled")
        if scheme not in _HARD_SCHEMES or scheme not in {s.lower() for s in self.allowed_schemes}:
            raise MediaURLError(f"media URL scheme not allowed: {scheme or '(none)'}")
        if not host:
            raise MediaURLError("media URL has no host")
        if parts.username is not None or parts.password is not None:
            raise MediaURLError("media URL must not contain credentials")
        if not self._host_allowed(host):
            raise MediaURLError(f"media URL host not allowed: {host}")

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return
        if safe_http.is_metadata_ip(ip):
            raise MediaURLError("media URL points to a cloud metadata address")
        if not safe_http.is_public_ip(ip) and not self.allows_private(host):
            raise MediaURLError("media URL points to a non-public address")

    def _content_type_allowed(self, content_type: str) -> bool:
        mime = content_type.split(";", 1)[0].strip().lower()
        return bool(mime) and any(
            fnmatch.fnmatchcase(mime, pattern.strip().lower())
            for pattern in self.allowed_content_types
        )


def resolve_media_policy() -> MediaFetchPolicy:
    """Return the policy bound by ``compile(media_policy=...)``, or the default."""
    try:
        from injectq import InjectQ

        bound = InjectQ.get_instance().try_get(MediaFetchPolicy)
    except Exception:
        bound = None
    return bound if isinstance(bound, MediaFetchPolicy) else MediaFetchPolicy()


def _download(url: str, policy: MediaFetchPolicy) -> tuple[bytes, str]:
    def resolve(host: str | None, port: int) -> str | None:
        return safe_http.resolve_public_ip(
            host, port, allow_private=bool(host) and policy.allows_private(host or "")
        )

    def check_content_type(content_type: str) -> None:
        if not policy._content_type_allowed(content_type):
            raise MediaURLError(
                f"content type not allowed: {content_type.split(';', 1)[0].strip() or '(none)'}"
            )

    try:
        result = safe_http.fetch(
            url,
            timeout=policy.timeout,
            max_bytes=policy.max_bytes,
            user_agent=_USER_AGENT,
            truncate=False,
            deadline=time.monotonic() + policy.timeout,
            resolve=resolve,
            check_url=policy.check_url,
            check_content_type=check_content_type,
        )
    except MediaURLError:
        raise
    except safe_http.SafeHTTPError as exc:
        raise MediaURLError(str(exc)) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise MediaURLError(f"download failed: {type(exc).__name__}") from exc

    if result.status >= http.client.BAD_REQUEST:
        raise MediaURLError(f"download failed: HTTP {result.status}")
    if not 200 <= result.status < 300:  # noqa: PLR2004
        raise MediaURLError(f"download failed: unexpected HTTP {result.status}")
    mime = result.content_type.split(";", 1)[0].strip().lower()
    return result.body, mime


async def fetch_media_url(url: str, policy: MediaFetchPolicy | None = None) -> tuple[bytes, str]:
    """Download an external media URL under ``policy`` and return ``(bytes, mime_type)``.

    Runs ``policy.check_url``, then downloads off the event loop: the host is resolved and
    must be public, the connection is pinned to that address, each redirect is checked again
    (max 5), the size cap and timeout apply, and the response Content-Type must match
    ``allowed_content_types``. Uses the default policy when ``policy`` is ``None``.

    Raises:
        MediaURLError: The URL is refused or the download failed.
    """
    policy = policy or MediaFetchPolicy()
    policy.check_url(url)
    if urlsplit(url).scheme.lower() not in _HARD_SCHEMES:
        raise MediaURLError("only http and https URLs can be downloaded")
    return await asyncio.to_thread(_download, url, policy)
