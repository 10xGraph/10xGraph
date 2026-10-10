"""Internal media URI scheme helpers.

New URIs are written with ``graph://media/{key}``. The legacy ``agentflow://media/{key}``
prefix is still accepted by every reader so previously stored data keeps resolving.
"""

MEDIA_SCHEME = "graph://media/"
LEGACY_MEDIA_SCHEME = "agentflow://media/"
MEDIA_SCHEMES: tuple[str, ...] = (MEDIA_SCHEME, LEGACY_MEDIA_SCHEME)


def is_internal_media_url(url: str | None) -> bool:
    """Return True if ``url`` uses the current or legacy internal media scheme."""
    return bool(url) and url.startswith(MEDIA_SCHEMES)  # type: ignore[union-attr]


def strip_media_scheme(url: str) -> str:
    """Return the storage key of an internal media URL (current or legacy scheme)."""
    for scheme in MEDIA_SCHEMES:
        if url.startswith(scheme):
            return url[len(scheme) :]
    return url
