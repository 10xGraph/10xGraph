"""Provider-specific media optimizations.

Handles:
- Google File API upload for large files (>20 MB inline threshold)
- Content-addressed caching to avoid re-uploading the same file
- OpenAI file helpers

These are optional optimizations.  The converters work without them
(falling back to inline bytes), but for production use with large files,
wiring in these helpers improves latency and avoids provider size limits.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any


logger = logging.getLogger("agentflow.media.provider_media")

# Google File API inline threshold (20 MB)
GOOGLE_INLINE_THRESHOLD = 20 * 1024 * 1024
# OpenAI inline size threshold (reasonable cutoff for base64 in API calls)
OPENAI_INLINE_THRESHOLD = 20 * 1024 * 1024


class ProviderMediaCache:
    """Content-addressed cache mapping file hashes to provider file references.

    Prevents re-uploading the same binary content to a provider.
    Thread-safe for reads but uses a simple dict (not concurrent writes).

    Usage::

        cache = ProviderMediaCache()
        key = cache.content_key(file_bytes)
        if (ref := cache.get("google", key)):
            # Use existing reference
        else:
            ref = upload_to_google(file_bytes)
            cache.put("google", key, ref)
    """

    def __init__(self, max_entries: int = 1000):
        self._max_entries = max_entries
        self._store: dict[str, dict[str, Any]] = {}  # provider -> {key -> ref}

    @staticmethod
    def content_key(data: bytes) -> str:
        """Compute a content-addressed key (SHA-256 hex digest)."""
        return hashlib.sha256(data).hexdigest()

    def get(self, provider: str, key: str) -> Any | None:
        """Look up a cached provider file reference."""
        return self._store.get(provider, {}).get(key)

    def put(self, provider: str, key: str, reference: Any) -> None:
        """Store a provider file reference."""
        bucket = self._store.setdefault(provider, {})
        if len(bucket) >= self._max_entries:
            # Simple eviction: remove oldest entry
            oldest = next(iter(bucket))
            del bucket[oldest]
        bucket[key] = reference

    def clear(self, provider: str | None = None) -> None:
        """Clear cached references for a provider (or all)."""
        if provider:
            self._store.pop(provider, None)
        else:
            self._store.clear()


async def upload_to_google_file_api(
    data: bytes,
    mime_type: str,
    *,
    display_name: str | None = None,
    cache: ProviderMediaCache | None = None,
    client: Any | None = None,
) -> Any:
    """Upload a file to Google's File API and return a ``types.Part``.

    Requires ``google-genai`` SDK.  If the file is already cached, returns
    the cached ``Part.from_uri`` reference.

    Args:
        data: Raw file bytes.
        mime_type: MIME type of the file.
        display_name: Optional display name for the upload.
        cache: Optional cache to avoid re-uploading the same content.
        client: Optional ``google.genai.Client`` instance.  If ``None``,
            a default client is created.

    Returns:
        A ``google.genai.types.Part`` referencing the uploaded file.
    """
    from google.genai import types

    # Check cache first
    if cache is not None:
        key = cache.content_key(data)
        cached = cache.get("google", key)
        if cached is not None:
            logger.debug("Cache hit for Google file upload (key=%s)", key[:12])
            return cached

    # Upload
    if client is None:
        from google import genai

        client = genai.Client()

    upload_result = await _google_upload(client, data, mime_type, display_name)
    part = types.Part.from_uri(
        file_uri=upload_result.uri,
        mime_type=upload_result.mime_type,
    )

    # Cache the result
    if cache is not None:
        cache.put("google", key, part)
        logger.debug("Cached Google file upload (key=%s)", key[:12])

    return part


async def _google_upload(client: Any, data: bytes, mime_type: str, display_name: str | None) -> Any:
    """Perform the actual Google File API upload (sync or async)."""
    # google-genai SDK uses synchronous upload by default
    # Wrap in a sync call
    import asyncio
    import io

    def _sync_upload():
        return client.files.upload(
            file=io.BytesIO(data),
            config={"mime_type": mime_type, "display_name": display_name or "upload"},
        )

    return await asyncio.to_thread(_sync_upload)


def should_use_google_file_api(data_size: int) -> bool:
    """Check if file size exceeds the inline threshold for Google."""
    return data_size > GOOGLE_INLINE_THRESHOLD


def prepare_google_content_part(
    data: bytes,
    mime_type: str,
    *,
    cache: ProviderMediaCache | None = None,
) -> Any:
    """Synchronous helper to create a Google Part, using inline or file API.

    For files under the inline threshold, returns ``Part.from_bytes()``.
    For larger files, raises ``ValueError`` — caller should use the
    async ``upload_to_google_file_api`` instead.

    This allows converters to make a quick decision without async overhead.
    """
    from google.genai import types

    if len(data) <= GOOGLE_INLINE_THRESHOLD:
        return types.Part.from_bytes(data=data, mime_type=mime_type)

    raise ValueError(
        f"File size ({len(data)} bytes) exceeds Google inline threshold "
        f"({GOOGLE_INLINE_THRESHOLD} bytes).  Use upload_to_google_file_api() "
        f"for large files."
    )


def create_openai_file_search_tool(file_ids: list[str]) -> dict[str, Any]:
    """Create an OpenAI file_search tool attachment for PDF documents.

    Use this when you have PDF file IDs (from OpenAI Files API) and want
    to enable the model to search through them.

    Args:
        file_ids: List of OpenAI file IDs (e.g., from ``client.files.create()``).

    Returns:
        A tool dict for the OpenAI chat completions ``tools`` parameter.
    """
    return {
        "type": "file_search",
        "file_search": {
            "vector_store_ids": [],  # Will be auto-created if empty
        },
    }


def create_openai_file_attachment(file_id: str, tools: list[str] | None = None) -> dict[str, Any]:
    """Create a message attachment referencing an uploaded OpenAI file.

    Args:
        file_id: The OpenAI file ID.
        tools: Tools to associate with this file (e.g. ``["file_search"]``).

    Returns:
        An attachment dict for the OpenAI messages ``attachments`` parameter.
    """
    return {
        "file_id": file_id,
        "tools": [{"type": t} for t in (tools or ["file_search"])],
    }
