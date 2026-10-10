"""Media processing module for multimodal support.

Document extraction (textxtract) lives in pyagenity-api, not here.
The core library provides config, media processing (images), storage,
resolution, and offloading.
"""

from .config import DocumentHandling, ImageHandling, MultimodalConfig
from .offload import MediaOffloadPolicy, ensure_media_offloaded
from .processor import MediaProcessor
from .provider_media import (
    GOOGLE_INLINE_THRESHOLD,
    ProviderMediaCache,
    create_openai_file_attachment,
    create_openai_file_search_tool,
    should_use_google_file_api,
)
from .resolver import MediaRefResolver
from .security import enforce_file_size, sanitize_filename, validate_magic_bytes
from .storage import BaseMediaStore, CloudMediaStore, InMemoryMediaStore, LocalFileMediaStore


__all__ = [
    "GOOGLE_INLINE_THRESHOLD",
    "BaseMediaStore",
    "CloudMediaStore",
    "DocumentHandling",
    "ImageHandling",
    "InMemoryMediaStore",
    "LocalFileMediaStore",
    "MediaOffloadPolicy",
    "MediaProcessor",
    "MediaRefResolver",
    "MultimodalConfig",
    "ProviderMediaCache",
    "create_openai_file_attachment",
    "create_openai_file_search_tool",
    "enforce_file_size",
    "ensure_media_offloaded",
    "sanitize_filename",
    "should_use_google_file_api",
    "validate_magic_bytes",
]
