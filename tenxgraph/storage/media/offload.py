"""Inline data guard — auto-offload large base64 blobs to a MediaStore.

This utility runs at the message ingestion boundary (NOT in the checkpointer).
If a message contains large inline ``data_base64`` and a ``MediaStore`` is
configured, the blob is stored externally and replaced with a lightweight
``agentflow://media/{key}`` reference.
"""

from __future__ import annotations

import base64
import logging
from enum import Enum
from typing import TYPE_CHECKING

from agentflow.core.state.message_block import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    VideoBlock,
)


if TYPE_CHECKING:
    from agentflow.core.state.message import Message
    from agentflow.storage.media.storage.base import BaseMediaStore

logger = logging.getLogger("agentflow.media.offload")


class MediaOffloadPolicy(str, Enum):
    """When to offload inline media to a ``BaseMediaStore``."""

    NEVER = "never"  # Allow inline base64 (testing / small images)
    THRESHOLD = "threshold"  # Offload if decoded size > max_inline_bytes
    ALWAYS = "always"  # Always offload to MediaStore


async def ensure_media_offloaded(
    message: Message,
    store: BaseMediaStore,
    policy: MediaOffloadPolicy = MediaOffloadPolicy.THRESHOLD,
    max_inline_bytes: int = 50_000,
) -> Message:
    """Replace large inline ``data_base64`` blobs with MediaStore references.

    Mutates the message in-place (replaces ``block.media``) and returns it.

    Args:
        message: The message to inspect.
        store: The media store to offload into.
        policy: When to offload.
        max_inline_bytes: Size threshold (decoded) for ``THRESHOLD`` policy.

    Returns:
        The (possibly mutated) message.
    """
    if policy == MediaOffloadPolicy.NEVER:
        return message

    media_block_types = (ImageBlock, AudioBlock, VideoBlock, DocumentBlock)

    for block in message.content:
        if not isinstance(block, media_block_types):
            continue

        media = block.media  # type: ignore[union-attr]
        if media.kind != "data" or not media.data_base64:
            continue

        decoded_size = len(media.data_base64) * 3 // 4  # approximate

        should_offload = policy == MediaOffloadPolicy.ALWAYS or decoded_size > max_inline_bytes

        if not should_offload:
            continue

        data = base64.b64decode(media.data_base64)
        mime = media.mime_type or "application/octet-stream"
        key = await store.store(data, mime)
        new_ref = store.to_media_ref(key, mime)

        # Preserve non-data fields from original MediaRef
        new_ref = new_ref.model_copy(
            update={
                "filename": media.filename,
                "width": media.width,
                "height": media.height,
                "duration_ms": media.duration_ms,
                "size_bytes": len(data),
            }
        )

        block.media = new_ref  # type: ignore[union-attr]
        logger.debug(
            "Offloaded %d bytes (key=%s) from inline base64",
            len(data),
            key,
        )

    return message
