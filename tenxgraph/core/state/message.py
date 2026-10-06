"""
Message and content block primitives for agent graphs.

This module defines the core message representation, multimodal content blocks,
token usage tracking, and utility functions for agent graph communication.

Classes:
    TokenUsages: Tracks token usage statistics for a message or model response.
    MediaRef: Reference to media content (image/audio/video/document/data).
    AnnotationRef: Reference to annotation metadata.
    TextBlock, ImageBlock, AudioBlock, VideoBlock, DocumentBlock, DataBlock,
        ToolCallBlock, ToolResultBlock, ReasoningBlock, AnnotationBlock, ErrorBlock:
        Multimodal content primitives for message composition.
    Message: Represents a message in a conversation, including content, role, metadata,
        and token usage.

Functions:
    generate_id: Generates a message or tool call ID based on DI context and type.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Sequence
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

import pydantic
from injectq import InjectQ
from pydantic import BaseModel, Field

from agentflow.core.state.message_block import (
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    MediaRef,
    TextBlock,
    ToolResultBlock,
    VideoBlock,
)


logger = logging.getLogger("agentflow.state")


def generate_id(default_id: str | int | None) -> str | int:
    """
    Generate a message or tool call ID based on DI context and type.

    Args:
        default_id (str | int | None): Default ID to use if provided and matches type.

    Returns:
        str | int: Generated or provided ID, type determined by DI context.

    Raises:
        None

    Example:
        >>> generate_id("abc123")
        'abc123'
        >>> generate_id(None)
        'a-uuid-string'
    """
    id_type = InjectQ.get_instance().try_get("generated_id_type", "string")
    generated_id = InjectQ.get_instance().try_get("generated_id", None)

    # if user provided an awaitable, resolve it
    if isinstance(generated_id, Awaitable):

        async def wait_for_id():
            return await generated_id

        generated_id = asyncio.run(wait_for_id())

    if generated_id:
        return generated_id

    if default_id:
        if id_type == "string" and isinstance(default_id, str):
            return default_id
        if id_type in ("int", "bigint") and isinstance(default_id, int):
            return default_id

    # if not matched or default_id is None, generate new id
    logger.debug(
        "Generating new id of type: %s. Default ID not provided or not matched %s",
        id_type,
        default_id,
    )

    if id_type == "int":
        return uuid4().int >> 96  # type: ignore
    if id_type == "bigint":
        return uuid4().int >> 64  # type: ignore
    return str(uuid4())


class TokenUsages(BaseModel):
    """
    Tracks token usage statistics for a message or model response.

    Attributes:
        completion_tokens (int): Number of completion tokens used.
        prompt_tokens (int): Number of prompt tokens used.
        total_tokens (int): Total tokens used.
        reasoning_tokens (int): Reasoning tokens used (optional).
        cache_creation_input_tokens (int): Cache creation input tokens (optional).
        cache_read_input_tokens (int): Cache read input tokens (optional).
        image_tokens (int | None): Image tokens for multimodal models (optional).
        audio_tokens (int | None): Audio tokens for multimodal models (optional).

    Example:
        >>> usage = TokenUsages(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        {'completion_tokens': 10, 'prompt_tokens': 20, 'total_tokens': 30, ...}
    """

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # Optional modality-specific usage fields for multimodal models
    image_tokens: int | None = 0
    audio_tokens: int | None = 0


class Message(BaseModel):
    """
    Represents a message in a conversation, including content, role, metadata, and token usage.

    Attributes:
        message_id (str | int): Unique identifier for the message.
        role (Literal["user", "assistant", "system", "tool"]): The role of the message sender.
        content (Sequence[ContentBlock]): The message content blocks.
        delta (bool): Indicates if this is a delta/partial message.
        tools_calls (list[dict[str, Any]] | None): Tool call information, if any.
        reasoning (str | None): Reasoning or explanation, if any.
        timestamp (datetime | None): Timestamp of the message.
        metadata (dict[str, Any]): Additional metadata.
        usages (TokenUsages | None): Token usage statistics.
        raw (dict[str, Any] | None): Raw data, if any.

    Example:
        >>> msg = Message(message_id="abc123", role="user", content=[TextBlock(text="Hello!")])
        {'message_id': 'abc123', 'role': 'user', 'content': [...], ...}
    """

    message_id: str | int = Field(default_factory=lambda: generate_id(None))
    role: Literal["user", "assistant", "system", "tool"]
    content: Sequence[ContentBlock]
    delta: bool = False  # Indicates if this is a delta/partial message
    tools_calls: list[dict[str, Any]] | None = None
    reasoning: str | None = None  # Remove it
    timestamp: float | None = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: dict[str, Any] = Field(default_factory=dict)
    usages: TokenUsages | None = None
    raw: dict[str, Any] | None = None
    parsed_content: dict | pydantic.BaseModel | None = None

    @classmethod
    def text_message(
        cls,
        content: str,
        role: Literal["user", "assistant", "system", "tool"] = "user",
        message_id: str | None = None,
    ) -> Message:
        """
        Create a Message instance from plain text.

        Args:
            content (str): The message content.
            role (Literal["user", "assistant", "system", "tool"]): The role of the sender.
            message_id (str | None): Optional message ID.

        Returns:
            Message: The created Message instance.

        Example:
            >>> Message.text_message("Hello!", role="user")
        """
        logger.debug("Creating message from text with role: %s", role)
        return cls(
            message_id=generate_id(message_id),
            role=role,
            content=[TextBlock(text=content)],
            metadata={},
        )

    @classmethod
    def tool_message(
        cls,
        content: Sequence[ContentBlock],
        message_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Message:
        """
        Create a tool message, optionally marking it as an error.

        Args:
        content (Sequence[ContentBlock]): The message content blocks.
            message_id (str | None): Optional message ID.
            meta (dict[str, Any] | None): Optional metadata.

        Returns:
            Message: The created tool message instance.

        Example:
            >>> Message.tool_message([ToolResultBlock(...)], message_id="tool1")
        """
        res = content
        msg_id = generate_id(message_id)
        return cls(
            message_id=msg_id,
            role="tool",
            content=res,
            metadata=meta or {},
        )

    # --- Convenience helpers ---
    def text(self) -> str:
        """
        Best-effort text extraction from content blocks.

        Returns:
            str: Concatenated text from TextBlock and ToolResultBlock outputs.

        Example:
            >>> msg.text()
            'Hello!Result text.'
        """
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        if not isinstance(self.content, list):
            return str(self.content)

        parts: list[str] = []
        for block in self.content:
            text = self._block_text(block)
            if text is not None:
                parts.append(text)

        return "".join(parts)

    @staticmethod
    def _block_text(block: Any) -> str | None:
        text: str | None = None
        if isinstance(block, TextBlock):
            text = block.text or ""
        elif isinstance(block, ToolResultBlock):
            text = (
                block.output
                if isinstance(block.output, str)
                else json.dumps(block.output, default=str)
            )
        elif isinstance(block, dict):
            if block.get("type") == "text":
                text = str(block.get("text") or "")
            elif block.get("type") == "tool_result":
                output = block.get("output")
                text = output if isinstance(output, str) else json.dumps(output, default=str)

        return text

    def attach_media(
        self,
        media: MediaRef,
        as_type: Literal["image", "audio", "video", "document"],
    ) -> None:
        """
        Append a media block to the content.

        If content was text, creates a block list. Supports image, audio, video, and document types.

        Args:
            media (MediaRef): Reference to media content.
            as_type (Literal["image", "audio", "video", "document"]): Type of media block to append.

        Returns:
            None

        Raises:
            ValueError: If an unsupported media type is provided.

        Example:
            >>> msg.attach_media(media_ref, as_type="image")
        """
        block: ContentBlock
        if as_type == "image":
            block = ImageBlock(media=media)
        elif as_type == "audio":
            block = AudioBlock(media=media)
        elif as_type == "video":
            block = VideoBlock(media=media)
        elif as_type == "document":
            block = DocumentBlock(media=media)
        else:
            raise ValueError(f"Unsupported media type: {as_type}")

        if isinstance(self.content, str):
            self.content = [TextBlock(text=self.content), block]
        elif isinstance(self.content, list):
            self.content.append(block)
        else:
            self.content = [block]

    # --- Multimodal convenience constructors ---

    @classmethod
    def image_message(
        cls,
        *,
        image_url: str | None = None,
        image_base64: str | None = None,
        mime_type: str = "image/png",
        text: str | None = None,
        role: Literal["user", "assistant", "system", "tool"] = "user",
        message_id: str | None = None,
    ) -> Message:
        """Create a message containing an image and optional text.

        Provide exactly one of *image_url* or *image_base64*.

        Args:
            image_url: URL of the image.
            image_base64: Base64-encoded image data.
            mime_type: MIME type of the image.
            text: Optional text to include alongside the image.
            role: Message role.
            message_id: Optional message ID.

        Returns:
            Message with image content block(s).
        """
        if not image_url and not image_base64:
            raise ValueError("Provide either image_url or image_base64")

        blocks: list[ContentBlock] = []
        if text:
            blocks.append(TextBlock(text=text))

        if image_url:
            media = MediaRef(kind="url", url=image_url, mime_type=mime_type)
        else:
            media = MediaRef(kind="data", data_base64=image_base64, mime_type=mime_type)

        blocks.append(ImageBlock(media=media))
        return cls(
            message_id=generate_id(message_id),
            role=role,
            content=blocks,
        )

    @classmethod
    def multimodal_message(
        cls,
        content_blocks: Sequence[ContentBlock],
        role: Literal["user", "assistant", "system", "tool"] = "user",
        message_id: str | None = None,
    ) -> Message:
        """Create a message from an explicit list of content blocks.

        Args:
            content_blocks: Pre-built content blocks.
            role: Message role.
            message_id: Optional message ID.

        Returns:
            Message with the provided content blocks.
        """
        return cls(
            message_id=generate_id(message_id),
            role=role,
            content=list(content_blocks),
        )

    @classmethod
    def from_file(
        cls,
        file_path: str,
        mime_type: str | None = None,
        text: str | None = None,
        role: Literal["user", "assistant", "system", "tool"] = "user",
        message_id: str | None = None,
    ) -> Message:
        """Create a message from a local file (reads and base64-encodes it).

        Auto-detects the block type (image vs document) from the MIME type.

        Args:
            file_path: Path to the file.
            mime_type: MIME type (auto-detected from extension if None).
            text: Optional text to include alongside the file.
            role: Message role.
            message_id: Optional message ID.

        Returns:
            Message with the appropriate content block.
        """
        import base64
        import mimetypes
        import pathlib

        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if mime_type is None:
            guessed, _ = mimetypes.guess_type(str(path))
            mime_type = guessed or "application/octet-stream"

        data = path.read_bytes()
        b64 = base64.b64encode(data).decode()

        blocks: list[ContentBlock] = []
        if text:
            blocks.append(TextBlock(text=text))

        media = MediaRef(
            kind="data",
            data_base64=b64,
            mime_type=mime_type,
            filename=path.name,
            size_bytes=len(data),
        )

        if mime_type.startswith("image/"):
            blocks.append(ImageBlock(media=media))
        elif mime_type.startswith("audio/"):
            blocks.append(AudioBlock(media=media))
        elif mime_type.startswith("video/"):
            blocks.append(VideoBlock(media=media))
        else:
            blocks.append(DocumentBlock(media=media))

        return cls(
            message_id=generate_id(message_id),
            role=role,
            content=blocks,
        )

    # --- Store-backed convenience helpers ---

    @classmethod
    async def with_image(
        cls,
        data: bytes,
        mime_type: str,
        store: Any,
        text: str | None = None,
        role: Literal["user", "assistant", "system", "tool"] = "user",
        message_id: str | None = None,
    ) -> Message:
        """Create a message with an image stored in a MediaStore.

        The image bytes are persisted to the store and only a lightweight
        ``MediaRef`` reference is kept in the message.

        Args:
            data: Raw image bytes.
            mime_type: MIME type (e.g. ``"image/jpeg"``).
            store: A ``BaseMediaStore`` instance.
            text: Optional text alongside the image.
            role: Message role.
            message_id: Optional message ID.
        """
        key = await store.store(data, mime_type)
        ref = store.to_media_ref(key, mime_type, size_bytes=len(data))

        blocks: list[ContentBlock] = []
        if text:
            blocks.append(TextBlock(text=text))
        blocks.append(ImageBlock(media=ref))

        return cls(
            message_id=generate_id(message_id),
            role=role,
            content=blocks,
        )

    @classmethod
    async def with_file(
        cls,
        file_path: str,
        store: Any,
        mime_type: str | None = None,
        text: str | None = None,
        role: Literal["user", "assistant", "system", "tool"] = "user",
        message_id: str | None = None,
    ) -> Message:
        """Create a message from a local file, stored via a MediaStore.

        Reads the file, stores it in the media store, and creates the
        appropriate content block type based on MIME type.

        Args:
            file_path: Path to the local file.
            store: A ``BaseMediaStore`` instance.
            mime_type: MIME type (auto-detected if ``None``).
            text: Optional text alongside the file.
            role: Message role.
            message_id: Optional message ID.
        """
        import mimetypes as _mt
        import pathlib

        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if mime_type is None:
            guessed, _ = _mt.guess_type(str(path))
            mime_type = guessed or "application/octet-stream"

        data = path.read_bytes()
        key = await store.store(data, mime_type)
        ref = store.to_media_ref(
            key,
            mime_type,
            filename=path.name,
            size_bytes=len(data),
        )

        blocks: list[ContentBlock] = []
        if text:
            blocks.append(TextBlock(text=text))

        if mime_type.startswith("image/"):
            blocks.append(ImageBlock(media=ref))
        elif mime_type.startswith("audio/"):
            blocks.append(AudioBlock(media=ref))
        elif mime_type.startswith("video/"):
            blocks.append(VideoBlock(media=ref))
        else:
            blocks.append(DocumentBlock(media=ref))

        return cls(
            message_id=generate_id(message_id),
            role=role,
            content=blocks,
        )
