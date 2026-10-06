from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class MediaRef(BaseModel):
    """
    Reference to media content (image/audio/video/document/data).

    Prefer referencing by URL or provider file_id over inlining base64 for large payloads.

    Attributes:
        kind (Literal["url", "file_id", "data"]): Type of reference.
        url (str | None): URL to media content.
        file_id (str | None): Provider-managed file ID.
        data_base64 (str | None): Base64-encoded data (small payloads only).
        mime_type (str | None): MIME type of the media.
        size_bytes (int | None): Size in bytes.
        sha256 (str | None): SHA256 hash of the media.
        filename (str | None): Filename of the media.
        width (int | None): Image width (if applicable).
        height (int | None): Image height (if applicable).
        duration_ms (int | None): Duration in milliseconds (if applicable).
        page (int | None): Page number (if applicable).
    """

    kind: Literal["url", "file_id", "data"] = "url"
    url: str | None = None  # http(s) or data: URL
    file_id: str | None = None  # provider-managed ID (e.g., OpenAI/Gemini)
    data_base64: str | None = None  # small payloads only
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    filename: str | None = None
    # Media-specific hints
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    page: int | None = None


class AnnotationRef(BaseModel):
    """
    Reference to annotation metadata (e.g., citation, note).

    Attributes:
        url (str | None): URL to annotation source.
        file_id (str | None): Provider-managed file ID.
        page (int | None): Page number (if applicable).
        index (int | None): Index within the annotation source.
        title (str | None): Title of the annotation.
    """

    url: str | None = None
    file_id: str | None = None
    page: int | None = None
    index: int | None = None
    title: str | None = None


class TextBlock(BaseModel):
    """
    Text content block for messages.

    Attributes:
        type (Literal["text"]): Block type discriminator.
        text (str): Text content.
        annotations (list[AnnotationRef]): List of annotation references.
    """

    type: Literal["text"] = "text"
    text: str
    annotations: list[AnnotationRef] = Field(default_factory=list)


class ImageBlock(BaseModel):
    """
    Image content block for messages.

    Attributes:
        type (Literal["image"]): Block type discriminator.
        media (MediaRef): Reference to image media.
        alt_text (str | None): Alternative text for accessibility.
        bbox (list[float] | None): Bounding box coordinates [x1, y1, x2, y2].
    """

    type: Literal["image"] = "image"
    media: MediaRef
    alt_text: str | None = None
    bbox: list[float] | None = None  # [x1,y1,x2,y2] if applicable


class AudioBlock(BaseModel):
    """
    Audio content block for messages.

    Attributes:
        type (Literal["audio"]): Block type discriminator.
        media (MediaRef): Reference to audio media.
        transcript (str | None): Transcript of audio.
        sample_rate (int | None): Sample rate in Hz.
        channels (int | None): Number of audio channels.
    """

    type: Literal["audio"] = "audio"
    media: MediaRef
    transcript: str | None = None
    sample_rate: int | None = None
    channels: int | None = None


class VideoBlock(BaseModel):
    """
    Video content block for messages.

    Attributes:
        type (Literal["video"]): Block type discriminator.
        media (MediaRef): Reference to video media.
        thumbnail (MediaRef | None): Reference to thumbnail image.
    """

    type: Literal["video"] = "video"
    media: MediaRef
    thumbnail: MediaRef | None = None


class DocumentBlock(BaseModel):
    """
    Document content block for messages.

    Attributes:
        type (Literal["document"]): Block type discriminator.
        media (MediaRef): Reference to document media.
        text (str | None): Extracted or provided document text content.
        pages (list[int] | None): List of page numbers.
        excerpt (str | None): Short excerpt/preview from the document.
    """

    type: Literal["document"] = "document"
    media: MediaRef
    text: str | None = None
    pages: list[int] | None = None
    excerpt: str | None = None


class DataBlock(BaseModel):
    """
    Data content block for messages.

    Attributes:
        type (Literal["data"]): Block type discriminator.
        mime_type (str): MIME type of the data.
        data_base64 (str | None): Base64-encoded data.
        media (MediaRef | None): Reference to associated media.
    """

    type: Literal["data"] = "data"
    mime_type: str
    data_base64: str | None = None
    media: MediaRef | None = None


class ToolCallBlock(BaseModel):
    """
    Tool call content block for messages.

    Attributes:
        type (Literal["tool_call"]): Block type discriminator.
        id (str): Tool call ID.
        name (str): Tool name.
        args (dict[str, Any]): Arguments for the tool call.
        tool_type (str | None): Type of tool (e.g., web_search, file_search).
    """

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    tool_type: str | None = None  # e.g., web_search, file_search, computer_use


class RemoteToolCallBlock(BaseModel):
    """
    Remote Tool call content block for messages.

    Attributes:
        type (Literal["remote_tool_call"]): Block type discriminator.
        id (str): Tool call ID.
        name (str): Tool name.
        args (dict[str, Any]): Arguments for the tool call.
        tool_type (str | None): Type of tool (e.g., web_search, file_search).
    """

    type: Literal["remote_tool_call"] = "remote_tool_call"
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    tool_type: str = "remote"


class ToolResultBlock(BaseModel):
    """
    Tool result content block for messages.

    Attributes:
        type (Literal["tool_result"]): Block type discriminator.
        call_id (str): Tool call ID.
        output (Any): Output from the tool (str, dict, MediaRef, or list of blocks).
        is_error (bool): Whether the result is an error.
        status (Literal["completed", "failed"] | None): Status of the tool call.
    """

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    output: Any = None  # str | dict | MediaRef | list[ContentBlock-like]
    is_error: bool = False
    status: Literal["completed", "failed"] | None = None


class ReasoningBlock(BaseModel):
    """
    Reasoning content block for messages.

    Attributes:
        type (Literal["reasoning"]): Block type discriminator.
        summary (str): Summary of reasoning.
        details (list[str] | None): Detailed reasoning steps.
    """

    type: Literal["reasoning"] = "reasoning"
    summary: str
    details: list[str] | None = None


class AnnotationBlock(BaseModel):
    """
    Annotation content block for messages.

    Attributes:
        type (Literal["annotation"]): Block type discriminator.
        kind (Literal["citation", "note"]): Kind of annotation.
        refs (list[AnnotationRef]): List of annotation references.
        spans (list[tuple[int, int]] | None): Spans covered by the annotation.
    """

    type: Literal["annotation"] = "annotation"
    kind: Literal["citation", "note"] = "citation"
    refs: list[AnnotationRef] = Field(default_factory=list)
    spans: list[tuple[int, int]] | None = None


class ErrorBlock(BaseModel):
    """
    Error content block for messages.

    Attributes:
        type (Literal["error"]): Block type discriminator.
        message (str): Error message.
        code (str | None): Error code.
        data (dict[str, Any] | None): Additional error data.
    """

    type: Literal["error"] = "error"
    message: str
    code: str | None = None
    data: dict[str, Any] | None = None


# Discriminated union over the "type" field
ContentBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        AudioBlock,
        VideoBlock,
        DocumentBlock,
        DataBlock,
        ToolCallBlock,
        RemoteToolCallBlock,
        ToolResultBlock,
        ReasoningBlock,
        AnnotationBlock,
        ErrorBlock,
    ],
    Field(discriminator="type"),
]
