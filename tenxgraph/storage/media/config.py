"""Configuration models for multimodal content handling."""

import warnings
from enum import Enum
from typing import Self

from pydantic import BaseModel, model_validator


class ImageHandling(str, Enum):
    """Strategy for sending images to the LLM provider."""

    BASE64 = "base64"
    URL = "url"
    FILE_ID = "file_id"


class DocumentHandling(str, Enum):
    """Strategy for handling document content."""

    EXTRACT_TEXT = "extract_text"
    FORWARD_RAW = "pass_raw"
    SKIP = "skip"


class MultimodalConfig(BaseModel):
    """Per-agent configuration for multimodal content processing.

    Attributes:
        image_handling: Deprecated, no effect, removed in 2.0.
        document_handling: Deprecated, no effect, removed in 2.0.
        max_image_size_mb: Maximum allowed image size in megabytes.
        max_image_dimension: Resize images if either dimension exceeds this.
        max_image_pixels: Refuse images with more pixels than this before decoding
            (``None`` disables the check).
        supported_image_types: Allowed image MIME types.
        supported_doc_types: Deprecated, no effect, removed in 2.0.
    """

    image_handling: ImageHandling = ImageHandling.BASE64
    document_handling: DocumentHandling = DocumentHandling.EXTRACT_TEXT
    max_image_size_mb: float = 10.0
    max_image_dimension: int = 2048
    max_image_pixels: int | None = 50_000_000
    supported_image_types: set[str] = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
    }
    supported_doc_types: set[str] = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    @model_validator(mode="after")
    def _warn_unused_fields(self) -> Self:
        for name in ("image_handling", "document_handling", "supported_doc_types"):
            if getattr(self, name) != type(self).model_fields[name].default:
                warnings.warn(
                    f"MultimodalConfig.{name} has no effect and will be removed in 2.0.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        return self
