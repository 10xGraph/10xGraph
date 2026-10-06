"""Configuration models for multimodal content handling."""

from enum import Enum

from pydantic import BaseModel


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
        image_handling: How to send images to the provider.
        document_handling: How to process documents before sending.
        max_image_size_mb: Maximum allowed image size in megabytes.
        max_image_dimension: Resize images if either dimension exceeds this.
        supported_image_types: Allowed image MIME types.
        supported_doc_types: Allowed document MIME types.
    """

    image_handling: ImageHandling = ImageHandling.BASE64
    document_handling: DocumentHandling = DocumentHandling.EXTRACT_TEXT
    max_image_size_mb: float = 10.0
    max_image_dimension: int = 2048
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
