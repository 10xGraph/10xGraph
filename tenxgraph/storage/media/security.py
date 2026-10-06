"""Security utilities for media upload handling.

Provides:
- Magic-bytes validation (verify file content matches claimed MIME type)
- Filename sanitization (prevent path traversal, null bytes, etc.)
- File size enforcement
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Magic bytes signatures for common image/document types
# ---------------------------------------------------------------------------

_MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/gif": [b"GIF87a", b"GIF89a"],
    "image/webp": [b"RIFF"],  # RIFF....WEBP
    "image/bmp": [b"BM"],
    "image/tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
    "application/pdf": [b"%PDF"],
    "application/zip": [b"PK\x03\x04"],
}

# MIME types that need WEBP sub-check (RIFF container)
_RIFF_SUBTYPES = {"image/webp"}
_RIFF_HEADER_LEN = 12


def validate_magic_bytes(data: bytes, claimed_mime: str) -> bool:
    """Check that the raw file bytes match the claimed MIME type.

    Args:
        data: Raw file bytes (at least the first 12 bytes are needed).
        claimed_mime: The MIME type the uploader claims the file is.

    Returns:
        ``True`` if the magic bytes match OR the MIME type is unknown
        (unrecognised MIME types pass by default).
        ``False`` if the data is empty or magic bytes contradict the claim.
    """
    if not data:
        return False

    signatures = _MAGIC_SIGNATURES.get(claimed_mime)
    if signatures is None:
        # Unknown MIME — we can't validate, so pass through
        return True

    header = data[:_RIFF_HEADER_LEN]
    for sig in signatures:
        if header[: len(sig)] == sig:
            # Extra check for RIFF-based formats (WEBP)
            if claimed_mime in _RIFF_SUBTYPES:
                # RIFF....WEBP  — bytes 8-12 should be "WEBP"
                return bool(len(data) >= _RIFF_HEADER_LEN and data[8:12] == b"WEBP")
            return True

    return False


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------

# Characters not allowed in filenames across major OSes
_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_FILENAME_LEN = 255


def sanitize_filename(filename: str) -> str:
    """Sanitise a user-provided filename for safe filesystem storage.

    - Strips path traversal components (``..``, ``/``, ``\\``)
    - Removes null bytes and control characters
    - Normalises Unicode (NFC)
    - Truncates to 255 characters
    - Returns ``"unnamed"`` for empty inputs

    Args:
        filename: Raw filename from user upload.

    Returns:
        A safe filename string.
    """
    if not filename:
        return "unnamed"

    # Normalise unicode
    filename = unicodedata.normalize("NFC", filename)

    # Take only the basename (strip any directory components)
    filename = Path(filename).name

    # Remove path traversal patterns that may survive basename on some OSes
    filename = filename.replace("..", "")

    # Remove unsafe characters
    filename = _UNSAFE_CHARS_RE.sub("", filename)

    # Strip leading/trailing whitespace and dots
    filename = filename.strip(" .")

    if not filename:
        return "unnamed"

    # Truncate preserving extension
    if len(filename) > _MAX_FILENAME_LEN:
        path = Path(filename)
        ext = "".join(path.suffixes)
        name = filename.removesuffix(ext) if ext else path.name
        max_name = _MAX_FILENAME_LEN - len(ext)
        filename = name[:max_name] + ext

    return filename


# ---------------------------------------------------------------------------
# File size enforcement
# ---------------------------------------------------------------------------


def enforce_file_size(data: bytes, max_mb: float) -> None:
    """Raise ``ValueError`` if data exceeds the size limit.

    Args:
        data: Raw file bytes.
        max_mb: Maximum allowed size in megabytes.

    Raises:
        ValueError: If data size exceeds the limit.
    """
    max_bytes = int(max_mb * 1_048_576)
    if len(data) > max_bytes:
        raise ValueError(
            f"File size ({len(data)} bytes) exceeds maximum "
            f"allowed size ({max_bytes} bytes / {max_mb} MB)"
        )
