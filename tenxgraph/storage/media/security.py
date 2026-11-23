"""Security utilities for media upload handling.

Provides:
- Magic-bytes validation (verify file content matches claimed MIME type)
- Filename sanitization (prevent path traversal, null bytes, etc.)
- File size enforcement
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from pathlib import Path


# ---------------------------------------------------------------------------
# Magic bytes signatures
# ---------------------------------------------------------------------------

_SYNC_MASK = 0xE0


def _prefix(*signatures: bytes) -> Callable[[bytes], bool]:
    return lambda data: data.startswith(signatures)


def _riff(kind: bytes) -> Callable[[bytes], bool]:
    return lambda data: data[:4] == b"RIFF" and data[8:12] == kind


def _mp3(data: bytes) -> bool:
    # An ID3 tag, or an MPEG audio frame sync (11 set bits).
    return data.startswith(b"ID3") or (
        data[:1] == b"\xff" and len(data) > 1 and (data[1] & _SYNC_MASK) == _SYNC_MASK
    )


def _iso_media(data: bytes) -> bool:
    return data[4:8] == b"ftyp"


def _pdf(data: bytes) -> bool:
    # Readers accept the header anywhere in the first KiB.
    return b"%PDF-" in data[:1024]


_ZIP = _prefix(b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_OLE = _prefix(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
_OGG = _prefix(b"OggS")
_EBML = _prefix(b"\x1a\x45\xdf\xa3")
_ICO = _prefix(b"\x00\x00\x01\x00")
_JPEG = _prefix(b"\xff\xd8\xff")
_WAV = _riff(b"WAVE")
_FLAC = _prefix(b"fLaC")

_MAGIC_SIGNATURES: dict[str, Callable[[bytes], bool]] = {
    "image/png": _prefix(b"\x89PNG\r\n\x1a\n"),
    "image/jpeg": _JPEG,
    "image/jpg": _JPEG,
    "image/gif": _prefix(b"GIF87a", b"GIF89a"),
    "image/webp": _riff(b"WEBP"),
    "image/bmp": _prefix(b"BM"),
    "image/tiff": _prefix(b"II*\x00", b"MM\x00*"),
    "image/x-icon": _ICO,
    "image/vnd.microsoft.icon": _ICO,
    "application/pdf": _pdf,
    "application/zip": _ZIP,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": _ZIP,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": _ZIP,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": _ZIP,
    "application/msword": _OLE,
    "application/vnd.ms-excel": _OLE,
    "application/vnd.ms-powerpoint": _OLE,
    "audio/mpeg": _mp3,
    "audio/mp3": _mp3,
    "audio/wav": _WAV,
    "audio/x-wav": _WAV,
    "audio/wave": _WAV,
    "audio/ogg": _OGG,
    "video/ogg": _OGG,
    "application/ogg": _OGG,
    "audio/flac": _FLAC,
    "audio/x-flac": _FLAC,
    "video/mp4": _iso_media,
    "audio/mp4": _iso_media,
    "audio/x-m4a": _iso_media,
    "video/webm": _EBML,
    "audio/webm": _EBML,
}


def validate_magic_bytes(data: bytes, claimed_mime: str) -> bool:
    """Check that the raw file bytes match the claimed MIME type.

    The type is compared without parameters and case (``IMAGE/PNG; x=1`` is ``image/png``).

    Args:
        data: Raw file bytes (the first KiB is enough).
        claimed_mime: The MIME type the uploader claims the file is.

    Returns:
        ``True`` if the leading bytes match OR the type has no known signature
        (plain text, CSV, JSON and other types pass).
        ``False`` if the data is empty or the leading bytes contradict the claim.
    """
    if not data:
        return False
    check = _MAGIC_SIGNATURES.get(claimed_mime.split(";", 1)[0].strip().lower())
    return True if check is None else check(data)


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------

# Characters not allowed in filenames across major OSes, C0 and C1 controls, the zero-width
# characters, the left-to-right and right-to-left marks, the line and paragraph separators,
# and the bidi embedding, override and isolate controls (U+202E can make ``a<RLO>gpj.exe``
# display as ``aexe.jpg``).
_UNSAFE_CHARS_RE = re.compile(
    '[<>:"/\\\\|?*\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202e\u2060\u2066-\u2069\ufeff]'
)
_MAX_FILENAME_LEN = 255


def sanitize_filename(filename: str) -> str:
    """Sanitise a user-provided filename for safe filesystem storage.

    - Keeps only the base name (after the last ``/`` or ``\\``) and removes ``..``
    - Removes control, zero-width and bidi control characters and ``<>:"/\\|?*``
    - Normalises Unicode (NFC)
    - Strips leading and trailing whitespace and dots
    - Truncates to 255 characters, keeping the extension
    - Returns ``"unnamed"`` for empty results

    Args:
        filename: Raw filename from user upload.

    Returns:
        A safe filename string.
    """
    if not filename:
        return "unnamed"

    # Normalise unicode
    filename = unicodedata.normalize("NFC", filename)

    # Take only the base name, splitting on both separators (a Windows path on Linux too)
    filename = re.split(r"[/\\]", filename)[-1]

    # Remove path traversal patterns
    filename = filename.replace("..", "")

    # Remove unsafe characters
    filename = _UNSAFE_CHARS_RE.sub("", filename)

    # Strip leading/trailing whitespace and dots
    filename = re.sub(r"^[\s.]+|[\s.]+$", "", filename)

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


# ---------------------------------------------------------------------------
# Pixel limit
# ---------------------------------------------------------------------------


def check_image_pixels(data: bytes, max_pixels: int | None) -> None:
    """Raise ``ValueError`` if an image has more than ``max_pixels`` pixels.

    Only the header is read (Pillow opens images lazily), so a small file that decodes to a
    huge bitmap is refused before any pixel is decoded. Does nothing when ``max_pixels`` is
    ``None``, when Pillow is not installed, or when ``data`` is not an image Pillow can open;
    other checks decide about those.

    Raises:
        ValueError: The image is larger than ``max_pixels``.
    """
    if not max_pixels:
        return
    try:
        import io

        from PIL import Image
    except ImportError:
        return
    try:
        with Image.open(io.BytesIO(data)) as img:
            pixels = img.width * img.height
    except Exception:
        return
    if pixels > max_pixels:
        raise ValueError(f"Image has {pixels} pixels; the limit is {max_pixels} pixels")
