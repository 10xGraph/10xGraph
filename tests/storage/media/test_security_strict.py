"""validate_magic_bytes and sanitize_filename are at least as strict as the API's checks."""

import pytest

from tenxgraph.storage.media.security import sanitize_filename, validate_magic_bytes


_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 8


@pytest.mark.parametrize(
    ("data", "mime", "expected"),
    [
        (b"hello", _DOCX, False),
        (b"  \n%PDF-1.7\n", "application/pdf", True),
        (b"hello", "image/jpg", False),
        (b"", "text/plain", False),
        (b"hello", "IMAGE/PNG", False),
        (b"hello", "audio/mpeg", False),
        (b"hello", "image/png; x=1", False),
    ],
)
def test_review_samples(data, mime, expected):
    assert validate_magic_bytes(data, mime) is expected


@pytest.mark.parametrize(
    ("mime", "data"),
    [
        ("image/png", _PNG),
        ("IMAGE/PNG; charset=binary", _PNG),
        ("image/jpg", b"\xff\xd8\xff\xe0"),
        ("image/webp", b"RIFF\x00\x00\x00\x00WEBPVP8 "),
        ("image/x-icon", b"\x00\x00\x01\x00"),
        (_DOCX, b"PK\x03\x04x"),
        ("application/msword", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"),
        ("audio/mpeg", b"ID3\x03\x00"),
        ("audio/mpeg", b"\xff\xfb\x90\x00"),
        ("audio/wav", b"RIFF\x00\x00\x00\x00WAVEfmt "),
        ("audio/ogg", b"OggS\x00"),
        ("audio/flac", b"fLaC\x00"),
        ("video/mp4", b"\x00\x00\x00\x18ftypmp42"),
        ("video/webm", b"\x1a\x45\xdf\xa3\x01"),
    ],
)
def test_matching_content_passes(mime, data):
    assert validate_magic_bytes(data, mime) is True


@pytest.mark.parametrize(
    ("mime", "data"),
    [
        ("image/webp", b"RIFF\x00\x00\x00\x00WAVE"),
        ("audio/wav", b"RIFF\x00\x00\x00\x00WEBP"),
        ("video/mp4", b"abcdefghijkl"),
        ("image/png", b"<html><script>alert(1)</script>"),
    ],
)
def test_mismatching_content_fails(mime, data):
    assert validate_magic_bytes(data, mime) is False


@pytest.mark.parametrize("mime", ["text/plain", "text/csv", "application/json"])
def test_types_without_a_signature_pass(mime):
    assert validate_magic_bytes(b"anything", mime) is True


@pytest.mark.parametrize(
    ("sent", "stored"),
    [
        ("a\u202egpj.exe", "agpj.exe"),
        ("\u202ax\u202b\u202c\u202dy.txt", "xy.txt"),
        ("\u2066x\u2067\u2068\u2069.txt", "x.txt"),
        ("z\u200bw\u200c\u200d\ufeff.txt", "zw.txt"),
        ("\u200ex\u200f.txt", "x.txt"),
        ("tab\x85\x9f\x7fx.txt", "tabx.txt"),
        ("line\r\nbreak.txt", "linebreak.txt"),
        ("C:\\Users\\bob\\report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..", "unnamed"),
        ("dir/", "unnamed"),
        ("\u202e\x00", "unnamed"),
        ("  spaced.txt  ", "spaced.txt"),
        ("r\u00e9sum\u00e9 \u65e5\u672c\u8a9e.pdf", "r\u00e9sum\u00e9 \u65e5\u672c\u8a9e.pdf"),
    ],
)
def test_sanitize_filename_drops_controls_and_bidi(sent, stored):
    assert sanitize_filename(sent) == stored
