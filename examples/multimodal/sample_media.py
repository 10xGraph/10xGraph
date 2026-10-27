"""Tiny generated media so the examples run without any asset files."""

import base64
import io
import math
import struct
import wave
import zlib


def make_png(size: int = 64) -> bytes:
    """A size x size PNG with a red-to-blue gradient."""
    rows = b""
    for y in range(size):
        rows += b"\x00" + b"".join(
            bytes((255 - x * 255 // size, y * 255 // size // 4, x * 255 // size))
            for x in range(size)
        )

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def make_wav(seconds: float = 1.0, freq: float = 440.0, rate: int = 16000) -> bytes:
    """A mono 16-bit sine tone."""
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(int(seconds * rate))
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
