"""Identity transforms for observability output.

``user_id_mode`` controls how a user id appears on OTEL spans (and therefore in
Logfire and LangSmith, which receive the same spans):

- ``"raw"`` (default): the id is emitted unchanged.
- ``"hash"``: a 16 hex character digest (the first 64 bits of SHA-256). When the
  environment variable ``TENXGRAPH_TRACE_USER_ID_KEY`` is set, HMAC-SHA256 with that
  key is used instead, so short or guessable ids cannot be recovered by hashing
  candidate values. Without the key it is plain SHA-256, which can be reversed by
  guessing for short ids (emails, small integers). The same id always gives the
  same digest, so spans of one user can still be grouped.
- ``"mask"``: a fixed-width partial view. Ids of 4 characters or fewer become
  ``"***"``; ids of 5 to 7 characters keep the first and last character
  (``"a***2"``); longer ids keep the first two and last two (``"al***42"``).
  The stars are always three, so the id length is not revealed.

Only observability spans use this. Event payloads sent by the Redis, RabbitMQ,
Kafka and Console publishers are not changed.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Literal, get_args


UserIdMode = Literal["raw", "hash", "mask"]

USER_ID_MODES: tuple[str, ...] = get_args(UserIdMode)
USER_ID_KEY_ENV = "TENXGRAPH_TRACE_USER_ID_KEY"
_HASH_HEX_CHARS = 16
_MASK_ALL_MAX_LEN = 4
_MASK_SHORT_MAX_LEN = 7


def validate_user_id_mode(mode: str) -> UserIdMode:
    """Return ``mode`` if allowed, else raise ``ValueError`` listing the allowed values."""
    if mode not in USER_ID_MODES:
        raise ValueError(
            f"Invalid user_id_mode {mode!r}. Allowed values: {', '.join(USER_ID_MODES)}"
        )
    return mode  # type: ignore[return-value]


def mask_user_id(user_id: str) -> str:
    """Partly hide ``user_id`` (deterministic; see module docstring)."""
    n = len(user_id)
    if n <= _MASK_ALL_MAX_LEN:
        return "***"
    keep = 1 if n <= _MASK_SHORT_MAX_LEN else 2
    return f"{user_id[:keep]}***{user_id[-keep:]}"


def hash_user_id(user_id: str, key: str | None = None) -> str:
    """Return a 16 hex char digest: HMAC-SHA256 with ``key`` if given, else SHA-256."""
    data = user_id.encode("utf-8")
    if key:
        digest = hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(data).hexdigest()
    return digest[:_HASH_HEX_CHARS]


def transform_user_id(user_id: object, mode: UserIdMode = "raw", key: str | None = None) -> str:
    """Apply ``mode`` to ``user_id`` and return the string to emit."""
    text = str(user_id)
    if mode == "hash":
        return hash_user_id(text, key if key is not None else os.environ.get(USER_ID_KEY_ENV))
    if mode == "mask":
        return mask_user_id(text)
    return text
