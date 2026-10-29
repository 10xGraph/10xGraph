"""Error text that is safe to show a client."""

from __future__ import annotations


def _is_ours(exc: BaseException) -> bool:
    return (type(exc).__module__ or "").split(".", 1)[0] in {"tenxgraph", "agentflow"}


def client_error_text(exc: BaseException) -> str:
    """Return a description of ``exc`` that can be sent to the client.

    Messages of this framework's own exceptions (``tenxgraph.*``) are written to be
    shown, unless one wraps a foreign exception (a node wraps whatever the node code
    raised): the wrapped text can carry request bodies, URLs, keys or SQL. For a
    foreign exception, or a wrapper around one, only the type of the innermost foreign
    exception is returned; the full text stays in the server log.
    """
    foreign: BaseException | None = None
    cur: BaseException | None = exc
    seen = 0
    while cur is not None and seen < 10:  # noqa: PLR2004
        if not _is_ours(cur):
            foreign = cur
        cur = cur.__cause__
        seen += 1
    if foreign is None:
        return str(exc)
    return f"{type(foreign).__name__}: the run failed (details are in the server log)"
