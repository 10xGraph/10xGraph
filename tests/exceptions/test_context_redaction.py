"""Exception context is redacted before it reaches the log (G-63)."""

import logging

from tenxgraph.core.exceptions import GraphError, NodeError
from tenxgraph.core.exceptions.storage_exceptions import StorageError


def test_secret_keys_are_redacted_in_log_but_kept_on_the_error(caplog):
    ctx = {"api_key": "sk-live-abcdefghijklmnopqrstuvwx", "nested": {"token": "t0k3n-value"}, "n": 1}
    with caplog.at_level(logging.DEBUG, logger="tenxgraph.exceptions"):
        err = GraphError("boom", context=ctx)
        NodeError("boom", context=ctx)
        StorageError("boom", context=ctx)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "sk-live-abcdefghijklmnopqrstuvwx" not in text
    assert "t0k3n-value" not in text
    assert "***REDACTED***" in text
    assert "'n': 1" in text
    # The structured error keeps the real context for the caller.
    assert err.context["api_key"] == "sk-live-abcdefghijklmnopqrstuvwx"
