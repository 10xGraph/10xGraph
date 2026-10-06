"""Tests for generate_id in agentflow/core/state/message.py.

Covers the int/bigint id_type paths, default_id matching, InjectQ override,
and awaitable resolution - previously uncovered branches.

Uses unittest.mock.patch to replace InjectQ.get_instance on the module level
so tests are fully isolated from the live InjectQ singleton.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.state.message import generate_id


def _mock_iq(id_type: str = "string", generated_id=None) -> MagicMock:
    """Create a mock InjectQ instance with controlled try_get behaviour."""
    mock = MagicMock()
    mock.try_get.side_effect = lambda key, default=None: (
        id_type if key == "generated_id_type"
        else generated_id if key == "generated_id"
        else default
    )
    return mock


def _patch(id_type: str = "string", generated_id=None):
    """Patch InjectQ.get_instance() on the current state.message module."""
    return patch(
        "tenxgraph.core.state.message.InjectQ.get_instance",
        return_value=_mock_iq(id_type=id_type, generated_id=generated_id),
    )


class TestGenerateIdIntType:
    def test_generates_int_when_type_is_int(self):
        with _patch(id_type="int"):
            result = generate_id(None)
        assert isinstance(result, int)
        assert result >= 0

    def test_generates_int_when_type_is_bigint(self):
        with _patch(id_type="bigint"):
            result = generate_id(None)
        assert isinstance(result, int)
        assert result >= 0

    def test_int_result_smaller_than_int_range(self):
        """int uses uuid4().int >> 96, giving a ~32-bit value."""
        with _patch(id_type="int"):
            result = generate_id(None)
        assert result < 2**32

    def test_bigint_result_in_64bit_range(self):
        """bigint uses uuid4().int >> 64, giving a 64-bit value."""
        with _patch(id_type="bigint"):
            result = generate_id(None)
        assert result < 2**64


class TestGenerateIdDefaultIdMatching:
    def test_returns_int_default_when_type_is_int(self):
        with _patch(id_type="int"):
            result = generate_id(42)
        assert result == 42

    def test_returns_int_default_when_type_is_bigint(self):
        with _patch(id_type="bigint"):
            result = generate_id(99999)
        assert result == 99999

    def test_discards_str_default_when_type_is_int(self):
        """String default_id does not match id_type='int'; a fresh int is generated."""
        with _patch(id_type="int"):
            result = generate_id("string-id")
        assert isinstance(result, int)

    def test_discards_int_default_when_type_is_string(self):
        """Int default_id does not match id_type='string'; a fresh UUID str is generated."""
        with _patch(id_type="string"):
            result = generate_id(123)
        assert isinstance(result, str)

    def test_returns_str_default_when_type_is_string(self):
        with _patch(id_type="string"):
            result = generate_id("my-custom-id")
        assert result == "my-custom-id"


class TestGenerateIdInjectQOverride:
    def test_injected_string_id_used_directly(self):
        with _patch(id_type="string", generated_id="injected-id-xyz"):
            result = generate_id(None)
        assert result == "injected-id-xyz"

    def test_injected_id_overrides_default(self):
        with _patch(id_type="string", generated_id="override"):
            result = generate_id("should-not-use-this")
        assert result == "override"

    def test_injected_int_id_used_directly(self):
        with _patch(id_type="int", generated_id=42):
            result = generate_id(None)
        assert result == 42


class TestGenerateIdAwaitablePath:
    def test_awaitable_resolved_in_sync_context(self):
        """An awaitable generated_id is resolved via asyncio.run()."""
        async def get_id():
            return "awaited-id-value"

        coro = get_id()
        with _patch(id_type="string", generated_id=coro):
            result = generate_id(None)

        assert result == "awaited-id-value"
