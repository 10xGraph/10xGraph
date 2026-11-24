"""generate_id: an explicit id wins, else the bound BaseIDGenerator makes it."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from tenxgraph.core.state.message import Message, generate_id
from tenxgraph.utils.id_generator import (
    AsyncIDGenerator,
    BaseIDGenerator,
    DefaultIDGenerator,
    IDType,
    IntIDGenerator,
    UUIDGenerator,
)


def _bound(generator: BaseIDGenerator | None):
    """Patch InjectQ so the given generator is the BaseIDGenerator binding (or none)."""
    mock = MagicMock()
    mock.try_get.side_effect = lambda key, default=None: (
        generator if key is BaseIDGenerator else default
    )
    return patch("tenxgraph.core.state.message.InjectQ.get_instance", return_value=mock)


class _AsyncExternal(BaseIDGenerator):
    @property
    def id_type(self) -> IDType:
        return IDType.STRING

    async def generate(self) -> str:
        return "external-1"


def test_default_generator_gives_uuid_strings():
    value = DefaultIDGenerator().generate()
    assert isinstance(value, str) and len(value) == 36 and value.count("-") == 4
    assert DefaultIDGenerator().id_type == IDType.STRING


def test_no_generator_bound_gives_uuid_string():
    with _bound(None):
        value = generate_id()
    assert isinstance(value, str) and len(value) == 36


def test_explicit_id_wins():
    with _bound(IntIDGenerator()):
        assert generate_id("my-id") == "my-id"
        assert Message.text_message("hi", message_id="m-7").message_id == "m-7"


def test_bound_generator_makes_the_id():
    with _bound(IntIDGenerator()):
        assert isinstance(generate_id(), int)
        assert isinstance(Message.text_message("hi").message_id, int)


def test_async_generator_is_resolved_in_sync_context():
    with _bound(_AsyncExternal()):
        assert generate_id() == "external-1"


@pytest.mark.asyncio
async def test_async_generator_is_resolved_inside_a_running_loop():
    with _bound(_AsyncExternal()):
        assert generate_id() == "external-1"


def test_uuid_generator_warns_and_still_works():
    with pytest.warns(DeprecationWarning, match="UUIDGenerator"):
        generator = UUIDGenerator()
    assert isinstance(generator, DefaultIDGenerator)
    assert len(generator.generate()) == 36


@pytest.mark.asyncio
async def test_async_id_generator_warns_and_still_works():
    with pytest.warns(DeprecationWarning, match="AsyncIDGenerator"):
        generator = AsyncIDGenerator()
    assert len(await generator.generate()) == 36


def test_default_generator_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DefaultIDGenerator()
