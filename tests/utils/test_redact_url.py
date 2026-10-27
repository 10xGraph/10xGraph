"""redact_url strips credentials from URLs shown in logs and errors."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
from tenxgraph.storage.media.url_policy import redact_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("redis://:s3cret@host:6379/0", "redis://host:6379/0"),
        ("amqp://user:s3cret@host/vh", "amqp://host/vh"),
        ("redis://host:6379/0", "redis://host:6379/0"),
        ("redis://:s3cret@[::1]:6379/0", "redis://[::1]:6379/0"),
        ("redis://u:p@host:6379/0?password=x#frag", "redis://host:6379/0"),
    ],
)
def test_redact_url(url, expected):
    assert redact_url(url) == expected


@pytest.mark.parametrize("url", ["not a url", "", "redis://:s3cret@[bad"])
def test_redact_url_non_url_never_raises_or_leaks(url):
    out = redact_url(url)
    assert "s3cret" not in out


def test_redis_publisher_connect_error_hides_password():
    pub = RedisPublisher({"url": "redis://user:s3cret@redis.example:6379/0"})
    fake_redis = MagicMock()
    fake_redis.asyncio.ConnectionPool.from_url.return_value = MagicMock()
    client = MagicMock()
    client.ping = AsyncMock(side_effect=ConnectionError("down"))
    fake_redis.asyncio.Redis.return_value = client
    with (
        patch("importlib.import_module", return_value=fake_redis.asyncio),
        pytest.raises(RuntimeError) as info,
    ):
        asyncio.run(pub._get_client())
    assert "s3cret" not in str(info.value)
    assert "redis.example:6379" in str(info.value)
