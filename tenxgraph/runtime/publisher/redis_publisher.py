"""Redis publisher implementation (optional dependency).

This publisher uses the redis-py asyncio client to publish events via:
- Pub/Sub channels (default), or
- Redis Streams (XADD) when configured with mode="stream".

Dependency: redis>=4.2 (provides redis.asyncio).
Not installed by default; install extra: `pip install 10xscale-agentflow[redis]`.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from typing import Any

from agentflow.runtime.publisher.events import EventModel

from .base_publisher import BasePublisher


logger = logging.getLogger("agentflow.publisher")


class RedisPublisher(BasePublisher):
    """Publish events to Redis via Pub/Sub channel or Stream.

    Supports connection pooling and resource limits for production use.

    Attributes:
        url: Redis URL.
        mode: Publishing mode ('pubsub' or 'stream').
        channel: Pub/Sub channel name.
        stream: Stream name.
        maxlen: Max length for streams.
        encoding: Encoding for messages.
        max_connections: Maximum number of connections in the pool.
        socket_timeout: Socket timeout in seconds.
        socket_connect_timeout: Socket connect timeout in seconds.
        socket_keepalive: Enable TCP keepalive.
        health_check_interval: Interval for connection health checks in seconds.
        _redis: Redis client instance.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the RedisPublisher.

        Args:
            config: Configuration dictionary. Supported keys:
                - url: Redis URL (default: "redis://localhost:6379/0").
                - mode: Publishing mode ('pubsub' or 'stream', default: 'pubsub').
                - channel: Pub/Sub channel name (default: "agentflow.events").
                - stream: Stream name (default: "agentflow.events").
                - maxlen: Max length for streams.
                - encoding: Encoding (default: "utf-8").
                - max_connections: Maximum connections in pool (default: 10).
                - socket_timeout: Socket timeout in seconds (default: 5.0).
                - socket_connect_timeout: Connect timeout in seconds (default: 5.0).
                - socket_keepalive: Enable TCP keepalive (default: True).
                - health_check_interval: Health check interval in seconds (default: 30).
        """
        super().__init__(config or {})
        self.url: str = self.config.get("url", "redis://localhost:6379/0")
        self.mode: str = self.config.get("mode", "pubsub")
        self.channel: str = self.config.get("channel", "agentflow.events")
        self.stream: str = self.config.get("stream", "agentflow.events")
        self.maxlen: int | None = self.config.get("maxlen")
        self.encoding: str = self.config.get("encoding", "utf-8")
        self.max_connections: int = self.config.get("max_connections", 10)
        self.socket_timeout: float = self.config.get("socket_timeout", 5.0)
        self.socket_connect_timeout: float = self.config.get("socket_connect_timeout", 5.0)
        self.socket_keepalive: bool = self.config.get("socket_keepalive", True)
        self.health_check_interval: int = self.config.get("health_check_interval", 30)

        # Lazy import & connect on first use to avoid ImportError at import-time.
        self._redis = None  # type: ignore[var-annotated]
        self._connection_lock = asyncio.Lock()

    async def _get_client(self):
        """Get or create the Redis client with connection pooling.

        Returns:
            The Redis client instance.

        Raises:
            RuntimeError: If connection fails or publisher is closed.
        """
        if self._is_closed:
            raise RuntimeError("RedisPublisher is closed")

        async with self._connection_lock:
            if self._redis is not None:
                return self._redis

            try:
                redis_asyncio = importlib.import_module("redis.asyncio")
            except Exception as exc:  # ImportError and others
                raise RuntimeError(
                    "RedisPublisher requires the 'redis' package. Install with "
                    "'pip install 10xscale-agentflow[redis]' or 'pip install redis'."
                ) from exc

            try:
                # Create connection pool with limits
                pool = redis_asyncio.ConnectionPool.from_url(
                    self.url,
                    encoding=self.encoding,
                    decode_responses=False,
                    max_connections=self.max_connections,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    socket_keepalive=self.socket_keepalive,
                    health_check_interval=self.health_check_interval,
                )
                self._redis = redis_asyncio.Redis(connection_pool=pool)

                # Test connection
                await self._redis.ping()
                logger.info(
                    "RedisPublisher connected successfully (max_connections=%d)",
                    self.max_connections,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"RedisPublisher failed to connect to Redis at {self.url}"
                ) from exc

            return self._redis

    async def publish(self, event: EventModel) -> Any:
        """Publish an event to Redis.

        Args:
            event: The event to publish.

        Returns:
            The result of the publish operation.

        Raises:
            RuntimeError: If publisher is closed.
        """
        if self._is_closed:
            raise RuntimeError("Cannot publish to closed RedisPublisher")

        client = await self._get_client()
        payload = json.dumps(event.model_dump()).encode(self.encoding)

        if self.mode == "stream":
            # XADD to stream
            fields = {"data": payload}
            if self.maxlen is not None:
                return await client.xadd(self.stream, fields, maxlen=self.maxlen, approximate=True)
            return await client.xadd(self.stream, fields)

        # Default: Pub/Sub channel
        return await client.publish(self.channel, payload)

    async def close(self):
        """Close the Redis client and release connection pool resources.

        This method is idempotent and can be called multiple times safely.
        """
        if self._is_closed:
            logger.debug("RedisPublisher already closed")
            return

        async with self._connection_lock:
            if self._redis is not None:
                try:
                    await self._redis.aclose()
                    logger.info("RedisPublisher closed successfully")
                except Exception:  # best-effort close
                    logger.debug("RedisPublisher close encountered an error", exc_info=True)
                finally:
                    self._redis = None
                    self._is_closed = True

    def sync_close(self):
        """Synchronously close the Redis client."""
        try:
            asyncio.run(self.close())
        except RuntimeError:
            # Already in an event loop; fall back to scheduling close
            logger.warning("sync_close called within an active event loop; skipping.")
