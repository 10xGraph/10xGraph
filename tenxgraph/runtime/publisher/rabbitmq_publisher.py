"""RabbitMQ publisher implementation (optional dependency).

Uses aio-pika to publish events to an exchange with a routing key.

Dependency: aio-pika
Not installed by default; install extra: `pip install 10xgraph[rabbitmq]`.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import time
from typing import Any

from tenxgraph.storage.media.url_policy import redact_url

from .base_publisher import BasePublisher
from .events import EventModel


logger = logging.getLogger("tenxgraph.publisher")


class RabbitMQPublisher(BasePublisher):
    """Publish events to RabbitMQ using aio-pika.

    Supports connection pooling and resource limits for production use.

    TLS is the developer's choice: use an ``amqps://`` URL to encrypt the connection
    (the default ``amqp://`` is plaintext).

    Attributes:
        url: RabbitMQ connection URL.
        exchange: Exchange name.
        routing_key: Routing key for messages.
        exchange_type: Type of exchange.
        declare: Whether to declare the exchange.
        durable: Whether the exchange is durable.
        connection_timeout: Connection timeout in seconds.
        heartbeat: Heartbeat interval in seconds.
        _conn: Connection instance.
        _channel: Channel instance.
        _exchange: Exchange instance.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the RabbitMQPublisher.

        Args:
            config: Configuration dictionary. Supported keys:
                - url: RabbitMQ URL (default: "amqp://guest:guest@localhost/").
                - exchange: Exchange name (default: "tenxgraph.events").
                - routing_key: Routing key (default: "tenxgraph.events").
                - exchange_type: Exchange type (default: "topic").
                - declare: Whether to declare exchange (default: True).
                - durable: Whether exchange is durable (default: True).
                - connection_timeout: Connection timeout in seconds (default: 10).
                - heartbeat: Heartbeat interval in seconds (default: 60).
                - publish_timeout: Optional seconds to wait for one send (default: none).
                - reconnect_interval: Seconds to wait after a failed connection before the
                  next attempt; events in between are dropped (default: 5).
        """
        super().__init__(config or {})
        self.url: str = self.config.get("url", "amqp://guest:guest@localhost/")
        self.exchange: str = self.config.get("exchange", "tenxgraph.events")
        self.routing_key: str = self.config.get("routing_key", "tenxgraph.events")
        self.exchange_type: str = self.config.get("exchange_type", "topic")
        self.declare: bool = self.config.get("declare", True)
        self.durable: bool = self.config.get("durable", True)
        self.connection_timeout: int = self.config.get("connection_timeout", 10)
        self.heartbeat: int = self.config.get("heartbeat", 60)
        self.reconnect_interval: float = self.config.get("reconnect_interval", 5)

        self._conn: Any = None
        self._channel: Any = None
        self._exchange: Any = None
        self._connection_lock = asyncio.Lock()
        self._auth_failed = False  # broker refused the credentials; never retried
        self._retry_at = 0.0  # monotonic time of the next connection attempt

    async def _ensure(self) -> bool:
        """Open the connection, channel and exchange when they are not open.

        A plain connection is used, not ``connect_robust``: a robust connection runs its own
        reconnect task that retries every few seconds forever (also after the broker refused
        the credentials) and, when the first attempt is cancelled, is never closed. Here a
        dropped connection is opened again by the next publish instead.

        Returns:
            True when the exchange is ready. False when the broker refused the credentials:
            that is permanent, so it is logged once and no further attempt is made.

        Raises:
            RuntimeError: If the publisher is closed, or a network failure happened less
                than ``reconnect_interval`` seconds ago.
        """
        if self._is_closed:
            raise RuntimeError("RabbitMQPublisher is closed")

        async with self._connection_lock:
            if self._is_closed:
                raise RuntimeError("RabbitMQPublisher is closed")
            if self._auth_failed:
                return False
            # The channel (not the connection object) tells when the broker side closed.
            if self._exchange is not None and not self._channel.is_closed:
                return True
            # The broker closed the old connection or channel: drop it and connect again.
            await self._discard_connection()
            wait = self._retry_at - time.monotonic()
            if wait > 0:
                raise RuntimeError(
                    f"RabbitMQPublisher is not connected; next attempt in {wait:.1f}s"
                )

            try:
                aio_pika = importlib.import_module("aio_pika")
            except Exception as exc:
                raise RuntimeError(
                    "RabbitMQPublisher requires the 'aio-pika' package. Install with "
                    "'pip install 10xgraph[rabbitmq]' or 'pip install aio-pika'."
                ) from exc

            # Every step is bounded by connection_timeout. A failure closes the half-open
            # connection, so the next attempt starts clean.
            try:
                self._conn = await asyncio.wait_for(
                    aio_pika.connect(
                        self.url, timeout=self.connection_timeout, heartbeat=self.heartbeat
                    ),
                    timeout=self.connection_timeout,
                )
                self._exchange = await asyncio.wait_for(
                    self._open_channel(aio_pika), timeout=self.connection_timeout
                )
            except aio_pika.exceptions.ProbableAuthenticationError as exc:
                await self._discard_connection()
                self._auth_failed = True
                logger.error(
                    "RabbitMQPublisher: the broker at %s refused the credentials (%s). "
                    "Not retrying; events are dropped until the publisher is created "
                    "again with valid credentials.",
                    redact_url(self.url),
                    exc,
                )
                return False
            except Exception:
                await self._discard_connection()
                self._retry_at = time.monotonic() + self.reconnect_interval
                raise
            except BaseException:
                await self._discard_connection()
                raise

            if self._is_closed:
                # close() ran while this connection was opening.
                await self._discard_connection()
                raise RuntimeError("RabbitMQPublisher is closed")

            logger.info(
                "RabbitMQPublisher connected successfully (heartbeat=%ds)",
                self.heartbeat,
            )
            return True

    async def _open_channel(self, aio_pika: Any) -> Any:
        """Open the channel and return the exchange to publish to."""
        self._channel = await self._conn.channel()
        if self.declare:
            ex_type = getattr(
                aio_pika.ExchangeType,
                self.exchange_type.upper(),
                aio_pika.ExchangeType.TOPIC,
            )
            return await self._channel.declare_exchange(
                self.exchange, ex_type, durable=self.durable
            )
        # Fall back to default exchange
        return self._channel.default_exchange

    async def _discard_connection(self) -> None:
        """Close and forget the connection (and its channel), waiting at most
        ``connection_timeout`` seconds for the broker."""
        conn, self._conn = self._conn, None
        self._channel = None
        self._exchange = None
        if conn is not None:
            try:
                await asyncio.wait_for(conn.close(), timeout=self.connection_timeout)
            except Exception:
                logger.debug("RabbitMQPublisher connection close error", exc_info=True)

    async def publish(self, event: EventModel) -> Any:
        """Publish an event to RabbitMQ.

        Args:
            event: The event to publish.

        Returns:
            True on success, False when the event was dropped because the broker
            refused the credentials.

        Raises:
            RuntimeError: If publisher is closed or not initialized.
        """
        if self._is_closed:
            raise RuntimeError("Cannot publish to closed RabbitMQPublisher")

        if not await self._ensure():
            return False  # credentials refused: logged once, event dropped
        payload = json.dumps(event.model_dump()).encode("utf-8")

        aio_pika = importlib.import_module("aio_pika")
        message = aio_pika.Message(body=payload)
        if self._exchange is None:
            raise RuntimeError("RabbitMQPublisher exchange not initialized")
        # Optional ``publish_timeout`` (seconds) bounds the send; unset waits for the broker.
        timeout = self.config.get("publish_timeout")
        if not isinstance(timeout, int | float) or isinstance(timeout, bool) or timeout <= 0:
            timeout = None
        if timeout:
            await asyncio.wait_for(
                self._exchange.publish(message, routing_key=self.routing_key), timeout=timeout
            )
        else:
            await self._exchange.publish(message, routing_key=self.routing_key)
        return True

    async def close(self):
        """Close the RabbitMQ connection and channel, releasing resources.

        Returns promptly in any state: it does not wait for a connection attempt in
        progress (that attempt sees the closed flag and closes its own connection), and
        the broker gets at most ``connection_timeout`` seconds to confirm the close.
        This method is idempotent and can be called multiple times safely.
        """
        if self._is_closed:
            logger.debug("RabbitMQPublisher already closed")
            return

        self._is_closed = True
        await self._discard_connection()
        logger.info("RabbitMQPublisher closed successfully")

    def sync_close(self):
        """Synchronously close the RabbitMQ connection."""
        try:
            asyncio.run(self.close())
        except RuntimeError:
            logger.warning("sync_close called within an active event loop; skipping.")
