"""RabbitMQ publisher implementation (optional dependency).

Uses aio-pika to publish events to an exchange with a routing key.

Dependency: aio-pika
Not installed by default; install extra: `pip install 10xscale-agentflow[rabbitmq]`.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from typing import Any

from .base_publisher import BasePublisher
from .events import EventModel


logger = logging.getLogger("agentflow.publisher")


class RabbitMQPublisher(BasePublisher):
    """Publish events to RabbitMQ using aio-pika.

    Supports connection pooling and resource limits for production use.

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
                - exchange: Exchange name (default: "agentflow.events").
                - routing_key: Routing key (default: "agentflow.events").
                - exchange_type: Exchange type (default: "topic").
                - declare: Whether to declare exchange (default: True).
                - durable: Whether exchange is durable (default: True).
                - connection_timeout: Connection timeout in seconds (default: 10).
                - heartbeat: Heartbeat interval in seconds (default: 60).
        """
        super().__init__(config or {})
        self.url: str = self.config.get("url", "amqp://guest:guest@localhost/")
        self.exchange: str = self.config.get("exchange", "agentflow.events")
        self.routing_key: str = self.config.get("routing_key", "agentflow.events")
        self.exchange_type: str = self.config.get("exchange_type", "topic")
        self.declare: bool = self.config.get("declare", True)
        self.durable: bool = self.config.get("durable", True)
        self.connection_timeout: int = self.config.get("connection_timeout", 10)
        self.heartbeat: int = self.config.get("heartbeat", 60)

        self._conn = None  # type: ignore[var-annotated]
        self._channel = None  # type: ignore[var-annotated]
        self._exchange = None  # type: ignore[var-annotated]
        self._connection_lock = asyncio.Lock()

    async def _ensure(self):
        """Ensure the connection, channel, and exchange are initialized with timeouts.

        Raises:
            RuntimeError: If publisher is closed or connection fails.
        """
        if self._is_closed:
            raise RuntimeError("RabbitMQPublisher is closed")

        async with self._connection_lock:
            if self._exchange is not None:
                return

            try:
                aio_pika = importlib.import_module("aio_pika")
            except Exception as exc:
                raise RuntimeError(
                    "RabbitMQPublisher requires the 'aio-pika' package. Install with "
                    "'pip install 10xscale-agentflow[rabbitmq]' or 'pip install aio-pika'."
                ) from exc

            # Connect with timeout and heartbeat
            self._conn = await aio_pika.connect_robust(
                self.url,
                timeout=self.connection_timeout,
                heartbeat=self.heartbeat,
            )
            self._channel = await self._conn.channel()

            if self.declare:
                ex_type = getattr(
                    aio_pika.ExchangeType,
                    self.exchange_type.upper(),
                    aio_pika.ExchangeType.TOPIC,
                )
                self._exchange = await self._channel.declare_exchange(
                    self.exchange, ex_type, durable=self.durable
                )
            else:
                # Fall back to default exchange
                self._exchange = self._channel.default_exchange

            logger.info(
                "RabbitMQPublisher connected successfully (heartbeat=%ds)",
                self.heartbeat,
            )

    async def publish(self, event: EventModel) -> Any:
        """Publish an event to RabbitMQ.

        Args:
            event: The event to publish.

        Returns:
            True on success.

        Raises:
            RuntimeError: If publisher is closed or not initialized.
        """
        if self._is_closed:
            raise RuntimeError("Cannot publish to closed RabbitMQPublisher")

        await self._ensure()
        payload = json.dumps(event.model_dump()).encode("utf-8")

        aio_pika = importlib.import_module("aio_pika")
        message = aio_pika.Message(body=payload)
        if self._exchange is None:
            raise RuntimeError("RabbitMQPublisher exchange not initialized")
        await self._exchange.publish(message, routing_key=self.routing_key)
        return True

    async def close(self):
        """Close the RabbitMQ connection and channel, releasing resources.

        This method is idempotent and can be called multiple times safely.
        """
        if self._is_closed:
            logger.debug("RabbitMQPublisher already closed")
            return

        async with self._connection_lock:
            try:
                if self._channel is not None:
                    await self._channel.close()
            except Exception:
                logger.debug("RabbitMQPublisher channel close error", exc_info=True)
            finally:
                self._channel = None

            try:
                if self._conn is not None:
                    await self._conn.close()
            except Exception:
                logger.debug("RabbitMQPublisher connection close error", exc_info=True)
            finally:
                self._conn = None
                self._exchange = None
                self._is_closed = True
                logger.info("RabbitMQPublisher closed successfully")

    def sync_close(self):
        """Synchronously close the RabbitMQ connection."""
        try:
            asyncio.run(self.close())
        except RuntimeError:
            logger.warning("sync_close called within an active event loop; skipping.")
