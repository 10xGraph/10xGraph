"""Kafka publisher implementation (optional dependency).

Uses aiokafka to publish events to a Kafka topic.

Dependency: aiokafka
Not installed by default; install extra: `pip install 10xscale-agentflow[kafka]`.
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


class KafkaPublisher(BasePublisher):
    """Publish events to a Kafka topic using aiokafka.

    This class provides an asynchronous interface for publishing events to a Kafka topic.
    It uses the aiokafka library to handle the producer operations. The publisher is
    lazily initialized and can be reused for multiple publishes.

    Supports connection pooling and resource limits for production use.

    Attributes:
        bootstrap_servers: Kafka bootstrap servers.
        topic: Kafka topic to publish to.
        client_id: Client ID for the producer.
        max_batch_size: Maximum size of a batch in bytes.
        linger_ms: Time to wait for additional messages before sending.
        compression_type: Compression type ('none', 'gzip', 'snappy', 'lz4', 'zstd').
        request_timeout_ms: Request timeout in milliseconds.
        _producer: Lazy-loaded Kafka producer instance.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the KafkaPublisher.

        Args:
            config: Configuration dictionary. Supported keys:
                - bootstrap_servers: Kafka bootstrap servers (default: "localhost:9092").
                - topic: Kafka topic to publish to (default: "agentflow.events").
                - client_id: Client ID for the producer.
                - max_batch_size: Maximum batch size in bytes (default: 16384).
                - linger_ms: Linger time in milliseconds (default: 0).
                - compression_type: Compression type (default: None).
                - request_timeout_ms: Request timeout in ms (default: 30000).
        """
        super().__init__(config or {})
        self.bootstrap_servers: str = self.config.get("bootstrap_servers", "localhost:9092")
        self.topic: str = self.config.get("topic", "agentflow.events")
        self.client_id: str | None = self.config.get("client_id")
        self.max_batch_size: int = self.config.get("max_batch_size", 16384)
        self.linger_ms: int = self.config.get("linger_ms", 0)
        self.compression_type: str | None = self.config.get("compression_type")
        self.request_timeout_ms: int = self.config.get("request_timeout_ms", 30000)
        self._producer = None  # type: ignore[var-annotated]
        self._producer_lock = asyncio.Lock()

    async def _get_producer(self):
        """Get or create the Kafka producer instance with connection limits.

        This method lazily initializes the producer if it hasn't been created yet.
        It imports aiokafka and starts the producer with connection pooling.

        Returns:
            The initialized producer instance.

        Raises:
            RuntimeError: If the 'aiokafka' package is not installed or publisher is closed.
        """
        if self._is_closed:
            raise RuntimeError("KafkaPublisher is closed")

        async with self._producer_lock:
            if self._producer is not None:
                return self._producer

            try:
                aiokafka = importlib.import_module("aiokafka")
            except Exception as exc:
                raise RuntimeError(
                    "KafkaPublisher requires the 'aiokafka' package. Install with "
                    "'pip install 10xscale-agentflow[kafka]' or 'pip install aiokafka'."
                ) from exc

            producer_cls = aiokafka.AIOKafkaProducer
            self._producer = producer_cls(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                max_batch_size=self.max_batch_size,
                linger_ms=self.linger_ms,
                compression_type=self.compression_type,
                request_timeout_ms=self.request_timeout_ms,
            )
            await self._producer.start()
            logger.info(
                "KafkaPublisher connected successfully (batch_size=%d, linger_ms=%d)",
                self.max_batch_size,
                self.linger_ms,
            )
            return self._producer

    async def publish(self, event: EventModel) -> Any:
        """Publish an event to the Kafka topic.

        Args:
            event: The event to publish.

        Returns:
            The result of the send_and_wait operation.

        Raises:
            RuntimeError: If publisher is closed.
        """
        if self._is_closed:
            raise RuntimeError("Cannot publish to closed KafkaPublisher")

        producer = await self._get_producer()
        payload = json.dumps(event.model_dump()).encode("utf-8")
        return await producer.send_and_wait(self.topic, payload)

    async def close(self):
        """Close the Kafka producer and release resources.

        Stops the producer and cleans up resources. Errors during stopping are logged
        but do not raise exceptions. This method is idempotent.
        """
        if self._is_closed:
            logger.debug("KafkaPublisher already closed")
            return

        async with self._producer_lock:
            if self._producer is not None:
                try:
                    await self._producer.stop()
                    logger.info("KafkaPublisher closed successfully")
                except Exception:
                    logger.debug("KafkaPublisher close encountered an error", exc_info=True)
                finally:
                    self._producer = None
                    self._is_closed = True

    def sync_close(self):
        """Synchronously close the Kafka producer.

        This method runs the async close in a new event loop. If called within an
        active event loop, it logs a warning and skips the operation.
        """
        try:
            asyncio.run(self.close())
        except RuntimeError:
            logger.warning("sync_close called within an active event loop; skipping.")
