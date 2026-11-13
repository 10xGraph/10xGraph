from abc import ABC, abstractmethod
from typing import Any, Literal

from .events import EventModel


PublisherKind = Literal["events", "observability"]


class BasePublisher(ABC):
    """Abstract base class for event publishers.

    This class defines the interface for publishing events. Subclasses should implement
    the publish, close, and sync_close methods to provide specific publishing logic.

    Supports async context manager for automatic resource cleanup:
        async with publisher:
            await publisher.publish(event)
        # Resources automatically cleaned up on exit

    Attributes:
        config: Configuration dictionary for the publisher.
        kind: What the publisher is for. ``"events"`` (default) is an event stream for your
            own consumers (Kafka, Redis, RabbitMQ, console, custom). ``"observability"`` is
            an AI tracing backend (OTEL, Logfire, LangSmith, Phoenix). Callers can send an
            event to one kind only with ``publish_event(..., only=kind)``.
        inline: When True, ``publish_event`` calls :meth:`publish_inline` on the caller's
            path instead of scheduling :meth:`publish` as a background task. For in-process
            publishers whose work is quick and must be done before the run goes on (OTEL
            spans that library spans nest under). Default False.
    """

    kind: PublisherKind = "events"
    inline: bool = False

    def __init__(self, config: dict[str, Any]):
        """Initialize the publisher with the given configuration.

        Args:
            config: Configuration dictionary for the publisher.
        """
        self.config = config
        self._is_closed = False

    @abstractmethod
    async def publish(self, event: EventModel) -> Any:
        """Publish an event.

        Args:
            event: The event to publish.

        Returns:
            The result of the publish operation.

        Raises:
            RuntimeError: If the publisher is closed.
        """
        raise NotImplementedError

    def publish_inline(self, event: EventModel) -> None:
        """Handle an event synchronously (``inline`` publishers only). Must never raise."""
        raise NotImplementedError

    @abstractmethod
    async def close(self):
        """Close the publisher and release any resources.

        This method should be overridden by subclasses to provide specific cleanup logic.
        It will be called externally and should be idempotent.
        """
        raise NotImplementedError

    @abstractmethod
    def sync_close(self):
        """Close the publisher and release any resources (synchronous version).

        This method should be overridden by subclasses to provide specific cleanup logic.
        It will be called externally and should be idempotent.
        """
        raise NotImplementedError

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager, ensuring cleanup."""
        await self.close()
        return False  # Don't suppress exceptions
