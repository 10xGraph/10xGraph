import asyncio

from .base_publisher import BasePublisher
from .events import EventModel


class CompositePublisher(BasePublisher):
    """Fan-out publisher that broadcasts every event to a list of publishers.

    Each publisher runs concurrently via asyncio.gather. A failure in one
    publisher is logged but does not prevent the others from receiving the event.
    """

    def __init__(self, publishers: list[BasePublisher]):
        super().__init__({})
        self._publishers = publishers

    def add_publisher(self, publisher: BasePublisher) -> None:
        """Add a publisher to the composite."""
        self._publishers.append(publisher)

    def remove_publisher(self, publisher: BasePublisher) -> None:
        """Remove a publisher from the composite."""
        self._publishers.remove(publisher)

    async def publish(self, event: EventModel) -> None:
        await asyncio.gather(
            *(p.publish(event) for p in self._publishers),
            return_exceptions=True,
        )

    async def close(self) -> None:
        await asyncio.gather(
            *(p.close() for p in self._publishers),
            return_exceptions=True,
        )

    def sync_close(self) -> None:
        for p in self._publishers:
            p.sync_close()
