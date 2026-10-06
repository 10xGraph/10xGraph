"""
EventCollector — stores all raw EventModel objects for debugging.

Useful when you need to inspect every event fired during a run, including
events that TrajectoryCollector doesn't process (e.g. GRAPH_EXECUTION start).
"""

from __future__ import annotations

from agentflow.runtime.publisher.events import Event, EventModel, EventType


class EventCollector:
    """Simple collector that stores all raw events for debugging and analysis.

    Example:
        ```python
        ec = EventCollector()
        # pass ec.on_event as a callback, then inspect:
        node_events = ec.filter_by_event(Event.NODE_EXECUTION)
        end_events = ec.filter_by_event_type(EventType.END)
        ```
    """

    def __init__(self) -> None:
        self.events: list[EventModel] = []

    def reset(self) -> None:
        self.events.clear()

    async def on_event(self, event: EventModel) -> None:
        self.events.append(event)

    def on_event_sync(self, event: EventModel) -> None:
        self.events.append(event)

    def filter_by_event(self, event_type: Event) -> list[EventModel]:
        return [e for e in self.events if e.event == event_type]

    def filter_by_event_type(self, event_type: EventType) -> list[EventModel]:
        return [e for e in self.events if e.event_type == event_type]

    def filter_by_node(self, node_name: str) -> list[EventModel]:
        return [e for e in self.events if e.node_name == node_name]

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"EventCollector(events={len(self.events)})"
