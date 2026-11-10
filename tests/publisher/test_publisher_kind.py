from unittest.mock import MagicMock

import pytest

from tenxgraph.runtime.publisher import (
    BasePublisher,
    CompositePublisher,
    Event,
    EventModel,
    EventType,
    LangsmithPublisher,
    LogfirePublisher,
    OtelPublisher,
    PhoenixPublisher,
    publish_event,
)
from tenxgraph.utils.background_task_manager import BackgroundTaskManager


class Events(BasePublisher):
    def __init__(self):
        super().__init__({})
        self.got = []

    async def publish(self, event):
        self.got.append(event)

    async def close(self):
        pass

    def sync_close(self):
        pass


class Tracing(Events):
    kind = "observability"


def _event():
    return EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START, node_name="n")


async def _send(publisher, only=None):
    manager = BackgroundTaskManager()
    publish_event(_event(), publisher=publisher, task_manager=manager, only=only)
    await manager.wait_for_all()


def test_kinds():
    assert Events.kind == "events"
    for cls in (OtelPublisher, LogfirePublisher, LangsmithPublisher, PhoenixPublisher):
        assert cls.kind == "observability"


@pytest.mark.asyncio
async def test_default_reaches_every_publisher():
    ev, tr = Events(), Tracing()
    await _send(CompositePublisher([ev, tr]))
    assert len(ev.got) == 1
    assert len(tr.got) == 1


@pytest.mark.asyncio
async def test_only_events_skips_tracing_in_composite():
    ev, tr = Events(), Tracing()
    composite = CompositePublisher([ev, tr])
    await _send(composite, only="events")
    assert len(ev.got) == 1
    assert tr.got == []
    assert composite._publishers == [ev, tr]


@pytest.mark.asyncio
async def test_only_observability_skips_events_in_composite():
    ev, tr = Events(), Tracing()
    await _send(CompositePublisher([ev, tr]), only="observability")
    assert ev.got == []
    assert len(tr.got) == 1


@pytest.mark.asyncio
async def test_single_publisher_of_other_kind_spawns_no_task():
    manager = MagicMock()
    tr = Tracing()
    publish_event(_event(), publisher=tr, task_manager=manager, only="events")
    manager.create_task.assert_not_called()
    assert tr.got == []


@pytest.mark.asyncio
async def test_composite_with_no_match_spawns_no_task():
    manager = MagicMock()
    publish_event(
        _event(), publisher=CompositePublisher([Tracing()]), task_manager=manager, only="events"
    )
    manager.create_task.assert_not_called()


@pytest.mark.asyncio
async def test_nested_composite():
    ev, tr, ev2 = Events(), Tracing(), Events()
    nested = CompositePublisher([CompositePublisher([ev, tr]), ev2])
    await _send(nested, only="events")
    assert len(ev.got) == 1
    assert len(ev2.got) == 1
    assert tr.got == []
