from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.runtime.publisher.events import Event, EventModel, EventType
from tenxgraph.runtime.publisher.publish import _publish_event_task, publish_event
from tenxgraph.utils.background_task_manager import BackgroundTaskManager


@pytest.mark.asyncio
async def test_publish_event_task_success():
    event = MagicMock(spec=EventModel)
    publisher = AsyncMock(spec=BasePublisher)

    await _publish_event_task(event, publisher)

    publisher.publish.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_publish_event_task_failure():
    event = MagicMock(spec=EventModel)
    publisher = AsyncMock(spec=BasePublisher)
    publisher.publish.side_effect = RuntimeError("publish boom")

    await _publish_event_task(event, publisher)

    publisher.publish.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_publish_event_task_no_publisher():
    event = MagicMock(spec=EventModel)
    await _publish_event_task(event, None)


def test_publish_event():
    from tenxgraph.runtime.publisher.events import Event, EventType

    event = EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START)
    publisher = MagicMock(spec=BasePublisher)
    task_manager = MagicMock(spec=BackgroundTaskManager)

    publish_event(event, publisher, task_manager)
    task_manager.create_task.assert_called_once()


# G-21: queued events are snapshots, not the live object
def test_publish_event_queues_a_snapshot():
    from tenxgraph.runtime.publisher.events import Event, EventType

    event = EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START)
    event.metadata["step"] = 0
    captured = []
    task_manager = MagicMock(spec=BackgroundTaskManager)

    def _capture(coro):
        captured.append(coro.cr_frame.f_locals["event"])
        coro.close()

    task_manager.create_task.side_effect = _capture
    publisher = MagicMock(spec=BasePublisher)

    publish_event(event, publisher, task_manager)
    event.event_type = EventType.END
    event.metadata["step"] = 5
    event.data["state"] = {"later": True}

    assert captured[0].event_type == EventType.START
    assert captured[0].metadata["step"] == 0
    assert "state" not in captured[0].data


# G-74: optional developer-set per-event timeout
@pytest.mark.asyncio
async def test_publish_timeout_drops_stalled_event():
    import asyncio

    class Slow(BasePublisher):
        async def publish(self, event):
            await asyncio.sleep(5)

        async def close(self):
            pass

        def sync_close(self):
            pass

    publisher = Slow({"publish_timeout": 0.05})
    loop = asyncio.get_running_loop()
    start = loop.time()
    await _publish_event_task(MagicMock(spec=EventModel), publisher)
    assert loop.time() - start < 2


# G-74: queued events carry a bounded state snapshot
def _capture_publish(event, config=None):
    captured = []
    task_manager = MagicMock(spec=BackgroundTaskManager)

    def _capture(coro):
        captured.append(coro.cr_frame.f_locals["event"])
        coro.close()

    task_manager.create_task.side_effect = _capture
    publisher = MagicMock(spec=BasePublisher)
    publisher.config = config or {}
    publish_event(event, publisher, task_manager)
    return captured[0]


def _big_state(n=500):
    return {
        "context": [
            {"message_id": f"m{i}", "role": "system" if i == 0 else "user", "content": "x" * 1000}
            for i in range(n)
        ],
        "context_summary": "s" * 5000,
        "execution_meta": {"current_node": "A"},
        "extra": 1,
    }


def test_large_state_is_replaced_by_bounded_summary():
    event = EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START)
    event.data["state"] = _big_state()
    snap = _capture_publish(event)
    state = snap.data["state"]
    assert state["truncated"] is True
    assert state["context_count"] == 500
    assert len(state["context"]) == 50
    assert len(state["context_summary"]) == 1000
    assert state["execution_meta"] == {"current_node": "A"}
    assert "extra" in state["fields"]
    assert len(str(state)) < 20000


def test_small_state_and_disabled_limit_keep_full_state():
    event = EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START)
    event.data["state"] = {"context": [], "k": 1}
    assert _capture_publish(event).data["state"] == {"context": [], "k": 1}
    big = _big_state()
    event.data["state"] = big
    assert _capture_publish(event, {"state_snapshot_limit": 0}).data["state"] == big
