"""Broker publishers encode an event once with pydantic; consumers parse the same JSON.

They used ``json.dumps(event.model_dump())``, which builds a dict copy of the event first.
``model_dump_json`` gives the same parsed value without it. A publish is timed per publisher.
"""

import json
from unittest.mock import AsyncMock

import pytest

from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.runtime.publisher.events import ContentType, Event, EventModel, EventType
from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
from tenxgraph.runtime.publisher.publish import _publish_event_task
from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
from tenxgraph.utils import metrics


def _event() -> EventModel:
    state = AgentState(context=[Message.text_message('héllo ☃ "quoted"\n')])
    event = EventModel.default(
        {"thread_id": "t1", "run_id": "r1"},
        data={"state": state.model_dump(), "step": 1},
        event=Event.NODE_EXECUTION,
        content_type=[ContentType.STATE],
        extra={"current_node": "MAIN"},
    )
    event.event_type = EventType.UPDATE
    event.content_blocks = state.context[0].content
    return event


def _old_payload(event: EventModel) -> object:
    return json.loads(json.dumps(event.model_dump()))


@pytest.mark.asyncio
async def test_kafka_payload_parses_to_the_same_value():
    publisher = KafkaPublisher({"topic": "events"})
    producer = AsyncMock()
    publisher._get_producer = AsyncMock(return_value=producer)
    event = _event()

    await publisher.publish(event)

    payload = producer.send_and_wait.call_args[0][1]
    assert json.loads(payload.decode("utf-8")) == _old_payload(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["utf-8", "ascii"])
async def test_redis_payload_parses_to_the_same_value(encoding):
    publisher = RedisPublisher({"channel": "events", "encoding": encoding})
    client = AsyncMock()
    publisher._get_client = AsyncMock(return_value=client)
    event = _event()

    await publisher.publish(event)

    payload = client.publish.call_args[0][1]
    assert json.loads(payload.decode(encoding)) == _old_payload(event)


@pytest.mark.asyncio
async def test_publish_is_timed_per_publisher():
    name = "tenxgraph.publish.duration"
    before = metrics.snapshot()["timers"].get(name, {}).get("count", 0)
    publisher = AsyncMock(spec=BasePublisher)
    publisher.config = {}

    await _publish_event_task(_event(), publisher)

    assert metrics.snapshot()["timers"][name]["count"] == before + 1
