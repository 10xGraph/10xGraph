"""Batched delivery of the broker publishers (``_buffered.BufferedPublisher``)."""

from __future__ import annotations

import asyncio
import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from injectq import InjectQ

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher._buffered import BufferedPublisher
from tenxgraph.runtime.publisher.events import Event, EventModel, EventType
from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
from tenxgraph.runtime.publisher.publish import publish_event
from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
from tenxgraph.utils import END, metrics


class Fake(BufferedPublisher):
    def __init__(self, config=None, delay=0.0):
        super().__init__(config or {})
        self.batches: list[list[int]] = []
        self.delay = delay

    async def publish_batch(self, events):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.batches.append([e.metadata.get("i") for e in events])

    async def publish(self, event):
        await self.publish_batch([event])

    async def close(self):
        await self.stop_buffer()
        self._is_closed = True

    def sync_close(self):
        pass


def _event(i: int = 0, thread_id: str = "t1") -> EventModel:
    return EventModel(
        event=Event.NODE_EXECUTION,
        event_type=EventType.END,
        thread_id=thread_id,
        metadata={"i": i},
    )


async def _settle(seconds: float = 0.05) -> None:
    await asyncio.sleep(seconds)


@pytest.mark.asyncio
async def test_flushes_when_a_batch_is_full():
    pub = Fake({"batch_size": 3, "flush_interval_ms": 10_000})
    for i in range(3):
        pub.enqueue(_event(i))
    await _settle()
    assert pub.batches == [[0, 1, 2]]
    await pub.close()


@pytest.mark.asyncio
async def test_flushes_after_the_interval():
    pub = Fake({"batch_size": 100, "flush_interval_ms": 10})
    pub.enqueue(_event(0))
    pub.enqueue(_event(1))
    assert pub.batches == []
    await _settle()
    assert pub.batches == [[0, 1]]
    await pub.close()


@pytest.mark.asyncio
async def test_order_is_kept_across_batches():
    pub = Fake({"batch_size": 100, "flush_interval_ms": 5}, delay=0.001)
    for i in range(250):
        pub.enqueue(_event(i))
    await pub.flush()
    assert [len(b) for b in pub.batches] == [100, 100, 50]
    assert [i for b in pub.batches for i in b] == list(range(250))
    await pub.close()


@pytest.mark.asyncio
async def test_queue_full_drops_counts_and_warns_once(caplog):
    pub = Fake({"queue_size": 2, "flush_interval_ms": 10_000})
    before = metrics.counter("tenxgraph.publish.dropped").value
    with caplog.at_level(logging.WARNING, logger="tenxgraph.publisher"):
        for i in range(5):
            pub.enqueue(_event(i))
    assert pub.dropped == 3
    assert metrics.counter("tenxgraph.publish.dropped").value - before == 3
    warnings = [r for r in caplog.records if "dropped an event" in r.getMessage()]
    assert len(warnings) == 1
    await pub.flush()
    assert pub.batches == [[0, 1]]
    await pub.close()


@pytest.mark.asyncio
async def test_publish_timeout_bounds_one_batch():
    pub = Fake({"publish_timeout": 0.01, "flush_interval_ms": 1}, delay=1.0)
    before = metrics.counter("tenxgraph.publish.dropped").value
    pub.enqueue(_event(0))
    pub.enqueue(_event(1))
    await pub.flush()
    assert pub.batches == []
    assert metrics.counter("tenxgraph.publish.dropped").value - before == 2
    await pub.close()


@pytest.mark.asyncio
async def test_batch_metrics_are_counted():
    pub = Fake({"flush_interval_ms": 1})
    batches = metrics.counter("tenxgraph.publish.batches").value
    sent = metrics.counter("tenxgraph.publish.events").value
    for i in range(3):
        pub.enqueue(_event(i))
    await pub.flush()
    assert metrics.counter("tenxgraph.publish.batches").value - batches == 1
    assert metrics.counter("tenxgraph.publish.events").value - sent == 3
    await pub.close()


@pytest.mark.asyncio
async def test_publish_event_enqueues_without_a_task():
    pub = Fake({"flush_interval_ms": 10_000})
    task_manager = MagicMock()
    publish_event(_event(7), pub, task_manager)
    task_manager.create_task.assert_not_called()
    assert len(pub._queue) == 1
    await pub.close()
    assert pub.batches == [[7]]


@pytest.mark.asyncio
async def test_closed_publisher_takes_no_events():
    pub = Fake()
    await pub.close()
    pub.enqueue(_event())
    assert not pub._queue


def test_worker_restarts_on_a_new_event_loop():
    pub = Fake({"flush_interval_ms": 1})

    async def one(i):
        pub.enqueue(_event(i))
        await _settle(0.02)

    asyncio.run(one(1))
    first_worker = pub._worker
    asyncio.run(one(2))
    assert pub._worker is not first_worker
    assert pub.batches == [[1], [2]]


def test_events_queued_when_a_loop_ends_go_out_on_the_next():
    pub = Fake({"flush_interval_ms": 10_000})

    async def put():
        pub.enqueue(_event(1))

    async def flush():
        await pub.flush()

    asyncio.run(put())  # the worker is cancelled with the loop before it sends
    assert pub.batches == []
    asyncio.run(flush())
    assert pub.batches == [[1]]


def _graph(publisher):
    def node(state: AgentState):
        return Message.text_message("ok", role="assistant")

    graph = StateGraph(container=InjectQ(), publisher=publisher)
    graph.add_node("A", node)
    graph.add_edge("A", END)
    graph.set_entry_point("A")
    return graph.compile()


def test_sync_invoke_flushes_before_its_loop_closes():
    pub = Fake({"flush_interval_ms": 10_000})
    _graph(pub).invoke({"messages": [Message.text_message("hi")]}, {"thread_id": "s1"})
    sent = [i for b in pub.batches for i in b]
    assert len(sent) == 3  # run.started, node.completed, run.completed
    assert pub._worker is None


def test_sync_stream_flushes_before_its_loop_closes():
    pub = Fake({"flush_interval_ms": 10_000})
    list(_graph(pub).stream({"messages": [Message.text_message("hi")]}, {"thread_id": "s2"}))
    assert sum(len(b) for b in pub.batches) == 3


@pytest.mark.asyncio
async def test_aclose_flushes_the_queue():
    pub = Fake({"flush_interval_ms": 10_000})
    app = _graph(pub)
    await app.ainvoke({"messages": [Message.text_message("hi")]}, {"thread_id": "s3"})
    assert pub.batches == []
    await app.aclose()
    assert sum(len(b) for b in pub.batches) == 3


# ── per-broker sends ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rabbitmq_sends_a_batch_together():
    pika = ModuleType("aio_pika")
    pika.Message = lambda body: SimpleNamespace(body=body)  # type: ignore[attr-defined]
    pub = RabbitMQPublisher({"routing_key": "rk"})
    exchange = MagicMock()
    pending: list[asyncio.Future] = []

    def publish(message, routing_key):
        fut = asyncio.get_running_loop().create_future()
        pending.append(fut)
        return fut

    exchange.publish = publish
    pub._ensure = AsyncMock(return_value=True)  # type: ignore[method-assign]
    pub._exchange = exchange
    with patch.dict(sys.modules, {"aio_pika": pika}):
        task = asyncio.create_task(pub.publish_batch([_event(0), _event(1), _event(2)]))
        await asyncio.sleep(0)
        # Every message went out before any confirm arrived: one wait for the batch.
        assert len(pending) == 3
        assert not task.done()
        for fut in pending:
            fut.set_result(None)
        assert await task is True


@pytest.mark.asyncio
async def test_kafka_sends_then_waits_once_keyed_by_thread():
    pub = KafkaPublisher({"topic": "events"})
    producer = MagicMock()
    sends = []

    async def send(topic, value, key=None):
        sends.append((topic, key))
        fut = asyncio.get_running_loop().create_future()
        fut.set_result("meta")
        return fut

    producer.send = send
    pub._get_producer = AsyncMock(return_value=producer)  # type: ignore[method-assign]
    result = await pub.publish_batch([_event(0, "ta"), _event(1, "tb"), _event(2, "")])
    assert sends == [("events", b"ta"), ("events", b"tb"), ("events", None)]
    assert result == ["meta", "meta", "meta"]


@pytest.mark.asyncio
async def test_kafka_single_publish_uses_the_thread_key():
    pub = KafkaPublisher({"topic": "events"})
    producer = MagicMock()
    producer.send_and_wait = AsyncMock(return_value="ok")
    pub._get_producer = AsyncMock(return_value=producer)  # type: ignore[method-assign]
    await pub.publish(_event(0, "ta"))
    assert producer.send_and_wait.call_args.kwargs["key"] == b"ta"


@pytest.mark.asyncio
async def test_redis_sends_a_batch_in_one_pipeline_without_transaction():
    pub = RedisPublisher({"channel": "ch"})
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, 1])
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.pipeline = MagicMock(return_value=pipe)
    pub._get_client = AsyncMock(return_value=client)  # type: ignore[method-assign]
    assert await pub.publish_batch([_event(0), _event(1)]) == [1, 1]
    client.pipeline.assert_called_once_with(transaction=False)
    assert [c.args[0] for c in pipe.publish.call_args_list] == ["ch", "ch"]
    pipe.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_redis_stream_mode_batches_xadd():
    pub = RedisPublisher({"mode": "stream", "stream": "st", "maxlen": 10})
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=["1-0"])
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.pipeline = MagicMock(return_value=pipe)
    pub._get_client = AsyncMock(return_value=client)  # type: ignore[method-assign]
    await pub.publish_batch([_event(0)])
    assert pipe.xadd.call_args.args[0] == "st"
    assert pipe.xadd.call_args.kwargs == {"maxlen": 10, "approximate": True}


@pytest.mark.asyncio
async def test_broker_close_flushes_first():
    pub = RedisPublisher({"flush_interval_ms": 10_000})
    pub.publish_batch = AsyncMock()  # type: ignore[method-assign]
    pub.enqueue(_event(0))
    await pub.close()
    pub.publish_batch.assert_awaited_once()
