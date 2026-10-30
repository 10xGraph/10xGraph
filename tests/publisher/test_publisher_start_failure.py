"""Kafka and Redis publishers retry after a failed start and stay closed once closed (G-73)."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.runtime.publisher.events import Event, EventModel, EventType
from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher


def _event() -> EventModel:
    return EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START)


@pytest.mark.asyncio
async def test_kafka_retries_after_failed_first_start(monkeypatch):
    producers = []

    def make(**kwargs):
        p = MagicMock()
        p.start = AsyncMock(side_effect=ConnectionError("down") if not producers else None)
        p.stop = AsyncMock()
        p.send_and_wait = AsyncMock(return_value="ok")
        producers.append(p)
        return p

    fake = MagicMock()
    fake.AIOKafkaProducer = make
    monkeypatch.setitem(sys.modules, "aiokafka", fake)

    pub = KafkaPublisher({"bootstrap_servers": "x:9092"})
    with pytest.raises(ConnectionError):
        await pub.publish(_event())
    assert await pub.publish(_event()) == "ok"
    assert len(producers) == 2


@pytest.mark.asyncio
async def test_kafka_close_without_connect_marks_closed():
    pub = KafkaPublisher({"bootstrap_servers": "x:9092"})
    await pub.close()
    with pytest.raises(RuntimeError, match="closed"):
        await pub.publish(_event())
    with pytest.raises(RuntimeError, match="closed"):
        await pub._get_producer()


@pytest.mark.asyncio
async def test_redis_failed_ping_is_not_cached(monkeypatch):
    clients = []

    def make_client(connection_pool=None):
        c = MagicMock()
        c.ping = AsyncMock(side_effect=ConnectionError("down") if not clients else None)
        c.aclose = AsyncMock()
        c.publish = AsyncMock(return_value=1)
        clients.append(c)
        return c

    fake = MagicMock()
    fake.Redis = make_client
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake)
    monkeypatch.setitem(sys.modules, "redis", MagicMock(asyncio=fake))

    pub = RedisPublisher({})
    with pytest.raises(RuntimeError):
        await pub.publish(_event())
    clients[0].aclose.assert_awaited()
    assert await pub.publish(_event()) == 1
    assert len(clients) == 2


@pytest.mark.asyncio
async def test_redis_close_without_connect_marks_closed():
    pub = RedisPublisher({})
    await pub.close()
    with pytest.raises(RuntimeError, match="closed"):
        await pub.publish(_event())


def _fake_aio_pika(conn):
    fake = MagicMock()
    fake.connect_robust = AsyncMock(return_value=conn)
    fake.Message = lambda body: body
    return fake


@pytest.mark.asyncio
async def test_rabbitmq_failed_channel_open_closes_connection(monkeypatch):
    from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher

    conn = MagicMock()
    conn.channel = AsyncMock(side_effect=ConnectionError("no channel"))
    conn.close = AsyncMock()
    monkeypatch.setitem(sys.modules, "aio_pika", _fake_aio_pika(conn))

    pub = RabbitMQPublisher({})
    with pytest.raises(ConnectionError):
        await pub.publish(_event())
    conn.close.assert_awaited_once()
    assert pub._conn is None and pub._exchange is None


@pytest.mark.asyncio
async def test_rabbitmq_publish_timeout_bounds_the_send(monkeypatch):
    import asyncio

    from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher

    async def stall(*a, **k):
        await asyncio.sleep(30)

    exchange = MagicMock()
    exchange.publish = stall
    channel = MagicMock()
    channel.declare_exchange = AsyncMock(return_value=exchange)
    conn = MagicMock()
    conn.channel = AsyncMock(return_value=channel)
    fake = _fake_aio_pika(conn)
    monkeypatch.setitem(sys.modules, "aio_pika", fake)

    pub = RabbitMQPublisher({"publish_timeout": 0.05})
    with pytest.raises(asyncio.TimeoutError):
        await pub.publish(_event())
