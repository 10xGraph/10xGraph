"""Kafka, Redis and RabbitMQ publishers recover from a failed start (G-73, D-07)."""

import asyncio
import logging
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.runtime.publisher.events import Event, EventModel, EventType
from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
from tenxgraph.utils import END


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


class _AuthError(Exception):
    """Stands in for aio_pika.exceptions.ProbableAuthenticationError."""


def _fake_aio_pika(conn):
    fake = MagicMock()
    fake.connect = AsyncMock(return_value=conn)
    fake.exceptions.ProbableAuthenticationError = _AuthError
    fake.Message = lambda body: body
    return fake


@pytest.mark.asyncio
async def test_rabbitmq_failed_channel_open_closes_connection(monkeypatch):
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
    async def stall(*a, **k):
        await asyncio.sleep(30)

    exchange = MagicMock()
    exchange.publish = stall
    channel = MagicMock(is_closed=False)
    channel.declare_exchange = AsyncMock(return_value=exchange)
    conn = MagicMock()
    conn.channel = AsyncMock(return_value=channel)
    fake = _fake_aio_pika(conn)
    monkeypatch.setitem(sys.modules, "aio_pika", fake)

    pub = RabbitMQPublisher({"publish_timeout": 0.05})
    with pytest.raises(asyncio.TimeoutError):
        await pub.publish(_event())


# D-07: the RabbitMQ publisher must not retry refused credentials forever, and close() must
# return promptly whatever state the connection is in.


@pytest.mark.asyncio
async def test_rabbitmq_refused_credentials_fail_fast_and_log_once(monkeypatch, caplog):
    fake = _fake_aio_pika(MagicMock())
    fake.connect = AsyncMock(side_effect=_AuthError("ACCESS_REFUSED"))
    monkeypatch.setitem(sys.modules, "aio_pika", fake)

    pub = RabbitMQPublisher({"url": "amqp://guest:wrong@broker:5672/"})
    with caplog.at_level(logging.ERROR, logger="tenxgraph.publisher"):
        for _ in range(5):
            assert await pub.publish(_event()) is False

    fake.connect.assert_awaited_once()
    refused = [r for r in caplog.records if "refused the credentials" in r.getMessage()]
    assert len(refused) == 1
    assert "wrong" not in refused[0].getMessage()
    await pub.close()


@pytest.mark.asyncio
async def test_rabbitmq_network_failure_waits_before_next_attempt(monkeypatch):
    fake = _fake_aio_pika(MagicMock())
    fake.connect = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    monkeypatch.setitem(sys.modules, "aio_pika", fake)

    pub = RabbitMQPublisher({"reconnect_interval": 0.1})
    with pytest.raises(ConnectionRefusedError):
        await pub.publish(_event())
    # Inside the interval the broker is not contacted again.
    with pytest.raises(RuntimeError, match="next attempt"):
        await pub.publish(_event())
    assert fake.connect.await_count == 1

    await asyncio.sleep(0.15)
    with pytest.raises(ConnectionRefusedError):
        await pub.publish(_event())
    assert fake.connect.await_count == 2
    await pub.close()


@pytest.mark.asyncio
async def test_rabbitmq_close_returns_while_connect_hangs(monkeypatch):
    async def hang(*a, **k):
        await asyncio.Event().wait()

    fake = _fake_aio_pika(MagicMock())
    fake.connect = hang
    monkeypatch.setitem(sys.modules, "aio_pika", fake)

    pub = RabbitMQPublisher({"connection_timeout": 0.3})
    publishing = asyncio.create_task(pub.publish(_event()))
    await asyncio.sleep(0.05)  # the publish now holds the connection lock

    await asyncio.wait_for(pub.close(), timeout=0.1)
    assert pub._is_closed
    # The stuck attempt is bounded by connection_timeout, and leaves nothing open.
    with pytest.raises((TimeoutError, RuntimeError)):
        await asyncio.wait_for(publishing, timeout=1)
    assert pub._conn is None


@pytest.mark.asyncio
async def test_rabbitmq_reconnects_after_broker_closed_connection(monkeypatch):
    def make_conn():
        exchange = MagicMock(publish=AsyncMock())
        channel = MagicMock(is_closed=False, declare_exchange=AsyncMock(return_value=exchange))
        return MagicMock(channel=AsyncMock(return_value=channel), close=AsyncMock())

    first, second = make_conn(), make_conn()
    fake = _fake_aio_pika(first)
    fake.connect = AsyncMock(side_effect=[first, second])
    monkeypatch.setitem(sys.modules, "aio_pika", fake)

    pub = RabbitMQPublisher({})
    assert await pub.publish(_event()) is True
    pub._channel.is_closed = True  # broker dropped the connection
    assert await pub.publish(_event()) is True
    assert fake.connect.await_count == 2
    first.close.assert_awaited_once()
    await pub.close()
    second.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_run_and_aclose_with_refused_rabbitmq_credentials(monkeypatch):
    fake = _fake_aio_pika(MagicMock())
    fake.connect = AsyncMock(side_effect=_AuthError("ACCESS_REFUSED"))
    monkeypatch.setitem(sys.modules, "aio_pika", fake)

    def node(state: AgentState):
        return state

    graph = StateGraph(publisher=RabbitMQPublisher({"url": "amqp://guest:wrong@broker/"}))
    graph.add_node("MAIN", node)
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)
    app = graph.compile()

    result = await asyncio.wait_for(
        app.ainvoke({"messages": [Message.text_message("hi")]}, config={"thread_id": "d07"}),
        timeout=5,
    )
    assert result["messages"]
    start = time.monotonic()
    await asyncio.wait_for(app.aclose(), timeout=5)
    assert time.monotonic() - start < 2
    fake.connect.assert_awaited_once()
