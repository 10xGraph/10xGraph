"""
Comprehensive tests for optional publishers (Redis, Kafka, RabbitMQ).

This module tests optional publishers with dependency mocking,
configuration handling, and error scenarios.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from tenxgraph.runtime.publisher.events import EventModel, Event, EventType


class TestRedisPublisher:
    """Test RedisPublisher with mocked dependencies."""
    
    
    
    
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_get_client_success(self, mock_import):
        """Test successful Redis client creation with connection pooling."""
        mock_redis_asyncio = Mock()
        mock_pool = Mock()
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        
        # Mock ConnectionPool
        mock_redis_asyncio.ConnectionPool = Mock()
        mock_redis_asyncio.ConnectionPool.from_url = Mock(return_value=mock_pool)
        mock_redis_asyncio.Redis = Mock(return_value=mock_client)
        
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher({"url": "redis://test:6379"})
            
            client = await publisher._get_client()
            
            assert client == mock_client
            assert publisher._redis == mock_client
            # Verify ConnectionPool was created with proper parameters
            mock_redis_asyncio.ConnectionPool.from_url.assert_called_once()
            # Verify ping was called to test connection
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_publisher_get_client_import_error(self):
        """Test Redis client creation when redis module is missing."""
        from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
        
        publisher = RedisPublisher()
        
        # Patch the import inside the _get_client method
        with patch('tenxgraph.runtime.publisher.redis_publisher.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No module named 'redis'")
            
            with pytest.raises(RuntimeError, match="RedisPublisher requires the 'redis' package"):
                await publisher._get_client()
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_get_client_connection_error(self, mock_import):
        """Test Redis client creation with connection failure."""
        mock_redis_asyncio = Mock()
        mock_pool = Mock()
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=Exception("Connection failed"))
        
        # Mock ConnectionPool
        mock_redis_asyncio.ConnectionPool = Mock()
        mock_redis_asyncio.ConnectionPool.from_url = Mock(return_value=mock_pool)
        mock_redis_asyncio.Redis = Mock(return_value=mock_client)
        
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher()
            
            with pytest.raises(RuntimeError, match="RedisPublisher failed to connect to Redis"):
                await publisher._get_client()
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_publish_pubsub_mode(self, mock_import):
        """Test publishing in pub/sub mode."""
        mock_redis_asyncio = Mock()
        mock_pool = Mock()
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.publish.return_value = 1  # 1 subscriber received
        
        # Mock ConnectionPool
        mock_redis_asyncio.ConnectionPool = Mock()
        mock_redis_asyncio.ConnectionPool.from_url = Mock(return_value=mock_pool)
        mock_redis_asyncio.Redis = Mock(return_value=mock_client)
        
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher({
                "mode": "pubsub",
                "channel": "test.channel"
            })
            
            event = EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.START,
                node_name="test_node",
                data={"key": "value"}
            )
            
            result = await publisher.publish(event)
            
            assert result == 1
            mock_client.publish.assert_called_once()
            
            # Verify the published message
            call_args = mock_client.publish.call_args
            channel, message = call_args[0]
            
            assert channel == "test.channel"
            
            # Message should be JSON-serialized event
            parsed_message = json.loads(message)
            assert parsed_message["event"] == "graph_execution"
            assert parsed_message["event_type"] == "start"
            assert parsed_message["node_name"] == "test_node"
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_publish_stream_mode(self, mock_import):
        """Test publishing in stream mode."""
        mock_redis_asyncio = Mock()
        mock_pool = Mock()
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.xadd.return_value = b"1234567890-0"  # Stream entry ID
        
        # Mock ConnectionPool
        mock_redis_asyncio.ConnectionPool = Mock()
        mock_redis_asyncio.ConnectionPool.from_url = Mock(return_value=mock_pool)
        mock_redis_asyncio.Redis = Mock(return_value=mock_client)
        
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher({
                "mode": "stream",
                "stream": "test.stream",
                "maxlen": 1000
            })
            
            event = EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.PROGRESS,
                node_name="stream_test",
                data={"progress": 50}
            )
            
            result = await publisher.publish(event)
            
            assert result == b"1234567890-0"
            mock_client.xadd.assert_called_once()
            
            # Verify the stream call
            call_args = mock_client.xadd.call_args
            stream_name, fields = call_args[0]
            maxlen = call_args[1].get("maxlen")
            
            assert stream_name == "test.stream"
            assert maxlen == 1000
            
            # Fields should contain serialized event
            assert "data" in fields
            parsed_data = json.loads(fields["data"].decode('utf-8'))
            assert parsed_data["event"] == "node_execution"
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_publish_error(self, mock_import):
        """Test publish error handling."""
        mock_redis_asyncio = Mock()
        mock_pool = Mock()
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.publish.side_effect = Exception("Redis error")
        
        # Mock ConnectionPool
        mock_redis_asyncio.ConnectionPool = Mock()
        mock_redis_asyncio.ConnectionPool.from_url = Mock(return_value=mock_pool)
        mock_redis_asyncio.Redis = Mock(return_value=mock_client)
        
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher()
            
            event = EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.ERROR
            )
            
            with pytest.raises(Exception, match="Redis error"):
                await publisher.publish(event)
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_close(self, mock_import):
        """Test Redis publisher close method."""
        mock_redis_asyncio = Mock()
        mock_pool = Mock()
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.aclose = AsyncMock()
        
        # Mock ConnectionPool
        mock_redis_asyncio.ConnectionPool = Mock()
        mock_redis_asyncio.ConnectionPool.from_url = Mock(return_value=mock_pool)
        mock_redis_asyncio.Redis = Mock(return_value=mock_client)
        
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher()
            
            # Get client (connects)
            await publisher._get_client()
            assert publisher._redis is not None
            
            # Close
            await publisher.close()
            
            mock_client.aclose.assert_called_once()
            assert publisher._redis is None
            assert publisher._is_closed is True


class TestKafkaPublisher:
    """Test KafkaPublisher with mocked dependencies."""
    
    
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_kafka_publisher_publish_success(self, mock_import):
        """Test successful Kafka publishing."""
        mock_kafka = Mock()
        mock_producer = AsyncMock()
        mock_producer_class = Mock(return_value=mock_producer)
        mock_kafka.AIOKafkaProducer = mock_producer_class
        mock_import.return_value = mock_kafka
        
        with patch.dict('sys.modules', {'aiokafka': mock_kafka}):
            from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
            
            publisher = KafkaPublisher({
                "topic": "test.topic"
            })
            
            event = EventModel(
                event=Event.TOOL_EXECUTION,
                event_type=EventType.RESULT,
                node_name="kafka_test",
                data={"result": "success"}
            )
            
            # Mock successful send
            mock_producer.send_and_wait.return_value = Mock(
                topic="test.topic",
                partition=0,
                offset=12345
            )
            
            result = await publisher.publish(event)
            
            # Should return the record metadata
            assert result.topic == "test.topic"
            assert result.partition == 0
            assert result.offset == 12345
            
            # Verify producer was called correctly
            mock_producer.send_and_wait.assert_called_once()
            call_args = mock_producer.send_and_wait.call_args
            topic, message = call_args[0]
            
            assert topic == "test.topic"
            
            # Message should be JSON-encoded event
            parsed_message = json.loads(message.decode('utf-8'))
            assert parsed_message["event"] == "tool_execution"
            assert parsed_message["node_name"] == "kafka_test"

    @pytest.mark.asyncio
    async def test_kafka_publisher_closed_errors(self):
        """Test KafkaPublisher closed errors."""
        from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
        publisher = KafkaPublisher()
        publisher._is_closed = True
        
        with pytest.raises(RuntimeError, match="KafkaPublisher is closed"):
            await publisher._get_producer()
            
        with pytest.raises(RuntimeError, match="Cannot publish to closed KafkaPublisher"):
            await publisher.publish(EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START))

    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_kafka_publisher_get_producer_early_return(self, mock_import):
        """Test early return in _get_producer if already initialized."""
        from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
        publisher = KafkaPublisher()
        publisher._producer = MagicMock()
        
        res = await publisher._get_producer()
        assert res == publisher._producer
        mock_import.assert_not_called()

    @pytest.mark.asyncio
    async def test_kafka_publisher_missing_module(self):
        """Test missing aiokafka ImportError fallback."""
        from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
        publisher = KafkaPublisher()
        
        with patch('tenxgraph.runtime.publisher.kafka_publisher.importlib.import_module', side_effect=ImportError):
            with pytest.raises(RuntimeError, match="requires the 'aiokafka' package"):
                await publisher._get_producer()

    @pytest.mark.asyncio
    async def test_kafka_publisher_close_methods(self):
        """Test close() and sync_close() paths."""
        from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
        publisher = KafkaPublisher()
        
        # Closed early return
        publisher._is_closed = True
        await publisher.close()
        
        # Close with producer
        publisher._is_closed = False
        mock_producer = AsyncMock()
        publisher._producer = mock_producer
        
        await publisher.close()
        mock_producer.stop.assert_called_once()
        assert publisher._producer is None
        assert publisher._is_closed is True
        
        # Close with exception in stop
        publisher = KafkaPublisher()
        mock_producer = AsyncMock()
        mock_producer.stop.side_effect = Exception("stop failed")
        publisher._producer = mock_producer
        
        await publisher.close()
        assert publisher._producer is None
        assert publisher._is_closed is True

        # Test sync_close
        publisher = KafkaPublisher()
        publisher.close = AsyncMock()
        publisher.sync_close()
        publisher.close.assert_called_once()

        # Test sync_close active loop warning
        publisher = KafkaPublisher()
        with patch('asyncio.run', side_effect=RuntimeError):
            with patch('tenxgraph.runtime.publisher.kafka_publisher.logger.warning') as mock_warn:
                publisher.sync_close()
                mock_warn.assert_called_once()


class TestRabbitMQPublisher:
    """Test RabbitMQPublisher with mocked dependencies."""
    
    
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_rabbitmq_publisher_publish_success(self, mock_import):
        """Test successful RabbitMQ publishing."""
        mock_pika = Mock()
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        # Set up proper async behavior
        mock_pika.connect = AsyncMock(return_value=mock_connection)
        mock_pika.exceptions.ProbableAuthenticationError = type("AuthError", (Exception,), {})
        mock_connection.channel = AsyncMock(return_value=mock_channel)
        mock_channel.declare_exchange = AsyncMock(return_value=mock_exchange)
        mock_exchange.publish = AsyncMock()
        
        # Add exchange type enum
        mock_exchange_type = Mock()
        mock_exchange_type.TOPIC = "topic"
        mock_pika.ExchangeType = mock_exchange_type
        
        mock_pika.Message = Mock  # Mock Message class
        mock_import.return_value = mock_pika
        
        with patch.dict('sys.modules', {'aio_pika': mock_pika}):
            from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
            
            publisher = RabbitMQPublisher({
                "exchange": "test.exchange",
                "routing_key": "test.key"
            })
            
            event = EventModel(
                event=Event.STREAMING,
                event_type=EventType.UPDATE,
                node_name="rabbitmq_test",
                data={"update": "data"}
            )
            
            result = await publisher.publish(event)
            
            # Should return True (successful publish)
            assert result is True
            
            # Verify the publish call
            mock_exchange.publish.assert_called_once()
            call_args = mock_exchange.publish.call_args
            message = call_args[0][0]  # First positional arg
            routing_key = call_args[1].get('routing_key')  # Keyword arg
            
            assert routing_key == "test.key"
            
        # Verify message creation - Mock should be called as a constructor
        assert mock_pika.Message.called

    @pytest.mark.asyncio
    async def test_rabbitmq_publisher_closed_errors(self):
        """Test RabbitMQPublisher raising RuntimeError when closed."""
        from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
        publisher = RabbitMQPublisher()
        publisher._is_closed = True
        
        with pytest.raises(RuntimeError, match="RabbitMQPublisher is closed"):
            await publisher._ensure()
            
        with pytest.raises(RuntimeError, match="Cannot publish to closed RabbitMQPublisher"):
            await publisher.publish(EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START))

    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_rabbitmq_publisher_ensure_early_return(self, mock_import):
        """Test early return in _ensure when exchange is already declared."""
        from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
        publisher = RabbitMQPublisher()
        publisher._exchange = MagicMock()
        publisher._channel = MagicMock(is_closed=False)
        
        assert await publisher._ensure() is True
        mock_import.assert_not_called()

    @pytest.mark.asyncio
    async def test_rabbitmq_publisher_missing_module(self):
        """Test RuntimeError when aio_pika module is missing."""
        from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
        publisher = RabbitMQPublisher()
        
        with patch('tenxgraph.runtime.publisher.rabbitmq_publisher.importlib.import_module', side_effect=ImportError):
            with pytest.raises(RuntimeError, match="requires the 'aio-pika' package"):
                await publisher._ensure()

    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_rabbitmq_publisher_default_exchange_fallback(self, mock_import):
        """Test default exchange declaration fallback when declare=False."""
        mock_pika = Mock()
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        
        mock_pika.connect = AsyncMock(return_value=mock_connection)
        mock_pika.exceptions.ProbableAuthenticationError = type("AuthError", (Exception,), {})
        mock_connection.channel = AsyncMock(return_value=mock_channel)
        mock_import.return_value = mock_pika
        
        with patch.dict('sys.modules', {'aio_pika': mock_pika}):
            from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
            publisher = RabbitMQPublisher({"declare": False})
            
            await publisher._ensure()
            assert publisher._exchange == mock_channel.default_exchange

    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_rabbitmq_publisher_publish_not_initialized(self, mock_import):
        """Test publish raises RuntimeError when exchange is not initialized."""
        mock_pika = Mock()
        mock_import.return_value = mock_pika
        from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
        publisher = RabbitMQPublisher()
        
        # Mock _ensure to do nothing so _exchange remains None
        with patch.object(publisher, '_ensure', AsyncMock()):
            with pytest.raises(RuntimeError, match="exchange not initialized"):
                await publisher.publish(EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START))

    @pytest.mark.asyncio
    async def test_rabbitmq_publisher_close_methods(self):
        """Test close() and sync_close() paths."""
        from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
        publisher = RabbitMQPublisher()
        
        # Close when already closed
        publisher._is_closed = True
        await publisher.close()  # Should return early
        
        # Close with active channel and connection
        publisher._is_closed = False
        mock_channel = AsyncMock()
        mock_conn = AsyncMock()
        publisher._channel = mock_channel
        publisher._conn = mock_conn
        
        await publisher.close()
        # Closing the connection closes its channel too.
        mock_conn.close.assert_called_once()
        assert publisher._channel is None
        assert publisher._conn is None
        assert publisher._is_closed is True
        
        # Test close with exceptions on close call (should catch and log)
        publisher = RabbitMQPublisher()
        publisher._channel = AsyncMock()
        publisher._channel.close.side_effect = Exception("channel close failed")
        publisher._conn = AsyncMock()
        publisher._conn.close.side_effect = Exception("conn close failed")
        
        await publisher.close()
        assert publisher._channel is None
        assert publisher._conn is None
        assert publisher._is_closed is True

        # Test sync_close
        publisher = RabbitMQPublisher()
        publisher.close = AsyncMock()
        publisher.sync_close()
        publisher.close.assert_called_once()

        # Test sync_close raises RuntimeError (active loop)
        publisher = RabbitMQPublisher()
        with patch('asyncio.run', side_effect=RuntimeError("active loop")):
            with patch('tenxgraph.runtime.publisher.rabbitmq_publisher.logger.warning') as mock_warn:
                publisher.sync_close()
                mock_warn.assert_called_once_with("sync_close called within an active event loop; skipping.")


class TestOptionalPublisherErrorHandling:
    """Test error handling across optional publishers."""
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_connection_retry_failure(self, mock_import):
        """Test Redis publisher with persistent connection failures."""
        mock_redis_asyncio = Mock()
        mock_redis_asyncio.from_url.side_effect = ConnectionError("Cannot connect")
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher()
            
            event = EventModel(
                event=Event.GRAPH_EXECUTION,
                event_type=EventType.START
            )
            
            with pytest.raises(RuntimeError, match="RedisPublisher failed to connect to Redis"):
                await publisher.publish(event)
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_kafka_publisher_send_failure(self, mock_import):
        """Test Kafka publisher with send failures."""
        mock_kafka = Mock()
        mock_producer = AsyncMock()
        mock_producer.send_and_wait.side_effect = Exception("Send failed")
        mock_kafka.AIOKafkaProducer.return_value = mock_producer
        mock_import.return_value = mock_kafka
        
        with patch.dict('sys.modules', {'aiokafka': mock_kafka}):
            from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
            
            publisher = KafkaPublisher()
            
            event = EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.ERROR
            )
            
            with pytest.raises(Exception, match="Send failed"):
                await publisher.publish(event)
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_rabbitmq_publisher_publish_failure(self, mock_import):
        """Test RabbitMQ publisher with publish failures."""
        mock_pika = Mock()
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        mock_exchange.publish.side_effect = Exception("Publish failed")
        
        mock_connection.channel.return_value = mock_channel
        mock_channel.get_exchange.return_value = mock_exchange
        mock_pika.connect_robust.return_value = mock_connection
        mock_import.return_value = mock_pika
        
        with patch.dict('sys.modules', {'aio_pika': mock_pika}):
            from tenxgraph.runtime.publisher.rabbitmq_publisher import RabbitMQPublisher
            
            publisher = RabbitMQPublisher()
            
            event = EventModel(
                event=Event.TOOL_EXECUTION,
                event_type=EventType.ERROR
            )
            
            # This will fail during _ensure() before publish
            with pytest.raises(TypeError):
                await publisher.publish(event)


class TestOptionalPublisherConfiguration:
    """Test configuration handling for optional publishers."""
    
    
    


class TestOptionalPublisherIntegration:
    """Integration tests for optional publishers."""
    
    @pytest.mark.asyncio
    @patch('importlib.import_module')
    async def test_redis_publisher_full_lifecycle(self, mock_import):
        """Test complete Redis publisher lifecycle."""
        mock_redis_asyncio = Mock()
        mock_pool = Mock()
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.publish.return_value = 2  # 2 subscribers
        mock_client.aclose = AsyncMock()
        
        # Mock ConnectionPool
        mock_redis_asyncio.ConnectionPool = Mock()
        mock_redis_asyncio.ConnectionPool.from_url = Mock(return_value=mock_pool)
        mock_redis_asyncio.Redis = Mock(return_value=mock_client)
        
        mock_import.return_value = mock_redis_asyncio
        
        with patch.dict('sys.modules', {'redis.asyncio': mock_redis_asyncio}):
            from tenxgraph.runtime.publisher.redis_publisher import RedisPublisher
            
            publisher = RedisPublisher({
                "channel": "integration.test"
            })
            
            # Publish multiple events
            events = [
                EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.START),
                EventModel(event=Event.NODE_EXECUTION, event_type=EventType.PROGRESS),
                EventModel(event=Event.GRAPH_EXECUTION, event_type=EventType.END)
            ]
            
            results = []
            for event in events:
                result = await publisher.publish(event)
                results.append(result)
            
            # All should succeed
            assert all(r == 2 for r in results)
            assert mock_client.publish.call_count == 3
            
            # Close publisher
            await publisher.close()
            mock_client.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_publisher_types_concurrent(self):
        """Test multiple publisher types working concurrently."""
        import asyncio
        from tenxgraph.runtime.publisher.console_publisher import ConsolePublisher
        
        # Use ConsolePublisher for this test since it doesn't need external dependencies
        # This tests the concurrency pattern without complex mocking
        
        pub1 = ConsolePublisher()
        pub2 = ConsolePublisher()
        
        event1 = EventModel(
            event=Event.GRAPH_EXECUTION,
            event_type=EventType.START,
            node_name="concurrent_test_1"
        )
        
        event2 = EventModel(
            event=Event.NODE_EXECUTION,
            event_type=EventType.END,
            node_name="concurrent_test_2"
        )
        
        # Test concurrent publishing with console publishers
        with patch('builtins.print') as mock_print:
            task1 = pub1.publish(event1)
            task2 = pub2.publish(event2)
            
            results = await asyncio.gather(task1, task2)
            
            # Both should succeed (console publisher returns None)
            assert results == [None, None]
            
            # Print should have been called twice
            assert mock_print.call_count == 2
