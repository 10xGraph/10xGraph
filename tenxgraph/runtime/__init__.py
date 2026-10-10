"""Runtime components for Agentflow.

This package provides the runtime infrastructure for agent execution:

- ``tenxgraph.runtime.adapters``   - LLM response converters
- ``tenxgraph.runtime.publisher``  - event publishers (console, Redis, Kafka, RabbitMQ)
- ``tenxgraph.runtime.protocols``  - agent communication protocol packages
"""

from . import adapters, protocols, publisher
from .adapters.llm import (
    BaseConverter,
    ConverterType,
    GoogleGenAIConverter,
    OpenAIConverter,
    OpenAIResponsesConverter,
)
from .publisher import (
    BasePublisher,
    CompositePublisher,
    ConsolePublisher,
    ContentType,
    Event,
    EventModel,
    EventType,
    KafkaPublisher,
    RabbitMQPublisher,
    RedisPublisher,
    publish_event,
)


__all__ = [
    # Adapters
    "BaseConverter",
    "BasePublisher",
    "CompositePublisher",
    "ConsolePublisher",
    "ContentType",
    "ConverterType",
    "Event",
    "EventModel",
    "EventType",
    "GoogleGenAIConverter",
    "KafkaPublisher",
    "OpenAIConverter",
    "OpenAIResponsesConverter",
    "RabbitMQPublisher",
    "RedisPublisher",
    "adapters",
    # Protocols
    "protocols",
    "publish_event",
    "publisher",
]
