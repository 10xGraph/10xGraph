import time

from tenxgraph.core.state.stream_chunks import StreamChunk


def test_stream_chunk_timestamp_is_per_instance():
    first = StreamChunk()
    time.sleep(0.02)
    second = StreamChunk()
    assert second.timestamp > first.timestamp


def test_publisher_event_timestamp_is_per_instance():
    from tenxgraph.runtime.publisher.events import EventModel

    factory = EventModel.model_fields["timestamp"].default_factory
    first = factory()
    time.sleep(0.02)
    assert factory() > first
