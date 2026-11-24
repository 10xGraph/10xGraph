"""Comprehensive streaming integration tests for TAF framework."""

import asyncio

import pytest

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.core.graph import StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.core.state.stream_chunks import StreamChunk
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter
from tenxgraph.utils import END


class MockStreamingResponse:
    """Mock streaming response that simulates incremental LLM chunks."""

    def __init__(self, content: str = "Streaming response"):
        self.content = content
        self.chunks = self._create_chunks()

    def _create_chunks(self):
        """Create mock chunks similar to OpenAI-style chat completions."""
        words = self.content.split()
        chunks = []

        for i, word in enumerate(words):
            chunk = type(
                "MockChunk",
                (),
                {
                    "id": "chatcmpl-stream",
                    "model": "mock-model",
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "delta": type(
                                    "Delta",
                                    (),
                                    {"content": word + (" " if i < len(words) - 1 else "")},
                                ),
                                "finish_reason": "stop" if i == len(words) - 1 else None,
                            },
                        )
                    ]
                },
            )
            chunks.append(chunk)

        return chunks

    def __iter__(self):
        return iter(self.chunks)


class MockAsyncStreamingResponse:
    """Mock async streaming response that simulates incremental LLM chunks."""

    def __init__(self, content: str = "Async streaming response"):
        self.content = content
        self.chunks = self._create_chunks()

    def _create_chunks(self):
        """Create mock chunks similar to OpenAI-style chat completions."""
        words = self.content.split()
        chunks = []

        for i, word in enumerate(words):
            chunk = type(
                "MockChunk",
                (),
                {
                    "id": "chatcmpl-stream",
                    "model": "mock-model",
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "delta": type(
                                    "Delta",
                                    (),
                                    {"content": word + (" " if i < len(words) - 1 else "")},
                                ),
                                "finish_reason": "stop" if i == len(words) - 1 else None,
                            },
                        )
                    ]
                },
            )
            chunks.append(chunk)

        return chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(0.01)  # Small delay to simulate async
            yield chunk


# Test node functions for different streaming scenarios


def non_streaming_string_node(state: AgentState) -> str:
    """Node that returns a simple string (non-streaming)."""
    return "This is a non-streaming string response"


STREAMED_TEXT = "This is a streaming response from the node"


def streaming_node(state: AgentState) -> ModelResponseConverter:
    """Node that returns a streamed LLM response, normalized by the OpenAI converter."""
    return ModelResponseConverter(MockStreamingResponse(STREAMED_TEXT), "openai")


ASYNC_STREAMED_TEXT = "This is an async streaming response from the node"


async def async_streaming_node(state: AgentState) -> ModelResponseConverter:
    """Node that returns an async streamed LLM response."""
    return ModelResponseConverter(MockAsyncStreamingResponse(ASYNC_STREAMED_TEXT), "openai")


class TestStreamingIntegration:
    """Test streaming integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checkpointer = InMemoryCheckpointer()

    def test_non_streaming_string_node_sync(self):
        """Test graph.stream() with a node that returns a string (non-streaming)."""
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("string_node", non_streaming_string_node)
        graph.set_entry_point("string_node")
        graph.add_edge("string_node", END)

        compiled = graph.compile(checkpointer=self.checkpointer)

        input_data = {"messages": [Message.text_message("Test input", role="user")]}
        config = {"thread_id": "test_string_sync"}

        chunks = list(compiled.stream(input_data, config))

        # Verify we got chunks
        assert len(chunks) > 0  # noqa: S101

    @pytest.mark.asyncio
    async def test_non_streaming_string_node_async(self):
        """Test graph.astream() with a node that returns a string (non-streaming)."""
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("string_node", non_streaming_string_node)
        graph.set_entry_point("string_node")
        graph.add_edge("string_node", END)

        compiled = graph.compile(checkpointer=self.checkpointer)

        input_data = {"messages": [Message.text_message("Test input", role="user")]}
        config = {"thread_id": "test_string_async"}

        chunks = []
        async for chunk in compiled.astream(input_data, config):
            chunks.append(chunk)

        # Verify we got chunks
        assert len(chunks) > 0  # noqa: S101

    def test_streaming_node_sync(self):
        """stream() yields the node's streamed chunks, then the final assistant message."""
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("streaming_node", streaming_node)
        graph.set_entry_point("streaming_node")
        graph.add_edge("streaming_node", END)

        compiled = graph.compile(checkpointer=self.checkpointer)

        input_data = {"messages": [Message.text_message("Test input", role="user")]}
        config = {"thread_id": "test_streaming_sync"}

        chunks = list(compiled.stream(input_data, config))
        assert all(isinstance(chunk, StreamChunk) for chunk in chunks)  # noqa: S101

        deltas = [c.message for c in chunks if c.message is not None and c.message.delta]
        assert len(deltas) > 1  # noqa: S101
        assert "".join(m.text() for m in deltas) == STREAMED_TEXT  # noqa: S101

        finals = [c.message for c in chunks if c.message is not None and not c.message.delta]
        assistant_finals = [m for m in finals if m.role == "assistant"]
        assert len(assistant_finals) == 1  # noqa: S101
        assert assistant_finals[0].text() == STREAMED_TEXT  # noqa: S101

    @pytest.mark.asyncio
    async def test_async_streaming_node_async(self):
        """astream() yields the async node's streamed chunks, then the final assistant message."""
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("async_streaming_node", async_streaming_node)
        graph.set_entry_point("async_streaming_node")
        graph.add_edge("async_streaming_node", END)

        compiled = graph.compile(checkpointer=self.checkpointer)

        input_data = {"messages": [Message.text_message("Test input", role="user")]}
        config = {"thread_id": "test_async_streaming"}

        chunks = []
        async for chunk in compiled.astream(input_data, config):
            chunks.append(chunk)
        assert all(isinstance(chunk, StreamChunk) for chunk in chunks)  # noqa: S101

        deltas = [c.message for c in chunks if c.message is not None and c.message.delta]
        assert len(deltas) > 1  # noqa: S101
        assert "".join(m.text() for m in deltas) == ASYNC_STREAMED_TEXT  # noqa: S101

        finals = [c.message for c in chunks if c.message is not None and not c.message.delta]
        assistant_finals = [m for m in finals if m.role == "assistant"]
        assert len(assistant_finals) == 1  # noqa: S101
        assert assistant_finals[0].text() == ASYNC_STREAMED_TEXT  # noqa: S101

    def test_stream_chunk_properties(self):
        """Test EventModel properties and methods."""
        graph = StateGraph[AgentState](AgentState())
        graph.add_node("string_node", non_streaming_string_node)
        graph.set_entry_point("string_node")
        graph.add_edge("string_node", END)

        compiled = graph.compile(checkpointer=self.checkpointer)

        input_data = {"messages": [Message.text_message("Test chunk properties", role="user")]}
        config = {"thread_id": "test_chunk_props"}

        chunks = list(compiled.stream(input_data, config))

        assert len(chunks) > 0  # noqa: S101
