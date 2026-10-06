from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, cast

from agentflow.core.state.message import (
    Message,
    TokenUsages,
    generate_id,
)
from agentflow.core.state.message_block import (
    AudioBlock,
    ImageBlock,
    MediaRef,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)

from .base_converter import BaseConverter
from .openai_responses_converter import OpenAIResponsesConverter, is_responses_api_response
from .reasoning_utils import parse_think_tags, parse_thought_tags


logger = logging.getLogger("agentflow.adapters.openai")


try:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    ChatCompletion = None  # type: ignore
    ChatCompletionChunk = None  # type: ignore


class OpenAIConverter(BaseConverter):
    """
    Converter for OpenAI responses to agentflow Message format.

    Handles both standard and streaming responses, extracting content, reasoning,
    tool calls, audio, images, and token usage details.

    Supports:
    - ChatCompletion responses
    - Streaming ChatCompletionChunk responses
    - Audio content (transcript + data)
    - Image content (URLs)
    - Reasoning content
    - Tool/function calls
    """

    async def convert_response(self, response: ChatCompletion) -> Message:  # type: ignore
        """
        Convert an OpenAI ChatCompletion to a Message.

        Args:
            response (ChatCompletion): The OpenAI ChatCompletion response object.

        Returns:
            Message: The converted message object.

        Raises:
            ImportError: If openai is not installed.
        """
        if not HAS_OPENAI:
            raise ImportError("openai is not installed. Please install it to use this converter.")

        # Auto-detect Responses API objects and delegate
        if is_responses_api_response(response):
            delegate = OpenAIResponsesConverter(state=self.state)
            return await delegate.convert_response(response)

        usages = self._extract_usage(response)

        choice = response.choices[0] if response.choices else None
        if not choice:
            logger.warning("OpenAI response has no choices: %s", response)
            return self._build_empty_response_message(response, usages)

        message = choice.message
        parsed = getattr(message, "parsed", None)
        content, reasoning_content = self._extract_message_text_and_reasoning(message)
        blocks = self._build_response_blocks(message, content, reasoning_content)
        final_tool_calls = self._append_tool_call_blocks(message, blocks)

        logger.debug("Creating message from OpenAI response with id: %s", response.id)

        return Message(
            message_id=generate_id(response.id),
            role=message.role,
            content=blocks,
            reasoning=reasoning_content,
            parsed_content=parsed,
            timestamp=getattr(response, "created", datetime.now().timestamp()),
            metadata={
                "provider": "openai",
                "model": response.model,
                "finish_reason": choice.finish_reason or "UNKNOWN",
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "service_tier": getattr(response, "service_tier", None),
            },
            usages=usages,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            tools_calls=final_tool_calls if final_tool_calls else None,
        )

    @staticmethod
    def _extract_usage(response: ChatCompletion) -> TokenUsages:  # type: ignore
        """Extract token usage from a ChatCompletion response."""
        usage = response.usage
        return TokenUsages(
            completion_tokens=usage.completion_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            reasoning_tokens=getattr(
                getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0
            )
            if usage
            else 0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
            )
            if usage
            else 0,
        )

    @staticmethod
    def _build_empty_response_message(
        response: ChatCompletion,  # type: ignore
        usages: TokenUsages,
    ) -> Message:
        """Build an empty assistant message when no choices were returned."""
        return Message(
            message_id=generate_id(response.id),
            role="assistant",
            content=[],
            timestamp=getattr(response, "created", datetime.now().timestamp()),
            metadata={
                "provider": "openai",
                "model": response.model,
                "finish_reason": "UNKNOWN",
            },
            usages=usages,
        )

    @staticmethod
    def _extract_message_text_and_reasoning(message: Any) -> tuple[str, str]:
        """Extract assistant text and reasoning, including inline reasoning tags."""
        content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", "") or ""
        if not reasoning_content:
            reasoning_content = getattr(message, "reasoning", "") or ""

        if not reasoning_content and content:
            content, think_reasoning = parse_think_tags(content)
            reasoning_content = think_reasoning or reasoning_content
        if not reasoning_content and content:
            content, thought_reasoning = parse_thought_tags(content)
            reasoning_content = thought_reasoning or reasoning_content

        return content, reasoning_content

    def _build_response_blocks(
        self,
        message: Any,
        content: str,
        reasoning_content: str,
    ) -> list:
        """Build non-tool content blocks for a response message."""
        blocks = []
        if reasoning_content:
            blocks.append(ReasoningBlock(summary=reasoning_content))
        if content:
            blocks.append(TextBlock(text=content))

        audio_block = self._extract_audio_block(getattr(message, "audio", None))
        if audio_block:
            blocks.append(audio_block)

        blocks.extend(self._extract_image_blocks(getattr(message, "images", None)))
        return blocks

    def _append_tool_call_blocks(self, message: Any, blocks: list) -> list[dict]:
        """Append tool call blocks and return raw tool call payloads."""
        final_tool_calls: list[dict] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            blocks.append(
                ToolCallBlock(
                    name=tool_call.function.name,
                    args=json.loads(tool_call.function.arguments),
                    id=tool_call.id,
                )
            )
            final_tool_calls.append(self._serialize_tool_call(tool_call))
        return final_tool_calls

    @staticmethod
    def _serialize_tool_call(tool_call: Any) -> dict:
        """Convert an OpenAI tool call object into a plain dictionary."""
        if hasattr(tool_call, "model_dump"):
            return tool_call.model_dump()
        return {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }

    def _extract_audio_block(self, audio_data: Any) -> AudioBlock | None:
        """Extract audio block from OpenAI audio data.

        Args:
            audio_data: Audio data from OpenAI response.

        Returns:
            AudioBlock | None: Audio block or None if invalid.
        """
        try:
            # OpenAI audio format: {id, data, transcript, expires_at}
            if hasattr(audio_data, "data"):
                data_base64 = audio_data.data
            elif isinstance(audio_data, dict):
                data_base64 = audio_data.get("data")
            else:
                return None

            if not data_base64:
                return None

            transcript = None
            if hasattr(audio_data, "transcript"):
                transcript = audio_data.transcript  # type: ignore
            elif isinstance(audio_data, dict):
                transcript = audio_data.get("transcript")

            media = MediaRef(
                kind="data",
                data_base64=data_base64,
                mime_type="audio/wav",  # OpenAI default
            )

            return AudioBlock(
                media=media,
                transcript=transcript,
            )
        except Exception as e:
            logger.warning("Failed to extract audio block: %s", e)
            return None

    def _extract_image_blocks(self, images_data: Any) -> list[ImageBlock]:
        """Extract image blocks from OpenAI images data.

        Args:
            images_data: List of image data from OpenAI response.

        Returns:
            list[ImageBlock]: List of image blocks.
        """
        blocks = []
        try:
            if not images_data:
                return blocks

            # Handle different formats
            if not isinstance(images_data, list):
                images_data = [images_data]

            for img in images_data:
                url = None
                if hasattr(img, "url"):
                    url = img.url
                elif isinstance(img, dict):
                    url = img.get("url")
                elif isinstance(img, str):
                    url = img

                if url:
                    media = MediaRef(
                        kind="url",
                        url=url,
                    )
                    blocks.append(ImageBlock(media=media))
        except Exception as e:
            logger.warning("Failed to extract image blocks: %s", e)

        return blocks

    def _extract_delta_content_blocks(
        self,
        delta: Any,
    ) -> tuple[str, str, list]:
        """Extract content blocks from a streaming delta.

        Args:
            delta: Delta object from streaming response.

        Returns:
            tuple: (text_part, reasoning_part, content_blocks)
        """
        text_part = delta.content or "" if hasattr(delta, "content") else ""
        content_blocks = []
        if text_part:
            content_blocks.append(TextBlock(text=text_part))

        reasoning_part = (
            getattr(delta, "reasoning_content", "") or ""
            if hasattr(delta, "reasoning_content")
            else ""
        )
        # OpenRouter streams reasoning via `delta.reasoning` (not `reasoning_content`)
        if not reasoning_part:
            reasoning_part = getattr(delta, "reasoning", "") or ""
        if reasoning_part:
            content_blocks.append(ReasoningBlock(summary=reasoning_part))

        # Extract audio if present in delta
        audio_data = getattr(delta, "audio", None) if hasattr(delta, "audio") else None
        if audio_data:
            audio_block = self._extract_audio_block(audio_data)
            if audio_block:
                content_blocks.append(audio_block)

        # Extract images if present in delta
        images_data = getattr(delta, "images", None) if hasattr(delta, "images") else None
        if images_data:
            image_blocks = self._extract_image_blocks(images_data)
            content_blocks.extend(image_blocks)

        return text_part, reasoning_part, content_blocks

    def _process_delta_tool_calls(
        self,
        delta: Any,
        tool_calls: list,
        tool_ids: set,
        content_blocks: list,
    ) -> None:
        """Process tool calls from delta.

        Args:
            delta: Delta object from streaming response.
            tool_calls: List to append tool calls to.
            tool_ids: Set to track tool call IDs.
            content_blocks: List to append tool call blocks to.
        """
        if not hasattr(delta, "tool_calls") or not delta.tool_calls:
            return

        for tc in delta.tool_calls:
            if not tc or not hasattr(tc, "id") or not tc.id:
                continue
            if tc.id in tool_ids:
                continue

            tool_ids.add(tc.id)

            # Extract function details
            func_name = tc.function.name if hasattr(tc, "function") and tc.function else ""
            func_args = (
                tc.function.arguments
                if hasattr(tc, "function") and tc.function and hasattr(tc.function, "arguments")
                else "{}"
            )

            if func_name and func_args:
                try:
                    args_dict = json.loads(func_args)
                    content_blocks.append(
                        ToolCallBlock(
                            name=func_name,
                            args=args_dict,
                            id=tc.id,
                        )
                    )

                    # Store raw tool call
                    if hasattr(tc, "model_dump"):
                        tool_calls.append(tc.model_dump())
                    else:
                        tool_calls.append(
                            {
                                "id": tc.id,
                                "type": getattr(tc, "type", "function"),
                                "function": {
                                    "name": func_name,
                                    "arguments": func_args,
                                },
                            }
                        )
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool call arguments: %s", func_args)

    def _process_chunk(
        self,
        chunk: ChatCompletionChunk | None,  # type: ignore
        accumulated_content: str,
        accumulated_reasoning_content: str,
        tool_calls: list,
        tool_ids: set,
    ) -> tuple[str, str, Message | None]:
        """
        Process a single chunk from an OpenAI streaming response.

        Args:
            chunk: The current chunk from the stream.
            accumulated_content: Accumulated text content so far.
            accumulated_reasoning_content: Accumulated reasoning content so far.
            tool_calls: List of tool calls detected so far.
            tool_ids: Set of tool call IDs to avoid duplicates.

        Returns:
            tuple: Updated accumulated content, reasoning, and Message (if any).
        """
        if not chunk:
            return accumulated_content, accumulated_reasoning_content, None

        if not chunk.choices or len(chunk.choices) == 0:
            return accumulated_content, accumulated_reasoning_content, None

        delta = chunk.choices[0].delta
        if delta is None:
            return accumulated_content, accumulated_reasoning_content, None

        # Extract content blocks
        text_part, reasoning_part, content_blocks = self._extract_delta_content_blocks(delta)
        accumulated_content += text_part
        accumulated_reasoning_content += reasoning_part

        # Handle tool calls if present
        self._process_delta_tool_calls(delta, tool_calls, tool_ids, content_blocks)

        output_message = Message(
            message_id=generate_id(chunk.id),
            role="assistant",
            content=content_blocks,
            reasoning=accumulated_reasoning_content,
            tools_calls=tool_calls,
            delta=True,
        )

        return accumulated_content, accumulated_reasoning_content, output_message

    async def _iterate_stream_chunks(self, stream: Any) -> AsyncGenerator[Any]:
        """Yield chunks from either async or sync OpenAI streams."""
        try:
            async for chunk in stream:  # type: ignore
                yield chunk
            return
        except Exception:  # noqa: S110 # nosec B110
            pass

        try:
            for chunk in stream:
                yield chunk
        except Exception:  # noqa: S110 # nosec B110
            pass

    @staticmethod
    def _extract_stream_reasoning(
        accumulated_content: str,
        accumulated_reasoning_content: str,
    ) -> tuple[str, str]:
        """Recover reasoning from inline tags when no dedicated reasoning delta exists."""
        if not accumulated_reasoning_content and accumulated_content:
            accumulated_content, think_reasoning = parse_think_tags(accumulated_content)
            accumulated_reasoning_content = think_reasoning or accumulated_reasoning_content
        if not accumulated_reasoning_content and accumulated_content:
            accumulated_content, thought_reasoning = parse_thought_tags(accumulated_content)
            accumulated_reasoning_content = thought_reasoning or accumulated_reasoning_content
        return accumulated_content, accumulated_reasoning_content

    @staticmethod
    def _build_stream_blocks(
        accumulated_content: str,
        accumulated_reasoning_content: str,
        tool_calls: list[dict],
    ) -> list:
        """Build the final block list for a streaming response."""
        blocks = []
        if accumulated_content:
            blocks.append(TextBlock(text=accumulated_content))
        if accumulated_reasoning_content:
            blocks.append(ReasoningBlock(summary=accumulated_reasoning_content))
        for tc in tool_calls:
            func_data = tc.get("function", {})
            blocks.append(
                ToolCallBlock(
                    name=func_data.get("name", ""),
                    args=json.loads(func_data.get("arguments", "{}")),
                    id=tc.get("id", ""),
                )
            )
        return blocks

    async def _handle_stream(
        self,
        config: dict,
        node_name: str,
        stream: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Handle an OpenAI streaming response and yield Message objects for each chunk.

        Args:
            config (dict): Node configuration parameters.
            node_name (str): Name of the node processing the response.
            stream: The OpenAI streaming response object.
            meta (dict | None): Optional metadata for conversion.

        Yields:
            Message: Converted message chunk from the stream.
        """
        accumulated_content = ""
        tool_calls = []
        tool_ids = set()
        accumulated_reasoning_content = ""

        is_awaitable = inspect.isawaitable(stream)

        # Await stream if necessary
        if is_awaitable:
            stream = await stream

        async for chunk in self._iterate_stream_chunks(stream):
            accumulated_content, accumulated_reasoning_content, message = self._process_chunk(
                chunk,
                accumulated_content,
                accumulated_reasoning_content,
                tool_calls,
                tool_ids,
            )
            if message:
                yield message

        # After streaming, yield final message
        metadata = meta or {}
        metadata["provider"] = "openai"
        metadata["node_name"] = node_name
        metadata["thread_id"] = config.get("thread_id")

        accumulated_content, accumulated_reasoning_content = self._extract_stream_reasoning(
            accumulated_content,
            accumulated_reasoning_content,
        )
        blocks = self._build_stream_blocks(
            accumulated_content,
            accumulated_reasoning_content,
            tool_calls,
        )

        logger.debug(
            "Stream done - Content: %s, Reasoning: %s, Tool Calls: %s",
            accumulated_content,
            accumulated_reasoning_content,
            len(tool_calls),
        )

        message = Message(
            role="assistant",
            message_id=generate_id(None),
            content=blocks,
            delta=False,
            reasoning=accumulated_reasoning_content,
            tools_calls=tool_calls,
            metadata=metadata,
        )
        yield message

    async def convert_streaming_response(
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Convert an OpenAI streaming or standard response to Message(s).

        Args:
            config (dict): Node configuration parameters.
            node_name (str): Name of the node processing the response.
            response (Any): The OpenAI response object (stream or standard).
            meta (dict | None): Optional metadata for conversion.

        Yields:
            Message: Converted message(s) from the response.

        Raises:
            ImportError: If openai is not installed.
            Exception: If response type is unsupported.
        """
        if not HAS_OPENAI:
            raise ImportError("openai is not installed. Please install it to use this converter.")

        # Auto-detect Responses API objects and delegate
        if is_responses_api_response(response):
            delegate = OpenAIResponsesConverter(state=self.state)
            async for msg in delegate.convert_streaming_response(config, node_name, response, meta):
                yield msg
            return

        # Check if it's a streaming response (async generator or iterator)
        if hasattr(response, "__aiter__") or hasattr(response, "__iter__"):
            async for event in self._handle_stream(
                config or {},
                node_name or "",
                response,
                meta,
            ):
                yield event
        elif isinstance(response, ChatCompletion):  # type: ignore
            message = await self.convert_response(cast(ChatCompletion, response))  # type: ignore
            yield message
        else:
            raise Exception("Unsupported response type for OpenAIConverter")
