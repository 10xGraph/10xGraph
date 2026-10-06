"""
Converter for Google Generative AI SDK responses to agentflow Message format.

This module provides conversion utilities for Google's google-genai SDK,
supporting both standard and streaming responses.
"""

from __future__ import annotations

import base64
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
    VideoBlock,
)

from .base_converter import BaseConverter


logger = logging.getLogger("agentflow.adapters.google_genai")


try:
    from google.genai.types import GenerateContentResponse

    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    GenerateContentResponse = None


class GoogleGenAIConverter(BaseConverter):
    """
    Converter for Google Generative AI responses to agentflow Message format.

    Handles both standard and streaming responses, extracting content, reasoning,
    tool calls, and token usage details from Google's GenerateContentResponse.
    """

    async def convert_response(self, response: GenerateContentResponse) -> Message:  # type: ignore[reportInvalidTypeForm]
        """
        Convert a Google GenAI GenerateContentResponse to a Message.

        Args:
            response (GenerateContentResponse): The Google GenAI model response object.

        Returns:
            Message: The converted message object.

        Raises:
            ImportError: If google-genai is not installed.
        """
        # Note: Import check removed to allow for mocking in tests
        # The typing will ensure correct usage in production

        # Extract candidates (Google GenAI can return multiple candidates)
        candidates = response.candidates or []
        if not candidates:
            logger.warning("Google GenAI response has no candidates: %s", response)
            # Return empty message if no candidates
            return self._create_empty_message()

        # Take the first candidate
        candidate = candidates[0]
        content = candidate.content

        # Extract usage metadata
        usages = self._extract_usage_metadata(response)

        # Extract parts from content
        parts = (content.parts or []) if content else []
        if parts is None:
            logger.warning("Google GenAI response content has no parts: %s", response)

        blocks, tools_calls, reasoning_content = self._process_parts(parts)

        # Get model version and other metadata
        model_version = response.model_version or ""
        finish_reason = candidate.finish_reason if candidate else "UNKNOWN"

        created_date = datetime.now().timestamp()
        if hasattr(response, "create_time") and response.create_time:
            created_date = response.create_time.timestamp()

        parsed = getattr(response, "parsed", None)

        return Message(
            message_id=generate_id(getattr(response, "response_id", None)),
            role="assistant",
            content=blocks,
            reasoning=reasoning_content,
            timestamp=created_date,
            metadata={
                "provider": "google_genai",
                "model": model_version,
                "finish_reason": str(finish_reason),
            },
            usages=usages,
            raw=None,  # Google GenAI responses are Pydantic models
            tools_calls=tools_calls if tools_calls else None,
            parsed_content=parsed,
        )

    def _create_empty_message(self) -> Message:
        """Create an empty assistant message."""
        return Message(
            message_id=generate_id(None),
            role="assistant",
            content=[],
            timestamp=datetime.now().timestamp(),
            metadata={"provider": "google_genai", "model": ""},
        )

    def _extract_usage_metadata(self, response: Any) -> TokenUsages:
        """Extract token usage metadata from response."""
        usage_metadata = response.usage_metadata or {}
        return TokenUsages(
            completion_tokens=getattr(usage_metadata, "candidates_token_count", 0) or 0,
            prompt_tokens=getattr(usage_metadata, "prompt_token_count", 0) or 0,
            total_tokens=getattr(usage_metadata, "total_token_count", 0) or 0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=getattr(usage_metadata, "cached_content_token_count", 0) or 0,
            reasoning_tokens=getattr(usage_metadata, "thoughts_token_count", 0) or 0,
        )

    def _process_parts(self, parts: list) -> tuple[list, list, str]:
        """Process content parts and extract blocks, tool calls, and reasoning."""
        blocks = []
        tools_calls = []
        reasoning_content = ""

        for part in parts:
            reasoning = self._process_reasoning_part(part, blocks)
            if reasoning:
                reasoning_content = reasoning
                continue

            self._process_text_part(part, blocks)
            self._process_function_call_part(part, blocks, tools_calls)
            self._process_inline_media_part(part, blocks)
            self._process_file_media_part(part, blocks)

        return blocks, tools_calls, reasoning_content

    def _process_text_part(self, part: Any, blocks: list) -> None:
        """Process text part."""
        if part.text:
            blocks.append(TextBlock(text=part.text))

    def _process_reasoning_part(self, part: Any, blocks: list) -> str:
        """Process reasoning (thought) part."""
        if hasattr(part, "thought") and part.thought:
            # When part.thought is True, the actual content is in part.text
            thought_content = part.text if hasattr(part, "text") else ""
            if thought_content:
                blocks.append(ReasoningBlock(summary=thought_content))
            return thought_content
        return ""

    def _process_function_call_part(self, part: Any, blocks: list, tools_calls: list) -> None:
        """Process function call part."""
        if hasattr(part, "function_call") and part.function_call:
            func_call = part.function_call
            tool_call_id = generate_id(None)

            # Parse args - they should be a dict
            args = dict(func_call.args) if func_call.args else {}

            blocks.append(
                ToolCallBlock(
                    name=func_call.name,
                    args=args,
                    id=tool_call_id,
                )
            )

            tool_call_dict: dict[str, Any] = {
                "id": tool_call_id,
                "function": {"name": func_call.name, "arguments": json.dumps(args)},
                "type": "function",
            }
            # Preserve thought_signature bytes so they can be replayed verbatim
            # in subsequent requests. Gemini 3 strictly requires signatures on
            # function call parts; Gemini 2.5 also benefits from them.
            sig = getattr(part, "thought_signature", None)
            if sig:
                tool_call_dict["thought_signature"] = base64.b64encode(sig).decode()

            tools_calls.append(tool_call_dict)

    def _process_inline_media_part(self, part: Any, blocks: list) -> None:
        """Process inline media (images, audio, video) part."""
        if hasattr(part, "inline_data") and part.inline_data:
            inline_data = part.inline_data
            mime_type = inline_data.mime_type or ""

            self._add_media_block_by_type(blocks, mime_type, inline_data.data, kind="data")

    def _process_file_media_part(self, part: Any, blocks: list) -> None:
        """Process file media (images, audio, video) part."""
        if hasattr(part, "file_data") and part.file_data:
            file_data = part.file_data
            mime_type = file_data.mime_type or ""

            self._add_media_block_by_type(blocks, mime_type, file_data.file_uri, kind="url")

    def _add_media_block_by_type(
        self, blocks: list, mime_type: str, data: str | Any, kind: str
    ) -> None:
        """Add appropriate media block based on MIME type."""
        media = MediaRef(
            kind=kind,
            data_base64=data if kind == "data" else None,
            url=data if kind == "url" else None,
            mime_type=mime_type,
        )

        if mime_type.startswith("image/"):
            blocks.append(ImageBlock(media=media))
        elif mime_type.startswith("audio/"):
            blocks.append(AudioBlock(media=media))
        elif mime_type.startswith("video/"):
            blocks.append(VideoBlock(media=media))

    def _extract_delta_content_blocks(
        self,
        candidate: Any,
    ) -> tuple[str, str, list, list]:
        """Extract content blocks from a streaming candidate.

        Args:
            candidate: Candidate object from streaming response.

        Returns:
            tuple: (text_part, reasoning_part, content_blocks, tools_calls)
        """
        text_part = ""
        reasoning_part = ""
        content_blocks = []
        tools_calls = []

        if not candidate or not candidate.content:
            return text_part, reasoning_part, content_blocks, tools_calls

        content = candidate.content
        parts = (content.parts or []) if content else []

        for part in parts:
            # Handle thought parts (reasoning) first
            if hasattr(part, "thought") and part.thought:
                thought_content = part.text if hasattr(part, "text") else ""
                if thought_content:
                    reasoning_part += thought_content
                    content_blocks.append(ReasoningBlock(summary=thought_content))
                # Skip text processing for thought parts
                continue

            # Handle text parts
            if hasattr(part, "text") and part.text:
                text_part += part.text
                content_blocks.append(TextBlock(text=part.text))

            # Handle function calls
            if hasattr(part, "function_call") and part.function_call:
                func_call = part.function_call
                tool_call_id = generate_id(None)

                # Parse args - they should be a dict
                args = dict(func_call.args) if func_call.args else {}

                content_blocks.append(
                    ToolCallBlock(
                        name=func_call.name,
                        args=args,
                        id=tool_call_id,
                    )
                )

                tool_call_dict: dict[str, Any] = {
                    "id": tool_call_id,
                    "function": {"name": func_call.name, "arguments": json.dumps(args)},
                    "type": "function",
                }
                # Preserve thought_signature bytes so they can be replayed verbatim
                # in subsequent requests (streaming path).
                sig = getattr(part, "thought_signature", None)
                if sig:
                    tool_call_dict["thought_signature"] = base64.b64encode(sig).decode()

                tools_calls.append(tool_call_dict)

            # Handle inline media (images, audio, video) in streaming chunks
            if hasattr(part, "inline_data") and part.inline_data:
                self._process_inline_media_part(part, content_blocks)

            # Handle file media in streaming chunks
            if hasattr(part, "file_data") and part.file_data:
                self._process_file_media_part(part, content_blocks)

        return text_part, reasoning_part, content_blocks, tools_calls

    def _process_chunk(
        self,
        chunk: Any,
        seq: int,
        accumulated_content: str,
        accumulated_reasoning_content: str,
        tool_calls: list,
        tool_ids: set,
    ) -> tuple[str, str, list, int, Message | None]:
        """
        Process a single chunk from a Google GenAI streaming response.

        Args:
            chunk: The current chunk from the stream.
            seq (int): Sequence number of the chunk.
            accumulated_content (str): Accumulated text content so far.
            accumulated_reasoning_content (str): Accumulated reasoning content so far.
            tool_calls (list): List of tool calls detected so far.
            tool_ids (set): Set of tool call IDs to avoid duplicates.

        Returns:
            tuple: Updated accumulated content, reasoning, tool calls, sequence,
                and Message (if any).
        """
        if not chunk:
            return accumulated_content, accumulated_reasoning_content, tool_calls, seq, None

        candidates = chunk.candidates or []
        if not candidates:
            return accumulated_content, accumulated_reasoning_content, tool_calls, seq, None

        candidate = candidates[0]

        # Extract content blocks
        text_part, reasoning_part, content_blocks, new_tools = self._extract_delta_content_blocks(
            candidate
        )
        accumulated_content += text_part
        accumulated_reasoning_content += reasoning_part

        # Handle tool calls
        for tool_call in new_tools:
            tool_id = tool_call.get("id")
            if tool_id and tool_id not in tool_ids:
                tool_ids.add(tool_id)
                tool_calls.append(tool_call)

        output_message = Message(
            message_id=generate_id(None),
            role="assistant",
            content=content_blocks,
            reasoning=accumulated_reasoning_content,
            tools_calls=tool_calls if tool_calls else None,
            delta=True,
        )

        return (
            accumulated_content,
            accumulated_reasoning_content,
            tool_calls,
            seq,
            output_message,
        )

    async def _handle_stream(
        self,
        config: dict,
        node_name: str,
        stream: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Handle a Google GenAI streaming response and yield Message objects for each chunk.

        Args:
            config (dict): Node configuration parameters.
            node_name (str): Name of the node processing the response.
            stream: The Google GenAI streaming response object.
            meta (dict | None): Optional metadata for conversion.

        Yields:
            Message: Converted message chunk from the stream.
        """
        accumulated_content = ""
        tool_calls = []
        tool_ids = set()
        accumulated_reasoning_content = ""
        seq = 0

        is_awaitable = inspect.isawaitable(stream)

        # Await stream if necessary
        if is_awaitable:
            stream = await stream

        # Try async iteration
        if stream is not None:
            try:
                async for chunk in stream:  # type: ignore
                    accumulated_content, accumulated_reasoning_content, tool_calls, seq, message = (
                        self._process_chunk(
                            chunk,
                            seq,
                            accumulated_content,
                            accumulated_reasoning_content,
                            tool_calls,
                            tool_ids,
                        )
                    )

                    if message:
                        yield message
            except Exception as e:
                logger.error("Google GenAI stream error: %s", e)
                raise

        # After streaming, yield final message
        metadata = meta or {}
        metadata["provider"] = "google_genai"
        metadata["node_name"] = node_name
        metadata["thread_id"] = config.get("thread_id")

        blocks = []
        if accumulated_content:
            blocks.append(TextBlock(text=accumulated_content))
        if accumulated_reasoning_content:
            blocks.append(ReasoningBlock(summary=accumulated_reasoning_content))
        if tool_calls:
            for tc in tool_calls:
                func_data = tc.get("function", {})
                args_str = func_data.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}
                blocks.append(
                    ToolCallBlock(
                        name=func_data.get("name", ""),
                        args=args,
                        id=tc.get("id", ""),
                    )
                )

        logger.debug(
            "Stream complete - Content: %s  Reasoning: %s Tool Calls: %s",
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
            tools_calls=tool_calls if tool_calls else None,
            metadata=metadata,
        )
        yield message

    async def convert_streaming_response(  # type: ignore
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Convert a Google GenAI streaming or standard response to Message(s).

        Args:
            config (dict): Node configuration parameters.
            node_name (str): Name of the node processing the response.
            response (Any): The Google GenAI response object (stream or standard).
            meta (dict | None): Optional metadata for conversion.

        Yields:
            Message: Converted message(s) from the response.

        Raises:
            ImportError: If google-genai is not installed.
            Exception: If response type is unsupported.
        """
        # Note: Import check removed to allow for mocking in tests
        # Check if it's a standard response
        if HAS_GOOGLE_GENAI and isinstance(response, GenerateContentResponse):  # type: ignore
            message = await self.convert_response(cast(GenerateContentResponse, response))  # type: ignore
            yield message
        else:
            # Assume it's a stream
            async for event in self._handle_stream(
                config or {},
                node_name or "",
                response,
                meta,
            ):
                yield event
