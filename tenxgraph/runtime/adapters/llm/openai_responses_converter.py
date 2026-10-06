"""Converter for OpenAI Responses API output.

The Responses API (``client.responses.create()``) uses a fundamentally
different response schema compared with Chat Completions.  This module
maps that schema into the same normalised ``Message`` objects used by
every other AgentFlow converter.

Key differences from Chat Completions:
- Response items live in ``response.output`` (list), not ``response.choices``
- Each item has a ``type``: ``"message"``, ``"reasoning"``, ``"function_call"``
- Token usage names differ: ``input_tokens`` / ``output_tokens``
- Streaming uses semantic events like ``response.output_text.delta``
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

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
from .reasoning_utils import (
    parse_think_tags,  # noqa: F401
)


logger = logging.getLogger("agentflow.adapters.openai_responses")


@dataclass
class _ResponseStreamState:
    """Mutable state accumulated while consuming a Responses API stream."""

    accumulated_text: str = ""
    accumulated_reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    current_fc_name: str = ""
    current_fc_args: str = ""
    current_fc_call_id: str = ""
    usages: TokenUsages = field(
        default_factory=lambda: TokenUsages(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
    )


# ---------------------------------------------------------------------------
# Helper: detect whether an object is a Responses API response
# ---------------------------------------------------------------------------


def is_responses_api_response(response: Any) -> bool:
    """Return *True* if *response* looks like an OpenAI Responses API object.

    We check for the ``output`` attribute (list of output items) **and** the
    absence of the ``choices`` attribute (which is specific to Chat
    Completions).
    """
    return hasattr(response, "output") and not hasattr(response, "choices")


# ---------------------------------------------------------------------------
# Helper: parse <reasoning>…</reasoning> tags from text
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


class OpenAIResponsesConverter(BaseConverter):
    """Convert OpenAI **Responses API** objects to AgentFlow ``Message``.

    Handles both non-streaming responses (``Response`` objects with an
    ``output`` list) and streaming event iterators.
    """

    # ---- non-streaming -------------------------------------------------

    async def convert_response(self, response: Any) -> Message:
        """Convert a non-streaming Responses API result to ``Message``."""

        # --- token usage --------------------------------------------------
        usages = self._extract_token_usage(response)

        # --- iterate output items -----------------------------------------
        blocks: list = []
        reasoning_text = ""
        tool_calls_raw: list[dict] = []
        output_items = getattr(response, "output", [])

        if not output_items:
            logger.warning("OpenAI Responses API response has no output items: %s", response)

        for item in output_items:
            item_type = getattr(item, "type", None)

            if item_type == "reasoning":
                r = self._extract_reasoning_from_item(item)
                # Create a ReasoningBlock even when summary is empty —
                # the API still consumed reasoning tokens and the item
                # signals that reasoning occurred.
                summary = r if r else "(reasoning performed - enable summary to see details)"
                blocks.append(ReasoningBlock(summary=summary))
                reasoning_text += ("\n" + summary) if reasoning_text else summary

            elif item_type == "message":
                t = self._extract_text_from_message_item(item)
                if t:
                    blocks.append(TextBlock(text=t))
                # Also extract any media blocks from message content
                media_blocks = self._extract_media_from_message_item(item)
                blocks.extend(media_blocks)

            elif item_type == "function_call":
                tb, raw = self._extract_tool_call_from_item(item)
                if tb:
                    blocks.append(tb)
                    tool_calls_raw.append(raw)

            elif item_type in ("image_generation_call", "image_generation"):
                img_block = self._extract_image_generation(item)
                if img_block:
                    blocks.append(img_block)

        # --- build Message ------------------------------------------------
        resp_id = getattr(response, "id", None)
        model = getattr(response, "model", "unknown")
        status = getattr(response, "status", "completed")
        created = getattr(response, "created_at", None) or datetime.now().timestamp()

        return Message(
            message_id=generate_id(resp_id),
            role="assistant",
            content=blocks,
            reasoning=reasoning_text,
            timestamp=created,
            metadata={
                "provider": "openai",
                "model": model,
                "finish_reason": status,
            },
            usages=usages,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            tools_calls=tool_calls_raw if tool_calls_raw else None,
        )

    # ---- streaming ------------------------------------------------------

    async def convert_streaming_response(
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """Convert a Responses API stream to ``Message`` chunks.

        Yields intermediate ``Message(delta=True)`` per event and a final
        ``Message(delta=False)`` when the stream completes.
        """
        # If it's not iterable at all, try the non-streaming path
        if not hasattr(response, "__aiter__") and not hasattr(response, "__iter__"):
            if is_responses_api_response(response):
                yield await self.convert_response(response)
                return
            raise TypeError("Unsupported response type for OpenAIResponsesConverter")

        async for msg in self._handle_stream(config, node_name, response, meta):
            yield msg

    async def _handle_stream(
        self,
        config: dict,
        node_name: str,
        stream: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """Internal stream handler."""
        state = _ResponseStreamState()

        is_awaitable = inspect.isawaitable(stream)
        if is_awaitable:
            stream = await stream

        async for event in self._iterate_stream_events(stream):
            messages, completed = self._process_stream_event(state, event)
            for message in messages:
                yield message
            if completed:
                break

        # --- final message -------------------------------------------------
        metadata = meta or {}
        metadata["provider"] = "openai"
        metadata["node_name"] = node_name
        metadata["thread_id"] = config.get("thread_id")
        final_blocks = self._build_final_blocks(state)

        yield Message(
            message_id=generate_id(None),
            role="assistant",
            content=final_blocks,
            reasoning=state.accumulated_reasoning,
            delta=False,
            tools_calls=state.tool_calls if state.tool_calls else None,
            metadata=metadata,
            usages=state.usages,
        )

    async def _iterate_stream_events(self, stream: Any) -> AsyncGenerator[Any]:
        """Yield stream events from async or sync Responses API iterators."""
        try:
            async for event in stream:
                yield event
            return
        except TypeError:
            pass

        for event in stream:
            yield event

    def _process_stream_event(
        self,
        state: _ResponseStreamState,
        event: Any,
    ) -> tuple[list[Message], bool]:
        """Apply one streaming event and return delta messages plus completion state."""
        event_type = getattr(event, "type", "")
        simple_handlers = {
            "response.output_text.delta": self._handle_text_delta,
            "response.reasoning_summary_text.delta": self._handle_reasoning_delta,
            "response.function_call_arguments.delta": self._handle_function_call_argument_delta,
            "response.output_item.added": self._handle_output_item_added,
        }

        handler = simple_handlers.get(event_type)
        if handler is not None:
            return handler(state, event), False
        if event_type == "response.output_item.done":
            return self._handle_output_item_done(state, getattr(event, "item", None)), False
        if event_type == "response.completed":
            state.usages = self._extract_token_usage(getattr(event, "response", None))
            return [], True
        return [], False

    def _handle_text_delta(self, state: _ResponseStreamState, event: Any) -> list[Message]:
        """Handle a text delta event."""
        delta_text = getattr(event, "delta", "")
        state.accumulated_text += delta_text
        return [self._build_delta_message([TextBlock(text=delta_text)] if delta_text else [])]

    def _handle_reasoning_delta(self, state: _ResponseStreamState, event: Any) -> list[Message]:
        """Accumulate streaming reasoning text."""
        state.accumulated_reasoning += getattr(event, "delta", "")
        return []

    def _handle_function_call_argument_delta(
        self,
        state: _ResponseStreamState,
        event: Any,
    ) -> list[Message]:
        """Accumulate partial function-call arguments."""
        state.current_fc_args += getattr(event, "delta", "")
        return []

    def _handle_output_item_added(self, state: _ResponseStreamState, event: Any) -> list[Message]:
        """Track metadata for newly opened output items."""
        item = getattr(event, "item", None)
        if item and getattr(item, "type", "") == "function_call":
            state.current_fc_name = getattr(item, "name", "")
            state.current_fc_call_id = getattr(item, "call_id", "")
            state.current_fc_args = ""
        return []

    def _handle_output_item_done(
        self,
        state: _ResponseStreamState,
        item: Any,
    ) -> list[Message]:
        """Process a completed output item."""
        item_type = getattr(item, "type", "") if item else ""
        handlers = {
            "function_call": self._finalize_function_call,
            "reasoning": self._handle_reasoning_item_done,
            "message": self._handle_message_item_done,
            "image_generation_call": self._handle_image_item_done,
            "image_generation": self._handle_image_item_done,
        }
        handler = handlers.get(item_type)
        if handler is None:
            return []
        return handler(state, item)

    def _finalize_function_call(self, state: _ResponseStreamState, item: Any) -> list[Message]:
        """Finalize a completed function call item."""
        name = getattr(item, "name", state.current_fc_name)
        raw_args = getattr(item, "arguments", state.current_fc_args)
        call_id = getattr(item, "call_id", state.current_fc_call_id)
        state.tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": raw_args or "{}"},
            }
        )
        state.current_fc_name = ""
        state.current_fc_args = ""
        state.current_fc_call_id = ""
        return [
            self._build_delta_message(
                [ToolCallBlock(name=name, args=self._parse_json_args(raw_args), id=call_id)]
            )
        ]

    def _handle_reasoning_item_done(
        self,
        state: _ResponseStreamState,
        item: Any,
    ) -> list[Message]:
        """Append completed reasoning summaries to the accumulated reasoning text."""
        reasoning = self._extract_reasoning_from_item(item)
        if reasoning:
            prefix = "\n" if state.accumulated_reasoning else ""
            state.accumulated_reasoning += f"{prefix}{reasoning}"
        return []

    def _handle_message_item_done(
        self,
        _state: _ResponseStreamState,
        item: Any,
    ) -> list[Message]:
        """Emit media blocks from completed message items."""
        media_blocks = self._extract_media_from_message_item(item)
        if not media_blocks:
            return []
        return [self._build_delta_message(media_blocks)]

    def _handle_image_item_done(
        self,
        _state: _ResponseStreamState,
        item: Any,
    ) -> list[Message]:
        """Emit an image block from image generation events."""
        img_block = self._extract_image_generation(item)
        if not img_block:
            return []
        return [self._build_delta_message([img_block])]

    @staticmethod
    def _build_delta_message(content: list) -> Message:
        """Build a streaming delta message."""
        return Message(
            message_id=generate_id(None),
            role="assistant",
            content=content,
            delta=True,
        )

    @staticmethod
    def _parse_json_args(raw_args: str) -> dict:
        """Parse tool call arguments, defaulting to an empty dict on invalid JSON."""
        try:
            return json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            return {}

    def _build_final_blocks(self, state: _ResponseStreamState) -> list:
        """Build the final content blocks after a Responses API stream ends."""
        final_blocks: list = []
        if state.accumulated_text:
            final_blocks.append(TextBlock(text=state.accumulated_text))
        if state.accumulated_reasoning:
            final_blocks.append(ReasoningBlock(summary=state.accumulated_reasoning))
        for tool_call in state.tool_calls:
            function_data = tool_call.get("function", {})
            final_blocks.append(
                ToolCallBlock(
                    name=function_data.get("name", ""),
                    args=self._parse_json_args(function_data.get("arguments", "{}")),
                    id=tool_call.get("id", ""),
                )
            )
        return final_blocks

    # ---- private helpers ------------------------------------------------

    @staticmethod
    def _extract_reasoning_from_item(item: Any) -> str:
        """Pull reasoning summary text out of a ``type='reasoning'`` item."""
        summary_list = getattr(item, "summary", None) or []
        parts: list[str] = []
        for entry in summary_list:
            # Each entry may be an object with .text or a dict with "text"
            if hasattr(entry, "text"):
                parts.append(entry.text)
            elif isinstance(entry, dict):
                parts.append(entry.get("text", ""))
        return "\n".join(parts)

    @staticmethod
    def _extract_text_from_message_item(item: Any) -> str:
        """Pull text out of a ``type='message'`` item."""
        content_list = getattr(item, "content", []) or []
        parts: list[str] = []
        for entry in content_list:
            entry_type = getattr(entry, "type", None)
            if entry_type == "output_text":
                parts.append(getattr(entry, "text", ""))
            elif isinstance(entry, dict) and entry.get("type") == "output_text":
                parts.append(entry.get("text", ""))
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _extract_media_from_message_item(item: Any) -> list:
        """Extract media blocks (images, audio) from a ``type='message'`` item.

        The Responses API can return ``output_image`` or ``output_audio``
        entries in the message content alongside ``output_text``.
        """
        content_list = getattr(item, "content", []) or []
        blocks: list = []
        for entry in content_list:
            entry_type = getattr(entry, "type", None) or (
                entry.get("type") if isinstance(entry, dict) else None
            )

            if entry_type == "output_image":
                # output_image has image_url (URL string) or image_data (base64)
                url = getattr(entry, "image_url", None) or (
                    entry.get("image_url") if isinstance(entry, dict) else None
                )
                data = getattr(entry, "image_data", None) or (
                    entry.get("image_data") if isinstance(entry, dict) else None
                )
                if data:
                    blocks.append(
                        ImageBlock(
                            media=MediaRef(
                                kind="data",
                                data_base64=data,
                                mime_type="image/png",
                            )
                        )
                    )
                elif url:
                    blocks.append(
                        ImageBlock(
                            media=MediaRef(
                                kind="url",
                                url=url,
                                mime_type="image/png",
                            )
                        )
                    )

            elif entry_type == "output_audio":
                data = getattr(entry, "data", None) or (
                    entry.get("data") if isinstance(entry, dict) else None
                )
                transcript = getattr(entry, "transcript", None) or (
                    entry.get("transcript") if isinstance(entry, dict) else None
                )
                if data:
                    blocks.append(
                        AudioBlock(
                            media=MediaRef(kind="data", data_base64=data, mime_type="audio/wav"),
                            transcript=transcript,
                        )
                    )

        return blocks

    @staticmethod
    def _extract_image_generation(item: Any) -> ImageBlock | None:
        """Extract image block from ``type='image_generation_call'`` items.

        The ``result`` field contains the base64-encoded image data.
        """
        result = getattr(item, "result", None)
        if result is None and isinstance(item, dict):
            result = item.get("result")

        if not result:
            return None

        # result can be base64 string or an object with .data / .b64_json
        if isinstance(result, str):
            b64 = result
        else:
            b64 = getattr(result, "b64_json", None) or getattr(result, "data", None)
            if not b64 and isinstance(result, dict):
                b64 = result.get("b64_json") or result.get("data")

        if not b64:
            return None

        return ImageBlock(
            media=MediaRef(
                kind="data",
                data_base64=b64,
                mime_type="image/png",
            )
        )

    @staticmethod
    def _extract_tool_call_from_item(item: Any) -> tuple[ToolCallBlock | None, dict]:
        """Extract a ``ToolCallBlock`` from a ``type='function_call'`` item."""
        name = getattr(item, "name", "")
        raw_args = getattr(item, "arguments", "")
        call_id = getattr(item, "call_id", "")

        try:
            args_dict = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args_dict = {}

        block = ToolCallBlock(name=name, args=args_dict, id=call_id)

        raw = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": raw_args or "{}",
            },
        }
        return block, raw

    @staticmethod
    def _extract_token_usage(response: Any) -> TokenUsages:
        """Map Responses API usage fields to ``TokenUsages``."""
        if response is None:
            return TokenUsages(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        usage = getattr(response, "usage", None)
        if not usage:
            return TokenUsages(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = input_tokens + output_tokens

        # Reasoning tokens
        output_details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) if output_details else 0

        # Cache tokens
        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", 0) if input_details else 0

        return TokenUsages(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens or 0,
            cache_read_input_tokens=cached_tokens or 0,
        )
