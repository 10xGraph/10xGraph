"""OpenAI request helpers for Agent."""

from __future__ import annotations

import logging
from typing import Any

from .constants import CALL_EXCLUDED_KWARGS, resolve_reasoning_config


logger = logging.getLogger("tenxgraph.agent")

# Models that answered the default reasoning parameter with a 400 (non-reasoning models such as
# gpt-4o). Process-local: later calls with the default config skip reasoning for them. An
# explicit reasoning_config is never affected.
_REASONING_REJECTED_MODELS: set[str] = set()

# Status codes meaning the endpoint does not exist or is not implemented on the server. Only
# these switch a custom base_url from the Responses API to Chat Completions.
_ENDPOINT_UNSUPPORTED_STATUS = frozenset({404, 405, 501})


def is_endpoint_unsupported(exc: BaseException) -> bool:
    """True when ``exc`` says the endpoint is missing or unsupported (404, 405, 501)."""
    return getattr(exc, "status_code", None) in _ENDPOINT_UNSUPPORTED_STATUS


def is_reasoning_rejected(exc: BaseException) -> bool:
    """True when ``exc`` is a 400 that names the reasoning parameter.

    The error's ``param`` decides when the server sets one, so a 400 about reasoning items in
    the input (``param="input"``) does not count; the message is checked only without it.
    """
    if getattr(exc, "status_code", None) != 400:  # noqa: PLR2004
        return False
    param = getattr(exc, "param", None)
    if param:
        return "reasoning" in str(param).lower()
    return "reasoning" in str(exc).lower()


def _to_responses_content(content: Any) -> Any:
    """Convert Chat Completions content format to Responses API format.

    The Responses API uses different content part types:
    - ``text``        → ``input_text``
    - ``image_url``   → ``input_image``  (flattened URL)
    - ``input_audio`` stays the same
    - ``document``    → ``input_file``  or ``input_text`` (excerpt)

    If *content* is a plain string, it is returned unchanged.
    """
    if not isinstance(content, list):
        return content

    converted: list[dict[str, Any]] = []
    for part in content:
        ptype = part.get("type", "")

        if ptype == "text":
            converted.append({"type": "input_text", "text": part.get("text", "")})

        elif ptype == "image_url":
            image_info = part.get("image_url", {})
            url = image_info.get("url", "") if isinstance(image_info, dict) else str(image_info)
            converted.append({"type": "input_image", "image_url": url})

        elif ptype == "input_audio":
            audio_info = part.get("input_audio", {})
            converted.append(
                {
                    "type": "input_audio",
                    "data": audio_info.get("data", ""),
                    "format": audio_info.get("format", "wav"),
                }
            )

        elif ptype == "document":
            # Documents with extracted text → input_text; raw docs → input_file
            doc_info = part.get("document", {})
            if isinstance(doc_info, dict) and doc_info.get("text"):
                converted.append({"type": "input_text", "text": doc_info["text"]})
            elif isinstance(doc_info, dict) and doc_info.get("url"):
                converted.append(
                    {
                        "type": "input_file",
                        "file_url": doc_info["url"],
                    }
                )
            else:
                # Fallback: stringify whatever we have
                converted.append({"type": "input_text", "text": str(doc_info)})

        elif ptype == "video":
            # Responses API doesn't natively support video input;
            # pass as input_text reference or input_image for frame URLs
            video_info = part.get("video", {})
            url = video_info.get("url", "") if isinstance(video_info, dict) else str(video_info)
            if url:
                converted.append({"type": "input_text", "text": f"[Video: {url}]"})

        else:
            # Pass unknown parts through unchanged
            converted.append(part)

    return converted


class AgentOpenAIMixin:
    """OpenAI and OpenAI-compatible API request helpers."""

    def _openai_reasoning_config(self) -> dict[str, Any] | None:
        """Reasoning config for this OpenAI request, or None.

        The default config is dropped for a model in ``_REASONING_REJECTED_MODELS``.
        """
        if getattr(self, "_reasoning_default", False) and (
            getattr(self, "model", None) in _REASONING_REJECTED_MODELS
        ):
            return None
        return resolve_reasoning_config(self)

    async def _call_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call OpenAI chat, image, or audio endpoints."""
        call_kwargs = {
            key: value
            for key, value in {**self.llm_kwargs, **kwargs}.items()
            if key not in CALL_EXCLUDED_KWARGS
        }

        output_schema = getattr(self, "output_schema", None)
        if output_schema:
            if tools:
                call_kwargs["tools"] = tools

            logger.debug("Calling OpenAI chat.completions.parse with model=%s", self.model)
            response = await self.client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=output_schema,
                stream=False,
                **call_kwargs,
            )
            details = getattr(getattr(response, "usage", None), "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            if cached:
                logger.debug("Cache hit: %d cached tokens (OpenAI chat completions)", cached)
            return response

        if self.output_type in ("text", "json"):
            if tools:
                call_kwargs["tools"] = tools
            if stream:
                # Without this OpenAI sends no usage chunk and streamed runs report 0 tokens.
                call_kwargs.setdefault("stream_options", {"include_usage": True})

            logger.debug("Calling OpenAI chat.completions.create with model=%s", self.model)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **call_kwargs,
            )
            if not stream:
                details = getattr(getattr(response, "usage", None), "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", 0) or 0
                if cached:
                    logger.debug("Cache hit: %d cached tokens (OpenAI chat completions)", cached)
            return response

        if self.output_type == "image":
            prompt = self._extract_prompt(messages)
            logger.debug("Calling OpenAI images.generate with model=%s", self.model)
            return await self.client.images.generate(
                model=self.model,
                prompt=prompt,
                **call_kwargs,
            )

        if self.output_type == "audio":
            text = self._extract_prompt(messages)
            logger.debug("Calling OpenAI audio.speech.create with model=%s", self.model)
            return await self.client.audio.speech.create(
                model=self.model,
                input=text,
                **call_kwargs,
            )

        raise ValueError(f"Unsupported output_type '{self.output_type}' for OpenAI provider")

    async def _call_openai_responses(  # noqa: PLR0912
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call the OpenAI Responses API using chat-style messages as input."""
        call_kwargs: dict[str, Any] = {
            key: value
            for key, value in {**self.llm_kwargs, **kwargs}.items()
            if key not in CALL_EXCLUDED_KWARGS
        }

        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role", "")
            if role == "system":
                instructions_parts.append(str(message.get("content", "")))
            elif role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.get("tool_call_id", ""),
                        "output": str(message.get("content", "")),
                    }
                )
            elif role == "assistant" and message.get("tool_calls"):
                text_content = message.get("content", "")
                if text_content:
                    input_items.append(
                        {
                            "role": "assistant",
                            "content": _to_responses_content(text_content),
                        }
                    )

                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function", {})
                    input_items.append(
                        {
                            "type": "function_call",
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", "{}"),
                            "call_id": tool_call.get("id", ""),
                        }
                    )
            else:
                input_items.append(
                    {
                        "role": role,
                        "content": _to_responses_content(message.get("content", "")),
                    }
                )

        instructions = "\n".join(instructions_parts) if instructions_parts else None

        responses_tools: list[dict[str, Any]] | None = None
        if tools:
            responses_tools = []
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    function = tool["function"]
                    response_tool: dict[str, Any] = {
                        "type": "function",
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                    }
                    if "parameters" in function:
                        response_tool["parameters"] = function["parameters"]
                    if "strict" in function:
                        response_tool["strict"] = function["strict"]
                    responses_tools.append(response_tool)
                else:
                    responses_tools.append(tool)

        if instructions:
            call_kwargs["instructions"] = instructions
        if responses_tools:
            call_kwargs["tools"] = responses_tools
        reasoning = self._openai_reasoning_config()
        if reasoning:
            # Without a summary the Responses API returns no reasoning text at all.
            reasoning = dict(reasoning)
            if "summary" not in reasoning and "generate_summary" not in reasoning:
                reasoning["summary"] = "auto"
            call_kwargs["reasoning"] = reasoning

        call_kwargs.pop("reasoning_effort", None)

        response = await self._send_responses_request(input_items, stream, call_kwargs)
        if getattr(self, "output_schema", None):
            stream = False
        if not stream:
            details = getattr(getattr(response, "usage", None), "input_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            if cached:
                logger.debug("Cache hit: %d cached tokens (OpenAI responses API)", cached)
        return response

    async def _send_responses_request(
        self, input_items: list[dict[str, Any]], stream: bool, call_kwargs: dict[str, Any]
    ) -> Any:
        """Send the request: ``responses.parse`` with an output schema, else ``create``."""
        output_schema = getattr(self, "output_schema", None)
        if output_schema:
            # responses.parse validates the reply into ``output_schema``. It returns the whole
            # response at once, so streaming is turned off, as on the chat parse path.
            logger.debug("Calling OpenAI responses.parse with model=%s", self.model)
            return await self.client.responses.parse(
                model=self.model,
                input=input_items,
                text_format=output_schema,
                stream=False,
                **call_kwargs,
            )
        logger.debug("Calling OpenAI responses.create with model=%s", self.model)
        return await self.client.responses.create(
            model=self.model,
            input=input_items,
            stream=stream,
            **call_kwargs,
        )
