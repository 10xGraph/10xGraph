import base64
import json
import logging
from typing import TYPE_CHECKING, Any, Union

from agentflow.core.state.message import Message
from agentflow.core.state.message_block import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    RemoteToolCallBlock,
    TextBlock,
    ToolResultBlock,
    VideoBlock,
)


if TYPE_CHECKING:
    from agentflow.core.state import AgentState
    from agentflow.storage.media.resolver import MediaRefResolver

logger = logging.getLogger("agentflow.utils")


_AGENTFLOW_SCHEME = "agentflow://media/"

_MEDIA_BLOCK_TYPES = ImageBlock | AudioBlock | VideoBlock | DocumentBlock
_MEDIA_DICT_TYPES = {"image", "audio", "video", "document"}


def strip_media_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove multimodal content parts from a list of LLM message dicts.

    When a text-only agent receives messages that originated from a
    multimodal agent (multi-agent workflow), images/audio/video/documents
    must be stripped so the text-only model doesn't receive content it
    cannot process.

    Operates on already-converted dicts (output of ``convert_messages``).

    * ``content: str`` → unchanged (already text-only).
    * ``content: list`` → non-text parts removed; if only text remains
      the list is collapsed back to a plain string.
    """
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        # Keep only text parts
        text_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "text"]

        if not text_parts:
            # All content was media — collapse to empty string
            cleaned.append({**msg, "content": ""})
        elif len(text_parts) == 1:
            # Single text part — collapse to plain string
            cleaned.append({**msg, "content": text_parts[0].get("text", "")})
        else:
            # Multiple text parts — keep as list
            cleaned.append({**msg, "content": text_parts})

    return cleaned


async def resolve_media_refs(
    messages: list[Message],
    resolver: "MediaRefResolver",
    provider: str | None = None,
    model: str | None = None,
) -> list[Message]:
    """Pre-resolve ``agentflow://media/`` URLs in message media blocks.

    Replaces internal media refs with inline base64 so the sync converter
    pipeline can handle them without async calls.  Only touches blocks
    whose ``media.url`` starts with ``agentflow://media/``.

    When *provider* and *model* are provided, uses the capability-aware
    resolution path (e.g. Google will not use signed URLs, will fall back
    to inline bytes).  Otherwise uses the legacy direct-resolution path.

    This should be called *before* ``convert_messages()`` when a MediaStore
    is configured.  If no MediaStore is in use, skip this step.
    """
    for msg in messages:
        for block in msg.content:
            if not isinstance(block, _MEDIA_BLOCK_TYPES):
                continue
            media = block.media  # type: ignore[union-attr]
            if not (media.kind == "url" and media.url and media.url.startswith(_AGENTFLOW_SCHEME)):
                continue

            # Use capability-aware resolution when provider+model are known
            if provider and model:
                resolved = await resolver.resolve_for_openai(media, model=model)
                if provider == "google":
                    resolved = await resolver.resolve_for_google(media, model=model)

                if resolved and "image_url" in resolved:
                    url = resolved["image_url"]["url"]
                    if url.startswith("data:"):
                        _, b64_data = url.partition(",")
                        mime = url.split(":")[1].split(";")[0]
                        block.media = media.model_copy(  # type: ignore[union-attr]
                            update={
                                "kind": "data",
                                "data_base64": b64_data,
                                "mime_type": mime,
                                "url": None,
                                "size_bytes": len(b64_data),
                            }
                        )
                    else:
                        block.media = media.model_copy(  # type: ignore[union-attr]
                            update={
                                "kind": "url",
                                "url": url,
                            }
                        )
                continue

            # Legacy path: try signed URL first, then base64 fallback
            direct_url = await resolver._get_direct_url(media)  # type: ignore[attr-defined]
            if direct_url:
                block.media = media.model_copy(  # type: ignore[union-attr]
                    update={
                        "kind": "url",
                        "url": direct_url,
                    }
                )
                continue

            # Resolve internal URL -> inline base64
            key = media.url.removeprefix(_AGENTFLOW_SCHEME)
            data, mime = await resolver.media_store.retrieve(key)  # type: ignore[union-attr]
            b64 = base64.b64encode(data).decode()
            block.media = media.model_copy(  # type: ignore[union-attr]
                update={
                    "kind": "data",
                    "data_base64": b64,
                    "mime_type": mime,
                    "url": None,
                    "size_bytes": len(data),
                }
            )

    return messages


def _get_message_content_blocks(message: Message) -> list[Any]:
    """Return a normalized list of content blocks for a message."""
    content = getattr(message, "content", None)
    if content is None:
        return []
    if isinstance(content, list):
        return content
    return [content]


def _has_remote_tool_call_block(blocks: list[Any]) -> bool:
    """Return True when any block represents a remote tool call."""
    for block in blocks:
        if isinstance(block, RemoteToolCallBlock):
            return True
        if isinstance(block, dict) and block.get("type") == "remote_tool_call":
            return True
    return False


def _has_multimodal_blocks(blocks: list[Any]) -> bool:
    """Return True when the block list contains any non-text media block."""
    for block in blocks:
        if isinstance(block, _MEDIA_BLOCK_TYPES):
            return True
        if isinstance(block, dict) and block.get("type") in _MEDIA_DICT_TYPES:
            return True
    return False


def _image_block_to_openai(block: ImageBlock) -> dict[str, Any]:
    """Convert an ImageBlock to OpenAI content part format."""
    media = block.media
    if media.kind == "data" and media.data_base64:
        mime = media.mime_type or "image/png"
        url = f"data:{mime};base64,{media.data_base64}"
    elif media.kind == "url" and media.url:
        url = media.url
    elif media.kind == "file_id" and media.file_id:
        url = media.file_id
    else:
        # Fallback: try url, then file_id
        url = media.url or media.file_id or ""
    return {"type": "image_url", "image_url": {"url": url}}


def _audio_block_to_openai(block: AudioBlock) -> dict[str, Any]:
    """Convert an AudioBlock to OpenAI content part format."""
    media = block.media
    if media.kind == "data" and media.data_base64:
        fmt = "wav"
        if media.mime_type:
            # e.g. "audio/mp3" -> "mp3"
            fmt = media.mime_type.split("/")[-1]
        return {
            "type": "input_audio",
            "input_audio": {"data": media.data_base64, "format": fmt},
        }
    # For URL/file_id audio, pass as text description (OpenAI doesn't support URL audio natively)
    return {"type": "text", "text": f"[Audio: {media.url or media.file_id or 'unknown'}]"}


def _document_block_to_openai(block: DocumentBlock) -> list[dict[str, Any]]:
    """Convert a DocumentBlock to content parts.

    Returns a list because a document may produce multiple parts (e.g.
    extracted text + the original reference).  For extracted text, returns
    a plain ``text`` part.  For raw documents, returns a ``document`` part
    with the data/url so downstream converters (Google, Responses API) can
    handle them appropriately.
    """
    parts: list[dict[str, Any]] = []
    # If there's an excerpt (extracted text), include it as text
    if block.excerpt:
        parts.append({"type": "text", "text": block.excerpt})
    else:
        media = block.media
        if media.kind == "data" and media.data_base64:
            mime = media.mime_type or "application/pdf"
            parts.append(
                {
                    "type": "document",
                    "document": {
                        "data": media.data_base64,
                        "mime_type": mime,
                        "url": None,
                        "text": None,
                    },
                }
            )
        elif media.kind == "url" and media.url:
            mime = media.mime_type or "application/pdf"
            parts.append(
                {
                    "type": "document",
                    "document": {
                        "url": media.url,
                        "mime_type": mime,
                        "data": None,
                        "text": None,
                    },
                }
            )
        elif media.kind == "file_id" and media.file_id:
            parts.append({"type": "text", "text": f"[Document file_id: {media.file_id}]"})
        else:
            parts.append({"type": "text", "text": "[Document: unsupported reference]"})
    return parts


def _video_block_to_openai(block: VideoBlock) -> dict[str, Any]:
    """Convert a VideoBlock to a content part.

    OpenAI Chat doesn't natively support video, but Google GenAI does.
    We produce a ``video`` typed part so downstream converters can handle it.
    """
    media = block.media
    if media.kind == "data" and media.data_base64:
        mime = media.mime_type or "video/mp4"
        return {
            "type": "video",
            "video": {"data": media.data_base64, "mime_type": mime, "url": None},
        }
    if media.kind == "url" and media.url:
        mime = media.mime_type or "video/mp4"
        return {
            "type": "video",
            "video": {"url": media.url, "mime_type": mime, "data": None},
        }
    return {"type": "text", "text": f"[Video: {media.url or media.file_id or 'unknown'}]"}


def _build_text_content(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append(block.text or "")
            continue

        if isinstance(block, ToolResultBlock):
            parts.append(
                block.output
                if isinstance(block.output, str)
                else json.dumps(block.output, default=str)
            )
            continue

        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))

    return "".join(parts)


def _append_content_part(content_parts: list[dict[str, Any]], block: Any) -> None:
    if isinstance(block, TextBlock):
        if block.text:
            content_parts.append({"type": "text", "text": block.text})
        return

    if isinstance(block, ImageBlock):
        content_parts.append(_image_block_to_openai(block))
        return

    if isinstance(block, AudioBlock):
        content_parts.append(_audio_block_to_openai(block))
        return

    if isinstance(block, DocumentBlock):
        content_parts.extend(_document_block_to_openai(block))
        return

    if isinstance(block, VideoBlock):
        content_parts.append(_video_block_to_openai(block))
        return

    if isinstance(block, dict) and block.get("type") == "text":
        text = str(block.get("text") or "")
        if text:
            content_parts.append({"type": "text", "text": text})


def _build_content(blocks: list[Any]) -> str | list[dict[str, Any]]:
    """Build content value for an OpenAI-style message dict.

    If the message is text-only, returns a plain string (backward compat).
    If multimodal blocks are present, returns a list of content parts.
    """
    if not _has_multimodal_blocks(blocks):
        return _build_text_content(blocks)

    # Multimodal: build list of content parts
    content_parts: list[dict[str, Any]] = []
    for block in blocks:
        _append_content_part(content_parts, block)

    return content_parts


def _convert_dict(message: Message) -> dict[str, Any] | None:
    """
    Convert a Message object to a dictionary for LLM/tool payloads.

    When a message contains multimodal content (images, audio, documents),
    the ``content`` value is returned as a list of content parts in OpenAI
    format.  Text-only messages keep the simple string ``content``.

    Args:
        message (Message): The message to convert.

    Returns:
        dict[str, Any]: Dictionary representation of the message.
    """
    # if any remote tool call exists we are skipping the tool result block
    # as remote tool calls are not supported in the current implementation
    blocks = _get_message_content_blocks(message)

    if _has_remote_tool_call_block(blocks):
        return None

    if message.role == "tool":
        call_id = ""
        for i in blocks:
            if isinstance(i, ToolResultBlock):
                call_id = i.call_id
                break
            if isinstance(i, dict) and i.get("type") == "tool_result":
                call_id = str(i.get("call_id", ""))
                break

        return {
            "role": message.role,
            "content": message.text(),
            "tool_call_id": call_id,
        }

    if message.role == "assistant" and message.tools_calls:
        return {
            "role": message.role,
            "content": message.text(),
            "tool_calls": message.tools_calls,
        }

    # Check for multimodal blocks
    content = _build_content(blocks)

    return {"role": message.role, "content": content}


def _interpolate_system_prompts(
    system_prompts: list[dict[str, Any]],
    state: Union["AgentState", None],
) -> list[dict[str, Any]]:
    """Interpolate state variables into system prompt content.

    Supports placeholders like {field_name} in system prompt strings.
    Uses model_dump() to get all state fields for interpolation.

    Args:
        system_prompts: List of system prompt dicts with "role" and "content".
        state: Current agent state with custom fields.

    Returns:
        List of system prompts with interpolated content.
    """
    if state is None:
        return system_prompts

    interpolated = []
    state_dict = state.model_dump()

    for prompt in system_prompts:
        if not isinstance(prompt.get("content"), str):
            # Non-string content (e.g., multimodal), pass through as-is
            interpolated.append(prompt)
            continue

        content = prompt["content"]
        try:
            # Interpolate placeholders with state variables
            interpolated_content = content.format(**state_dict)
            interpolated.append({**prompt, "content": interpolated_content})
        except KeyError as e:
            # Missing field in state - log warning and use original
            logger.warning(
                "Failed to interpolate system prompt: missing field %s. "
                "Using original prompt without interpolation.",
                e,
            )
            interpolated.append(prompt)
        except (ValueError, IndexError) as e:
            # Invalid format string or other formatting issues
            logger.warning(
                "Failed to interpolate system prompt due to formatting error: %s. "
                "Using original prompt without interpolation.",
                e,
            )
            interpolated.append(prompt)

    return interpolated


def convert_messages(
    system_prompts: list[dict[str, Any]],
    state: Union["AgentState", None] = None,
    extra_messages: list[Message] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert system prompts, agent state, and extra messages to a list of dicts for
    LLM/tool payloads.

    Args:
        system_prompts (list[dict[str, Any]]): List of system prompt dicts.
        state (AgentState | None): Optional agent state containing context and summary.
        extra_messages (list[Message] | None): Optional extra messages to include.

    Returns:
        list[dict[str, Any]]: List of message dicts for payloads.

    Raises:
        ValueError: If system_prompts is None.
    """
    if system_prompts is None:
        logger.error("System prompts are None")
        raise ValueError("System prompts cannot be None")

    # Interpolate state variables into system prompts
    interpolated_prompts = _interpolate_system_prompts(system_prompts, state)

    res = []
    res += interpolated_prompts

    if state and state.context_summary:
        summary = {
            "role": "assistant",
            "content": state.context_summary if state.context_summary else "",
        }
        res.append(summary)

    if state and state.context:
        for msg in state.context:
            formatted = _convert_dict(msg)
            if formatted:
                res.append(formatted)

    if extra_messages:
        for msg in extra_messages:
            formatted = _convert_dict(msg)
            if formatted:
                res.append(formatted)

    logger.debug("Number of Converted messages: %s", len(res))
    return res
