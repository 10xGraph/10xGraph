"""Pure request translation for the Anthropic Messages API.

Every function here is side-effect free: no client, no network, no ``AgentState``.
Agentflow speaks an OpenAI-shaped internal dialect (system messages inside the
message list, ``tool_calls`` arrays, one ``role: "tool"`` message per result).
Anthropic wants something structurally different, and this module is the whole of
that difference so it can be tested exhaustively without a request.

The four structural gaps:

===========================  ==========================================
Agentflow (OpenAI-shaped)    Anthropic
===========================  ==========================================
``role: "system"`` messages  top-level ``system=`` parameter
``tool_calls`` array         ``tool_use`` content blocks
one ``role: "tool"`` each    all ``tool_result`` blocks in ONE user turn
``{"type": "function"...}``  ``{"name", "description", "input_schema"}``
===========================  ==========================================
"""

from __future__ import annotations

import logging
from functools import cache
from typing import Any

from tenxgraph.runtime.adapters.llm.tool_arguments import parse_tool_arguments
from tenxgraph.utils.converter import THINKING_BLOCKS_KEY
from tenxgraph.utils.media_scheme import parse_data_uri


logger = logging.getLogger("tenxgraph.agent.anthropic")

# Content part types Anthropic has no equivalent for. Dropped with a warning,
# matching how the OpenAI path degrades video rather than inventing a failure.
_UNSUPPORTED_PART_TYPES = ("input_audio", "video")

_DATA_URI_PREFIX = "data:"

_BEDROCK_PREFIX = "anthropic."


@cache
def _sdk_keys(param_type: str) -> frozenset[str]:
    """Keys the Anthropic SDK defines on a request type (``TextBlockParam``, ``ToolParam``).

    Imported lazily so the anthropic package stays optional. The SDK's own types
    decide which developer-written fields are kept, so new API fields need no
    change here.
    """
    from anthropic import types

    return frozenset(getattr(types, param_type).__annotations__)


def _keep_sdk_keys(
    source: dict[str, Any], param_type: str, skip: tuple[str, ...] = ()
) -> dict[str, Any]:
    keys = _sdk_keys(param_type)
    return {k: v for k, v in source.items() if k in keys and k not in skip}


def _text_block(part: dict[str, Any]) -> dict[str, Any]:
    return {**_keep_sdk_keys(part, "TextBlockParam"), "type": "text", "text": part.get("text", "")}


def strip_bedrock_prefix(model: str) -> str:
    """Strip a Bedrock ``anthropic.`` prefix for capability lookups only.

    The prefix is part of the real Bedrock model id and must be preserved in the
    string actually sent to the client, so this is for set-membership checks
    (does this model reject sampling params?), never for building the request.
    """
    return model[len(_BEDROCK_PREFIX) :] if model.startswith(_BEDROCK_PREFIX) else model


def bare_model_id(model: str) -> str:
    """Bare lowercase Claude id for capability lookups only, never for the request.

    Strips a provider prefix, a Bedrock ``anthropic.`` or cross-region ``us.anthropic.``
    prefix, and a Vertex ``@date`` suffix.
    """
    bare = model.split("/")[-1].split("@")[0]
    if "anthropic." in bare:
        bare = bare.split("anthropic.", 1)[1]
    return strip_bedrock_prefix(bare).lower()


def split_system(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Split ``role: "system"`` entries out into Anthropic's top-level ``system``.

    Anthropic takes the system prompt as a separate request parameter, not as a
    message. Multiple system prompts become a list of text blocks rather than
    being concatenated, because that list is also the correct attachment point
    for ``cache_control`` breakpoints later.

    Returns:
        A ``(system, messages)`` tuple. ``system`` is ``None`` when the input
        carried no system prompt.
    """
    system_blocks: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []

    for message in messages:
        if message.get("role") != "system":
            remainder.append(message)
            continue

        content = message.get("content", "")
        if isinstance(content, str):
            if content:
                system_blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    system_blocks.append(_text_block(part))
                elif isinstance(part, str):
                    system_blocks.append({"type": "text", "text": part})
        elif content:
            system_blocks.append({"type": "text", "text": str(content)})

    return (system_blocks or None), remainder


def convert_content_parts(content: Any) -> Any:
    """Convert internal content parts into Anthropic content blocks.

    A plain string passes through unchanged: Anthropic accepts a bare string as
    message content, so there is no reason to wrap it.
    """
    if not isinstance(content, list):
        return content

    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            converted.append({"type": "text", "text": str(part)})
            continue

        ptype = part.get("type", "")

        if ptype == "text":
            converted.append(_text_block(part))

        elif ptype == "image_url":
            block = _convert_image_part(part)
            if block is not None:
                converted.append(block)

        elif ptype == "document":
            converted.append(_convert_document_part(part))

        elif ptype in _UNSUPPORTED_PART_TYPES:
            logger.warning(
                "Anthropic has no %s content block; dropping this part from the request.",
                ptype,
            )

        else:
            # Already-Anthropic-shaped blocks (text, image, document, tool_use,
            # tool_result, thinking) pass through untouched.
            converted.append(part)

    return converted


def _convert_image_part(part: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAI ``image_url`` part into an Anthropic ``image`` block."""
    image_info = part.get("image_url", {})
    url = image_info.get("url", "") if isinstance(image_info, dict) else str(image_info)

    if not url:
        logger.warning("Dropping image content part with no URL.")
        return None

    if url.startswith(_DATA_URI_PREFIX):
        media_type, data = parse_data_uri(url, "application/octet-stream")
        if data is None:
            logger.warning("Dropping malformed image data URI.")
            return None
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }

    return {"type": "image", "source": {"type": "url", "url": url}}


def _convert_document_part(part: dict[str, Any]) -> dict[str, Any]:
    """Convert a ``document`` part into an Anthropic ``document`` (or text) block.

    Documents that already carry extracted text degrade to a text block, which is
    what the OpenAI path does too: re-sending the raw PDF when the text is already
    in hand only costs tokens.
    """
    doc_info = part.get("document", {})
    if not isinstance(doc_info, dict):
        return {"type": "text", "text": str(doc_info)}

    if doc_info.get("text"):
        return {"type": "text", "text": doc_info["text"]}

    url = doc_info.get("url", "")
    if url.startswith(_DATA_URI_PREFIX):
        media_type, data = parse_data_uri(url, "application/octet-stream")
        if data is not None:
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
    elif url:
        return {"type": "document", "source": {"type": "url", "url": url}}

    return {"type": "text", "text": str(doc_info)}


def convert_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Unwrap OpenAI-shaped tool definitions into Anthropic's flat shape.

    Entries that are already Anthropic-shaped pass through untouched, so server
    tools (``web_search_20260209`` and friends) and MCP toolsets are not mangled.
    """
    if not tools:
        return None

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            converted.append(tool)
            continue

        function = tool.get("function")
        if tool.get("type") == "function" and isinstance(function, dict):
            # Developer fields (cache_control, strict, defer_loading, ...) are kept
            # when the SDK's ToolParam defines them, from the wrapper or from
            # ``function``. Other keys (e.g. x-function-capabilities) are not sent.
            entry = _keep_sdk_keys(tool, "ToolParam", skip=("type",))
            entry.update(_keep_sdk_keys(function, "ToolParam", skip=("type",)))
            entry.update(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
            if entry.get("strict") is None:
                entry.pop("strict", None)
            converted.append(entry)
        else:
            converted.append(tool)

    return converted


def convert_assistant_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Turn an assistant ``tool_calls`` array into ``tool_use`` content blocks."""
    # Signed thinking blocks go first, unchanged: Anthropic rejects an edited or reordered one.
    blocks: list[dict[str, Any]] = list(message.get(THINKING_BLOCKS_KEY) or [])

    content = message.get("content")
    if content:
        converted = convert_content_parts(content)
        if isinstance(converted, str):
            blocks.append({"type": "text", "text": converted})
        elif isinstance(converted, list):
            blocks.extend(converted)

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        name = function.get("name", "")
        # Anthropic wants a decoded object; never string-match: Claude varies its JSON escaping.
        args, _ = parse_tool_arguments(function.get("arguments", "{}"), tool_name=name)
        blocks.append(
            {"type": "tool_use", "id": tool_call.get("id", ""), "name": name, "input": args}
        )

    return {"role": "assistant", "content": blocks}


def merge_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive ``role: "tool"`` messages into one user turn.

    This is the one structurally non-mechanical transform. Agentflow emits one
    ``role: "tool"`` message per tool call. Anthropic requires every
    ``tool_result`` for a given assistant turn to arrive in a **single**
    ``role: "user"`` message. Splitting them across messages is accepted by the
    API but degrades parallel tool calling, so they must be merged rather than
    emitted one apiece.
    """
    merged: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            merged.append({"role": "user", "content": list(pending)})
            pending.clear()

    for message in messages:
        role = message.get("role")

        if role == "tool":
            pending.append(_to_tool_result_block(message))
            continue

        flush()

        if role == "assistant" and message.get("tool_calls"):
            merged.append(convert_assistant_tool_calls(message))
        else:
            content = convert_content_parts(message.get("content", ""))
            thinking = message.get(THINKING_BLOCKS_KEY) if role == "assistant" else None
            if thinking:
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}] if content else []
                content = [*thinking, *(content or [])]
            merged.append({"role": role, "content": content})

    flush()
    return merged


def _to_tool_result_block(message: dict[str, Any]) -> dict[str, Any]:
    """Build a single ``tool_result`` block from a ``role: "tool"`` message."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": message.get("tool_call_id", ""),
        "content": _stringify_tool_content(message.get("content", "")),
    }
    if message.get("is_error"):
        block["is_error"] = True
    return block


def _stringify_tool_content(content: Any) -> Any:
    """Normalise tool result content into something Anthropic accepts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return convert_content_parts(content)
    return str(content)


def _has_cache_control(
    system: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    call_kwargs: dict[str, Any],
) -> bool:
    """True when the developer placed ``cache_control`` on the request themselves."""
    if "cache_control" in call_kwargs:
        return True
    blocks = [*(system or []), *(tools or [])]
    for message in messages:
        if isinstance(message.get("content"), list):
            blocks.extend(message["content"])
    return any(isinstance(block, dict) and "cache_control" in block for block in blocks)


def apply_cache_control(
    cache: bool | dict[str, Any] | None,
    system: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    call_kwargs: dict[str, Any],
) -> None:
    """Turn on Anthropic prompt caching for the request, in place.

    Two parts, both using the same control (so the same TTL): a breakpoint on the
    end of the stable prefix (the last system block, or the last tool when there
    is no system prompt), and a top-level ``cache_control`` in ``call_kwargs``
    (automatic caching, which caches the conversation and moves forward each call).

    If the developer placed ``cache_control`` anywhere (top level, system blocks,
    tools, message blocks) nothing is added, so their own strategy is never mixed
    with ours.

    Args:
        cache: ``False``/``None`` disables. ``True`` uses ``{"type": "ephemeral"}``.
            A dict is used verbatim as the control, so
            ``{"type": "ephemeral", "ttl": "1h"}`` extends retention.
        system: System blocks, mutated in place.
        tools: Anthropic-shaped tool definitions, mutated in place.
        messages: Anthropic-shaped messages, only read.
        call_kwargs: Request kwargs, gets the top-level ``cache_control``.

    Note:
        The minimum cacheable prefix depends on the model; a shorter prefix
        silently does not cache. Verify with ``usage.cache_read_input_tokens``.
    """
    if not cache or _has_cache_control(system, tools, messages, call_kwargs):
        return

    control = cache if isinstance(cache, dict) else {"type": "ephemeral"}

    if system:
        system[-1] = {**system[-1], "cache_control": control}
    elif tools:
        tools[-1] = {**tools[-1], "cache_control": control}

    call_kwargs["cache_control"] = control


def drop_trailing_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove a trailing assistant turn, which Anthropic rejects as a prefill.

    Assistant prefills return a 400 on every current Claude model. This matters
    specifically because agentflow injects ``state.context_summary`` as an
    assistant message, which becomes the trailing turn whenever ``state.context``
    is empty.

    Dropping that turn must not leave the request with no messages at all, so
    when it is the *only* turn it is demoted to a ``user`` turn instead of being
    discarded: the summary survives as context and the request stays sendable.
    Losing it silently would send the model an empty conversation.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return messages

    if len(messages) == 1:
        logger.debug(
            "Demoting the lone trailing assistant message to a user turn: "
            "Anthropic rejects assistant prefills, but an empty message list is "
            "not a sendable request either."
        )
        return [{**messages[0], "role": "user"}]

    logger.debug(
        "Dropping trailing assistant message: Anthropic rejects assistant "
        "prefills with a 400 on current models."
    )
    return messages[:-1]
