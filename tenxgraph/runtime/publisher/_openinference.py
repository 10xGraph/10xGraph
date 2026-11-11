"""OpenInference span attributes, read by Arize Phoenix and other OpenInference viewers.

Pure helpers: each returns a plain ``dict`` of attributes (or a string) and never touches a
span, so ``OtelPublisher`` decides what to set per observability level. Attribute names follow
the OpenInference semantic conventions (``openinference.semconv.trace``).
"""

from __future__ import annotations

import json
from typing import Any


SPAN_KIND = "openinference.span.kind"
USER_ID = "user.id"
INPUT_VALUE = "input.value"
INPUT_MIME = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME = "output.mime_type"
LLM_MODEL_NAME = "llm.model_name"
LLM_PROVIDER = "llm.provider"
TOKENS_PROMPT = "llm.token_count.prompt"
TOKENS_COMPLETION = "llm.token_count.completion"
TOKENS_TOTAL = "llm.token_count.total"

KIND_CHAIN = "CHAIN"
KIND_LLM = "LLM"
KIND_TOOL = "TOOL"

JSON_MIME = "application/json"
TEXT_MIME = "text/plain"


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def content_text(content: Any) -> str:
    """Plain text of a message ``content``: a string, or a list of text parts or blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text is None and part.get("type") == "tool_result":
                    text = part.get("output")
                if text is not None:
                    parts.append(text if isinstance(text, str) else _dumps(text))
        return "\n".join(parts)
    return _dumps(content)


def message_text(message: Any) -> str:
    """Text of one message dict (``Message.model_dump()`` or an LLM request message)."""
    return content_text(message.get("content")) if isinstance(message, dict) else ""


def last_message_text(messages: Any, role: str) -> str:
    """Text of the last message with ``role`` in a list of message dicts, else ``""``."""
    for m in reversed(messages if isinstance(messages, list) else []):
        if isinstance(m, dict) and m.get("role") == role:
            text = message_text(m)
            if text:
                return text
    return ""


def node_input_attrs(event_data: Any) -> dict[str, Any]:
    """``input.value`` for a node span: the state going into the node, as JSON.

    Empty when ``event_data`` carries no dict ``state``.
    """
    state = event_data.get("state") if isinstance(event_data, dict) else None
    if not isinstance(state, dict):
        return {}
    return {INPUT_VALUE: _dumps(state), INPUT_MIME: JSON_MIME}


def node_output_attrs(event_data: Any) -> dict[str, Any]:
    """``output.value`` for a node span: state after the node, messages produced, next node.

    Keys that are absent or ``None`` are omitted; empty when none remain.
    """
    if not isinstance(event_data, dict):
        return {}
    state = event_data.get("state")
    out: dict[str, Any] = {}
    if isinstance(state, dict):
        out["state"] = state
    if event_data.get("messages") is not None:
        out["messages"] = event_data["messages"]
    if event_data.get("next_node") is not None:
        out["next_node"] = event_data["next_node"]
    if not out:
        return {}
    return {OUTPUT_VALUE: _dumps(out), OUTPUT_MIME: JSON_MIME}


def kind_attrs(kind: str) -> dict[str, Any]:
    return {SPAN_KIND: kind}


def _call_fields(call: Any) -> tuple[str, str, str]:
    """``(id, name, arguments)`` of a tool call in OpenAI request form or ``{"id","name",...}``."""
    if not isinstance(call, dict):
        return "", "", ""
    raw_fn = call.get("function")
    fn: dict[str, Any] = raw_fn if isinstance(raw_fn, dict) else call
    args = fn.get("arguments")
    if args is not None and not isinstance(args, str):
        args = _dumps(args)
    return str(call.get("id") or ""), str(fn.get("name", "")), args or ""


def llm_request_messages(input_messages: Any, system_prompt: Any = None) -> list[dict[str, Any]]:
    """Request messages as ``[{"role", "content"}]``; the system prompt is added when absent.

    Assistant ``tool_calls`` and a tool message's ``tool_call_id`` are kept when present.
    """
    out: list[dict[str, Any]] = []
    for m in input_messages if isinstance(input_messages, list) else []:
        if isinstance(m, dict):
            msg: dict[str, Any] = {"role": str(m.get("role", "")), "content": message_text(m)}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            out.append(msg)
    if system_prompt and not any(m["role"] == "system" for m in out):
        text = system_prompt if isinstance(system_prompt, str) else content_text(system_prompt)
        out.insert(0, {"role": "system", "content": text})
    return out


def llm_standard_attrs(data: dict[str, Any], model: str, system_name: str) -> dict[str, Any]:
    """Model, provider and token counts (STANDARD and above)."""
    attrs: dict[str, Any] = {}
    if model:
        attrs[LLM_MODEL_NAME] = model
    if system_name:
        attrs[LLM_PROVIDER] = system_name
    inp = int(data.get("input_tokens") or 0)
    out = int(data.get("output_tokens") or 0)
    if inp:
        attrs[TOKENS_PROMPT] = inp
    if out:
        attrs[TOKENS_COMPLETION] = out
    if inp or out:
        attrs[TOKENS_TOTAL] = inp + out
    return attrs


def llm_input_attrs(input_messages: Any, system_prompt: Any = None) -> dict[str, Any]:
    """Indexed input messages plus ``input.value`` (FULL only)."""
    messages = llm_request_messages(input_messages, system_prompt)
    if not messages:
        return {}
    attrs: dict[str, Any] = {INPUT_VALUE: _dumps(messages), INPUT_MIME: JSON_MIME}
    for i, m in enumerate(messages):
        attrs[f"llm.input_messages.{i}.message.role"] = m["role"]
        attrs[f"llm.input_messages.{i}.message.content"] = m["content"]
        if m.get("tool_call_id"):
            attrs[f"llm.input_messages.{i}.message.tool_call_id"] = m["tool_call_id"]
        for j, call in enumerate(m.get("tool_calls", [])):
            prefix = f"llm.input_messages.{i}.message.tool_calls.{j}.tool_call"
            call_id, name, arguments = _call_fields(call)
            if call_id:
                attrs[f"{prefix}.id"] = call_id
            attrs[f"{prefix}.function.name"] = name
            attrs[f"{prefix}.function.arguments"] = arguments
    return attrs


def llm_output_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The assistant reply as ``[{"role": "assistant", "content": ..., "tool_calls"?}]``."""
    text = data.get("output_response") or ""
    calls = data.get("output_tool_calls") or []
    if not text and not calls:
        return []
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = calls
    return [msg]


def llm_output_attrs(data: dict[str, Any]) -> dict[str, Any]:
    """Indexed output message plus ``output.value`` (FULL only)."""
    messages = llm_output_messages(data)
    if not messages:
        return {}
    msg = messages[0]
    attrs: dict[str, Any] = {
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": msg["content"],
        OUTPUT_VALUE: msg["content"],
        OUTPUT_MIME: TEXT_MIME,
    }
    for j, call in enumerate(msg.get("tool_calls", [])):
        prefix = f"llm.output_messages.0.message.tool_calls.{j}.tool_call"
        if call.get("id"):
            attrs[f"{prefix}.id"] = str(call["id"])
        attrs[f"{prefix}.function.name"] = str(call.get("name", ""))
        attrs[f"{prefix}.function.arguments"] = str(call.get("arguments", ""))
    if not msg["content"]:
        attrs[OUTPUT_VALUE] = _dumps(msg.get("tool_calls", []))
        attrs[OUTPUT_MIME] = JSON_MIME
    return attrs


def graph_input_text(event_data: dict[str, Any]) -> str:
    """The user's last message in the state carried by a graph START event."""
    state = event_data.get("state")
    context = state.get("context") if isinstance(state, dict) else None
    return last_message_text(context, "user")


def graph_output_text(event_data: dict[str, Any]) -> str:
    """The final assistant text of a graph END event."""
    return last_message_text(event_data.get("messages"), "assistant")


def tool_input_attrs(args: Any) -> dict[str, Any]:
    if not args:
        return {}
    return {INPUT_VALUE: args if isinstance(args, str) else _dumps(args), INPUT_MIME: JSON_MIME}


def tool_output_attrs(event_data: dict[str, Any]) -> dict[str, Any]:
    """Result text from a tool END event (``message`` is a dumped tool ``Message``)."""
    message = event_data.get("message")
    text = message_text(message) if isinstance(message, dict) else ""
    if not text:
        raw = event_data.get("tool_output") or message
        text = raw if isinstance(raw, str) else (_dumps(raw) if raw else "")
    return {OUTPUT_VALUE: text, OUTPUT_MIME: TEXT_MIME} if text else {}
