"""OTel GenAI semconv message builders for ``gen_ai.input.messages`` / ``gen_ai.output.messages``.

Pure helpers: each returns plain dicts, so ``OtelPublisher`` decides when to set them. Each message
is ``{"role", "parts": [...]}`` with parts of type ``text``, ``tool_call``, ``tool_call_response``.
"""

from __future__ import annotations

import json
from typing import Any

from . import _openinference as oi


def _text_parts(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "content": text}] if text else []


def _arguments(raw: Any) -> Any:
    """Tool call arguments as an object when the provider sent a JSON string."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def _tool_call_parts(calls: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict):
            continue
        raw_fn = call.get("function")
        fn: dict[str, Any] = raw_fn if isinstance(raw_fn, dict) else call
        part: dict[str, Any] = {
            "type": "tool_call",
            "name": str(fn.get("name", "")),
            "arguments": _arguments(fn.get("arguments")),
        }
        if call.get("id"):
            part["id"] = call["id"]
        parts.append(part)
    return parts


def _system_text(system_prompt: Any) -> str:
    """Text of a system prompt given as a string or as a list of message dicts or strings."""
    if isinstance(system_prompt, list):
        texts = [s if isinstance(s, str) else oi.message_text(s) for s in system_prompt]
        return "\n".join(t for t in texts if t)
    return oi.content_text(system_prompt)


def input_messages(messages: Any, system_prompt: Any = None) -> list[dict[str, Any]]:
    """Request messages in semconv form; the system prompt leads when no system message exists."""
    out: list[dict[str, Any]] = []
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", ""))
        text = oi.message_text(m)
        if role == "tool" and m.get("tool_call_id"):
            parts: list[dict[str, Any]] = [
                {"type": "tool_call_response", "id": m["tool_call_id"], "response": text}
            ]
        else:
            parts = _text_parts(text) + _tool_call_parts(m.get("tool_calls"))
        out.append({"role": role, "parts": parts})
    if system_prompt and not any(m["role"] == "system" for m in out):
        text = _system_text(system_prompt)
        if text:
            out.insert(0, {"role": "system", "parts": _text_parts(text)})
    return out


def output_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The assistant reply of an LLM END event in semconv form, with ``finish_reason``."""
    parts = _text_parts(data.get("output_response") or "") + _tool_call_parts(
        data.get("output_tool_calls")
    )
    if not parts:
        return []
    msg: dict[str, Any] = {"role": "assistant", "parts": parts}
    if data.get("finish_reason"):
        msg["finish_reason"] = data["finish_reason"]
    return [msg]
