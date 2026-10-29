"""G-81: a malformed base64 payload must not fail the whole Google request."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tenxgraph.core.graph.agent import Agent


def _agent() -> Agent:
    with patch.object(Agent, "_create_client", return_value=MagicMock()):
        return Agent(model="gemini-2.0-flash", provider="google", reasoning_config=None)


def test_bad_data_url_becomes_failure_text():
    agent = _agent()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abcde"}}],
        }
    ]
    _, contents = agent._convert_to_google_format(messages)
    assert "Failed to load" in contents[0].parts[0].text


def test_bad_audio_and_document_base64_become_failure_text():
    agent = _agent()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": "abcde", "format": "wav"}},
                {"type": "document", "document": {"data": "abcde", "mime_type": "application/pdf"}},
            ],
        }
    ]
    _, contents = agent._convert_to_google_format(messages)
    assert all("Failed to load" in p.text for p in contents[0].parts)
