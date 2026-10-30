"""G-86: failed requests in a finished OpenAI batch must appear in results()."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tenxgraph.core.llm.openai_batch import OpenAIBatch


def _jsonl(*entries) -> SimpleNamespace:
    return SimpleNamespace(text="\n".join(json.dumps(e) for e in entries))


@pytest.mark.asyncio
async def test_results_include_requests_from_the_error_file():
    ok = {
        "custom_id": "a",
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]},
        },
    }
    bad = {
        "custom_id": "b",
        "response": {"status_code": 400, "body": {"error": {"message": "nope"}}},
    }
    files = {"out": _jsonl(ok), "err": _jsonl(bad)}
    client = SimpleNamespace(
        batches=SimpleNamespace(
            retrieve=AsyncMock(
                return_value=SimpleNamespace(
                    status="completed", output_file_id="out", error_file_id="err"
                )
            )
        ),
        files=SimpleNamespace(content=AsyncMock(side_effect=lambda fid: files[fid])),
    )
    results = await OpenAIBatch(client=client, model="gpt-4o").results("batch_1")
    assert results["a"].ok
    assert results["b"].status == "errored"
