"""Streaming LLM spans get token usage from the stream's final message (G-47)."""

from types import SimpleNamespace

import pytest

from tenxgraph.runtime.adapters.llm.base_converter import BaseConverter
from tenxgraph.runtime.adapters.llm.model_response_converter import ModelResponseConverter


class _Conv(BaseConverter):
    async def convert_response(self, response):
        raise NotImplementedError

    def __init__(self, items, fail=False):
        self.items, self.fail = items, fail

    async def convert_streaming_response(self, config, node_name, response, meta=None):
        for i in self.items:
            yield i
        if self.fail:
            raise RuntimeError("dropped")


def _conv(items, fail=False):
    c = ModelResponseConverter(response=object(), converter=_Conv(items, fail))
    got = []
    c.on_stream_end = lambda final, err: got.append((final, err))
    return c, got


@pytest.mark.asyncio
async def test_hook_gets_last_message_with_usage():
    chunk = SimpleNamespace(usages=None)
    final = SimpleNamespace(usages=SimpleNamespace(prompt_tokens=5), metadata={})
    c, got = _conv([chunk, final])
    async for _ in c.stream({"k": 1}, "n"):
        pass
    assert got == [(final, None)]


@pytest.mark.asyncio
async def test_hook_fires_once_on_error_and_on_early_close():
    c, got = _conv([SimpleNamespace(usages=None)], fail=True)
    with pytest.raises(RuntimeError):
        async for _ in c.stream({"k": 1}, "n"):
            pass
    assert len(got) == 1 and isinstance(got[0][1], RuntimeError)

    c2, got2 = _conv([SimpleNamespace(usages=None), SimpleNamespace(usages=None)])
    gen = c2.stream({"k": 1}, "n")
    await gen.__anext__()
    await gen.aclose()
    assert len(got2) == 1
