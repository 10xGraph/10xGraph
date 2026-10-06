"""Legacy ``agentflow://`` media URIs must still be readable; new ones use ``graph://``."""

import pytest

from tenxgraph.core.state import MediaRef
from tenxgraph.storage.media.resolver import MediaRefResolver
from tenxgraph.utils.media_scheme import is_internal_media_url, strip_media_scheme


def test_scheme_helpers():
    assert is_internal_media_url("graph://media/k")
    assert is_internal_media_url("agentflow://media/k")
    assert not is_internal_media_url("https://x/k")
    assert not is_internal_media_url(None)
    assert strip_media_scheme("graph://media/k") == "k"
    assert strip_media_scheme("agentflow://media/k") == "k"


class _Store:
    async def retrieve(self, key):
        return key.encode(), "image/png"


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["agentflow://media/abc", "graph://media/abc"])
async def test_resolver_reads_old_and_new_uri(url):
    resolver = MediaRefResolver(media_store=_Store())
    data, mime = await resolver._retrieve(url)
    assert data == b"abc"
    assert mime == "image/png"
