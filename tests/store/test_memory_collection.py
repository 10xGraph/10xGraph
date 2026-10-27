"""Collection precedence for memory scopes and QdrantStore defaults (security Q6)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.prebuilt.tools.memory import _memory_scope_config
from tenxgraph.storage.store import AgentMemoryConfig, MemoryConfig, UserMemoryConfig
from tenxgraph.storage.store.qdrant_store import DEFAULT_COLLECTION, QdrantStore


def _scope_cfg(memory, scope, run_config):
    scope_config = memory.user_memory if scope == "user" else memory.agent_memory
    return _memory_scope_config(run_config, memory, scope_config, scope=scope)


def _agent_memory(**kw):
    return AgentMemoryConfig(enabled=True, agent_id="a1", **kw)


class TestScopeCollection:
    def test_scope_collection_beats_run_config(self):
        memory = MemoryConfig(user_memory=UserMemoryConfig(collection="dev"))
        cfg = _scope_cfg(memory, "user", {"user_id": "u", "collection": "attacker"})
        assert cfg["collection"] == "dev"

    def test_scope_collection_beats_developer_config(self):
        memory = MemoryConfig(
            config={"collection": "mc"},
            user_memory=UserMemoryConfig(collection="dev", config={"collection": "sc"}),
        )
        assert _scope_cfg(memory, "user", {})["collection"] == "dev"

    def test_run_config_collection_used_when_scope_has_none(self):
        memory = MemoryConfig(user_memory=UserMemoryConfig())
        cfg = _scope_cfg(memory, "user", {"collection": "from_run"})
        assert cfg["collection"] == "from_run"

    def test_no_collection_left_to_store_default(self):
        memory = MemoryConfig(user_memory=UserMemoryConfig())
        assert "collection" not in _scope_cfg(memory, "user", {})

    def test_callable_gets_run_config_and_beats_run_collection(self):
        memory = MemoryConfig(
            user_memory=UserMemoryConfig(
                collection=lambda cfg: f"company_{cfg['user']['tenant_id']}"
            )
        )
        run = {"user": {"tenant_id": "acme"}, "collection": "attacker"}
        assert _scope_cfg(memory, "user", run)["collection"] == "company_acme"

    def test_agent_and_user_scopes_differ(self):
        memory = MemoryConfig(
            user_memory=UserMemoryConfig(collection="users"),
            agent_memory=_agent_memory(collection="shared"),
        )
        assert _scope_cfg(memory, "user", {})["collection"] == "users"
        assert _scope_cfg(memory, "agent", {})["collection"] == "shared"

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_string_rejected(self, bad):
        with pytest.raises(ValueError, match="collection"):
            UserMemoryConfig(collection=bad)

    def test_string_is_stripped(self):
        assert UserMemoryConfig(collection=" docs ").collection == "docs"

    @pytest.mark.parametrize("bad", ["", "  ", None, 5])
    def test_callable_bad_result_raises(self, bad):
        memory = MemoryConfig(user_memory=UserMemoryConfig(collection=lambda cfg: bad))
        with pytest.raises(ValueError, match="non-empty string"):
            _scope_cfg(memory, "user", {})


class _Emb:
    dimension = 4

    async def aembed(self, text):
        return [0.0] * 4

    async def aembed_batch(self, texts):
        return [[0.0] * 4 for _ in texts]

    def embed(self, text):
        return [0.0] * 4

    def embed_batch(self, texts):
        return [[0.0] * 4 for _ in texts]

    @property
    def dim(self):
        return 4


def _store(collection=None):
    client = AsyncMock()
    client.get_collections.return_value = MagicMock(collections=[])
    with patch("qdrant_client.AsyncQdrantClient", return_value=client):
        store = QdrantStore(embedding=_Emb(), path="./x", collection=collection)
    return store, client


class TestStoreDefault:
    def test_default_constant(self):
        store, _ = _store()
        assert store._extract_config_values({"user_id": "u"})[2] == DEFAULT_COLLECTION

    def test_string_default(self):
        store, _ = _store("mine")
        assert store._extract_config_values({})[2] == "mine"

    def test_config_collection_beats_default(self):
        store, _ = _store(lambda cfg: "fn")
        assert store._extract_config_values({"collection": "cfg"})[2] == "cfg"

    def test_callable_default_gets_config(self):
        store, _ = _store(lambda cfg: f"co_{cfg['user']['tenant_id']}")
        assert store._extract_config_values({"user": {"tenant_id": "t1"}})[2] == "co_t1"

    @pytest.mark.parametrize("bad", ["", None, 3])
    def test_callable_default_bad_result_raises(self, bad):
        store, _ = _store(lambda cfg: bad)
        with pytest.raises(ValueError, match="collection"):
            store._extract_config_values({})

    @pytest.mark.asyncio
    async def test_asetup_with_callable_creates_nothing(self):
        store, client = _store(lambda cfg: "x")
        assert await store.asetup() is True
        client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_asetup_with_string_creates_default(self):
        store, client = _store("named")
        await store.asetup()
        assert client.create_collection.call_args.kwargs["collection_name"] == "named"
