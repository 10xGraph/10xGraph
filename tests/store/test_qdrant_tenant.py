"""QdrantStore tenant columns: declaration, payload, indexes and filters.

Behaviour across backends (cross-tenant reads, legacy points) is in
``tests/contract/test_store_contract.py``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http.models import IsEmptyCondition, KeywordIndexParams

from tenxgraph.storage.columns import ColumnSpec
from tenxgraph.storage.store.qdrant_store import QdrantStore


TENANTS = {"org_id": "text", "project_id": "text"}


class _Embedding:
    dimension = 4

    async def aembed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


def _cfg(user="alice", org="A", project="P1", reach="own"):
    tenant = {"org_id": org, "project_id": project}
    return {"user": {"user_id": user, "tenant": tenant, "reach": reach, "claims": {}}}


@pytest.fixture
def client():
    mock = AsyncMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.retrieve.return_value = []
    mock.query_points.return_value = MagicMock(points=[])
    mock.scroll.return_value = ([], None)
    return mock


@pytest.fixture
def qdrant(client):
    with patch("qdrant_client.AsyncQdrantClient", return_value=client):
        return QdrantStore(embedding=_Embedding(), path="./test_data", tenant_columns=TENANTS)


def _conditions(flt) -> dict:
    return {c.key: c.match.value for c in flt.must}


def test_tenant_columns_declaration():
    with patch("qdrant_client.AsyncQdrantClient"):
        assert QdrantStore(embedding=_Embedding(), path="x").tenant_column_names == ()
        listed = QdrantStore(embedding=_Embedding(), path="x", tenant_columns=["org_id"])
        assert listed.tenant_column_names == ("org_id",)
        assert listed.columns.tenant[0].type == "text"
        for bad in ({"org_id": "jsonb"}, ["content"], ["user_id"], ["memory_key"], "org_id"):
            with pytest.raises(ValueError):
                QdrantStore(embedding=_Embedding(), path="x", tenant_columns=bad)


@pytest.mark.asyncio
async def test_write_stores_identity_tenant_values_not_metadata(qdrant, client):
    await qdrant.astore(_cfg(), "x", metadata={"org_id": "B", "project_id": "P9", "a": 1})
    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert payload["org_id"] == "A"
    assert payload["project_id"] == "P1"
    assert payload["a"] == 1


@pytest.mark.asyncio
async def test_write_without_tenant_value_raises(qdrant, client):
    with pytest.raises(ValueError):
        await qdrant.astore({"user": {"user_id": "alice"}}, "x")
    client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_tenant_columns_get_tenant_keyword_indexes(qdrant, client):
    await qdrant._ensure_collection_exists("c")
    schemas = {
        c.kwargs["field_name"]: c.kwargs["field_schema"]
        for c in client.create_payload_index.call_args_list
    }
    for name in TENANTS:
        assert schemas[name] == KeywordIndexParams(type="keyword", is_tenant=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "expected"),
    [
        ("own", {"user_id": "alice", "org_id": "A", "project_id": "P1"}),
        ("tenant", {"org_id": "A", "project_id": "P1"}),
        ("all", {}),
    ],
)
async def test_read_filter_follows_reach(qdrant, client, reach, expected):
    await qdrant.asearch(_cfg(reach=reach), "q")
    flt = client.query_points.call_args.kwargs["query_filter"]
    assert (_conditions(flt) if flt else {}) == expected
    await qdrant.aget_all(_cfg(reach=reach))
    flt = client.scroll.call_args.kwargs["scroll_filter"]
    assert (_conditions(flt) if flt else {}) == expected


@pytest.mark.asyncio
async def test_aget_all_orders_newest_first(qdrant, client):
    await qdrant.aget_all(_cfg(), filters={"color": "red"})
    kwargs = client.scroll.call_args.kwargs
    assert kwargs["order_by"].key == "timestamp"
    assert kwargs["order_by"].direction == "desc"
    assert _conditions(kwargs["scroll_filter"])["color"] == "red"


def test_missing_write_value_matches_only_points_without_it(qdrant):
    spec = ColumnSpec(name="org_id", type="text", required=False, kind="tenant")
    flt = qdrant._build_qdrant_filter(tenant=[(spec, None)])
    assert isinstance(flt.must[0], IsEmptyCondition)
    assert flt.must[0].is_empty.key == "org_id"


@pytest.mark.asyncio
async def test_local_mode_hides_tenant_values_and_keeps_key_on_replace(tmp_path):
    store = QdrantStore(embedding=_Embedding(), path=str(tmp_path / "q"), tenant_columns=TENANTS)
    try:
        mid = await store.astore(_cfg(), "x", category="food", metadata={"memory_key": "k"})
        got = await store.aget(_cfg(), mid)
        assert not set(TENANTS) & set(got.metadata)
        assert got.metadata["memory_key"] == "k"
        await store.aupdate(_cfg(), mid, "y", metadata={"b": 2}, replace_metadata=True)
        got = await store.aget(_cfg(), mid)
        assert got.metadata == {
            "b": 2,
            "memory_key": "k",
            "category": "food",
            "memory_type": "episodic",
        }
        point = (await store.client.retrieve(store.collection, ids=[mid]))[0]
        assert point.payload["org_id"] == "A"
    finally:
        await store.arelease()
