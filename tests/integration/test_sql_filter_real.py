"""Safe filters (``tenxgraph.storage.sql_filter``) against a REAL Postgres (+ pgvector, + Redis).

Every operator on every column type returns the rows a Python model of SQL semantics expects,
on ``PgVectorStore`` (all three search strategies, ``aget_all``, ``aforget_memory``) and on
``PgCheckpointer.alist_threads``. Filters combined with tenant isolation can only narrow it,
and injection attempts leave the tables intact.

    POSTGRES_DSN=... REDIS_URL=... pytest tests/integration/test_sql_filter_real.py --integration
"""

# SQL here is built only from identifiers the fixtures generate (uuid hex), never input.
# ruff: noqa: S608

import re
import uuid
from datetime import UTC, datetime

import pytest

from tenxgraph.storage.sql_filter import OPERATORS_BY_TYPE, FilterError
from tenxgraph.storage.store import PgVectorStore
from tenxgraph.utils import ThreadInfo
from tests.integration.test_pg_checkpointer_tenant_real import (  # noqa: F401 - fixtures
    admin_pool,
    make_cp,
    schema,
)
from tests.integration.test_pgvector_tenant_real import (  # noqa: F401 - fixtures
    WordEmbedding,
    pool,
    table,
)


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

TENANT = {"org_id": "text"}
EXTRA = {
    "plan": "text",
    "seats": "integer",
    "big": "bigint",
    "active": "boolean",
    "ref": "uuid",
    "renewed": "timestamptz",
}
U = [uuid.UUID(int=i + 1) for i in range(4)]


def _ts(year, month=1, day=1):
    return datetime(year, month, day, tzinfo=UTC)


# Rows owned by alice in org A. r3 has no extra values (NULL in every extra column).
ROWS = {
    "r0": {"plan": "pro", "seats": 5, "big": 10**12, "active": True, "ref": U[0],
           "renewed": _ts(2026)},
    "r1": {"plan": "free", "seats": 1, "big": -3, "active": False, "ref": U[1],
           "renewed": _ts(2025, 6)},
    "r2": {"plan": "Pro_Max", "seats": 50, "big": 0, "active": True, "ref": U[2],
           "renewed": _ts(2027)},
    "r3": {},
}  # fmt: skip
# Same user id and same values as r0, but in org B: never visible to alice@A.
OTHER_ORG = ("x0", dict(ROWS["r0"]))


def _like(pattern, text, insensitive):
    regex = ""
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            regex += re.escape(pattern[i + 1])
            i += 2
            continue
        regex += ".*" if ch == "%" else "." if ch == "_" else re.escape(ch)
        i += 1
    return re.fullmatch(regex, text, re.DOTALL | (re.IGNORECASE if insensitive else 0))


def _matches(value, op, arg):  # noqa: PLR0911
    """SQL three-valued logic, where NULL comparisons are not true."""
    if op == "is_null":
        return (value is None) == arg
    if op == "ne":
        return value != arg
    if op == "nin":
        return value is None or value not in arg
    if value is None:
        return False
    if op == "eq":
        return value == arg
    if op == "in":
        return value in arg
    if op in ("like", "ilike"):
        return bool(_like(arg, value, op == "ilike"))
    return {"gt": value > arg, "gte": value >= arg, "lt": value < arg, "lte": value <= arg}[op]


# (column, op, value sent in the filter, the same value in Python for the model)
def _cases():
    probes = {
        "plan": ("pro", ["pro", "free"], "pro%", "PRO%"),
        "seats": (5, [1, 50], None, None),
        "big": (0, [0, 10**12], None, None),
        "active": (True, [False], None, None),
        "ref": (U[1], [U[0], U[2]], None, None),
        "renewed": (_ts(2026), [_ts(2025, 6)], None, None),
    }
    cases = []
    for col, type_name in EXTRA.items():
        scalar, items, like, ilike = probes[col]
        for op in OPERATORS_BY_TYPE[type_name]:
            if op == "is_null":
                cases += [(col, op, True, True), (col, op, False, False)]
            elif op in ("in", "nin"):
                sent = [str(v) if col == "ref" else v for v in items]
                cases.append((col, op, sent, items))
            elif op == "like":
                cases += [(col, op, like, like), (col, op, "%\\_%", "%\\_%")]
            elif op == "ilike":
                cases.append((col, op, ilike, ilike))
            elif col == "renewed":
                cases.append((col, op, scalar.isoformat(), scalar))
            elif col == "ref":
                cases.append((col, op, str(scalar), scalar))
            else:
                cases.append((col, op, scalar, scalar))
    return cases


CASES = _cases()
CASE_IDS = [f"{c}-{o}-{i}" for i, (c, o, _s, _p) in enumerate(CASES)]


def _expected(col, op, py):
    return {name for name, values in ROWS.items() if _matches(values.get(col), op, py)}


def _identity(name_values, org="A", user="alice"):
    return {"user_id": user, "tenant": {"org_id": org}, "claims": dict(name_values)}


# ---------------------------------------------------------------------------
# PgVectorStore
# ---------------------------------------------------------------------------


async def _filled_store(pool, table):  # noqa: F811
    store = PgVectorStore(
        WordEmbedding(), pool=pool, table=table, tenant_columns=TENANT, extra_columns=EXTRA
    )
    await store.asetup()
    for name, values in ROWS.items():
        cfg = {"user_id": "alice", "user": _identity(values)}
        await store.astore(cfg, f"note {name}", metadata={"label": name})
    other, values = OTHER_ORG
    await store.astore(
        {"user_id": "alice", "user": _identity(values, org="B")},
        f"note {other}",
        metadata={"label": other},
    )
    return store


ALICE_A = {"user_id": "alice", "user": {"user_id": "alice", "tenant": {"org_id": "A"}}}


def _labels(results):
    return {r.metadata["label"] for r in results}


class TestStoreReal:
    async def test_every_operator_aget_all(self, pool, table):  # noqa: F811
        store = await _filled_store(pool, table)
        for (col, op, sent, py), case_id in zip(CASES, CASE_IDS, strict=True):
            got = _labels(await store.aget_all(ALICE_A, limit=50, filters={col: {op: sent}}))
            assert got == _expected(col, op, py), case_id

    @pytest.mark.parametrize("strategy", ["similarity", "text", "hybrid"])
    async def test_every_operator_search(self, pool, table, strategy):  # noqa: F811
        store = await _filled_store(pool, table)
        for (col, op, sent, py), case_id in zip(CASES, CASE_IDS, strict=True):
            got = _labels(
                await store.asearch(
                    ALICE_A,
                    "note",
                    limit=50,
                    retrieval_strategy=strategy,
                    filters={col: {op: sent}},
                )
            )
            assert got == _expected(col, op, py), (strategy, case_id)

    async def test_builtin_columns_and_combinations(self, pool, table):  # noqa: F811
        store = await _filled_store(pool, table)
        got = await store.aget_all(
            ALICE_A,
            filters={
                "created_at": {"lte": datetime.now(UTC).isoformat(), "is_null": False},
                "memory_type": {"in": ["episodic"]},
                "seats": {"gte": 2},
                "label": "r2",
            },
        )
        assert _labels(got) == {"r2"}

    async def test_filters_never_widen_isolation(self, pool, table):  # noqa: F811
        store = await _filled_store(pool, table)
        wide = [
            {"plan": {"nin": ["nothing"]}},
            {"plan": {"ne": "nothing"}},
            {"plan": {"like": "%"}},
            {"created_at": {"is_null": False}},
        ]
        for filters in wide:
            got = _labels(await store.aget_all(ALICE_A, limit=50, filters=filters))
            assert "x0" not in got
        bob = {"user_id": "bob", "user": {"user_id": "bob", "tenant": {"org_id": "A"}}}
        assert await store.aget_all(bob, filters={"plan": {"nin": ["x"]}}) == []
        with pytest.raises(FilterError, match="isolation column"):
            await store.aget_all(ALICE_A, filters={"org_id": "B"})
        with pytest.raises(FilterError, match="isolation column"):
            await store.aget_all(ALICE_A, filters={"user_id": {"ne": "alice"}})

    async def test_forget_with_operators(self, pool, table):  # noqa: F811
        store = await _filled_store(pool, table)
        await store.aforget_memory(ALICE_A, filters={"seats": {"gt": 2}})
        assert _labels(await store.aget_all(ALICE_A)) == {"r1", "r3"}
        # The org B row with the same values survives.
        b = {"user_id": "alice", "user": {"user_id": "alice", "tenant": {"org_id": "B"}}}
        assert _labels(await store.aget_all(b)) == {"x0"}

    async def test_injection_leaves_table_intact(self, pool, table):  # noqa: F811
        store = await _filled_store(pool, table)
        payloads = [
            f'x\'; DROP TABLE "{table}"; --',
            "') OR ('1'='1",
            "1; DELETE FROM pg_class; --",
            "%' OR '1'='1",
        ]
        for payload in payloads:
            for filters in (
                {"plan": payload},
                {"plan": {"like": payload, "ne": payload}},
                {"plan": {"in": [payload]}},
                {"category": {"ilike": payload}},
                {payload: payload},  # metadata key: travels inside a JSON parameter
            ):
                assert await store.aget_all(ALICE_A, filters=filters) == []
                await store.aforget_memory(ALICE_A, filters=filters)
            with pytest.raises(FilterError):
                await store.aget_all(ALICE_A, filters={"plan": {payload: 1}})
            with pytest.raises(FilterError):
                await store.aget_all(ALICE_A, filters={"seats": payload})
        count = await pool.fetchval(f'SELECT count(*) FROM "public"."{table}"')
        assert count == len(ROWS) + 1


# ---------------------------------------------------------------------------
# PgCheckpointer.alist_threads
# ---------------------------------------------------------------------------


async def _filled_cp(make_cp):  # noqa: F811
    cp = await make_cp(tenant_columns=TENANT, extra_columns=EXTRA)
    await cp.asetup()
    ids = {}
    for name, values in [*ROWS.items(), OTHER_ORG]:
        org = "B" if name == OTHER_ORG[0] else "A"
        thread = f"t-{name}-{uuid.uuid4().hex[:6]}"
        ids[thread] = name
        cfg = {"thread_id": thread, "user_id": "alice", "user": _identity(values, org=org)}
        await cp.aput_thread(cfg, ThreadInfo(thread_id=thread, thread_name=f"Thread {name}"))
    return cp, ids


def _cfg_a():
    return {"user_id": "alice", "user": {"user_id": "alice", "tenant": {"org_id": "A"}}}


def _names(threads, ids):
    return {ids[t.thread_id] for t in threads}


class TestCheckpointerReal:
    async def test_every_operator(self, make_cp):  # noqa: F811
        cp, ids = await _filled_cp(make_cp)
        for (col, op, sent, py), case_id in zip(CASES, CASE_IDS, strict=True):
            got = _names(await cp.alist_threads(_cfg_a(), filters={col: {op: sent}}), ids)
            assert got == _expected(col, op, py), case_id

    async def test_builtins_search_and_pagination(self, make_cp):  # noqa: F811
        cp, ids = await _filled_cp(make_cp)
        got = await cp.alist_threads(
            _cfg_a(),
            search="thread",
            filters={
                "thread_name": {"ilike": "thread r%", "nin": ["Thread r1"]},
                "created_at": {"lte": datetime.now(UTC)},
                "updated_at": {"is_null": False},
            },
        )
        assert _names(got, ids) == {"r0", "r2", "r3"}
        page = await cp.alist_threads(_cfg_a(), filters={"active": True}, limit=1, offset=1)
        assert len(page) == 1 and _names(page, ids) <= {"r0", "r2"}

    async def test_filters_never_widen_isolation(self, make_cp):  # noqa: F811
        cp, ids = await _filled_cp(make_cp)
        for filters in ({"plan": {"nin": ["x"]}}, {"thread_name": {"like": "%"}}, {}):
            assert "x0" not in _names(await cp.alist_threads(_cfg_a(), filters=filters), ids)
        bob = {"user_id": "bob", "user": {"user_id": "bob", "tenant": {"org_id": "A"}}}
        assert await cp.alist_threads(bob, filters={"plan": {"ne": "x"}}) == []
        with pytest.raises(FilterError, match="isolation column"):
            await cp.alist_threads(_cfg_a(), filters={"org_id": "B"})

    async def test_injection_leaves_tables_intact(self, make_cp, admin_pool, schema):  # noqa: F811
        cp, _ids = await _filled_cp(make_cp)
        payloads = ['x\'; DROP TABLE "threads"; --', "') OR ('1'='1", "\\'; SELECT 1; --"]
        for payload in payloads:
            for filters in (
                {"plan": payload},
                {"thread_name": {"like": payload, "ne": payload}},
                {"plan": {"nin": [payload]}, "thread_name": payload},
            ):
                assert await cp.alist_threads(_cfg_a(), filters=filters) == []
            with pytest.raises(FilterError):
                await cp.alist_threads(_cfg_a(), filters={payload: "x"})
            with pytest.raises(FilterError):
                await cp.alist_threads(_cfg_a(), filters={"renewed": {"gt": payload}})
        count = await admin_pool.fetchval(f'SELECT count(*) FROM "{schema}"."threads"')
        assert count == len(ROWS) + 1
