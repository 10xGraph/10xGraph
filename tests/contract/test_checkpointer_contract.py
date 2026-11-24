"""One behaviour contract, run against every checkpointer.

Backends: InMemoryCheckpointer, SqliteCheckpointer, PgCheckpointer (integration). Where the
docs or the ``BaseCheckpointer`` docstrings define a behaviour the test asserts it. Where
backends disagree, the failing combination is listed in ``conftest.KNOWN_GAPS`` as a strict
xfail with the gap id from ``cleanup-review/CORE_B.md`` section 3.
"""

import asyncio
from datetime import timedelta

import pytest

from tenxgraph.core.exceptions import MessageNotFoundError, ThreadNotFoundError
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer.base_checkpointer import MAX_LIST_LIMIT
from tenxgraph.utils import ThreadInfo
from tenxgraph.utils.validators import ValidationError

from .conftest import cfg


pytestmark = pytest.mark.asyncio


def _msg(i, text=None, role="user"):
    return Message.text_message(text or f"msg {i}", role=role, message_id=f"m{i}")


def _state(*texts):
    state = AgentState()
    state.context = [_msg(i, t) for i, t in enumerate(texts)]
    return state


def _ids(messages):
    return [m.message_id for m in messages]


# ------------------------------------------------------------------ thread CRUD


async def test_thread_roundtrip(cp):
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1", thread_name="first", metadata={"k": 1}))
    got = await cp.aget_thread(c)
    assert got is not None
    assert got.thread_id == "t1"
    assert got.thread_name == "first"
    assert got.user_id == "alice"
    assert (got.metadata or {}).get("k") == 1


async def test_get_missing_thread_is_none(cp):
    assert await cp.aget_thread(cfg("nope")) is None


async def test_thread_update_changes_name(cp):
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1", thread_name="old"))
    await cp.aput_thread(c, ThreadInfo(thread_id="t1", thread_name="new"))
    assert (await cp.aget_thread(c)).thread_name == "new"


async def test_clean_thread_returns_one_then_zero(cp):
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1"))
    assert await cp.aclean_thread(c) == 1
    assert await cp.aget_thread(c) is None
    assert await cp.aclean_thread(c) == 0


async def test_list_threads_scoped_to_caller(cp):
    for t in ("a", "b"):
        await cp.aput_thread(cfg(t), ThreadInfo(thread_id=t, thread_name=f"name-{t}"))
    await cp.aput_thread(cfg("c", user="bob"), ThreadInfo(thread_id="c"))
    got = {t.thread_id for t in await cp.alist_threads(cfg("a"))}
    assert got == {"a", "b"}


async def test_list_threads_order_is_most_recently_updated_first(cp):
    """G11: S and P order by updated_at DESC; M keeps insertion order."""
    await cp.aput_thread(cfg("a"), ThreadInfo(thread_id="a"))
    await asyncio.sleep(0.01)
    await cp.aput_thread(cfg("b"), ThreadInfo(thread_id="b"))
    await asyncio.sleep(0.01)
    await cp.aput_thread(cfg("a"), ThreadInfo(thread_id="a", thread_name="touched"))
    assert [t.thread_id for t in await cp.alist_threads(cfg("a"))] == ["a", "b"]


async def test_list_threads_search_matches_thread_name(cp):
    await cp.aput_thread(cfg("a"), ThreadInfo(thread_id="a", thread_name="invoices"))
    await cp.aput_thread(cfg("b"), ThreadInfo(thread_id="b", thread_name="travel"))
    assert [t.thread_id for t in await cp.alist_threads(cfg("a"), search="invoice")] == ["a"]


async def test_list_threads_search_ignores_other_fields(cp):
    """Only the name is searched, not the owner or any other field."""
    await cp.aput_thread(cfg("a"), ThreadInfo(thread_id="a", thread_name="x", user_id="alice"))
    assert await cp.alist_threads(cfg("a"), search="alice") == []


async def test_thread_listing_and_search_are_isolated_by_user(cp):
    await cp.aput_thread(cfg("a"), ThreadInfo(thread_id="a", thread_name="secret plan"))
    bob = cfg("x", user="bob")
    assert await cp.alist_threads(bob) == []
    assert await cp.alist_threads(bob, search="secret") == []
    assert [t.thread_id for t in await cp.alist_threads(cfg("x"), search="secret")] == ["a"]


async def test_thread_listing_and_search_are_isolated_by_tenant(backend, cp):
    if backend != "pg":
        pytest.skip("tenant columns need PgCheckpointer")
    await cp.aput_thread(cfg("a", org="A"), ThreadInfo(thread_id="a", thread_name="secret plan"))
    # The same user id in another tenant, and another user in the other tenant.
    for other in (cfg("x", user="alice", org="B"), cfg("x", user="bob", org="B")):
        assert await cp.alist_threads(other) == []
        assert await cp.alist_threads(other, search="secret") == []
    # Reach tenant widens to the tenant's users, never to another tenant.
    peer = cfg("x", user="bob", org="A", reach="tenant")
    assert [t.thread_id for t in await cp.alist_threads(peer, search="secret")] == ["a"]
    outsider = cfg("x", user="bob", org="B", reach="tenant")
    assert await cp.alist_threads(outsider, search="secret") == []


async def test_list_threads_pagination(cp):
    for t in "abcde":
        await cp.aput_thread(cfg(t), ThreadInfo(thread_id=t, thread_name=f"n-{t}"))
        await asyncio.sleep(0.005)
    everything = [t.thread_id for t in await cp.alist_threads(cfg("a"))]
    assert len(everything) == 5
    assert [t.thread_id for t in await cp.alist_threads(cfg("a"), offset=1, limit=2)] == (
        everything[1:3]
    )
    assert len(await cp.alist_threads(cfg("a"), search="n-", offset=1, limit=2)) == 2
    assert len(await cp.alist_threads(cfg("a"), search="n-", offset=4, limit=2)) == 1


async def test_negative_page_values_rejected(cp):
    for kw in ({"limit": -1}, {"offset": -1}):
        with pytest.raises(ValueError):
            await cp.alist_threads(cfg("t"), **kw)
        with pytest.raises(ValueError):
            await cp.alist_messages(cfg("t"), **kw)


async def test_limit_is_capped(cp):
    await cp.aput_messages(cfg("t"), [_msg(i) for i in range(3)])
    assert len(await cp.alist_messages(cfg("t"), limit=MAX_LIST_LIMIT + 5)) == 3


async def test_put_thread_rejects_mismatched_thread_id(cp):
    """A thread_info.thread_id that differs from the config id is refused."""
    with pytest.raises(ValueError):
        await cp.aput_thread(cfg("t1"), ThreadInfo(thread_id="other"))


async def test_put_thread_does_not_mutate_input(cp):
    """The caller's metadata is stored as a copy, never changed."""
    info = ThreadInfo(thread_id="t1", metadata={"k": 1})
    await cp.aput_thread(cfg("t1"), info)
    assert info.metadata == {"k": 1}


async def test_thread_roundtrip_equals_input(cp):
    """G14: aget_thread returns the same ThreadInfo that was put (modulo owner and time)."""
    info = ThreadInfo(thread_id="t1", thread_name="n", metadata={"k": "v"})
    await cp.aput_thread(cfg("t1"), info)
    got = await cp.aget_thread(cfg("t1"))
    assert got.metadata == {"k": "v"}
    assert got.thread_name == "n"


# ------------------------------------------------------------------ is_new_thread


async def test_is_new_thread_true_once(cp):
    c = cfg("t1")
    assert await cp.aput_thread(c, ThreadInfo(thread_id="t1")) is True
    assert await cp.aput_thread(c, ThreadInfo(thread_id="t1")) is False
    assert await cp.aput_thread(c, ThreadInfo(thread_id="t1", thread_name="x")) is False


async def test_is_new_thread_true_again_after_clean(cp):
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1"))
    await cp.aclean_thread(c)
    assert await cp.aput_thread(c, ThreadInfo(thread_id="t1")) is True


async def test_is_new_thread_per_thread(cp):
    assert await cp.aput_thread(cfg("a"), ThreadInfo(thread_id="a")) is True
    assert await cp.aput_thread(cfg("b"), ThreadInfo(thread_id="b")) is True


# ------------------------------------------------------------------ messages


async def test_message_put_get(cp):
    c = cfg("t1")
    await cp.aput_messages(c, [_msg(1, "hello")])
    got = await cp.aget_message(c, "m1")
    assert got.text() == "hello"
    assert got.role == "user"


async def test_get_missing_message_raises_index_error(cp):
    """A missing id raises MessageNotFoundError, which is an IndexError and a ValueError."""
    with pytest.raises(MessageNotFoundError, match="Message not found") as info:
        await cp.aget_message(cfg("t1"), "missing")
    assert isinstance(info.value, IndexError)
    assert isinstance(info.value, ValueError)


async def test_list_messages_insertion_order(cp):
    """G7: 50 messages in one put, then the same list put again (upsert)."""
    c = cfg("t1")
    msgs = [_msg(i) for i in range(50)]
    await cp.aput_messages(c, msgs)
    expected = [f"m{i}" for i in range(50)]
    assert _ids(await cp.alist_messages(c)) == expected
    await cp.aput_messages(c, msgs)
    assert _ids(await cp.alist_messages(c)) == expected


async def test_list_messages_order_across_puts(cp):
    c = cfg("t1")
    await cp.aput_messages(c, [_msg(0), _msg(1)])
    await cp.aput_messages(c, [_msg(2)])
    assert _ids(await cp.alist_messages(c)) == ["m0", "m1", "m2"]


async def test_list_messages_pagination(cp):
    c = cfg("t1")
    await cp.aput_messages(c, [_msg(i) for i in range(10)])
    assert _ids(await cp.alist_messages(c, offset=2, limit=3)) == ["m2", "m3", "m4"]
    assert _ids(await cp.alist_messages(c, offset=8)) == ["m8", "m9"]


async def test_list_messages_search_with_pagination(cp):
    """G22: S pages in Python when ``search`` is set."""
    c = cfg("t1")
    await cp.aput_messages(c, [_msg(i, f"apple {i}") for i in range(6)] + [_msg(9, "pear")])
    got = await cp.alist_messages(c, search="apple", offset=1, limit=3)
    assert _ids(got) == ["m1", "m2", "m3"]


async def test_message_search_matches_text_only(cp):
    """Only the message text is searched; block keys such as "text" and "type" never match."""
    c = cfg("t1")
    await cp.aput_messages(c, [_msg(1, "hello world")])
    assert await cp.alist_messages(c, search="text") == []
    assert await cp.alist_messages(c, search="type") == []
    assert _ids(await cp.alist_messages(c, search="HELLO")) == ["m1"]


async def test_message_search_treats_wildcards_literally(cp):
    c = cfg("t1")
    await cp.aput_messages(c, [_msg(1, "100% sure"), _msg(2, "a_b"), _msg(3, "axb")])
    assert _ids(await cp.alist_messages(c, search="100%")) == ["m1"]
    assert _ids(await cp.alist_messages(c, search="a_b")) == ["m2"]
    assert _ids(await cp.alist_messages(c, search="%")) == ["m1"]


async def test_delete_message(cp):
    c = cfg("t1")
    await cp.aput_messages(c, [_msg(1), _msg(2)])
    assert await cp.adelete_message(c, "m1") == 1
    assert _ids(await cp.alist_messages(c)) == ["m2"]
    assert await cp.adelete_message(c, "m1") == 0
    assert await cp.adelete_message(c, "never") == 0


async def test_delete_message_removes_it_from_state_and_cache(cp):
    """G19: adelete_message strips the message from the state and the state cache."""
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1"))
    state = _state("keep", "drop")
    await cp.aput_state(c, state)
    await cp.aput_state_cache(c, state)
    await cp.aput_messages(c, state.context)
    assert await cp.adelete_message(c, "m1") == 1
    for got in (await cp.aget_state(c), await cp.aget_state_cache(c)):
        assert got is not None
        assert "m1" not in _ids(got.context)
        assert "m0" in _ids(got.context)


async def test_update_message_metadata_merges(cp):
    c = cfg("t1")
    m = _msg(1)
    m.metadata = {"a": 1, "b": 2}
    await cp.aput_messages(c, [m])
    updated = await cp.aupdate_message_metadata(c, "m1", {"b": 3, "c": 4})
    assert updated is not None
    assert updated.metadata == {"a": 1, "b": 3, "c": 4}
    assert (await cp.aget_message(c, "m1")).metadata == {"a": 1, "b": 3, "c": 4}


async def test_update_message_metadata_missing_is_none(cp):
    assert await cp.aupdate_message_metadata(cfg("t1"), "missing", {"a": 1}) is None


async def test_put_messages_metadata_argument(cp):
    """``aput_messages(metadata=...)`` is merged into each message; the message's keys win."""
    c = cfg("t1")
    own = _msg(2)
    own.metadata = {"source": "own"}
    await cp.aput_messages(c, [_msg(1), own], metadata={"source": "import", "batch": 7})
    assert (await cp.aget_message(c, "m1")).metadata == {"source": "import", "batch": 7}
    assert (await cp.aget_message(c, "m2")).metadata == {"source": "own", "batch": 7}
    listed = await cp.alist_messages(c)
    assert listed[0].metadata == {"source": "import", "batch": 7}
    assert own.metadata == {"source": "own"}


async def test_message_roundtrip_keeps_full_model(cp):
    """parsed_content and an explicit timestamp survive a round trip.

    ``raw`` (a large provider payload) and ``delta`` (a flag of partial messages) are not part
    of the contract: Postgres does not keep them.
    """
    c = cfg("t1")
    m = _msg(1, "full")
    m.parsed_content = {"answer": 42}
    m.timestamp = 1_700_000_000.5
    await cp.aput_messages(c, [m])
    got = (await cp.alist_messages(c))[0]
    assert got.parsed_content == {"answer": 42}
    assert got.metadata == {}
    assert got.timestamp == pytest.approx(1_700_000_000.5)


async def test_same_message_id_in_two_threads(cp):
    a, b = cfg("a", user="u1"), cfg("b", user="u2")
    await cp.aput_messages(a, [_msg(1, "from a")])
    await cp.aput_messages(b, [_msg(1, "from b")])
    assert (await cp.aget_message(a, "m1")).text() == "from a"
    assert (await cp.aget_message(b, "m1")).text() == "from b"
    await cp.adelete_message(a, "m1")
    assert (await cp.aget_message(b, "m1")).text() == "from b"


# ------------------------------------------------------------------ ids


async def test_int_and_str_thread_ids_are_the_same_thread_text(cp):
    """Direct callers pass int ids; ``"7"`` and ``7`` must address the same data."""
    await cp.aput_messages(cfg(7), [_msg(1, "seven")])
    assert _ids(await cp.alist_messages(cfg("7"))) == ["m1"]


async def test_int_thread_id_roundtrip(cp):
    await cp.aput_thread(cfg(11), ThreadInfo(thread_id=11, thread_name="int"))
    got = await cp.aget_thread(cfg(11))
    assert got is not None
    assert str(got.thread_id) == "11"


async def test_int_message_id_found_by_str_and_int(cp):
    """G5: store id 5 (int); get "5" and 5. M compares with ==, so "5" misses."""
    c = cfg("t1")
    m = Message.text_message("five", role="user")
    m.message_id = 5
    await cp.aput_messages(c, [m])
    assert (await cp.aget_message(c, "5")).text() == "five"
    assert (await cp.aget_message(c, 5)).text() == "five"


async def test_str_message_id_found_by_int(cp):
    """G5, other direction: store "5", get 5."""
    c = cfg("t1")
    await cp.aput_messages(c, [Message.text_message("five", message_id="5")])
    assert (await cp.aget_message(c, 5)).text() == "five"


async def test_int_message_ids_dedupe_with_str(cp):
    """G5: re-putting id 5 as "5" updates the same row, not a second one."""
    c = cfg("t1")
    a = Message.text_message("one", role="user")
    a.message_id = 5
    b = Message.text_message("two", role="user", message_id="5")
    await cp.aput_messages(c, [a])
    await cp.aput_messages(c, [b])
    got = await cp.alist_messages(c)
    assert len(got) == 1
    assert got[0].text() == "two"


async def test_int_message_id_delete_and_update(cp):
    c = cfg("t1")
    m = Message.text_message("five", role="user")
    m.message_id = 5
    await cp.aput_messages(c, [m])
    assert (await cp.aupdate_message_metadata(c, "5", {"k": 1})) is not None
    assert await cp.adelete_message(c, "5") == 1


async def test_pg_integer_id_tables_accept_string_ids(backend, make_cp):
    """G23: id_type="bigint" tables take the string ids the API passes, in every message method."""
    if backend != "pg":
        pytest.skip("Postgres only")
    cp = await make_cp(id_type="bigint")
    c = cfg("42")
    await cp.aput_thread(c, ThreadInfo(thread_id="42", thread_name="n"))
    m = Message.text_message("hi", role="user")
    m.message_id = 9
    await cp.aput_messages(c, [m])
    assert [x.text() for x in await cp.alist_messages(c)] == ["hi"]
    assert (await cp.aget_message(c, "9")).text() == "hi"
    assert await cp.aupdate_message_metadata(c, "9", {"k": 1}) is not None
    assert await cp.adelete_message(c, "9") == 1
    assert await cp.aget_thread_owner("42") == "alice"
    assert await cp.aclean_thread(c) == 1


# ------------------------------------------------------------------ tool ledger


async def test_ledger_roundtrip(cp):
    c = cfg("t1")
    assert await cp.aget_tool_result(c, "call-1") is None
    await cp.aput_tool_result(c, "call-1", {"output": "charged", "is_error": False})
    assert await cp.aget_tool_result(c, "call-1") == {"output": "charged", "is_error": False}
    assert await cp.aget_tool_result(c, "call-2") is None


async def test_ledger_first_result_wins(cp):
    """G1: P docstring and S: first write wins; M overwrites."""
    c = cfg("t1")
    await cp.aput_tool_result(c, "call-1", {"output": "first"})
    await cp.aput_tool_result(c, "call-1", {"output": "second"})
    assert await cp.aget_tool_result(c, "call-1") == {"output": "first"}


async def test_ledger_is_per_thread(cp):
    await cp.aput_tool_result(cfg("a"), "call-1", {"output": "x"})
    assert await cp.aget_tool_result(cfg("b"), "call-1") is None


async def test_ledger_cleared_by_clean_thread(cp):
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1"))
    await cp.aput_tool_result(c, "m1:call-1", {"output": "x"})
    await cp.aclean_thread(c)
    assert await cp.aget_tool_result(c, "m1:call-1") is None


async def test_clear_tool_results_by_origin_message(cp):
    """Keys are ``<origin_message_id>:<tool_call_id>``; an empty list deletes nothing."""
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1"))
    await cp.aput_tool_result(c, "m1:a", {"output": 1})
    await cp.aput_tool_result(c, "m2:b", {"output": 2})
    assert await cp.aclear_tool_results(c, []) == 0
    assert await cp.aclear_tool_results(c, ["m1"]) == 1
    assert await cp.aget_tool_result(c, "m1:a") is None
    assert await cp.aget_tool_result(c, "m2:b") == {"output": 2}
    assert await cp.aclear_tool_results(c) == 1


async def test_prune_tool_results_keeps_recent_rows(cp):
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1"))
    await cp.aput_tool_result(c, "m1:a", {"output": 1})
    assert await cp.aprune_tool_results(timedelta(days=1)) == 0
    assert await cp.aget_tool_result(c, "m1:a") == {"output": 1}


# ------------------------------------------------------------------ delete leaves nothing


async def _fill(cp, c):
    await cp.aput_thread(c, ThreadInfo(thread_id=c["thread_id"], thread_name="n"))
    state = _state("a", "b")
    await cp.aput_state(c, state)
    await cp.aput_state_cache(c, state)
    await cp.aput_messages(c, state.context)
    await cp.aput_tool_result(c, "m0:call", {"output": "x"})


async def test_clean_thread_leaves_nothing_behind(cp):
    """G17: every read is empty afterwards."""
    c = cfg("t1")
    await _fill(cp, c)
    assert await cp.aclean_thread(c) == 1
    assert await cp.aget_state(c) is None
    assert await cp.aget_state_cache(c) is None
    assert await cp.alist_messages(c) == []
    assert await cp.aget_tool_result(c, "m0:call") is None
    assert await cp.aget_thread(c) is None
    assert await cp.aget_thread_owner("t1") is None


async def test_deleted_thread_id_can_be_created_by_another_user(cp):
    """G17: ``A deleted id can be created again by anyone``."""
    await _fill(cp, cfg("t1", user="alice"))
    await cp.aclean_thread(cfg("t1", user="alice"))
    assert await cp.aput_thread(cfg("t1", user="bob"), ThreadInfo(thread_id="t1")) is True
    assert await cp.aget_thread_owner("t1") == "bob"
    assert await cp.alist_messages(cfg("t1", user="bob")) == []


async def test_clean_thread_with_state_but_no_record_returns_one(backend, cp):
    """G18: data without a thread row still counts (P cannot have that shape)."""
    if backend == "pg":
        pytest.skip("Postgres cannot hold state without a thread row")
    c = cfg("t1")
    await cp.aput_state(c, _state("a"))
    assert await cp.aclean_thread(c) == 1
    assert await cp.aget_state(c) is None


async def test_thread_owner_follows_the_claimed_owner(cp):
    c = cfg("t1")
    assert await cp.aget_thread_owner("t1") is None
    await cp.aput_thread(c, ThreadInfo(thread_id="t1"))
    assert await cp.aget_thread_owner("t1") == "alice"


async def test_thread_owner_with_state_but_no_record(backend, cp):
    """G15: M and S return None although the owner is stored with the state."""
    if backend == "pg":
        pytest.skip("Postgres cannot hold state without a thread row")
    await cp.aput_state(cfg("t1"), _state("a"))
    assert await cp.aget_thread_owner("t1") == "alice"


async def test_stop_request_and_active_run_cleared_with_thread(cp):
    """The stop flag and the active run marker die with the thread."""
    c = cfg("t1")
    await _fill(cp, c)
    await cp.abegin_run(c)
    await cp.arequest_stop(c)
    await cp.aclean_thread(c)
    assert await cp.aget_active_run(c) is None
    assert await cp.ais_stop_requested(c) is False


# ------------------------------------------------------------------ reach and tenants


async def test_foreign_thread_is_invisible_under_own(cp):
    await cp.aput_thread(cfg("t1", user="alice"), ThreadInfo(thread_id="t1"))
    await cp.aput_messages(cfg("t1", user="alice"), [_msg(1, "secret")])
    bob = cfg("t1", user="bob")
    assert await cp.aget_thread(bob) is None
    assert await cp.alist_messages(bob) == []
    assert await cp.aclean_thread(bob) == 0
    assert await cp.aget_thread(cfg("t1", user="alice")) is not None


async def test_foreign_write_is_refused(cp):
    await cp.aput_thread(cfg("t1", user="alice"), ThreadInfo(thread_id="t1"))
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_thread(cfg("t1", user="bob"), ThreadInfo(thread_id="t1"))


async def test_foreign_write_is_refused_under_reach_all(cp):
    """``ThreadNotFoundError: The thread belongs to another user or tenant (any reach)``."""
    await cp.aput_thread(cfg("t1", user="alice"), ThreadInfo(thread_id="t1"))
    with pytest.raises(ThreadNotFoundError):
        await cp.aput_thread(cfg("t1", user="root", reach="all"), ThreadInfo(thread_id="t1"))


async def test_reach_all_sees_every_thread(cp):
    await cp.aput_thread(cfg("a", user="alice", org="A"), ThreadInfo(thread_id="a"))
    await cp.aput_thread(cfg("b", user="bob", org="B"), ThreadInfo(thread_id="b"))
    root = cfg("a", user="root", org="C", reach="all")
    assert {t.thread_id for t in await cp.alist_threads(root)} == {"a", "b"}
    got = await cp.aget_thread(root)
    assert got is not None
    assert got.user_id == "alice"  # the owner, not the caller


async def test_reach_all_cannot_delete_foreign_thread_by_accident(cp):
    """Reads widen with reach; the owner stays the owner."""
    await cp.aput_thread(cfg("a", user="alice"), ThreadInfo(thread_id="a"))
    await cp.aput_messages(cfg("a", user="alice"), [_msg(1)])
    root = cfg("a", user="root", reach="all")
    assert _ids(await cp.alist_messages(root)) == ["m1"]
    assert await cp.aget_thread_owner("a") == "alice"


async def test_reach_tenant(backend, cp):
    """G21: M and S have no tenant columns and must raise ValueError; P filters by tenant."""
    await cp.aput_thread(cfg("a", user="alice", org="A"), ThreadInfo(thread_id="a"))
    await cp.aput_thread(cfg("b", user="bob", org="A"), ThreadInfo(thread_id="b"))
    await cp.aput_thread(cfg("c", user="carol", org="B"), ThreadInfo(thread_id="c"))
    team = cfg("a", user="dave", org="A", reach="tenant")
    if backend != "pg":
        with pytest.raises(ValueError):
            await cp.alist_threads(team)
        return
    assert {t.thread_id for t in await cp.alist_threads(team)} == {"a", "b"}
    assert await cp.aget_thread(cfg("c", user="dave", org="A", reach="tenant")) is None


async def test_reach_own_sees_only_own_threads(cp):
    await cp.aput_thread(cfg("a", user="alice"), ThreadInfo(thread_id="a"))
    await cp.aput_thread(cfg("b", user="bob"), ThreadInfo(thread_id="b"))
    assert {t.thread_id for t in await cp.alist_threads(cfg("a", user="alice"))} == {"a"}


async def test_tenant_filters_on_list_threads(backend, cp):
    """``filters`` on a backend without declared columns raise FilterError."""
    from tenxgraph.storage.sql_filter import FilterError

    await cp.aput_thread(cfg("a", org="A"), ThreadInfo(thread_id="a"))
    if backend == "pg":
        got = await cp.alist_threads(cfg("a", org="A"), filters={"org_id": "A"})
        assert [t.thread_id for t in got] == ["a"]
    else:
        with pytest.raises(FilterError):
            await cp.alist_threads(cfg("a"), filters={"org_id": "A"})


async def test_same_user_id_in_two_tenants_is_isolated(backend, cp):
    if backend != "pg":
        pytest.skip("tenant columns need PgCheckpointer")
    await cp.aput_thread(cfg("a", user="alice", org="A"), ThreadInfo(thread_id="a"))
    assert await cp.aget_thread(cfg("a", user="alice", org="B")) is None


async def test_anonymous_use(backend, make_cp):
    """No user_id: M and S work anonymously; P (the production backend) refuses with ValueError."""
    cp = await make_cp(tenant_columns=None)
    anon = {"thread_id": "t1"}
    if backend == "pg":
        with pytest.raises(ValueError):
            await cp.aput_thread(anon, ThreadInfo(thread_id="t1"))
        with pytest.raises(ValueError):
            await cp.aput_tool_result(anon, "c", {"output": 1})
        return
    await cp.aput_thread(anon, ThreadInfo(thread_id="t1"))
    assert await cp.aget_thread(anon) is not None
    await cp.aput_tool_result(anon, "c", {"output": 1})
    assert await cp.aget_tool_result(anon, "c") == {"output": 1}


# ------------------------------------------------------------------ NUL and surrogates


@pytest.mark.parametrize("bad", ["a\x00b", "a\ud800b"], ids=["nul", "lone-surrogate"])
async def test_message_text_with_nul_or_surrogate(cp, bad):
    """G20: round trip exactly, or refuse with ValidationError. Never corrupt, never a raw
    driver error."""
    c = cfg("t1")
    try:
        await cp.aput_messages(c, [_msg(1, bad)])
    except ValidationError:
        return
    assert (await cp.aget_message(c, "m1")).text() == bad


@pytest.mark.parametrize("bad", ["a\x00b", "a\ud800b"], ids=["nul", "lone-surrogate"])
async def test_state_text_with_nul_or_surrogate(cp, bad):
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1"))
    try:
        await cp.aput_state(c, _state(bad))
    except ValidationError:
        return
    got = await cp.aget_state(c)
    assert got.context[0].text() == bad


@pytest.mark.parametrize("bad", ["a\x00b", "a\ud800b"], ids=["nul", "lone-surrogate"])
async def test_ledger_and_cache_with_nul_or_surrogate(cp, bad):
    c = cfg("t1")
    await cp.aput_thread(c, ThreadInfo(thread_id="t1"))
    try:
        await cp.aput_tool_result(c, "m0:c", {"output": bad})
        await cp.aput_cache_value("ns", "k", {"v": bad})
    except ValidationError:
        return
    assert await cp.aget_tool_result(c, "m0:c") == {"output": bad}
    assert await cp.aget_cache_value("ns", "k") == {"v": bad}


# ------------------------------------------------------------------ cache values


async def test_cache_roundtrip(cp):
    assert await cp.aget_cache_value("ns", "k") is None
    await cp.aput_cache_value("ns", "k", {"a": [1, 2]})
    assert await cp.aget_cache_value("ns", "k") == {"a": [1, 2]}
    assert await cp.aget_cache_value("other", "k") is None


async def test_cache_overwrite(cp):
    await cp.aput_cache_value("ns", "k", "one")
    await cp.aput_cache_value("ns", "k", "two")
    assert await cp.aget_cache_value("ns", "k") == "two"


async def test_cache_ttl_expires(cp):
    await cp.aput_cache_value("ns", "k", "v", ttl_seconds=1)
    assert await cp.aget_cache_value("ns", "k") == "v"
    await asyncio.sleep(1.3)
    assert await cp.aget_cache_value("ns", "k") is None


async def test_cache_clear_returns_a_plain_value(cp):
    """True when a value was removed."""
    await cp.aput_cache_value("ns", "k", "v")
    assert await cp.aclear_cache_value("ns", "k") is True
    assert await cp.aget_cache_value("ns", "k") is None


async def test_cache_clear_missing_key(cp):
    assert await cp.aclear_cache_value("ns", "absent") is False


async def test_cache_clear_if_and_put_if_absent(cp):
    assert await cp.aput_cache_value_if_absent("ns", "k", "a") is True
    assert await cp.aput_cache_value_if_absent("ns", "k", "b") is False
    assert await cp.aget_cache_value("ns", "k") == "a"
    assert await cp.aclear_cache_value_if("ns", "k", "zzz") is False
    assert await cp.aclear_cache_value_if("ns", "k", "a") is True
    assert await cp.aget_cache_value("ns", "k") is None


async def test_cache_list_keys_with_prefix(cp):
    await cp.aput_cache_value("ns", "a:1", 1)
    await cp.aput_cache_value("ns", "a:2", 2)
    await cp.aput_cache_value("ns", "b:1", 3)
    assert sorted(await cp.alist_cache_keys("ns", prefix="a:")) == ["a:1", "a:2"]
    assert sorted(await cp.alist_cache_keys("ns")) == ["a:1", "a:2", "b:1"]
