# 10xgraph 0.10.1

A bug-fix release for the checkpointers' state cache. No API changes; upgrading is safe.

```bash
pip install -U "10xgraph==0.10.1"
```

## Fixed

- **State written with `aput_state` between runs is now what the next run starts from.**
  - A run reads its state from the cache first, but `aput_state` only wrote the durable store.
  - `InMemoryCheckpointer` and `SqliteCheckpointer` silently lost the update.
  - `PgCheckpointer` failed the next run with `StaleStateError` (`STORAGE_CONFLICT_001`).
  - All three now refresh the cache on write. On Postgres the Redis refresh is best-effort, so a Redis outage never fails the write.
- **Clearing a thread's state now also clears its cached copy.**
  - Before, `aclear_state` on `InMemoryCheckpointer` and `SqliteCheckpointer` left the cache behind, so the next run resumed from the cleared state.
  - `InMemoryCheckpointer.aclean_thread` now removes the thread's state, cached state and messages, as the SQLite and Postgres checkpointers already did.

See `CHANGELOG.md` for details.
