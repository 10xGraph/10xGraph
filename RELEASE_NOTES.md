# 10xgraph 0.10.3

Python 3.14 and 3.15 support. No API changes.

```bash
pip install -U "10xgraph==0.10.3"
```

## Added

- **Python 3.14 and 3.15 are officially supported.**
  - CI runs the full test suite on 3.12, 3.13, 3.14 and 3.15.

## Changed

- **`injectq` 0.5 is now allowed.**
  - 0.5.0 fixes a crash under Python 3.15 global lazy imports (`PYTHON_LAZY_IMPORTS=all`).
  - 0.4.x keeps working.
- **Code that Python 3.16 removes is replaced.**
  - `asyncio.iscoroutinefunction` and the event-loop policy calls are gone, so 3.14 and 3.15 no longer warn.
  - The sync `CompiledGraph.stream()` no longer changes the thread's current event loop.

See `CHANGELOG.md` for details.
