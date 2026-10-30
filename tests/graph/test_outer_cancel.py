"""An outer cancel that arrives while core waits for its own cancelled task is not absorbed (G-27)."""

import asyncio

import pytest

from tenxgraph.core.graph.utils.guards import execute_with_guards
from tenxgraph.utils.shutdown import shutdown_with_timeout


async def stubborn_work():
    """Work whose cleanup after a cancel takes a while."""
    try:
        await asyncio.sleep(30)
    except asyncio.CancelledError:
        await asyncio.sleep(0.5)
        raise


@pytest.mark.asyncio
async def test_outer_cancel_during_timeout_cleanup_is_not_swallowed():
    class Timeout(Exception):
        pass

    outer = asyncio.ensure_future(
        execute_with_guards(stubborn_work(), timeout=0.05, on_timeout=lambda: Timeout())
    )
    await asyncio.sleep(0.2)  # the deadline passed; core is waiting for the cleanup
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer


@pytest.mark.asyncio
async def test_outer_cancel_during_stop_cleanup_is_not_swallowed():
    class Stopped(Exception):
        pass

    async def stop_check():
        return True

    outer = asyncio.ensure_future(
        execute_with_guards(
            stubborn_work(),
            stop_check=stop_check,
            poll_interval=0.05,
            on_stop=lambda: Stopped(),
        )
    )
    await asyncio.sleep(0.2)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer


@pytest.mark.asyncio
async def test_guards_still_raise_the_timeout_when_nobody_cancels():
    class Timeout(Exception):
        pass

    with pytest.raises(Timeout):
        await execute_with_guards(stubborn_work(), timeout=0.05, on_timeout=lambda: Timeout())


@pytest.mark.asyncio
async def test_outer_cancel_during_shutdown_cleanup_is_not_swallowed():
    async def unkillable_for_a_while():
        # Ignores the first cancel (wait_for's), so the helper's own cancel has to wait.
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise

    outer = asyncio.ensure_future(shutdown_with_timeout(unkillable_for_a_while(), 0.05, "t"))
    await asyncio.sleep(0.2)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer


@pytest.mark.asyncio
async def test_shutdown_with_timeout_reports_timeout():
    result = await shutdown_with_timeout(asyncio.sleep(30), 0.05, "t")
    assert result["status"] == "timeout"
