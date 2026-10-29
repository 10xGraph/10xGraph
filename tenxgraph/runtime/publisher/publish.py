import asyncio
import json
import logging
from typing import Any

from injectq import Inject

from tenxgraph.runtime.publisher.base_publisher import BasePublisher
from tenxgraph.runtime.publisher.events import EventModel
from tenxgraph.utils.background_task_manager import BackgroundTaskManager, resolve_task_manager
from tenxgraph.utils.injection import fresh


logger = logging.getLogger("tenxgraph.publisher")


async def _publish_event_task(
    event: EventModel,
    publisher: BasePublisher | None,
) -> None:
    """Publish an event asynchronously if publisher is configured.

    Args:
        event: The event to publish.
        publisher: The publisher instance, or None.
    """
    if publisher:
        # Optional bound on one send, set by the developer as ``publish_timeout`` (seconds)
        # in the publisher config. Unset means wait for the broker.
        config = getattr(publisher, "config", None)
        timeout = config.get("publish_timeout") if isinstance(config, dict) else None
        if not isinstance(timeout, int | float) or isinstance(timeout, bool):
            timeout = None
        try:
            if timeout:
                await asyncio.wait_for(publisher.publish(event), timeout=timeout)
            else:
                await publisher.publish(event)
            logger.debug("Published event: %s", event)
        except TimeoutError:
            logger.error("Publishing event timed out after %ss; event dropped", timeout)
        except Exception as e:
            logger.error("Failed to publish event: %s", e)


# Default size above which an event's ``data["state"]`` is replaced by a bounded summary.
DEFAULT_STATE_SNAPSHOT_LIMIT = 65536
_SUMMARY_MAX_CONTEXT = 50
_SUMMARY_PART_LIMIT = 8192


def _json_len(value: Any) -> int:
    try:
        return len(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return len(str(value))


def _state_limit(publisher: BasePublisher) -> int | None:
    """Size limit from the publisher's ``state_snapshot_limit`` config; None means no limit."""
    config = getattr(publisher, "config", None)
    if not isinstance(config, dict) or "state_snapshot_limit" not in config:
        return DEFAULT_STATE_SNAPSHOT_LIMIT
    limit = config["state_snapshot_limit"]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return None
    return limit


def bound_state_snapshot(state: Any, limit: int | None) -> Any:
    """Return ``state`` as is when small, else a bounded summary of it.

    The summary keeps: ``truncated`` (True), ``bytes`` (original JSON size), ``fields``
    (top-level field names), ``context_count``, ``context`` (``message_id`` and ``role`` of the
    last 50 messages, plus full system messages up to 8 KiB each), ``context_summary`` (cut
    to 1000 characters) and ``execution_meta`` when it is under 8 KiB.
    """
    if limit is None or not isinstance(state, dict):
        return state
    size = _json_len(state)
    if size <= limit:
        return state
    context = state.get("context")
    entries: list[dict[str, Any]] = []
    if isinstance(context, list):
        for m in context[-_SUMMARY_MAX_CONTEXT:]:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "system" and _json_len(m) <= _SUMMARY_PART_LIMIT:
                entries.append(m)
            else:
                entries.append({"message_id": m.get("message_id"), "role": m.get("role")})
    summary: dict[str, Any] = {
        "truncated": True,
        "bytes": size,
        "fields": sorted(state),
        "context_count": len(context) if isinstance(context, list) else 0,
        "context": entries,
    }
    ctx_summary = state.get("context_summary")
    if isinstance(ctx_summary, str):
        summary["context_summary"] = ctx_summary[:1000]
    meta = state.get("execution_meta")
    if meta is not None and _json_len(meta) <= _SUMMARY_PART_LIMIT:
        summary["execution_meta"] = meta
    return summary


def publish_event(
    event: EventModel,
    publisher: BasePublisher | None = Inject[BasePublisher],
    task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
) -> None:
    """Publish an event asynchronously using the background task manager.

    Delivery is best-effort: events run as background tasks, and when more than the
    manager's cap (1000 by default) are pending, new events are shed and a warning is
    logged. ``CompiledGraph.aclose()`` and sync ``invoke()`` wait for pending events.
    ``data["state"]`` larger than the publisher's ``state_snapshot_limit`` config (JSON
    characters, default 65536; 0 or None disables) is replaced by a bounded summary.

    Args:
        event: The event to publish.
        publisher: The publisher instance (injected).
        task_manager: The background task manager (injected).
    """
    publisher = fresh(publisher)
    task_manager = resolve_task_manager(fresh(task_manager))
    # No sink bound -> nothing to publish. Spawning a task per event just to
    # discover there is nowhere to send it cost a task and kept the event alive
    # for no reason, on the hot path of every node in every run.
    if publisher is None:
        return

    # Store the task to prevent it from being garbage collected. May return None
    # when the manager is shedding load (see BackgroundTaskManager.create_task).
    # Snapshot the event first: the run loop reuses one event object and edits its
    # data, metadata and phase between publishes, so the queued task would otherwise
    # send the last values for every event.
    data = dict(event.data)
    # A queued event must not pin a whole run state while the sink is slow, so a large
    # ``state`` is cut to a bounded summary (see ``bound_state_snapshot``).
    if "state" in data:
        data["state"] = bound_state_snapshot(data["state"], _state_limit(publisher))
    snapshot = event.model_copy(update={"data": data, "metadata": dict(event.metadata)})
    task_manager.create_task(_publish_event_task(snapshot, publisher))
