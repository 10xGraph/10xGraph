"""TEMP-PERF: temporary timers for the per-request overhead hunt. Delete this file when done.

``now()`` takes a start time, ``done(part, t0)`` records the elapsed ms in the histogram
``tenxgraph.perf.<part>.duration`` (the core metrics registry, bridged to OTel/Prometheus).
"""  # TEMP-PERF

import time  # TEMP-PERF
from typing import Any  # TEMP-PERF

from tenxgraph.utils import metrics  # TEMP-PERF


now = time.perf_counter  # TEMP-PERF
_M: dict[str, Any] = {}  # TEMP-PERF


def done(part: str, t0: float, attrs: dict[str, Any] | None = None) -> None:  # TEMP-PERF
    m = _M.get(part)  # TEMP-PERF
    if m is None:  # TEMP-PERF
        m = _M[part] = metrics.timer(f"tenxgraph.perf.{part}.duration").metric  # TEMP-PERF
    m.observe((time.perf_counter() - t0) * 1000.0, attrs)  # TEMP-PERF
