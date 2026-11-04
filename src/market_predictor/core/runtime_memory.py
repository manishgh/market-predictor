"""Shared bounded memory-pressure policy for news and saved-return workloads.

System memory must be strictly below 91 percent, with no absolute free-memory
floor. Preserve the 5 GiB process budget, 0.75 GiB headroom, 120-second maximum
pause and 5-second polling used by these workloads.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime

from market_predictor.core.errors import MemoryBudgetError
from market_predictor.core.system_memory import system_memory_snapshot
from market_predictor.resources import assert_memory_budget

MAXIMUM_SYSTEM_USED_PERCENT = 91.0
MAXIMUM_SYSTEM_PAUSE_SECONDS = 120.0
SYSTEM_MEMORY_POLL_SECONDS = 5.0


def _used_percent(stage: str) -> float:
    assert_memory_budget(stage=stage, hard_budget_gib=5.0, headroom_gib=0.75)
    snapshot = system_memory_snapshot()
    if (snapshot is None
            or type(snapshot.total_bytes) is not int
            or type(snapshot.available_bytes) is not int
            or snapshot.total_bytes <= 0
            or not 0 <= snapshot.available_bytes <= snapshot.total_bytes):
        raise MemoryBudgetError(f"memory guard stopped {stage}: system memory measurement unavailable")
    used = snapshot.used_percent
    if not math.isfinite(used) or not 0 <= used <= 100:
        raise MemoryBudgetError(f"memory guard stopped {stage}: system memory measurement unavailable")
    return used


def _event(event: str, stage: str, used: float, elapsed: float) -> None:
    print(json.dumps({
        "event": event, "timestamp_utc": datetime.now(UTC).isoformat(), "stage": stage,
        "used_percent": used, "ceiling_percent": MAXIMUM_SYSTEM_USED_PERCENT,
        "elapsed_seconds": elapsed, "maximum_wait_seconds": MAXIMUM_SYSTEM_PAUSE_SECONDS,
    }, allow_nan=False, sort_keys=True), file=sys.stderr, flush=True)


def guard(stage: str = "news source links") -> None:
    """Pause only valid system pressure; callers retain their lease and transaction.

    No caller data work runs during the bounded wait. Invalid measurements and
    process-budget failures propagate immediately, including during a pause.
    """
    used = _used_percent(stage)
    if used < MAXIMUM_SYSTEM_USED_PERCENT:
        return
    started = time.monotonic()
    deadline = started + MAXIMUM_SYSTEM_PAUSE_SECONDS
    _event("pause", stage, used, 0.0)
    while True:
        now = time.monotonic()
        if now >= deadline:
            _event("timeout", stage, used, now - started)
            raise MemoryBudgetError(
                f"memory guard stopped {stage}: system memory {used:.2f}% used; "
                f"requires strictly below {MAXIMUM_SYSTEM_USED_PERCENT:g}% "
                f"within {MAXIMUM_SYSTEM_PAUSE_SECONDS:g} seconds"
            )
        time.sleep(min(SYSTEM_MEMORY_POLL_SECONDS, deadline - now))
        used = _used_percent(stage)
        now = time.monotonic()
        if used < MAXIMUM_SYSTEM_USED_PERCENT and now <= deadline:
            _event("resume", stage, used, now - started)
            return
