from __future__ import annotations

from dataclasses import dataclass

from market_predictor.process_memory import (
    process_memory_snapshot as process_memory_snapshot,
)
from market_predictor.process_memory import (
    release_process_memory as release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError


@dataclass(frozen=True, slots=True)
class MemoryAudit:
    hard_budget_gib: float
    safety_threshold_gib: float
    current_working_set_gib: float | None
    peak_working_set_gib: float | None

    def to_record(self) -> dict[str, float | None]:
        return {
            "hard_budget_gib": self.hard_budget_gib,
            "safety_threshold_gib": self.safety_threshold_gib,
            "current_working_set_gib": self.current_working_set_gib,
            "peak_working_set_gib": self.peak_working_set_gib,
        }


def assert_memory_budget(
    *,
    hard_budget_gib: float,
    headroom_gib: float,
    stage: str,
) -> None:
    if hard_budget_gib <= 0 or headroom_gib <= 0 or headroom_gib >= hard_budget_gib:
        raise ValueError("memory budget and headroom are invalid")
    snapshot = process_memory_snapshot()
    if snapshot is None:
        return
    threshold = int((hard_budget_gib - headroom_gib) * 1024**3)
    if snapshot[0] > threshold:
        raise DataReadinessError(
            f"memory guard stopped {stage}: RSS {_gib(snapshot[0]):.2f} GiB exceeds "
            f"the {_gib(threshold):.2f} GiB safety threshold for the {hard_budget_gib:.2f} GiB hard budget"
        )


def memory_audit(*, hard_budget_gib: float, headroom_gib: float) -> MemoryAudit:
    snapshot = process_memory_snapshot()
    return MemoryAudit(
        hard_budget_gib=hard_budget_gib,
        safety_threshold_gib=hard_budget_gib - headroom_gib,
        current_working_set_gib=_gib(snapshot[0]) if snapshot is not None else None,
        peak_working_set_gib=_gib(snapshot[1]) if snapshot is not None else None,
    )


def _gib(value: int) -> float:
    return value / 1024**3
