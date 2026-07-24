from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from market_predictor.prediction_contracts import (
    PredictionCapacityError,
    PredictionMemoryPressureError,
)
from market_predictor.resources import process_memory_snapshot


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    active_requests: int
    reserved_gib: float
    max_concurrent_requests: int
    memory_budget_gib: float
    memory_safety_threshold_gib: float

    def to_record(self) -> dict[str, int | float]:
        return {
            "active_requests": self.active_requests,
            "reserved_gib": self.reserved_gib,
            "max_concurrent_requests": self.max_concurrent_requests,
            "memory_budget_gib": self.memory_budget_gib,
            "memory_safety_threshold_gib": self.memory_safety_threshold_gib,
        }


class InferenceAdmissionController:
    """Reject overload before feature/model allocation; never queue requests."""

    def __init__(
        self,
        *,
        max_concurrent_requests: int,
        memory_budget_gib: float,
        memory_headroom_gib: float,
        reject_unknown_memory: bool = False,
    ) -> None:
        if max_concurrent_requests != 1:
            raise ValueError(
                "one inference request per process is required until measured "
                "memory evidence approves higher concurrency"
            )
        if (
            memory_budget_gib <= 0
            or memory_headroom_gib <= 0
            or memory_headroom_gib >= memory_budget_gib
        ):
            raise ValueError("admission memory budget and headroom are invalid")
        self._max_concurrent_requests = max_concurrent_requests
        self._memory_budget_gib = memory_budget_gib
        self._memory_safety_threshold_gib = memory_budget_gib - memory_headroom_gib
        self._reject_unknown_memory = reject_unknown_memory
        self._active_requests = 0
        self._reserved_gib = 0.0
        self._lock = threading.Lock()

    @contextmanager
    def lease(self, *, estimated_incremental_gib: float) -> Iterator[None]:
        if estimated_incremental_gib <= 0:
            raise ValueError("inference memory reservation must be positive")
        with self._lock:
            if self._active_requests >= self._max_concurrent_requests:
                raise PredictionCapacityError
            current_gib = _current_rss_gib()
            if current_gib is None and self._reject_unknown_memory:
                raise PredictionMemoryPressureError
            projected = (
                (current_gib or 0.0)
                + self._reserved_gib
                + estimated_incremental_gib
            )
            if projected > self._memory_safety_threshold_gib:
                raise PredictionMemoryPressureError
            self._active_requests += 1
            self._reserved_gib += estimated_incremental_gib
        try:
            yield
        finally:
            with self._lock:
                self._active_requests -= 1
                self._reserved_gib -= estimated_incremental_gib

    def snapshot(self) -> AdmissionSnapshot:
        with self._lock:
            return AdmissionSnapshot(
                active_requests=self._active_requests,
                reserved_gib=self._reserved_gib,
                max_concurrent_requests=self._max_concurrent_requests,
                memory_budget_gib=self._memory_budget_gib,
                memory_safety_threshold_gib=self._memory_safety_threshold_gib,
            )


def _current_rss_gib() -> float | None:
    snapshot = process_memory_snapshot()
    return snapshot[0] / 1024**3 if snapshot is not None else None
