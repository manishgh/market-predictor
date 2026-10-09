"""Synthetic UNIT memory boundaries; never load or reconstruct parent data."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from market_predictor.core.errors import MemoryBudgetError
from market_predictor.core.system_memory import SystemMemory
from market_predictor.research import news_parent_inputs as parent
from market_predictor.research import news_runtime_memory as runtime


def _snapshot(monkeypatch: pytest.MonkeyPatch, percentage: float) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    snapshot = (SystemMemory(total_bytes=1_000_000_000, available_bytes=round(10_000_000 * (100 - percentage)))
                if math.isfinite(percentage) else
                SimpleNamespace(used_percent=percentage, total_bytes=1_000_000_000, available_bytes=100_000_000))
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: snapshot)
    return calls


@pytest.mark.parametrize("percentage", [85.0, 89.999])
def test_parent_allows_below_90_without_old_absolute_floor(monkeypatch: pytest.MonkeyPatch, percentage: float) -> None:
    calls = _snapshot(monkeypatch, percentage)
    parent._guard()
    assert calls == [{"stage": "saved news parent", "hard_budget_gib": 5.0, "headroom_gib": 0.75}]


@pytest.mark.parametrize("percentage", [90.0, 90.001, float("nan"), float("inf")])
def test_parent_rejects_boundary_or_invalid_system_percentage(monkeypatch: pytest.MonkeyPatch, percentage: float) -> None:
    _snapshot(monkeypatch, percentage)
    with pytest.raises(MemoryBudgetError):
        parent._guard()


def test_parent_rejects_unknown_system_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _snapshot(monkeypatch, 80.0)
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: None)
    with pytest.raises(MemoryBudgetError):
        parent._guard()


def test_parent_keeps_process_budget_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _snapshot(monkeypatch, 80.0)

    def exceeded(**kwargs: Any) -> None:
        assert kwargs == {"stage": "saved news parent", "hard_budget_gib": 5.0, "headroom_gib": 0.75}
        raise MemoryBudgetError("UNIT process budget exceeded")

    monkeypatch.setattr(runtime, "assert_memory_budget", exceeded)
    with pytest.raises(MemoryBudgetError, match="UNIT process budget exceeded"):
        parent._guard()


def test_parent_pins_executed_runtime_and_existing_memory_owners() -> None:
    assert {"research/news_runtime_memory.py", "resources.py", "process_memory.py", "core/system_memory.py"}.issubset(
        parent.IMPLEMENTATION_PATHS)
