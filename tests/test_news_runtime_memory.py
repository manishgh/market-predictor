"""Synthetic UNIT resource snapshots and tiny files; no operational data jobs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import SystemMemory
from market_predictor.research import news_runtime_memory as runtime


@pytest.mark.parametrize("available,allowed", [(200, True), (150, True), (101, True), (100, False), (99, False)])
def test_percentage_only_below_ninety_boundary(monkeypatch: pytest.MonkeyPatch, available: int, allowed: bool) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    # Even this tiny synthetic available byte count is admitted below90: no2GiB floor.
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: SystemMemory(1000, available))
    if allowed:
        runtime.guard(stage="unit")
    else:
        with pytest.raises(MemoryBudgetError, match="strictly below 90"):
            runtime.guard(stage="unit")
    assert calls == [{"stage": "unit", "hard_budget_gib": 5.0, "headroom_gib": 0.75}]


@pytest.mark.parametrize("snapshot", [None, SystemMemory(0, 0), SystemMemory(100, -1), SystemMemory(100, 101)])
def test_unknown_or_invalid_system_snapshot_rejects(monkeypatch: pytest.MonkeyPatch, snapshot: SystemMemory | None) -> None:
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: snapshot)
    with pytest.raises(MemoryBudgetError, match="measurement unavailable"):
        runtime.guard()


def test_process_budget_failure_is_not_relaxed(monkeypatch: pytest.MonkeyPatch) -> None:
    def process(**kwargs: object) -> None:
        assert kwargs["hard_budget_gib"] == 5.0 and kwargs["headroom_gib"] == 0.75
        raise MemoryBudgetError("synthetic process pressure")

    monkeypatch.setattr(runtime, "assert_memory_budget", process)
    with pytest.raises(MemoryBudgetError, match="process pressure"):
        runtime.guard()


def test_bounded_io_uses_new_guard_preserves_hashes_and_rejects_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: SystemMemory(1000, 110))
    path = tmp_path / "unit.bin"
    path.write_bytes(b"synthetic source")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    files: dict[str, str] = {}
    assert runtime.file_sha256(path) == expected
    assert runtime.pin_file(tmp_path, "unit.bin", expected, files) == path
    assert runtime.capture_file(tmp_path, path, files) == expected
    runtime.recheck_files(tmp_path, files)
    path.write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="changed"):
        runtime.recheck_files(tmp_path, files)


def test_parquet_batch_io_admits_above_old_eightyfive_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: SystemMemory(1000, 120))
    path = tmp_path / "unit.parquet"
    pd.DataFrame({"value": range(600)}).to_parquet(path, index=False)
    assert list(runtime.parquet_rows(path, ("value",))) == [{"value": number} for number in range(600)]
