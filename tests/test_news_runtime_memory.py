"""Synthetic UNIT resource snapshots and tiny files; no operational data jobs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import SystemMemory
from market_predictor.research import news_runtime_memory as runtime


@pytest.mark.parametrize("available,allowed", [(200, True), (150, True), (101, True), (100, True), (91, True), (90, False), (89, False)])
def test_percentage_only_requested_boundary(monkeypatch: pytest.MonkeyPatch, available: int, allowed: bool) -> None:
    monkeypatch.setattr(runtime, "MAXIMUM_SYSTEM_PAUSE_SECONDS", 0.0)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    # Even this tiny synthetic available byte count is admitted below91: no2GiB floor.
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: SystemMemory(1000, available))
    if allowed:
        runtime.guard(stage="unit")
    else:
        with pytest.raises(MemoryBudgetError, match="strictly below 91"):
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


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.during_sleep: Callable[[], None] = lambda: None

    def sleep(self, seconds: float) -> None:
        self.during_sleep()
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    value = FakeClock()
    monkeypatch.setattr(runtime.time, "monotonic", lambda: value.now)
    monkeypatch.setattr(runtime.time, "sleep", value.sleep)
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: None)
    return value


def test_healthy_guard_does_not_sleep_or_log(clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: SystemMemory(10000, 901))
    runtime.guard()
    assert clock.sleeps == []
    assert capsys.readouterr().err == ""


def test_exact_boundary_recovers_and_logs_actual_facts(clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    snapshots = iter([SystemMemory(10000, 900), SystemMemory(10000, 901)])
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    runtime.guard(stage="unit recovery")
    assert clock.sleeps == [5.0]
    assert calls == [{"stage": "unit recovery", "hard_budget_gib": 5.0, "headroom_gib": 0.75}] * 2
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [event["event"] for event in events] == ["pause", "resume"]
    assert [event["used_percent"] for event in events] == pytest.approx([91.0, 90.99])
    assert [event["elapsed_seconds"] for event in events] == [0.0, 5.0]
    for event in events:
        assert event["stage"] == "unit recovery"
        assert event["ceiling_percent"] == 91.0
        assert event["maximum_wait_seconds"] == 120.0
        assert datetime.fromisoformat(event["timestamp_utc"]).utcoffset().total_seconds() == 0


def test_sustained_pressure_times_out_at_monotonic_deadline(clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: SystemMemory(100, 8))
    with pytest.raises(MemoryBudgetError, match="within 120 seconds"):
        runtime.guard()
    assert clock.now == 120.0
    assert clock.sleeps == [5.0] * 24
    assert len(calls) == 25
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [event["event"] for event in events] == ["pause", "timeout"]
    assert events[-1]["elapsed_seconds"] == 120.0
    assert events[-1]["used_percent"] == 92.0


@pytest.mark.parametrize("invalid", [None, SystemMemory(0, 0), SystemMemory(100, -1),
    SimpleNamespace(total_bytes=float("inf"), available_bytes=1, used_percent=99.0),
    SimpleNamespace(total_bytes=100, available_bytes=10, used_percent=float("nan")),
    SimpleNamespace(total_bytes=100, available_bytes=10, used_percent=float("inf"))])
def test_invalid_measurement_during_pause_fails_immediately(clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
                                                          invalid: Any, capsys: pytest.CaptureFixture[str]) -> None:
    snapshots = iter([SystemMemory(100, 8), invalid])
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: next(snapshots))
    with pytest.raises(MemoryBudgetError, match="measurement unavailable"):
        runtime.guard()
    assert clock.sleeps == [5.0]
    assert [json.loads(line)["event"] for line in capsys.readouterr().err.splitlines()] == ["pause"]


def test_process_failure_during_pause_is_immediate(clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def process(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryBudgetError("UNIT process pressure")

    monkeypatch.setattr(runtime, "assert_memory_budget", process)
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: SystemMemory(100, 8))
    with pytest.raises(MemoryBudgetError, match="UNIT process pressure"):
        runtime.guard()
    assert clock.sleeps == [5.0]


@pytest.mark.parametrize("kind", ["hash", "parquet"])
def test_data_io_does_not_start_while_paused_and_results_unchanged(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    path = tmp_path / "synthetic.parquet"
    expected_rows = [{"value": value} for value in range(600)]
    pd.DataFrame(expected_rows).to_parquet(path, index=False)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    io_calls: list[str] = []
    original_open = Path.open
    original_parquet = runtime.pq.ParquetFile

    def tracked_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        io_calls.append("open")
        return original_open(self, *args, **kwargs)

    def tracked_parquet(*args: Any, **kwargs: Any) -> Any:
        io_calls.append("parquet")
        return original_parquet(*args, **kwargs)

    def during_pause() -> None:
        assert io_calls == []

    clock.during_sleep = during_pause
    snapshots = iter([SystemMemory(100, 8)])
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: next(snapshots, SystemMemory(100, 10)))
    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(runtime.pq, "ParquetFile", tracked_parquet)
    if kind == "hash":
        assert runtime.file_sha256(path) == expected_hash
    else:
        assert list(runtime.parquet_rows(path, ("value",))) == expected_rows
    assert clock.sleeps == [5.0]
    assert io_calls
