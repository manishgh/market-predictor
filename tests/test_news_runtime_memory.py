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

from market_predictor.core import runtime_memory as memory
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import SystemMemory
from market_predictor.research import news_runtime_memory as runtime


@pytest.mark.parametrize("available,allowed", [(200, True), (150, True), (101, True), (100, False), (91, False), (90, False), (89, False)])
def test_percentage_only_requested_boundary(monkeypatch: pytest.MonkeyPatch, available: int, allowed: bool) -> None:
    monkeypatch.setattr(memory, "MAXIMUM_SYSTEM_PAUSE_SECONDS", 0.0)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(memory, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    # Even this tiny synthetic available byte count is admitted below90: no2GiB floor.
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: SystemMemory(1000, available))
    if allowed:
        memory.guard(stage="unit")
    else:
        with pytest.raises(MemoryBudgetError, match="strictly below 90"):
            memory.guard(stage="unit")
    assert calls == [{"stage": "unit", "hard_budget_gib": 5.0, "headroom_gib": 0.75}]


@pytest.mark.parametrize("snapshot", [None, SystemMemory(0, 0), SystemMemory(100, -1), SystemMemory(100, 101)])
def test_unknown_or_invalid_system_snapshot_rejects(monkeypatch: pytest.MonkeyPatch, snapshot: SystemMemory | None) -> None:
    monkeypatch.setattr(memory, "assert_memory_budget", lambda **kwargs: None)
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: snapshot)
    with pytest.raises(MemoryBudgetError, match="measurement unavailable"):
        memory.guard()


def test_process_budget_failure_is_not_relaxed(monkeypatch: pytest.MonkeyPatch) -> None:
    def process(**kwargs: object) -> None:
        assert kwargs["hard_budget_gib"] == 5.0 and kwargs["headroom_gib"] == 0.75
        raise MemoryBudgetError("synthetic process pressure")

    monkeypatch.setattr(memory, "assert_memory_budget", process)
    with pytest.raises(MemoryBudgetError, match="process pressure"):
        memory.guard()


def test_bounded_io_uses_new_guard_preserves_hashes_and_rejects_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "assert_memory_budget", lambda **kwargs: None)
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: SystemMemory(1000, 110))
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
    monkeypatch.setattr(memory, "assert_memory_budget", lambda **kwargs: None)
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: SystemMemory(1000, 120))
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
    monkeypatch.setattr(memory.time, "monotonic", lambda: value.now)
    monkeypatch.setattr(memory.time, "sleep", value.sleep)
    monkeypatch.setattr(memory, "assert_memory_budget", lambda **kwargs: None)
    return value


def test_healthy_guard_does_not_sleep_or_log(clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: SystemMemory(10000, 1001))
    memory.guard()
    assert clock.sleeps == []
    assert capsys.readouterr().err == ""


def test_exact_boundary_recovers_and_logs_actual_facts(clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    snapshots = iter([SystemMemory(10000, 1000), SystemMemory(10000, 1001)])
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(memory, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    memory.guard(stage="unit recovery")
    assert clock.sleeps == [5.0]
    assert calls == [{"stage": "unit recovery", "hard_budget_gib": 5.0, "headroom_gib": 0.75}] * 2
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [event["event"] for event in events] == ["pause", "resume"]
    assert [event["used_percent"] for event in events] == pytest.approx([90.0, 89.99])
    assert [event["elapsed_seconds"] for event in events] == [0.0, 5.0]
    for event in events:
        assert event["stage"] == "unit recovery"
        assert event["ceiling_percent"] == 90.0
        assert event["maximum_wait_seconds"] == 120.0
        assert datetime.fromisoformat(event["timestamp_utc"]).utcoffset().total_seconds() == 0


def test_sustained_pressure_times_out_at_monotonic_deadline(clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[object] = []
    monkeypatch.setattr(memory, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: SystemMemory(100, 8))
    with pytest.raises(MemoryBudgetError, match="within 120 seconds"):
        memory.guard()
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
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: next(snapshots))
    with pytest.raises(MemoryBudgetError, match="measurement unavailable"):
        memory.guard()
    assert clock.sleeps == [5.0]
    assert [json.loads(line)["event"] for line in capsys.readouterr().err.splitlines()] == ["pause"]


def test_process_failure_during_pause_is_immediate(clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def process(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryBudgetError("UNIT process pressure")

    monkeypatch.setattr(memory, "assert_memory_budget", process)
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: SystemMemory(100, 8))
    with pytest.raises(MemoryBudgetError, match="UNIT process pressure"):
        memory.guard()
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
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: next(snapshots, SystemMemory(100, 11)))
    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(runtime.pq, "ParquetFile", tracked_parquet)
    if kind == "hash":
        assert runtime.file_sha256(path) == expected_hash
    else:
        assert list(runtime.parquet_rows(path, ("value",))) == expected_rows
    assert clock.sleeps == [5.0]
    assert io_calls


@pytest.mark.parametrize("reclaim", [True, False])
def test_cleanup_keyword_preserves_real_process_and_system_measurements(monkeypatch, reclaim):
    from market_predictor import resources

    events = []
    monkeypatch.setattr(memory, "assert_memory_budget", resources.assert_memory_budget)
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: events.append("process") or (1024**3, 1024**3))
    monkeypatch.setattr(resources, "release_unused_process_memory", lambda: events.append("cleanup"))
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: events.append("system") or SystemMemory(1000, 101))
    memory.guard("unit cadence", reclaim_unused_memory=reclaim)
    assert events == ["process", *(["cleanup"] if reclaim else []), "system"]


@pytest.mark.parametrize("snapshot", [None, SystemMemory(0, 0), SystemMemory(100, -1), SystemMemory(100, 101)])
def test_cleanup_opt_out_cannot_admit_unknown_or_invalid_system_measurement(monkeypatch, snapshot):
    from market_predictor import resources

    events = []
    monkeypatch.setattr(memory, "assert_memory_budget", resources.assert_memory_budget)
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: events.append("process") or (1024**3, 1024**3))
    monkeypatch.setattr(resources, "release_unused_process_memory", lambda: pytest.fail("initial cleanup opted out"))
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: events.append("system") or snapshot)
    with pytest.raises(MemoryBudgetError, match="measurement unavailable"):
        memory.guard("unit unknown", reclaim_unused_memory=False)
    assert events == ["process", "system"]


@pytest.mark.parametrize("snapshot", [None, (6 * 1024**3, 6 * 1024**3)])
def test_cleanup_opt_out_cannot_bypass_unknown_or_exceeded_process_budget(monkeypatch, snapshot):
    from market_predictor import resources

    monkeypatch.setattr(memory, "assert_memory_budget", resources.assert_memory_budget)
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr(resources, "release_unused_process_memory", lambda: pytest.fail("process refusal precedes cleanup"))
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: pytest.fail("process refusal precedes system check"))
    with pytest.raises(MemoryBudgetError):
        memory.guard("unit process refusal", reclaim_unused_memory=False)


def test_cleanup_opt_out_recovers_from_exact90_with_full_cleanup_every_poll(clock, monkeypatch, capsys):
    from market_predictor import resources

    events = []
    snapshots = iter([SystemMemory(10000, 1000), SystemMemory(10000, 1000), SystemMemory(10000, 1001)])
    monkeypatch.setattr(memory, "assert_memory_budget", resources.assert_memory_budget)
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: events.append("process") or (1024**3, 1024**3))
    monkeypatch.setattr(resources, "release_unused_process_memory", lambda: events.append("cleanup"))
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: events.append("system") or next(snapshots))
    memory.guard("unit pressure recovery", reclaim_unused_memory=False)
    assert events == ["process", "system", "process", "cleanup", "system", "process", "cleanup", "system"]
    assert clock.sleeps == [5.0, 5.0]
    logs = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [entry["event"] for entry in logs] == ["pause", "resume"]
    assert [entry["used_percent"] for entry in logs] == pytest.approx([90.0, 89.99])
    assert logs[-1]["elapsed_seconds"] == 10.0


def test_cleanup_opt_out_sustained90_times_out_and_reclaims_every_poll(clock, monkeypatch, capsys):
    from market_predictor import resources

    events = []
    monkeypatch.setattr(memory, "assert_memory_budget", resources.assert_memory_budget)
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: events.append("process") or (1024**3, 1024**3))
    monkeypatch.setattr(resources, "release_unused_process_memory", lambda: events.append("cleanup"))
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: events.append("system") or SystemMemory(10000, 1000))
    with pytest.raises(MemoryBudgetError, match="within 120 seconds"):
        memory.guard("unit pressure timeout", reclaim_unused_memory=False)
    assert clock.now == 120.0 and clock.sleeps == [5.0] * 24
    assert events.count("process") == events.count("system") == 25
    assert events.count("cleanup") == 24
    assert [json.loads(line)["event"] for line in capsys.readouterr().err.splitlines()] == ["pause", "timeout"]


def test_cleanup_opt_out_does_not_hide_invalid_measurement_during_pressure_poll(clock, monkeypatch):
    from market_predictor import resources

    events = []
    snapshots = iter([SystemMemory(10000, 1000), None])
    monkeypatch.setattr(memory, "assert_memory_budget", resources.assert_memory_budget)
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: events.append("process") or (1024**3, 1024**3))
    monkeypatch.setattr(resources, "release_unused_process_memory", lambda: events.append("cleanup"))
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: events.append("system") or next(snapshots))
    with pytest.raises(MemoryBudgetError, match="measurement unavailable"):
        memory.guard("unit unknown poll", reclaim_unused_memory=False)
    assert clock.sleeps == [5.0]
    assert events == ["process", "system", "process", "cleanup", "system"]


@pytest.mark.parametrize("failed", [None, (6 * 1024**3, 6 * 1024**3)])
def test_cleanup_opt_out_rechecks_process_budget_before_pressure_poll_cleanup(clock, monkeypatch, failed):
    from market_predictor import resources

    events = []
    process = iter([(1024**3, 1024**3), failed])
    monkeypatch.setattr(memory, "assert_memory_budget", resources.assert_memory_budget)
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: events.append("process") or next(process))
    monkeypatch.setattr(resources, "release_unused_process_memory", lambda: pytest.fail("failed RSS check precedes cleanup"))
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: events.append("system") or SystemMemory(10000, 1000))
    with pytest.raises(MemoryBudgetError):
        memory.guard("unit process failure while paused", reclaim_unused_memory=False)
    assert clock.sleeps == [5.0]
    assert events == ["process", "system", "process"]
