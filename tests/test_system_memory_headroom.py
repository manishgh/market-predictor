"""Synthetic OS measurements test the guard without stressing the laptop."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from market_predictor.core import system_memory as memory
from market_predictor.core.errors import MemoryBudgetError

GIB = 1024**3


@pytest.mark.parametrize(("total", "available", "allowed"), [
    (10 * GIB, 2 * GIB, True), (10 * GIB, 2 * GIB - 1, False),
    (20 * GIB, 3 * GIB, False), (20 * GIB, 3 * GIB + 1, True),
])
def test_memory_thresholds(total: int, available: int, allowed: bool) -> None:
    with patch.object(memory, "system_memory_snapshot", return_value=memory.SystemMemory(total, available)):
        if allowed:
            assert memory.assert_system_memory_available().available_bytes == available
        else:
            with pytest.raises(MemoryBudgetError, match="system memory pressure"):
                memory.assert_system_memory_available()


def test_missing_measurement_stops_work() -> None:
    with patch.object(memory, "system_memory_snapshot", return_value=None):
        with pytest.raises(MemoryBudgetError, match="measurement unavailable"):
            memory.assert_system_memory_available()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_settings(value: float) -> None:
    with pytest.raises(ValueError):
        memory.assert_system_memory_available(minimum_available_gib=value)


@pytest.mark.parametrize("text", ["MemTotal: 20 kB\n", "MemTotal: x kB\nMemAvailable: 3 kB",
    "MemTotal: 20 bytes\nMemAvailable: 3 kB", "MemTotal: 2 kB\nMemAvailable: 3 kB",
    "MemTotal: 0 kB\nMemAvailable: 0 kB"])
def test_bad_linux_measurements(text: str) -> None:
    with patch.object(memory.os, "name", "posix"), patch.object(memory, "Path") as path:
        path.return_value.read_text.return_value = text
        assert memory.system_memory_snapshot() is None


def test_linux_uses_available_not_free_or_swap() -> None:
    with patch.object(memory.os, "name", "posix"), patch.object(memory, "Path") as path:
        path.return_value.read_text.return_value = "MemTotal: 20 kB\nMemAvailable: 8 kB\nMemFree: 1 kB\nSwapFree: 90 kB"
        assert memory.system_memory_snapshot() == memory.SystemMemory(20480, 8192)


@pytest.mark.parametrize("succeeds", [False, True])
def test_windows_api(succeeds: bool) -> None:
    class Probe:
        def __call__(self, pointer: object) -> int:
            status = ctypes.cast(pointer, ctypes.POINTER(memory._MemoryStatus)).contents
            assert status.length == ctypes.sizeof(memory._MemoryStatus)
            status.total_physical, status.available_physical = 16 * GIB, 5 * GIB
            return int(succeeds)
    dll = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=Probe()))
    with patch.object(memory.os, "name", "nt"), patch.dict(memory.ctypes.__dict__, {"windll": dll}):
        assert memory.system_memory_snapshot() == (memory.SystemMemory(16 * GIB, 5 * GIB) if succeeds else None)
