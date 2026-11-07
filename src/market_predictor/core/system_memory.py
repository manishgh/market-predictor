"""Physical-memory headroom, independent of the current process working set."""
from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_predictor.core.errors import MemoryBudgetError

MAXIMUM_SYSTEM_USED_PERCENT = 90.0


@dataclass(frozen=True)
class SystemMemory:
    total_bytes: int
    available_bytes: int

    @property
    def used_percent(self) -> float:
        return 100.0 * (1.0 - self.available_bytes / self.total_bytes)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
        *[(name, ctypes.c_uint64) for name in (
            "total_physical", "available_physical", "total_pagefile", "available_pagefile",
            "total_virtual", "available_virtual", "available_extended")]]


def system_memory_snapshot() -> SystemMemory | None:
    try:
        if os.name == "nt":
            status = _MemoryStatus()
            status.length = ctypes.sizeof(status)
            windll: Any = ctypes.__dict__["windll"]
            probe = windll.kernel32.GlobalMemoryStatusEx
            probe.argtypes = [ctypes.POINTER(_MemoryStatus)]
            probe.restype = ctypes.c_int
            if not probe(ctypes.byref(status)):
                return None
            total, available = int(status.total_physical), int(status.available_physical)
        else:
            fields = {}
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                name, _, raw = line.partition(":")
                if name in {"MemTotal", "MemAvailable"}:
                    value, unit = raw.split()
                    if unit != "kB":
                        return None
                    fields[name] = int(value) * 1024
            total, available = fields["MemTotal"], fields["MemAvailable"]
        if total <= 0 or not 0 <= available <= total:
            return None
        return SystemMemory(total, available)
    except (OSError, ValueError, KeyError, AttributeError):
        return None


def validate_system_memory_snapshot(snapshot: SystemMemory | None) -> SystemMemory:
    """Reject unknown or incoherent measurements before applying any threshold."""
    if (snapshot is None or type(snapshot.total_bytes) is not int or type(snapshot.available_bytes) is not int
            or snapshot.total_bytes <= 0 or not 0 <= snapshot.available_bytes <= snapshot.total_bytes
            or not math.isfinite(snapshot.used_percent) or not 0 <= snapshot.used_percent <= 100):
        raise MemoryBudgetError("system memory measurement unavailable; no new data request permitted")
    return snapshot


def assert_system_memory_available(*, minimum_available_gib: float = 0.0,
    maximum_used_percent: float = MAXIMUM_SYSTEM_USED_PERCENT) -> SystemMemory:
    """Percentage-only by default; explicitly stricter caller limits remain supported."""
    if (not math.isfinite(minimum_available_gib) or minimum_available_gib < 0
            or not math.isfinite(maximum_used_percent) or not 0 < maximum_used_percent <= MAXIMUM_SYSTEM_USED_PERCENT):
        raise ValueError("invalid system memory thresholds")
    snapshot = validate_system_memory_snapshot(system_memory_snapshot())
    if (snapshot.available_bytes < minimum_available_gib * 1024**3
            or snapshot.used_percent >= maximum_used_percent):
        floor = f" and at least {minimum_available_gib:.2f} GiB free" if minimum_available_gib > 0 else ""
        raise MemoryBudgetError(f"system memory pressure: {snapshot.used_percent:.1f}% used, "
            f"{snapshot.available_bytes / 1024**3:.2f} GiB available; "
            f"requires strictly below {maximum_used_percent:.1f}%{floor}")
    return snapshot
