from __future__ import annotations

import ctypes
import gc
import os
from dataclasses import dataclass
from pathlib import Path

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


def release_process_memory() -> None:
    gc.collect()
    if os.name != "nt":
        return
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    empty_working_set = ctypes.windll.psapi.EmptyWorkingSet
    empty_working_set.argtypes = [ctypes.c_void_p]
    empty_working_set.restype = ctypes.c_int
    empty_working_set(get_current_process())


def process_memory_snapshot() -> tuple[int, int] | None:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        if get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
        return None
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    parts = statm.read_text(encoding="ascii").split()
    if len(parts) < 2:
        return None
    sysconf = os.__dict__.get("sysconf")
    if not callable(sysconf):
        return None
    rss = int(parts[1]) * int(sysconf("SC_PAGE_SIZE"))
    return rss, rss


def _gib(value: int) -> float:
    return value / 1024**3
