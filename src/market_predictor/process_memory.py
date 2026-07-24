"""Cross-platform process-memory primitives with no domain-layer dependencies."""

from __future__ import annotations

import ctypes
import gc
import os
from pathlib import Path
from typing import Any


def release_process_memory() -> None:
    gc.collect()
    if os.name != "nt":
        return
    kernel32, psapi = _windows_dlls()
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    empty_working_set = psapi.EmptyWorkingSet
    empty_working_set.argtypes = [ctypes.c_void_p]
    empty_working_set.restype = ctypes.c_int
    empty_working_set(get_current_process())


def process_memory_snapshot() -> tuple[int, int] | None:
    if os.name == "nt":
        return _windows_memory_snapshot()
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


def _windows_memory_snapshot() -> tuple[int, int] | None:
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
    kernel32, psapi = _windows_dlls()
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    if get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    return None


def _windows_dlls() -> tuple[Any, Any]:
    windll: Any = ctypes.__dict__["windll"]
    return windll.kernel32, windll.psapi
