from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from market_predictor.core.errors import DataReadinessError
from market_predictor.process_memory import process_memory_snapshot
from market_predictor.resources import assert_peak_memory_budget


class ProcessMemoryTests(unittest.TestCase):
    @patch(
        "market_predictor.resources.process_memory_snapshot",
        return_value=(1 * 1024**3, 4 * 1024**3),
    )
    def test_peak_guard_rejects_transient_budget_breach(
        self,
        _snapshot: object,
    ) -> None:
        with self.assertRaisesRegex(
            DataReadinessError,
            "peak RSS 4.00 GiB",
        ):
            assert_peak_memory_budget(
                hard_budget_gib=4.0,
                headroom_gib=0.75,
                stage="test",
            )

    def test_concurrent_snapshots_use_stable_native_types(self) -> None:
        initial = process_memory_snapshot()
        if initial is None:
            self.skipTest("process memory is not available on this platform")

        with ThreadPoolExecutor(max_workers=8) as pool:
            snapshots = list(pool.map(lambda _: process_memory_snapshot(), range(64)))

        self.assertTrue(all(snapshot is not None for snapshot in snapshots))
        for snapshot in snapshots:
            assert snapshot is not None
            working_set, peak_working_set = snapshot
            self.assertGreater(working_set, 0)
            self.assertGreaterEqual(peak_working_set, working_set)


if __name__ == "__main__":
    unittest.main()


def test_release_unused_arrow_precedes_native_trim(monkeypatch):
    import sys
    from types import SimpleNamespace

    from market_predictor import process_memory as memory

    events = []
    def current_process():
        return 42
    def trim(handle):
        assert handle == 42
        events.append("trim")
        return 1
    monkeypatch.setattr(memory.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(memory, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(memory, "_windows_dlls", lambda: (SimpleNamespace(GetCurrentProcess=current_process),
        SimpleNamespace(EmptyWorkingSet=trim)))
    pool = SimpleNamespace(release_unused=lambda: events.append("arrow"))
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(default_memory_pool=lambda: pool))
    memory.release_process_memory()
    assert events == ["gc", "arrow", "trim"]


def test_arrow_release_failure_still_trims_and_does_not_relax_guard(monkeypatch):
    import sys
    from types import SimpleNamespace

    import pytest

    from market_predictor import process_memory as memory
    from market_predictor import resources
    from market_predictor.core.errors import MemoryBudgetError

    events = []
    def current_process():
        return 42
    def trim(handle):
        events.append("trim")
        return 1
    def release():
        raise RuntimeError("allocator cleanup unavailable")
    monkeypatch.setattr(memory.gc, "collect", lambda: None)
    monkeypatch.setattr(memory, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(memory, "_windows_dlls", lambda: (SimpleNamespace(GetCurrentProcess=current_process),
        SimpleNamespace(EmptyWorkingSet=trim)))
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(default_memory_pool=lambda: SimpleNamespace(release_unused=release)))
    memory.release_process_memory()
    assert events == ["trim"]
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: (6 * 1024**3, 6 * 1024**3))
    with pytest.raises(MemoryBudgetError, match="exceeds"):
        resources.assert_memory_budget(hard_budget_gib=5, headroom_gib=0.75, stage="test")


def test_release_without_arrow_does_not_import_it(monkeypatch):
    import sys
    from types import SimpleNamespace

    from market_predictor import process_memory as memory
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.setattr(memory, "os", SimpleNamespace(name="posix"))
    memory.release_process_memory()
    assert "pyarrow" not in sys.modules


def test_releasing_unused_arrow_preserves_live_array():
    import pyarrow as pa

    from market_predictor.process_memory import release_process_memory
    values = pa.array([1.0, None, 3.0])
    release_process_memory()
    assert values.to_pylist() == [1.0, None, 3.0]


def test_current_rss_guard_measures_before_cleanup_without_native_trim(monkeypatch):
    from market_predictor import process_memory as memory
    from market_predictor import resources
    events = []
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: (events.append("measure") or (1024**3, 1024**3)))
    monkeypatch.setattr(memory.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(memory, "_windows_dlls", lambda: (_ for _ in ()).throw(AssertionError("native trim forbidden")))
    resources.assert_memory_budget(hard_budget_gib=5, headroom_gib=0.75, stage="test")
    assert events == ["measure", "gc"]


def test_current_rss_breach_fails_before_cleanup(monkeypatch):
    import pytest

    from market_predictor import resources
    from market_predictor.core.errors import MemoryBudgetError
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: (6 * 1024**3, 6 * 1024**3))
    def forbidden_cleanup():
        raise AssertionError("cleanup must not hide breach")
    monkeypatch.setattr(resources, "release_unused_process_memory", forbidden_cleanup)
    with pytest.raises(MemoryBudgetError, match="exceeds"):
        resources.assert_memory_budget(hard_budget_gib=5, headroom_gib=0.75, stage="test")


def test_unused_cleanup_cannot_authorize_unresolved_system_pressure(monkeypatch):
    import pytest

    from market_predictor import resources
    from market_predictor.core import system_memory
    from market_predictor.core.errors import MemoryBudgetError
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: (1024**3, 1024**3))
    monkeypatch.setattr(system_memory, "system_memory_snapshot", lambda: system_memory.SystemMemory(16 * 1024**3, 1024**3))
    resources.assert_memory_budget(hard_budget_gib=5, headroom_gib=0.75, stage="test")
    with pytest.raises(MemoryBudgetError, match="system memory pressure"):
        system_memory.assert_system_memory_available()
