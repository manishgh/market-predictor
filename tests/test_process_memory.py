from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from market_predictor.process_memory import process_memory_snapshot
from market_predictor.resources import assert_peak_memory_budget
from market_predictor.v3.errors import DataReadinessError


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
