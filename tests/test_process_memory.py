from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from market_predictor.process_memory import process_memory_snapshot


class ProcessMemoryTests(unittest.TestCase):
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
