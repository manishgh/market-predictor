"""R3: portable file lock + concurrency-safe quota reservation."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from market_predictor.locking import LockTimeout, file_lock
from market_predictor.quota import MonthlyQuotaTracker


class FileLockTest(unittest.TestCase):
    def test_lock_is_exclusive_then_reacquirable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "artifact.bin"
            with file_lock(target):
                with self.assertRaises(LockTimeout):
                    with file_lock(target, timeout=0.1):
                        pass
            # Once released the lock can be re-acquired.
            with file_lock(target, timeout=0.5):
                pass


class QuotaReserveTest(unittest.TestCase):
    def test_reserve_enforces_the_limit(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tracker = MonthlyQuotaTracker(Path(temp_dir) / "quota.json", "seeking_alpha", monthly_limit=2)
            tracker.reserve()
            tracker.reserve()
            with self.assertRaises(RuntimeError):
                tracker.reserve()
            self.assertEqual(tracker.status().used, 2)

    def test_reserve_persists_across_instances(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "quota.json"
            MonthlyQuotaTracker(path, "seeking_alpha", monthly_limit=5).reserve()
            self.assertEqual(MonthlyQuotaTracker(path, "seeking_alpha", monthly_limit=5).status().used, 1)

    def test_record_headers_does_not_consume_quota(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tracker = MonthlyQuotaTracker(Path(temp_dir) / "quota.json", "seeking_alpha", monthly_limit=5)
            tracker.reserve()
            tracker.record_headers({"x-ratelimit-remaining": "4", "irrelevant": "x"})
            status = tracker.status()
            self.assertEqual(status.used, 1)
            self.assertEqual(status.last_headers, {"x-ratelimit-remaining": "4"})


if __name__ == "__main__":
    unittest.main()
