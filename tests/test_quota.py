from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from market_predictor.quota import MonthlyQuotaTracker


class MonthlyQuotaTrackerTests(unittest.TestCase):
    def test_rejects_non_object_persisted_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "quota.json"
            path.write_text("[]", encoding="utf-8")
            tracker = MonthlyQuotaTracker(path, "seeking_alpha", 10_000)

            with self.assertRaisesRegex(ValueError, "quota state must contain a JSON object"):
                tracker.status()


if __name__ == "__main__":
    unittest.main()
