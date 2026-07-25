from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease


class HeavyJobLeaseTests(unittest.TestCase):
    def test_lease_is_non_queueing_and_recovers_after_release(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            with heavy_job_lease(
                "first",
                runtime_dir=runtime,
            ) as owner:
                owner_path = runtime / "heavy-job.owner.json"
                self.assertTrue(owner_path.exists())
                persisted = json.loads(owner_path.read_text(encoding="utf-8"))
                self.assertEqual(persisted["run_id"], owner["run_id"])
                self.assertEqual(persisted["command"], "first")
                with self.assertRaises(HeavyJobBusyError):
                    with heavy_job_lease("second", runtime_dir=runtime):
                        self.fail("competing heavy job acquired the lease")

            self.assertFalse((runtime / "heavy-job.owner.json").exists())
            with heavy_job_lease(
                "third",
                runtime_dir=runtime,
            ) as recovered:
                self.assertEqual(recovered["command"], "third")


if __name__ == "__main__":
    unittest.main()
