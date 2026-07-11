from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_predictor.v3.errors import ArtifactIntegrityError, LeakageAuditError
from market_predictor.v3.partitions import assert_development_only, partition_development_shadow, write_shadow_partition


class V3PartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "ticker": ["MSFT", "MSFT"],
                "decision_time_utc": ["2026-07-08T20:00:00Z", "2026-07-09T14:30:00Z"],
                "value": [1.0, 2.0],
            }
        )

    def test_partition_cutoff_and_development_guard(self) -> None:
        development, shadow = partition_development_shadow(self.frame)
        self.assertEqual(len(development), 1)
        self.assertEqual(len(shadow), 1)
        assert_development_only(development)
        with self.assertRaises(LeakageAuditError):
            assert_development_only(self.frame)

    def test_shadow_partition_is_immutable(self) -> None:
        _, shadow = partition_development_shadow(self.frame)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shadow.parquet"
            manifest = write_shadow_partition(shadow, output)
            self.assertEqual(manifest["rows"], 1)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".manifest.json").exists())
            with self.assertRaises(ArtifactIntegrityError):
                write_shadow_partition(shadow, output)

    def test_partition_rejects_naive_timestamps(self) -> None:
        frame = self.frame.copy()
        frame["decision_time_utc"] = ["2026-07-08 20:00:00", "2026-07-09 14:30:00"]
        with self.assertRaises(LeakageAuditError):
            partition_development_shadow(frame)


if __name__ == "__main__":
    unittest.main()
