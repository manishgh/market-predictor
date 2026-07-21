from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from market_predictor.feature_store import LiveFeatureStore, LiveFeatureStoreConfig


class LiveFeatureStoreTests(unittest.TestCase):
    def test_publishes_and_loads_integrity_checked_registered_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LiveFeatureStore(root)
            generated = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
            frame = _frame()

            manifest = store.publish("swing", frame, price_feed="sip", generated_at=generated)
            loaded = store.load("swing", as_of=generated + timedelta(hours=1))

            self.assertEqual(manifest["rows"], len(frame))
            self.assertEqual(len(loaded), len(frame))
            self.assertTrue(loaded["price_feed"].eq("sip").all())
            self.assertTrue(loaded["stale_cache"].eq(False).all())

    def test_rejects_stale_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = LiveFeatureStoreConfig(swing_max_age=timedelta(hours=2))
            store = LiveFeatureStore(root, config)
            generated = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
            store.publish("swing", _frame(), price_feed="sip", generated_at=generated)

            with self.assertRaisesRegex(ValueError, "is stale"):
                store.load("swing", as_of=generated + timedelta(hours=3))

    def test_rejects_modified_feature_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LiveFeatureStore(root)
            generated = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
            store.publish("swing", _frame(), price_feed="sip", generated_at=generated)
            path = root / "data/live/features/swing.parquet"
            modified = _frame().assign(close=999.0)
            modified.to_parquet(path, index=False)

            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                store.load("swing", as_of=generated + timedelta(hours=1))

    def test_rejects_freshly_published_snapshot_with_stale_feature_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LiveFeatureStore(root)
            generated = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
            store.publish("swing", _frame(), price_feed="sip", generated_at=generated)

            with self.assertRaisesRegex(ValueError, "feature rows are stale"):
                store.validate("swing", as_of=generated + timedelta(hours=1))


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["MSFT", "MSFT"],
            "date": ["2026-07-09", "2026-07-10"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
        }
    )


if __name__ == "__main__":
    unittest.main()
