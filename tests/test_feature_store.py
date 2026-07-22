from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from market_predictor.feature_store import LiveFeatureStore, LiveFeatureStoreConfig
from market_predictor.live_features import live_feature_columns


class LiveFeatureStoreTests(unittest.TestCase):
    def test_publishes_and_loads_integrity_checked_registered_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LiveFeatureStore(root)
            generated = datetime(2026, 7, 10, 21, 0, tzinfo=UTC)
            frame = _frame()

            manifest = _publish(store, frame, generated)
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
            generated = datetime(2026, 7, 10, 21, 0, tzinfo=UTC)
            _publish(store, _frame(), generated)

            with self.assertRaisesRegex(ValueError, "is stale"):
                store.load("swing", as_of=generated + timedelta(hours=3))

    def test_rejects_modified_feature_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LiveFeatureStore(root)
            generated = datetime(2026, 7, 10, 21, 0, tzinfo=UTC)
            _publish(store, _frame(), generated)
            path = root / "data/live/features/swing.parquet"
            modified = _frame().assign(close=999.0)
            modified.to_parquet(path, index=False)

            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                store.load("swing", as_of=generated + timedelta(hours=1))

    def test_rejects_stale_feature_rows_at_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LiveFeatureStore(root)
            generated = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
            with self.assertRaisesRegex(ValueError, "contains stale rows"):
                _publish(store, _frame(), generated)

    def test_rejects_one_stale_row_and_any_future_derived_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LiveFeatureStore(Path(tmp))
            generated = datetime(2026, 7, 10, 21, 0, tzinfo=UTC)
            mixed_freshness = _frame()
            mixed_freshness.loc[0, "feature_available_at_utc"] = pd.Timestamp("2026-07-01T20:00:00Z")
            with self.assertRaisesRegex(ValueError, "contains stale rows"):
                _publish(store, mixed_freshness, generated)

            with self.assertRaisesRegex(ValueError, "labels or future paths"):
                _publish(store, _frame().assign(target_net_positive_5d=1), generated)


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ticker": ["MSFT", "AAPL"],
            "date": ["2026-07-10", "2026-07-10"],
            "decision_time_utc": [
                pd.Timestamp("2026-07-10T20:00:00Z"),
                pd.Timestamp("2026-07-10T20:00:00Z"),
            ],
            "feature_available_at_utc": [
                pd.Timestamp("2026-07-10T20:00:00Z"),
                pd.Timestamp("2026-07-10T20:00:00Z"),
            ],
            "price_feed": ["sip", "sip"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
        }
    )
    missing = {
        column: pd.Series(0.0, index=frame.index)
        for column in live_feature_columns("swing")
        if column not in frame
    }
    return pd.concat([frame, pd.DataFrame(missing)], axis=1)


def _publish(
    store: LiveFeatureStore,
    frame: pd.DataFrame,
    generated: datetime,
) -> dict[str, object]:
    return store.publish(
        "swing",
        frame,
        price_feed="sip",
        feature_schema_version="swing.features.v1",
        source_artifact_sha256="a" * 64,
        source_artifact_type="swing_inference_features",
        generated_at=generated,
    )


if __name__ == "__main__":
    unittest.main()
