from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from market_predictor.canonical.cutoffs import SWING_NIGHTLY_CUTOFF
from market_predictor.feature_store import LiveFeatureStore, LiveFeatureStoreConfig
from market_predictor.live_features import live_feature_columns, select_and_audit_live_features
from market_predictor.swing.contracts import SWING_FEATURE_SCHEMA_VERSION


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

    def test_live_audit_uses_frozen_cutoff_for_features_and_source_freshness(self) -> None:
        frame = _frame()
        cutoff = pd.Timestamp("2026-07-10T22:00:00Z")
        frame["decision_time_utc"] = cutoff
        frame["feature_available_at_utc"] = pd.Timestamp("2026-07-10T21:31:00Z")
        frame["bar_available_at_utc"] = pd.Timestamp("2026-07-10T20:15:00Z")
        frame["prediction_cutoff_policy_id"] = SWING_NIGHTLY_CUTOFF.policy_id
        frame["decision_group_id"] = cutoff.isoformat()
        frame["session_date_et"] = pd.Timestamp("2026-07-10").date()
        frame["feature_eligible"] = True
        frame["cross_section_eligible"] = True
        frame["daily_bar_count"] = 250
        frame["adjustment"] = "all"
        frame["swing_feature_schema_version"] = SWING_FEATURE_SCHEMA_VERSION
        frame["source_status_alpaca"] = "observed"
        frame["source_status_available_at_utc_alpaca"] = pd.Timestamp("2026-07-10T21:31:00Z")
        frame["source_coverage_end_utc_alpaca"] = pd.Timestamp("2026-07-10T21:30:00Z")

        selected, audit = select_and_audit_live_features(
            frame,
            mode="swing",
            required_price_feed="sip",
            required_adjustment="all",
            minimum_bar_count=250,
            minimum_cross_section=2,
            source_coverage_max_age_minutes=45,
            required_global_sources=(),
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(audit.passed, msg=audit.to_frame().to_string(index=False))

        after_cutoff = frame.copy()
        after_cutoff["source_status_available_at_utc_alpaca"] = cutoff + pd.Timedelta(minutes=1)
        _, rejected = select_and_audit_live_features(
            after_cutoff,
            mode="swing",
            required_price_feed="sip",
            required_adjustment="all",
            minimum_bar_count=250,
            minimum_cross_section=2,
            source_coverage_max_age_minutes=45,
            required_global_sources=(),
        )
        checks = {check.name: check for check in rejected.checks}
        self.assertEqual(checks["swing_live_no_future_features"].status, "fail")
        self.assertEqual(checks["swing_live_sources"].status, "fail")

        wrong_policy = frame.copy()
        wrong_policy["prediction_cutoff_policy_id"] = "unreviewed-policy"
        _, rejected_policy = select_and_audit_live_features(
            wrong_policy,
            mode="swing",
            required_price_feed="sip",
            required_adjustment="all",
            minimum_bar_count=250,
            minimum_cross_section=2,
            source_coverage_max_age_minutes=45,
            required_global_sources=(),
        )
        policy_check = next(check for check in rejected_policy.checks if check.name == "swing_live_cutoff_contract")
        self.assertEqual(policy_check.status, "fail")

    def test_live_swing_audit_rejects_missing_cutoff_identity(self) -> None:
        frame = _frame()
        frame["session_date_et"] = pd.Timestamp("2026-07-10").date()
        frame["feature_eligible"] = True
        frame["cross_section_eligible"] = True
        frame["daily_bar_count"] = 250
        frame["adjustment"] = "all"
        frame["swing_feature_schema_version"] = SWING_FEATURE_SCHEMA_VERSION

        selected, audit = select_and_audit_live_features(
            frame,
            mode="swing",
            required_price_feed="sip",
            required_adjustment="all",
            minimum_bar_count=250,
            minimum_cross_section=2,
            source_coverage_max_age_minutes=45,
            required_global_sources=(),
        )

        self.assertTrue(selected.empty)
        schema_check = next(check for check in audit.checks if check.name == "swing_live_schema")
        self.assertEqual(schema_check.status, "fail")
        self.assertIn("bar_available_at_utc", schema_check.detail)
        self.assertIn("prediction_cutoff_policy_id", schema_check.detail)


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
    missing = {column: pd.Series(0.0, index=frame.index) for column in live_feature_columns("swing") if column not in frame}
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
