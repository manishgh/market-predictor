from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pydantic import ValidationError

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_canonical_bars,
    audit_canonical_events,
    audit_decision_availability,
    audit_decision_source_coverage,
    audit_source_collections,
    audit_universe_memberships,
)
from market_predictor.canonical.contracts import (
    CanonicalBar,
    CanonicalEvent,
    CanonicalUniverseMembership,
    SourceCollection,
)
from market_predictor.canonical.joins import (
    aggregate_event_features,
    decisions_from_completed_bars,
    join_fundamentals_asof,
    join_source_collection_status,
    join_universe_membership,
)
from market_predictor.canonical.normalize import (
    canonicalize_bars,
    canonicalize_events,
    canonicalize_universe_memberships,
)
from market_predictor.canonical.store import load_canonical_artifact, write_canonical_artifact
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError


class CanonicalContractTests(unittest.TestCase):
    def test_bar_cannot_be_available_before_left_edge_interval_ends(self) -> None:
        start = datetime(2026, 7, 21, 13, 30, tzinfo=UTC)
        with self.assertRaises(ValidationError):
            CanonicalBar(
                ticker="MSFT",
                timeframe="5m",
                bar_start_utc=start,
                bar_end_utc=datetime(2026, 7, 21, 13, 35, tzinfo=UTC),
                available_at_utc=start,
                ingested_at_utc=datetime(2026, 7, 21, 13, 36, tzinfo=UTC),
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=1_000,
                source="alpaca",
                price_feed="sip",
                adjustment="all",
                availability_policy="market_interval_close",
            )

    def test_event_requires_sentiment_scoring_availability(self) -> None:
        published = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        with self.assertRaises(ValidationError):
            CanonicalEvent(
                event_id="a" * 64,
                ticker="MSFT",
                source_family="alpaca",
                source="alpaca:benzinga",
                published_at_utc=published,
                first_seen_at_utc=published,
                available_at_utc=published,
                feature_available_at_utc=published,
                title="Microsoft announces an update",
                sentiment_numeric=0.5,
                availability_policy="observed",
                raw_sha256="b" * 64,
            )

    def test_source_empty_is_distinct_from_failed(self) -> None:
        now = datetime.now(UTC)
        observed_empty = SourceCollection(
            collection_id="collection-1",
            ticker="MSFT",
            source_family="reddit",
            requested_start_utc=now,
            requested_end_utc=now,
            started_at_utc=now,
            completed_at_utc=now,
            status="observed_empty",
            row_count=0,
        )
        self.assertEqual(observed_empty.status, "observed_empty")
        with self.assertRaises(ValidationError):
            SourceCollection(**{**observed_empty.model_dump(), "status": "failed"})

    def test_membership_requires_explicit_availability(self) -> None:
        raw = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "effective_from_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
                "effective_to_utc": [pd.NaT],
                "sector": ["Technology"],
                "industry": ["Software"],
                "market_cap_bucket": ["mega"],
                "liquidity_bucket": ["high"],
                "primary_benchmark": ["XLK"],
                "universe_snapshot_id": ["snapshot-1"],
            }
        )
        with self.assertRaises(SchemaMismatchError):
            canonicalize_universe_memberships(raw, source="finviz", availability_policy="observed")


class CanonicalNormalizationTests(unittest.TestCase):
    def test_intraday_bar_is_available_only_after_interval_and_finalization_delay(self) -> None:
        bars = pd.DataFrame(
            {
                "symbol": ["MSFT"],
                "timeframe": ["5m"],
                "timestamp": [pd.Timestamp("2026-07-21T13:30:00Z")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1_000],
                "price_feed": ["sip"],
                "ingested_at_utc": [pd.Timestamp("2026-07-21T14:00:00Z")],
            }
        )
        canonical = canonicalize_bars(bars)
        row = canonical.iloc[0]
        self.assertEqual(row["bar_start_utc"], pd.Timestamp("2026-07-21T13:30:00Z"))
        self.assertEqual(row["bar_end_utc"], pd.Timestamp("2026-07-21T13:35:00Z"))
        self.assertEqual(row["available_at_utc"], pd.Timestamp("2026-07-21T13:35:30Z"))

    def test_daily_bar_uses_actual_early_market_close(self) -> None:
        bars = pd.DataFrame(
            {
                "symbol": ["MSFT"],
                "timeframe": ["1d"],
                "timestamp": [pd.Timestamp("2025-11-28T05:00:00Z")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1_000],
                "price_feed": ["sip"],
                "ingested_at_utc": [pd.Timestamp("2025-11-29T00:00:00Z")],
            }
        )
        canonical = canonicalize_bars(bars)
        self.assertEqual(canonical.iloc[0]["bar_end_utc"], pd.Timestamp("2025-11-28T18:00:00Z"))
        self.assertEqual(canonical.iloc[0]["available_at_utc"], pd.Timestamp("2025-11-28T18:15:00Z"))

    def test_observed_event_uses_provider_update_first_seen_and_score_times(self) -> None:
        events = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "timestamp": [pd.Timestamp("2026-07-21T12:05:00Z")],
                "source": ["alpaca:benzinga"],
                "title": ["Microsoft announces an update"],
                "url": ["https://example.test/news/1"],
                "summary": ["Summary"],
                "text": ["Text"],
                "sentiment_numeric": [0.7],
                "raw": [
                    {
                        "id": 123,
                        "created_at": "2026-07-21T12:00:00Z",
                        "updated_at": "2026-07-21T12:05:00Z",
                    }
                ],
            }
        )
        canonical = canonicalize_events(
            events,
            collected_at_utc=pd.Timestamp("2026-07-21T12:10:00Z"),
            sentiment_scored_at_utc=pd.Timestamp("2026-07-21T12:11:00Z"),
        )
        row = canonical.iloc[0]
        self.assertEqual(row["published_at_utc"], pd.Timestamp("2026-07-21T12:00:00Z"))
        self.assertEqual(row["provider_updated_at_utc"], pd.Timestamp("2026-07-21T12:05:00Z"))
        self.assertEqual(row["available_at_utc"], pd.Timestamp("2026-07-21T12:10:00Z"))
        self.assertEqual(row["feature_available_at_utc"], pd.Timestamp("2026-07-21T12:11:00Z"))

    def test_historical_event_proxy_is_explicit_and_cannot_hide_missing_score_time(self) -> None:
        events = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "timestamp": [pd.Timestamp("2026-07-21T12:00:00Z")],
                "source": ["finviz"],
                "title": ["Microsoft headline"],
                "sentiment_numeric": [0.4],
            }
        )
        with self.assertRaises(DataReadinessError):
            canonicalize_events(
                events,
                collected_at_utc=pd.Timestamp("2026-07-21T13:00:00Z"),
                availability_policy="provider_publication_proxy",
            )


class CanonicalJoinAndAuditTests(unittest.TestCase):
    def test_empty_production_artifacts_fail_readiness(self) -> None:
        empty_bars = pd.DataFrame(columns=CanonicalBar.model_fields)
        bar_checks = audit_canonical_bars(empty_bars)
        self.assertEqual(next(check for check in bar_checks if check.name == "bar_rows").status, "fail")

        empty_decisions = pd.DataFrame(columns=["decision_time_utc", "feature_available_at_utc"])
        decision_checks = audit_decision_availability(
            empty_decisions,
            feature_timestamp_columns=["feature_available_at_utc"],
        )
        self.assertEqual(next(check for check in decision_checks if check.name == "decision_rows").status, "fail")

    def test_event_join_waits_for_ingestion_and_sentiment_scoring(self) -> None:
        decisions = pd.DataFrame(
            {
                "ticker": ["MSFT", "MSFT"],
                "decision_time_utc": [
                    pd.Timestamp("2026-07-21T10:05:00Z"),
                    pd.Timestamp("2026-07-21T10:10:00Z"),
                ],
            }
        )
        event = self._canonical_event(
            published="2026-07-21T09:00:00Z",
            first_seen="2026-07-21T10:06:00Z",
            scored="2026-07-21T10:07:00Z",
        )
        joined = aggregate_event_features(decisions, event)
        self.assertEqual(joined["event_count_2h"].tolist(), [0, 1])
        self.assertEqual(joined["sentiment_mean_2h"].tolist(), [0.0, 0.8])
        self.assertTrue(
            joined.loc[1, "latest_event_feature_available_at_utc"] <= joined.loc[1, "decision_time_utc"]
        )

    def test_proxy_events_fail_production_join_and_audit(self) -> None:
        decisions = pd.DataFrame(
            {"ticker": ["MSFT"], "decision_time_utc": [pd.Timestamp("2026-07-21T10:10:00Z")]}
        )
        event = self._canonical_event(
            published="2026-07-21T09:00:00Z",
            first_seen="2026-07-21T12:00:00Z",
            scored="2026-07-21T12:01:00Z",
            policy="provider_publication_proxy",
        )
        with self.assertRaises(DataReadinessError):
            aggregate_event_features(decisions, event)
        checks = audit_canonical_events(event, require_observed=True)
        self.assertEqual(next(check for check in checks if check.name == "event_observed_history").status, "fail")

    def test_source_status_distinguishes_not_collected_empty_and_failed(self) -> None:
        decisions = pd.DataFrame(
            {
                "ticker": ["MSFT", "MSFT"],
                "decision_time_utc": [
                    pd.Timestamp("2026-07-21T10:00:00Z"),
                    pd.Timestamp("2026-07-21T11:00:00Z"),
                ],
            }
        )
        collections = pd.DataFrame(
            [
                SourceCollection(
                    collection_id="reddit-1",
                    ticker="MSFT",
                    source_family="reddit",
                    requested_start_utc=datetime(2026, 7, 20, tzinfo=UTC),
                    requested_end_utc=datetime(2026, 7, 21, 10, 0, tzinfo=UTC),
                    started_at_utc=datetime(2026, 7, 21, 10, 5, tzinfo=UTC),
                    completed_at_utc=datetime(2026, 7, 21, 10, 6, tzinfo=UTC),
                    status="observed_empty",
                    row_count=0,
                ).model_dump()
            ]
        )
        joined = join_source_collection_status(decisions, collections, source_families=["reddit"])
        self.assertEqual(joined["source_status_reddit"].tolist(), ["not_collected", "observed_empty"])
        self.assertTrue(all(check.status == "pass" for check in audit_source_collections(collections)))
        coverage = audit_decision_source_coverage(joined, required_sources=["reddit"])
        self.assertEqual(coverage[0].status, "fail")
        self.assertEqual(coverage[0].failures, 1)

    def test_fundamental_join_uses_fact_availability_not_current_snapshot(self) -> None:
        decisions = pd.DataFrame(
            {
                "ticker": ["MSFT", "MSFT"],
                "decision_time_utc": [
                    pd.Timestamp("2026-04-01T14:00:00Z"),
                    pd.Timestamp("2026-05-01T14:00:00Z"),
                ],
            }
        )
        facts = pd.DataFrame(
            {
                "fact_id": ["old-fact-00000001", "new-fact-00000001"],
                "ticker": ["MSFT", "MSFT"],
                "metric": ["revenue", "revenue"],
                "value": [100.0, 120.0],
                "available_at_utc": [
                    pd.Timestamp("2026-02-01T12:00:00Z"),
                    pd.Timestamp("2026-04-20T12:00:00Z"),
                ],
                "availability_policy": ["observed", "observed"],
            }
        )
        joined = join_fundamentals_asof(decisions, facts, metrics=["revenue"])
        self.assertEqual(joined["fundamental_revenue"].tolist(), [100.0, 120.0])
        self.assertTrue(
            (joined["fundamental_available_at_utc_revenue"] <= joined["decision_time_utc"]).all()
        )

    def test_membership_join_rejects_hindsight_before_snapshot_was_known(self) -> None:
        membership = CanonicalUniverseMembership(
            ticker="MSFT",
            effective_from_utc=datetime(2026, 1, 1, tzinfo=UTC),
            available_at_utc=datetime(2026, 7, 1, tzinfo=UTC),
            sector="Technology",
            industry="Software",
            market_cap_bucket="mega",
            liquidity_bucket="high",
            primary_benchmark="XLK",
            universe_snapshot_id="snapshot-1",
            source="finviz",
            availability_policy="observed",
        )
        memberships = pd.DataFrame([membership.model_dump()])
        decisions = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "decision_time_utc": [pd.Timestamp("2026-06-01T14:00:00Z")],
            }
        )
        checks = audit_universe_memberships(memberships, decisions=decisions)
        coverage = next(check for check in checks if check.name == "universe_membership_coverage")
        self.assertEqual(coverage.status, "fail")
        with self.assertRaises(DataReadinessError):
            join_universe_membership(decisions, memberships)

        decisions["decision_time_utc"] = pd.Timestamp("2026-07-02T14:00:00Z")
        joined = join_universe_membership(decisions, memberships)
        self.assertEqual(joined.loc[0, "primary_benchmark"], "XLK")
        self.assertLessEqual(joined.loc[0, "membership_available_at_utc"], joined.loc[0, "decision_time_utc"])

    def test_membership_audit_rejects_overlapping_windows(self) -> None:
        base = {
            "ticker": "MSFT",
            "available_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
            "sector": "Technology",
            "industry": "Software",
            "market_cap_bucket": "mega",
            "liquidity_bucket": "high",
            "primary_benchmark": "XLK",
            "source": "finviz",
            "availability_policy": "observed",
        }
        memberships = pd.DataFrame(
            [
                CanonicalUniverseMembership(
                    **base,
                    effective_from_utc=datetime(2026, 1, 1, tzinfo=UTC),
                    effective_to_utc=datetime(2026, 7, 1, tzinfo=UTC),
                    universe_snapshot_id="snapshot-1",
                ).model_dump(),
                CanonicalUniverseMembership(
                    **base,
                    effective_from_utc=datetime(2026, 6, 1, tzinfo=UTC),
                    effective_to_utc=None,
                    universe_snapshot_id="snapshot-2",
                ).model_dump(),
            ]
        )
        checks = audit_universe_memberships(memberships)
        windows = next(check for check in checks if check.name == "universe_membership_windows")
        self.assertEqual(windows.status, "fail")

    def test_audit_rejects_partial_feed_and_future_feature_join(self) -> None:
        raw = pd.DataFrame(
            {
                "symbol": ["MSFT"],
                "timeframe": ["5m"],
                "timestamp": [pd.Timestamp("2026-07-21T13:30:00Z")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1_000],
                "price_feed": ["iex"],
                "ingested_at_utc": [pd.Timestamp("2026-07-21T14:00:00Z")],
            }
        )
        bars = canonicalize_bars(raw)
        self.assertEqual(next(check for check in audit_canonical_bars(bars) if check.name == "bar_price_feed").status, "fail")
        decisions = decisions_from_completed_bars(bars)
        decisions["event_available_at_utc"] = decisions["decision_time_utc"] + pd.Timedelta(seconds=1)
        availability = audit_decision_availability(
            decisions,
            feature_timestamp_columns=["feature_available_at_utc", "event_available_at_utc"],
        )
        self.assertEqual(next(check for check in availability if check.name == "decision_no_future_features").status, "fail")

    def test_canonical_artifact_manifest_is_written_last_and_detects_tampering(self) -> None:
        raw = pd.DataFrame(
            {
                "symbol": ["MSFT"],
                "timeframe": ["5m"],
                "timestamp": [pd.Timestamp("2026-07-21T13:30:00Z")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1_000],
                "price_feed": ["sip"],
                "ingested_at_utc": [pd.Timestamp("2026-07-21T14:00:00Z")],
            }
        )
        bars = canonicalize_bars(raw)
        audit = CanonicalAuditReport(checks=audit_canonical_bars(bars))
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bars.parquet"
            manifest = write_canonical_artifact(bars, path, artifact_type="bars", audit=audit)
            loaded, loaded_manifest = load_canonical_artifact(path, expected_type="bars")
            pd.testing.assert_frame_equal(loaded, bars)
            self.assertEqual(loaded_manifest["artifact_sha256"], manifest["artifact_sha256"])
            changed = bars.copy()
            changed.loc[0, "close"] = 100.75
            changed.to_parquet(path, index=False)
            with self.assertRaises(DataReadinessError):
                load_canonical_artifact(path, expected_type="bars")

    @staticmethod
    def _canonical_event(
        *,
        published: str,
        first_seen: str,
        scored: str,
        policy: str = "observed",
    ) -> pd.DataFrame:
        available = pd.Timestamp(first_seen) if policy == "observed" else pd.Timestamp(published)
        feature_available = max(available, pd.Timestamp(scored))
        return pd.DataFrame(
            {
                "event_id": ["a" * 64],
                "ticker": ["MSFT"],
                "source_family": ["alpaca"],
                "source": ["alpaca:benzinga"],
                "published_at_utc": [pd.Timestamp(published)],
                "provider_updated_at_utc": [pd.NaT],
                "first_seen_at_utc": [pd.Timestamp(first_seen)],
                "available_at_utc": [available],
                "sentiment_scored_at_utc": [pd.Timestamp(scored)],
                "feature_available_at_utc": [feature_available],
                "title": ["Microsoft announces an update"],
                "url": [""],
                "summary": [""],
                "text": [""],
                "sentiment_numeric": [0.8],
                "relevance": [1.0],
                "availability_policy": [policy],
                "raw_sha256": ["b" * 64],
                "schema_version": ["market_data.v1"],
            }
        )


if __name__ == "__main__":
    unittest.main()
