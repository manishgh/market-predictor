from __future__ import annotations

import unittest
from datetime import timedelta

import pandas as pd

from market_predictor.v3.audits import audit_bars, build_data_audit
from market_predictor.v3.errors import DataReadinessError


class V3AuditTests(unittest.TestCase):
    def test_complete_point_in_time_inputs_pass(self) -> None:
        bars = pd.DataFrame(
            {
                "ticker": ["MSFT", "MSFT"],
                "timestamp": ["2026-07-08T14:30:00Z", "2026-07-08T14:35:00Z"],
                "open": [500.0, 501.0],
                "high": [502.0, 503.0],
                "low": [499.0, 500.0],
                "close": [501.0, 502.0],
                "volume": [1000, 1200],
                "price_feed": ["sip", "sip"],
            }
        )
        events = pd.DataFrame(
            {
                "event_id": ["event-1"],
                "ticker": ["MSFT"],
                "published_at_utc": ["2026-07-08T13:00:00Z"],
                "ingested_at_utc": ["2026-07-08T13:00:05Z"],
                "source_family": ["alpaca"],
            }
        )
        decisions = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "decision_time_utc": ["2026-07-08T14:35:00Z"],
                "universe_snapshot_id": ["sp500-20260708"],
                "primary_benchmark": ["QQQ"],
                "benchmark_close": [620.0],
            }
        )
        memberships = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "effective_from_utc": ["2026-01-01T00:00:00Z"],
                "effective_to_utc": [None],
                "sector": ["Technology"],
                "industry": ["Software"],
                "market_cap_bucket": ["mega"],
                "liquidity_bucket": ["high"],
                "primary_benchmark": ["QQQ"],
                "universe_snapshot_id": ["sp500-20260708"],
                "schema_version": ["ml_v3.v1"],
            }
        )
        report = build_data_audit(bars=bars, events=events, decisions=decisions, memberships=memberships)
        self.assertTrue(report.passed, report.to_frame().to_dict(orient="records"))
        report.raise_for_failure()

    def test_gap_and_partial_feed_fail_strict_audit(self) -> None:
        bars = pd.DataFrame(
            {
                "ticker": ["RGTI", "RGTI"],
                "timestamp": ["2026-07-08T14:30:00Z", "2026-07-08T14:45:00Z"],
                "open": [10, 10],
                "high": [11, 11],
                "low": [9, 9],
                "close": [10, 10],
                "volume": [100, 100],
                "price_feed": ["iex", "iex"],
            }
        )
        checks = audit_bars(bars, interval=timedelta(minutes=5), require_sip=True)
        failures = {check.name for check in checks if check.status == "fail"}
        self.assertEqual(failures, {"bars_gaps", "bars_sip_feed"})
        with self.assertRaises(DataReadinessError):
            from market_predictor.v3.audits import DataAuditReport

            DataAuditReport(checks=checks).raise_for_failure()

    def test_naive_bar_timestamps_fail_instead_of_assuming_utc(self) -> None:
        bars = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "timestamp": ["2026-07-08 14:30:00"],
                "open": [500],
                "high": [501],
                "low": [499],
                "close": [500],
                "volume": [100],
                "price_feed": ["sip"],
            }
        )
        failures = {check.name for check in audit_bars(bars, interval=timedelta(minutes=5)) if check.status == "fail"}
        self.assertIn("bars_timestamp", failures)


if __name__ == "__main__":
    unittest.main()
