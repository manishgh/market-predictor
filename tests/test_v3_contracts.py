from __future__ import annotations

import unittest
from datetime import UTC, date, datetime

from pydantic import ValidationError

from market_predictor.v3.contracts import DecisionRowIdentity, SourceAvailability, UniverseMembership


class V3ContractTests(unittest.TestCase):
    def test_decision_row_enforces_point_in_time_ordering(self) -> None:
        decision = datetime(2026, 7, 8, 15, 0, tzinfo=UTC)
        identity = DecisionRowIdentity(
            ticker="msft",
            decision_time_utc=decision,
            feature_available_at_utc=decision,
            entry_time_utc=datetime(2026, 7, 8, 15, 5, tzinfo=UTC),
            session_date_et=date(2026, 7, 8),
            decision_group_id="20260708T150000Z",
            universe_snapshot_id="sp500-20260708",
            price_feed="SIP",
        )
        self.assertEqual(identity.ticker, "MSFT")
        self.assertEqual(identity.price_feed, "sip")
        with self.assertRaises(ValidationError):
            DecisionRowIdentity(**{**identity.model_dump(), "feature_available_at_utc": identity.entry_time_utc})
        with self.assertRaises(ValidationError):
            DecisionRowIdentity(**{**identity.model_dump(), "decision_time_utc": decision.replace(tzinfo=None)})

    def test_membership_uses_half_open_effective_window(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 7, 1, tzinfo=UTC)
        membership = UniverseMembership(
            ticker="BRK/B",
            effective_from_utc=start,
            effective_to_utc=end,
            sector="Financials",
            industry="Insurance",
            market_cap_bucket="mega",
            liquidity_bucket="high",
            primary_benchmark="SPY",
            universe_snapshot_id="snapshot-1",
        )
        self.assertEqual(membership.ticker, "BRK.B")
        self.assertTrue(membership.contains(start))
        self.assertFalse(membership.contains(end))

    def test_source_availability_does_not_invent_missing_history(self) -> None:
        now = datetime.now(UTC)
        missing = SourceAvailability(
            ticker="RGTI",
            source_family="reddit",
            available=False,
            row_count=0,
            collected_at_utc=now,
        )
        self.assertFalse(missing.available)
        with self.assertRaises(ValidationError):
            SourceAvailability(
                ticker="RGTI",
                source_family="reddit",
                available=True,
                row_count=0,
                collected_at_utc=now,
            )


if __name__ == "__main__":
    unittest.main()
