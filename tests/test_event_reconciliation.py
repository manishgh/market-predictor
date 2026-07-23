"""R3 P0-3b: event-to-feature reconciliation categorises every accepted event."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_predictor.canonical.reconciliation import (
    reconcile_events,
    reconciliation_sha256,
    reconciliation_summary,
)


class EventReconciliationTest(unittest.TestCase):
    def _artifact(self) -> pd.DataFrame:
        decision = pd.Timestamp("2026-01-20 21:00", tz="UTC")
        hour = pd.Timedelta(hours=1)
        decisions = pd.DataFrame({"ticker": ["AAA"], "decision_time_utc": [decision]})
        events = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "ZZZ", "AAA", "AAA", "AAA", "AAA"],
                "event_id": ["m1", "m1", "w1", "f1", "u1", "i1", "o1"],
                "feature_available_at_utc": [
                    decision - hour,          # matched (inside window)
                    decision - hour,          # duplicate id
                    decision - hour,          # wrong ticker
                    decision + hour,          # available only after the decision
                    decision - hour,          # unknown relevance
                    decision - hour,          # irrelevant (below floor)
                    decision - pd.Timedelta(days=10),  # outside the 3d window
                ],
                "relevance": [1.0, 1.0, 1.0, 1.0, np.nan, 0.1, 1.0],
            }
        )
        return reconcile_events(decisions, events, windows={"3d": pd.Timedelta(days=3)})

    def test_every_event_gets_exactly_one_expected_status(self) -> None:
        artifact = self._artifact()
        self.assertEqual(
            list(artifact["status"]),
            [
                "matched",
                "duplicate",
                "wrong_ticker",
                "unavailable_future",
                "unknown_relevance",
                "irrelevant",
                "outside_window",
            ],
        )

    def test_summary_has_zero_unexplained(self) -> None:
        summary = reconciliation_summary(self._artifact())
        self.assertEqual(summary["total_events"], 7)
        self.assertEqual(summary["unexplained_events"], 0)
        self.assertEqual(summary["matched"], 1)
        # duplicate + wrong_ticker + unavailable_future + outside_window are lineage errors.
        self.assertEqual(summary["lineage_error_events"], 4)

    def test_hash_is_deterministic(self) -> None:
        self.assertEqual(reconciliation_sha256(self._artifact()), reconciliation_sha256(self._artifact()))


class ReconciliationAuditCheckTest(unittest.TestCase):
    def test_check_passes_when_every_event_is_explained(self) -> None:
        from market_predictor.canonical.audits import event_reconciliation_checks

        checks = event_reconciliation_checks({"total_events": 5, "unexplained_events": 0, "matched": 5})
        self.assertEqual(checks[0].name, "event_reconciliation")
        self.assertEqual(checks[0].status, "pass")

    def test_check_fails_on_any_unexplained_event(self) -> None:
        from market_predictor.canonical.audits import event_reconciliation_checks

        checks = event_reconciliation_checks({"total_events": 5, "unexplained_events": 1})
        self.assertEqual(checks[0].status, "fail")


if __name__ == "__main__":
    unittest.main()
