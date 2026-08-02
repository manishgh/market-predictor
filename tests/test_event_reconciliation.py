"""R7.3 exact event-to-decision assignment and aggregate reconciliation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import (
    event_assignment_checks,
    event_reconciliation_checks,
)
from market_predictor.canonical.reconciliation import (
    ASSIGNMENT_FEATURE_COLUMNS,
    aggregate_reconciliation_summary,
    assignment_integrity_summary,
    build_event_assignments,
    event_aggregate_sha256,
    reconciliation_sha256,
    reconciliation_summary,
    reproduce_event_features,
)


class EventReconciliationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.decision = pd.Timestamp("2026-01-20 21:00", tz="UTC")
        self.decisions = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "security_id": ["security:aaa"],
                "decision_time_utc": [self.decision],
                "prediction_cutoff_policy_id": ["test-cutoff-v1"],
                "timeframe": ["1d"],
                "bar_start_utc": [pd.Timestamp("2026-01-20", tz="UTC")],
            }
        )
        hour = pd.Timedelta(hours=1)
        self.events = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "ZZZ", "AAA", "AAA"],
                "security_id": [
                    "security:aaa",
                    "security:aaa",
                    "security:zzz",
                    "security:aaa",
                    "security:aaa",
                ],
                "source_family": ["alpaca", "alpaca", "open_source", "sec", "finviz"],
                "event_id": ["m1", "m1", "w1", "f1", "o1"],
                "feature_available_at_utc": [
                    self.decision - hour,
                    self.decision - hour,
                    self.decision - hour,
                    self.decision + hour,
                    self.decision - pd.Timedelta(days=10),
                ],
                "sentiment_numeric": [0.8, 0.8, -0.2, 0.5, np.nan],
                "relevance": [1.0, 1.0, 0.8, 0.4, np.nan],
            }
        )

    def _artifact(self) -> pd.DataFrame:
        return build_event_assignments(self.decisions, self.events)

    def test_assignments_name_exact_decision_and_each_matching_window(self) -> None:
        artifact = self._artifact()
        assigned = artifact[artifact["status"].eq("assigned")]
        self.assertEqual(assigned["event_id"].tolist(), ["m1", "m1", "m1"])
        self.assertEqual(set(assigned["window_name"]), {"2h", "1d", "3d"})
        self.assertEqual(assigned["decision_id"].nunique(), 1)
        self.assertTrue(
            (assigned["feature_available_at_utc"] <= assigned["decision_time_utc"]).all()
        )

    def test_every_excluded_input_gets_a_deterministic_reason(self) -> None:
        artifact = self._artifact()
        statuses = artifact.groupby("event_id")["status"].apply(set).to_dict()
        self.assertEqual(statuses["m1"], {"assigned", "duplicate_event_id"})
        self.assertEqual(statuses["w1"], {"security_not_in_decisions"})
        self.assertEqual(statuses["f1"], {"no_future_decision"})
        self.assertEqual(statuses["o1"], {"outside_all_windows"})
        summary = reconciliation_summary(artifact)
        self.assertEqual(summary["unexplained_events"], 0)
        self.assertEqual(summary["duplicate_event_id"], 1)

    def test_time_index_assignment_handles_unsorted_multi_security_decisions(self) -> None:
        decisions = pd.DataFrame(
            {
                "ticker": ["BBB", "AAA", "AAA", "BBB"],
                "security_id": [
                    "security:bbb",
                    "security:aaa",
                    "security:aaa",
                    "security:bbb",
                ],
                "decision_time_utc": [
                    self.decision + pd.Timedelta(days=1),
                    self.decision,
                    self.decision + pd.Timedelta(days=1),
                    self.decision,
                ],
                "prediction_cutoff_policy_id": ["test-cutoff-v1"] * 4,
                "timeframe": ["1d"] * 4,
                "bar_start_utc": [
                    self.decision + pd.Timedelta(days=1),
                    self.decision,
                    self.decision + pd.Timedelta(days=1),
                    self.decision,
                ],
            }
        )
        events = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "security_id": ["security:aaa", "security:bbb"],
                "source_family": ["alpaca", "alpaca"],
                "event_id": ["aaa-event", "bbb-event"],
                "feature_available_at_utc": [
                    self.decision - pd.Timedelta(hours=1),
                    self.decision + pd.Timedelta(hours=1),
                ],
                "sentiment_numeric": [0.5, -0.5],
                "relevance": [1.0, 1.0],
            }
        )

        assigned = build_event_assignments(decisions, events)

        aaa = assigned.loc[
            assigned["event_id"].eq("aaa-event")
            & assigned["status"].eq("assigned")
        ]
        bbb = assigned.loc[
            assigned["event_id"].eq("bbb-event")
            & assigned["status"].eq("assigned")
        ]
        self.assertEqual(aaa["decision_id"].nunique(), 2)
        self.assertEqual(bbb["decision_id"].nunique(), 1)
        self.assertTrue(
            (
                pd.to_datetime(assigned["feature_available_at_utc"], utc=True)
                <= pd.to_datetime(assigned["decision_time_utc"], utc=True)
            )
            .fillna(True)
            .all()
        )

    def test_assigned_identity_uses_decision_ticker_after_symbol_change(self) -> None:
        for event_ticker, decision_ticker in (
            ("FB", "META"),
            ("NLOK", "GEN"),
            ("VIAC", "PARA"),
        ):
            with self.subTest(event_ticker=event_ticker, decision_ticker=decision_ticker):
                decisions = self.decisions.copy()
                decisions["ticker"] = decision_ticker
                events = self.events.iloc[[0]].copy()
                events["ticker"] = event_ticker

                assignments = build_event_assignments(decisions, events)
                assigned = assignments.loc[assignments["status"].eq("assigned")]

                self.assertEqual(set(assigned["ticker"]), {decision_ticker})
                self.assertEqual(set(assigned["security_id"]), {"security:aaa"})
                self.assertEqual(assigned["decision_id"].nunique(), 1)

    def test_aggregates_reproduce_from_assignment_rows(self) -> None:
        assignments = self._artifact()
        decisions = reproduce_event_features(self.decisions, assignments)
        self.assertEqual(decisions.loc[0, "event_count_2h"], 1)
        self.assertEqual(decisions.loc[0, "sentiment_mean_2h"], 0.8)
        self.assertEqual(decisions.loc[0, "source_family_count_2h"], 1)
        self.assertEqual(
            decisions.loc[0, "latest_event_feature_available_at_utc"],
            self.decision - pd.Timedelta(hours=1),
        )
        summary = aggregate_reconciliation_summary(decisions, assignments)
        self.assertEqual(summary["aggregate_reconciliation_errors"], 0)
        self.assertGreater(summary["aggregate_cells_checked"], 0)

    def test_compact_assignment_projection_reproduces_exact_aggregates(
        self,
    ) -> None:
        assignments = self._artifact()
        complete = reproduce_event_features(self.decisions, assignments)
        compact = reproduce_event_features(
            self.decisions,
            assignments.loc[:, ASSIGNMENT_FEATURE_COLUMNS],
        )

        pd.testing.assert_frame_equal(compact, complete, check_exact=True)

    def test_inplace_reproduction_is_exact_and_cleans_assignment_scratch(
        self,
    ) -> None:
        assignments = self._artifact()
        assigned = assignments.loc[
            assignments["status"].eq("assigned"),
            ASSIGNMENT_FEATURE_COLUMNS,
        ].copy()
        expected = reproduce_event_features(self.decisions, assigned.copy())
        target = self.decisions.copy()

        actual = reproduce_event_features(
            target,
            assigned,
            inplace=True,
        )

        self.assertIs(actual, target)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        self.assertFalse(
            any(column.startswith("_") for column in assigned.columns)
        )

    def test_hashes_are_deterministic_and_content_sensitive(self) -> None:
        assignments = self._artifact()
        self.assertEqual(
            reconciliation_sha256(assignments),
            reconciliation_sha256(assignments.sample(frac=1, random_state=3)),
        )
        decisions = reproduce_event_features(self.decisions, assignments)
        original = event_aggregate_sha256(decisions)
        decisions.loc[0, "event_count_2h"] += 1
        self.assertNotEqual(original, event_aggregate_sha256(decisions))

    def test_deleted_assignment_is_rejected(self) -> None:
        assignments = self._artifact().iloc[1:].reset_index(drop=True)
        summary = assignment_integrity_summary(
            self.decisions,
            self.events,
            assignments,
        )
        self.assertGreater(summary["deleted_assignment_rows"], 0)
        self.assertGreater(summary["assignment_integrity_errors"], 0)

    def test_duplicated_assignment_is_rejected(self) -> None:
        assignments = self._artifact()
        poisoned = pd.concat(
            [assignments, assignments.iloc[[0]]],
            ignore_index=True,
        )
        summary = assignment_integrity_summary(
            self.decisions,
            self.events,
            poisoned,
        )
        self.assertGreater(summary["duplicate_assignment_rows"], 0)
        self.assertGreater(summary["assignment_integrity_errors"], 0)

    def test_wrong_ticker_assignment_is_rejected(self) -> None:
        poisoned = self._artifact()
        poisoned.loc[poisoned["status"].eq("assigned"), "ticker"] = "BBB"
        summary = assignment_integrity_summary(
            self.decisions,
            self.events,
            poisoned,
        )
        self.assertGreater(summary["assignment_integrity_errors"], 0)

    def test_wrong_window_assignment_is_rejected(self) -> None:
        poisoned = self._artifact()
        index = poisoned.index[poisoned["status"].eq("assigned")][0]
        poisoned.loc[index, "window_name"] = "4h"
        poisoned.loc[index, "window_seconds"] = 14_400
        summary = assignment_integrity_summary(
            self.decisions,
            self.events,
            poisoned,
        )
        self.assertGreater(summary["assignment_integrity_errors"], 0)

    def test_poisoned_aggregate_is_rejected(self) -> None:
        assignments = self._artifact()
        decisions = reproduce_event_features(self.decisions, assignments)
        decisions.loc[0, "event_count_2h"] += 1
        summary = aggregate_reconciliation_summary(decisions, assignments)
        self.assertGreater(summary["aggregate_value_mismatches"], 0)
        checks = event_assignment_checks(
            assignment_integrity_summary(
                self.decisions,
                self.events,
                assignments,
            ),
            summary,
        )
        self.assertEqual(checks[1].status, "fail")


class ReconciliationAuditCheckTest(unittest.TestCase):
    def test_check_passes_when_every_event_is_explained(self) -> None:
        checks = event_reconciliation_checks(
            {"total_events": 5, "unexplained_events": 0, "assigned": 5}
        )
        self.assertEqual(checks[0].name, "event_reconciliation")
        self.assertEqual(checks[0].status, "pass")

    def test_check_fails_on_any_unexplained_event(self) -> None:
        checks = event_reconciliation_checks(
            {"total_events": 5, "unexplained_events": 1}
        )
        self.assertEqual(checks[0].status, "fail")


if __name__ == "__main__":
    unittest.main()
