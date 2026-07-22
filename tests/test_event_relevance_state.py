"""R3 P1-1: unknown event relevance must not masquerade as fully relevant."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_predictor.canonical.joins import aggregate_event_features


class UnknownRelevanceTest(unittest.TestCase):
    def test_unknown_relevance_is_excluded_and_counted_low(self) -> None:
        decision_time = pd.Timestamp("2026-01-05 20:00", tz="UTC")
        decisions = pd.DataFrame({"ticker": ["AAA"], "decision_time_utc": [decision_time]})
        events = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "source_family": ["alpaca", "reddit"],
                "feature_available_at_utc": [
                    decision_time - pd.Timedelta(hours=1),
                    decision_time - pd.Timedelta(minutes=30),
                ],
                "availability_policy": ["observed", "observed"],
                "event_id": ["validated_event_0001", "unknown_event_0002"],
                "sentiment_numeric": [0.8, 0.8],
                "relevance": [1.0, np.nan],  # one validated-relevant, one unknown
            }
        )
        out = aggregate_event_features(decisions, events, windows={"1d": pd.Timedelta(days=1)}, require_observed=True)
        row = out.iloc[0]
        self.assertEqual(int(row["event_count_1d"]), 2)
        self.assertAlmostEqual(float(row["unknown_relevance_event_fraction_1d"]), 0.5)
        # Under the old fillna(1.0) this would be 1.0; unknown is not fully relevant.
        self.assertAlmostEqual(float(row["event_relevance_mean_1d"]), 0.5)
        # Under the old rule (relevance < 0.5) unknown scored 0.0; now it counts as low.
        self.assertAlmostEqual(float(row["low_relevance_event_fraction_1d"]), 0.5)
        # Unknown carries zero relevance weight, so it does not dilute validated sentiment.
        self.assertAlmostEqual(float(row["sentiment_mean_1d"]), 0.8)


if __name__ == "__main__":
    unittest.main()
