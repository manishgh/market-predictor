from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

import pandas as pd

from market_predictor.global_context import score_flashpoints


class GlobalContextTests(unittest.TestCase):
    def test_scores_oil_chokepoint_flashpoint(self) -> None:
        now = datetime(2026, 7, 8, tzinfo=UTC)
        events = pd.DataFrame(
            [
                {
                    "timestamp": now - timedelta(hours=1),
                    "title": "Hormuz blockade threat disrupts oil shipment routes",
                    "summary": "Tanker traffic in the Persian Gulf faces missile attack risk.",
                    "sentiment_numeric": -0.6,
                },
                {
                    "timestamp": now - timedelta(hours=2),
                    "title": "Persian Gulf tanker seizure raises oil supply risk",
                    "summary": "",
                    "sentiment_numeric": -0.4,
                },
            ]
        )

        scored = score_flashpoints(events, now=now, lookback_hours=24)

        self.assertFalse(scored.empty)
        first = scored.iloc[0]
        self.assertEqual(first["commodity_channel"], "oil")
        self.assertGreater(first["shock_score"], 0.0)
        self.assertIn("energy_oil_gas", first["positive_themes"])



if __name__ == "__main__":
    unittest.main()
