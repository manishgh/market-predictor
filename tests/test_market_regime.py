from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.market_regime import add_market_regime_labels


class MarketRegimeTests(unittest.TestCase):
    def test_labels_intraday_risk_on_and_risk_off_from_benchmarks(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "date": pd.date_range("2026-07-01 13:30:00Z", periods=3, freq="5min"),
                "spy_return_6bar": [0.003, -0.004, 0.0],
                "qqq_return_6bar": [0.002, -0.003, 0.0],
            }
        )

        labeled = add_market_regime_labels(frame)

        self.assertEqual(labeled["market_regime"].tolist(), ["risk_on", "risk_off", "neutral"])
        self.assertIn("market_regime_score", labeled.columns)
        self.assertEqual(labeled["market_regime_risk_on"].tolist(), [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
