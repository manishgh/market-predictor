from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.intraday_enrichment import build_enriched_intraday_dataset


class IntradayEnrichmentTests(unittest.TestCase):
    def test_builds_setup_features_and_filters_rows(self) -> None:
        timestamps = pd.date_range("2026-07-08 13:30:00Z", periods=40, freq="5min")
        frame = pd.DataFrame(
            {
                "ticker": ["MSFT"] * len(timestamps),
                "timestamp": timestamps,
                "date": timestamps,
                "open": [100 + idx * 0.1 for idx in range(len(timestamps))],
                "high": [100.2 + idx * 0.1 for idx in range(len(timestamps))],
                "low": [99.9 + idx * 0.1 for idx in range(len(timestamps))],
                "close": [100.1 + idx * 0.1 for idx in range(len(timestamps))],
                "volume": [1000] * 10 + [2500] * 30,
                "volume_z20": [0.0] * 10 + [1.0] * 30,
                "ema_10": [100 + idx * 0.1 for idx in range(len(timestamps))],
                "ema_20": [99 + idx * 0.1 for idx in range(len(timestamps))],
                "ema_50": [98 + idx * 0.1 for idx in range(len(timestamps))],
                "macd_signal_diff": [0.1] * len(timestamps),
                "prior_macd_signal_diff": [0.0] * len(timestamps),
                "target_entry_success_12b": [0, 1] * 20,
            }
        )

        enriched, audit = build_enriched_intraday_dataset(frame, setup_only=True, min_setup_score=2.0)

        self.assertFalse(enriched.empty)
        self.assertIn("setup_candidate_score", enriched.columns)
        self.assertIn("dist_session_vwap", enriched.columns)
        self.assertIn("market_regime", enriched.columns)
        self.assertTrue(enriched["is_intraday_setup_candidate"].all())
        self.assertEqual(audit["ticker"].tolist(), ["MSFT"])


if __name__ == "__main__":
    unittest.main()
