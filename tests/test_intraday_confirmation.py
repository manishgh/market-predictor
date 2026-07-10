from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.intraday_confirmation import latest_one_minute_confirmation


class IntradayConfirmationTests(unittest.TestCase):
    def test_latest_one_minute_confirmation_detects_bullish_vwap(self) -> None:
        timestamps = pd.date_range("2026-07-08 13:30:00Z", periods=40, freq="min")
        close = [10 + idx * 0.02 for idx in range(40)]
        bars = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": close,
                "high": [value + 0.03 for value in close],
                "low": [value - 0.02 for value in close],
                "close": close,
                "volume": [1000] * 25 + [2500] * 15,
            }
        )

        result = latest_one_minute_confirmation("MSFT", bars)

        self.assertEqual(result["one_minute_status"], "ok")
        self.assertIn(result["one_minute_confirmation_signal"], {"bullish_breakout_confirmation", "bullish_vwap_confirmation"})
        self.assertGreater(result["one_minute_dist_vwap"], 0)


if __name__ == "__main__":
    unittest.main()
