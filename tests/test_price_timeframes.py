from __future__ import annotations

import unittest

from market_predictor.price import INTRADAY_TIMEFRAMES


class PriceTimeframeTests(unittest.TestCase):
    def test_intraday_timeframes_include_entry_exit_granularities(self) -> None:
        self.assertEqual(INTRADAY_TIMEFRAMES["1m"]["alpaca"], "1Min")
        self.assertEqual(INTRADAY_TIMEFRAMES["5m"]["alpaca"], "5Min")
        self.assertEqual(INTRADAY_TIMEFRAMES["1h"]["alpaca"], "1Hour")


if __name__ == "__main__":
    unittest.main()
